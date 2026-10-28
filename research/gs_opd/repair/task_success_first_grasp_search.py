# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded physical grasp search driven by active collision and pad geometry."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import socket
import subprocess
import sys
import time
from collections import Counter
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from research.gs_opd.repair.task_success_first_search import (
    SearchFailureMemory,
    blocked_candidate_ids_from_report,
    candidate_route_preflight_admitted,
    generated_geometry_high_approach_admitted,
    candidate_matches_failure_memory,
    diversify_candidates,
    high_approach_route,
    object_relative_grasp_targets,
    planner_seed,
    route_control_action_budget,
    route_signature,
    robot_contact_escape_direction,
    target_finger_contact_sides,
)
from research.gs_opd.repair.task_skill_experience import (
    alternate_experience_approach_candidates,
    append_verified_search_grasp_template,
    instantiate_grasp_candidate,
    load_experience_library,
    matching_grasp_templates,
    matching_operation_skill,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = Path("/workspace/zy_workspace/LIBERO-Safety")
BRIDGE_PYTHON = LIBERO_ROOT / ".venv/bin/python"
BRIDGE_SCRIPT = REPO_ROOT / "research/gs_opd/smoke/safety_bridge.py"
DEFAULT_SNAPSHOT_ROOT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "obstacle_avoidance_L0T1_raw_student_seed0_init0_20261025_234627"
)
DEFAULT_OUTPUT_ROOT = Path("/opt/data/private/gs_opd/rollouts")
BRIDGE_AUTHKEY = b"gs-opd-local"
EXPECTED_MUJOCO_SUBSTEPS_PER_ACTION = 25
MAX_GEOMETRY_CANDIDATES = 16
MAX_OFFLINE_GEOMETRY_SCREEN_BATCH = 24
MAX_OFFLINE_GEOMETRY_GENERATION = 48
MAX_RRT_PREFLIGHTS = 8
MAX_PHYSICAL_BRANCHES = 3
MAX_BRANCH_CONTROL_ACTIONS = 440
MAX_PREFIX_CONTROL_ACTIONS = 320
MAX_APPROACH_CONTROL_ACTIONS = 60
MAX_CLOSE_CONTROL_ACTIONS = 18
MAX_LIVE_CONTACT_SETTLE_ACTIONS = 2
LIVE_CONTACT_SETTLE_LINEAR_SPEED_M_S = 0.015
LIVE_CONTACT_SETTLE_ANGULAR_SPEED_RAD_S = 0.08
LIVE_CONTACT_SETTLE_REQUIRED_SAMPLES = 3
LIVE_CONTACT_CLOSE_POSITION_TOLERANCE_M = 0.004
LIVE_CONTACT_CLOSE_ROTATION_TOLERANCE_RAD = 0.045
MAX_LIFT_CONTROL_ACTIONS = 40
MINIMUM_OBJECT_LIFT_M = 0.030
LIFT_COMMAND_M = 0.040


class _PythonObjectState:
    """Compatibility wrapper for opaque runtime objects in archived Snapshots."""

    def __init__(self, module: str, qualname: str, attributes: dict[str, Any]):
        self.module = module
        self.qualname = qualname
        self.attributes = attributes


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pose(position: Any, rotation: Any) -> np.ndarray:
    p = np.asarray(position, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    if p.shape != (3,) or r.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("active_runtime_pose_is_malformed")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3], result[:3, 3] = r, p
    return result


def _wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()
    value = np.r_[xyzw[3], xyzw[:3]]
    return value / np.linalg.norm(value)


def _rotation_error(left_wxyz: Any, right_wxyz: Any) -> float:
    left = np.asarray(left_wxyz, dtype=np.float64).copy()
    right = np.asarray(right_wxyz, dtype=np.float64).copy()
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))))


def _box_half_extents(geometry: dict[str, Any]) -> np.ndarray:
    kind = str(geometry.get("geom_type_name", ""))
    size = np.asarray(geometry.get("size_m", []), dtype=np.float64)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
        raise ValueError("target_collision_geom_has_invalid_compiled_size")
    if kind in {"box", "mesh"}:
        return size.copy()
    if kind == "sphere":
        return np.repeat(float(size[0]), 3)
    if kind in {"cylinder", "capsule"}:
        return np.asarray([size[0], size[0], size[1]], dtype=np.float64)
    if kind == "ellipsoid":
        return size.copy()
    raise ValueError(f"unsupported_target_collision_geom_type:{kind or 'unknown'}")


def generate_geometry_candidates(
    context: dict[str, Any], *, maximum_candidates: int = MAX_GEOMETRY_CANDIDATES
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate opposing-pad grasps on the active target's compiled geometry."""
    if isinstance(maximum_candidates, bool) or not 1 <= maximum_candidates <= MAX_OFFLINE_GEOMETRY_GENERATION:
        raise ValueError(
            f"maximum_candidates_must_be_in_1..{MAX_OFFLINE_GEOMETRY_GENERATION}"
        )
    target_pose = _pose(
        context["target_root_pose_world"]["position_m"],
        context["target_root_pose_world"]["rotation"],
    )
    pad_center = np.asarray(context["symmetric_pad_center_eef_m"], dtype=np.float64)
    if pad_center.shape != (3,) or not np.isfinite(pad_center).all():
        raise ValueError("active_bilateral_pad_center_is_invalid")
    max_opening = float(context["maximum_pad_separation_m"])
    eef_start = np.asarray(context["eef_pose_world"]["position_m"], dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    skipped_geometries = []
    for geometry in context.get("target_collision_geometries", []):
        try:
            half = _box_half_extents(geometry)
        except ValueError as exc:
            skipped_geometries.append({"geom_id": geometry.get("geom_id"), "reason": str(exc)})
            continue
        geom_center = np.asarray(geometry["center_root_m"], dtype=np.float64)
        geom_rotation = np.asarray(geometry["rotation_root"], dtype=np.float64)
        if geom_center.shape != (3,) or geom_rotation.shape != (3, 3):
            raise ValueError("active_target_collision_geom_transform_is_invalid")
        for jaw_axis_index in range(3):
            grasp_width = float(2.0 * half[jaw_axis_index])
            for approach_axis_index in range(3):
                if approach_axis_index == jaw_axis_index:
                    continue
                approach_distance = max(0.060, float(half[approach_axis_index]) + 0.035)
                for approach_sign in (-1, 1):
                    jaw_base = geom_rotation[:, jaw_axis_index]
                    approach = geom_rotation[:, approach_axis_index] * approach_sign
                    for jaw_sign in (-1, 1):
                        jaw = jaw_base * jaw_sign
                        eef_y = np.cross(approach, jaw)
                        norm = float(np.linalg.norm(eef_y))
                        if norm <= 1e-8:
                            continue
                        eef_y /= norm
                        eef_x = np.cross(eef_y, approach)
                        eef_rotation_root = np.column_stack((eef_x, eef_y, approach))
                        if not np.isclose(np.linalg.det(eef_rotation_root), 1.0, atol=1e-6):
                            continue
                        root_to_eef = np.eye(4, dtype=np.float64)
                        root_to_eef[:3, :3] = eef_rotation_root
                        root_to_eef[:3, 3] = geom_center - eef_rotation_root @ pad_center
                        world_eef = target_pose @ root_to_eef
                        pregrasp_position = world_eef[:3, 3] - target_pose[:3, :3] @ approach * approach_distance
                        width_overrun = max(0.0, grasp_width - max_opening)
                        start_distance = float(np.linalg.norm(pregrasp_position - eef_start))
                        candidate_id = (
                            f"geom{int(geometry['geom_id'])}_jaw{jaw_axis_index}_"
                            f"approach{approach_axis_index}_{approach_sign:+d}_flip{jaw_sign:+d}"
                        )
                        candidates.append({
                            "candidate_id": candidate_id,
                            "target_geom_id": int(geometry["geom_id"]),
                            "target_geom_name": str(geometry["geom_name"]),
                            "geom_type": str(geometry["geom_type_name"]),
                            "object_to_eef": root_to_eef.tolist(),
                            "approach_direction_object": approach.tolist(),
                            "approach_distance_m": approach_distance,
                            "jaw_axis_index_in_geom": jaw_axis_index,
                            "approach_axis_index_in_geom": approach_axis_index,
                            "approach_sign": approach_sign,
                            "jaw_sign": jaw_sign,
                            "estimated_clamp_width_m": grasp_width,
                            "maximum_opening_m": max_opening,
                            "opening_overrun_m_record_only": width_overrun,
                            "pregrasp_eef_distance_m": start_distance,
                            "pad_center_eef_m_from_active_open_pose": pad_center.tolist(),
                            "candidate_source": "active_BDDL_target_compiled_collision_geom_and_live_Panda_pad_centers",
                            "target_geometry_shape_model": geometry.get("geometry_source", "compiled_geom_size"),
                            "official_demo_or_historical_action_prior_used": False,
                        })
    candidates.sort(
        key=lambda item: (
            item["opening_overrun_m_record_only"] > 1e-9,
            item["opening_overrun_m_record_only"],
            item["pregrasp_eef_distance_m"],
            item["candidate_id"],
        )
    )
    generated_candidate_count = len(candidates)
    unique_candidates = []
    seen_candidates: dict[tuple[float, ...], str] = {}
    duplicate_candidates = []
    for candidate in candidates:
        # Keep the same grasp transform when its approach direction differs;
        # those routes can pass on different sides of a scene obstacle.
        signature = tuple(np.round(np.r_[
            np.asarray(candidate["object_to_eef"], dtype=np.float64).ravel(),
            np.asarray(candidate["approach_direction_object"], dtype=np.float64),
        ], 8))
        original = seen_candidates.get(signature)
        if original is not None:
            duplicate_candidates.append({
                "candidate_id": candidate["candidate_id"],
                "duplicate_of": original,
            })
            continue
        seen_candidates[signature] = candidate["candidate_id"]
        unique_candidates.append(candidate)
        if len(unique_candidates) >= maximum_candidates:
            break
    candidates = unique_candidates
    return candidates, {
        "target_object_id_from_active_BDDL": context["target_object_id"],
        "target_native_goal_predicate": context["native_goal_predicate"],
        "active_target_collision_geom_count": len(context.get("target_collision_geometries", [])),
        "supported_collision_geometries": len(context.get("target_collision_geometries", [])) - len(skipped_geometries),
        "skipped_collision_geometries": skipped_geometries,
        "candidate_count": len(candidates),
        "generated_candidate_count_before_deduplication": generated_candidate_count,
        "duplicate_candidates_deduplicated": duplicate_candidates,
        "deduplicated_candidate_count": len(candidates),
        "maximum_candidate_budget": maximum_candidates,
        "jaw_pad_center_source": "active_runtime_MuJoCo_geom_xpos_relative_to_eef_site",
        "pad_separation_open_m": context["gripper_pad_separation_m"],
        "maximum_pad_separation_m": max_opening,
        "all_candidates_unique": True,
    }


def _candidate_window(
    candidates: list[dict[str, Any]], *, start: int, count: int
) -> list[dict[str, Any]]:
    if isinstance(start, bool) or start < 0 or isinstance(count, bool) or count < 1:
        raise ValueError("candidate_window_start_and_count_must_be_nonnegative_and_positive")
    return candidates[start : start + count]


def _candidate_generation_limit(args: argparse.Namespace) -> int:
    """Generate the full pool when an exact candidate ID pins a pose."""
    if args.candidate_id:
        return MAX_OFFLINE_GEOMETRY_GENERATION
    if args.strict_grasp_screen_only:
        return int(args.candidate_start_index + args.maximum_candidates)
    return int(args.maximum_candidates)


def _combine_candidate_sources(
    geometry_candidates: list[dict[str, Any]],
    experience_candidates: list[dict[str, Any]],
    *,
    generation_limit: int,
    requested_candidate_ids: list[str],
) -> list[dict[str, Any]]:
    """Keep pinned IDs addressable even when experience poses fill the pool cap."""
    if requested_candidate_ids:
        requested = set(requested_candidate_ids)
        by_id: dict[str, dict[str, Any]] = {}
        for candidate in (*experience_candidates, *geometry_candidates):
            candidate_id = str(candidate.get("candidate_id", ""))
            if candidate_id in requested:
                by_id.setdefault(candidate_id, candidate)
        return [by_id[candidate_id] for candidate_id in requested_candidate_ids if candidate_id in by_id]
    geometry_capacity = max(0, generation_limit - len(experience_candidates))
    return [*experience_candidates, *geometry_candidates[:geometry_capacity]]


def generate_grasp_tilt_variants(
    candidates: list[dict[str, Any]],
    *,
    target_pose_world: np.ndarray,
    eef_position_world: np.ndarray,
    tilt_degrees: list[float],
    rotation_axis: str = "jaw",
) -> list[dict[str, Any]]:
    """Rotate candidate orientation while anchoring its bilateral pad center."""
    if rotation_axis not in {"jaw", "approach"}:
        raise ValueError("grasp_orientation_rotation_axis_must_be_jaw_or_approach")
    target_pose = np.asarray(target_pose_world, dtype=np.float64)
    eef_position = np.asarray(eef_position_world, dtype=np.float64)
    if target_pose.shape != (4, 4) or not np.isfinite(target_pose).all():
        raise ValueError("grasp_tilt_target_pose_must_be_finite_4x4")
    if eef_position.shape != (3,) or not np.isfinite(eef_position).all():
        raise ValueError("grasp_tilt_eef_position_must_be_finite_xyz")
    angles = [float(value) for value in tilt_degrees]
    if (
        not angles
        or not np.isfinite(angles).all()
        or any(abs(value) <= 1e-8 or abs(value) > 30.0 for value in angles)
        or len(set(angles)) != len(angles)
    ):
        raise ValueError("grasp_tilt_angles_must_be_unique_nonzero_values_within_30_degrees")
    variants = []
    for candidate in candidates:
        transform = np.asarray(candidate.get("object_to_eef"), dtype=np.float64)
        pad_center = np.asarray(candidate.get("pad_center_eef_m_from_active_open_pose"), dtype=np.float64)
        approach_distance = float(candidate.get("approach_distance_m", 0.0606))
        if (
            transform.shape != (4, 4)
            or pad_center.shape != (3,)
            or not np.isfinite(transform).all()
            or not np.isfinite(pad_center).all()
            or not np.isfinite(approach_distance)
            or approach_distance <= 0.0
        ):
            raise ValueError("grasp_tilt_base_candidate_transform_or_pad_center_is_invalid")
        original_rotation = transform[:3, :3]
        object_pad_center = transform[:3, 3] + original_rotation @ pad_center
        for angle in angles:
            tilted = dict(candidate)
            new_transform = transform.copy()
            local_axis = "y" if rotation_axis == "jaw" else "z"
            axis_rotation = Rotation.from_euler(local_axis, angle, degrees=True).as_matrix()
            new_rotation = original_rotation @ axis_rotation
            new_transform[:3, :3] = new_rotation
            new_transform[:3, 3] = object_pad_center - new_rotation @ pad_center
            direction = new_rotation[:, 2]
            world_eef = target_pose @ new_transform
            pregrasp_world = world_eef[:3, 3] - target_pose[:3, :3] @ direction * approach_distance
            variant_label = "jaw_tilt" if rotation_axis == "jaw" else "approach_roll"
            variant_id = f"{candidate['candidate_id']}__{variant_label}_{angle:+.1f}deg"
            tilted.update({
                "candidate_id": variant_id,
                "source_candidate_id": str(candidate["candidate_id"]),
                "object_to_eef": new_transform.tolist(),
                "approach_direction_object": direction.tolist(),
                "pregrasp_eef_distance_m": float(np.linalg.norm(pregrasp_world - eef_position)),
                "candidate_source": str(candidate.get("candidate_source", "")) + "+jaw_axis_tilt_variant",
                "orientation_perturbation": {
                    "type": f"rotation_about_gripper_{rotation_axis}_axis",
                    "angle_degrees": angle,
                    "pad_center_anchor_preserved": True,
                },
            })
            variants.append(tilted)
    return variants


def generate_grasp_roll_variants(
    candidates: list[dict[str, Any]],
    *,
    target_pose_world: np.ndarray,
    eef_position_world: np.ndarray,
    roll_degrees: list[float],
) -> list[dict[str, Any]]:
    """Roll the jaw pair around the approach axis without shifting pad center."""
    return generate_grasp_tilt_variants(
        candidates,
        target_pose_world=target_pose_world,
        eef_position_world=eef_position_world,
        tilt_degrees=roll_degrees,
        rotation_axis="approach",
    )


def generate_grasp_surface_offset_variants(
    candidates: list[dict[str, Any]],
    *,
    target_pose_world: np.ndarray,
    eef_position_world: np.ndarray,
    offset_object_mm: list[float],
    offset_object_axis: str = "z",
) -> list[dict[str, Any]]:
    """Shift pad centers along one active-object axis while preserving contact orientation."""
    target_pose = np.asarray(target_pose_world, dtype=np.float64)
    eef_position = np.asarray(eef_position_world, dtype=np.float64)
    offsets = [float(value) for value in offset_object_mm]
    if offset_object_axis not in {"x", "y", "z"}:
        raise ValueError("grasp_surface_offset_object_axis_must_be_x_y_or_z")
    if target_pose.shape != (4, 4) or not np.isfinite(target_pose).all():
        raise ValueError("grasp_surface_offset_target_pose_must_be_finite_4x4")
    if eef_position.shape != (3,) or not np.isfinite(eef_position).all():
        raise ValueError("grasp_surface_offset_eef_position_must_be_finite_xyz")
    if (
        not offsets
        or not np.isfinite(offsets).all()
        or any(abs(value) <= 1e-8 or abs(value) > 20.0 for value in offsets)
        or len(set(offsets)) != len(offsets)
    ):
        raise ValueError("grasp_surface_offsets_must_be_unique_nonzero_values_within_20mm")
    variants = []
    for candidate in candidates:
        transform = np.asarray(candidate.get("object_to_eef"), dtype=np.float64)
        pad_center = np.asarray(candidate.get("pad_center_eef_m_from_active_open_pose"), dtype=np.float64)
        approach = np.asarray(candidate.get("approach_direction_object"), dtype=np.float64)
        approach_distance = float(candidate.get("approach_distance_m", 0.0606))
        if (
            transform.shape != (4, 4)
            or pad_center.shape != (3,)
            or approach.shape != (3,)
            or not all(np.isfinite(value).all() for value in (transform, pad_center, approach))
            or not np.isfinite(approach_distance)
            or approach_distance <= 0.0
        ):
            raise ValueError("grasp_surface_offset_base_candidate_is_invalid")
        for offset_mm in offsets:
            tilted = dict(candidate)
            new_transform = transform.copy()
            axis_index = {"x": 0, "y": 1, "z": 2}[offset_object_axis]
            delta = np.zeros(3, dtype=np.float64)
            delta[axis_index] = offset_mm / 1000.0
            new_transform[:3, 3] += delta
            world_eef = target_pose @ new_transform
            approach_world = target_pose[:3, :3] @ (approach / np.linalg.norm(approach))
            pregrasp_world = world_eef[:3, 3] - approach_world * approach_distance
            tilted.update({
                "candidate_id": (
                    f"{candidate['candidate_id']}__object_{offset_object_axis}_{offset_mm:+.1f}mm"
                ),
                "source_candidate_id": str(candidate["candidate_id"]),
                "object_to_eef": new_transform.tolist(),
                "pregrasp_eef_distance_m": float(np.linalg.norm(pregrasp_world - eef_position)),
                "candidate_source": (
                    str(candidate.get("candidate_source", ""))
                    + f"+object_surface_{offset_object_axis}_offset"
                ),
                "surface_contact_offset": {
                    "object_axis": offset_object_axis,
                    "offset_mm": offset_mm,
                    "rotation_preserved": True,
                },
            })
            variants.append(tilted)
    return variants


def generate_grasp_coupled_tilt_offset_variants(
    candidates: list[dict[str, Any]],
    *,
    target_pose_world: np.ndarray,
    eef_position_world: np.ndarray,
    tilt_degrees: list[float],
    rotation_axis: str,
    offset_object_mm: list[float],
    offset_object_axis: str,
) -> list[dict[str, Any]]:
    """Combine a pad-center-anchored wrist rotation with one object-frame shift axis."""
    return generate_grasp_coupled_pose_grid_variants(
        candidates,
        target_pose_world=target_pose_world,
        eef_position_world=eef_position_world,
        tilt_degrees=tilt_degrees,
        rotation_axis=rotation_axis,
        offsets_by_object_axis={offset_object_axis: offset_object_mm},
    )


def generate_grasp_coupled_pose_grid_variants(
    candidates: list[dict[str, Any]],
    *,
    target_pose_world: np.ndarray,
    eef_position_world: np.ndarray,
    tilt_degrees: list[float],
    rotation_axis: str,
    offsets_by_object_axis: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Generate a bounded Cartesian grid of wrist rotations and object-frame shifts."""
    if not candidates:
        raise ValueError("coupled_grasp_perturbations_require_base_candidates")
    if not isinstance(offsets_by_object_axis, dict) or not offsets_by_object_axis:
        raise ValueError("coupled_grasp_perturbations_require_object_frame_offsets")
    if any(axis not in {"x", "y", "z"} for axis in offsets_by_object_axis):
        raise ValueError("coupled_grasp_offset_axes_must_be_x_y_or_z")
    oriented = generate_grasp_tilt_variants(
        candidates,
        target_pose_world=target_pose_world,
        eef_position_world=eef_position_world,
        tilt_degrees=tilt_degrees,
        rotation_axis=rotation_axis,
    )
    for candidate in oriented:
        candidate["_coupled_base_candidate_id"] = str(candidate.get("source_candidate_id", ""))
        candidate["_coupled_offsets_by_axis"] = {}
    coupled = oriented
    for axis in ("x", "y", "z"):
        if axis not in offsets_by_object_axis:
            continue
        coupled = generate_grasp_surface_offset_variants(
            coupled,
            target_pose_world=target_pose_world,
            eef_position_world=eef_position_world,
            offset_object_mm=offsets_by_object_axis[axis],
            offset_object_axis=axis,
        )
        for variant in coupled:
            offsets = dict(variant["_coupled_offsets_by_axis"])
            offsets[axis] = float(variant["surface_contact_offset"]["offset_mm"])
            variant["_coupled_offsets_by_axis"] = offsets
    for variant in coupled:
        orientation_candidate_id = next(
            str(variant["candidate_id"]).split(f"__object_{axis}_", 1)[0]
            for axis in ("x", "y", "z")
            if f"__object_{axis}_" in str(variant["candidate_id"])
        )
        base_candidate_id = str(variant.pop("_coupled_base_candidate_id"))
        offsets = dict(variant.pop("_coupled_offsets_by_axis"))
        variant["source_candidate_id"] = base_candidate_id
        variant["coupled_surface_contact_offsets"] = {
            "object_axes_mm": offsets,
            "rotation_preserved_after_offsets": True,
        }
        variant["coupled_pose_perturbation"] = {
            "base_candidate_id": base_candidate_id,
            "orientation_candidate_id": orientation_candidate_id,
            "composition_order": "rotate_about_gripper_axis_then_translate_in_object_frame_axes",
        }
    return coupled


def load_historical_grasp_pose_prior_candidates(
    path: Path,
    *,
    benchmark: str,
    level: int,
    task_id: int,
    init_state: int,
    seed: int,
    target_object_id: str,
    target_pose_world: np.ndarray,
    eef_position_world: np.ndarray,
    active_pad_center_eef_m: np.ndarray,
    maximum_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load task-matched grasp poses recomputed from hashed physical-state samples."""
    path = Path(path).resolve()
    prior_set = json.loads(path.read_text(encoding="utf-8"))
    if prior_set.get("schema_version") != "gs_opd_historical_grasp_pose_priors.v1":
        raise ValueError("historical_grasp_pose_prior_schema_version_mismatch")
    expected_task = {
        "benchmark": benchmark,
        "level": int(level),
        "task_id": int(task_id),
        "init_state": int(init_state),
        "seed": int(seed),
        "target_object_id": target_object_id,
    }
    if prior_set.get("task") != expected_task:
        raise ValueError("historical_grasp_pose_prior_task_identity_mismatch")
    if prior_set.get("safety_qualification_inherited") is not False:
        raise ValueError("historical_grasp_pose_prior_must_not_inherit_safety_qualification")
    source_rows = prior_set.get("source_episodes")
    candidate_rows = prior_set.get("candidates")
    if not isinstance(source_rows, list) or not source_rows or not isinstance(candidate_rows, list):
        raise ValueError("historical_grasp_pose_prior_sources_or_candidates_missing")
    if not 1 <= len(candidate_rows) <= int(maximum_candidates):
        raise ValueError("historical_grasp_pose_prior_candidate_count_exceeds_search_budget")

    source_by_id: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        source_id = str(source.get("source_episode_id", ""))
        if not source_id or source_id in source_by_id:
            raise ValueError("historical_grasp_pose_prior_source_episode_id_invalid")
        trajectory = Path(str(source.get("trajectory_npz", ""))).resolve()
        expected_sha = str(source.get("trajectory_npz_sha256", ""))
        if not trajectory.is_file() or _sha256_file(trajectory) != expected_sha:
            raise ValueError(f"historical_grasp_pose_source_npz_sha256_mismatch:{source_id}")
        result_path = Path(str(source.get("result_json", ""))).resolve()
        if not result_path.is_file() or _sha256_file(result_path) != source.get("result_json_sha256"):
            raise ValueError(f"historical_grasp_pose_source_result_sha256_mismatch:{source_id}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("success_once") is not True or result.get("done") is not True:
            raise ValueError(f"historical_grasp_pose_source_not_native_success:{source_id}")
        if source.get("native_task_success") is not True:
            raise ValueError(f"historical_grasp_pose_source_success_attestation_missing:{source_id}")
        if source.get("safety_qualification_inherited") is not False:
            raise ValueError(f"historical_grasp_pose_source_safety_label_inheritance_forbidden:{source_id}")
        contact_report_path = Path(str(source.get("contact_evidence_report", ""))).resolve()
        if (
            not contact_report_path.is_file()
            or _sha256_file(contact_report_path) != source.get("contact_evidence_report_sha256")
        ):
            raise ValueError(f"historical_grasp_pose_source_contact_report_sha256_mismatch:{source_id}")
        evaluation_path = Path(str(source.get("evaluation_report_json", ""))).resolve()
        if (
            not evaluation_path.is_file()
            or _sha256_file(evaluation_path) != source.get("evaluation_report_json_sha256")
        ):
            raise ValueError(f"historical_grasp_pose_source_evaluation_report_sha256_mismatch:{source_id}")
        evaluation_report = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation_task = evaluation_report.get("task", {})
        evaluation_model = evaluation_report.get("model", {})
        if (
            evaluation_task.get("level") != int(level)
            or evaluation_task.get("task_id") != int(task_id)
            or evaluation_task.get("initial_state_index") != int(init_state)
            or benchmark not in str(evaluation_task.get("suite", ""))
            or evaluation_model.get("base_checkpoint") != source.get("base_checkpoint_path")
            or evaluation_model.get("adapter_sha256") != source.get("adapter_sha256")
        ):
            raise ValueError(f"historical_grasp_pose_source_model_or_task_identity_mismatch:{source_id}")
        if not (
            evaluation_model.get("adapter_loaded_during_rollout") is True
            or evaluation_model.get("adapter_loaded_marker") is True
        ):
            raise ValueError(f"historical_grasp_pose_source_adapter_load_unverified:{source_id}")
        with np.load(trajectory, allow_pickle=False) as trace:
            if "metadata_json" not in trace or "step_dones" not in trace:
                raise ValueError(f"historical_grasp_pose_source_trace_fields_missing:{source_id}")
            metadata = json.loads(str(trace["metadata_json"].item()))
            dones = np.asarray(trace["step_dones"], dtype=bool)
            actual = {
                "suite": metadata.get("suite"),
                "level": metadata.get("level"),
                "task_id": metadata.get("task_id"),
                "initial_state_index": metadata.get("initial_state_index"),
                "env_seed": metadata.get("env_seed"),
                "action_dim": metadata.get("action_dim"),
                "steps": metadata.get("steps"),
            }
            required = {
                "suite": benchmark,
                "level": int(level),
                "task_id": int(task_id),
                "initial_state_index": int(init_state),
                "env_seed": int(seed),
                "action_dim": 7,
                "steps": int(source.get("steps", -1)),
            }
            if (
                actual != required
                or metadata.get("checkpoint_path") != source.get("base_checkpoint_path")
                or metadata.get("lora_checkpoint_path") != source.get("lora_adapter_path")
                or not len(dones) or not bool(dones[-1])
            ):
                raise ValueError(f"historical_grasp_pose_source_trace_identity_mismatch:{source_id}")
        if source.get("strict_safety_qualified") is True:
            raise ValueError(f"historical_grasp_pose_source_qualification_must_remain_episode_local:{source_id}")
        source_by_id[source_id] = {
            "source_episode_id": source_id,
            "trajectory_npz": str(trajectory),
            "trajectory_npz_sha256": expected_sha,
            "result_json": str(result_path),
            "result_json_sha256": source["result_json_sha256"],
            "evaluation_report_json": str(evaluation_path),
            "evaluation_report_json_sha256": source["evaluation_report_json_sha256"],
            "contact_evidence_report": str(contact_report_path),
            "contact_evidence_report_sha256": source["contact_evidence_report_sha256"],
            "base_checkpoint_path": source["base_checkpoint_path"],
            "base_checkpoint_sha256": source.get("base_checkpoint_sha256"),
            "lora_adapter_path": source["lora_adapter_path"],
            "adapter_sha256": source["adapter_sha256"],
            "steps": int(source["steps"]),
            "native_task_success": True,
            "strict_safety_qualified": False,
            "safety_qualification_inherited": False,
        }

    target_pose = np.asarray(target_pose_world, dtype=np.float64)
    eef_position = np.asarray(eef_position_world, dtype=np.float64)
    pad_center = np.asarray(active_pad_center_eef_m, dtype=np.float64)
    if (
        target_pose.shape != (4, 4)
        or not np.isfinite(target_pose).all()
        or any(arr.shape != shape or not np.isfinite(arr).all() for arr, shape in (
            (eef_position, (3,)), (pad_center, (3,))
        ))
    ):
        raise ValueError("historical_grasp_pose_active_scene_context_invalid")

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in candidate_rows:
        candidate_id = str(row.get("candidate_id", ""))
        source_id = str(row.get("source_episode_id", ""))
        if not candidate_id or candidate_id in seen_ids or source_id not in source_by_id:
            raise ValueError("historical_grasp_pose_candidate_id_or_source_invalid")
        seen_ids.add(candidate_id)
        window = row.get("stable_grasp_window_control_steps")
        samples = row.get("physical_pose_samples")
        if (
            not isinstance(window, list) or len(window) != 2
            or not isinstance(samples, list) or len(samples) < 3
            or row.get("bilateral_pad_contact_verified") is not True
            or row.get("object_lift_verified") is not True
        ):
            raise ValueError(f"historical_grasp_pose_contact_lift_evidence_missing:{candidate_id}")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in window):
            raise ValueError(f"historical_grasp_pose_window_indices_invalid:{candidate_id}")
        sample_indices = [sample.get("control_step_index") for sample in samples]
        if (
            sample_indices != list(range(window[0], window[1] + 1))
            or window[1] >= source_by_id[source_id]["steps"]
        ):
            raise ValueError(f"historical_grasp_pose_samples_not_contiguous:{candidate_id}")
        contact_evidence = row.get("bilateral_pad_contact_evidence")
        lift_evidence = row.get("object_lift_evidence")
        if (
            not isinstance(contact_evidence, dict)
            or contact_evidence.get("left_pad_first_control_step") is None
            or contact_evidence.get("right_pad_first_control_step") is None
            or max(
                int(contact_evidence["left_pad_first_control_step"]),
                int(contact_evidence["right_pad_first_control_step"]),
            ) > window[0]
            or not isinstance(lift_evidence, dict)
        ):
            raise ValueError(f"historical_grasp_pose_bilateral_contact_evidence_invalid:{candidate_id}")
        lift_threshold = float(lift_evidence.get("threshold_m", 0.03))
        initial_object_z = float(lift_evidence.get("initial_object_root_z_m", float("nan")))
        if not np.isfinite(lift_threshold) or lift_threshold < 0.03 or not np.isfinite(initial_object_z):
            raise ValueError(f"historical_grasp_pose_lift_evidence_invalid:{candidate_id}")
        transforms = []
        sample_object_z = []
        for sample in samples:
            try:
                object_pos = np.asarray(sample["object_root_position_world_m"], dtype=np.float64)
                object_quat = np.asarray(sample["object_root_quaternion_world_wxyz"], dtype=np.float64)
                eef_pos = np.asarray(sample["eef_position_world_m"], dtype=np.float64)
                eef_quat = np.asarray(sample["eef_quaternion_world_wxyz"], dtype=np.float64)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"historical_grasp_pose_sample_fields_invalid:{candidate_id}") from exc
            if (
                object_pos.shape != (3,) or eef_pos.shape != (3,)
                or object_quat.shape != (4,) or eef_quat.shape != (4,)
                or not all(np.isfinite(value).all() for value in (
                    object_pos, object_quat, eef_pos, eef_quat
                ))
                or np.linalg.norm(object_quat) < 1e-8
                or np.linalg.norm(eef_quat) < 1e-8
            ):
                raise ValueError(f"historical_grasp_pose_sample_pose_invalid:{candidate_id}")
            object_transform = np.eye(4, dtype=np.float64)
            object_transform[:3, :3] = Rotation.from_quat(
                object_quat[[1, 2, 3, 0]] / np.linalg.norm(object_quat)
            ).as_matrix()
            object_transform[:3, 3] = object_pos
            sample_object_z.append(float(object_pos[2]))
            eef_transform = np.eye(4, dtype=np.float64)
            eef_transform[:3, :3] = Rotation.from_quat(
                eef_quat[[1, 2, 3, 0]] / np.linalg.norm(eef_quat)
            ).as_matrix()
            eef_transform[:3, 3] = eef_pos
            transforms.append(np.linalg.inv(object_transform) @ eef_transform)
        if sample_object_z[0] - initial_object_z < lift_threshold:
            raise ValueError(f"historical_grasp_pose_lift_threshold_not_met:{candidate_id}")
        translation = np.mean([transform[:3, 3] for transform in transforms], axis=0)
        rotation_mean = np.mean([transform[:3, :3] for transform in transforms], axis=0)
        u, _, vt = np.linalg.svd(rotation_mean)
        rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation

        approach_evidence = row.get("approach_direction_evidence")
        if not isinstance(approach_evidence, dict):
            raise ValueError(f"historical_grasp_approach_evidence_missing:{candidate_id}")
        def relative_eef_position(sample: dict[str, Any]) -> np.ndarray:
            object_pos = np.asarray(sample["object_root_position_world_m"], dtype=np.float64)
            object_quat = np.asarray(sample["object_root_quaternion_world_wxyz"], dtype=np.float64)
            eef_pos = np.asarray(sample["eef_position_world_m"], dtype=np.float64)
            object_rotation = Rotation.from_quat(
                object_quat[[1, 2, 3, 0]] / np.linalg.norm(object_quat)
            ).as_matrix()
            return object_rotation.T @ (eef_pos - object_pos)
        approach_start = approach_evidence.get("start_pose")
        approach_end = approach_evidence.get("end_pose")
        if not isinstance(approach_start, dict) or not isinstance(approach_end, dict):
            raise ValueError(f"historical_grasp_approach_samples_missing:{candidate_id}")
        first_gripper_contact = int(row.get("first_gripper_target_contact_control_step", -1))
        approach_start_index = int(approach_start.get("control_step_index", -1))
        approach_end_index = int(approach_end.get("control_step_index", -1))
        if not 0 <= approach_start_index < approach_end_index < first_gripper_contact:
            raise ValueError(f"historical_grasp_approach_not_contact_free_prefix:{candidate_id}")
        approach = relative_eef_position(approach_end) - relative_eef_position(approach_start)
        approach_norm = float(np.linalg.norm(approach))
        if approach_norm < 1e-5:
            raise ValueError(f"historical_grasp_approach_direction_degenerate:{candidate_id}")
        approach /= approach_norm
        if row.get("approach_direction_object") is not None:
            recorded = np.asarray(row["approach_direction_object"], dtype=np.float64)
            if recorded.shape != (3,) or not np.isfinite(recorded).all() or not np.allclose(
                recorded / max(float(np.linalg.norm(recorded)), 1e-12), approach, atol=2e-4
            ):
                raise ValueError(f"historical_grasp_approach_direction_mismatch:{candidate_id}")
        approach_distance = float(row.get("approach_distance_m", 0.0606))
        if not np.isfinite(approach_distance) or not 0.02 <= approach_distance <= 0.12:
            raise ValueError(f"historical_grasp_approach_distance_invalid:{candidate_id}")
        world_eef = target_pose @ transform
        pregrasp = world_eef[:3, 3] - target_pose[:3, :3] @ approach * approach_distance
        candidates.append({
            "candidate_id": candidate_id,
            "candidate_source": "hashed_success_episode_object_relative_grasp_pose_prior",
            "source_candidate_id": candidate_id,
            "source_episode_id": source_id,
            "source_episode": source_by_id[source_id],
            "object_to_eef": transform.tolist(),
            "approach_direction_object": approach.tolist(),
            "approach_distance_m": approach_distance,
            "pregrasp_eef_distance_m": float(np.linalg.norm(pregrasp - eef_position)),
            "pad_center_eef_m_from_active_open_pose": pad_center.tolist(),
            "ik_seed_qpos": [],
            "jaw_axis_index_in_geom": -1,
            "approach_axis_index_in_geom": -1,
            "approach_sign": 0,
            "jaw_sign": 0,
            "historical_episode_stable_bilateral_grasp_prior_used": True,
            "historical_source_episode_qualification_inherited": False,
            "source_stable_grasp_window_control_steps": [int(window[0]), int(window[1])],
            "source_bilateral_pad_contact_evidence": row["bilateral_pad_contact_evidence"],
            "source_object_lift_evidence": row["object_lift_evidence"],
        })
    return candidates, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": prior_set["schema_version"],
        "task_identity_verified": True,
        "source_episode_count": len(source_by_id),
        "candidate_count": len(candidates),
        "source_action_replay": False,
        "source_safety_qualification_inherited": False,
    }


class BridgeSession:
    _READ_ONLY_RECONNECT_OPS = frozenset({
        "observe", "native_task_status", "native_goal_query", "inspect_contacts",
        "grasp_state", "geometry_inventory", "gripper_geometry_inventory",
        "task_semantic_geometry_bindings", "repair_context",
        "task_success_first_context", "task_success_first_articulation_context",
        "task_scene_object_collision_envelopes",
        "robot_joint_limit_inventory", "compiled_model_fingerprint_details",
        "trajectory_tracking_capabilities", "scratch_robot_fk_audit",
        "scratch_carried_geom_distance_audit",
    })

    def __init__(self, address: tuple[str, int], log_path: Path, args: argparse.Namespace):
        self.address = address
        self.log_path = log_path
        self.args = args
        self.process: subprocess.Popen[str] | None = None
        self.connection = None
        self.log_handle = None

    def start(self) -> None:
        with socket.socket() as sock:
            if sock.connect_ex(self.address) == 0:
                raise RuntimeError(f"Safety Bridge port {self.address[1]} is already in use")
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"
        env["LIBERO_CONFIG_PATH"] = "/workspace/zy_workspace/.libero_safety_config"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(LIBERO_ROOT), str(LIBERO_ROOT / "third_party/robosuite-1.4"), str(REPO_ROOT)]
        )
        self.log_handle = self.log_path.open("ab", buffering=0)
        command = [
            str(BRIDGE_PYTHON), str(BRIDGE_SCRIPT),
            "--benchmark", self.args.benchmark,
            "--level", str(self.args.level),
            "--task-id", str(self.args.task_id),
            "--init-state", str(self.args.init_state),
            "--seed", str(self.args.seed),
            "--camera-resolution", str(self.args.camera_resolution),
            "--port", str(self.address[1]),
            "--demonstration-horizon", str(getattr(self.args, "demonstration_horizon", 1000)),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Safety Bridge exited with code {self.process.returncode}")
            try:
                self.connection = Client(self.address, authkey=BRIDGE_AUTHKEY)
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)
        raise TimeoutError("Safety Bridge did not become ready within 120 seconds")

    def request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.connection is None:
            raise RuntimeError("Safety Bridge is not connected")
        operation = request.get("op")
        try:
            self.connection.send(request)
            result = self.connection.recv()
        except (EOFError, BrokenPipeError, ConnectionError, OSError):
            # A read-only reply can be retried safely after reconnecting to
            # the Bridge listener. Physical actions and Snapshot mutations
            # are deliberately never replayed on an ambiguous connection.
            if operation not in self._READ_ONLY_RECONNECT_OPS:
                raise
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None
            last_error: Exception | None = None
            for _ in range(20):
                try:
                    self.connection = Client(self.address, authkey=BRIDGE_AUTHKEY)
                    break
                except (ConnectionRefusedError, OSError) as exc:
                    last_error = exc
                    time.sleep(0.05)
            if self.connection is None:
                raise EOFError(
                    f"Safety Bridge read-only reconnect failed for {operation}: {last_error}"
                ) from last_error
            self.connection.send(request)
            result = self.connection.recv()
        if not isinstance(result, dict):
            raise RuntimeError(f"Bridge returned malformed response to {request.get('op')}")
        return result

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def _snapshot_state_vector(value: dict[str, Any]) -> np.ndarray:
    state = np.asarray(value.get("mujoco_integration_state", []), dtype=np.float64)
    if state.ndim != 1 or state.size == 0 or not np.isfinite(state).all():
        raise ValueError("runtime_snapshot_integration_state_is_missing_or_invalid")
    return state


def _verify_cold_start_reset(bridge: BridgeSession, expected_state: np.ndarray) -> dict[str, Any]:
    """Compare a claimed t0 snapshot to the freshly reset live environment."""
    observed = bridge.request({"op": "observe", "include_snapshot_state": True})
    if observed.get("timestep") != 0:
        raise RuntimeError("cold_start_fresh_reset_timestep_is_not_zero")
    actual = _snapshot_state_vector(observed.get("snapshot_state", {}))
    if actual.shape != expected_state.shape or not np.array_equal(actual, expected_state):
        raise RuntimeError("cold_start_snapshot_differs_from_fresh_reset_integration_state")
    return {
        "status": "pass",
        "fresh_reset_timestep": 0,
        "exact_integration_state_match": True,
        "integration_state_sha256": hashlib.sha256(
            np.ascontiguousarray(actual, dtype=np.float64).tobytes()
        ).hexdigest(),
    }


def _restore_initial(bridge: BridgeSession, payload: bytes, expected_state: np.ndarray, model_sha: str) -> dict[str, Any]:
    result = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
    if result.get("status") != "restored" or result.get("model_geometry_sha256") != model_sha:
        raise RuntimeError(f"initial_snapshot_restore_failed:{result.get('error') or result.get('status')}")
    actual = _snapshot_state_vector(result.get("snapshot_state", {}))
    if actual.shape != expected_state.shape or not np.array_equal(actual, expected_state):
        raise RuntimeError("initial_snapshot_restore_did_not_reproduce_exact_integration_state")
    return result


def _verified_resume_runtime_snapshot(
    path: Path,
    *,
    benchmark: str,
    level: int,
    task_id: int,
    seed: int,
    init_state: int,
    model_sha: str,
    bddl_sha: str,
    instruction: str,
) -> dict[str, Any]:
    """Load an exact partial-task prefix after validating its scene identity."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resume_Runtime_Snapshot_missing:{path}")
    payload = path.read_bytes()
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    envelope = pickle.loads(payload)
    scene = envelope.get("scene_metadata", {})
    expected = {
        "benchmark": benchmark,
        "level": level,
        "task_id": task_id,
        "environment_seed": seed,
        "init_state_index": init_state,
        "bddl_sha256": bddl_sha,
        "instruction": instruction,
    }
    mismatches = {
        key: {"expected": value, "actual": scene.get(key)}
        for key, value in expected.items()
        if scene.get(key) != value
    }
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != model_sha
        or mismatches
    ):
        raise ValueError(f"resume_Runtime_Snapshot_identity_mismatch:{mismatches}")
    mujoco_state = envelope.get("snapshot", {}).get("mujoco_state")
    state = _snapshot_state_vector({"mujoco_integration_state": mujoco_state})
    return {
        "path": path,
        "payload": payload,
        "payload_sha256": _sha256_bytes(payload),
        "snapshot_state": {"mujoco_integration_state": state},
        "integration_state_sha256": _state_sha(state),
        "scene_metadata": scene,
    }


def _object_state(snapshot_state: dict[str, Any], body_id: int) -> dict[str, Any]:
    values = [
        item for item in snapshot_state.get("object_states", {}).values()
        if int(item.get("body_id", -1)) == int(body_id)
    ]
    if len(values) != 1:
        raise ValueError(f"target_runtime_body_{body_id}_resolved_to_{len(values)}_states")
    return values[0]


def _runtime_object_pose(item: dict[str, Any]) -> np.ndarray:
    quaternion_wxyz = np.asarray(item.get("root_quat"), dtype=np.float64)
    if quaternion_wxyz.shape != (4,):
        raise ValueError("runtime_object_root_quaternion_is_invalid")
    rotation = Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()
    return _pose(item.get("root_pos"), rotation)


def _runtime_eef_pose(snapshot_state: dict[str, Any]) -> np.ndarray:
    robots = snapshot_state.get("robot_states", [])
    if not robots:
        raise ValueError("runtime_robot_pose_is_unavailable")
    robot = robots[0]
    quaternion_wxyz = np.asarray(robot.get("eef_quat"), dtype=np.float64)
    if quaternion_wxyz.shape != (4,):
        raise ValueError("runtime_eef_quaternion_is_invalid")
    rotation = Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()
    return _pose(robot.get("eef_pos"), rotation)


def _object_relative_target_pose(
    snapshot_state: dict[str, Any],
    target_body_id: int,
    object_to_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_transform = _runtime_object_pose(
        _object_state(snapshot_state, target_body_id)
    ) @ np.asarray(object_to_target, dtype=np.float64)
    return target_transform[:3, 3], _wxyz(target_transform[:3, :3])


def _pose_tracking_error_components(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
) -> tuple[float, float]:
    current_pose = np.asarray(current_pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)
    if (
        current_pose.shape != (4, 4)
        or target_pose.shape != (4, 4)
        or not np.isfinite(current_pose).all()
        or not np.isfinite(target_pose).all()
    ):
        raise ValueError("pose_tracking_error_requires_finite_4x4_poses")
    position_error = float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3]))
    rotation_error = float(np.linalg.norm(Rotation.from_matrix(
        target_pose[:3, :3] @ current_pose[:3, :3].T
    ).as_rotvec()))
    return position_error, rotation_error


def _live_contact_pose_is_close_to_candidate(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
) -> tuple[bool, float, float]:
    position_error, rotation_error = _pose_tracking_error_components(
        current_pose, target_pose,
    )
    aligned = _pose_tracking_error_within_tolerance(
        position_error,
        rotation_error,
        LIVE_CONTACT_CLOSE_POSITION_TOLERANCE_M,
        LIVE_CONTACT_CLOSE_ROTATION_TOLERANCE_RAD,
    )
    return aligned, position_error, rotation_error


def _learned_grasp_template(
    candidate: dict[str, Any],
    branch: dict[str, Any],
    context: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    benchmark: str,
    level: int,
    task_id: int,
    seed: int,
    init_state: int,
    output_dir: Path,
) -> dict[str, Any]:
    target_id = str(context["target_object_id"])
    family = target_id.rsplit("_", 1)[0] if target_id.rsplit("_", 1)[-1].isdigit() else target_id
    target_body_id = int(context["target_binding"]["root_body_id"])
    path_rows = []
    for item in records:
        if item.get("candidate_id") != candidate.get("candidate_id"):
            continue
        phase = str(item.get("phase", ""))
        if phase not in {
            "task_success_first_rrt_osc_pregrasp_route",
            "migrated_object_relative_approach",
            "migrated_object_relative_grasp_endpoint",
        }:
            continue
        try:
            state = item["snapshot_state"]
            object_pose = _runtime_object_pose(_object_state(state, target_body_id))
            relative_pose = np.linalg.inv(object_pose) @ _runtime_eef_pose(state)
            path_rows.append(np.r_[relative_pose[:3, 3], _wxyz(relative_pose[:3, :3])].tolist())
        except (KeyError, ValueError, TypeError):
            continue
    if len(path_rows) > 6:
        indices = np.linspace(0, len(path_rows) - 1, 6).round().astype(int)
        path_rows = [path_rows[index] for index in sorted(set(indices.tolist()))]
    grasp_transform = np.asarray(branch.get("verified_object_to_eef_at_grasp"), dtype=np.float64)
    if grasp_transform.shape != (4, 4) or not np.isfinite(grasp_transform).all():
        raise ValueError("verified_grasp_object_relative_transform_is_unavailable")
    lift = float(branch.get("target_lift_delta_m", 0.0) or 0.0)
    snapshot_hash = str(branch.get("final_snapshot", {}).get("sha256", ""))[:10]
    return {
        "template_id": (
            f"{family}:stable_grasp:{benchmark}_L{level}T{task_id}_seed{seed}_"
            f"init{init_state}_{snapshot_hash}"
        ),
        "object_id_in_source": target_id,
        "object_family": family.lower(),
        "operation": "bilateral_grasp_with_object_relative_approach",
        "source_kind": "GS_OPD_task_success_first_physical_skill_search",
        "source_task": context["instruction"],
        "source_scene": f"{benchmark}/L{level}T{task_id}",
        "source_archive": str(output_dir),
        "source_native_task_success": False,
        "skill_success": True,
        "stable_bilateral_grasp_verified": True,
        "verified_lift_m": lift,
        "source_control_actions": int(branch.get("branch_control_actions", 0)),
        "object_to_eef": grasp_transform.tolist(),
        "approach_direction_object": np.asarray(
            candidate["approach_direction_object"], dtype=np.float64
        ).tolist(),
        "approach_distance_m": float(candidate["approach_distance_m"]),
        "approach_waypoints_object_xyz_quaternion_wxyz": path_rows,
        "approach_waypoint_count": len(path_rows),
        "official_safety_demonstrations_used": False,
        "safety_certified": False,
        "historical_actions_replayed": False,
        "source_actions_used_as_direct_cross_scene_labels": False,
        "runtime_physics_revalidation_required": True,
    }


def _image(reply: dict[str, Any], key: str) -> np.ndarray:
    item = reply["images"][key]
    return np.frombuffer(item["data"], dtype=np.uint8).reshape(tuple(item["shape"])).copy()


def _build_safety_policy(context: dict[str, Any], semantic: dict[str, Any], geometry: dict[str, Any], gripper: dict[str, Any]) -> dict[str, Any]:
    """Create an active-binding diagnostic policy without object-name rules."""
    target_id = context["target_object_id"]
    bindings = semantic.get("bindings", {})
    if target_id not in bindings:
        raise ValueError("active_BDDL_target_has_no_current_semantic_geometry_binding")
    robot_geoms = {
        int(item["geom_id"])
        for entry in gripper.get("grippers", [])
        for item in entry.get("geometries", [])
    }
    pad_ids = {
        int(item["geom_id"])
        for side in context["fingertip_pad_geometries"].values()
        for item in side
    }
    covered_bodies: set[int] = set()
    groups: dict[str, list[dict[str, Any]]] = {
        "forbidden_objects": [], "allowed_target_objects": [], "other_scene_objects": []
    }
    for object_id, binding in bindings.items():
        body_ids = sorted(int(value) for value in binding.get("collision_body_ids", []))
        geom_ids = sorted(int(value) for value in binding.get("geom_ids", []))
        body_names = list(binding.get("collision_body_names", []))
        if not body_ids or not geom_ids:
            continue
        if len(body_names) != len(body_ids):
            raise ValueError(f"active_collision_body_names_incomplete_for_{object_id}")
        group = {
            "object_id": str(object_id),
            "body_ids": body_ids,
            "body_names": [str(value) for value in body_names],
            "geom_ids": geom_ids,
            "minimum_clearance_m": 0.010,
        }
        groups["allowed_target_objects" if object_id == target_id else "forbidden_objects"].append(group)
        covered_bodies.update(body_ids)
    if not groups["allowed_target_objects"] or not groups["forbidden_objects"]:
        raise ValueError("active_scene_does_not_expose_target_and_forbidden_geometry_groups")

    inventory_geoms = geometry.get("geometries", [])
    all_scene_bodies = {
        int(item["body_id"])
        for item in inventory_geoms
        if item.get("collision_enabled") and item.get("category") == "scene"
    }
    residual_bodies = sorted(all_scene_bodies - covered_bodies)
    residual_geom_ids = sorted(
        int(item["geom_id"])
        for item in inventory_geoms
        if item.get("collision_enabled")
        and item.get("category") == "scene"
        and int(item.get("body_id", -1)) in set(residual_bodies)
    )
    if residual_bodies and residual_geom_ids:
        names_by_body = {
            int(item["body_id"]): str(item["body_name"])
            for item in geometry.get("bodies", [])
            if int(item.get("body_id", -1)) in set(residual_bodies)
        }
        groups["other_scene_objects"].append({
            "object_id": "unmapped_scene_environment",
            "body_ids": residual_bodies,
            "body_names": [names_by_body[value] for value in residual_bodies],
            "geom_ids": residual_geom_ids,
            "minimum_clearance_m": 0.0,
        })
    target = groups["allowed_target_objects"][0]
    finger_pairs = [
        {"robot_geom_id": int(finger), "scene_geom_id": int(scene), "minimum_clearance_m": -0.002}
        for finger in sorted(pad_ids)
        for scene in target["geom_ids"]
    ]
    forbidden_pairs = [
        {
            "object_id_a": target["object_id"],
            "object_id_b": item["object_id"],
            "body_ids_a": target["body_ids"],
            "body_ids_b": item["body_ids"],
            "geom_ids_a": target["geom_ids"],
            "geom_ids_b": item["geom_ids"],
        }
        for item in groups["forbidden_objects"]
    ]
    return {
        "source": "active_BDDL_semantic_bindings_and_compiled_geometry_task_success_first_diagnostic_only",
        "minimum_clearance_m": 0.010,
        "forbidden_objects": groups["forbidden_objects"],
        "allowed_target_objects": groups["allowed_target_objects"],
        "other_scene_objects": groups["other_scene_objects"],
        "allowed_contact_pairs": finger_pairs,
        "forbidden_object_pairs": forbidden_pairs,
        "allowed_carried_support_contact_geom_ids": [],
        "carried_object": None,
    }


def _build_bddl_grasp_screen_policy(
    context: dict[str, Any],
    semantic: dict[str, Any],
    geometry: dict[str, Any],
    bddl_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a scratch-screen policy from the active BDDL contact constraints."""
    from research.gs_opd.repair.automatic_recovery_runtime import _parse_bddl_entities

    _fixtures, _objects, forbidden_robot, forbidden_pairs, goal_targets = _parse_bddl_entities(bddl_text)
    target_id = str(context["target_object_id"])
    if target_id not in goal_targets or not forbidden_robot:
        raise ValueError("active_BDDL_screen_policy_has_no_target_or_robot_contact_constraints")
    bindings = semantic.get("bindings", {})
    required = set(forbidden_robot) | set(goal_targets) | {
        name for pair in forbidden_pairs for name in pair
    }
    missing = sorted(required - set(bindings))
    if missing:
        raise ValueError(f"active_BDDL_screen_policy_entities_are_unbound:{missing}")

    groups_by_name: dict[str, dict[str, Any]] = {}
    categories = {
        "forbidden_objects": [],
        "allowed_target_objects": [],
        "other_scene_objects": [],
    }
    covered_bodies: set[int] = set()
    for object_id, binding in bindings.items():
        body_ids = sorted(int(value) for value in binding.get("collision_body_ids", []))
        geom_ids = sorted(int(value) for value in binding.get("geom_ids", []))
        if not body_ids or not geom_ids:
            if object_id in required:
                raise ValueError(f"active_BDDL_screen_policy_entity_has_no_collision_geometry:{object_id}")
            continue
        category = (
            "forbidden_objects" if object_id in forbidden_robot
            else "allowed_target_objects" if object_id in goal_targets
            else "other_scene_objects"
        )
        margin = 0.010 if category != "other_scene_objects" else 0.0
        group = {
            "object_id": str(object_id),
            "body_ids": body_ids,
            "body_names": [str(value) for value in binding.get("collision_body_names", [])],
            "geom_ids": geom_ids,
            "minimum_clearance_m": margin,
        }
        if len(group["body_names"]) != len(body_ids):
            raise ValueError(f"active_BDDL_screen_policy_body_names_incomplete:{object_id}")
        categories[category].append(group)
        groups_by_name[str(object_id)] = group
        covered_bodies.update(body_ids)

    scene_geoms = [
        item for item in geometry.get("geometries", [])
        if item.get("collision_enabled") is True and item.get("category") == "scene"
    ]
    all_scene_bodies = {int(item["body_id"]) for item in scene_geoms}
    residual_bodies = sorted(all_scene_bodies - covered_bodies)
    if residual_bodies:
        body_names = {
            int(item["body_id"]): str(item["body_name"])
            for item in geometry.get("bodies", [])
        }
        residual_geom_ids = sorted(
            int(item["geom_id"]) for item in scene_geoms
            if int(item["body_id"]) in set(residual_bodies)
        )
        categories["other_scene_objects"].append({
            "object_id": "unmapped_scene_environment",
            "body_ids": residual_bodies,
            "body_names": [body_names.get(value, f"body_{value}") for value in residual_bodies],
            "geom_ids": residual_geom_ids,
            "minimum_clearance_m": 0.0,
        })
    if not categories["forbidden_objects"] or not categories["allowed_target_objects"]:
        raise ValueError("active_BDDL_screen_policy_collision_groups_are_incomplete")

    target_binding = bindings[target_id]
    pad_geom_ids = sorted({
        int(item["geom_id"])
        for side in context["fingertip_pad_geometries"].values()
        for item in side
    })
    if len(pad_geom_ids) != 2:
        raise ValueError("strict_grasp_screen_requires_two_active_Panda_fingertip_pad_geometries")
    allowed_pairs = [
        {
            "robot_geom_id": pad_id,
            "scene_geom_id": int(scene_id),
            "minimum_clearance_m": -0.002,
        }
        for pad_id in pad_geom_ids
        for scene_id in target_binding["geom_ids"]
    ]
    normalized_forbidden_pairs = []
    for object_a, object_b in forbidden_pairs:
        if object_a not in groups_by_name or object_b not in groups_by_name:
            raise ValueError(f"active_BDDL_forbidden_pair_has_unclassified_geometry:{object_a}:{object_b}")
        a, b = groups_by_name[object_a], groups_by_name[object_b]
        normalized_forbidden_pairs.append({
            "object_id_a": object_a,
            "object_id_b": object_b,
            "body_ids_a": a["body_ids"],
            "body_ids_b": b["body_ids"],
            "geom_ids_a": a["geom_ids"],
            "geom_ids_b": b["geom_ids"],
        })
    policy = {
        "source": "active_BDDL_CheckRobotContact_CheckContact_candidate_screen_only",
        "minimum_clearance_m": 0.010,
        "forbidden_objects": categories["forbidden_objects"],
        "allowed_target_objects": categories["allowed_target_objects"],
        "other_scene_objects": categories["other_scene_objects"],
        "allowed_contact_pairs": allowed_pairs,
        "forbidden_object_pairs": normalized_forbidden_pairs,
        "allowed_carried_support_contact_geom_ids": [],
        "carried_object": None,
    }
    semantics = {
        "forbidden_robot_contact_objects": sorted(forbidden_robot),
        "forbidden_object_contact_pairs": [list(pair) for pair in forbidden_pairs],
        "goal_target_objects": sorted(goal_targets),
        "support_contacts_added_as_exceptions": False,
        "source_and_goal_support_semantics": "recorded from BDDL relations; no additional support exception is introduced by this candidate screen",
        "nonpad_target_clearance_m": 0.010,
        "authorized_pad_target_penetration_m": -0.002,
    }
    return policy, semantics


def _record_step(
    bridge: BridgeSession,
    action: np.ndarray,
    phase: str,
    candidate_id: str,
    action_index: int,
    target_binding: dict[str, Any],
    instruction: str,
    policy: dict[str, Any],
    monitor,
    records: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    *,
    raise_on_done: bool = True,
    planning_tracking: dict[str, Any] | None = None,
) -> None:
    reply = bridge.request({
        "op": "step_monitored",
        "action": np.asarray(action, dtype=np.float32).tolist(),
        "action_index": int(action_index),
        "safety_geometry_policy": policy,
    })
    try:
        grasp = bridge.request({
            "op": "grasp_state",
            "carried_binding": target_binding,
            "expected_instruction": instruction,
        })
    except Exception as exc:
        grasp = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    sample = monitor.update(grasp, rollout_step=action_index)
    contact_monitoring = reply.get("contact_monitoring", {})
    monitoring_complete = _contact_substep_monitoring_complete(contact_monitoring)
    snapshot_state = reply.get("snapshot_state", {})
    record = {
        "candidate_id": candidate_id,
        "step_index": len(records),
        "phase": phase,
        "action_7d": np.asarray(action, dtype=np.float32).tolist(),
        "state": reply.get("state", {}),
        "snapshot_state": snapshot_state,
        "target_body_id": int(target_binding["body_id"]),
        "contacts": reply.get("contacts", {}),
        "start_contacts": reply.get("start_contacts", {}),
        "contact_monitoring": contact_monitoring,
        "substep_monitoring_complete": monitoring_complete,
        "safety_geometry": reply.get("safety_geometry"),
        "grasp_state": grasp,
        "grasp_monitor_sample": sample,
        "reward": reply.get("reward", 0.0),
        "done": reply.get("done", False),
        "task_success": reply.get("task_success"),
        "timestep": reply.get("timestep"),
    }
    if planning_tracking is not None:
        tracking = dict(planning_tracking)
        actual_joint_qpos = snapshot_state.get("robot_states", [{}])[0].get("joint_qpos")
        reference_joint_qpos = tracking.get("rrt_reference_qpos_rad")
        if actual_joint_qpos is not None:
            tracking["actual_joint_qpos_after_rad"] = actual_joint_qpos
        if actual_joint_qpos is not None and reference_joint_qpos is not None:
            actual_q = np.asarray(actual_joint_qpos, dtype=np.float64)
            reference_q = np.asarray(reference_joint_qpos, dtype=np.float64)
            if actual_q.shape == reference_q.shape == (7,):
                tracking["rrt_reference_joint_error_rad"] = float(np.linalg.norm(actual_q - reference_q))
        record["planning_tracking"] = tracking
    records.append(record)
    frames.append({
        "candidate_id": candidate_id,
        "step_index": len(records) - 1,
        "phase": phase,
        "agentview_rgb": _image(reply, "agentview_image"),
        "wrist_rgb": _image(reply, "robot0_eye_in_hand_image"),
    })
    if reply.get("done") and raise_on_done:
        raise RuntimeError("environment_done_during_grasp_only_branch")


def _contact_substep_monitoring_complete(contact_monitoring: dict[str, Any]) -> bool:
    """Check observation coverage separately from the observed safety verdict.

    The Safety Bridge marks measured geometry samples as ``unsafe`` when a
    clearance/contact rule is violated. Those samples are still present and
    must not be counted as missing monitoring.
    """
    if not isinstance(contact_monitoring, dict):
        return False
    expected = EXPECTED_MUJOCO_SUBSTEPS_PER_ACTION
    if (
        contact_monitoring.get("status") != "complete"
        or contact_monitoring.get("expected_physics_substeps_per_control_step") != expected
        or contact_monitoring.get("captured_physics_substeps") != expected
        or contact_monitoring.get("unobserved_expected_substeps", 0) != 0
        or contact_monitoring.get("unavailable_contact_substeps", 0) != 0
    ):
        return False
    control_steps = contact_monitoring.get("control_steps")
    if not isinstance(control_steps, list) or len(control_steps) != 1:
        return False
    for step in control_steps:
        if (
            step.get("status") != "complete"
            or step.get("expected_physics_substeps") != expected
            or step.get("captured_physics_substeps") != expected
            or step.get("unobserved_expected_substeps", 0) != 0
            or step.get("capture_error") is not None
        ):
            return False
    geometry = contact_monitoring.get("geometry_monitoring", {})
    substeps = geometry.get("substeps", []) if isinstance(geometry, dict) else []
    complete_geometry_statuses = {"complete", "unsafe"}
    return bool(
        geometry.get("status") in complete_geometry_statuses
        and len(substeps) == expected
        and all(
            isinstance(item, dict)
            and item.get("status") in complete_geometry_statuses
            for item in substeps
        )
    )


def _advance_pose(
    bridge: BridgeSession,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    phase: str,
    candidate_id: str,
    action_counter: list[int],
    maximum_steps: int,
    target_binding: dict[str, Any],
    instruction: str,
    policy: dict[str, Any],
    monitor,
    records: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    *,
    stop_condition=None,
    step_callback=None,
    stagnation_window_actions: int = 0,
    position_tolerance_m: float = 0.004,
    rotation_tolerance_rad: float = 0.045,
    planning_tracking: dict[str, Any] | None = None,
    target_pose_provider: Callable[
        [dict[str, Any]], tuple[np.ndarray, np.ndarray]
    ] | None = None,
    absolute_action_limit: int | None = MAX_BRANCH_CONTROL_ACTIONS,
) -> bool:
    from research.gs_opd.repair.trajectory_tracking import (
        quaternion_conjugate,
        quaternion_multiply,
        quaternion_to_rotation_vector,
        world_pose_tracking_action,
    )

    target_position = np.asarray(target_position, dtype=np.float64)
    target_quaternion = np.asarray(target_quaternion, dtype=np.float64)
    if (
        not np.isfinite([position_tolerance_m, rotation_tolerance_rad]).all()
        or position_tolerance_m <= 0.0 or rotation_tolerance_rad <= 0.0
    ):
        raise ValueError("pose_tracking_tolerances_must_be_positive_and_finite")
    progress_history: list[tuple[float, float]] = []
    for _ in range(maximum_steps):
        if stop_condition is not None and stop_condition():
            return False
        observed = bridge.request({"op": "observe", "include_snapshot_state": True})
        if target_pose_provider is not None:
            target_position, target_quaternion = target_pose_provider(
                observed.get("snapshot_state", {})
            )
            target_position = np.asarray(target_position, dtype=np.float64)
            target_quaternion = np.asarray(target_quaternion, dtype=np.float64)
            if (
                target_position.shape != (3,)
                or target_quaternion.shape != (4,)
                or not np.isfinite(target_position).all()
                or not np.isfinite(target_quaternion).all()
            ):
                raise ValueError("dynamic_target_pose_provider_returned_invalid_pose")
        robot = observed["snapshot_state"]["robot_states"][0]
        current_position = np.asarray(robot["eef_pos"], dtype=np.float64)
        current_quaternion = np.asarray(robot["eef_quat"], dtype=np.float64)
        position_error = target_position - current_position
        distance = float(np.linalg.norm(position_error))
        rotation_error = quaternion_to_rotation_vector(
            quaternion_multiply(target_quaternion, quaternion_conjugate(current_quaternion))
        )
        rotation_norm = float(np.linalg.norm(rotation_error))
        if _pose_tracking_error_within_tolerance(
            distance, rotation_norm, position_tolerance_m, rotation_tolerance_rad,
        ):
            return True
        if stagnation_window_actions:
            if (
                isinstance(stagnation_window_actions, bool)
                or not isinstance(stagnation_window_actions, int)
                or stagnation_window_actions < 2
            ):
                raise ValueError("pose_tracking_stagnation_window_must_be_zero_or_at_least_two")
            progress_history.append((distance, rotation_norm))
            progress_history = progress_history[-(stagnation_window_actions + 1):]
            if _pose_tracking_window_stagnated(
                progress_history, window_actions=stagnation_window_actions,
            ):
                raise RuntimeError(
                    f"{phase}_waypoint_stagnated_for_{stagnation_window_actions}_actions"
                )
        delta_position = position_error * min(1.0, 0.008 / max(distance, 1e-12))
        delta_rotation = rotation_error * min(1.0, 0.10 / max(rotation_norm, 1e-12))
        next_quaternion_xyzw = Rotation.from_rotvec(delta_rotation).as_quat()
        next_quaternion = quaternion_multiply(
            next_quaternion_xyzw[[3, 0, 1, 2]], current_quaternion
        )
        action = world_pose_tracking_action(
            current_position,
            current_quaternion,
            current_position + delta_position,
            next_quaternion,
            _advance_pose.controller_fields,
            gripper_action=0.0,
        )
        step_tracking = None
        if planning_tracking is not None:
            step_tracking = {
                **planning_tracking,
                "osc_command_target_eef_pose_world": np.r_[
                    current_position + delta_position, next_quaternion,
                ].tolist(),
                "active_subgoal_eef_pose_world": np.r_[target_position, target_quaternion].tolist(),
                "observed_eef_pose_before_action_world": np.r_[
                    current_position, current_quaternion,
                ].tolist(),
                "actual_joint_qpos_before_rad": robot.get("joint_qpos"),
                "requested_translation_delta_world_m": delta_position.tolist(),
                "requested_rotation_delta_world_rad": delta_rotation.tolist(),
            }
        _record_step(
            bridge,
            action,
            phase,
            candidate_id,
            action_counter[0],
            target_binding,
            instruction,
            policy,
            monitor,
            records,
            frames,
            raise_on_done=False,
            planning_tracking=step_tracking,
        )
        action_counter[0] += 1
        if step_callback is not None:
            step_callback(records[-1], frames[-1])
        if records[-1].get("done"):
            raise RuntimeError("environment_done_during_grasp_only_branch")
        if absolute_action_limit is not None and action_counter[0] > absolute_action_limit:
            raise RuntimeError("physical_branch_control_action_budget_exceeded")
        actual = np.asarray(
            records[-1]["snapshot_state"]["robot_states"][0]["eef_pos"], dtype=np.float64
        )
        if float(np.linalg.norm(actual - current_position)) > 0.030:
            raise RuntimeError("actual_EEF_displacement_exceeded_30mm_per_control_action")
        if stop_condition is not None and stop_condition():
            return False
    raise TimeoutError(f"{phase}_did_not_reach_pose_within_{maximum_steps}_actions")


def _pose_tracking_window_stagnated(
    history: list[tuple[float, float]],
    *,
    window_actions: int,
    minimum_position_progress_m: float = 0.006,
    minimum_rotation_progress_rad: float = 0.06,
) -> bool:
    """Detect a blocked OSC waypoint while allowing measured route progress."""
    if len(history) < window_actions + 1:
        return False
    window = history[-(window_actions + 1):]
    position_progress = max(0.0, window[0][0] - min(row[0] for row in window[1:]))
    rotation_progress = max(0.0, window[0][1] - min(row[1] for row in window[1:]))
    return bool(
        position_progress < minimum_position_progress_m
        and rotation_progress < minimum_rotation_progress_rad
    )


def _pose_tracking_error_within_tolerance(
    position_error_m: float,
    rotation_error_rad: float,
    position_tolerance_m: float,
    rotation_tolerance_rad: float,
) -> bool:
    return bool(
        np.isfinite([position_error_m, rotation_error_rad, position_tolerance_m, rotation_tolerance_rad]).all()
        and position_tolerance_m > 0.0
        and rotation_tolerance_rad > 0.0
        and position_error_m <= position_tolerance_m
        and rotation_error_rad <= rotation_tolerance_rad
    )


def _compress_eef_route_with_indices(route: Any) -> tuple[np.ndarray, list[int]]:
    poses = np.asarray(route, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError("RRT_EEF_route_is_not_a_finite_N_by_7_pose_array")
    keep = [0]
    last = 0
    for index in range(1, len(poses)):
        translation = float(np.linalg.norm(poses[index, :3] - poses[last, :3]))
        rotation = _rotation_error(poses[index, 3:], poses[last, 3:])
        if translation >= 0.020 or rotation >= 0.10 or index == len(poses) - 1:
            keep.append(index)
            last = index
    if len(keep) > MAX_PREFIX_CONTROL_ACTIONS:
        keep = sorted(set(np.linspace(0, len(poses) - 1, MAX_PREFIX_CONTROL_ACTIONS).round().astype(int).tolist()))
    return poses[keep], keep


def _compress_eef_route(route: Any) -> np.ndarray:
    """Compatibility wrapper retaining the original pose-only return value."""
    return _compress_eef_route_with_indices(route)[0]


def _run_physical_branch(
    bridge: BridgeSession,
    candidate: dict[str, Any],
    rrt: dict[str, Any],
    *,
    initial_payload: bytes,
    initial_state: np.ndarray,
    model_sha: str,
    context: dict[str, Any],
    target_binding: dict[str, Any],
    policy: dict[str, Any],
    controller_fields: dict[str, Any],
    monitor_type,
    records: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    output_dir: Path,
    abort_target_displacement_m: float | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    candidate_id = candidate["candidate_id"]
    _restore_initial(bridge, initial_payload, initial_state, model_sha)
    start_export = bridge.request({"op": "export_runtime_snapshot"})
    if start_export.get("status") != "complete" or not isinstance(start_export.get("payload"), bytes):
        raise RuntimeError("physical_branch_initial_snapshot_export_failed")
    if not np.array_equal(
        _snapshot_state_vector(start_export.get("snapshot_state", {})), initial_state
    ):
        raise RuntimeError("physical_branch_did_not_start_at_exact_archived_t0_state")
    start_path = output_dir / f"{candidate_id}_branch_start_runtime_snapshot.pkl"
    start_path.write_bytes(start_export["payload"])
    branch: dict[str, Any] = {
        "candidate_id": candidate_id,
        "status": "in_progress",
        "branch_start_snapshot": str(start_path),
        "route_safety_is_diagnostic_only": True,
        "safety_certified": False,
        "action_budget": MAX_BRANCH_CONTROL_ACTIONS,
    }
    action_counter = [0]
    monitor = monitor_type(required_samples=3)
    initial_grasp_state = bridge.request({
            "op": "grasp_state",
            "carried_binding": target_binding,
            "expected_instruction": context["instruction"],
        })
    monitor.update(initial_grasp_state, rollout_step=-1)
    initial_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
    initial_runtime_state = initial_observation.get("snapshot_state", {})
    initial_object = _object_state(
        initial_runtime_state, int(context["target_binding"]["root_body_id"])
    )
    initial_grasp_world = (
        _runtime_object_pose(initial_object)
        @ np.asarray(candidate["object_to_eef"], dtype=np.float64)
    )
    initial_eef_world = _runtime_eef_pose(initial_runtime_state)
    initial_pose_aligned, initial_grasp_distance_m, initial_grasp_rotation_error_rad = (
        _live_contact_pose_is_close_to_candidate(
            initial_eef_world, initial_grasp_world,
        )
    )
    initial_contact_sides = target_finger_contact_sides(initial_grasp_state)
    close_from_live_contact = bool(initial_contact_sides and initial_pose_aligned)
    branch["initial_target_finger_contact_sides"] = list(initial_contact_sides)
    branch["initial_eef_to_migrated_grasp_pose_m"] = initial_grasp_distance_m
    branch["initial_eef_to_migrated_grasp_rotation_error_rad"] = initial_grasp_rotation_error_rad
    branch["live_contact_close_pose_aligned_with_candidate"] = initial_pose_aligned
    branch["live_contact_close_pose_tolerance"] = {
        "position_error_max_m": LIVE_CONTACT_CLOSE_POSITION_TOLERANCE_M,
        "rotation_error_max_rad": LIVE_CONTACT_CLOSE_ROTATION_TOLERANCE_RAD,
    }
    branch["pregrasp_route_skipped_due_to_live_target_contact"] = close_from_live_contact
    if close_from_live_contact:
        if len(initial_contact_sides) == 2:
            settle_actions = 0
            settled_samples = 0
            maximum_linear_speed = 0.0
            maximum_angular_speed = 0.0
            settle_start_object = _object_state(
                initial_runtime_state, int(target_binding["body_id"])
            )
            settle_start_position = np.asarray(
                settle_start_object.get("root_pos"), dtype=np.float64
            )
            while settle_actions < MAX_LIVE_CONTACT_SETTLE_ACTIONS:
                object_state = _object_state(
                    initial_runtime_state, int(target_binding["body_id"])
                ) if settle_actions == 0 else _object_state(
                    records[-1]["snapshot_state"], int(target_binding["body_id"])
                )
                velocity = np.asarray(object_state.get("joint_qvel"), dtype=np.float64)
                if velocity.shape != (6,) or not np.isfinite(velocity).all():
                    raise RuntimeError("live_contact_settle_object_velocity_unavailable")
                linear_speed = float(np.linalg.norm(velocity[:3]))
                angular_speed = float(np.linalg.norm(velocity[3:]))
                maximum_linear_speed = max(maximum_linear_speed, linear_speed)
                maximum_angular_speed = max(maximum_angular_speed, angular_speed)
                if (
                    linear_speed <= LIVE_CONTACT_SETTLE_LINEAR_SPEED_M_S
                    and angular_speed <= LIVE_CONTACT_SETTLE_ANGULAR_SPEED_RAD_S
                ):
                    settled_samples += 1
                else:
                    settled_samples = 0
                if settled_samples >= LIVE_CONTACT_SETTLE_REQUIRED_SAMPLES:
                    break
                _record_step(
                    bridge,
                    np.zeros(7, dtype=np.float32),
                    "task_success_first_live_bilateral_contact_settle",
                    candidate_id,
                    action_counter[0],
                    target_binding,
                    context["instruction"],
                    policy,
                    monitor,
                    records,
                    frames,
                )
                action_counter[0] += 1
                settle_actions += 1
                settled_sides = target_finger_contact_sides(
                    records[-1].get("grasp_state", {})
                )
                if len(settled_sides) != 2:
                    branch["live_contact_settle_failure"] = {
                        "reason": "bilateral_target_contact_lost",
                        "actions": settle_actions,
                        "remaining_contact_sides": list(settled_sides),
                    }
                    raise RuntimeError("live_bilateral_target_contact_lost_during_settle")
            settle_end_state = (
                records[-1]["snapshot_state"] if settle_actions else initial_runtime_state
            )
            settle_end_object = _object_state(
                settle_end_state, int(target_binding["body_id"])
            )
            branch["live_contact_settle"] = {
                "actions": settle_actions,
                "status": (
                    "settled_with_bilateral_contact"
                    if settled_samples >= LIVE_CONTACT_SETTLE_REQUIRED_SAMPLES
                    else "action_budget_exhausted_bilateral_contact_preserved"
                ),
                "required_consecutive_low_velocity_samples": LIVE_CONTACT_SETTLE_REQUIRED_SAMPLES,
                "linear_speed_threshold_m_s": LIVE_CONTACT_SETTLE_LINEAR_SPEED_M_S,
                "angular_speed_threshold_rad_s": LIVE_CONTACT_SETTLE_ANGULAR_SPEED_RAD_S,
                "maximum_observed_linear_speed_m_s": maximum_linear_speed,
                "maximum_observed_angular_speed_rad_s": maximum_angular_speed,
                "consecutive_low_velocity_samples_at_end": settled_samples,
                "object_displacement_m": float(np.linalg.norm(
                    np.asarray(settle_end_object["root_pos"], dtype=np.float64)
                    - settle_start_position
                )),
                "state_was_only_observed_and_held_with_native_zero_action": True,
            }
        else:
            branch["live_contact_settle"] = {
                "actions": 0,
                "status": "skipped_without_bilateral_target_contact",
            }
        branch["live_contact_close_probe_action_budget"] = MAX_CLOSE_CONTROL_ACTIONS
        live_close_actions = 0
        while live_close_actions < MAX_CLOSE_CONTROL_ACTIONS and not monitor.stable:
            _record_step(
                bridge,
                np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
                (
                    "task_success_first_live_bilateral_contact_close_probe"
                    if len(initial_contact_sides) == 2
                    else "task_success_first_live_unilateral_contact_close_probe"
                ),
                candidate_id,
                action_counter[0],
                target_binding,
                context["instruction"],
                policy,
                monitor,
                records,
                frames,
            )
            action_counter[0] += 1
            live_close_actions += 1
        branch["live_contact_close_probe_actions"] = live_close_actions
        branch["live_contact_close_probe_grasp"] = monitor.diagnostics()
        if not monitor.stable:
            raise RuntimeError("live_target_contact_close_probe_did_not_stabilize")
        branch["live_contact_close_probe_succeeded"] = True
    else:
        branch["live_contact_close_probe_succeeded"] = False
    _advance_pose.controller_fields = controller_fields
    route = _compress_eef_route(rrt.get("eef_reference_poses_world"))
    branch["rrt_eef_waypoint_count"] = int(len(route))
    route_action_budget = route_control_action_budget(
        route,
        minimum_actions=74,
        maximum_actions=MAX_PREFIX_CONTROL_ACTIONS,
    )
    branch["pregrasp_route_action_budget"] = route_action_budget
    branch["local_replans"] = []

    def advance_with_local_replan(
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        phase: str,
        maximum_steps: int,
        target_pose_provider: Callable[
            [dict[str, Any]], tuple[np.ndarray, np.ndarray]
        ] | None = None,
    ) -> None:
        try:
            _advance_pose(
                bridge, target_position, target_quaternion, phase, candidate_id,
                action_counter, maximum_steps, target_binding, context["instruction"],
                policy, monitor, records, frames,
                target_pose_provider=target_pose_provider,
            )
            return
        except TimeoutError as original_error:
            if target_pose_provider is not None:
                branch["local_replans"].append({
                    "phase": phase,
                    "status": "dynamic_object_relative_target_did_not_converge",
                    "failure_reason": str(original_error),
                    "safety_certified": False,
                })
                raise original_error
            if len(branch["local_replans"]) >= 2:
                raise original_error
            remaining_budget = MAX_BRANCH_CONTROL_ACTIONS - action_counter[0]
            if remaining_budget <= 0:
                raise original_error
            observed_now = bridge.request({"op": "observe", "include_snapshot_state": True})
            current_state = observed_now.get("snapshot_state", {})
            current_object = _object_state(current_state, int(target_binding["body_id"]))
            source_object_position = np.asarray(
                context["target_root_pose_world"]["position_m"], dtype=np.float64
            )
            object_displacement = float(np.linalg.norm(
                np.asarray(current_object["root_pos"], dtype=np.float64) - source_object_position
            ))
            if object_displacement > 0.005:
                branch["local_replans"].append({
                    "phase": phase,
                    "status": "not_attempted_target_object_moved",
                    "object_displacement_m": object_displacement,
                    "failure_reason": str(original_error),
                })
                raise original_error
            exported = bridge.request({"op": "export_runtime_snapshot"})
            if exported.get("status") != "complete" or not isinstance(exported.get("payload"), bytes):
                branch["local_replans"].append({
                    "phase": phase,
                    "status": "not_attempted_snapshot_export_failed",
                    "failure_reason": str(original_error),
                })
                raise original_error
            snapshot_path = output_dir / (
                f"{candidate_id}_local_replan_{len(branch['local_replans']) + 1}_runtime_snapshot.pkl"
            )
            snapshot_path.write_bytes(exported["payload"])
            live_integration = _snapshot_state_vector(current_state)
            snapshot_integration = _snapshot_state_vector(exported.get("snapshot_state", {}))
            if (
                exported.get("model_geometry_sha256") != model_sha
                or not np.array_equal(live_integration, snapshot_integration)
            ):
                branch["local_replans"].append({
                    "phase": phase,
                    "status": "not_attempted_snapshot_identity_mismatch",
                    "snapshot": str(snapshot_path),
                    "failure_reason": str(original_error),
                })
                raise original_error
            start_pose = np.r_[
                current_state["robot_states"][0]["eef_pos"],
                current_state["robot_states"][0]["eef_quat"],
            ].astype(np.float64)
            target_pose = np.r_[
                np.asarray(target_position, dtype=np.float64),
                np.asarray(target_quaternion, dtype=np.float64),
            ]
            plan = bridge.request({
                "op": "task_success_first_rrt",
                "expected_instruction": context["instruction"],
                "model_geometry_sha256": model_sha,
                "start_state_sha256": _state_sha(live_integration),
                "goal_pose_world": {
                    "position_world_m": target_pose[:3].tolist(),
                    "quaternion_wxyz": target_pose[3:].tolist(),
                },
                "safety_geometry_policy": policy,
                "planner_config": {
                    "step_rad": 0.22,
                    "connect_max_steps": 12,
                    "goal_bias": 0.2,
                    "max_iterations": 400,
                    "max_config_checks": 4000,
                    "max_candidate_routes": 1,
                    "search_time_budget_s": 2.0,
                    "shortcut_attempts": 20,
                    "minimum_clearance_m": 0.0,
                    "require_safe_configurations": False,
                    "seed": int(20261026 + len(branch["local_replans"])),
                },
            })
            if (
                plan.get("status") == "offline_feasible"
                and plan.get("live_state_unchanged") is True
                and plan.get("eef_reference_poses_world")
            ):
                replan_route = _compress_eef_route(plan["eef_reference_poses_world"])
                route_source = "local_RRT_from_verified_failure_snapshot"
            else:
                replan_route = np.asarray([start_pose, target_pose], dtype=np.float64)
                route_source = "direct_OSC_segment_after_local_RRT_failure"
            route_action_cap = MAX_BRANCH_CONTROL_ACTIONS - action_counter[0]
            branch["local_replans"].append({
                "phase": phase,
                "status": "physical_replan_started",
                "failure_reason": str(original_error),
                "snapshot": str(snapshot_path),
                "snapshot_sha256": _sha256_bytes(exported["payload"]),
                "object_displacement_m": object_displacement,
                "RRT_status": plan.get("status"),
                "RRT_failure_reason": plan.get("failure_reason"),
                "route_source": route_source,
                "remaining_physical_action_budget": route_action_cap,
                "safety_certified": False,
            })
            candidate["_local_replans"] = list(branch["local_replans"])
            for local_waypoint in replan_route[1:]:
                local_remaining = MAX_BRANCH_CONTROL_ACTIONS - action_counter[0]
                if local_remaining <= 0:
                    raise RuntimeError("local_segment_replan_physical_action_budget_exhausted")
                _advance_pose(
                    bridge,
                    local_waypoint[:3],
                    local_waypoint[3:],
                    f"local_replan_{phase}",
                    candidate_id,
                    action_counter,
                    local_remaining,
                    target_binding,
                    context["instruction"],
                    policy,
                    monitor,
                    records,
                    frames,
                )
            branch["local_replans"][-1]["status"] = "physical_replan_segment_reached"
            candidate["_local_replans"] = list(branch["local_replans"])

    if not close_from_live_contact:
        for waypoint in route[1:]:
            advance_with_local_replan(
                waypoint[:3],
                waypoint[3:],
                "task_success_first_rrt_osc_pregrasp_route",
                max(1, route_action_budget - action_counter[0]),
            )
            if action_counter[0] > route_action_budget:
                raise RuntimeError("RRT_OSC_pregrasp_route_action_budget_exceeded")
    branch["pregrasp_route_reached"] = True

    root_pose = _pose(
        context["target_root_pose_world"]["position_m"],
        context["target_root_pose_world"]["rotation"],
    )
    object_to_eef = np.asarray(candidate["object_to_eef"], dtype=np.float64)
    grasp_world = root_pose @ object_to_eef

    def object_relative_target_provider(
        object_to_target: np.ndarray,
    ) -> Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]]:
        relative_transform = np.asarray(object_to_target, dtype=np.float64)

        def provide(snapshot_state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
            if abort_target_displacement_m is not None:
                obj = _object_state(snapshot_state, int(target_binding["body_id"]))
                position = np.asarray(obj["root_pos"], dtype=np.float64)
                initial = np.asarray(context["target_root_pose_world"]["position_m"], dtype=np.float64)
                displacement = float(np.linalg.norm(position - initial))
                if displacement > abort_target_displacement_m:
                    raise RuntimeError(
                        f"ungrasped_target_displaced_during_approach:{displacement:.6f}m"
                    )
            return _object_relative_target_pose(
                snapshot_state,
                int(target_binding["body_id"]),
                relative_transform,
            )

        return provide

    if not close_from_live_contact:
        approach_start_actions = action_counter[0]
        migrated_waypoints = candidate.get("historical_approach_waypoints_object_xyz_quaternion_wxyz", [])
        if migrated_waypoints:
            branch["historical_object_relative_approach_waypoints_migrated"] = len(migrated_waypoints)
            for waypoint in migrated_waypoints:
                relative_pose = np.eye(4, dtype=np.float64)
                relative_pose[:3, :3] = Rotation.from_quat(
                    np.asarray(waypoint[3:], dtype=np.float64)[[1, 2, 3, 0]]
                ).as_matrix()
                relative_pose[:3, 3] = np.asarray(waypoint[:3], dtype=np.float64)
                world_waypoint = root_pose @ relative_pose
                remaining = MAX_APPROACH_CONTROL_ACTIONS - (action_counter[0] - approach_start_actions)
                if remaining <= 0:
                    raise RuntimeError("migrated_object_relative_approach_action_budget_exhausted")
                advance_with_local_replan(
                    world_waypoint[:3, 3],
                    _wxyz(world_waypoint[:3, :3]),
                    "migrated_object_relative_approach",
                    remaining,
                    target_pose_provider=object_relative_target_provider(relative_pose),
                )
            remaining = MAX_APPROACH_CONTROL_ACTIONS - (action_counter[0] - approach_start_actions)
            if remaining <= 0:
                raise RuntimeError("migrated_object_relative_grasp_endpoint_action_budget_exhausted")
        else:
            remaining = MAX_APPROACH_CONTROL_ACTIONS
        advance_with_local_replan(
            grasp_world[:3, 3],
            _wxyz(grasp_world[:3, :3]),
            "task_success_first_surface_approach" if not migrated_waypoints else "migrated_object_relative_grasp_endpoint",
            remaining,
            target_pose_provider=object_relative_target_provider(object_to_eef),
        )
        branch["surface_approach_target_frame"] = "live_target_object_root_pose_times_generated_object_to_eef"
    branch["surface_approach_reached"] = True

    close_actions = 0
    while close_actions < MAX_CLOSE_CONTROL_ACTIONS and not monitor.stable:
        action = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        _record_step(
            bridge,
            action,
            "task_success_first_bilateral_close",
            candidate_id,
            action_counter[0],
            target_binding,
            context["instruction"],
            policy,
            monitor,
            records,
            frames,
        )
        action_counter[0] += 1
        close_actions += 1
        if action_counter[0] > MAX_BRANCH_CONTROL_ACTIONS:
            raise RuntimeError("physical_branch_control_action_budget_exceeded")
    branch["close_actions"] = close_actions
    branch["grasp_before_lift"] = monitor.diagnostics()
    if not monitor.stable:
        raise RuntimeError("bilateral_grasp_not_stable_within_close_action_budget")

    closed = bridge.request({"op": "export_runtime_snapshot"})
    if closed.get("status") != "complete" or not isinstance(closed.get("payload"), bytes):
        raise RuntimeError("closed_grasp_runtime_snapshot_export_failed")
    closed_path = output_dir / f"{candidate_id}_closed_runtime_snapshot.pkl"
    closed_path.write_bytes(closed["payload"])
    branch["closed_grasp_snapshot"] = str(closed_path)

    before = bridge.request({"op": "observe", "include_snapshot_state": True})
    target_before = _object_state(before["snapshot_state"], int(context["target_binding"]["root_body_id"]))
    object_pose_at_grasp = _runtime_object_pose(target_before)
    eef_pose_at_grasp = _runtime_eef_pose(before["snapshot_state"])
    branch["verified_object_to_eef_at_grasp"] = (
        np.linalg.inv(object_pose_at_grasp) @ eef_pose_at_grasp
    ).tolist()
    eef_before = np.asarray(before["snapshot_state"]["robot_states"][0]["eef_pos"], dtype=np.float64)
    orientation = np.asarray(before["snapshot_state"]["robot_states"][0]["eef_quat"], dtype=np.float64)
    _advance_pose(
        bridge,
        eef_before + np.asarray([0.0, 0.0, LIFT_COMMAND_M]),
        orientation,
        "task_success_first_40mm_lift",
        candidate_id,
        action_counter,
        MAX_LIFT_CONTROL_ACTIONS,
        target_binding,
        context["instruction"],
        policy,
        monitor,
        records,
        frames,
    )
    after = bridge.request({"op": "observe", "include_snapshot_state": True})
    target_after = _object_state(after["snapshot_state"], int(context["target_binding"]["root_body_id"]))
    lift_delta = float(target_after["root_pos"][2] - target_before["root_pos"][2])
    branch.update({
        "target_root_position_before_lift_m": target_before["root_pos"],
        "target_root_position_after_lift_m": target_after["root_pos"],
        "target_lift_delta_m": lift_delta,
        "grasp_after_lift": monitor.diagnostics(),
        "branch_control_actions": action_counter[0],
    })
    if action_counter[0] > MAX_BRANCH_CONTROL_ACTIONS:
        raise RuntimeError("physical_branch_control_action_budget_exceeded")
    if lift_delta < MINIMUM_OBJECT_LIFT_M:
        raise RuntimeError(f"object_lift_below_30mm:{lift_delta:.6f}m")
    if not monitor.stable:
        raise RuntimeError("bilateral_grasp_did_not_remain_stable_through_lift")
    final = bridge.request({"op": "export_runtime_snapshot"})
    if final.get("status") != "complete" or not isinstance(final.get("payload"), bytes):
        raise RuntimeError("verified_lifted_grasp_snapshot_export_failed")
    final_state = _snapshot_state_vector(final.get("snapshot_state", {}))
    if final.get("model_geometry_sha256") != model_sha:
        raise RuntimeError("lifted_snapshot_model_fingerprint_changed")
    final_path = output_dir / "task_success_candidate_lifted_runtime_snapshot.pkl"
    final_path.write_bytes(final["payload"])
    branch["final_snapshot"] = {
        "path": str(final_path),
        "sha256": _sha256_file(final_path),
        "integration_state_sha256": _sha256_bytes(np.ascontiguousarray(final_state).tobytes()),
        "snapshot_state": final.get("snapshot_state"),
    }
    branch["status"] = "task_success_candidate_bilateral_grasp_and_30mm_lift"
    branch["task_success_candidate"] = True
    branch["safety_certified"] = False
    return branch, final["payload"]


def _write_recording(output_dir: Path, records: list[dict[str, Any]], frames: list[dict[str, Any]]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    compressed = output_dir / "physical_step_records.jsonl.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=lambda item: item.tolist() if isinstance(item, np.ndarray) else str(item)))
            handle.write("\n")
    actions = np.asarray([item["action_7d"] for item in records], dtype=np.float32).reshape(-1, 7)
    robot_states = [item.get("snapshot_state", {}).get("robot_states", [{}])[0] for item in records]
    targets = []
    for row in records:
        target = next(
            (
                value for value in row.get("snapshot_state", {}).get("object_states", {}).values()
                if value.get("body_id") == row.get("target_body_id")
            ),
            None,
        )
        targets.append(target or {})
    arrays = {
        "candidate_ids": np.asarray([item["candidate_id"] for item in records]),
        "phases": np.asarray([item["phase"] for item in records]),
        "actions_7d": actions,
        "eef_position_world_m": np.asarray([item.get("eef_pos", [np.nan] * 3) for item in robot_states], dtype=np.float64).reshape(-1, 3),
        "eef_quaternion_wxyz": np.asarray([item.get("eef_quat", [np.nan] * 4) for item in robot_states], dtype=np.float64).reshape(-1, 4),
        "arm_joint_qpos_rad": np.asarray([item.get("joint_qpos", [np.nan] * 7) for item in robot_states], dtype=np.float64).reshape(-1, 7),
        "gripper_qpos": np.asarray([item.get("gripper_qpos", [np.nan] * 2) for item in robot_states], dtype=np.float64).reshape(-1, 2),
        "target_root_position_world_m": np.asarray([item.get("root_pos", [np.nan] * 3) for item in targets], dtype=np.float64).reshape(-1, 3),
        "target_root_quaternion_wxyz": np.asarray([item.get("root_quat", [np.nan] * 4) for item in targets], dtype=np.float64).reshape(-1, 4),
        "rewards": np.asarray([item.get("reward", 0.0) for item in records], dtype=np.float32),
        "dones": np.asarray([item.get("done", False) for item in records], dtype=bool),
        "task_success": np.asarray([item.get("task_success", False) for item in records], dtype=bool),
    }
    np.savez_compressed(output_dir / "task_success_first_actions_states.npz", **arrays)
    video_path = output_dir / "dual_camera_agentview_wrist_20fps.mp4"
    if frames:
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
            (frames[0]["agentview_rgb"].shape[1] * 2, frames[0]["agentview_rgb"].shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the dual-camera grasp video")
        for frame in frames:
            combined = np.concatenate((frame["agentview_rgb"], frame["wrist_rgb"]), axis=1)
            writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        writer.release()
    _write_json(output_dir / "video_frame_index.json", [
        {"index": index, "candidate_id": item["candidate_id"], "step_index": item["step_index"], "phase": item["phase"]}
        for index, item in enumerate(frames)
    ])
    return {
        "physical_control_actions": int(len(records)),
        "action_shape": list(actions.shape),
        "captured_physics_substeps": int(sum(
            item.get("contact_monitoring", {}).get("captured_physics_substeps", 0)
            for item in records
        )),
        "actions_with_incomplete_substep_monitoring": int(sum(
            not bool(item.get("substep_monitoring_complete", False)) for item in records
        )),
        "dual_camera_video_frame_count": int(len(frames)),
        "per_action_records_gzip": str(compressed),
        "actions_and_robot_states_npz": str(output_dir / "task_success_first_actions_states.npz"),
        "dual_camera_video": str(video_path) if frames else None,
    }


def _state_sha(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value, dtype=np.float64).tobytes())


def _cold_start_geometry_fallback_allowed(cold_start: bool, candidate: dict[str, Any], planner: dict[str, Any]) -> bool:
    return bool(cold_start and generated_geometry_high_approach_admitted(candidate, planner))


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_started = time.monotonic()
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor

    cold_start = bool(getattr(args, "cold_start", False))
    if cold_start and (
        args.experience_library is not None
        or args.previous_search_report
        or args.historical_grasp_pose_priors_json is not None
        or args.strict_grasp_screen_only
        or args.strict_screen_reference_report is not None
        or args.resume_from_runtime_snapshot is not None
        or args.candidate_id
        or args.candidate_start_index != 0
        or args.candidate_tilt_degrees
        or args.candidate_roll_degrees
        or args.candidate_offset_object_x_mm
        or args.candidate_offset_object_y_mm
        or args.candidate_offset_object_z_mm
    ):
        raise ValueError("cold_start_cannot_use_historical_experience_or_partial_search_inputs")
    snapshot_root = args.snapshot_root.resolve()
    snapshot_path = snapshot_root / "snapshots/initial_runtime_snapshot.pkl"
    manifest_path = snapshot_root / "snapshots/runtime_snapshot_manifest.json"
    bddl_path = snapshot_root / "task.bddl"
    if not all(path.is_file() for path in (snapshot_path, manifest_path, bddl_path)):
        raise FileNotFoundError("the_source_t0_runtime_snapshot_manifest_or_BDDL_is_missing")
    output_root = args.output_root.resolve()
    output_dir = output_root / (
        f"task_success_first_grasp_{args.benchmark}_L{args.level}T{args.task_id}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_meta = manifest.get("snapshots", {}).get("initial", {})
    payload = snapshot_path.read_bytes()
    if _sha256_bytes(payload) != source_meta.get("payload_sha256"):
        raise ValueError("source_t0_runtime_snapshot_payload_sha256_mismatch")
    if _sha256_file(bddl_path) != manifest.get("scene_metadata", {}).get("bddl_sha256"):
        raise ValueError("source_snapshot_BDDL_sha256_mismatch")
    envelope = pickle.loads(payload)
    model_sha = str(source_meta.get("model_geometry_sha256"))
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != model_sha
        or envelope.get("scene_metadata", {}).get("instruction") != manifest.get("instruction")
    ):
        raise ValueError("source_t0_runtime_snapshot_envelope_identity_mismatch")
    initial_state = _snapshot_state_vector({"mujoco_integration_state": envelope["snapshot"]["mujoco_state"]})
    if _state_sha(initial_state) != source_meta.get("mujoco_integration_state_sha256"):
        raise ValueError("source_t0_integration_state_sha256_mismatch")
    search_start_payload = payload
    search_start_state = initial_state
    resume_descriptor = None
    if args.resume_from_runtime_snapshot is not None:
        resume_descriptor = _verified_resume_runtime_snapshot(
            args.resume_from_runtime_snapshot,
            benchmark=args.benchmark,
            level=args.level,
            task_id=args.task_id,
            seed=args.seed,
            init_state=args.init_state,
            model_sha=model_sha,
            bddl_sha=str(manifest["scene_metadata"]["bddl_sha256"]),
            instruction=str(manifest["instruction"]),
        )
        search_start_payload = resume_descriptor["payload"]
        search_start_state = _snapshot_state_vector(resume_descriptor["snapshot_state"])
    (output_dir / "source_initial_runtime_snapshot.pkl").write_bytes(payload)
    (output_dir / "source_task.bddl").write_bytes(bddl_path.read_bytes())
    _write_json(output_dir / "source_snapshot_manifest.json", manifest)
    if resume_descriptor is not None:
        (output_dir / "source_resume_runtime_snapshot.pkl").write_bytes(search_start_payload)

    address = ("127.0.0.1", args.port)
    bridge = BridgeSession(address, output_dir / "safety_bridge.log", args)
    records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": "gs_opd_task_success_first_grasp_search.v1",
        "status": "preflight_in_progress",
        "cold_start_protocol": (
            {
                "enabled": True,
                "target_success_demonstration_used": False,
                "target_success_keyposes_used": False,
                "official_demonstration_used": False,
                "historical_actions_replayed": False,
                "initial_policy": "base_pi05",
                "search_stage": "geometry_search_after_base_pi05_attempt",
                "experience_library_loaded": False,
                "previous_search_reports_loaded": False,
                "historical_grasp_pose_priors_loaded": False,
                "start_state": "identity_checked_legal_t0_snapshot",
            }
            if cold_start else {"enabled": False}
        ),
        "scope": {
            "benchmark": args.benchmark,
            "level": args.level,
            "task_id": args.task_id,
            "init_state": args.init_state,
            "environment_seed": args.seed,
            "source_snapshot": str(snapshot_path),
            "source_snapshot_sha256": _sha256_bytes(payload),
            "search_start_runtime_snapshot": (
                str(resume_descriptor["path"]) if resume_descriptor is not None else str(snapshot_path)
            ),
            "search_start_runtime_snapshot_sha256": (
                resume_descriptor["payload_sha256"] if resume_descriptor is not None
                else _sha256_bytes(payload)
            ),
            "model_geometry_sha256": model_sha,
            "bddl_sha256": manifest["scene_metadata"]["bddl_sha256"],
        },
        "route_constraints": {
            "student_rollout_executed": False,
            "official_safety_demo_actions_keyframes_trajectories_or_labels_used": False,
            "LoRA_loaded": False,
            "object_pose_or_live_joint_qpos_written_directly": False,
            "strict_safety_acceptance_modified": False,
            "RRT_safety_gates_bypassed_only_in_this_task_success_first_mode": True,
            "search_results_are_automatically_safe": False,
            "training_or_LeRobot_conversion": False,
            "full_task_or_goal_region_execution": False,
            "official_safety_demonstrations_used": False,
            "target_success_demonstration_used": False if cold_start else None,
            "target_success_keyposes_used": False if cold_start else None,
            "historical_actions_replayed": False,
            "cross_scene_experience_library_path": (
                str(args.experience_library) if args.experience_library else None
            ),
        },
        "search_budget": {
            "geometry_candidates_max": args.maximum_candidates,
            "candidate_start_index": int(args.candidate_start_index),
            "candidate_generation_limit": (
                int(args.candidate_start_index + args.maximum_candidates)
                if args.strict_grasp_screen_only else int(args.maximum_candidates)
            ),
            "candidate_tilt_degrees": list(args.candidate_tilt_degrees),
            "candidate_roll_degrees": list(args.candidate_roll_degrees),
            "candidate_offset_object_x_mm": list(args.candidate_offset_object_x_mm),
            "candidate_offset_object_z_mm": list(args.candidate_offset_object_z_mm),
            "candidate_offset_object_y_mm": list(args.candidate_offset_object_y_mm),
            "candidate_coupled_perturbations": bool(args.candidate_coupled_perturbations),
            "historical_grasp_pose_priors_json": (
                str(args.historical_grasp_pose_priors_json.resolve())
                if args.historical_grasp_pose_priors_json else None
            ),
            "RRT_preflights_max": args.maximum_rrt_preflights,
            "physical_snapshot_branches_max": args.maximum_physical_branches,
            "branch_control_actions_max": MAX_BRANCH_CONTROL_ACTIONS,
            "pregrasp_route_actions_max": MAX_PREFIX_CONTROL_ACTIONS,
            "approach_actions_max": MAX_APPROACH_CONTROL_ACTIONS,
            "close_actions_max": MAX_CLOSE_CONTROL_ACTIONS,
            "live_contact_settle_actions_max": MAX_LIVE_CONTACT_SETTLE_ACTIONS,
            "live_contact_settle_linear_speed_threshold_m_s": LIVE_CONTACT_SETTLE_LINEAR_SPEED_M_S,
            "live_contact_settle_angular_speed_threshold_rad_s": LIVE_CONTACT_SETTLE_ANGULAR_SPEED_RAD_S,
            "live_contact_settle_required_consecutive_samples": LIVE_CONTACT_SETTLE_REQUIRED_SAMPLES,
            "lift_actions_max": MAX_LIFT_CONTROL_ACTIONS,
            "MuJoCo_substeps_per_control_action": EXPECTED_MUJOCO_SUBSTEPS_PER_ACTION,
            "candidate_id_filter": args.candidate_id,
        },
        "candidates": [],
        "physical_branches": [],
        "success": None,
    }
    if resume_descriptor is not None:
        report["resumed_from_runtime_snapshot"] = {
            "path": str(resume_descriptor["path"]),
            "payload_sha256": resume_descriptor["payload_sha256"],
            "integration_state_sha256": resume_descriptor["integration_state_sha256"],
            "status": "scene_and_model_identity_verified",
            "other_completed_native_predicates_preserved": True,
            "safety_certified": False,
        }
    final_snapshot_payload: bytes | None = None
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        if reset.get("scene_metadata", {}).get("model_geometry_sha256") != model_sha:
            raise RuntimeError("bridge_reset_model_does_not_match_archived_t0_snapshot")
        if reset.get("scene_metadata", {}).get("bddl_sha256") != manifest["scene_metadata"].get("bddl_sha256"):
            raise RuntimeError("bridge_reset_BDDL_does_not_match_archived_t0_snapshot")
        if cold_start:
            report["cold_start_reset_verification"] = _verify_cold_start_reset(bridge, initial_state)
        _restore_initial(bridge, search_start_payload, search_start_state, model_sha)
        initial = bridge.request({"op": "observe", "include_snapshot_state": True})
        instruction = str(manifest["instruction"])
        if initial.get("instruction") != instruction:
            raise RuntimeError("restored_snapshot_instruction_does_not_match_source_BDDL")
        native = bridge.request({"op": "native_task_status"})
        if native.get("status") != "complete" or native.get("task_success") is not False:
            raise RuntimeError("source_t0_native_task_state_is_not_false_or_complete")
        geometry = bridge.request({"op": "geometry_inventory"})
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        gripper = bridge.request({"op": "gripper_geometry_inventory"})
        limits = bridge.request({"op": "robot_joint_limit_inventory"})
        controller = bridge.request({"op": "repair_context"})
        fingerprint = bridge.request({"op": "compiled_model_fingerprint_details"})
        context = bridge.request({
            "op": "task_success_first_context",
            "expected_instruction": instruction,
            "target_predicate": args.target_predicate,
        })
        if context.get("status") != "complete" or context.get("read_only") is not True:
            raise RuntimeError(f"active_BDDL_target_or_pad_context_failed:{context.get('error')}")
        if geometry.get("status") != "complete" or semantic.get("status") != "complete":
            raise RuntimeError("active_geometry_inventory_or_semantic_binding_is_incomplete")
        if gripper.get("status") != "complete" or fingerprint.get("model_geometry_sha256") != model_sha:
            raise RuntimeError("active_gripper_or_model_fingerprint_verification_failed")
        if context.get("model_geometry_sha256") != model_sha:
            raise RuntimeError("active_target_context_model_fingerprint_mismatch")
        if context.get("native_goal_value") is not False:
            raise RuntimeError("source_t0_target_goal_predicate_is_not_false")
        goal_key = json.dumps(context["native_goal_predicate"], separators=(",", ":"))
        target_id = str(context["target_object_id"])
        imported_failure_memory: dict[str, Any] = {
            "source_reports": [],
            "identity_verified": False,
            "blocked_candidate_ids": [],
            "events": [],
        }
        blocked_candidate_ids: set[str] = set()
        imported_blocked_candidate_routes: set[tuple[str, str, str]] = set()
        previous_search_reports = getattr(args, "previous_search_report", None) or []
        if isinstance(previous_search_reports, (str, Path)):
            previous_search_reports = [Path(previous_search_reports)]
        imported_event_rows = []
        for previous_search_report in previous_search_reports:
            previous_search_report = Path(previous_search_report)
            previous = json.loads(previous_search_report.read_text(encoding="utf-8"))
            previous_scope = previous.get("scope", {})
            expected_identity = {
                "benchmark": args.benchmark,
                "level": args.level,
                "task_id": args.task_id,
                "init_state": args.init_state,
                "environment_seed": args.seed,
                "model_geometry_sha256": model_sha,
                "bddl_sha256": manifest["scene_metadata"]["bddl_sha256"],
            }
            mismatches = {
                key: {"expected": value, "actual": previous_scope.get(key)}
                for key, value in expected_identity.items()
                if previous_scope.get(key) != value
            }
            previous_goal = previous.get("active_target", {}).get("native_goal_predicate")
            if previous_goal != context.get("native_goal_predicate"):
                mismatches["native_goal_predicate"] = {
                    "expected": context.get("native_goal_predicate"),
                    "actual": previous_goal,
                }
            if mismatches:
                raise ValueError(f"previous_search_report_identity_mismatch:{mismatches}")
            report_blocked_ids, imported_events = blocked_candidate_ids_from_report(
                previous, goal_predicate=goal_key
            )
            blocked_candidate_ids.update(report_blocked_ids)
            imported_blocked_candidate_routes.update(
                (goal_key, str(event["candidate_id"]), str(event.get("route_id", "")))
                for event in imported_events if event.get("route_id")
            )
            imported_event_rows.extend({
                **event, "source_report": str(previous_search_report.resolve())
            } for event in imported_events)
            imported_failure_memory["source_reports"].append({
                "path": str(previous_search_report.resolve()),
                "source_report_status": previous.get("status"),
                "identity_verified": True,
            })
        if imported_failure_memory["source_reports"]:
            imported_failure_memory.update({
                "identity_verified": True,
                "blocked_candidate_ids": sorted(blocked_candidate_ids),
                "events": imported_event_rows,
            })
        target_semantic_binding = semantic["bindings"][target_id]
        target_binding = {
            "object_id": target_id,
            "body_id": int(target_semantic_binding["root_body_id"]),
            "body_name": str(target_semantic_binding["root_body_name"]),
            "geom_ids": sorted(int(value) for value in target_semantic_binding["root_geom_ids"]),
            "geom_names": list(target_semantic_binding["root_geom_names"]),
        }
        if cold_start:
            policy, policy_semantics = _build_bddl_grasp_screen_policy(
                context, semantic, geometry, bddl_path.read_text(encoding="utf-8"),
            )
            policy["source"] = "active_BDDL_contact_constraints_no_teacher_attestation"
            policy["measure_forbidden_object_pair_clearances"] = True
            report["safety_policy_provenance"] = {
                "source": "active_BDDL", "teacher_attested": False,
                "semantics": policy_semantics, "safety_certified": False,
            }
        else:
            policy = _build_safety_policy(context, semantic, geometry, gripper)
        initial_safety = bridge.request({"op": "safety_geometry_check", "safety_geometry_policy": policy})
        initial_contact_escape = robot_contact_escape_direction(
            initial_safety.get("contact_identities", []),
            ignore_object_id=str(context["target_object_id"]),
        )
        report["initial_robot_contact_escape"] = initial_contact_escape
        candidate_generation_limit = _candidate_generation_limit(args)
        candidates, generation = generate_geometry_candidates(
            context, maximum_candidates=candidate_generation_limit
        )
        experience_use = {
            "library_path": str(args.experience_library) if args.experience_library else None,
            "library_loaded": False,
            "matching_object_family_template_found": False,
            "template_id": None,
            "matching_template_ids": [],
            "inserted_template_ids": [],
            "operation_skill_id": None,
            "operation_phase_sequence": [],
            "migrated_grasp_prior_inserted": False,
            "historical_GS_OPD_data_used": False,
            "official_safety_demonstration_data_used": False,
            "matching_template_skipped_due_to_failure_memory": False,
        }
        if args.experience_library and args.experience_library.is_file():
            library = load_experience_library(args.experience_library)
            experience_use["library_loaded"] = True
            operation_skill = matching_operation_skill(library, target_id)
            if operation_skill is not None:
                experience_use.update({
                    "operation_skill_id": operation_skill.get("skill_id"),
                    "operation_phase_sequence": operation_skill.get("phase_sequence", []),
                    "operation_phase_structure_scheduled": True,
                })
            templates = matching_grasp_templates(library, target_id)
            if templates:
                template = templates[0]
                experience_use.update({
                    "matching_object_family_template_found": True,
                    "template_id": template["template_id"],
                    "matching_template_ids": [
                        str(item["template_id"]) for item in templates
                    ],
                })
                experience_candidates = []
                inserted_template_ids = []
                alternate_count = 0
                for template_rank, matched_template in enumerate(templates):
                    if len(experience_candidates) >= candidate_generation_limit:
                        break
                    prior = instantiate_grasp_candidate(
                        matched_template, context, source_library=str(args.experience_library)
                    )
                    prior["experience_template_rank"] = template_rank
                    experience_candidates.append(prior)
                    inserted_template_ids.append(str(matched_template["template_id"]))
                    if candidate_matches_failure_memory(
                        matched_template["template_id"], blocked_candidate_ids
                    ):
                        alternates = alternate_experience_approach_candidates(prior)
                        capacity = candidate_generation_limit - len(experience_candidates)
                        experience_candidates.extend(alternates[:capacity])
                        alternate_count += min(len(alternates), capacity)
                candidates = _combine_candidate_sources(
                    candidates,
                    experience_candidates,
                    generation_limit=candidate_generation_limit,
                    requested_candidate_ids=args.candidate_id,
                )
                experience_use.update({
                    "inserted_template_ids": inserted_template_ids,
                    "migrated_grasp_prior_inserted": bool(experience_candidates),
                    "historical_GS_OPD_data_used": bool(experience_candidates),
                    "historical_grasp_template_count_inserted": len(inserted_template_ids),
                    "alternate_object_relative_approach_candidates_inserted": alternate_count,
                    "exact_historical_route_skipped_due_to_failure_memory": bool(alternate_count),
                    "failure_memory_applied_at_candidate_route_pair_level": bool(alternate_count),
                    "approach_waypoints_migrated": experience_candidates[0].get(
                        "historical_approach_waypoints_object_xyz_quaternion_wxyz", []
                    ) if experience_candidates else [],
                })
        historical_pose_prior_set = None
        if args.historical_grasp_pose_priors_json is not None:
            candidates, historical_pose_prior_set = load_historical_grasp_pose_prior_candidates(
                args.historical_grasp_pose_priors_json,
                benchmark=args.benchmark,
                level=args.level,
                task_id=args.task_id,
                init_state=args.init_state,
                seed=args.seed,
                target_object_id=target_id,
                target_pose_world=_pose(
                    context["target_root_pose_world"]["position_m"],
                    context["target_root_pose_world"]["rotation"],
                ),
                eef_position_world=np.asarray(
                    context["eef_pose_world"]["position_m"], dtype=np.float64
                ),
                active_pad_center_eef_m=np.asarray(
                    context["symmetric_pad_center_eef_m"], dtype=np.float64
                ),
                maximum_candidates=args.maximum_candidates,
            )
            copied_prior_path = output_dir / "historical_grasp_pose_priors.json"
            copied_prior_path.write_bytes(args.historical_grasp_pose_priors_json.read_bytes())
            historical_pose_prior_set["archived_input_copy"] = str(copied_prior_path)
            report["historical_grasp_pose_prior_set"] = historical_pose_prior_set
            experience_use["hashed_success_episode_pose_priors_used"] = True
            experience_use["historical_pose_prior_set"] = historical_pose_prior_set
        report["experience_transfer"] = experience_use
        report["imported_search_failure_memory"] = imported_failure_memory
        generated_candidate_count = len(candidates)
        if args.candidate_id:
            generated_ids = {candidate.get("candidate_id") for candidate in candidates}
            missing_ids = set(args.candidate_id) - generated_ids
            if missing_ids:
                raise ValueError(f"requested_candidate_ids_not_generated:{sorted(missing_ids)}")
            candidates = [
                candidate for candidate in candidates
                if candidate.get("candidate_id") in set(args.candidate_id)
            ]
        orientation_angles = args.candidate_tilt_degrees or args.candidate_roll_degrees
        offsets_by_axis = {
            axis: values
            for axis, values in (
                ("x", args.candidate_offset_object_x_mm),
                ("y", args.candidate_offset_object_y_mm),
                ("z", args.candidate_offset_object_z_mm),
            )
            if values
        }
        offset_values = next(iter(offsets_by_axis.values())) if offsets_by_axis else []
        target_pose_for_variants = _pose(
            context["target_root_pose_world"]["position_m"],
            context["target_root_pose_world"]["rotation"],
        )
        eef_position_for_variants = np.asarray(
            context["eef_pose_world"]["position_m"], dtype=np.float64
        )
        offset_axis = next(iter(offsets_by_axis), "z")
        if args.candidate_coupled_perturbations:
            candidates = generate_grasp_coupled_pose_grid_variants(
                candidates,
                target_pose_world=target_pose_for_variants,
                eef_position_world=eef_position_for_variants,
                tilt_degrees=orientation_angles,
                rotation_axis="approach" if args.candidate_roll_degrees else "jaw",
                offsets_by_object_axis=offsets_by_axis,
            )
        elif orientation_angles:
            candidates = generate_grasp_tilt_variants(
                candidates,
                target_pose_world=target_pose_for_variants,
                eef_position_world=eef_position_for_variants,
                tilt_degrees=orientation_angles,
                rotation_axis="approach" if args.candidate_roll_degrees else "jaw",
            )
        elif offset_values:
            candidates = generate_grasp_surface_offset_variants(
                candidates,
                target_pose_world=target_pose_for_variants,
                eef_position_world=eef_position_for_variants,
                offset_object_mm=offset_values,
                offset_object_axis=offset_axis,
            )
        if args.strict_grasp_screen_only:
            candidates = _candidate_window(
                candidates,
                start=args.candidate_start_index,
                count=args.maximum_candidates,
            )
        candidates = diversify_candidates(candidates)
        if not candidates:
            raise RuntimeError("no_supported_collision_geometry_produced_a_pad_grasp_candidate")
        _write_json(output_dir / "geometry_candidates.json", candidates)
        _write_json(output_dir / "experience_transfer.json", experience_use)
        _write_json(output_dir / "active_target_context.json", context)
        _write_json(output_dir / "initial_safety_diagnostics.json", initial_safety)
        _write_json(output_dir / "safety_policy_diagnostic_only.json", policy)
        _advance_pose.controller_fields = controller["robots"][0]["fields"]

        if args.strict_grasp_screen_only:
            reference_report_path = args.strict_screen_reference_report.resolve()
            reference_report = json.loads(reference_report_path.read_text(encoding="utf-8"))
            reference_scope = reference_report.get("scope", {})
            expected_identity = {
                "benchmark": args.benchmark,
                "level": args.level,
                "task_id": args.task_id,
                "init_state": args.init_state,
                "environment_seed": args.seed,
                "model_geometry_sha256": model_sha,
                "bddl_sha256": manifest["scene_metadata"]["bddl_sha256"],
            }
            mismatches = {
                key: {"expected": value, "actual": reference_scope.get(key)}
                for key, value in expected_identity.items()
                if reference_scope.get(key) != value
            }
            stable_grasp = reference_report.get("success", {}).get("stable_bilateral_grasp", {})
            if reference_report.get("success", {}).get("target_object_id") != target_id:
                mismatches["target_object_id"] = {
                    "expected": target_id,
                    "actual": reference_report.get("success", {}).get("target_object_id"),
                }
            if stable_grasp.get("stable_grasp_observed") is not True:
                mismatches["stable_grasp_observed"] = {
                    "expected": True,
                    "actual": stable_grasp.get("stable_grasp_observed"),
                }
            if float(reference_report.get("success", {}).get("target_lift_delta_m") or 0.0) < MINIMUM_OBJECT_LIFT_M:
                mismatches["target_lift_delta_m"] = {
                    "expected_at_least_m": MINIMUM_OBJECT_LIFT_M,
                    "actual": reference_report.get("success", {}).get("target_lift_delta_m"),
                }
            if mismatches:
                raise ValueError(f"strict_screen_reference_identity_or_grasp_evidence_mismatch:{mismatches}")
            reference_npz = Path(
                reference_report.get("artifacts", {}).get("actions_and_robot_states_npz", "")
            ).resolve()
            if not reference_npz.is_file():
                raise FileNotFoundError("strict_screen_reference_action_state_npz_is_missing")
            reference_archive = Path(reference_report.get("archive", "")).resolve()
            hash_manifest_path = reference_archive / "artifact_hashes.json"
            if not hash_manifest_path.is_file():
                raise FileNotFoundError("strict_screen_reference_artifact_hash_manifest_is_missing")
            relative_reference_npz = reference_npz.relative_to(reference_archive).as_posix()
            reference_hashes = json.loads(hash_manifest_path.read_text(encoding="utf-8"))
            reference_npz_sha = _sha256_file(reference_npz)
            expected_npz_hash = reference_hashes.get(relative_reference_npz, {}).get("sha256")
            if expected_npz_hash != reference_npz_sha:
                raise ValueError("strict_screen_reference_action_state_npz_sha256_mismatch")
            with np.load(reference_npz, allow_pickle=False) as trace:
                required_trace_keys = {
                    "arm_joint_qpos_rad", "gripper_qpos", "target_root_position_world_m",
                    "target_root_quaternion_wxyz",
                }
                if not required_trace_keys.issubset(trace.files):
                    raise ValueError("strict_screen_reference_trace_is_missing_required_pose_or_gripper_arrays")
                reference_arm = np.asarray(trace["arm_joint_qpos_rad"], dtype=np.float64)
                reference_gripper = np.asarray(trace["gripper_qpos"], dtype=np.float64)
                reference_position = np.asarray(trace["target_root_position_world_m"], dtype=np.float64)
                reference_quaternion = np.asarray(trace["target_root_quaternion_wxyz"], dtype=np.float64)
            if (
                reference_arm.ndim != 2 or reference_arm.shape[1] != 7 or not len(reference_arm)
                or reference_gripper.ndim != 2 or reference_gripper.shape[1] != 2
                or reference_position.ndim != 2 or reference_position.shape[1] != 3
                or reference_quaternion.ndim != 2 or reference_quaternion.shape[1] != 4
                or len({len(reference_arm), len(reference_gripper), len(reference_position), len(reference_quaternion)}) != 1
                or not all(np.isfinite(value).all() for value in (reference_arm, reference_gripper, reference_position, reference_quaternion))
            ):
                raise ValueError("strict_screen_reference_trace_arrays_are_malformed")
            closed_gripper_qpos = reference_gripper[-1].tolist()
            reference_arm_qpos = reference_arm[-1].tolist()
            predicted_object_pose = np.eye(4, dtype=np.float64)
            predicted_object_pose[:3, :3] = Rotation.from_quat(
                reference_quaternion[-1][[1, 2, 3, 0]]
            ).as_matrix()
            predicted_object_pose[:3, 3] = reference_position[-1]
            for candidate in candidates:
                candidate["ik_seed_qpos"] = [
                    np.asarray(context["arm_qpos"], dtype=np.float64).tolist(),
                    reference_arm_qpos,
                ]
            screen_policy, task_safety_semantics = _build_bddl_grasp_screen_policy(
                context, semantic, geometry, bddl_path.read_text(encoding="utf-8")
            )
            state_sha = _state_sha(search_start_state)
            screen = bridge.request({
                "op": "screen_grasp_candidates",
                "expected_instruction": instruction,
                "model_geometry_sha256": model_sha,
                "start_state_sha256": state_sha,
                "safety_geometry_policy": screen_policy,
                "target_object_id": target_id,
                "target_root_body_id": int(target_semantic_binding["root_body_id"]),
                "left_finger_geom_ids": None,
                "right_finger_geom_ids": None,
                "clearance_m": 0.010,
                "strict_target_contact_policy": True,
                "closed_gripper_qpos": closed_gripper_qpos,
                "predicted_stable_object_pose_world": predicted_object_pose.tolist(),
                "candidates": candidates,
            })
            if screen.get("status") != "complete" or screen.get("live_state_unchanged") is not True:
                raise RuntimeError(
                    f"strict_offline_grasp_screen_incomplete:{screen.get('failure_reason') or screen.get('status')}"
                )
            strict_screen_report = {
                "schema_version": "gs_opd_strict_pad_only_grasp_screen.v1",
                "status": "complete",
                "task_safety_semantics": task_safety_semantics,
                "task": expected_identity,
                "native_goal_predicate": context["native_goal_predicate"],
                "start_snapshot_payload_sha256": _sha256_bytes(search_start_payload),
                "candidate_window": {
                    "start_index": int(args.candidate_start_index),
                    "screened_candidate_count": len(candidates),
                    "generated_candidate_count": generated_candidate_count,
                    "geometry_generation_limit": int(candidate_generation_limit),
                    "batch_limit": MAX_OFFLINE_GEOMETRY_SCREEN_BATCH,
                    "orientation_perturbation_count": len(orientation_angles),
                    "orientation_perturbation_axis": (
                        "approach" if args.candidate_roll_degrees else "jaw"
                    ) if orientation_angles else None,
                    "orientation_perturbation_degrees": list(orientation_angles),
                    "surface_contact_offset_mm": list(offset_values),
                    "surface_contact_offset_object_axis": (
                        offset_axis if len(offsets_by_axis) == 1
                        else ("multiple" if offsets_by_axis else None)
                    ),
                    "surface_contact_offsets_by_object_axis_mm": {
                        axis: list(values) for axis, values in offsets_by_axis.items()
                    },
                    "coupled_pose_perturbations": bool(args.candidate_coupled_perturbations),
                },
                "screen_policy_source": screen_policy["source"],
                "reference_prior": {
                    "report": str(reference_report_path),
                    "report_status": reference_report.get("status"),
                    "actions_and_robot_states_npz": str(reference_npz),
                    "npz_sha256": reference_npz_sha,
                    "stable_grasp_verified_in_source": True,
                    "used_only_for_closed_gripper_qpos_and_post_close_object_pose_and_IK_seed": True,
                    "source_actions_replayed": False,
                    "safety_qualification_inherited": False,
                },
                "screening": screen,
                "physical_actions": 0,
                "live_integration_state_unchanged": True,
                "safety_certified": False,
                "strict_safety_candidate_only": True,
                "route_preflight_performed": False,
                "next_step": (
                    "Run bounded exact-scene RRT preflight only for ranked candidates; physical execution requires a separate explicit selection and remains prohibited by this screen-only mode."
                ),
            }
            screen_path = output_dir / "strict_grasp_geometry_screen.json"
            _write_json(screen_path, strict_screen_report)
            report.update({
                "status": (
                    "offline_strict_grasp_candidates_available_route_preflight_pending"
                    if int(screen.get("feasible_count", 0)) > 0
                    else "no_strict_pad_only_grasp_candidate_in_generated_set"
                ),
                "strict_grasp_screen_only": True,
                "strict_grasp_geometry_screen": str(screen_path),
                "strict_grasp_geometry_screen_sha256": _sha256_file(screen_path),
                "strict_safety_screen_policy_source": screen_policy["source"],
                "safety_certified": False,
                "physical_actions": 0,
                "archive": str(output_dir),
                "artifacts": {
                    "physical_control_actions": 0,
                    "screen_report": str(screen_path),
                    "candidate_count": len(candidates),
                    "offline_feasible_candidate_count": int(screen.get("feasible_count", 0)),
                    "source_actions_replayed": False,
                },
            })
            _write_json(output_dir / "run_report.json", report)
            return report

        candidate_start = int(args.candidate_start_index)
        candidates_to_plan = candidates[candidate_start : candidate_start + args.maximum_rrt_preflights]
        state_sha = _state_sha(search_start_state)
        for candidate in candidates_to_plan:
            object_pose = _pose(
                context["target_root_pose_world"]["position_m"],
                context["target_root_pose_world"]["rotation"],
            )
            target_pose, pregrasp_pose = object_relative_grasp_targets(
                object_pose,
                candidate["object_to_eef"],
                candidate["approach_direction_object"],
                float(candidate["approach_distance_m"]),
            )
            pregrasp = pregrasp_pose[:3, 3]
            rrt = bridge.request({
                "op": "task_success_first_rrt",
                "expected_instruction": instruction,
                "model_geometry_sha256": model_sha,
                "start_state_sha256": state_sha,
                "goal_pose_world": {
                    "position_world_m": pregrasp.tolist(),
                    "quaternion_wxyz": _wxyz(target_pose[:3, :3]).tolist(),
                },
                "safety_geometry_policy": policy,
                "planner_config": {
                    "step_rad": 0.22,
                    "connect_max_steps": 20,
                    "goal_bias": 0.15,
                    "max_iterations": 1200,
                    "max_config_checks": 12000,
                    # Keep several record-only alternatives and select the
                    # one with the best diagnostic clearance before physical
                    # execution. A failed physical route still rolls back.
                    "max_candidate_routes": 3,
                    "search_time_budget_s": args.rrt_time_budget_s,
                    "shortcut_attempts": 40,
                    "minimum_clearance_m": 0.0,
                    "require_safe_configurations": False,
                    "seed": planner_seed(
                        int(args.seed),
                        int(candidate["target_geom_id"]),
                        len(report["candidates"]),
                    ),
                },
            })
            summary = {
                "candidate_id": candidate["candidate_id"],
                "candidate_geometry": candidate,
                "rrt_status": rrt.get("status"),
                "rrt_failure_reason": rrt.get("failure_reason"),
                "ik_candidates": rrt.get("ik_candidates", []),
                "rrt_planning_seconds": rrt.get("planning_seconds"),
                "rrt_configuration_checks": rrt.get("config_checks"),
                "rrt_minimum_diagnostic_clearance_m": rrt.get("minimum_forbidden_clearance_m"),
                "rrt_route_candidates": rrt.get("route_candidates", []),
                "rrt_live_state_unchanged": rrt.get("live_state_unchanged"),
                "rrt_safety_certified": False,
                "pregrasp_pose_world": np.r_[pregrasp, _wxyz(target_pose[:3, :3])].tolist(),
                "planner_response": rrt,
            }
            report["candidates"].append(summary)
        failure_memory = SearchFailureMemory()
        failure_memory.attempted.update(imported_blocked_candidate_routes)
        report["imported_failed_candidate_route_pairs"] = [
            {"goal_predicate": goal, "candidate_id": candidate_id, "route_id": route_id}
            for goal, candidate_id, route_id in sorted(imported_blocked_candidate_routes)
        ]
        report["candidate_routes_skipped_by_failure_memory"] = []
        physical_candidates = []
        seen_route_signatures: set[str] = set()
        route_options = []
        for candidate_index, (candidate, summary) in enumerate(zip(candidates_to_plan, report["candidates"])):
            planner_response = summary["planner_response"]
            routes = list(planner_response.get("route_candidates", []))
            start_pose = np.r_[
                context["eef_pose_world"]["position_m"],
                _wxyz(context["eef_pose_world"]["rotation"]),
            ]
            escape_pose = None
            if initial_contact_escape is not None:
                escape_pose = start_pose.copy()
                escape_pose[:3] += (
                    np.asarray(initial_contact_escape["direction_world"], dtype=np.float64) * 0.12
                )
            if not routes:
                planned_poses = planner_response.get("eef_reference_poses_world")
                if planned_poses:
                    routes = [{
                        "route_index": planner_response.get("selected_route_index", 0),
                        "eef_reference_poses_world": planned_poses,
                    }]
            live_grasp_state = bridge.request({
                "op": "grasp_state",
                "carried_binding": target_binding,
                "expected_instruction": instruction,
            })
            live_contact_sides = target_finger_contact_sides(live_grasp_state)
            live_eef_transform = _pose(
                start_pose[:3],
                Rotation.from_quat(start_pose[3:][[1, 2, 3, 0]]).as_matrix(),
            )
            live_pose_aligned, live_grasp_distance_m, live_grasp_rotation_error_rad = (
                _live_contact_pose_is_close_to_candidate(
                    live_eef_transform, target_pose,
                )
            )
            if (
                (not routes or len(live_contact_sides) == 2)
                and live_contact_sides
                and live_pose_aligned
            ):
                # Close only when measured opposing contacts coincide with the
                # generated candidate pose; incidental contact elsewhere is not
                # treated as a grasp-ready state.
                routes = [{
                    "route_index": "live_contact_pose_aligned_close_probe",
                    "eef_reference_poses_world": [start_pose.tolist(), start_pose.tolist()],
                    "route_source": "live_target_contact_pose_aligned_close_probe",
                    "live_contact_sides": list(live_contact_sides),
                }]
                summary["live_contact_close_probe_admitted_after_RRT_failure"] = {
                    "finger_sides": list(live_contact_sides),
                    "eef_to_migrated_grasp_pose_m": live_grasp_distance_m,
                    "eef_to_migrated_grasp_rotation_error_rad": live_grasp_rotation_error_rad,
                    "pose_alignment_tolerances": {
                        "position_m": LIVE_CONTACT_CLOSE_POSITION_TOLERANCE_M,
                        "rotation_rad": LIVE_CONTACT_CLOSE_ROTATION_TOLERANCE_RAD,
                    },
                    "RRT_status": planner_response.get("status"),
                    "RRT_failure_reason": planner_response.get("failure_reason"),
                    "no_object_or_robot_state_written": True,
                }
            if not routes:
                experience_route_available = bool(
                    (
                        candidate.get("historical_GS_OPD_object_relative_waypoint_prior_used")
                        or candidate.get("experience_object_relative_grasp_prior_used")
                    )
                    and summary.get("ik_candidates")
                    and summary.get("rrt_live_state_unchanged") is True
                )
                generated_geometry_route_available = _cold_start_geometry_fallback_allowed(
                    cold_start, candidate, planner_response,
                )
                if experience_route_available or generated_geometry_route_available:
                    object_pose = _pose(
                        context["target_root_pose_world"]["position_m"],
                        context["target_root_pose_world"]["rotation"],
                    )
                    target_pose, pregrasp_pose = object_relative_grasp_targets(
                        object_pose,
                        candidate["object_to_eef"],
                        candidate["approach_direction_object"],
                        float(candidate["approach_distance_m"]),
                    )
                    pregrasp_pose_7d = np.r_[pregrasp_pose[:3, 3], _wxyz(pregrasp_pose[:3, :3])]
                    route_start = escape_pose if escape_pose is not None else start_pose
                    if generated_geometry_route_available:
                        route_source = "active_BDDL_geometry_high_OSC_approach_after_RRT_failure"
                    else:
                        route_source = (
                            "contact_normal_escape_then_elevated_object_relative_approach"
                            if escape_pose is not None
                            else "object_relative_experience_high_OSC_approach_after_RRT_preflight_failure"
                        )
                    routes = [{
                        "route_index": "generated_geometry_high_approach" if generated_geometry_route_available else "historical_high_approach",
                        "eef_reference_poses_world": high_approach_route(
                            route_start,
                            pregrasp_pose_7d,
                            elevation_m=0.18 if escape_pose is not None else 0.10,
                        ),
                        "route_source": route_source,
                        "contact_escape_prepended": escape_pose is not None,
                    }]
                    if escape_pose is not None:
                        routes[0]["eef_reference_poses_world"] = [
                            start_pose.tolist(), escape_pose.tolist(),
                            *routes[0]["eef_reference_poses_world"][1:],
                        ]
                    summary["historical_direct_OSC_fallback_added"] = experience_route_available
                    summary["generated_geometry_high_approach_fallback_added"] = generated_geometry_route_available
                else:
                    continue
            if escape_pose is not None:
                for route in routes:
                    if route.get("contact_escape_prepended"):
                        continue
                    poses = route.get("eef_reference_poses_world", [])
                    if poses:
                        route["eef_reference_poses_world"] = [
                            start_pose.tolist(), escape_pose.tolist(), *poses[1:]
                        ]
                        route["contact_escape_prepended"] = True
                        route["route_source"] = "MuJoCo_contact_normal_escape_then_planner_route"
            elif (
                not candidate_route_preflight_admitted(
                    summary.get("rrt_status"),
                    summary.get("rrt_live_state_unchanged"),
                    direct_osc_fallback_added=bool(
                        summary.get("historical_direct_OSC_fallback_added")
                        or summary.get("generated_geometry_high_approach_fallback_added")
                    ),
                    live_contact_probe=any(
                        route.get("live_contact_sides") for route in routes
                    ),
                )
            ):
                continue
            for route_rank, route in enumerate(routes):
                poses = route.get("eef_reference_poses_world")
                if not poses:
                    continue
                signature = (
                    f"{route.get('route_source', 'live_target_contact_close_probe')}_"
                    + "_".join(route.get("live_contact_sides", []))
                    if route.get("live_contact_sides")
                    else route_signature(poses)
                )
                route_id = f"rrt{route.get('route_index', 0)}:{signature}"
                if not live_pose_aligned and not route.get("live_contact_sides"):
                    route_id = f"{route_id}:live_object_relative_target"
                if signature in seen_route_signatures:
                    continue
                seen_route_signatures.add(signature)
                memory_candidate_id = candidate.get("geometry_candidate_id", candidate["candidate_id"])
                if (goal_key, str(memory_candidate_id), route_id) in imported_blocked_candidate_routes:
                    report["candidate_routes_skipped_by_failure_memory"].append({
                        "candidate_id": str(memory_candidate_id),
                        "route_id": route_id,
                        "reason": "exact_candidate_route_pair_failed_in_prior_physical_search",
                    })
                    continue
                route_options.append((
                    route_rank, candidate_index, candidate, planner_response, route, route_id
                ))
        # Try one path from each different grasp family before revisiting a
        # candidate with another path; this switches approach side sooner.
        for _, _, candidate, planner_response, route, route_id in sorted(
            route_options, key=lambda row: (row[0], row[1])
        ):
            if not failure_memory.admit(
                goal_key,
                candidate["candidate_id"],
                route_id,
            ):
                continue
            planned_route = dict(planner_response)
            planned_route["eef_reference_poses_world"] = route["eef_reference_poses_world"]
            planned_route["selected_route_index"] = route.get("route_index")
            planned_route["planner_source"] = route.get(
                "route_source", planner_response.get("planner_source", "jointspace_rrt_connect.v1")
            )
            attempted_candidate = dict(candidate)
            attempted_candidate["geometry_candidate_id"] = candidate["candidate_id"]
            attempted_candidate["candidate_id"] = (
                f"{candidate['candidate_id']}__rrt{route.get('route_index', 0)}"
            )
            physical_candidates.append((attempted_candidate, planned_route, route_id))
            if len(physical_candidates) >= args.maximum_physical_branches:
                break
        report.update({
            "status": "physical_search_in_progress",
            "active_target": {
                "object_id_from_native_BDDL": target_id,
                "native_goal_predicate": context["native_goal_predicate"],
                "semantic_binding": target_semantic_binding,
                "initial_root_pose_world": context["target_root_pose_world"],
                "instruction": instruction,
            },
            "generation": generation,
            "preflight": {
                "reset_model_geometry_sha256": reset.get("scene_metadata", {}).get("model_geometry_sha256"),
                "restored_search_start_state_sha256": state_sha,
                "native_task_status": native,
                "initial_safety_diagnostics": initial_safety,
                "semantic_geometry_binding_status": semantic.get("status"),
                "active_gripper_geometry_inventory": gripper,
                "joint_limit_inventory": limits,
                "model_fingerprint_components": fingerprint.get("component_digests"),
                "safety_policy_is_diagnostic_only": True,
            },
        })
        for candidate, rrt, route_id in physical_candidates:
            branch: dict[str, Any] = {
                "candidate_id": candidate["candidate_id"],
                "geometry_candidate_id": candidate.get("geometry_candidate_id"),
                "selected_RRT_route_index": rrt.get("selected_route_index"),
                "route_source": rrt.get("planner_source", "jointspace_rrt_connect.v1"),
                "route_signature": route_id,
                "status": "in_progress",
                "action_record_start_index": len(records),
            }
            report["physical_branches"].append(branch)
            try:
                result, candidate_snapshot = _run_physical_branch(
                    bridge,
                    candidate,
                    rrt,
                    initial_payload=search_start_payload,
                    initial_state=search_start_state,
                    model_sha=model_sha,
                    context=context,
                    target_binding=target_binding,
                    policy=policy,
                    controller_fields=controller["robots"][0]["fields"],
                    monitor_type=GraspStabilityMonitor,
                    records=records,
                    frames=frames,
                    output_dir=output_dir,
                    abort_target_displacement_m=0.015 if cold_start else None,
                )
                branch.update(result)
                final_snapshot_payload = candidate_snapshot
                skill_success_time_s = time.monotonic() - run_started
                report["time_to_first_verified_grasp_skill_success_s"] = skill_success_time_s
                if args.experience_library:
                    try:
                        learned_template = _learned_grasp_template(
                            candidate,
                            branch,
                            context,
                            records,
                            benchmark=args.benchmark,
                            level=args.level,
                            task_id=args.task_id,
                            seed=args.seed,
                            init_state=args.init_state,
                            output_dir=output_dir,
                        )
                        added = append_verified_search_grasp_template(
                            args.experience_library, learned_template
                        )
                        report.setdefault("experience_learning", {}).update({
                            "status": "skill_added" if added else "skill_already_present",
                            "template_id": learned_template["template_id"],
                            "library_path": str(args.experience_library),
                            "task_success": False,
                            "safety_certified": False,
                            "source_actions_replayed": False,
                        })
                    except Exception as exc:
                        report["experience_learning"] = {
                            "status": "not_recorded",
                            "error": f"{type(exc).__name__}: {exc}",
                            "task_success": False,
                            "safety_certified": False,
                        }
                report["success"] = {
                    "candidate_id": candidate["candidate_id"],
                    "target_object_id": target_id,
                    "target_lift_delta_m": branch.get("target_lift_delta_m"),
                    "stable_bilateral_grasp": branch.get("grasp_after_lift"),
                    "skill_success_time_s": skill_success_time_s,
                    "snapshot": branch.get("final_snapshot"),
                    "task_success_candidate": True,
                    "safety_certified": False,
                }
                report["status"] = "task_success_candidate_created"
                break
            except Exception as exc:
                if candidate.get("_local_replans"):
                    branch["local_replans"] = candidate["_local_replans"]
                candidate_records = [
                    item for item in records if item.get("candidate_id") == candidate["candidate_id"]
                ]
                last_record = candidate_records[-1] if candidate_records else {}
                last_target = _object_state(
                    last_record.get("snapshot_state", {}), int(target_binding["body_id"])
                ) if last_record else {}
                branch.update({
                    "status": "physical_attempt_failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "task_success_candidate": False,
                    "safety_certified": False,
                    "attempted_control_actions_archived": len(candidate_records),
                    "phase_action_counts": dict(Counter(item.get("phase", "unknown") for item in candidate_records)),
                    "captured_physics_substeps": int(sum(
                        item.get("contact_monitoring", {}).get("captured_physics_substeps", 0)
                        for item in candidate_records
                    )),
                    "incomplete_substep_monitoring_actions": int(sum(
                        not bool(item.get("substep_monitoring_complete", False))
                        for item in candidate_records
                    )),
                    "last_archived_target_root_position_m": last_target.get("root_pos"),
                    "last_archived_grasp_state": last_record.get("grasp_state"),
                    "last_archived_safety_diagnostics": last_record.get("safety_geometry"),
                })
                initial_target_z = float(
                    report.get("active_target", {})
                    .get("initial_root_pose_world", {})
                    .get("position_m", [0.0, 0.0, 0.0])[2]
                )
                stable_lift_values = []
                stable_bilateral_grasp_observed = False
                for item in candidate_records:
                    grasp_sample = item.get("grasp_monitor_sample", {})
                    if (
                        grasp_sample.get("grasp_confirmed") is True
                        and int(grasp_sample.get("consecutive_stable_samples", 0) or 0) >= 3
                    ):
                        stable_bilateral_grasp_observed = True
                        object_states = item.get("snapshot_state", {}).get("object_states", {})
                        object_state = next((
                            state for state in object_states.values()
                            if int(state.get("body_id", -1)) == int(target_binding["body_id"])
                        ), None) if isinstance(object_states, dict) else None
                        if object_state and object_state.get("root_pos"):
                            stable_lift_values.append(
                                float(object_state["root_pos"][2]) - initial_target_z
                            )
                branch["stable_bilateral_grasp_observed"] = stable_bilateral_grasp_observed
                branch["maximum_bilateral_grasp_lift_m"] = max(
                    0.0, max(stable_lift_values, default=0.0)
                )
                contact_payload = last_record.get("contacts", {})
                contact_rows = (
                    contact_payload.get("contacts", [])
                    if isinstance(contact_payload, dict) else contact_payload
                )
                robot_contact_pairs = []
                for contact in contact_rows if isinstance(contact_rows, list) else []:
                    if not isinstance(contact, dict):
                        continue
                    body_names = [
                        str(contact.get(side, {}).get("name", ""))
                        for side in ("body1", "body2")
                        if isinstance(contact.get(side), dict)
                    ]
                    geometry_names = [
                        str(contact.get(side, {}).get("name", ""))
                        for side in ("geom1", "geom2")
                        if isinstance(contact.get(side), dict)
                    ]
                    if any(
                        name.startswith(("robot0_", "gripper0_"))
                        for name in body_names + geometry_names
                    ) and len(body_names) == 2:
                        robot_contact_pairs.append("|".join(sorted(body_names)))
                failure_event = failure_memory.record(
                    goal=goal_key,
                    candidate_id=candidate.get("geometry_candidate_id", candidate["candidate_id"]),
                    route_id=route_id,
                    reason=branch["failure_reason"],
                    contact_pairs=robot_contact_pairs,
                )
                branch["failure_memory_event"] = failure_event
                try:
                    terminal = bridge.request({"op": "export_runtime_snapshot"})
                    if terminal.get("status") == "complete" and isinstance(terminal.get("payload"), bytes):
                        candidate_slug = "".join(
                            char if char.isalnum() or char in "._-" else "_"
                            for char in str(candidate.get("candidate_id", "candidate"))
                        )[:120]
                        terminal_path = output_dir / f"{candidate_slug}__failed_runtime_snapshot.pkl"
                        terminal_path.write_bytes(terminal["payload"])
                        terminal_state = _snapshot_state_vector(terminal.get("snapshot_state", {}))
                        branch["final_snapshot"] = {
                            "status": "saved_before_failure_rollback",
                            "path": str(terminal_path),
                            "payload_sha256": _sha256_bytes(terminal["payload"]),
                            "integration_state_sha256": _state_sha(terminal_state),
                            "model_geometry_sha256": model_sha,
                            "failure_state_recoverable": True,
                            "task_success_candidate": False,
                            "safety_certified": False,
                        }
                except Exception as snapshot_exc:
                    branch["final_snapshot"] = {
                        "status": "unavailable_bridge_ended_before_snapshot_export",
                        "error": f"{type(snapshot_exc).__name__}: {snapshot_exc}",
                        "failure_state_recoverable": False,
                        "task_success_candidate": False,
                        "safety_certified": False,
                    }
                try:
                    _restore_initial(bridge, search_start_payload, search_start_state, model_sha)
                    branch["exact_initial_snapshot_restored_after_failure"] = True
                except Exception as restore_exc:
                    branch["exact_initial_snapshot_restored_after_failure"] = False
                    branch["failure_branch_recovery_error"] = f"{type(restore_exc).__name__}: {restore_exc}"
                    branch["bridge_unavailable_after_failure"] = True
                    break
        if report.get("success") is None:
            if not physical_candidates:
                report["status"] = "no_record_only_RRT_route_for_physical_attempt"
                report["failure_classification"] = Counter(
                    str(item.get("rrt_failure_reason") or item.get("rrt_status"))
                    for item in report["candidates"]
                ).most_common()
            else:
                report["status"] = "physical_branches_exhausted_without_grasp_and_lift"
                report["failure_classification"] = Counter(
                    str(item.get("failure_reason")) for item in report["physical_branches"]
                ).most_common()
        report["search_failure_memory"] = failure_memory.summary()
        report["archive"] = str(output_dir)
        report["artifacts"] = _write_recording(output_dir, records, frames)
        _write_json(output_dir / "run_report.json", report)
    except Exception as exc:
        report["status"] = "preflight_or_runner_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["archive"] = str(output_dir)
        if records or frames:
            report["artifacts"] = _write_recording(output_dir, records, frames)
        _write_json(output_dir / "run_report.json", report)
        raise
    finally:
        bridge.close()

    if final_snapshot_payload is not None:
        recovery = BridgeSession(address, output_dir / "snapshot_recovery_bridge.log", args)
        try:
            recovery.start()
            recovery_reset = recovery.request({"op": "reset"})
            if recovery_reset.get("scene_metadata", {}).get("model_geometry_sha256") != model_sha:
                raise RuntimeError("fresh_recovery_bridge_model_fingerprint_mismatch")
            restored = recovery.request({"op": "restore_runtime_snapshot", "payload": final_snapshot_payload})
            expected = report["success"]["snapshot"]["snapshot_state"]
            if restored.get("status") != "restored" or restored.get("model_geometry_sha256") != model_sha:
                raise RuntimeError("fresh_bridge_could_not_restore_task_success_candidate_snapshot")
            if not np.array_equal(
                _snapshot_state_vector(restored.get("snapshot_state", {})),
                _snapshot_state_vector(expected),
            ):
                raise RuntimeError("fresh_bridge_snapshot_integration_state_mismatch")
            grasp = recovery.request({
                "op": "grasp_state",
                "carried_binding": report["preflight"].get("target_binding", {
                    "object_id": report["active_target"]["object_id_from_native_BDDL"],
                    "body_id": int(report["active_target"]["semantic_binding"]["root_body_id"]),
                    "body_name": str(report["active_target"]["semantic_binding"]["root_body_name"]),
                    "geom_ids": report["active_target"]["semantic_binding"]["root_geom_ids"],
                    "geom_names": report["active_target"]["semantic_binding"]["root_geom_names"],
                }),
                "expected_instruction": report["active_target"]["instruction"],
            })
            report["snapshot_recovery_verification"] = {
                "status": "pass",
                "model_geometry_sha256": model_sha,
                "integration_state_exact_match": True,
                "restored_grasp_state": grasp,
            }
        except Exception as exc:
            report["snapshot_recovery_verification"] = {
                "status": "fail", "error": f"{type(exc).__name__}: {exc}"
            }
            report["status"] = "task_success_candidate_snapshot_recovery_failed"
        finally:
            recovery.close()
        report["artifacts"]["task_success_candidate_snapshot"] = report["success"]["snapshot"]["path"]
        report["artifacts"]["dual_camera_video"] = report["artifacts"].get("dual_camera_video")
        _write_json(output_dir / "run_report.json", report)

    hashes = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.json":
            hashes[str(path.relative_to(output_dir))] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    _write_json(output_dir / "artifact_hashes.json", hashes)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resume-from-runtime-snapshot", type=Path, default=None,
        help="continue grasp search from a model-verified partial-task Runtime Snapshot",
    )
    parser.add_argument("--benchmark", default="obstacle_avoidance")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--port", type=int, default=38569)
    parser.add_argument("--demonstration-horizon", type=int, default=1000)
    parser.add_argument("--target-predicate", nargs=3, default=None)
    parser.add_argument("--experience-library", type=Path, default=None)
    parser.add_argument("--previous-search-report", type=Path, action="append", default=[])
    parser.add_argument(
        "--cold-start", action="store_true",
        help="forbid experience, keypose, prior-report, resume, and candidate-answer inputs",
    )
    parser.add_argument(
        "--candidate-id", action="append", default=[],
        help="limit a local retry to a previously generated candidate; repeat only in screen-only mode",
    )
    parser.add_argument(
        "--candidate-tilt-degrees", type=float, action="append", default=[],
        help="screen-only jaw-axis tilt variants for selected candidate IDs; repeat for bounded angles",
    )
    parser.add_argument(
        "--candidate-roll-degrees", type=float, action="append", default=[],
        help="screen-only jaw-pair roll around approach axis for selected candidate IDs",
    )
    parser.add_argument(
        "--candidate-offset-object-z-mm", type=float, action="append", default=[],
        help="screen-only contact-height offsets along active object local z for selected candidate IDs",
    )
    parser.add_argument(
        "--candidate-offset-object-y-mm", type=float, action="append", default=[],
        help="screen-only contact-patch offsets along active object local y for selected candidate IDs",
    )
    parser.add_argument(
        "--candidate-offset-object-x-mm", type=float, action="append", default=[],
        help="screen-only contact-patch offsets along active object local x for selected candidate IDs",
    )
    parser.add_argument(
        "--candidate-coupled-perturbations", action="store_true",
        help="explicitly combine one orientation family and one or more object-axis offset families in screen-only mode",
    )
    parser.add_argument(
        "--historical-grasp-pose-priors-json", type=Path, default=None,
        help="screen-only, task-matched grasp poses recomputed from hashed successful episode physical-state samples",
    )
    parser.add_argument(
        "--maximum-candidates", type=int, default=MAX_GEOMETRY_CANDIDATES,
        help="candidate count; physical search <=16, scratch screen <=24 per batch",
    )
    parser.add_argument(
        "--candidate-start-index", type=int, default=0,
        help="physical candidate start, or scratch-screen window offset",
    )
    parser.add_argument("--maximum-rrt-preflights", type=int, default=MAX_RRT_PREFLIGHTS)
    parser.add_argument("--maximum-physical-branches", type=int, default=MAX_PHYSICAL_BRANCHES)
    parser.add_argument("--rrt-time-budget-s", type=float, default=12.0)
    parser.add_argument(
        "--strict-grasp-screen-only", action="store_true",
        help="run scratch-only full-geometry pad-only/10mm screening, then exit before RRT or physical actions",
    )
    parser.add_argument(
        "--strict-screen-reference-report", type=Path, default=None,
        help="identity-checked prior physical grasp report used only for close-joint and post-close pose priors",
    )
    args = parser.parse_args()
    maximum_batch = (
        MAX_OFFLINE_GEOMETRY_SCREEN_BATCH
        if args.strict_grasp_screen_only else MAX_GEOMETRY_CANDIDATES
    )
    if not 1 <= args.maximum_candidates <= maximum_batch:
        parser.error(f"--maximum-candidates must be in 1..{maximum_batch} for this mode")
    if not 1 <= args.maximum_rrt_preflights <= MAX_RRT_PREFLIGHTS:
        parser.error(f"--maximum-rrt-preflights must be in 1..{MAX_RRT_PREFLIGHTS}")
    if args.strict_grasp_screen_only:
        if args.candidate_start_index < 0 or (
            args.candidate_start_index + args.maximum_candidates
            > MAX_OFFLINE_GEOMETRY_GENERATION
        ):
            parser.error(
                "screen-only candidate window must fit within generated indices "
                f"0..{MAX_OFFLINE_GEOMETRY_GENERATION - 1}"
            )
    elif not 0 <= args.candidate_start_index < args.maximum_candidates:
        parser.error("--candidate-start-index must select an index inside the geometry candidate budget")
    if args.candidate_id and args.candidate_start_index != 0:
        parser.error("--candidate-id requires --candidate-start-index 0")
    if len(args.candidate_id) > 1 and not args.strict_grasp_screen_only:
        parser.error("multiple --candidate-id values require --strict-grasp-screen-only")
    if args.candidate_tilt_degrees and args.candidate_roll_degrees:
        parser.error("choose either --candidate-tilt-degrees or --candidate-roll-degrees")
    perturbation_flags = sum(bool(values) for values in (
        args.candidate_tilt_degrees,
        args.candidate_roll_degrees,
        args.candidate_offset_object_x_mm,
        args.candidate_offset_object_z_mm,
        args.candidate_offset_object_y_mm,
    ))
    orientation_family_count = int(bool(args.candidate_tilt_degrees)) + int(bool(args.candidate_roll_degrees))
    offset_families = {
        axis: values
        for axis, values in (
            ("x", args.candidate_offset_object_x_mm),
            ("y", args.candidate_offset_object_y_mm),
            ("z", args.candidate_offset_object_z_mm),
        )
        if values
    }
    if args.candidate_coupled_perturbations:
        if orientation_family_count != 1 or not offset_families:
            parser.error("--candidate-coupled-perturbations requires exactly one orientation family and at least one object-axis offset family")
    elif orientation_family_count + len(offset_families) > 1:
        parser.error("combine orientation and multiple contact-offset families only with --candidate-coupled-perturbations")
    if (
        args.candidate_tilt_degrees or args.candidate_roll_degrees
        or offset_families
    ):
        if not args.strict_grasp_screen_only:
            parser.error("candidate orientation perturbations require --strict-grasp-screen-only")
        if not args.candidate_id:
            parser.error("candidate orientation perturbations require one or more --candidate-id base poses")
        orientation_count = len(args.candidate_tilt_degrees or args.candidate_roll_degrees)
        offset_count = 1
        for values in offset_families.values():
            offset_count *= len(values)
        perturbation_count = orientation_count * offset_count if args.candidate_coupled_perturbations else (orientation_count or offset_count)
        if len(args.candidate_id) * perturbation_count > MAX_OFFLINE_GEOMETRY_SCREEN_BATCH:
            parser.error("selected base candidates and perturbations must fit one 24-candidate screen batch")
    if args.historical_grasp_pose_priors_json is not None:
        if not args.strict_grasp_screen_only:
            parser.error("--historical-grasp-pose-priors-json requires --strict-grasp-screen-only")
        if args.candidate_id or args.candidate_start_index != 0:
            parser.error("historical pose priors cannot be combined with candidate ID/window filters")
        if perturbation_flags:
            parser.error("historical pose priors cannot be combined with orientation/contact perturbations")
    if not 1 <= args.maximum_physical_branches <= MAX_PHYSICAL_BRANCHES:
        parser.error(f"--maximum-physical-branches must be in 1..{MAX_PHYSICAL_BRANCHES}")
    if not 0.5 <= args.rrt_time_budget_s <= 60.0:
        parser.error("--rrt-time-budget-s must be in 0.5..60")
    if args.strict_grasp_screen_only and args.strict_screen_reference_report is None:
        parser.error("--strict-grasp-screen-only requires --strict-screen-reference-report")
    if args.strict_screen_reference_report is not None and not args.strict_grasp_screen_only:
        parser.error("--strict-screen-reference-report requires --strict-grasp-screen-only")
    if args.cold_start and (
        args.experience_library is not None
        or args.previous_search_report
        or args.historical_grasp_pose_priors_json is not None
        or args.strict_grasp_screen_only
        or args.strict_screen_reference_report is not None
        or args.resume_from_runtime_snapshot is not None
        or args.candidate_id
        or args.candidate_start_index != 0
        or args.candidate_tilt_degrees
        or args.candidate_roll_degrees
        or args.candidate_offset_object_x_mm
        or args.candidate_offset_object_y_mm
        or args.candidate_offset_object_z_mm
    ):
        parser.error("--cold-start cannot use historical experience or partial search inputs")
    if not 1000 <= args.demonstration_horizon <= 1800:
        parser.error("--demonstration-horizon must be in 1000..1800")
    if args.target_predicate is not None and len(args.target_predicate) != 3:
        parser.error("--target-predicate needs exactly three active BDDL predicate fields")
    return args


def main() -> int:
    report = run(_parse_args())
    candidate_count = report.get("artifacts", {}).get("candidate_count")
    if candidate_count is None:
        candidate_count = len(report.get("candidates", []))
    print(json.dumps({
        "status": report.get("status"),
        "archive": report.get("archive"),
        "target": report.get("active_target"),
        "candidate_count": candidate_count,
        "physical_branch_count": len(report.get("physical_branches", [])),
        "success": report.get("success"),
        "failure_classification": report.get("failure_classification"),
        "snapshot_recovery_verification": report.get("snapshot_recovery_verification"),
    }, ensure_ascii=False, indent=2), flush=True)
    if report.get("strict_grasp_screen_only"):
        return 0 if report.get("strict_grasp_geometry_screen") else 3
    if report.get("success") is None:
        return 2
    return 0 if report.get("snapshot_recovery_verification", {}).get("status") == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
