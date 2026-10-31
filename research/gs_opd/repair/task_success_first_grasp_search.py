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
from scipy.spatial.transform import Rotation, Slerp
from research.gs_opd.repair.task_success_first_search import (
    MAX_DEMONSTRATION_HORIZON,
    SearchFailureMemory,
    blocked_candidate_ids_from_report,
    candidate_route_preflight_admitted,
    filter_mechanically_infeasible_grasp_candidates,
    generated_geometry_high_approach_admitted,
    candidate_matches_failure_memory,
    diversify_candidates,
    high_approach_route,
    object_relative_grasp_targets,
    planner_seed,
    route_control_action_budget,
    route_signature,
    robot_contact_escape_direction,
    task_approach_feasible_candidate_ids,
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
MAX_LIVE_UNILATERAL_CLOSE_PROBE_ACTIONS = 8
# A unilateral pad contact can occur well before the generated grasp pose is
# reached.  Only use a pose-held close probe when that contact is already near
# the nominal frame; otherwise finish the live object-relative surface approach.
LIVE_UNILATERAL_CLOSE_PROBE_POSITION_M = 0.010
LIVE_UNILATERAL_APPROACH_CONTINUE_POSITION_M = 0.075
MAX_LIVE_CONTACT_SETTLE_ACTIONS = 2
MAX_LIVE_CLOSE_BILATERAL_CONTACT_LOSS_ACTIONS = 2
LIVE_CONTACT_UNOPPOSED_NORMAL_DOT_THRESHOLD = 0.5
NEAR_SURFACE_START_POSITION_M = 0.025
NEAR_SURFACE_START_ROTATION_RAD = 0.08
NEAR_PREGRASP_START_POSITION_M = 0.075
NEAR_PREGRASP_START_ROTATION_RAD = 0.08
LIVE_CONTACT_SETTLE_LINEAR_SPEED_M_S = 0.015
LIVE_CONTACT_SETTLE_ANGULAR_SPEED_RAD_S = 0.08
LIVE_CONTACT_SETTLE_REQUIRED_SAMPLES = 3
LIVE_CONTACT_CLOSE_POSITION_TOLERANCE_M = 0.004
LIVE_CONTACT_CLOSE_ROTATION_TOLERANCE_RAD = 0.045
MAX_LIFT_CONTROL_ACTIONS = 40
LIVE_CONTACT_LIFT_CONTROL_ACTIONS = 120
LIVE_CONTACT_LIFT_TRANSLATION_STEP_M = 0.002
# Stage A needs a short physical lift to confirm that a grasp moves the target;
# it does not need the former arbitrary 30 mm excursion.  Some valid thin
# features (mug handles, pot rims) lose contact near that distance.  Stop the
# lift as soon as this smaller displacement is reached while the live grasp
# monitor still confirms bilateral contact, then let native-task transport
# determine whether the grasp is useful.
MINIMUM_OBJECT_LIFT_M = 0.022
LIFT_COMMAND_M = 0.040

# Stage A can search for native task success without computing exhaustive
# all-Geom clearances after every MuJoCo microstep.  Physical contact samples
# remain complete when this option is enabled; only the separate Stage B style
# clearance sweep is skipped.
_STAGE_A_CONTACT_ONLY_SUBSTEP_AUDIT = False


def configure_stage_a_contact_only_substep_audit(enabled: bool) -> None:
    global _STAGE_A_CONTACT_ONLY_SUBSTEP_AUDIT
    if not isinstance(enabled, bool):
        raise TypeError("stage_a_contact_only_substep_audit_requires_bool")
    _STAGE_A_CONTACT_ONLY_SUBSTEP_AUDIT = enabled


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
    context: dict[str, Any], *, maximum_candidates: int = MAX_GEOMETRY_CANDIDATES,
    required_candidate_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enumerate opposing-pad grasps on the active target's compiled geometry."""
    if isinstance(maximum_candidates, bool) or not 1 <= maximum_candidates <= MAX_OFFLINE_GEOMETRY_GENERATION:
        raise ValueError(
            f"maximum_candidates_must_be_in_1..{MAX_OFFLINE_GEOMETRY_GENERATION}"
        )
    required_candidate_ids = set(required_candidate_ids or ())
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
            if candidate["candidate_id"] not in required_candidate_ids:
                continue
            # Preserve an explicitly pinned active-scene geom ID even when a
            # different compiled geom yields the same physical grasp transform.
            candidate["duplicate_pose_alias_of"] = original
        else:
            seen_candidates[signature] = candidate["candidate_id"]
        if (
            len(unique_candidates) < maximum_candidates
            or candidate["candidate_id"] in required_candidate_ids
        ):
            unique_candidates.append(candidate)
        selected_ids = {str(row["candidate_id"]) for row in unique_candidates}
        if len(unique_candidates) >= maximum_candidates and required_candidate_ids <= selected_ids:
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
        "required_active_geometry_candidate_ids": sorted(required_candidate_ids),
        "jaw_pad_center_source": "active_runtime_MuJoCo_geom_xpos_relative_to_eef_site",
        "pad_separation_open_m": context["gripper_pad_separation_m"],
        "maximum_pad_separation_m": max_opening,
        "all_candidates_unique": True,
    }


def _rank_by_endpoint_ik_joint_travel(
    candidates: list[dict[str, Any]],
    approach_rows: dict[str, dict[str, Any]],
    start_arm_qpos: np.ndarray,
    screen_pass_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prefer reachable grasp poses whose IK posture is close to the live arm.

    A Cartesian-nearest pregrasp can require a large wrist/elbow winding from
    the current posture. OSC then may stall at an intermediate pose even when
    both endpoint IK checks pass. Rank within the existing screen evidence
    tier by estimated joint travel; this only changes search order and does not
    certify or gate a physical attempt.
    """
    start = np.asarray(start_arm_qpos, dtype=np.float64)
    screen_pass_ids = set(screen_pass_ids or ())
    if start.shape != (7,) or not np.isfinite(start).all():
        return list(candidates), []
    scored: list[tuple[tuple[float, int, float, float, str], dict[str, Any], dict[str, Any]]] = []
    diagnostics: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        reachability = approach_rows.get(candidate_id, {}).get("joint_reachability", {})
        pregrasp = reachability.get("pregrasp", {})
        grasp = reachability.get("grasp_open_gripper", {})
        pre_q = np.asarray(pregrasp.get("qpos", []), dtype=np.float64)
        grasp_q = np.asarray(grasp.get("qpos", []), dtype=np.float64)
        endpoint_ik = bool(
            pregrasp.get("reachable") is True
            and grasp.get("reachable") is True
            and pre_q.shape == grasp_q.shape == (7,)
            and np.isfinite(pre_q).all()
            and np.isfinite(grasp_q).all()
        )
        if endpoint_ik:
            pregrasp_delta = pre_q - start
            surface_delta = grasp_q - pre_q
            pregrasp_travel = float(np.linalg.norm(pregrasp_delta))
            surface_travel = float(np.linalg.norm(surface_delta))
            maximum_joint_delta = float(np.max(np.abs(pregrasp_delta)))
            score = pregrasp_travel + 0.5 * surface_travel
        else:
            pregrasp_travel = None
            surface_travel = None
            maximum_joint_delta = None
            score = float("inf")
        row = {
            "candidate_id": candidate_id,
            "endpoint_ik_verified": endpoint_ik,
            "task_approach_screen_pass": candidate_id in screen_pass_ids,
            "pregrasp_joint_travel_l2_rad": pregrasp_travel,
            "grasp_surface_joint_travel_l2_rad": surface_travel,
            "maximum_pregrasp_joint_delta_rad": maximum_joint_delta,
            "joint_travel_score_rad": score if np.isfinite(score) else None,
            "pregrasp_eef_distance_m": float(candidate.get("pregrasp_eef_distance_m") or 0.0),
        }
        diagnostics.append(row)
        # Candidates without endpoint IK retain their prior relative order.
        key = (
            score,
            0 if candidate_id in screen_pass_ids else 1,
            maximum_joint_delta if maximum_joint_delta is not None else float("inf"),
            float(candidate.get("pregrasp_eef_distance_m") or 0.0),
            candidate_id,
        )
        scored.append((key, candidate, row))
    scored.sort(key=lambda item: item[0])
    return [item[1] for item in scored], [item[2] for item in scored]


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
    return int(getattr(args, "geometry_candidate_pool_offset", 0) + args.maximum_candidates)


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


def expand_task_approach_depth_candidates(
    candidates: list[dict[str, Any]],
    *,
    target_pose_world: np.ndarray,
    eef_position_world: np.ndarray,
    target_collision_geometries: list[dict[str, Any]],
    screen_batch_limit: int = MAX_OFFLINE_GEOMETRY_SCREEN_BATCH,
    depth_offset_fractions: tuple[float, ...] = (0.30, 0.60),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Shift candidate pad centers along the object-geom approach axis."""
    if isinstance(screen_batch_limit, bool) or not 1 <= screen_batch_limit <= MAX_OFFLINE_GEOMETRY_SCREEN_BATCH:
        raise ValueError("task_approach_screen_batch_limit_must_be_in_1_to_24")
    if len(candidates) > screen_batch_limit:
        raise ValueError("task_approach_base_candidates_exceed_screen_batch_limit")
    target_pose = np.asarray(target_pose_world, dtype=np.float64)
    eef_position = np.asarray(eef_position_world, dtype=np.float64)
    if target_pose.shape != (4, 4) or not np.isfinite(target_pose).all():
        raise ValueError("task_approach_target_pose_must_be_finite_4x4")
    if eef_position.shape != (3,) or not np.isfinite(eef_position).all():
        raise ValueError("task_approach_eef_position_must_be_finite_xyz")
    fractions = np.asarray(depth_offset_fractions, dtype=np.float64)
    if (
        fractions.ndim != 1
        or len(fractions) == 0
        or not np.isfinite(fractions).all()
        or np.any(fractions <= 0.0)
        or np.any(fractions > 1.5)
        or len(set(float(item) for item in fractions)) != len(fractions)
    ):
        raise ValueError("task_approach_depth_offset_fractions_must_be_unique_in_(0,1.5]")
    base_ids = [str(candidate.get("candidate_id", "")) for candidate in candidates]
    if any(not candidate_id for candidate_id in base_ids) or len(set(base_ids)) != len(base_ids):
        raise ValueError("task_approach_base_candidate_ids_must_be_nonempty_and_unique")
    capacity = screen_batch_limit - len(candidates)
    geometry_by_id = {
        int(geometry["geom_id"]): geometry for geometry in target_collision_geometries
    }
    variants = []
    offset_records = []
    prioritized_candidates = sorted(
        enumerate(candidates),
        key=lambda item: (
            int(item[1].get("approach_sign", 0) != 1),
            item[0],
        ),
    )
    for _candidate_index, candidate in prioritized_candidates:
        if len(variants) >= capacity:
            break
        geometry_id = int(candidate.get("target_geom_id", -1))
        geometry = geometry_by_id.get(geometry_id)
        if geometry is None:
            raise ValueError(f"task_approach_candidate_target_geom_missing:{geometry_id}")
        half_extents = _box_half_extents(geometry)
        jaw_axis = int(candidate.get("jaw_axis_index_in_geom", -1))
        approach_axis = int(candidate.get("approach_axis_index_in_geom", -1))
        if jaw_axis not in (0, 1, 2) or approach_axis not in (0, 1, 2) or jaw_axis == approach_axis:
            raise ValueError("task_approach_candidate_geom_axis_indices_are_invalid")
        geometry_rotation = np.asarray(geometry["rotation_root"], dtype=np.float64)
        if geometry_rotation.shape != (3, 3) or not np.isfinite(geometry_rotation).all():
            raise ValueError("task_approach_target_geom_rotation_is_invalid")
        half_extent = float(half_extents[approach_axis])
        offset_magnitudes = [
            min(float(fraction) * half_extent, 0.050)
            for fraction in fractions
        ]
        offset_magnitudes = list(dict.fromkeys(value for value in offset_magnitudes if value > 1e-6))
        if not offset_magnitudes:
            continue
        transform = np.asarray(candidate.get("object_to_eef"), dtype=np.float64)
        approach = np.asarray(candidate.get("approach_direction_object"), dtype=np.float64)
        if transform.shape != (4, 4) or approach.shape != (3,) or not np.isfinite(transform).all() or not np.isfinite(approach).all():
            raise ValueError("task_approach_base_candidate_pose_is_invalid")
        axis_object = geometry_rotation[:, approach_axis]
        signed_offsets = [
            (sign * magnitude)
            for magnitude in offset_magnitudes
            for sign in (-1, 1)
        ]
        for signed_offset_m in signed_offsets:
            if len(variants) >= capacity:
                break
            variant = dict(candidate)
            shifted_transform = transform.copy()
            shifted_transform[:3, 3] += axis_object * signed_offset_m
            world_eef = target_pose @ shifted_transform
            approach_world = target_pose[:3, :3] @ (approach / np.linalg.norm(approach))
            pregrasp_world = world_eef[:3, 3] - approach_world * float(
                candidate.get("approach_distance_m", 0.0606)
            )
            offset_mm = signed_offset_m * 1000.0
            variant.update({
                "candidate_id": (
                    f"{candidate['candidate_id']}__approach_axis{approach_axis}_{offset_mm:+.1f}mm"
                ),
                "source_candidate_id": str(candidate["candidate_id"]),
                "object_to_eef": shifted_transform.tolist(),
                "pregrasp_eef_distance_m": float(np.linalg.norm(pregrasp_world - eef_position)),
                "candidate_source": str(candidate.get("candidate_source", ""))
                    + "+geometry_approach_depth_offset",
                "approach_depth_offset": {
                    "basis": "target_collision_geom_local_axis_in_object_root_frame",
                    "geom_id": geometry_id,
                    "axis_index": approach_axis,
                    "offset_mm": offset_mm,
                    "grasp_orientation_preserved": True,
                },
            })
            variants.append(variant)
            offset_records.append({
                "candidate_id": variant["candidate_id"],
                "source_candidate_id": str(candidate["candidate_id"]),
                "geom_id": geometry_id,
                "axis_index": approach_axis,
                "offset_mm": offset_mm,
            })
    expanded = [*candidates, *variants]
    ids = [str(candidate["candidate_id"]) for candidate in expanded]
    if len(ids) != len(set(ids)):
        raise ValueError("task_approach_expansion_produced_duplicate_candidate_ids")
    return expanded, {
        "applied": bool(variants),
        "source": "active_BDDL_geometry_candidates_only",
        "base_candidate_count": len(candidates),
        "candidate_count_after_expansion": len(expanded),
        "approach_depth_offset_variant_count": len(variants),
        "offset_rule": "plus_or_minus_30pct_and_60pct_of_approach_axis_half_extent_capped_at_20mm_and_40mm",
        "offsets_by_candidate": offset_records,
        "grasp_orientation_preserved": True,
        "screen_batch_limit": screen_batch_limit,
        "historical_actions_or_success_pose_priors_used": False,
    }


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
    axis_index = {"x": 0, "y": 1, "z": 2}[offset_object_axis]
    object_up_in_world = target_pose[:3, :3][:, axis_index]
    gravity_alignment = abs(float(np.dot(object_up_in_world, np.asarray([0.0, 0.0, 1.0]))))
    maximum_offset_mm = 60.0 if gravity_alignment >= 0.80 else 20.0
    if (
        not offsets
        or not np.isfinite(offsets).all()
        or any(abs(value) <= 1e-8 or abs(value) > maximum_offset_mm for value in offsets)
        or len(set(offsets)) != len(offsets)
    ):
        raise ValueError(
            "grasp_surface_offsets_must_be_unique_nonzero_values_within_60mm_on_a_gravity_aligned_axis_or_20mm_elsewhere"
        )
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
    """Generate a bounded grid of wrist rotations and/or object-frame shifts."""
    if not candidates:
        raise ValueError("coupled_grasp_perturbations_require_base_candidates")
    if not isinstance(offsets_by_object_axis, dict) or not offsets_by_object_axis:
        raise ValueError("coupled_grasp_perturbations_require_object_frame_offsets")
    if any(axis not in {"x", "y", "z"} for axis in offsets_by_object_axis):
        raise ValueError("coupled_grasp_offset_axes_must_be_x_y_or_z")
    oriented = (
        generate_grasp_tilt_variants(
            candidates,
            target_pose_world=target_pose_world,
            eef_position_world=eef_position_world,
            tilt_degrees=tilt_degrees,
            rotation_axis=rotation_axis,
        )
        if tilt_degrees else [dict(candidate) for candidate in candidates]
    )
    for candidate in oriented:
        candidate["_coupled_base_candidate_id"] = str(candidate.get("source_candidate_id", ""))
        candidate["_coupled_orientation_candidate_id"] = str(candidate.get("candidate_id", ""))
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
        orientation_candidate_id = str(variant.pop("_coupled_orientation_candidate_id"))
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
            "composition_order": (
                "rotate_about_gripper_axis_then_translate_in_object_frame_axes"
                if tilt_degrees else "translate_in_object_frame_axes"
            ),
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


def _target_pad_contact_normal_diagnostics(
    contact_reply: dict[str, Any], target_body_id: int,
) -> dict[str, Any]:
    """Measure whether left/right live pad contacts face opposing object surfaces."""
    contacts = contact_reply.get("contacts", [])
    if not isinstance(contacts, list):
        return {"status": "unavailable", "reason": "contact_list_is_unavailable"}
    normals_by_side: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    for contact in contacts:
        if not isinstance(contact, dict) or contact.get("normal_status") != "available":
            continue
        body1 = contact.get("body1", {})
        body2 = contact.get("body2", {})
        try:
            body1_id, body2_id = int(body1.get("id", -1)), int(body2.get("id", -1))
            normal = np.asarray(contact.get("normal_geom1_to_geom2"), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if normal.shape != (3,) or not np.isfinite(normal).all():
            continue
        geom1 = str(contact.get("geom1", {}).get("name", ""))
        geom2 = str(contact.get("geom2", {}).get("name", ""))
        if body1_id == target_body_id and "finger1" in geom2:
            side, object_to_finger = "left", normal
        elif body2_id == target_body_id and "finger1" in geom1:
            side, object_to_finger = "left", -normal
        elif body1_id == target_body_id and "finger2" in geom2:
            side, object_to_finger = "right", normal
        elif body2_id == target_body_id and "finger2" in geom1:
            side, object_to_finger = "right", -normal
        else:
            continue
        magnitude = float(np.linalg.norm(object_to_finger))
        if magnitude > 1e-12:
            normals_by_side[side].append(object_to_finger / magnitude)

    means: dict[str, np.ndarray] = {}
    for side, values in normals_by_side.items():
        if values:
            mean = np.mean(np.stack(values), axis=0)
            magnitude = float(np.linalg.norm(mean))
            if magnitude > 1e-12:
                means[side] = mean / magnitude
    if len(means) != 2:
        return {
            "status": "unavailable",
            "contact_counts_by_side": {side: len(values) for side, values in normals_by_side.items()},
            "reason": "both_pad_contact_normals_not_available",
        }
    dot = float(np.clip(np.dot(means["left"], means["right"]), -1.0, 1.0))
    return {
        "status": "complete",
        "contact_counts_by_side": {side: len(values) for side, values in normals_by_side.items()},
        "left_object_to_finger_normal": means["left"].tolist(),
        "right_object_to_finger_normal": means["right"].tolist(),
        "left_right_normal_dot": dot,
        "surfaces_are_opposed": dot <= -LIVE_CONTACT_UNOPPOSED_NORMAL_DOT_THRESHOLD,
        "source": "live_MuJoCo_contact_frames_normalized_object_to_finger",
    }


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
    # BDDL can repeat an identical CheckContact constraint.  The predicates
    # are idempotent, but the Safety Bridge correctly rejects duplicate body
    # pairs in a compiled policy; canonicalize them here without changing the
    # active task semantics.
    unique_forbidden_pairs = []
    seen_forbidden_pairs = set()
    for object_a, object_b in forbidden_pairs:
        canonical_pair = tuple(sorted((str(object_a), str(object_b))))
        if canonical_pair in seen_forbidden_pairs:
            continue
        seen_forbidden_pairs.add(canonical_pair)
        unique_forbidden_pairs.append((str(object_a), str(object_b)))
    duplicate_constraint_count = len(forbidden_pairs) - len(unique_forbidden_pairs)
    forbidden_pairs = unique_forbidden_pairs
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
        "duplicate_forbidden_contact_constraints_collapsed": duplicate_constraint_count,
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
    if action_index >= MAX_BRANCH_CONTROL_ACTIONS:
        raise RuntimeError("physical_branch_control_action_budget_exhausted_before_step")
    monitored_step_started = time.perf_counter()
    reply = bridge.request({
        "op": "step_monitored",
        "action": np.asarray(action, dtype=np.float32).tolist(),
        "action_index": int(action_index),
        "include_grasp_state": True,
        "grasp_binding": target_binding,
        "expected_instruction": instruction,
        "safety_geometry_policy": (
            None if _STAGE_A_CONTACT_ONLY_SUBSTEP_AUDIT else policy
        ),
    })
    monitored_step_wall_s = time.perf_counter() - monitored_step_started
    grasp_state_inline = isinstance(reply.get("grasp_state"), dict)
    grasp_state_wall_s = 0.0
    grasp = reply.get("grasp_state") if grasp_state_inline else None
    if grasp is None:
        grasp_started = time.perf_counter()
        try:
            grasp = bridge.request({
                "op": "grasp_state",
                "carried_binding": target_binding,
                "expected_instruction": instruction,
            })
        except Exception as exc:
            grasp = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
        grasp_state_wall_s = time.perf_counter() - grasp_started
    sample = monitor.update(grasp, rollout_step=action_index)
    contact_monitoring = reply.get("contact_monitoring", {})
    contact_monitoring_complete = _contact_substep_monitoring_complete(
        contact_monitoring, require_geometry=False
    )
    clearance_monitoring_complete = _contact_substep_monitoring_complete(
        contact_monitoring, require_geometry=True
    )
    monitoring_complete = contact_monitoring_complete
    snapshot_state = reply.get("snapshot_state", {})
    record = {
        "candidate_id": candidate_id,
        "step_index": len(records),
        "phase": phase,
        "action_7d": np.asarray(action, dtype=np.float32).tolist(),
        "state": reply.get("state", {}),
        "snapshot_state": snapshot_state,
        "target_body_id": int(target_binding["body_id"]),
        "grasp_state_source": (
            "inline_step_monitored" if grasp_state_inline else "separate_rpc_fallback"
        ),
        "contacts": reply.get("contacts", {}),
        "start_contacts": reply.get("start_contacts", {}),
        "contact_monitoring": contact_monitoring,
        "substep_monitoring_complete": monitoring_complete,
        "substep_contact_monitoring_complete": contact_monitoring_complete,
        "substep_clearance_monitoring_complete": clearance_monitoring_complete,
        "physics_audit_profile": {
            "substep_contacts": "complete_per_MuJoCo_step",
            "substep_clearance_sweeps": (
                "disabled_for_stage_a_search"
                if _STAGE_A_CONTACT_ONLY_SUBSTEP_AUDIT else "enabled"
            ),
            "safety_certified": False,
        },
        "safety_geometry": reply.get("safety_geometry"),
        "grasp_state": grasp,
        "grasp_monitor_sample": sample,
        "reward": reply.get("reward", 0.0),
        "done": reply.get("done", False),
        "task_success": reply.get("task_success"),
        "timestep": reply.get("timestep"),
        "execution_timing_s": {
            "step_monitored_rpc": monitored_step_wall_s,
            "grasp_state_rpc": grasp_state_wall_s,
        },
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


def _contact_substep_monitoring_complete(
    contact_monitoring: dict[str, Any], *, require_geometry: bool = True
) -> bool:
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
    if not require_geometry:
        return True
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
    max_translation_step_m: float = 0.008,
    planning_tracking: dict[str, Any] | None = None,
    target_pose_provider: Callable[
        [dict[str, Any]], tuple[np.ndarray, np.ndarray]
    ] | None = None,
    object_progress_target_world: np.ndarray | None = None,
    object_progress_rotation_only: bool = False,
    object_progress_horizontal_only: bool = False,
    gripper_action: float = 0.0,
    absolute_action_limit: int | None = None,
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
    if (
        isinstance(max_translation_step_m, bool)
        or not isinstance(max_translation_step_m, (int, float))
        or not np.isfinite(float(max_translation_step_m))
        or not 0.001 <= float(max_translation_step_m) <= 0.02
    ):
        raise ValueError("pose_tracking_max_translation_step_must_be_in_0.001_to_0.02")
    if (
        isinstance(gripper_action, bool)
        or not isinstance(gripper_action, (int, float))
        or not np.isfinite(float(gripper_action))
        or not -1.0 <= float(gripper_action) <= 1.0
    ):
        raise ValueError("pose_tracking_gripper_action_must_be_finite_and_in_minus1_to_1")
    if not isinstance(object_progress_rotation_only, bool):
        raise ValueError("pose_tracking_object_rotation_only_must_be_boolean")
    if not isinstance(object_progress_horizontal_only, bool):
        raise ValueError("pose_tracking_object_horizontal_only_must_be_boolean")
    if absolute_action_limit is None:
        absolute_action_limit = MAX_BRANCH_CONTROL_ACTIONS
    if object_progress_rotation_only and object_progress_target_world is None:
        raise ValueError("pose_tracking_rotation_only_requires_an_object_progress_target")
    progress_history: list[tuple[float, float]] = []
    object_progress_history: list[tuple[float, float]] = []
    object_progress_target = None
    if object_progress_target_world is not None:
        object_progress_target = np.asarray(object_progress_target_world, dtype=np.float64)
        if object_progress_target.shape != (4, 4) or not np.isfinite(object_progress_target).all():
            raise ValueError("pose_tracking_object_progress_target_must_be_a_finite_4x4_pose")

    def previous_bilateral_grasp_confirmed() -> bool:
        if not records:
            return True
        previous_grasp = records[-1].get("grasp_state", {})
        previous_fingers = previous_grasp.get("finger_contacts", {})
        return bool(
            previous_grasp.get("grasp_confirmed") is True
            and previous_fingers.get("left") is True
            and previous_fingers.get("right") is True
        )

    for _ in range(maximum_steps):
        if action_counter[0] >= absolute_action_limit:
            raise RuntimeError("physical_branch_control_action_budget_exhausted_before_step")
        if stop_condition is not None and stop_condition():
            return False
        observed = bridge.request({"op": "observe", "include_snapshot_state": True})
        bilateral_for_tracking = previous_bilateral_grasp_confirmed()
        if target_pose_provider is not None and bilateral_for_tracking:
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
        object_progress_window = None
        object_position_error = None
        object_rotation_error = None
        if object_progress_target is not None and bilateral_for_tracking:
            live_object = _object_state(
                observed.get("snapshot_state", {}), int(target_binding["body_id"]),
            )
            live_object_world = _runtime_object_pose(live_object)
            position_delta = (
                object_progress_target[:2, 3] - live_object_world[:2, 3]
                if object_progress_horizontal_only
                else object_progress_target[:3, 3] - live_object_world[:3, 3]
            )
            object_position_error = float(np.linalg.norm(position_delta))
            object_rotation_error = float(np.linalg.norm(Rotation.from_matrix(
                object_progress_target[:3, :3] @ live_object_world[:3, :3].T
            ).as_rotvec()))
            if (
                object_rotation_error <= rotation_tolerance_rad
                if object_progress_rotation_only
                else _pose_tracking_error_within_tolerance(
                    object_position_error, object_rotation_error,
                    position_tolerance_m, rotation_tolerance_rad,
                )
            ):
                return True
        elif object_progress_target is None and _pose_tracking_error_within_tolerance(
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
            if object_progress_target is not None:
                if bilateral_for_tracking:
                    object_progress_history.append((
                        0.0 if object_progress_rotation_only else object_position_error,
                        object_rotation_error,
                    ))
                    object_progress_history = object_progress_history[-(stagnation_window_actions + 1):]
                    object_progress_window = object_progress_history
                else:
                    object_progress_history.clear()
            if _pose_tracking_window_stagnated(
                progress_history, window_actions=stagnation_window_actions,
                object_progress_history=object_progress_window,
                object_minimum_position_progress_m=0.002,
                object_minimum_rotation_progress_rad=0.02,
                rotation_only=object_progress_rotation_only,
            ):
                raise RuntimeError(
                    f"{phase}_waypoint_stagnated_for_{stagnation_window_actions}_actions"
                )
        delta_position = position_error * min(
            1.0, float(max_translation_step_m) / max(distance, 1e-12),
        )
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
            gripper_action=float(gripper_action),
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
                "gripper_action_command": float(gripper_action),
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
    object_progress_history: list[tuple[float, float]] | None = None,
    object_minimum_position_progress_m: float | None = None,
    object_minimum_rotation_progress_rad: float | None = None,
    rotation_only: bool = False,
) -> bool:
    """Detect blocked tracking from EEF error and held-object progress when available."""
    object_position_threshold = (
        minimum_position_progress_m
        if object_minimum_position_progress_m is None
        else float(object_minimum_position_progress_m)
    )
    object_rotation_threshold = (
        minimum_rotation_progress_rad
        if object_minimum_rotation_progress_rad is None
        else float(object_minimum_rotation_progress_rad)
    )
    if not np.isfinite([object_position_threshold, object_rotation_threshold]).all() or min(
        object_position_threshold, object_rotation_threshold,
    ) <= 0.0:
        raise ValueError("object_pose_tracking_progress_thresholds_must_be_positive_and_finite")
    if not isinstance(rotation_only, bool):
        raise ValueError("object_pose_tracking_rotation_only_must_be_boolean")

    def stagnant(
        rows: list[tuple[float, float]],
        *,
        position_threshold_m: float,
        rotation_threshold_rad: float,
    ) -> bool:
        if len(rows) < window_actions + 1:
            return False
        window = rows[-(window_actions + 1):]
        position_progress = max(0.0, window[0][0] - min(row[0] for row in window[1:]))
        rotation_progress = max(0.0, window[0][1] - min(row[1] for row in window[1:]))
        position_stalled = position_progress < position_threshold_m
        rotation_stalled = rotation_progress < rotation_threshold_rad
        return bool(
            (position_stalled and rotation_stalled)
            # A component that remains far outside tolerance must make its own
            # progress. Otherwise, improvement in the other component can hide
            # a blocked or diverging OSC axis for an entire long waypoint.
            or (position_stalled and min(row[0] for row in window[1:]) > 0.020)
            or (rotation_stalled and min(row[1] for row in window[1:]) > 0.12)
        )

    if object_progress_history is None:
        return stagnant(
            history,
            position_threshold_m=minimum_position_progress_m,
            rotation_threshold_rad=minimum_rotation_progress_rad,
        )
    if rotation_only:
        if len(object_progress_history) < window_actions + 1:
            return False
        window = object_progress_history[-(window_actions + 1):]
        rotation_progress = max(0.0, window[0][1] - min(row[1] for row in window[1:]))
        return bool(
            rotation_progress < object_rotation_threshold
            and min(row[1] for row in window[1:]) > 0.12
        )
    # During a confirmed grasp, object motion is the meaningful progress
    # signal; EEF error can remain large while the object is still moving.
    return stagnant(
        object_progress_history,
        position_threshold_m=object_position_threshold,
        rotation_threshold_rad=object_rotation_threshold,
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
    local_replan_rrt_time_budget_s: float = 2.0,
    abort_target_displacement_m: float | None = None,
    force_route_before_contact_close: bool = False,
    stage_a_unilateral_close_probe_position_m: float = LIVE_UNILATERAL_CLOSE_PROBE_POSITION_M,
    stage_a_pregrasp_staged_clearance_height_m: float = 0.0,
    stage_a_pregrasp_lateral_detour_m: float = 0.0,
    stage_a_pregrasp_lift_before_lateral_detour: bool = False,
) -> tuple[dict[str, Any], bytes | None]:
    from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action

    _advance_pose.controller_fields = controller_fields
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
    branch_record_start = len(records)

    def unintended_robot_contacts(record: dict[str, Any]) -> list[dict[str, Any]]:
        """Return physical robot contacts with scene objects other than the grasp target/table."""
        contacts = record.get("contacts", {}).get("contacts", [])
        if not isinstance(contacts, list):
            return []
        result = []
        target_body_id = int(target_binding["body_id"])
        for contact in contacts:
            geom1 = contact.get("geom1", {})
            geom2 = contact.get("geom2", {})
            body1 = contact.get("body1", {})
            body2 = contact.get("body2", {})
            name1 = str(geom1.get("name", ""))
            name2 = str(geom2.get("name", ""))
            robot1 = name1.startswith(("robot0_", "gripper0_"))
            robot2 = name2.startswith(("robot0_", "gripper0_"))
            if robot1 == robot2:
                continue
            other_body = body2 if robot1 else body1
            if int(other_body.get("id", -1)) == target_body_id:
                continue
            if str(other_body.get("name", "")).lower() in {"table", "table_main"}:
                continue
            result.append({
                "robot_geom": name1 if robot1 else name2,
                "obstacle_geom": name2 if robot1 else name1,
                "obstacle_body_id": int(other_body.get("id", -1)),
                "obstacle_body_name": str(other_body.get("name", "")),
                "contact_position_world_m": contact.get("position"),
                "signed_contact_distance_m": contact.get("signed_contact_distance_m"),
            })
        return result

    def record_robot_obstacle_contact() -> bool:
        """Record incidental Stage A contact without treating it as a stop condition.

        Task-success search permits safety violations.  The MuJoCo contact is
        retained in the per-action audit and the first event is archived for
        diagnosis, while OSC continues until the requested pose, target contact,
        or its normal physical stagnation/action limit decides feasibility.
        """
        if len(records) <= branch_record_start:
            return False
        hits = unintended_robot_contacts(records[-1])
        if not hits:
            return False
        branch.setdefault("first_robot_obstacle_contact", {
            "timestep": records[-1].get("timestep"),
            "physical_action_index": action_counter[0] - 1,
            "contacts": hits,
            "source": "per_action_MuJoCo_contacts",
            "stage_a_policy": "record_and_continue_until_target_contact_or_physical_tracking_failure",
            "action_taken_before_contact_detection": True,
        })
        return False

    def checkpoint_runtime_snapshot(
        checkpoint_name: str,
        observed_state: dict[str, Any],
    ) -> None:
        """Save an exact live checkpoint at expensive grasp-phase boundaries."""
        exported = bridge.request({"op": "export_runtime_snapshot"})
        observed_vector = _snapshot_state_vector(observed_state)
        exported_vector = _snapshot_state_vector(exported.get("snapshot_state", {}))
        checkpoint: dict[str, Any] = {
            "name": checkpoint_name,
            "export_status": exported.get("status"),
            "model_geometry_sha256": exported.get("model_geometry_sha256"),
            "integration_state_matches_observation": bool(
                observed_vector.shape == exported_vector.shape
                and np.array_equal(observed_vector, exported_vector)
            ),
        }
        if (
            exported.get("status") == "complete"
            and isinstance(exported.get("payload"), bytes)
            and exported.get("model_geometry_sha256") == model_sha
            and checkpoint["integration_state_matches_observation"]
        ):
            path = output_dir / f"{candidate_id}__{checkpoint_name}_runtime_snapshot.pkl"
            path.write_bytes(exported["payload"])
            checkpoint.update({
                "status": "saved_exact_runtime_snapshot",
                "path": str(path),
                "sha256": _sha256_bytes(exported["payload"]),
                "timestep": float(observed_vector[0]),
            })
        else:
            checkpoint["status"] = "snapshot_export_or_identity_check_failed"
        branch.setdefault("runtime_snapshot_checkpoints", []).append(checkpoint)

    action_counter = [0]
    monitor = monitor_type(required_samples=3)
    initial_grasp_state = bridge.request({
            "op": "grasp_state",
            "carried_binding": target_binding,
            "expected_instruction": context["instruction"],
        })
    initial_live_target_contact_sides = target_finger_contact_sides(initial_grasp_state)
    monitor.update(initial_grasp_state, rollout_step=-1)
    initial_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
    initial_runtime_state = initial_observation.get("snapshot_state", {})
    # Geometry screening assumes the jaws can open to their model limits, but
    # the live Panda often starts halfway open. A 39 mm live aperture cannot
    # admit a 42 mm target even when the scratch candidate is reachable. Open
    # with real controller actions before approaching; retain a verified grasp.
    gripper_ranges = np.asarray(context.get("gripper_joint_ranges", []), dtype=np.float64)
    initial_gripper_qpos = np.asarray(
        initial_runtime_state.get("robot_states", [{}])[0].get("gripper_qpos", []),
        dtype=np.float64,
    )
    if (
        gripper_ranges.shape != (2, 2)
        or initial_gripper_qpos.shape != (2,)
        or not np.isfinite(gripper_ranges).all()
        or not np.isfinite(initial_gripper_qpos).all()
    ):
        raise RuntimeError("pregrasp_live_gripper_joint_limits_or_qpos_unavailable")
    open_gripper_qpos_target = np.asarray(
        [gripper_ranges[0, 1], gripper_ranges[1, 0]], dtype=np.float64
    )
    pregrasp_open_actions = 0
    pregrasp_open_qpos_tolerance_m = 0.0005
    pregrasp_open_qpos_start = initial_gripper_qpos.copy()
    if (
        initial_grasp_state.get("grasp_confirmed") is not True
        and not initial_live_target_contact_sides
    ):
        while (
            pregrasp_open_actions < 16
            and float(np.max(np.abs(open_gripper_qpos_target - initial_gripper_qpos)))
            > pregrasp_open_qpos_tolerance_m
        ):
            _record_step(
                bridge,
                np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
                "task_success_first_pregrasp_open_gripper",
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
            pregrasp_open_actions += 1
            initial_runtime_state = records[-1]["snapshot_state"]
            initial_gripper_qpos = np.asarray(
                initial_runtime_state.get("robot_states", [{}])[0].get("gripper_qpos", []),
                dtype=np.float64,
            )
            if initial_gripper_qpos.shape != (2,) or not np.isfinite(initial_gripper_qpos).all():
                raise RuntimeError("pregrasp_live_gripper_qpos_became_unavailable")
        if pregrasp_open_actions:
            initial_grasp_state = records[-1].get("grasp_state", {})
    branch["pregrasp_gripper_opening"] = {
        "status": (
            "already_verified_grasp_preserved"
            if initial_grasp_state.get("grasp_confirmed") is True
            else (
                "opening_deferred_to_preserve_measured_target_finger_contact"
                if initial_live_target_contact_sides else (
                    "opened_to_model_joint_limits"
                    if float(np.max(np.abs(open_gripper_qpos_target - initial_gripper_qpos)))
                    <= pregrasp_open_qpos_tolerance_m
                    else "open_action_budget_exhausted_before_joint_limits"
                )
            )
        ),
        "measured_target_contact_sides_at_branch_start": list(initial_live_target_contact_sides),
        "opening_skipped_to_preserve_live_contact": bool(initial_live_target_contact_sides),
        "actions": pregrasp_open_actions,
        "gripper_qpos_start_m": pregrasp_open_qpos_start.tolist(),
        "gripper_qpos_target_m": open_gripper_qpos_target.tolist(),
        "gripper_qpos_end_m": initial_gripper_qpos.tolist(),
        "joint_limit_tolerance_m": pregrasp_open_qpos_tolerance_m,
        "model_maximum_pad_separation_m": context.get("maximum_pad_separation_m"),
        "state_was_changed_only_by_recorded_physical_OSC_actions": True,
    }
    initial_object = _object_state(
        initial_runtime_state, int(context["target_binding"]["root_body_id"])
    )
    initial_grasp_world = (
        _runtime_object_pose(initial_object)
        @ np.asarray(candidate["object_to_eef"], dtype=np.float64)
    )
    initial_eef_world = _runtime_eef_pose(initial_runtime_state)
    close_hold_eef_world = initial_eef_world.copy()
    close_probe_start_runtime_state = initial_runtime_state
    initial_pose_aligned, initial_grasp_distance_m, initial_grasp_rotation_error_rad = (
        _live_contact_pose_is_close_to_candidate(
            initial_eef_world, initial_grasp_world,
        )
    )
    initial_contact_sides = target_finger_contact_sides(initial_grasp_state)
    live_bilateral_target_contact = len(initial_contact_sides) == 2
    live_unilateral_target_contact = len(initial_contact_sides) == 1
    contact_free_candidate_pose_close_probe = bool(
        not initial_contact_sides
        and initial_grasp_state.get("grasp_confirmed") is not True
        and initial_grasp_distance_m <= 0.010
        and initial_grasp_rotation_error_rad <= 0.08
    )
    if contact_free_candidate_pose_close_probe:
        # No contact must be preserved, so finish the last few millimeters to
        # the geometry-derived pad-center pose while the real gripper closes.
        close_hold_eef_world = initial_grasp_world.copy()
    initial_pad_contact_normal_diagnostics: dict[str, Any] = {
        "status": "not_applicable_without_bilateral_target_contact"
    }
    bilateral_contact_surfaces_admissible = False
    if live_bilateral_target_contact:
        live_contacts = bridge.request({"op": "inspect_contacts"})
        initial_pad_contact_normal_diagnostics = _target_pad_contact_normal_diagnostics(
            live_contacts, int(target_binding["body_id"]),
        )
        # Missing normals remain eligible for a bounded physical probe. A
        # complete pair of same-facing surfaces, however, is a top/edge brush
        # rather than opposing jaw contact and should not trigger a costly
        # close-plus-lift sequence.
        bilateral_contact_surfaces_admissible = bool(
            initial_pad_contact_normal_diagnostics.get("status") != "complete"
            or initial_pad_contact_normal_diagnostics.get("surfaces_are_opposed") is True
        )
    candidate_orientation_aligned = bool(
        initial_grasp_rotation_error_rad <= LIVE_CONTACT_CLOSE_ROTATION_TOLERANCE_RAD
    )
    unilateral_close_probe_position_match = bool(
        live_unilateral_target_contact
        # A contact alone is insufficient: the other pad may still be outside
        # the object, as happens when a downward approach first touches a can's
        # upper rim.  Keep the pose-held close probe local to the grasp frame.
        # Once the active target finger itself is in MuJoCo contact, however,
        # the candidate's nominal orientation is only a prior: the physical
        # close probe holds the measured EEF pose and the live stability
        # monitor decides whether this contact is actually graspable.
        and initial_grasp_distance_m <= stage_a_unilateral_close_probe_position_m
    )
    unilateral_surface_approach_start = bool(
        live_unilateral_target_contact
        and stage_a_unilateral_close_probe_position_m < initial_grasp_distance_m
        <= LIVE_UNILATERAL_APPROACH_CONTINUE_POSITION_M
        and initial_grasp_rotation_error_rad <= NEAR_SURFACE_START_ROTATION_RAD
    )
    # Real bilateral pad contacts provide stronger evidence than matching the
    # nominal candidate pose. A physically obstructed surface approach can
    # stop short of that pose while both pads already clamp opposite sides.
    # A closed, bilateral-contact Snapshot is stronger still: slight object
    # rotation during closure must not route it back through repeated close
    # pulses, which can push the object out of the gripper.
    preclosed_bilateral_contact = bool(
        live_bilateral_target_contact
        and initial_grasp_state.get("gripper_closed") is True
    )
    close_from_live_contact_evidence = bool(
        preclosed_bilateral_contact
        # A measured contact from each active target-finger pad is direct
        # physical evidence that the gripper is already at the object.  The
        # nominal generated grasp frame can differ substantially when a
        # fixture or earlier contact deflects the EEF, so do not route away
        # from an otherwise useful bilateral-contact Snapshot merely because
        # that stale frame is misaligned.  Closure remains a bounded real
        # MuJoCo probe and must still pass the live grasp stability monitor.
        or (live_bilateral_target_contact and bilateral_contact_surfaces_admissible)
        or unilateral_close_probe_position_match
        or contact_free_candidate_pose_close_probe
    )
    close_from_live_contact = bool(
        close_from_live_contact_evidence and not force_route_before_contact_close
    )
    skip_pregrasp_route = bool(
        close_from_live_contact or unilateral_surface_approach_start
    )
    branch["initial_target_finger_contact_sides"] = list(initial_contact_sides)
    branch["live_bilateral_target_contact_observed"] = live_bilateral_target_contact
    branch["initial_pad_contact_normal_diagnostics"] = initial_pad_contact_normal_diagnostics
    branch["live_bilateral_target_contact_surfaces_admissible"] = bilateral_contact_surfaces_admissible
    branch["contact_free_candidate_pose_close_probe"] = {
        "admitted": contact_free_candidate_pose_close_probe,
        "position_error_m": initial_grasp_distance_m,
        "position_limit_m": 0.010,
        "rotation_error_rad": initial_grasp_rotation_error_rad,
        "rotation_limit_rad": 0.08,
        "physical_success_still_requires_opposing_contacts_and_lift": True,
    }
    branch["live_unilateral_target_contact_observed"] = live_unilateral_target_contact
    branch["live_unilateral_close_probe_position_match"] = unilateral_close_probe_position_match
    branch["live_unilateral_close_probe_action_limit"] = (
        MAX_LIVE_UNILATERAL_CLOSE_PROBE_ACTIONS
        if unilateral_close_probe_position_match else 0
    )
    branch["live_unilateral_surface_approach_continuation"] = {
        "enabled": unilateral_surface_approach_start,
        "position_error_to_candidate_grasp_m": initial_grasp_distance_m,
        "position_limit_m": LIVE_UNILATERAL_APPROACH_CONTINUE_POSITION_M,
        "rotation_error_rad": initial_grasp_rotation_error_rad,
        "reason": (
            "single_pad_contact_is_above_or_offset_from_generated_grasp_frame"
            if unilateral_surface_approach_start else None
        ),
    }
    branch["live_unilateral_close_probe_position_threshold_m"] = (
        stage_a_unilateral_close_probe_position_m
    )
    branch["preclosed_bilateral_contact_snapshot"] = preclosed_bilateral_contact
    branch["initial_gripper_closed"] = bool(
        initial_grasp_state.get("gripper_closed") is True
    )
    branch["live_contact_close_candidate_orientation_aligned"] = candidate_orientation_aligned
    branch["live_contact_close_position_match_required"] = not (
        preclosed_bilateral_contact
        or (live_bilateral_target_contact and bilateral_contact_surfaces_admissible)
        or unilateral_close_probe_position_match
        or contact_free_candidate_pose_close_probe
    )
    branch["live_contact_close_basis"] = (
        "verified_preclosed_bilateral_target_pad_contacts_on_runtime_snapshot"
        if preclosed_bilateral_contact else (
            "verified_bilateral_target_pad_contacts_without_nominal_pose_gate"
            if live_bilateral_target_contact and close_from_live_contact else (
                "verified_unilateral_target_contact_and_near_candidate_pose"
                if unilateral_close_probe_position_match else (
                    "contact_free_live_EEF_near_candidate_grasp_pose"
                    if contact_free_candidate_pose_close_probe else "candidate_pose_alignment"
                )
            )
        )
    )
    branch["live_unilateral_close_probe_candidate_rotation_error_ignored"] = bool(
        unilateral_close_probe_position_match
    )
    branch["initial_eef_to_migrated_grasp_pose_m"] = initial_grasp_distance_m
    branch["initial_eef_to_migrated_grasp_rotation_error_rad"] = initial_grasp_rotation_error_rad
    branch["live_contact_close_pose_aligned_with_candidate"] = initial_pose_aligned
    branch["live_contact_close_pose_tolerance"] = {
        "position_error_max_m": LIVE_CONTACT_CLOSE_POSITION_TOLERANCE_M,
        "rotation_error_max_rad": LIVE_CONTACT_CLOSE_ROTATION_TOLERANCE_RAD,
    }
    branch["pregrasp_route_skipped_due_to_live_target_contact"] = skip_pregrasp_route
    branch["live_contact_close_deferred_until_selected_route_end"] = bool(
        force_route_before_contact_close and close_from_live_contact_evidence
    )
    if close_from_live_contact:
        if live_bilateral_target_contact and not preclosed_bilateral_contact:
            # Do not commit an unverified sidewall slide while the open pads
            # already make real bilateral target contact. A scratch route can
            # be blocked by the same object that produced those contacts, and
            # the old 20 mm normal-guided slide could erase the only useful
            # state before testing closure. First spend the bounded, pose-held
            # physical close probe; if it cannot stabilize a grasp, its exact
            # Runtime Snapshot remains available for a later side-entry branch.
            branch["live_contact_sidewall_reposition"] = {
                "status": "deferred_until_after_bounded_live_contact_close_probe",
                "reason": "preserve_measured_bilateral_contact_before_testing_grasp_closure",
                "normal_diagnostics": initial_pad_contact_normal_diagnostics,
            }
        if len(initial_contact_sides) == 2:
            settle_actions = 0
            settled_samples = 0
            maximum_linear_speed = 0.0
            maximum_angular_speed = 0.0
            # If both pads touch but the gripper is still open, a zero action
            # leaves the object unsupported and can erase the only useful
            # contact state before the close probe. Close immediately; only
            # an already-closed bilateral grasp can safely spend actions on
            # velocity settling.
            settle_action_limit = (
                MAX_LIVE_CONTACT_SETTLE_ACTIONS
                if preclosed_bilateral_contact and initial_pose_aligned
                else 0
            )
            settle_start_object = _object_state(
                initial_runtime_state, int(target_binding["body_id"])
            )
            settle_start_position = np.asarray(
                settle_start_object.get("root_pos"), dtype=np.float64
            )
            while settle_actions < settle_action_limit:
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
                    "skipped_to_preserve_bilateral_contact_before_close_probe"
                    if settle_action_limit == 0
                    else (
                        "settled_with_bilateral_contact"
                        if settled_samples >= LIVE_CONTACT_SETTLE_REQUIRED_SAMPLES
                        else "action_budget_exhausted_bilateral_contact_preserved"
                    )
                ),
                "skip_reason": (
                    "live_bilateral_pad_contact_is_outside_nominal_candidate_position_tolerance"
                    if settle_action_limit == 0 else None
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
        live_close_action_limit = (
            MAX_CLOSE_CONTROL_ACTIONS if live_bilateral_target_contact
            else (
                MAX_CLOSE_CONTROL_ACTIONS
                if contact_free_candidate_pose_close_probe
                else MAX_LIVE_UNILATERAL_CLOSE_PROBE_ACTIONS
            )
        )
        branch["live_contact_close_probe_action_budget"] = live_close_action_limit
        branch["live_contact_close_probe_kind"] = (
            "bilateral_contact_close"
            if live_bilateral_target_contact else (
                "contact_free_candidate_pose_close"
                if contact_free_candidate_pose_close_probe
                else "bounded_unilateral_contact_close"
            )
        )
        branch["live_contact_close_pose_hold_target_world"] = np.r_[
            close_hold_eef_world[:3, 3],
            _wxyz(close_hold_eef_world[:3, :3]),
        ].tolist()
        branch["live_contact_close_pose_hold_controller"] = (
            "feedback_OSC_tracks_the_contact_snapshot_EEF_pose_on_every_close_action"
        )
        live_close_actions = 0
        bilateral_contact_loss_actions = 0
        closure_stable_samples = 0
        closure_position_tolerance_m = 0.0002
        prior_gripper_qpos = np.asarray(
            initial_runtime_state.get("robot_states", [{}])[0].get("gripper_qpos", []),
            dtype=np.float64,
        )
        maximum_gripper_qpos_step_m = 0.0
        branch["live_contact_close_probe_contact_sides"] = []
        branch["live_contact_physical_closure_settle"] = {
            "required_consecutive_stable_samples": 2,
            "maximum_per_joint_qpos_change_m": closure_position_tolerance_m,
            "gripper_qpos_start_m": prior_gripper_qpos.tolist(),
            "cartesian_motion_during_physical_settle": "zero_OSC_delta",
            "verification_basis": "measured_runtime_gripper_joint_qpos_and_live_bilateral_target_pad_contacts",
        }
        while (
            not preclosed_bilateral_contact
            and live_close_actions < live_close_action_limit
            and (not monitor.stable or closure_stable_samples < 2)
        ):
            live_runtime_state = (
                close_probe_start_runtime_state
                if live_close_actions == 0
                else records[-1]["snapshot_state"]
            )
            if monitor.stable:
                # The stability monitor verifies real opposing contacts and a
                # fully issued close command, but actuator qpos can still be
                # moving several millimeters per control step. Hold the live
                # OSC target exactly in place while MuJoCo finishes the
                # physical squeeze; do not begin lifting on command state.
                close_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            else:
                live_eef_world = _runtime_eef_pose(live_runtime_state)
                close_action = world_pose_tracking_action(
                    live_eef_world[:3, 3], _wxyz(live_eef_world[:3, :3]),
                    close_hold_eef_world[:3, 3], _wxyz(close_hold_eef_world[:3, :3]),
                    controller_fields,
                    gripper_action=1.0,
                )
            _record_step(
                bridge,
                close_action,
                (
                    "task_success_first_live_bilateral_contact_close_probe"
                    if len(initial_contact_sides) == 2 else (
                        "task_success_first_contact_free_candidate_pose_close_probe"
                        if contact_free_candidate_pose_close_probe
                        else "task_success_first_live_unilateral_contact_close_probe"
                    )
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
            contact_sides = target_finger_contact_sides(records[-1]["grasp_state"])
            branch["live_contact_close_probe_contact_sides"].append(contact_sides)
            current_gripper_qpos = np.asarray(
                records[-1].get("snapshot_state", {}).get("robot_states", [{}])[0].get("gripper_qpos", []),
                dtype=np.float64,
            )
            if (
                prior_gripper_qpos.shape != (2,)
                or current_gripper_qpos.shape != (2,)
                or not np.isfinite(prior_gripper_qpos).all()
                or not np.isfinite(current_gripper_qpos).all()
            ):
                raise RuntimeError("live_contact_close_probe_gripper_qpos_unavailable")
            gripper_qpos_step_m = float(np.max(np.abs(current_gripper_qpos - prior_gripper_qpos)))
            maximum_gripper_qpos_step_m = max(maximum_gripper_qpos_step_m, gripper_qpos_step_m)
            if (
                monitor.stable
                and len(contact_sides) == 2
                and gripper_qpos_step_m <= closure_position_tolerance_m
            ):
                closure_stable_samples += 1
            elif monitor.stable:
                closure_stable_samples = 0
            prior_gripper_qpos = current_gripper_qpos
            if len(initial_contact_sides) == 2 and len(contact_sides) < 2:
                bilateral_contact_loss_actions += 1
            else:
                bilateral_contact_loss_actions = 0
            if live_unilateral_target_contact and not contact_sides:
                branch["live_unilateral_close_probe_contact_lost"] = {
                    "action": live_close_actions,
                    "contact_sides": contact_sides,
                }
                raise RuntimeError(
                    "live_unilateral_target_contact_lost_during_pose_held_close_probe"
                )
            if (
                not monitor.stable
                and bilateral_contact_loss_actions
                >= MAX_LIVE_CLOSE_BILATERAL_CONTACT_LOSS_ACTIONS
            ):
                branch["live_contact_close_probe_early_stop"] = {
                    "reason": "bilateral_target_pad_contact_lost_during_pose_held_close",
                    "consecutive_actions_without_bilateral_contact": bilateral_contact_loss_actions,
                    "maximum_consecutive_actions": MAX_LIVE_CLOSE_BILATERAL_CONTACT_LOSS_ACTIONS,
                }
                raise RuntimeError(
                    "live_bilateral_target_contact_lost_during_pose_held_close_probe"
                )
        branch["live_contact_close_probe_actions"] = live_close_actions
        branch["live_contact_close_probe_grasp"] = monitor.diagnostics()
        branch["live_contact_close_probe_skipped"] = preclosed_bilateral_contact
        branch["live_contact_physical_closure_settle"].update({
            "actions": live_close_actions,
            "stable_physical_closure_samples": closure_stable_samples,
            "maximum_observed_per_joint_qpos_step_m": maximum_gripper_qpos_step_m,
            "gripper_qpos_end_m": prior_gripper_qpos.tolist(),
            "status": (
                "verified" if preclosed_bilateral_contact or closure_stable_samples >= 2
                else "not_settled_within_close_action_budget"
            ),
        })
        if not monitor.stable and not preclosed_bilateral_contact:
            raise RuntimeError("live_target_contact_close_probe_did_not_stabilize")
        if (
            not preclosed_bilateral_contact
            and live_bilateral_target_contact
            and closure_stable_samples < 2
        ):
            raise RuntimeError("live_bilateral_grasp_physical_closure_not_settled")
        branch["live_contact_close_probe_succeeded"] = bool(
            monitor.stable or preclosed_bilateral_contact
        )
    else:
        branch["live_contact_close_probe_succeeded"] = False
    _advance_pose.controller_fields = controller_fields
    route = (
        np.asarray(rrt.get("eef_reference_poses_world"), dtype=np.float64)
        if rrt.get("preserve_eef_waypoints") is True
        else _compress_eef_route(rrt.get("eef_reference_poses_world"))
    )
    branch["rrt_eef_waypoint_count"] = int(len(route))
    route_action_budget = route_control_action_budget(
        route,
        minimum_actions=74,
        maximum_actions=MAX_PREFIX_CONTROL_ACTIONS,
    )
    branch["pregrasp_route_action_budget"] = route_action_budget
    branch["local_replans"] = []
    # Keep screen-derived IK with the candidate before crossing into this
    # branch helper.  approach_rows is owned by the caller's screening scope
    # and is intentionally not a global/implicit dependency here.
    candidate_pregrasp_ik_seed_qpos = candidate.get(
        "_pregrasp_ik_seed_qpos"
    )
    if candidate_pregrasp_ik_seed_qpos is not None:
        candidate_pregrasp_ik_seed_qpos = np.asarray(
            candidate_pregrasp_ik_seed_qpos, dtype=np.float64,
        )
        if (
            candidate_pregrasp_ik_seed_qpos.shape != (7,)
            or not np.isfinite(candidate_pregrasp_ik_seed_qpos).all()
        ):
            candidate_pregrasp_ik_seed_qpos = None

    def advance_with_local_replan(
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        phase: str,
        maximum_steps: int,
        target_pose_provider: Callable[
            [dict[str, Any]], tuple[np.ndarray, np.ndarray]
        ] | None = None,
        max_translation_step_m: float = 0.008,
        position_tolerance_m: float = 0.004,
        rotation_tolerance_rad: float = 0.045,
        step_callback=None,
        stop_condition=None,
    ) -> bool | None:
        try:
            reached = _advance_pose(
                bridge, target_position, target_quaternion, phase, candidate_id,
                action_counter, maximum_steps, target_binding, context["instruction"],
                policy, monitor, records, frames,
                stop_condition=stop_condition,
                stagnation_window_actions=(
                    12 if target_pose_provider is None or stop_condition is not None else 0
                ),
                position_tolerance_m=position_tolerance_m,
                rotation_tolerance_rad=rotation_tolerance_rad,
                max_translation_step_m=max_translation_step_m,
                target_pose_provider=target_pose_provider,
                step_callback=step_callback,
            )
            return reached
        except (TimeoutError, RuntimeError) as original_error:
            is_stagnation = (
                isinstance(original_error, RuntimeError)
                and "_waypoint_stagnated_for_" in str(original_error)
            )
            if not isinstance(original_error, TimeoutError) and not is_stagnation:
                raise
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
                "ik_seed_qpos": (
                    [candidate_pregrasp_ik_seed_qpos.tolist()]
                    if candidate_pregrasp_ik_seed_qpos is not None else None
                ),
                "safety_geometry_policy": policy,
                "planner_config": {
                    "step_rad": 0.22,
                    "connect_max_steps": 12,
                    "goal_bias": 0.2,
                    "max_iterations": 400,
                    "max_config_checks": 4000,
                    "max_candidate_routes": 1,
                    "search_time_budget_s": local_replan_rrt_time_budget_s,
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
                "candidate_pregrasp_endpoint_ik_seed_used": (
                    candidate_pregrasp_ik_seed_qpos is not None
                ),
                "ik_selection_seed_source": "exact_live_arm_qpos_at_local_replan_start",
                "remaining_physical_action_budget": route_action_cap,
                "safety_certified": False,
            })
            candidate["_local_replans"] = list(branch["local_replans"])
            for local_waypoint in replan_route[1:]:
                local_remaining = MAX_BRANCH_CONTROL_ACTIONS - action_counter[0]
                if local_remaining <= 0:
                    raise RuntimeError("local_segment_replan_physical_action_budget_exhausted")
                reached = _advance_pose(
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
                    stop_condition=record_robot_obstacle_contact,
                    stagnation_window_actions=12,
                    position_tolerance_m=position_tolerance_m,
                    rotation_tolerance_rad=rotation_tolerance_rad,
                    max_translation_step_m=max_translation_step_m,
                )
            branch["local_replans"][-1]["status"] = "physical_replan_segment_reached"
            candidate["_local_replans"] = list(branch["local_replans"])

    if not skip_pregrasp_route:
        route_source = str(rrt.get("planner_source", ""))
        adaptive_jointspace_route = bool(
            rrt.get("preserve_eef_waypoints") is True
            or route_source.startswith("adaptive_jointspace_continuation_")
        )
        route_translation_step_m = (
            0.02
            if route_source == "near_target_direct_pregrasp_approach.v1"
            or route_source in {
                "active_BDDL_geometry_high_OSC_approach_after_RRT_failure",
                "explicit_active_BDDL_geometry_direct_OSC_approach",
            }
            or adaptive_jointspace_route
            or any(
                item.get("route_source")
                == "active_BDDL_geometry_high_OSC_approach_after_RRT_failure"
                for item in rrt.get("route_candidates", [])
            )
            else 0.008
        )
        branch["pregrasp_route_max_translation_step_m"] = route_translation_step_m
        route_waypoints = route[1:]
        if stage_a_pregrasp_staged_clearance_height_m > 0.0 and len(route) >= 2:
            # A grasp can start with the wrist above a container after an
            # earlier predicate was completed. Moving directly toward the
            # next object can rotate the forearm through that container even
            # when the EEF path is collision free. First lift while holding
            # the live wrist orientation, translate above the candidate
            # pregrasp, rotate at clearance height, then descend.
            live_eef = _runtime_eef_pose(initial_runtime_state)
            live_quaternion = _wxyz(live_eef[:3, :3])
            pregrasp = np.asarray(route[-1], dtype=np.float64)
            if pregrasp.shape != (7,) or not np.isfinite(pregrasp).all():
                raise RuntimeError("staged_pregrasp_clearance_endpoint_unavailable")
            clearance_z = max(
                float(stage_a_pregrasp_staged_clearance_height_m),
                float(live_eef[2, 3]),
                float(pregrasp[2]) + 0.025,
            )
            held_orientation_start = np.r_[
                live_eef[:2, 3], clearance_z, live_quaternion,
            ]
            if abs(stage_a_pregrasp_lateral_detour_m) > 1e-6:
                displacement_xy = pregrasp[:2] - live_eef[:2, 3]
                displacement_norm = float(np.linalg.norm(displacement_xy))
                if displacement_norm <= 1e-6:
                    raise RuntimeError("staged_pregrasp_lateral_detour_has_degenerate_xy_route")
                # Keep the full transit parallel to the direct start-to-goal
                # line but offset to one side. This lets the forearm clear a
                # container even when a point-to-point wrist route crosses
                # its rim; return to the candidate side only after passing it.
                side_xy = np.array(
                    [-displacement_xy[1], displacement_xy[0]], dtype=np.float64,
                ) / displacement_norm * float(stage_a_pregrasp_lateral_detour_m)
                start_side_at_live_height = np.r_[
                    live_eef[:2, 3] + side_xy, float(live_eef[2, 3]), live_quaternion,
                ]
                start_side_clearance = np.r_[
                    live_eef[:2, 3] + side_xy, clearance_z, live_quaternion,
                ]
                target_side = np.r_[
                    pregrasp[:2] + side_xy, clearance_z, live_quaternion,
                ]
                target_side_rotated = np.r_[
                    pregrasp[:2] + side_xy, clearance_z, pregrasp[3:],
                ]
                target_at_clearance = np.r_[
                    pregrasp[:2], clearance_z, pregrasp[3:],
                ]
                if stage_a_pregrasp_lift_before_lateral_detour:
                    route_waypoints = [
                        # Some held-object snapshots start above a container.
                        # Leaving that opening sideways can strike its rim, so
                        # first lift with the live wrist orientation, then take
                        # the obstacle-offset corridor at clearance height.
                        held_orientation_start,
                        start_side_clearance,
                        target_side,
                        target_side_rotated,
                        target_at_clearance,
                        pregrasp,
                    ]
                    detour_waypoint_order = [
                        "vertical_lift_with_live_wrist_orientation",
                        "lateral_shift_at_clearance_with_live_wrist_orientation",
                        "parallel_transit_beyond_obstacle_with_live_wrist_orientation",
                        "rotate_at_clearance_beyond_obstacle",
                        "lateral_return_to_candidate_at_clearance",
                        "descend_to_candidate_pregrasp",
                    ]
                else:
                    route_waypoints = [
                        # Move the wrist sideways before lifting. Prior physical
                        # branches showed that raising in the starting arm
                        # configuration can stall without contact, or sweep link6
                        # into a nearby obstacle before the detour begins.
                        start_side_at_live_height,
                        start_side_clearance,
                        target_side,
                        target_side_rotated,
                        target_at_clearance,
                        pregrasp,
                    ]
                    detour_waypoint_order = [
                        "lateral_shift_at_live_height_with_live_wrist_orientation",
                        "vertical_lift_after_leaving_start_corridor",
                        "parallel_transit_beyond_obstacle_with_live_wrist_orientation",
                        "rotate_at_clearance_beyond_obstacle",
                        "lateral_return_to_candidate_at_clearance",
                        "descend_to_candidate_pregrasp",
                    ]
            else:
                horizontal_transit = np.r_[
                    pregrasp[:2], clearance_z, live_quaternion,
                ]
                orientation_at_clearance = np.r_[
                    pregrasp[:2], clearance_z, pregrasp[3:],
                ]
                route_waypoints = [
                    held_orientation_start,
                    horizontal_transit,
                    orientation_at_clearance,
                    pregrasp,
                ]
            branch["pregrasp_staged_clearance_route"] = {
                "enabled": True,
                "clearance_height_world_m": clearance_z,
                "lateral_detour_signed_m": float(stage_a_pregrasp_lateral_detour_m),
                "lift_before_lateral_detour": bool(
                    stage_a_pregrasp_lift_before_lateral_detour
                ),
                "waypoint_order": (
                    detour_waypoint_order
                    if abs(stage_a_pregrasp_lateral_detour_m) > 1e-6 else [
                        "vertical_lift_with_live_wrist_orientation",
                        "horizontal_transit_with_live_wrist_orientation",
                        "rotate_at_clearance_over_candidate_pregrasp",
                        "descend_to_candidate_pregrasp",
                    ]
                ),
                "lateral_detour_start_waypoint_world": (
                    (
                        start_side_clearance
                        if stage_a_pregrasp_lift_before_lateral_detour
                        else start_side_at_live_height
                    ).tolist()
                    if abs(stage_a_pregrasp_lateral_detour_m) > 1e-6 else None
                ),
                "lateral_detour_target_waypoint_world": (
                    target_side.tolist()
                    if abs(stage_a_pregrasp_lateral_detour_m) > 1e-6 else None
                ),
                "start_eef_pose_world": [
                    live_eef[:3, 3].tolist(), live_quaternion.tolist(),
                ],
                "candidate_pregrasp_pose_world": pregrasp.tolist(),
                "state_changes_only_through_recorded_MuJoCo_OSC_actions": True,
                "safety_certified": False,
            }
            route_translation_step_m = 0.02
        else:
            branch["pregrasp_staged_clearance_route"] = {
                "enabled": False,
                "configured_clearance_height_world_m": float(
                    stage_a_pregrasp_staged_clearance_height_m
                ),
            }
        for route_waypoint_index, waypoint in enumerate(route_waypoints):
            final_adaptive_endpoint = bool(
                adaptive_jointspace_route
                and route_waypoint_index == len(route_waypoints) - 1
            )
            reached = advance_with_local_replan(
                waypoint[:3],
                waypoint[3:],
                "task_success_first_rrt_osc_pregrasp_route",
                max(1, route_action_budget - action_counter[0]),
                max_translation_step_m=route_translation_step_m,
                position_tolerance_m=(
                    0.005
                    if final_adaptive_endpoint
                    else 0.025
                    if adaptive_jointspace_route
                    else 0.004
                ),
                rotation_tolerance_rad=(
                    0.045
                    if final_adaptive_endpoint
                    else 0.12
                    if adaptive_jointspace_route
                    else 0.045
                ),
                stop_condition=record_robot_obstacle_contact,
            )
            if action_counter[0] > route_action_budget:
                raise RuntimeError("RRT_OSC_pregrasp_route_action_budget_exceeded")
    branch["pregrasp_route_reached"] = not skip_pregrasp_route
    branch["pregrasp_route_skipped_due_to_live_target_contact"] = skip_pregrasp_route
    if not skip_pregrasp_route:
        pregrasp_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
        checkpoint_runtime_snapshot("pregrasp", pregrasp_observation.get("snapshot_state", {}))

    def checkpoint_first_target_contact(record: dict[str, Any], _frame: dict[str, Any]) -> None:
        if branch.get("first_target_contact_snapshot") is not None:
            return
        sides = target_finger_contact_sides(record.get("grasp_state", {}))
        if sides:
            checkpoint_runtime_snapshot("first_target_pad_contact", record["snapshot_state"])
            branch["first_target_contact_snapshot"] = {
                "finger_sides": sides,
                "timestep": record.get("timestep"),
                "snapshot_checkpoint_index": len(branch["runtime_snapshot_checkpoints"]) - 1,
            }

    surface_approach_record_start = [len(records)]

    def target_pad_contact_sides_observed() -> list[str]:
        if len(records) <= surface_approach_record_start[0]:
            return []
        return target_finger_contact_sides(records[-1].get("grasp_state", {}))

    def target_pad_contact_observed() -> bool:
        return bool(target_pad_contact_sides_observed())

    def bilateral_target_pad_contact_observed() -> bool:
        return len(target_pad_contact_sides_observed()) == 2

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
                    step_callback=checkpoint_first_target_contact,
                )
            remaining = MAX_APPROACH_CONTROL_ACTIONS - (action_counter[0] - approach_start_actions)
            if remaining <= 0:
                raise RuntimeError("migrated_object_relative_grasp_endpoint_action_budget_exhausted")
        else:
            remaining = MAX_APPROACH_CONTROL_ACTIONS
        surface_approach_record_start[0] = len(records)
        branch["surface_approach_contact_alignment_policy"] = {
            "stop_on_bilateral_target_pad_contact": True,
            "stop_on_unilateral_contact_only_when_candidate_pose_aligned": True,
            "unilateral_position_tolerance_m": 0.012,
            "unilateral_rotation_tolerance_rad": 0.08,
            "unaligned_unilateral_contact_action": "continue_object_relative_approach_with_existing_target_displacement_abort",
            "safety_certified": False,
        }
        def stop_surface_approach() -> bool:
            nonlocal close_hold_eef_world
            # An open gripper that has touched the target is already at a
            # useful physical branch point only when the nominal grasp pose is
            # already aligned. A unilateral pad can touch the top edge of a
            # can or box several centimetres before the jaw center reaches the
            # intended grasp. Continue the short object-relative approach in
            # that case; the existing target-displacement abort bounds any
            # physical push, while bilateral contact or a near-aligned single
            # pad still stops immediately for a feedback-held close.
            record_robot_obstacle_contact()
            sides = target_pad_contact_sides_observed()
            if sides:
                live_state = records[-1]["snapshot_state"]
                if len(sides) == 2:
                    should_stop = True
                    alignment = {"position_error_m": None, "rotation_error_rad": None}
                    stop_reason = "bilateral_target_pad_contact"
                else:
                    live_eef = _runtime_eef_pose(live_state)
                    target_position, target_quaternion = object_relative_target_provider(
                        object_to_eef
                    )(live_state)
                    target_pose = np.eye(4, dtype=np.float64)
                    target_pose[:3, :3] = Rotation.from_quat(
                        np.asarray(target_quaternion, dtype=np.float64)[[1, 2, 3, 0]]
                    ).as_matrix()
                    target_pose[:3, 3] = np.asarray(target_position, dtype=np.float64)
                    position_error, rotation_error = _pose_tracking_error_components(
                        live_eef, target_pose,
                    )
                    should_stop = _pose_tracking_error_within_tolerance(
                        position_error, rotation_error, 0.012, 0.08,
                    )
                    alignment = {
                        "position_error_m": position_error,
                        "rotation_error_rad": rotation_error,
                    }
                    stop_reason = (
                        "aligned_unilateral_target_pad_contact"
                        if should_stop else "unaligned_unilateral_contact_continued"
                    )
                branch.setdefault("surface_approach_contact_events", []).append({
                    "timestep": records[-1].get("timestep"),
                    "finger_contact_sides": sides,
                    "candidate_pose_alignment": alignment,
                    "decision": stop_reason,
                    "target_object_root_position_m": _object_state(
                        live_state, int(target_binding["body_id"])
                    ).get("root_pos"),
                })
                if not should_stop:
                    return False
                # Preserve the exact measured contact pose for a feedback
                # controlled close. A zero OSC delta is not a pose hold under
                # contact: the arm can drift while the gripper actuator closes
                # and lose the only useful unilateral target contact.
                close_hold_eef_world = _runtime_eef_pose(
                    live_state
                )
                branch["surface_approach_stop_reason"] = stop_reason
                branch["surface_contact_pose_hold_target_world"] = np.r_[
                    close_hold_eef_world[:3, 3],
                    _wxyz(close_hold_eef_world[:3, :3]),
                ].tolist()
                branch["surface_contact_pose_hold_controller"] = (
                    "feedback_OSC_tracks_the_first_target_pad_contact_pose_during_close"
                )
                return True
            return False

        approach_reached = advance_with_local_replan(
            grasp_world[:3, 3],
            _wxyz(grasp_world[:3, :3]),
            "task_success_first_surface_approach" if not migrated_waypoints else "migrated_object_relative_grasp_endpoint",
            remaining,
            target_pose_provider=object_relative_target_provider(object_to_eef),
            step_callback=checkpoint_first_target_contact,
            stop_condition=stop_surface_approach,
            # Drive long free-space grasp approaches with the controller's
            # full validated translation increment. The old 8 mm increment
            # needed >60 actions for a 75 mm reach from a legal snapshot.
            max_translation_step_m=0.02,
        )
        branch["surface_approach_target_frame"] = "live_target_object_root_pose_times_generated_object_to_eef"
        branch["surface_approach_stopped_on_target_contact"] = bool(
            approach_reached is False and branch.get("surface_approach_stop_reason")
        )
        branch["surface_approach_target_pad_contact_sides_at_stop"] = (
            target_pad_contact_sides_observed()
            if branch["surface_approach_stopped_on_target_contact"] else []
        )
        if (
            not approach_reached
            and not target_pad_contact_observed()
            and branch.get("first_robot_obstacle_contact")
        ):
            raise RuntimeError("surface_approach_stopped_on_first_robot_obstacle_contact")
    branch["surface_approach_reached"] = bool(
        close_from_live_contact or approach_reached is not False
        or target_pad_contact_observed()
    )

    close_actions = 0
    while (
        not preclosed_bilateral_contact
        and close_actions < MAX_CLOSE_CONTROL_ACTIONS
        and not monitor.stable
    ):
        if branch.get("surface_approach_stopped_on_target_contact"):
            live_eef_world = _runtime_eef_pose(records[-1]["snapshot_state"])
            action = world_pose_tracking_action(
                live_eef_world[:3, 3],
                _wxyz(live_eef_world[:3, :3]),
                close_hold_eef_world[:3, 3],
                _wxyz(close_hold_eef_world[:3, :3]),
                controller_fields,
                gripper_action=1.0,
            )
        else:
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
    if not monitor.stable and not preclosed_bilateral_contact:
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
    live_contact_lift = bool(close_from_live_contact)
    lift_action_budget = (
        LIVE_CONTACT_LIFT_CONTROL_ACTIONS if live_contact_lift
        else MAX_LIFT_CONTROL_ACTIONS
    )
    lift_translation_step_m = (
        LIVE_CONTACT_LIFT_TRANSLATION_STEP_M if live_contact_lift else 0.008
    )
    lift_phase = (
        "task_success_first_live_contact_incremental_40mm_lift"
        if live_contact_lift else "task_success_first_40mm_lift"
    )
    branch["lift_control_profile"] = {
        "strategy": (
            "incremental_low_speed_lift_from_verified_live_bilateral_contact"
            if live_contact_lift else "standard_pose_tracking_lift"
        ),
        "target_lift_m": LIFT_COMMAND_M,
        "minimum_verified_object_lift_m": MINIMUM_OBJECT_LIFT_M,
        "early_stop_when_minimum_lift_and_stable_bilateral_contact": True,
        "max_translation_step_m": lift_translation_step_m,
        "action_budget": lift_action_budget,
    }

    def stable_minimum_lift_reached() -> bool:
        """Avoid overshooting a short valid grasp lift into contact loss."""
        if not records or not monitor.stable:
            return False
        latest_state = records[-1].get("snapshot_state", {})
        try:
            latest_target = _object_state(
                latest_state, int(target_binding["body_id"]),
            )
            latest_lift = float(
                latest_target["root_pos"][2] - target_before["root_pos"][2]
            )
        except (KeyError, TypeError, ValueError):
            return False
        return bool(
            np.isfinite(latest_lift)
            and latest_lift >= MINIMUM_OBJECT_LIFT_M
            and records[-1].get("grasp_state", {}).get("grasp_confirmed") is True
        )

    _advance_pose(
        bridge,
        eef_before + np.asarray([0.0, 0.0, LIFT_COMMAND_M]),
        orientation,
        lift_phase,
        candidate_id,
        action_counter,
        lift_action_budget,
        target_binding,
        context["instruction"],
        policy,
        monitor,
        records,
        frames,
        stop_condition=stable_minimum_lift_reached,
        max_translation_step_m=lift_translation_step_m,
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


def _generated_geometry_fallback_allowed(
    candidate: dict[str, Any], planner: dict[str, Any],
) -> bool:
    """Admit a bounded physical OSC approach after scratch IK succeeds but RRT does not.

    Stage A continuation snapshots are valid search starts too: limiting this
    fallback to cold starts discarded physically reachable candidates whenever
    RRT timed out, even though its scratch query had left the live state intact.
    """
    return generated_geometry_high_approach_admitted(candidate, planner)


def _forbidden_object_root_positions(
    safety_geometry_policy: dict[str, Any], snapshot_state: dict[str, Any],
    *, active_target_object_id: str,
) -> list[dict[str, Any]]:
    """Read current root positions for active-BDDL forbidden scene objects."""
    object_states = snapshot_state.get("object_states", {})
    if not isinstance(object_states, dict):
        return []
    rows: list[dict[str, Any]] = []
    for item in safety_geometry_policy.get("forbidden_objects", []):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id", ""))
        if not object_id or object_id == active_target_object_id:
            continue
        state = object_states.get(f"objects_dict:{object_id}")
        if not isinstance(state, dict):
            continue
        position = np.asarray(state.get("root_pos", []), dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            continue
        rows.append({
            "object_id": object_id,
            "position_world_m": position.tolist(),
            "position_source": "read_only_live_runtime_snapshot_object_root",
        })
    return rows


def _adaptive_geometry_approach_route(
    bridge: BridgeSession,
    *,
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    forbidden_obstacle_positions_world: list[dict[str, Any]] | None = None,
    minimum_transit_height_m: float | None = None,
    approach_direction_world: np.ndarray | None,
    live_arm_qpos: np.ndarray,
    instruction: str,
    model_sha: str,
    state_sha: str,
    policy: dict[str, Any],
) -> tuple[list[list[float]] | None, dict[str, Any]]:
    """Choose a short lift-and-traverse OSC route whose waypoints all have IK.

    The previous fixed +10 cm route could demand an unreachable high pose and
    then waste physical actions discovering that OSC saturation. This samples
    five progressively lower transit planes, using read-only scratch IK and a
    warm joint seed between short Cartesian segments. Only the returned route
    is physically executed; collisions and contacts remain real Stage A
    dynamics and no state is written during preflight.
    """
    start = np.asarray(start_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    live_q = np.asarray(live_arm_qpos, dtype=np.float64)
    if (
        start.shape != (7,) or target.shape != (7,)
        or live_q.shape != (7,)
        or not all(np.isfinite(x).all() for x in (start, target, live_q))
    ):
        return None, {"status": "invalid_route_seed", "physical_actions": 0}
    if minimum_transit_height_m is not None and (
        not np.isfinite(float(minimum_transit_height_m))
        or float(minimum_transit_height_m) <= 0.0
    ):
        return None, {"status": "invalid_contact_clearance_transit_height", "physical_actions": 0}

    route_trials: list[dict[str, Any]] = []
    route_start_q = live_q.copy()
    # Keep the robot at its current clearance while leaving the source area,
    # then descend along the final traverse.  Using one shared height for the
    # whole route made a floor-level grasp inherit the initial 0.5 m EEF
    # height: the later surface-approach controller spent its entire 60-action
    # budget just descending, before it could reach the grasp pose.
    target_clearances = (0.14, 0.18, 0.22, 0.26, 0.30)
    approach = None
    if approach_direction_world is not None:
        approach = np.asarray(approach_direction_world, dtype=np.float64)
        if approach.shape != (3,) or not np.isfinite(approach).all():
            return None, {"status": "invalid_approach_direction", "physical_actions": 0}
        approach_norm = float(np.linalg.norm(approach))
        if approach_norm <= 1e-8:
            return None, {"status": "degenerate_approach_direction", "physical_actions": 0}
        approach /= approach_norm
    side_detour_enabled = bool(
        approach is not None and float(np.linalg.norm(approach[:2])) >= 0.15
    )
    start_rotation = Rotation.from_quat(start[3:][[1, 2, 3, 0]])
    target_rotation = Rotation.from_quat(target[3:][[1, 2, 3, 0]])
    horizontal_delta = target[:2] - start[:2]
    horizontal_length = float(np.linalg.norm(horizontal_delta))
    horizontal_direction = (
        horizontal_delta / horizontal_length
        if horizontal_length > 1e-6 else np.zeros(2, dtype=np.float64)
    )
    horizontal_perpendicular = np.asarray(
        [-horizontal_direction[1], horizontal_direction[0]], dtype=np.float64,
    )
    obstacle_route_candidates: list[dict[str, Any]] = []

    for target_clearance in target_clearances:
        target_height = max(
            float(target[2]) + target_clearance,
            float(minimum_transit_height_m)
            if minimum_transit_height_m is not None else 0.12,
            0.12,
        )
        start_height = max(float(start[2]), target_height)
        start_high = start.copy()
        start_high[2] = start_height
        target_high = target.copy()
        target_high[2] = target_height
        segment_rotations = Slerp(
            [0.0, 1.0], Rotation.from_quat(np.stack([
                start_rotation.as_quat(), target_rotation.as_quat(),
            ]))
        )([1.0 / 3.0, 2.0 / 3.0])
        poses = [start_high]
        if not side_detour_enabled:
            obstacle_route_candidates = []
            if horizontal_length > 1e-6:
                for obstacle in forbidden_obstacle_positions_world or []:
                    obstacle_position = np.asarray(
                        obstacle.get("position_world_m", []), dtype=np.float64,
                    )
                    if obstacle_position.shape != (3,) or not np.isfinite(obstacle_position).all():
                        continue
                    relative_xy = obstacle_position[:2] - start[:2]
                    projection = float(np.dot(relative_xy, horizontal_direction))
                    if projection < 0.10 or projection > horizontal_length - 0.10:
                        continue
                    nearest_xy = start[:2] + projection * horizontal_direction
                    lateral_distance = float(np.linalg.norm(obstacle_position[:2] - nearest_xy))
                    vertical_distance = abs(float(obstacle_position[2]) - start_height)
                    if lateral_distance >= 0.19 or vertical_distance >= 0.24:
                        continue
                    signed_lateral = float(np.dot(relative_xy, horizontal_perpendicular))
                    if abs(signed_lateral) < 1e-6:
                        side_sign = 1.0 if horizontal_perpendicular[0] >= 0.0 else -1.0
                    else:
                        side_sign = -1.0 if signed_lateral > 0.0 else 1.0
                    obstacle_route_candidates.append({
                        "object_id": str(obstacle.get("object_id", "unknown")),
                        "projection_m": projection,
                        "lateral_distance_m": lateral_distance,
                        "vertical_distance_m": vertical_distance,
                        "signed_lateral_m": signed_lateral,
                        "side_sign": side_sign,
                    })
            if obstacle_route_candidates:
                blocker = min(
                    obstacle_route_candidates,
                    key=lambda row: (row["lateral_distance_m"], row["projection_m"]),
                )
                projection = float(blocker["projection_m"])
                side_offset = horizontal_perpendicular * float(blocker["side_sign"]) * 0.16
                before_distance = max(0.05, projection - 0.12)
                after_distance = min(horizontal_length - 0.05, projection + 0.12)
                detour_points = [
                    (before_distance / horizontal_length,
                     start[:2] + horizontal_direction * before_distance + side_offset),
                    (after_distance / horizontal_length,
                     start[:2] + horizontal_direction * after_distance + side_offset),
                ]
                blocker["lateral_clearance_m"] = 0.16
                blocker["waypoint_source"] = "active_BDDL_forbidden_object_root_projection"
            else:
                blocker = None
                detour_points = [
                    (1.0 / 3.0, start[:2] + horizontal_delta / 3.0),
                    (2.0 / 3.0, start[:2] + 2.0 * horizontal_delta / 3.0),
                ]
            for fraction, position_xy in detour_points:
                rotation = Slerp(
                    [0.0, 1.0], Rotation.from_quat(np.stack([
                        start_rotation.as_quat(), target_rotation.as_quat(),
                    ]))
                )([fraction])[0]
                pose = np.empty(7, dtype=np.float64)
                pose[:2] = position_xy
                pose[2] = start_height
                pose[3:] = rotation.as_quat()[[3, 0, 1, 2]]
                poses.append(pose)
            poses.append(target_high)
            # `target` is the actual candidate pregrasp, while `target_high`
            # is only a transit waypoint.  Omitting this final descent made
            # the later surface-approach controller start as much as 30 cm
            # above the grasp and cut diagonally through the target instead
            # of following the candidate's collision-aware approach axis.
            poses.append(target.copy())
            detour_margin_m = (
                float(blocker["lateral_clearance_m"])
                if blocker is not None else 0.0
            )
        else:
            # Stay high while crossing the workspace, reach beyond the target
            # on the chosen grasp approach side, lower there, then enter along
            # the object's free-space approach axis. This avoids dropping the
            # wrist through an intervening fixture while descending diagonally.
            detour_margin_m = 0.06
            detour = target_high.copy()
            detour[:3] -= approach * detour_margin_m
            detour[2] = start_height
            for fraction, rotation in zip((1.0 / 3.0, 2.0 / 3.0), segment_rotations):
                pose = np.empty(7, dtype=np.float64)
                pose[:3] = start_high[:3] + fraction * (detour[:3] - start_high[:3])
                pose[2] = start_height
                pose[3:] = rotation.as_quat()[[3, 0, 1, 2]]
                poses.append(pose)
            detour[3:] = target_high[3:]
            poses.extend([detour, target_high])
            detour_low = detour.copy()
            detour_low[2] = float(target[2])
            # Descend on the outside of the target to the actual pregrasp
            # height, then approach horizontally to that pregrasp.  The old
            # route stopped at `target_high`; the following surface approach
            # consequently descended diagonally through the target geometry.
            poses[-1:] = [detour_low, target.copy()]

        previous_q = route_start_q.copy()
        previous_pose = start
        checks: list[dict[str, Any]] = []
        jointspace_reference_poses: list[list[float]] = [start.tolist()]
        rejected: dict[str, Any] | None = None
        for index, pose in enumerate(poses):
            if np.linalg.norm(pose[:3] - previous_pose[:3]) < 1e-5 and index == 0:
                # An already occupied lift endpoint needs no redundant IK RPC.
                previous_pose = pose
                continue
            reply = bridge.request({
                "op": "task_success_first_ik_preflight",
                "expected_instruction": instruction,
                "model_geometry_sha256": model_sha,
                "start_state_sha256": state_sha,
                "goal_pose_world": {
                    "position_world_m": pose[:3].tolist(),
                    "quaternion_wxyz": pose[3:].tolist(),
                },
                "ik_seed_qpos": [previous_q.tolist()],
                "ik_selection_seed_qpos": previous_q.tolist(),
                "ik_solver_max_nfev": 120,
                "jointspace_reference_start_qpos": previous_q.tolist(),
                "jointspace_reference_max_step_rad": 0.10,
                "safety_geometry_policy": policy,
            })
            qpos = reply.get("ik_goal_qpos")
            if (
                reply.get("status") != "goal_ik_preflight_passed"
                or reply.get("live_state_unchanged") is not True
                or not isinstance(qpos, list)
                or len(qpos) != 7
                or not np.isfinite(np.asarray(qpos, dtype=np.float64)).all()
            ):
                rejected = {
                    "waypoint_index": index,
                    "failure_reason": reply.get("failure_reason") or reply.get("status"),
                    "live_state_unchanged": reply.get("live_state_unchanged"),
                }
                break
            qpos_array = np.asarray(qpos, dtype=np.float64)
            joint_step = float(np.linalg.norm(qpos_array - previous_q))
            reference_path = reply.get("eef_jointspace_reference_path_world")
            if not isinstance(reference_path, list) or len(reference_path) < 2:
                rejected = {
                    "waypoint_index": index,
                    "failure_reason": "jointspace_reference_path_missing_or_empty",
                }
                break
            checks.append({
                "waypoint_index": index,
                "position_world_m": pose[:3].tolist(),
                "endpoint_joint_step_rad": joint_step,
                "ik_goal_qpos": qpos_array.tolist(),
                "jointspace_reference_waypoints": len(reference_path) - 1,
                "jointspace_reference_path_status": "bridge_verified_continuous_IK_chain",
                "endpoint_joint_step_is_not_a_route_continuity_test": True,
                "IK_solver_evaluations": [
                    row.get("solver_evaluations")
                    for row in reply.get("ik_candidates", [])
                ],
            })
            # The bridge has already proven a bounded joint-space reference
            # chain from `previous_q` to this endpoint. Rejecting the route on
            # the endpoint L2 delta alone discards continuous paths that wind
            # through multiple small IK steps (for example, 1.43 rad over 11
            # verified waypoints). Stage A must test such routes with real OSC
            # and MuJoCo feedback; the endpoint delta is retained as a metric.
            previous_q = qpos_array
            previous_pose = pose
            jointspace_reference_poses.extend(reference_path[1:])
        route_trials.append({
            "transit_height_m": target_height,
            "start_transit_height_m": start_height,
            "target_transit_height_m": target_height,
            "target_clearance_above_grasp_m": target_clearance,
            "target_side_detour_margin_m": detour_margin_m,
            "obstacle_route_candidate_count": len(obstacle_route_candidates),
            "selected_obstacle_detour": (
                blocker if not side_detour_enabled and blocker is not None else None
            ),
            "status": "IK_waypoints_passed" if rejected is None else "rejected",
            "waypoint_checks": checks,
            "rejection": rejected,
        })
        if rejected is None:
            route = jointspace_reference_poses
            return route, {
                "status": "ik_feasible_route_selected",
                "route_source": "adaptive_jointspace_continuation_descending_to_target_clearance_with_OSC_EEF_references.v2",
                "selected_transit_height_m": target_height,
                "selected_start_transit_height_m": start_height,
                "selected_target_transit_height_m": target_height,
                "contact_clearance_transit_floor_m": (
                    float(minimum_transit_height_m)
                    if minimum_transit_height_m is not None else None
                ),
                "target_side_detour_margin_m": detour_margin_m,
                "final_waypoint_is_candidate_pregrasp": True,
                "side_approach_descends_outside_then_translates_to_pregrasp": side_detour_enabled,
                "transit_height_trials": route_trials,
                "obstacle_route_candidates": obstacle_route_candidates,
                "all_waypoints_preflighted_without_live_state_change": True,
                "jointspace_reference_max_step_rad": 0.10,
                "physical_controller": "MuJoCo OSC_POSE tracks scratch-FK references; no qpos writes",
                "safety_certified": False,
            }
    return None, {
        "status": "no_IK_continuous_adaptive_transit_route",
        "transit_height_trials": route_trials,
        "all_waypoint_preflights_were_read_only": True,
        "safety_certified": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    global MAX_BRANCH_CONTROL_ACTIONS
    run_started = time.monotonic()
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor

    configure_stage_a_contact_only_substep_audit(
        bool(getattr(args, "contact_only_substep_audit", False))
    )
    cold_start_continuation = bool(getattr(args, "cold_start_continuation", False))
    cold_start = bool(getattr(args, "cold_start", False) or cold_start_continuation)
    if cold_start and (
        args.experience_library is not None
        or args.previous_search_report
        or args.historical_grasp_pose_priors_json is not None
        or args.strict_grasp_screen_only
        or args.strict_screen_reference_report is not None
        or (args.resume_from_runtime_snapshot is not None and not cold_start_continuation)
        or args.candidate_start_index != 0
        or (
            not cold_start_continuation
            and (
                args.candidate_tilt_degrees
                or args.candidate_roll_degrees
            )
        )
    ):
        # Object-local contact offsets are generated from the active BDDL
        # collision geometry. They are valid fresh-t0 candidates and import
        # no historical poses, actions, or partial-task state.
        raise ValueError("cold_start_cannot_use_historical_experience_or_partial_search_inputs")
    if cold_start_continuation and args.resume_from_runtime_snapshot is None:
        raise ValueError("cold_start_continuation_requires_a_runtime_snapshot_from_the_active_episode")
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
                "start_state": (
                    "same_run_identity_checked_partial_task_snapshot"
                    if cold_start_continuation else "identity_checked_legal_t0_snapshot"
                ),
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
            "geometry_candidate_pool_offset": int(getattr(args, "geometry_candidate_pool_offset", 0)),
            "candidate_generation_limit": (
                int(args.candidate_start_index + args.maximum_candidates)
                if args.strict_grasp_screen_only else _candidate_generation_limit(args)
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
            "stage_a_pregrasp_staged_clearance_height_world_m": float(
                args.stage_a_pregrasp_staged_clearance_height_m
            ),
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
        restored_state = _snapshot_state_vector(initial.get("snapshot_state", {}))
        if restored_state.shape != search_start_state.shape or not np.array_equal(restored_state, search_start_state):
            raise RuntimeError("cold_start_search_runtime_snapshot_restore_not_exact")
        if resume_descriptor is not None:
            report["resumed_from_runtime_snapshot"] = {
                "path": str(resume_descriptor["path"]),
                "payload_sha256": resume_descriptor["payload_sha256"],
                "integration_state_sha256": _state_sha(restored_state),
                "status": "verified_exact",
                "other_completed_native_predicates_preserved": True,
                "safety_certified": False,
            }
        instruction = str(manifest["instruction"])
        if initial.get("instruction") != instruction:
            raise RuntimeError("restored_snapshot_instruction_does_not_match_source_BDDL")
        native = bridge.request({"op": "native_task_status"})
        if native.get("status") != "complete" or native.get("task_success") is not False:
            raise RuntimeError("source_t0_native_task_state_is_not_false_or_complete")
        resumed_timestep = int(native.get("timestep", 0))
        horizon_remaining = int(args.demonstration_horizon) - resumed_timestep
        configured_branch_action_limit = int(args.branch_control_action_budget)
        # Do not issue a physical action after the active Bridge episode horizon.
        # MuJoCo terminates the environment at the horizon but may report done=False;
        # a following step raises and prevents terminal Snapshot export. Reserve one
        # action so an unsuccessful branch remains recoverable at the exact boundary.
        MAX_BRANCH_CONTROL_ACTIONS = min(
            configured_branch_action_limit,
            max(1, horizon_remaining - 1),
        )
        report["search_budget"]["branch_action_budget_clamp"] = {
            "configured_branch_control_actions_max": configured_branch_action_limit,
            "search_start_timestep": resumed_timestep,
            "demonstration_horizon": int(args.demonstration_horizon),
            "horizon_remaining_actions": horizon_remaining,
            "effective_branch_control_actions_max": MAX_BRANCH_CONTROL_ACTIONS,
            "reserved_actions_before_horizon": 1,
            "reason": "preserve_recoverable_snapshot_without_stepping_a_terminated_episode",
        }
        report["search_budget"]["branch_control_actions_max"] = MAX_BRANCH_CONTROL_ACTIONS
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
        route_observation = bridge.request({
            "op": "observe", "include_snapshot_state": True,
        })
        def arm_joint_limit_diagnostics(inventory: dict[str, Any]) -> dict[str, Any]:
            rows = []
            for joint in inventory.get("joints", []):
                if not isinstance(joint, dict):
                    continue
                name = str(joint.get("joint_name", ""))
                if not name.startswith("robot0_joint"):
                    continue
                try:
                    index = int(name.removeprefix("robot0_joint")) - 1
                    qpos = float(joint.get("qpos"))
                    lower = float(joint.get("lower"))
                    upper = float(joint.get("upper"))
                except (TypeError, ValueError):
                    continue
                if not 0 <= index < 7 or not np.isfinite([qpos, lower, upper]).all():
                    continue
                rows.append({
                    "joint_name": name,
                    "arm_index": index,
                    "qpos_rad": qpos,
                    "lower_rad": lower,
                    "upper_rad": upper,
                    "lower_excess_rad": max(0.0, lower - qpos),
                    "upper_excess_rad": max(0.0, qpos - upper),
                    "excess_rad": max(0.0, lower - qpos, qpos - upper),
                })
            return {
                "within_limits": bool(rows) and all(row["excess_rad"] <= 1e-8 for row in rows),
                "maximum_excess_rad": max((row["excess_rad"] for row in rows), default=None),
                "joints": rows,
            }

        initial_limit_state = arm_joint_limit_diagnostics(limits)
        if (
            not initial_limit_state["within_limits"]
            and initial_limit_state["maximum_excess_rad"] is not None
            and initial_limit_state["maximum_excess_rad"] > 0.0002
            and initial_limit_state["maximum_excess_rad"] <= 0.02
        ):
            from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action

            controller_fields = controller["robots"][0]["fields"]
            recovery_monitor = GraspStabilityMonitor(required_samples=3)
            recovery: dict[str, Any] = {
                "status": "bounded_unheld_OSC_recovery_in_progress",
                "initial_joint_limits": initial_limit_state,
                "maximum_actions": 4,
                "command": "joint-limit-directed world-Z OSC correction; real MuJoCo only",
                "robot_or_object_state_written_directly": False,
                "safety_certified": False,
                "action_indices": [],
            }
            for recovery_index in range(4):
                current_limit_state = arm_joint_limit_diagnostics(limits)
                if current_limit_state["within_limits"]:
                    break
                if current_limit_state["maximum_excess_rad"] is None or current_limit_state["maximum_excess_rad"] > 0.02:
                    break
                current_robot = route_observation["snapshot_state"]["robot_states"][0]
                current_position = np.asarray(current_robot["eef_pos"], dtype=np.float64)
                current_quaternion = np.asarray(current_robot["eef_quat"], dtype=np.float64)
                limiting_joint = max(
                    current_limit_state["joints"], key=lambda row: row["excess_rad"],
                )
                z_correction_m = 0.002
                if (
                    limiting_joint["joint_name"].endswith("joint4")
                    and limiting_joint["upper_excess_rad"] > 0.0
                ):
                    # A physical finite-difference probe on the exact failure
                    # Snapshot showed +Z increased the joint4 upper-limit
                    # excess.  Reverse the local direction and use a bounded
                    # 8 mm target so OSC tracking clears integration noise in
                    # a few audited steps without writing arm qpos.
                    z_correction_m = -0.008
                target_position = current_position + np.asarray([0.0, 0.0, z_correction_m])
                target_quaternion = current_quaternion.copy()
                recovery_rotation = None
                if limiting_joint["joint_name"].endswith("joint7"):
                    current_rotation = Rotation.from_quat(current_quaternion[[1, 2, 3, 0]])
                    axis_world = current_rotation.as_matrix()[:, 2]
                    direction = 1.0 if limiting_joint["lower_excess_rad"] > 0.0 else -1.0
                    target_rotation = Rotation.from_rotvec(axis_world * direction * 0.01) * current_rotation
                    target_quaternion = target_rotation.as_quat()[[3, 0, 1, 2]]
                    recovery_rotation = {
                        "joint_name": limiting_joint["joint_name"],
                        "axis_world": axis_world.tolist(),
                        "direction": direction,
                        "command_rotation_rad": 0.01,
                    }
                action = world_pose_tracking_action(
                    current_position, current_quaternion,
                    target_position, target_quaternion,
                    controller_fields, gripper_action=0.0,
                ).astype(np.float32)
                action_index = len(records)
                _record_step(
                    bridge, action, "pregrasp_joint_limit_OSC_recovery",
                    "unheld_joint_limit_micro_recovery", action_index,
                    target_binding, instruction, policy, recovery_monitor,
                    records, frames,
                    planning_tracking={
                        "tracking_mode": "bounded_unheld_OSC_pose_recovery_from_joint_limit_violation",
                        "target_eef_pose_world": np.r_[target_position, target_quaternion].tolist(),
                        "world_z_correction_m": z_correction_m,
                        "joint_limits_before": current_limit_state,
                        "limiting_joint_wrist_unwind": recovery_rotation,
                        "state_write": False,
                        "safety_certified": False,
                    },
                )
                recovery["action_indices"].append(action_index)
                route_observation = bridge.request({
                    "op": "observe", "include_snapshot_state": True,
                })
                limits = bridge.request({"op": "robot_joint_limit_inventory"})
                after_limit_state = arm_joint_limit_diagnostics(limits)
                recovery["latest_joint_limits"] = after_limit_state
                if after_limit_state["within_limits"]:
                    break
                if (
                    after_limit_state["maximum_excess_rad"] is None
                    or after_limit_state["maximum_excess_rad"]
                    >= current_limit_state["maximum_excess_rad"] - 1e-6
                ):
                    break
            final_limit_state = arm_joint_limit_diagnostics(limits)
            recovery["final_joint_limits"] = final_limit_state
            recovery["status"] = (
                "recovered_by_real_OSC"
                if final_limit_state["within_limits"]
                else "OSC_recovery_did_not_clear_joint_limit"
            )
            if recovery["action_indices"]:
                exported_recovery = bridge.request({"op": "export_runtime_snapshot"})
                if exported_recovery.get("status") != "complete" or not isinstance(exported_recovery.get("payload"), bytes):
                    raise RuntimeError("pregrasp_joint_limit_recovery_snapshot_export_failed")
                recovered_state = _snapshot_state_vector(exported_recovery.get("snapshot_state", {}))
                observed_state = _snapshot_state_vector(route_observation.get("snapshot_state", {}))
                if (
                    exported_recovery.get("model_geometry_sha256") != model_sha
                    or not np.array_equal(recovered_state, observed_state)
                ):
                    raise RuntimeError("pregrasp_joint_limit_recovery_snapshot_state_mismatch")
                recovery_path = output_dir / "pregrasp_joint_limit_recovery_runtime_snapshot.pkl"
                recovery_path.write_bytes(exported_recovery["payload"])
                recovery["terminal_snapshot"] = {
                    "path": str(recovery_path),
                    "payload_sha256": _sha256_bytes(exported_recovery["payload"]),
                    "integration_state_sha256": _state_sha(recovered_state),
                }
                search_start_payload = exported_recovery["payload"]
                search_start_state = recovered_state
                context = bridge.request({
                    "op": "task_success_first_context",
                    "expected_instruction": instruction,
                    "target_predicate": args.target_predicate,
                })
                if context.get("status") != "complete" or context.get("native_goal_value") is not False:
                    raise RuntimeError("pregrasp_joint_limit_recovery_changed_active_BDDL_goal_state")
                report["pregrasp_joint_limit_recovery"] = recovery
            else:
                report["pregrasp_joint_limit_recovery"] = recovery

        if (
            not initial_limit_state["within_limits"]
            and initial_limit_state["maximum_excess_rad"] is not None
            and initial_limit_state["maximum_excess_rad"] <= 0.0002
        ):
            # At OAH/L1T3 t248, one-step +Z and -Z OSC probes both increased
            # the joint5 excursion. Do not spend more physical actions on an
            # uncalibrated retreat for this sub-0.2-mrad integration error;
            # scratch-only IK screening may clip that tiny value, while the
            # live Snapshot and every subsequent action remain untouched and
            # fully audited.
            report["pregrasp_joint_limit_recovery"] = {
                "status": "micro_violation_deferred_to_physical_task_search",
                "initial_joint_limits": initial_limit_state,
                "physical_actions": 0,
                "scratch_only_screen_tolerance_rad": 0.0002,
                "robot_or_object_state_written_directly": False,
                "safety_certified": False,
            }

        initial_safety = bridge.request({
            "op": "safety_geometry_check", "safety_geometry_policy": policy,
        })
        initial_contact_escape = robot_contact_escape_direction(
            initial_safety.get("contact_identities", []),
            ignore_object_id=str(context["target_object_id"]),
        )
        report["initial_robot_contact_escape"] = initial_contact_escape
        route_obstacle_positions = _forbidden_object_root_positions(
            policy,
            route_observation.get("snapshot_state", {}),
            active_target_object_id=str(context["target_object_id"]),
        )
        report["route_obstacle_positions_world"] = route_obstacle_positions
        candidate_generation_limit = _candidate_generation_limit(args)
        required_geometry_ids = set(args.candidate_id)
        required_geometry_ids.update(
            candidate_id.split("__approach_axis", 1)[0]
            for candidate_id in args.candidate_id
            if "__approach_axis" in candidate_id
        )
        candidate_pool, generation = generate_geometry_candidates(
            context,
            maximum_candidates=candidate_generation_limit,
            required_candidate_ids=required_geometry_ids,
        )
        pool_offset = int(getattr(args, "geometry_candidate_pool_offset", 0))
        if args.candidate_id:
            requested_ids = set(args.candidate_id)
            # A same-scene physical retry may pin a geometry candidate from an
            # earlier branch even though the live robot pose changes its rank.
            # Keep exact generated IDs addressable across the full active-scene
            # pool instead of silently making the default ranked window a gate.
            candidates = [
                candidate for candidate in candidate_pool
                if candidate.get("candidate_id") in requested_ids
            ]
            generation["geometry_candidate_pool_window"] = {
                "offset": pool_offset,
                "count": len(candidates),
                "generated_pool_count": len(candidate_pool),
                "generation_limit": candidate_generation_limit,
                "source": "active_scene_compiled_target_geometry_exact_id_selection",
                "requested_candidate_ids": sorted(requested_ids),
            }
        else:
            candidates = candidate_pool[pool_offset:pool_offset + args.maximum_candidates]
            generation["geometry_candidate_pool_window"] = {
                "offset": pool_offset,
                "count": len(candidates),
                "generated_pool_count": len(candidate_pool),
                "generation_limit": candidate_generation_limit,
                "source": "active_scene_compiled_target_geometry",
            }
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
            if missing_ids and cold_start:
                # Task-approach depth variants are normally generated before the
                # screen. A cold-start pin may name one such same-task variant;
                # regenerate it from its deterministic active-geometry parent
                # instead of importing a serialized candidate or replaying a path.
                target_pose_for_pin = _pose(
                    context["target_root_pose_world"]["position_m"],
                    context["target_root_pose_world"]["rotation"],
                )
                eef_position_for_pin = np.asarray(
                    context["eef_pose_world"]["position_m"], dtype=np.float64
                )
                base_candidates = {
                    str(candidate.get("candidate_id")): candidate
                    for candidate in candidate_pool
                }
                regenerated = []
                for requested_id in sorted(missing_ids):
                    if "__approach_axis" not in requested_id:
                        continue
                    parent_id = requested_id.split("__approach_axis", 1)[0]
                    parent = base_candidates.get(parent_id)
                    if parent is None:
                        continue
                    expanded, expansion_report = expand_task_approach_depth_candidates(
                        [parent],
                        target_pose_world=target_pose_for_pin,
                        eef_position_world=eef_position_for_pin,
                        target_collision_geometries=context["target_collision_geometries"],
                        depth_offset_fractions=(0.30, 0.60, 0.90, 1.20),
                    )
                    regenerated.extend(
                        candidate for candidate in expanded[1:]
                        if candidate.get("candidate_id") == requested_id
                    )
                    if regenerated:
                        generation["pinned_candidate_regenerated_from_active_geometry"] = {
                            "requested_candidate_id": requested_id,
                            "parent_candidate_id": parent_id,
                            "expansion": expansion_report,
                            "serialized_candidate_or_prior_action_used": False,
                        }
                candidates.extend(regenerated)
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
        task_approach_pass_ids: set[str] | None = None
        if not args.strict_grasp_screen_only:
            unfiltered_candidate_count = len(candidates)
            candidates, mechanically_infeasible = filter_mechanically_infeasible_grasp_candidates(
                candidates
            )
            generation["mechanical_opening_filter"] = {
                "applied": True,
                "source": "candidate_estimated_clamp_width_vs_active_maximum_pad_separation",
                "unfiltered_candidate_count": unfiltered_candidate_count,
                "feasible_candidate_count": len(candidates),
                "rejected_candidate_count": len(mechanically_infeasible),
                "rejected_candidates": mechanically_infeasible,
                "is_strict_safety_gate": False,
            }
            if cold_start and not args.candidate_id:
                candidates, generation["task_approach_depth_expansion"] = (
                    expand_task_approach_depth_candidates(
                        candidates,
                        target_pose_world=target_pose_for_variants,
                        eef_position_world=eef_position_for_variants,
                        target_collision_geometries=context["target_collision_geometries"],
                    )
                )
            task_approach_screen = bridge.request({
                "op": "screen_grasp_candidates",
                "expected_instruction": instruction,
                "model_geometry_sha256": model_sha,
                "start_state_sha256": _state_sha(search_start_state),
                "safety_geometry_policy": policy,
                "target_object_id": target_id,
                "target_root_body_id": int(target_binding["body_id"]),
                "clearance_m": 0.0,
                "strict_target_contact_policy": False,
                "task_approach_only": True,
                "closed_gripper_qpos": context["active_open_gripper_qpos"],
                "predicted_stable_object_pose_world": target_pose_for_variants.tolist(),
                "candidates": candidates,
            })
            if (
                task_approach_screen.get("screening_type")
                != "offline_scratch_task_approach_feasibility_not_safety_certification"
            ):
                raise RuntimeError(
                    "task_approach_feasibility_screen_unavailable_or_wrong_scope:"
                    f"{task_approach_screen.get('failure_reason') or task_approach_screen.get('status')}"
                )
            feasible_ids = task_approach_feasible_candidate_ids(task_approach_screen)
            approach_rows = {
                row["candidate_id"]: row
                for row in task_approach_screen.get("candidates", [])
            }
            screened_ids = {candidate["candidate_id"] for candidate in candidates}
            if feasible_ids - screened_ids or set(approach_rows) != screened_ids:
                raise RuntimeError("task_approach_screen_candidate_identity_set_mismatch")
            approach_rejections = [
                {
                    "candidate_id": candidate_id,
                    "failure_reasons": approach_rows[candidate_id].get("failure_reasons", []),
                    "failure_class": approach_rows[candidate_id].get("failure_class"),
                }
                for candidate_id in sorted(screened_ids - feasible_ids)
            ]
            live_eef_start_pose = _pose(
                context["eef_pose_world"]["position_m"],
                context["eef_pose_world"]["rotation"],
            )
            live_initial_grasp_state = bridge.request({
                "op": "grasp_state",
                "carried_binding": target_binding,
                "expected_instruction": instruction,
            })
            live_initial_contact_sides = target_finger_contact_sides(
                live_initial_grasp_state
            )
            near_surface_start_candidates: dict[str, dict[str, float]] = {}
            near_pregrasp_start_candidates: dict[str, dict[str, float]] = {}
            unilateral_contact_pose_candidates: dict[str, dict[str, Any]] = {}
            for candidate in candidates:
                candidate_grasp_pose = (
                    target_pose_for_variants
                    @ np.asarray(candidate["object_to_eef"], dtype=np.float64)
                )
                position_error, rotation_error = _pose_tracking_error_components(
                    live_eef_start_pose, candidate_grasp_pose,
                )
                if (
                    position_error <= NEAR_SURFACE_START_POSITION_M
                    and rotation_error <= NEAR_SURFACE_START_ROTATION_RAD
                ):
                    near_surface_start_candidates[str(candidate["candidate_id"])] = {
                        "eef_to_grasp_position_error_m": position_error,
                        "eef_to_grasp_rotation_error_rad": rotation_error,
                    }
                if (
                    len(live_initial_contact_sides) == 1
                    and position_error <= args.stage_a_unilateral_close_probe_position_m
                ):
                    unilateral_contact_pose_candidates[str(candidate["candidate_id"])] = {
                        "measured_target_finger_contact_sides": live_initial_contact_sides,
                        "eef_to_grasp_position_error_m": position_error,
                        "eef_to_grasp_rotation_error_rad": rotation_error,
                        "position_threshold_m": args.stage_a_unilateral_close_probe_position_m,
                        "candidate_rotation_error_ignored_after_measured_target_pad_contact": True,
                        "requires_bounded_real_MuJoCo_close_and_stability_monitor": True,
                        "live_state_modified": False,
                    }
                reachability = approach_rows[str(candidate["candidate_id"])].get(
                    "joint_reachability", {}
                )
                pregrasp_endpoint_ik = bool(
                    reachability.get("pregrasp", {}).get("reachable") is True
                )
                grasp_endpoint_ik = bool(
                    reachability.get("grasp_open_gripper", {}).get("reachable") is True
                )
                if grasp_endpoint_ik:
                    _, candidate_pregrasp_pose = object_relative_grasp_targets(
                        target_pose_for_variants,
                        candidate["object_to_eef"],
                        candidate["approach_direction_object"],
                        float(candidate["approach_distance_m"]),
                    )
                    pregrasp_position_error, pregrasp_rotation_error = (
                        _pose_tracking_error_components(
                            live_eef_start_pose, candidate_pregrasp_pose,
                        )
                    )
                    if (
                        pregrasp_position_error <= NEAR_PREGRASP_START_POSITION_M
                        and pregrasp_rotation_error <= NEAR_PREGRASP_START_ROTATION_RAD
                    ):
                        near_pregrasp_start_candidates[str(candidate["candidate_id"])] = {
                            "eef_to_pregrasp_position_error_m": pregrasp_position_error,
                            "eef_to_pregrasp_rotation_error_rad": pregrasp_rotation_error,
                            "grasp_endpoint_IK_verified_by_task_approach_screen": True,
                            "pregrasp_endpoint_IK_verified_by_task_approach_screen": pregrasp_endpoint_ik,
                            "live_pose_near_pregrasp_is_a_physical_fallback": not pregrasp_endpoint_ik,
                        }
            _write_json(output_dir / "task_approach_feasibility_screen.json", task_approach_screen)
            task_approach_pass_ids = set(feasible_ids)
            rejected_ids = screened_ids - feasible_ids
            physically_reachable_reject_ids = set()
            ik_unreachable_ids = set()
            for candidate_id in rejected_ids:
                reachability = approach_rows[candidate_id].get("joint_reachability", {})
                pregrasp_reachable = (
                    reachability.get("pregrasp", {}).get("reachable") is True
                )
                grasp_reachable = (
                    reachability.get("grasp_open_gripper", {}).get("reachable") is True
                )
                if pregrasp_reachable and grasp_reachable:
                    physically_reachable_reject_ids.add(candidate_id)
                else:
                    ik_unreachable_ids.add(candidate_id)
            generation["task_approach_feasibility_screen"] = {
                "applied": True,
                "scope": "open_gripper_pregrasp_to_grasp_cartesian_path_only",
                "candidate_count_before_screen": len(candidates),
                "candidate_count_after_screen": len(feasible_ids),
                "candidate_count_retained_as_stage_a_physical_fallback": len(
                    physically_reachable_reject_ids
                ),
                "candidate_count_excluded_for_missing_endpoint_ik": len(ik_unreachable_ids),
                "stage_a_selection_policy": (
                    "endpoint_IK_joint_travel_first_with_screen_outcome_only_as_tiebreaker"
                ),
                "screen_rejection_is_not_a_hard_stage_a_safety_gate": True,
                "rejected_candidates": approach_rejections,
                "live_state_unchanged": task_approach_screen.get("live_state_unchanged"),
                "strict_safety_gate_used": False,
                "safety_certified": False,
                "screen_report": str(output_dir / "task_approach_feasibility_screen.json"),
            }
            generation["near_surface_start_candidate_fallback"] = {
                "candidate_ids": sorted(near_surface_start_candidates),
                "diagnostics": near_surface_start_candidates,
                "position_threshold_m": NEAR_SURFACE_START_POSITION_M,
                "rotation_threshold_rad": NEAR_SURFACE_START_ROTATION_RAD,
                "retained_even_if_pregrasp_endpoint_IK_unavailable": True,
                "requires_real_MuJoCo_surface_approach_and_contact_close": True,
                "live_state_modified": False,
            }
            generation["near_pregrasp_start_candidate_fallback"] = {
                "candidate_ids": sorted(near_pregrasp_start_candidates),
                "diagnostics": near_pregrasp_start_candidates,
                "position_threshold_m": NEAR_PREGRASP_START_POSITION_M,
                "rotation_threshold_rad": NEAR_PREGRASP_START_ROTATION_RAD,
                "requires_task_approach_screen_grasp_endpoint_IK": True,
                "pregrasp_endpoint_IK_failure_does_not_block_near_live_pose_fallback": True,
                "requires_real_MuJoCo_pregrasp_and_surface_approach": True,
                "live_state_modified": False,
            }
            generation["live_unilateral_contact_close_candidate_fallback"] = {
                "candidate_ids": sorted(unilateral_contact_pose_candidates),
                "diagnostics": unilateral_contact_pose_candidates,
                "measured_target_finger_contact_sides": live_initial_contact_sides,
                "candidate_pose_position_threshold_m": args.stage_a_unilateral_close_probe_position_m,
                "candidate_pose_rotation_threshold_rad": NEAR_SURFACE_START_ROTATION_RAD,
                "screen_and_endpoint_IK_rejections_do_not_suppress_this_physical_probe": True,
                "live_state_modified": False,
                "safety_certified": False,
            }
            candidates = [
                candidate for candidate in candidates
                if candidate["candidate_id"] in (
                    feasible_ids
                    | physically_reachable_reject_ids
                    | set(near_surface_start_candidates)
                    | set(near_pregrasp_start_candidates)
                    | set(unilateral_contact_pose_candidates)
                )
            ]
        else:
            generation["mechanical_opening_filter"] = {
                "applied": False,
                "reason": "strict_screen_reports_all_candidates_including_impossible_widths",
                "is_strict_safety_gate": False,
            }
        if args.strict_grasp_screen_only:
            candidates = _candidate_window(
                candidates,
                start=args.candidate_start_index,
                count=args.maximum_candidates,
            )
        if task_approach_pass_ids is None:
            candidates = diversify_candidates(candidates)
        else:
            # Stage A seeks task completion and accepts physical contact. Keep
            # screen passes as a tie-breaker, while endpoint IK and live-posture
            # travel rank all reachable poses in one pool. This prevents a
            # conservative open-path screen tier from promoting a much harder
            # OSC joint winding above a lower-travel physical fallback.
            live_arm_qpos = np.asarray(context.get("arm_qpos", []), dtype=np.float64)
            candidates, posture_scores = _rank_by_endpoint_ik_joint_travel(
                candidates, approach_rows, live_arm_qpos, task_approach_pass_ids,
            )
            generation["endpoint_ik_joint_travel_ranking"] = {
                "applied": bool(posture_scores),
                "joint_travel_precedes_screen_outcome": True,
                "score": "L2(pregrasp_qpos-live_arm_qpos)+0.5*L2(grasp_qpos-pregrasp_qpos)",
                "live_arm_qpos_rad": live_arm_qpos.tolist() if live_arm_qpos.shape == (7,) else None,
                "ordered_candidates": posture_scores,
                "physical_attempt_still_requires_real_MuJoCo_and_native_BDDL": True,
                "safety_certified": False,
            }
        if not args.strict_grasp_screen_only:
            candidates = candidates[:args.maximum_candidates]
        if not candidates:
            if generation.get("task_approach_feasibility_screen", {}).get("applied"):
                raise RuntimeError(
                    "no_stage_a_grasp_candidate_with_pregrasp_and_grasp_endpoint_ik"
                )
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
            live_eef_pose = np.r_[
                context["eef_pose_world"]["position_m"],
                _wxyz(context["eef_pose_world"]["rotation"]),
            ]
            live_eef_transform = _pose(
                live_eef_pose[:3],
                Rotation.from_quat(live_eef_pose[3:][[1, 2, 3, 0]]).as_matrix(),
            )
            near_position_error, near_rotation_error = _pose_tracking_error_components(
                live_eef_transform, target_pose,
            )
            approach_reachability = approach_rows.get(
                str(candidate["candidate_id"]), {}
            ).get("joint_reachability", {})
            pregrasp_endpoint_ik_verified = bool(
                approach_reachability.get("pregrasp", {}).get("reachable") is True
                and approach_reachability.get("grasp_open_gripper", {}).get("reachable") is True
            )
            candidate_pregrasp_ik = approach_reachability.get(
                "pregrasp", {}
            ).get("qpos") if pregrasp_endpoint_ik_verified else None
            if (
                not pregrasp_endpoint_ik_verified
                and approach_reachability.get("grasp_open_gripper", {}).get("reachable") is True
            ):
                original_approach_distance = float(candidate["approach_distance_m"])
                short_approach_trials: list[dict[str, Any]] = []
                tested_distances = list(dict.fromkeys(
                    round(value, 6)
                    for value in (
                        max(0.045, original_approach_distance - 0.015),
                        max(0.030, original_approach_distance - 0.030),
                        max(0.015, original_approach_distance - 0.045),
                    )
                    if value < original_approach_distance - 1e-6
                ))
                seed_qpos_values = [
                    np.asarray(context["arm_qpos"], dtype=np.float64).tolist()
                ]
                grasp_seed = approach_reachability.get(
                    "grasp_open_gripper", {}
                ).get("qpos")
                if isinstance(grasp_seed, list) and len(grasp_seed) == 7:
                    grasp_seed_array = np.asarray(grasp_seed, dtype=np.float64)
                    if np.isfinite(grasp_seed_array).all():
                        seed_qpos_values.append(grasp_seed_array.tolist())
                for short_distance in tested_distances:
                    trial_target, trial_pregrasp = object_relative_grasp_targets(
                        object_pose,
                        candidate["object_to_eef"],
                        candidate["approach_direction_object"],
                        short_distance,
                    )
                    ik_reply = bridge.request({
                        "op": "task_success_first_ik_preflight",
                        "expected_instruction": instruction,
                        "model_geometry_sha256": model_sha,
                        "start_state_sha256": state_sha,
                        "goal_pose_world": {
                            "position_world_m": trial_pregrasp[:3, 3].tolist(),
                            "quaternion_wxyz": _wxyz(trial_pregrasp[:3, :3]).tolist(),
                        },
                        "ik_seed_qpos": seed_qpos_values,
                        "ik_selection_seed_qpos": seed_qpos_values[0],
                        "ik_solver_max_nfev": 120,
                        "jointspace_reference_start_qpos": seed_qpos_values[0],
                        "jointspace_reference_max_step_rad": 0.10,
                        "safety_geometry_policy": policy,
                    })
                    trial_qpos = ik_reply.get("ik_goal_qpos")
                    trial_passed = bool(
                        ik_reply.get("status") == "goal_ik_preflight_passed"
                        and ik_reply.get("live_state_unchanged") is True
                        and isinstance(trial_qpos, list)
                        and len(trial_qpos) == 7
                        and np.isfinite(np.asarray(trial_qpos, dtype=np.float64)).all()
                    )
                    short_approach_trials.append({
                        "approach_distance_m": short_distance,
                        "status": "read_only_pregrasp_IK_passed" if trial_passed else "pregrasp_IK_unreachable",
                        "failure_reason": ik_reply.get("failure_reason"),
                        "live_state_unchanged": ik_reply.get("live_state_unchanged"),
                    })
                    if trial_passed:
                        candidate["approach_distance_m"] = short_distance
                        candidate["_short_approach_ik_seed_qpos"] = trial_qpos
                        candidate["_short_approach_ik_trials"] = short_approach_trials
                        target_pose, pregrasp_pose = trial_target, trial_pregrasp
                        pregrasp_endpoint_ik_verified = True
                        candidate_pregrasp_ik = trial_qpos
                        approach_reachability["pregrasp"] = {
                            "reachable": True,
                            "qpos": trial_qpos,
                            "approach_distance_m": short_distance,
                            "source": "shortened_object_geometry_approach_distance_read_only_IK",
                        }
                        break
                if candidate.get("_short_approach_ik_trials") is None:
                    candidate["_short_approach_ik_trials"] = short_approach_trials
            if candidate_pregrasp_ik is not None:
                candidate_pregrasp_ik = np.asarray(candidate_pregrasp_ik, dtype=np.float64)
                if candidate_pregrasp_ik.shape == (7,) and np.isfinite(candidate_pregrasp_ik).all():
                    candidate["_pregrasp_ik_seed_qpos"] = candidate_pregrasp_ik.tolist()
            pregrasp_position_error, pregrasp_rotation_error = (
                _pose_tracking_error_components(live_eef_transform, pregrasp_pose)
            )
            near_pregrasp_start = bool(
                approach_reachability.get("grasp_open_gripper", {}).get("reachable") is True
                and pregrasp_position_error <= NEAR_PREGRASP_START_POSITION_M
                and pregrasp_rotation_error <= NEAR_PREGRASP_START_ROTATION_RAD
            )
            near_surface_start = bool(
                near_position_error <= NEAR_SURFACE_START_POSITION_M
                and near_rotation_error <= NEAR_SURFACE_START_ROTATION_RAD
            )
            approach_route_strategy = str(
                getattr(args, "approach_route_strategy", "auto")
            )
            explicit_geometry_high_approach = approach_route_strategy in {
                "geometry_high_approach", "geometry_direct_approach",
            }
            if explicit_geometry_high_approach:
                # This mode is an explicit physical experiment: the candidate
                # pose came from active-scene geometry, and OSC will test its
                # reachability while tracking. Do not let a broad "near enough"
                # shortcut or a one-finger close probe bypass the requested
                # direct or lifted OSC route.
                rrt = {
                    "status": "offline_feasible",
                    "planner_source": (
                        "explicit_active_geometry_direct_approach.v1"
                        if approach_route_strategy == "geometry_direct_approach"
                        else "explicit_active_geometry_high_approach.v1"
                    ),
                    "planning_seconds": 0.0,
                    "config_checks": 0,
                    "iterations": 0,
                    "selected_route_index": (
                        "explicit_geometry_direct_approach"
                        if approach_route_strategy == "geometry_direct_approach"
                        else "explicit_geometry_high_approach"
                    ),
                    "minimum_forbidden_clearance_m": None,
                    "live_state_unchanged": True,
                    "eef_reference_poses_world": [],
                    "route_candidates": [],
                    "explicit_route_strategy": {
                        "name": approach_route_strategy,
                        "RRT_skipped_by_explicit_experiment_design": True,
                        "pregrasp_endpoint_ik_verified_by_task_approach_screen": (
                            pregrasp_endpoint_ik_verified
                        ),
                        "OSC_physical_reachability_test_required": True,
                        "physical_state_modified": False,
                    },
                }
            elif near_surface_start:
                # A legal snapshot can already be at the grasp surface after a
                # prior physical branch. Returning to pregrasp would waste a
                # full RRT and duplicate the expensive prefix. Continue with a
                # real MuJoCo OSC correction from the exact live pose instead.
                rrt = {
                    "status": "offline_feasible",
                    "planner_source": "near_target_direct_surface_approach.v1",
                    "planning_seconds": 0.0,
                    "config_checks": 0,
                    "iterations": 0,
                    "selected_route_index": "near_surface_start",
                    "minimum_forbidden_clearance_m": None,
                    "live_state_unchanged": True,
                    "eef_reference_poses_world": [live_eef_pose.tolist(), live_eef_pose.tolist()],
                    "route_candidates": [{
                        "route_index": "near_surface_start",
                        "route_source": "near_target_direct_surface_approach",
                        "metrics": {
                            "direct_surface_approach_start": True,
                            "eef_to_grasp_position_error_m": near_position_error,
                            "eef_to_grasp_rotation_error_rad": near_rotation_error,
                            "RRT_skipped_to_avoid_repeating_pregrasp_prefix": True,
                        },
                        "eef_reference_poses_world": [
                            live_eef_pose.tolist(), live_eef_pose.tolist(),
                        ],
                    }],
                    "near_surface_start_diagnostic": {
                        "status": "pass",
                        "position_error_m": near_position_error,
                        "rotation_error_rad": near_rotation_error,
                        "position_threshold_m": NEAR_SURFACE_START_POSITION_M,
                        "rotation_threshold_rad": NEAR_SURFACE_START_ROTATION_RAD,
                        "physical_state_modified": False,
                    },
                }
            elif near_pregrasp_start:
                # A restored failure snapshot can already be within a short,
                # IK-verified final approach to pregrasp. Replan-free OSC from
                # that exact live pose avoids repeating a long elevated prefix.
                pregrasp_pose_7d = np.r_[
                    pregrasp_pose[:3, 3], _wxyz(pregrasp_pose[:3, :3]),
                ]
                rrt = {
                    "status": "offline_feasible",
                    "planner_source": "near_target_direct_pregrasp_approach.v1",
                    "planning_seconds": 0.0,
                    "config_checks": 0,
                    "iterations": 0,
                    "selected_route_index": "near_pregrasp_start",
                    "minimum_forbidden_clearance_m": None,
                    "live_state_unchanged": True,
                    "eef_reference_poses_world": [
                        live_eef_pose.tolist(), pregrasp_pose_7d.tolist(),
                    ],
                    "route_candidates": [{
                        "route_index": "near_pregrasp_start",
                        "route_source": "near_target_direct_pregrasp_approach",
                        "metrics": {
                            "direct_pregrasp_approach_start": True,
                            "eef_to_pregrasp_position_error_m": pregrasp_position_error,
                            "eef_to_pregrasp_rotation_error_rad": pregrasp_rotation_error,
                            "grasp_endpoint_IK_verified_by_task_approach_screen": (
                                approach_reachability.get("grasp_open_gripper", {}).get("reachable") is True
                            ),
                            "pregrasp_endpoint_IK_verified_by_task_approach_screen": pregrasp_endpoint_ik_verified,
                            "admitted_by_live_pregrasp_pose_and_physical_OSC_validation": True,
                            "RRT_skipped_to_avoid_repeating_elevated_prefix": True,
                        },
                        "eef_reference_poses_world": [
                            live_eef_pose.tolist(), pregrasp_pose_7d.tolist(),
                        ],
                    }],
                    "near_pregrasp_start_diagnostic": {
                        "status": "pass",
                        "position_error_m": pregrasp_position_error,
                        "rotation_error_rad": pregrasp_rotation_error,
                        "position_threshold_m": NEAR_PREGRASP_START_POSITION_M,
                        "rotation_threshold_rad": NEAR_PREGRASP_START_ROTATION_RAD,
                        "grasp_endpoint_IK_verified_by_task_approach_screen": (
                            approach_reachability.get("grasp_open_gripper", {}).get("reachable") is True
                        ),
                        "pregrasp_endpoint_IK_verified_by_task_approach_screen": pregrasp_endpoint_ik_verified,
                        "admitted_by_live_pregrasp_pose_and_physical_OSC_validation": True,
                        "physical_state_modified": False,
                    },
                }
            else:
                rrt = bridge.request({
                    "op": "task_success_first_rrt",
                    "expected_instruction": instruction,
                    "model_geometry_sha256": model_sha,
                    "start_state_sha256": state_sha,
                    "goal_pose_world": {
                        "position_world_m": pregrasp.tolist(),
                        "quaternion_wxyz": _wxyz(target_pose[:3, :3]).tolist(),
                    },
                    "ik_seed_qpos": (
                        [np.asarray(
                            approach_reachability["pregrasp"]["qpos"],
                            dtype=np.float64,
                        ).tolist()]
                        if pregrasp_endpoint_ik_verified else None
                    ),
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
                        # A few microradians of OSC integration noise may place
                        # the exact live start just outside a hard stop. The
                        # planner projects only its private seed onto the limit;
                        # live qpos stays untouched and IK goals remain strict.
                        "start_joint_limit_tolerance_rad": 5e-6,
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
                "rrt_ik_seed_source": (
                    "task_approach_screen_pregrasp_endpoint_ik"
                    if (
                        pregrasp_endpoint_ik_verified
                        and not explicit_geometry_high_approach
                        and not near_surface_start
                        and not near_pregrasp_start
                    ) else None
                ),
                "rrt_ik_selection_seed_source": (
                    "planner_start_qpos_clipped_to_joint_limits"
                    if (
                        pregrasp_endpoint_ik_verified
                        and not explicit_geometry_high_approach
                        and not near_surface_start
                        and not near_pregrasp_start
                    ) else None
                ),
                "rrt_safety_certified": False,
                "pregrasp_pose_world": np.r_[pregrasp, _wxyz(target_pose[:3, :3])].tolist(),
                "near_surface_start_bypass": rrt.get("near_surface_start_diagnostic"),
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
                not explicit_geometry_high_approach
                and (
                    not routes
                    or len(live_contact_sides) == 2
                    or str(candidate["candidate_id"])
                    in unilateral_contact_pose_candidates
                )
                and live_contact_sides
                and (
                    live_pose_aligned
                    or str(candidate["candidate_id"])
                    in unilateral_contact_pose_candidates
                )
            ):
                # A measured unilateral target-pad contact near an orientation-
                # aligned generated grasp frame also merits a bounded physical
                # close probe.  This avoids replanning away from useful contact
                # on low objects while retaining the live grasp monitor as the
                # only success oracle.
                routes = [{
                    "route_index": "live_contact_pose_aligned_close_probe",
                    "eef_reference_poses_world": [start_pose.tolist(), start_pose.tolist()],
                    "route_source": "live_target_contact_pose_aligned_close_probe",
                    "live_contact_sides": list(live_contact_sides),
                }]
                summary["live_contact_close_probe_admitted_from_measured_target_contact"] = {
                    "finger_sides": list(live_contact_sides),
                    "eef_to_migrated_grasp_pose_m": live_grasp_distance_m,
                    "eef_to_migrated_grasp_rotation_error_rad": live_grasp_rotation_error_rad,
                    "configured_unilateral_candidate_position_threshold_m": (
                        args.stage_a_unilateral_close_probe_position_m
                    ),
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
                explicit_geometry_route_available = bool(
                    explicit_geometry_high_approach
                    and summary.get("rrt_live_state_unchanged") is True
                )
                generated_geometry_route_available = (
                    explicit_geometry_route_available
                    or _generated_geometry_fallback_allowed(candidate, planner_response)
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
                    if explicit_geometry_route_available:
                        route_source = "explicit_active_BDDL_geometry_high_OSC_approach"
                    elif generated_geometry_route_available:
                        route_source = "active_BDDL_geometry_high_OSC_approach_after_RRT_failure"
                    else:
                        route_source = (
                            "contact_normal_escape_then_elevated_object_relative_approach"
                            if escape_pose is not None
                            else "object_relative_experience_high_OSC_approach_after_RRT_preflight_failure"
                        )
                    if explicit_geometry_route_available:
                        # In an explicitly requested Stage A physical search,
                        # intermediate scratch-IK failure is not evidence that
                        # OSC cannot track the route.  The previous path
                        # rejected every target before taking an action when
                        # one of its conservative transit waypoints had no
                        # bounded IK solution.  Keep the route grounded in the
                        # active-scene grasp geometry, then let real MuJoCo
                        # execution measure tracking and contact outcomes.
                        # Automatic fallback mode retains its adaptive IK
                        # preflight below.
                        if approach_route_strategy == "geometry_direct_approach":
                            # A fixed elevated transit can itself be outside
                            # the live OSC workspace even when both endpoints
                            # have IK.  From an exact Snapshot, test the
                            # direct Cartesian segment with bounded OSC steps;
                            # contact and BDDL remain the physical oracles.
                            route_poses = [route_start.tolist(), pregrasp_pose_7d.tolist()]
                            route_index = "explicit_geometry_direct_approach_physical_validation"
                            route_source = "explicit_active_BDDL_geometry_direct_OSC_approach"
                        else:
                            route_poses = high_approach_route(
                                route_start,
                                pregrasp_pose_7d,
                                elevation_m=0.10,
                                lift_before_translate=True,
                            )
                            route_index = "explicit_geometry_high_approach_physical_validation"
                            route_source = "explicit_active_BDDL_geometry_high_OSC_approach_without_intermediate_IK_veto"
                        summary["adaptive_geometry_route_preflight"] = {
                            "status": "bypassed_for_explicit_physical_validation",
                            "reason": "read_only_intermediate_IK_preflight_would_prevent_any_MuJoCo_attempt",
                            "route_waypoint_count": len(route_poses),
                            "physical_OSC_tracking_is_required": True,
                            "live_state_modified_during_preflight": False,
                            "safety_certified": False,
                        }
                    elif generated_geometry_route_available:
                        candidate_goal_ik = candidate.get("_pregrasp_ik_seed_qpos")
                        if candidate_goal_ik is None:
                            summary["adaptive_geometry_route_preflight"] = {
                                "status": "missing_candidate_endpoint_ik_seed",
                                "physical_actions": 0,
                            }
                            continue
                        adaptive_route, adaptive_report = _adaptive_geometry_approach_route(
                            bridge,
                            start_pose=route_start,
                            target_pose=pregrasp_pose_7d,
                            forbidden_obstacle_positions_world=route_obstacle_positions,
                            minimum_transit_height_m=(
                                float(initial_contact_escape["contact_clearance_waypoint_z_m"])
                                if initial_contact_escape is not None
                                and initial_contact_escape.get("contact_clearance_waypoint_z_m") is not None
                                else None
                            ),
                            approach_direction_world=(
                                target_pose[:3, 3] - pregrasp_pose[:3, 3]
                            ) / max(
                                float(np.linalg.norm(
                                    target_pose[:3, 3] - pregrasp_pose[:3, 3]
                                )),
                                1e-12,
                            ),
                            live_arm_qpos=np.asarray(context["arm_qpos"], dtype=np.float64),
                            instruction=instruction,
                            model_sha=model_sha,
                            state_sha=state_sha,
                            policy=policy,
                        )
                        summary["adaptive_geometry_route_preflight"] = adaptive_report
                        if adaptive_route is None:
                            continue
                        route_poses = adaptive_route
                        route_index = "adaptive_geometry_ik_waypoints"
                        route_source = adaptive_report["route_source"]
                    else:
                        route_poses = high_approach_route(
                            route_start,
                            pregrasp_pose_7d,
                            elevation_m=0.18 if escape_pose is not None else 0.10,
                            lift_before_translate=True,
                        )
                        route_index = "historical_high_approach"
                    routes = [{
                        "route_index": route_index,
                        "eef_reference_poses_world": route_poses,
                        "route_source": route_source,
                        "preserve_eef_waypoints": generated_geometry_route_available,
                        "contact_escape_prepended": escape_pose is not None,
                        "defer_live_contact_close_until_route_end": explicit_geometry_route_available,
                    }]
                    if escape_pose is not None:
                        routes[0]["eef_reference_poses_world"] = [
                            start_pose.tolist(), escape_pose.tolist(),
                            *routes[0]["eef_reference_poses_world"][1:],
                        ]
                    summary["historical_direct_OSC_fallback_added"] = experience_route_available
                    summary["generated_geometry_high_approach_fallback_added"] = generated_geometry_route_available
                    summary["explicit_geometry_high_approach_selected"] = bool(
                        explicit_geometry_route_available
                        and approach_route_strategy == "geometry_high_approach"
                    )
                    summary["explicit_geometry_physical_approach_strategy"] = (
                        approach_route_strategy
                        if explicit_geometry_route_available else None
                    )
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
                if route.get("route_source") == "near_target_direct_surface_approach":
                    signature = f"near_target_direct_surface_approach:{candidate['candidate_id']}"
                elif route.get("live_contact_sides"):
                    signature = (
                        f"{route.get('route_source', 'live_target_contact_close_probe')}_"
                        + "_".join(route.get("live_contact_sides", []))
                    )
                else:
                    signature = route_signature(poses)
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
            planned_route["preserve_eef_waypoints"] = bool(
                route.get("preserve_eef_waypoints", False)
            )
            planned_route["defer_live_contact_close_until_route_end"] = bool(
                route.get("defer_live_contact_close_until_route_end", False)
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
                    local_replan_rrt_time_budget_s=args.local_replan_rrt_time_budget_s,
                    abort_target_displacement_m=(
                        0.015
                        if cold_start and not bool(rrt.get("defer_live_contact_close_until_route_end"))
                        else None
                    ),
                    force_route_before_contact_close=bool(
                        rrt.get("defer_live_contact_close_until_route_end")
                    ),
                    stage_a_unilateral_close_probe_position_m=(
                        args.stage_a_unilateral_close_probe_position_m
                    ),
                    stage_a_pregrasp_staged_clearance_height_m=(
                        args.stage_a_pregrasp_staged_clearance_height_m
                    ),
                    stage_a_pregrasp_lateral_detour_m=(
                        args.stage_a_pregrasp_lateral_detour_m
                    ),
                    stage_a_pregrasp_lift_before_lateral_detour=(
                        args.stage_a_pregrasp_lift_before_lateral_detour
                    ),
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
                # A grasp-search branch can complete the whole BDDL task by
                # physically pushing an object into its goal. Preserve that
                # terminal Episode instead of misclassifying MuJoCo's done
                # signal as a failed grasp-only branch.
                if (
                    last_record.get("done") is True
                    and last_record.get("task_success") is True
                ):
                    native_terminal = bridge.request({"op": "native_task_status"})
                    if native_terminal.get("task_success") is True:
                        terminal = bridge.request({"op": "export_runtime_snapshot"})
                        if terminal.get("status") != "complete" or not isinstance(
                            terminal.get("payload"), bytes
                        ):
                            raise RuntimeError(
                                "native_task_success_terminal_snapshot_export_failed"
                            )
                        terminal_slug = "".join(
                            char if char.isalnum() or char in "._-" else "_"
                            for char in str(candidate.get("candidate_id", "candidate"))
                        )[:120]
                        terminal_path = output_dir / (
                            f"{terminal_slug}__native_task_success_terminal_runtime_snapshot.pkl"
                        )
                        terminal_path.write_bytes(terminal["payload"])
                        terminal_state = _snapshot_state_vector(
                            terminal.get("snapshot_state", {})
                        )
                        terminal_snapshot = {
                            "status": "saved_native_task_success_episode_terminal",
                            "path": str(terminal_path),
                            "payload_sha256": _sha256_bytes(terminal["payload"]),
                            "integration_state_sha256": _state_sha(terminal_state),
                            "model_geometry_sha256": model_sha,
                            "timestep": native_terminal.get("timestep"),
                            "task_success_candidate": False,
                            "safety_certified": False,
                        }
                        branch.update({
                            "status": "native_task_success_terminated_episode",
                            "failure_reason": None,
                            "task_success_candidate": False,
                            "episode_native_task_success": True,
                            "native_task_status": native_terminal,
                            "stable_bilateral_grasp_observed": False,
                            "final_snapshot": terminal_snapshot,
                            "terminal_action_record_index": len(records) - 1,
                        })
                        report["episode_native_success"] = {
                            "status": "native_task_success_terminated_episode",
                            "native_task_status": native_terminal,
                            "terminal_action": {
                                "step_index": last_record.get("step_index"),
                                "timestep": last_record.get("timestep"),
                                "reward": last_record.get("reward"),
                                "done": last_record.get("done"),
                                "task_success": last_record.get("task_success"),
                                "candidate_id": last_record.get("candidate_id"),
                            },
                            "stable_bilateral_grasp_observed": False,
                            "snapshot": terminal_snapshot,
                            "episode_action_count_from_legal_t0": native_terminal.get("timestep"),
                            "safety_certified": False,
                        }
                        final_snapshot_payload = terminal["payload"]
                        report["status"] = "native_task_success_terminated_episode"
                        break
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
        if report.get("success") is None and report.get("episode_native_success") is None:
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
        # Keep the cold-recovery verifier off the expert search port. Some
        # environments retain the just-closed listener briefly; a separate
        # port avoids colliding with that socket during immediate rebind.
        recovery_address = (address[0], int(address[1]) + 2)
        recovery = BridgeSession(
            recovery_address, output_dir / "snapshot_recovery_bridge.log", args,
        )
        try:
            recovery.start()
            recovery_reset = recovery.request({"op": "reset"})
            if recovery_reset.get("scene_metadata", {}).get("model_geometry_sha256") != model_sha:
                raise RuntimeError("fresh_recovery_bridge_model_fingerprint_mismatch")
            restored = recovery.request({"op": "restore_runtime_snapshot", "payload": final_snapshot_payload})
            episode_native_success = report.get("episode_native_success")
            if restored.get("status") != "restored" or restored.get("model_geometry_sha256") != model_sha:
                raise RuntimeError("fresh_bridge_could_not_restore_terminal_snapshot")
            if episode_native_success is not None:
                recovered_state_sha = _state_sha(
                    _snapshot_state_vector(restored.get("snapshot_state", {}))
                )
                if recovered_state_sha != episode_native_success["snapshot"].get(
                    "integration_state_sha256"
                ):
                    raise RuntimeError("fresh_bridge_snapshot_integration_state_mismatch")
                recovered_native = recovery.request({"op": "native_task_status"})
                if (
                    recovered_native.get("task_success") is not True
                    or recovered_native.get("timestep")
                    != episode_native_success["native_task_status"].get("timestep")
                ):
                    raise RuntimeError("restored_terminal_snapshot_native_success_mismatch")
                episode_native_success["snapshot"]["integration_state_exact_match"] = True
                episode_native_success["snapshot_recovery_native_task_status"] = recovered_native
                report["snapshot_recovery_verification"] = {
                    "status": "pass",
                    "port": recovery_address[1],
                    "model_geometry_sha256": model_sha,
                    "integration_state_exact_match": True,
                    "native_task_status": recovered_native,
                }
            else:
                expected = report["success"]["snapshot"].get("snapshot_state")
                if expected is None or not np.array_equal(
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
                    "port": recovery_address[1],
                    "model_geometry_sha256": model_sha,
                    "integration_state_exact_match": True,
                    "restored_grasp_state": grasp,
                }
        except Exception as exc:
            report["snapshot_recovery_verification"] = {
                "status": "fail", "port": recovery_address[1],
                "error": f"{type(exc).__name__}: {exc}"
            }
            report["status"] = "task_success_candidate_snapshot_recovery_failed"
        finally:
            recovery.close()
        if report.get("episode_native_success") is not None:
            report["artifacts"]["native_task_success_terminal_snapshot"] = (
                report["episode_native_success"]["snapshot"]["path"]
            )
        else:
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
        "--cold-start-continuation", action="store_true",
        help="continue a cold-start Episode from its own V3-generated, identity-checked stage Snapshot",
    )
    parser.add_argument(
        "--candidate-id", action="append", default=[],
        help="limit physical search to one or more deterministic generated candidate IDs",
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
    parser.add_argument(
        "--geometry-candidate-pool-offset", type=int, default=0,
        help="select a later window from the deterministic active-geometry candidate pool",
    )
    parser.add_argument("--maximum-rrt-preflights", type=int, default=MAX_RRT_PREFLIGHTS)
    parser.add_argument("--local-replan-rrt-time-budget-s", type=float, default=2.0)
    parser.add_argument("--maximum-physical-branches", type=int, default=MAX_PHYSICAL_BRANCHES)
    parser.add_argument(
        "--branch-control-action-budget", type=int, default=MAX_BRANCH_CONTROL_ACTIONS,
        help=(
            "maximum real MuJoCo control actions for each physical grasp branch; "
            "use a short value for targeted contact/route validation"
        ),
    )
    parser.add_argument(
        "--stage-a-unilateral-close-probe-position-m", type=float, default=LIVE_UNILATERAL_CLOSE_PROBE_POSITION_M,
        help=(
            "maximum candidate-pose error for a bounded physical close probe after a real unilateral target-pad contact; "
            "the live MuJoCo grasp monitor must still verify closure"
        ),
    )
    parser.add_argument(
        "--stage-a-pregrasp-staged-clearance-height-m", type=float, default=0.0,
        help=(
            "optional absolute world-Z clearance for the physical pregrasp route: "
            "lift with the live wrist orientation, translate above the candidate, "
            "rotate, then descend; 0 disables it"
        ),
    )
    parser.add_argument(
        "--stage-a-pregrasp-lateral-detour-m", type=float, default=0.0,
        help=(
            "at staged clearance, offset the transit corridor to one side of the "
            "direct wrist route, then return beyond the obstacle; signed meters"
        ),
    )
    parser.add_argument(
        "--stage-a-pregrasp-lift-before-lateral-detour", action="store_true",
        help=(
            "when a staged lateral detour is enabled, first lift vertically at the "
            "live wrist pose, then sidestep at clearance height"
        ),
    )
    parser.add_argument("--rrt-time-budget-s", type=float, default=12.0)
    parser.add_argument(
        "--approach-route-strategy",
        choices=("auto", "geometry_high_approach", "geometry_direct_approach"),
        default="auto",
        help=(
            "use normal scratch RRT routing, or explicitly select a generated-geometry high-lift/direct "
            "OSC route after endpoint IK verification"
        ),
    )
    parser.add_argument(
        "--contact-only-substep-audit", action="store_true",
        help="retain all per-substep contacts and skip exhaustive per-substep all-Geom clearance sweeps",
    )
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
    if (
        args.geometry_candidate_pool_offset < 0
        or args.geometry_candidate_pool_offset + args.maximum_candidates > MAX_OFFLINE_GEOMETRY_GENERATION
    ):
        parser.error(
            "--geometry-candidate-pool-offset plus --maximum-candidates must fit the generated geometry pool"
        )
    if args.strict_grasp_screen_only and args.geometry_candidate_pool_offset != 0:
        parser.error("--geometry-candidate-pool-offset is available only for physical task-success search")
    if not 1 <= args.maximum_rrt_preflights <= MAX_RRT_PREFLIGHTS:
        parser.error(f"--maximum-rrt-preflights must be in 1..{MAX_RRT_PREFLIGHTS}")
    if not np.isfinite(args.stage_a_unilateral_close_probe_position_m) or not (
        0.010 <= args.stage_a_unilateral_close_probe_position_m <= 0.080
    ):
        parser.error("--stage-a-unilateral-close-probe-position-m must be in 0.010..0.080")
    if (
        not np.isfinite(args.stage_a_pregrasp_staged_clearance_height_m)
        or args.stage_a_pregrasp_staged_clearance_height_m != 0.0
            and not 0.20 <= args.stage_a_pregrasp_staged_clearance_height_m <= 2.50
    ):
        parser.error(
            "--stage-a-pregrasp-staged-clearance-height-m must be 0 or in 0.20..2.50"
        )
    if (
        not np.isfinite(args.stage_a_pregrasp_lateral_detour_m)
        or not -0.40 <= args.stage_a_pregrasp_lateral_detour_m <= 0.40
    ):
        parser.error("--stage-a-pregrasp-lateral-detour-m must be in -0.40..0.40")
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
        if not offset_families:
            parser.error("--candidate-coupled-perturbations requires at least one object-axis offset family")
    elif orientation_family_count + len(offset_families) > 1:
        parser.error("combine orientation and multiple contact-offset families only with --candidate-coupled-perturbations")
    if (
        args.candidate_tilt_degrees or args.candidate_roll_degrees
        or offset_families
    ):
        if not args.candidate_id:
            parser.error("candidate geometry perturbations require one or more generated --candidate-id base poses")
        orientation_count = len(args.candidate_tilt_degrees or args.candidate_roll_degrees)
        offset_count = 1
        for values in offset_families.values():
            offset_count *= len(values)
        perturbation_count = (orientation_count or 1) * offset_count if args.candidate_coupled_perturbations else (orientation_count or offset_count)
        if len(args.candidate_id) * perturbation_count > MAX_OFFLINE_GEOMETRY_SCREEN_BATCH:
            parser.error("selected base candidates and perturbations must fit one task-approach screen batch of at most 24")
    if args.historical_grasp_pose_priors_json is not None:
        if not args.strict_grasp_screen_only:
            parser.error("--historical-grasp-pose-priors-json requires --strict-grasp-screen-only")
        if args.candidate_id or args.candidate_start_index != 0:
            parser.error("historical pose priors cannot be combined with candidate ID/window filters")
        if perturbation_flags:
            parser.error("historical pose priors cannot be combined with orientation/contact perturbations")
    if not 1 <= args.maximum_physical_branches <= MAX_PHYSICAL_BRANCHES:
        parser.error(f"--maximum-physical-branches must be in 1..{MAX_PHYSICAL_BRANCHES}")
    if not 1 <= args.branch_control_action_budget <= MAX_BRANCH_CONTROL_ACTIONS:
        parser.error(
            f"--branch-control-action-budget must be in 1..{MAX_BRANCH_CONTROL_ACTIONS}"
        )
    if not 0.5 <= args.rrt_time_budget_s <= 60.0:
        parser.error("--rrt-time-budget-s must be in 0.5..60")
    if not 0.5 <= args.local_replan_rrt_time_budget_s <= 60.0:
        parser.error("--local-replan-rrt-time-budget-s must be in 0.5..60")
    if args.strict_grasp_screen_only and args.strict_screen_reference_report is None:
        parser.error("--strict-grasp-screen-only requires --strict-screen-reference-report")
    if args.strict_screen_reference_report is not None and not args.strict_grasp_screen_only:
        parser.error("--strict-screen-reference-report requires --strict-grasp-screen-only")
    cold_start_mode = args.cold_start or args.cold_start_continuation
    if args.cold_start_continuation and (
        not args.cold_start or args.resume_from_runtime_snapshot is None
    ):
        parser.error("--cold-start-continuation requires --cold-start and --resume-from-runtime-snapshot")
    if cold_start_mode and (
        args.experience_library is not None
        or args.previous_search_report
        or args.historical_grasp_pose_priors_json is not None
        or args.strict_grasp_screen_only
        or args.strict_screen_reference_report is not None
        or (args.resume_from_runtime_snapshot is not None and not args.cold_start_continuation)
        or args.candidate_start_index != 0
        or (
            not args.cold_start_continuation
            and (
                args.candidate_tilt_degrees
                or args.candidate_roll_degrees
            )
        )
    ):
        # Object-local contact offsets are generated from the active BDDL
        # collision geometry. They are valid fresh-t0 search candidates and do
        # not import historical poses, actions, or partial-task state.
        parser.error("--cold-start cannot use historical experience or partial search inputs")
    if not 1000 <= args.demonstration_horizon <= MAX_DEMONSTRATION_HORIZON:
        parser.error(
            f"--demonstration-horizon must be in 1000..{MAX_DEMONSTRATION_HORIZON}"
        )
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
