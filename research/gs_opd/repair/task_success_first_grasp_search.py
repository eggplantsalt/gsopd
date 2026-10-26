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
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from research.gs_opd.repair.task_success_first_search import (
    SearchFailureMemory,
    blocked_candidate_ids_from_report,
    candidate_route_preflight_admitted,
    candidate_matches_failure_memory,
    diversify_candidates,
    high_approach_route,
    object_relative_grasp_targets,
    planner_seed,
    route_control_action_budget,
    route_signature,
    robot_contact_escape_direction,
    unilateral_target_contact_sides,
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
MAX_RRT_PREFLIGHTS = 8
MAX_PHYSICAL_BRANCHES = 3
MAX_BRANCH_CONTROL_ACTIONS = 440
MAX_PREFIX_CONTROL_ACTIONS = 320
MAX_APPROACH_CONTROL_ACTIONS = 60
MAX_CLOSE_CONTROL_ACTIONS = 18
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
    if isinstance(maximum_candidates, bool) or not 1 <= maximum_candidates <= MAX_GEOMETRY_CANDIDATES:
        raise ValueError(f"maximum_candidates_must_be_in_1..{MAX_GEOMETRY_CANDIDATES}")
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


class BridgeSession:
    _READ_ONLY_RECONNECT_OPS = frozenset({
        "observe", "native_task_status", "native_goal_query", "inspect_contacts",
        "grasp_state", "geometry_inventory", "gripper_geometry_inventory",
        "task_semantic_geometry_bindings", "repair_context",
        "task_success_first_context", "task_success_first_articulation_context",
        "robot_joint_limit_inventory", "compiled_model_fingerprint_details",
        "trajectory_tracking_capabilities",
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
    substeps = contact_monitoring.get("geometry_monitoring", {}).get("substeps", [])
    expected = contact_monitoring.get("expected_physics_substeps_per_control_step")
    captured = contact_monitoring.get("captured_physics_substeps")
    monitoring_complete = not bool(
        contact_monitoring.get("status") != "complete"
        or expected != EXPECTED_MUJOCO_SUBSTEPS_PER_ACTION
        or captured != EXPECTED_MUJOCO_SUBSTEPS_PER_ACTION
        or len(substeps) != EXPECTED_MUJOCO_SUBSTEPS_PER_ACTION
        or any(item.get("status") != "complete" for item in substeps)
    )
    snapshot_state = reply.get("snapshot_state", {})
    records.append({
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
    })
    frames.append({
        "candidate_id": candidate_id,
        "step_index": len(records) - 1,
        "phase": phase,
        "agentview_rgb": _image(reply, "agentview_image"),
        "wrist_rgb": _image(reply, "robot0_eye_in_hand_image"),
    })
    if reply.get("done") and raise_on_done:
        raise RuntimeError("environment_done_during_grasp_only_branch")


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
        )
        action_counter[0] += 1
        if step_callback is not None:
            step_callback(records[-1], frames[-1])
        if records[-1].get("done"):
            raise RuntimeError("environment_done_during_grasp_only_branch")
        if action_counter[0] > MAX_BRANCH_CONTROL_ACTIONS:
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


def _compress_eef_route(route: Any) -> np.ndarray:
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
    return poses[keep]


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
    initial_grasp_distance_m = float(np.linalg.norm(
        initial_eef_world[:3, 3] - initial_grasp_world[:3, 3]
    ))
    unilateral_sides = unilateral_target_contact_sides(initial_grasp_state)
    close_from_live_contact = bool(
        unilateral_sides
        and initial_grasp_distance_m <= max(
            0.12,
            float(candidate.get("approach_distance_m", 0.0))
            + float(candidate.get("pregrasp_eef_distance_m", 0.0))
            + 0.035,
        )
    )
    branch["initial_target_finger_contact_sides"] = list(unilateral_sides)
    branch["initial_eef_to_migrated_grasp_pose_m"] = initial_grasp_distance_m
    branch["pregrasp_route_skipped_due_to_live_target_contact"] = close_from_live_contact
    if close_from_live_contact:
        branch["live_contact_close_probe_action_budget"] = MAX_CLOSE_CONTROL_ACTIONS
        live_close_actions = 0
        while live_close_actions < MAX_CLOSE_CONTROL_ACTIONS and not monitor.stable:
            _record_step(
                bridge,
                np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
                "task_success_first_live_unilateral_contact_close_probe",
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
            raise RuntimeError("unilateral_live_contact_close_probe_did_not_stabilize")
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
    ) -> None:
        try:
            _advance_pose(
                bridge, target_position, target_quaternion, phase, candidate_id,
                action_counter, maximum_steps, target_binding, context["instruction"],
                policy, monitor, records, frames,
            )
            return
        except TimeoutError as original_error:
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
    approach_world = root_pose[:3, :3] @ np.asarray(
        candidate["approach_direction_object"], dtype=np.float64
    )
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
        )
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_started = time.monotonic()
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor

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
            "cross_scene_experience_library_path": (
                str(args.experience_library) if args.experience_library else None
            ),
        },
        "search_budget": {
            "geometry_candidates_max": args.maximum_candidates,
            "candidate_start_index": int(args.candidate_start_index),
            "RRT_preflights_max": args.maximum_rrt_preflights,
            "physical_snapshot_branches_max": args.maximum_physical_branches,
            "branch_control_actions_max": MAX_BRANCH_CONTROL_ACTIONS,
            "pregrasp_route_actions_max": MAX_PREFIX_CONTROL_ACTIONS,
            "approach_actions_max": MAX_APPROACH_CONTROL_ACTIONS,
            "close_actions_max": MAX_CLOSE_CONTROL_ACTIONS,
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
        policy = _build_safety_policy(context, semantic, geometry, gripper)
        initial_safety = bridge.request({"op": "safety_geometry_check", "safety_geometry_policy": policy})
        initial_contact_escape = robot_contact_escape_direction(
            initial_safety.get("contact_identities", []),
            ignore_object_id=str(context["target_object_id"]),
        )
        report["initial_robot_contact_escape"] = initial_contact_escape
        candidates, generation = generate_geometry_candidates(
            context, maximum_candidates=args.maximum_candidates
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
                    if len(experience_candidates) >= args.maximum_candidates:
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
                        capacity = args.maximum_candidates - len(experience_candidates)
                        experience_candidates.extend(alternates[:capacity])
                        alternate_count += min(len(alternates), capacity)
                candidates = [
                    *experience_candidates,
                    *candidates[:max(0, args.maximum_candidates - len(experience_candidates))],
                ]
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
        report["experience_transfer"] = experience_use
        report["imported_search_failure_memory"] = imported_failure_memory
        candidates = diversify_candidates(candidates)
        if args.candidate_id is not None:
            candidates = [
                candidate for candidate in candidates
                if candidate.get("candidate_id") == args.candidate_id
            ]
            if not candidates:
                raise ValueError(f"requested_candidate_id_not_generated:{args.candidate_id}")
        if not candidates:
            raise RuntimeError("no_supported_collision_geometry_produced_a_pad_grasp_candidate")
        _write_json(output_dir / "geometry_candidates.json", candidates)
        _write_json(output_dir / "experience_transfer.json", experience_use)
        _write_json(output_dir / "active_target_context.json", context)
        _write_json(output_dir / "initial_safety_diagnostics.json", initial_safety)
        _write_json(output_dir / "safety_policy_diagnostic_only.json", policy)
        _advance_pose.controller_fields = controller["robots"][0]["fields"]

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
            live_contact_sides = unilateral_target_contact_sides(live_grasp_state)
            live_grasp_distance_m = float(np.linalg.norm(start_pose[:3] - target_pose[:3, 3]))
            live_close_distance_limit_m = max(
                0.12,
                float(candidate.get("approach_distance_m", 0.0))
                + float(candidate.get("pregrasp_eef_distance_m", 0.0))
                + 0.035,
            )
            if (
                not routes
                and live_contact_sides
                and live_grasp_distance_m <= live_close_distance_limit_m
            ):
                # A live target contact is itself a physically validated local
                # starting point. Preserve it and try closing in place even if
                # the migrated EEF target has no IK solution from this state.
                routes = [{
                    "route_index": "live_contact_close_probe",
                    "eef_reference_poses_world": [start_pose.tolist(), start_pose.tolist()],
                    "route_source": "live_unilateral_target_contact_close_probe",
                    "live_contact_sides": list(live_contact_sides),
                }]
                summary["live_contact_close_probe_admitted_after_RRT_failure"] = {
                    "finger_sides": list(live_contact_sides),
                    "eef_to_migrated_grasp_pose_m": live_grasp_distance_m,
                    "maximum_close_probe_distance_m": live_close_distance_limit_m,
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
                if experience_route_available:
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
                    routes = [{
                        "route_index": "historical_high_approach",
                        "eef_reference_poses_world": high_approach_route(
                            route_start,
                            pregrasp_pose_7d,
                            elevation_m=0.18 if escape_pose is not None else 0.10,
                        ),
                        "route_source": (
                            "contact_normal_escape_then_elevated_object_relative_approach"
                            if escape_pose is not None
                            else "object_relative_experience_high_OSC_approach_after_RRT_preflight_failure"
                        ),
                        "contact_escape_prepended": escape_pose is not None,
                    }]
                    if escape_pose is not None:
                        routes[0]["eef_reference_poses_world"] = [
                            start_pose.tolist(), escape_pose.tolist(),
                            *routes[0]["eef_reference_poses_world"][1:],
                        ]
                    summary["historical_direct_OSC_fallback_added"] = True
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
                    "live_contact_close_probe_"
                    + "_".join(route.get("live_contact_sides", []))
                    if route.get("live_contact_sides")
                    else route_signature(poses)
                )
                route_id = f"rrt{route.get('route_index', 0)}:{signature}"
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
        "--candidate-id", default=None,
        help="limit a local retry to one previously generated object-relative candidate",
    )
    parser.add_argument("--maximum-candidates", type=int, default=MAX_GEOMETRY_CANDIDATES)
    parser.add_argument("--candidate-start-index", type=int, default=0)
    parser.add_argument("--maximum-rrt-preflights", type=int, default=MAX_RRT_PREFLIGHTS)
    parser.add_argument("--maximum-physical-branches", type=int, default=MAX_PHYSICAL_BRANCHES)
    parser.add_argument("--rrt-time-budget-s", type=float, default=12.0)
    args = parser.parse_args()
    if not 1 <= args.maximum_rrt_preflights <= MAX_RRT_PREFLIGHTS:
        parser.error(f"--maximum-rrt-preflights must be in 1..{MAX_RRT_PREFLIGHTS}")
    if not 0 <= args.candidate_start_index < args.maximum_candidates:
        parser.error("--candidate-start-index must select an index inside the geometry candidate budget")
    if not 1 <= args.maximum_physical_branches <= MAX_PHYSICAL_BRANCHES:
        parser.error(f"--maximum-physical-branches must be in 1..{MAX_PHYSICAL_BRANCHES}")
    if not 0.5 <= args.rrt_time_budget_s <= 60.0:
        parser.error("--rrt-time-budget-s must be in 0.5..60")
    if not 1000 <= args.demonstration_horizon <= 1800:
        parser.error("--demonstration-horizon must be in 1000..1800")
    if args.target_predicate is not None and len(args.target_predicate) != 3:
        parser.error("--target-predicate needs exactly three active BDDL predicate fields")
    return args


def main() -> int:
    report = run(_parse_args())
    print(json.dumps({
        "status": report.get("status"),
        "archive": report.get("archive"),
        "target": report.get("active_target"),
        "candidate_count": len(report.get("candidates", [])),
        "physical_branch_count": len(report.get("physical_branches", [])),
        "success": report.get("success"),
        "failure_classification": report.get("failure_classification"),
        "snapshot_recovery_verification": report.get("snapshot_recovery_verification"),
    }, ensure_ascii=False, indent=2), flush=True)
    if report.get("success") is None:
        return 2
    return 0 if report.get("snapshot_recovery_verification", {}).get("status") == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
