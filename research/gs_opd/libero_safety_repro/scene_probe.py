# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Capture read-only OA/L0T0 MuJoCo inputs for the CuRobo route stage."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = Path("/workspace/zy_workspace/LIBERO-Safety")
LIBERO_PYTHON = LIBERO_ROOT / ".venv/bin/python"
BRIDGE_SCRIPT = REPO_ROOT / "research/gs_opd/smoke/safety_bridge.py"
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "oa_l0t0.json"
AUTHKEY = b"gs-opd-local"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": __import__("base64").b64encode(value).decode("ascii")}
    return value


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_only_rpc(connection, request: dict[str, Any]) -> dict[str, Any]:
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError(f"bridge_returned_non_mapping_for:{request['op']}")
    return response


def _pose_matrix(position_m: Any, rotation: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    position = np.asarray(position_m, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64)
    if position.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("MuJoCo_pose_fields_have_invalid_shapes")
    if not np.isfinite(position).all() or not np.isfinite(matrix).all():
        raise ValueError("MuJoCo_pose_fields_are_not_finite")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-6
    ):
        raise ValueError("MuJoCo_pose_rotation_is_not_SO3")
    transform[:3, :3] = matrix
    transform[:3, 3] = position
    return transform


def _capture(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if not LIBERO_PYTHON.is_file() or not BRIDGE_SCRIPT.is_file():
        raise FileNotFoundError("existing_LIBERO_environment_or_Safety_Bridge_is_unavailable")
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "safety_bridge.log"
    port = _port()
    env = os.environ.copy()
    env.update({
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LIBERO_CONFIG_PATH": "/workspace/zy_workspace/.libero_safety_config",
        "PYTHONPATH": os.pathsep.join((
            str(LIBERO_ROOT),
            str(LIBERO_ROOT / "third_party/robosuite-1.4"),
            str(REPO_ROOT),
        )),
    })
    command = [
        str(LIBERO_PYTHON), str(BRIDGE_SCRIPT),
        "--benchmark", str(config["benchmark"]),
        "--level", str(config["level"]),
        "--task-id", str(config["task_id"]),
        "--init-state", str(config["init_state"]),
        "--seed", str(config["seed"]),
        "--camera-resolution", str(config["camera_resolution"]),
        "--port", str(port),
        "--demonstration-horizon", str(config["demonstration_horizon"]),
    ]
    with log_path.open("wb", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        connection = None
        deadline = time.monotonic() + 120.0
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Safety_Bridge_exited_before_ready:{process.returncode}")
                try:
                    connection = Client(("127.0.0.1", port), authkey=AUTHKEY)
                    break
                except (ConnectionRefusedError, OSError):
                    time.sleep(0.25)
            if connection is None:
                raise TimeoutError("Safety_Bridge_did_not_start_within_120_seconds")

            initial = _read_only_rpc(connection, {"op": "observe", "include_snapshot_state": True})
            instruction = str(config["instruction"])
            if initial.get("instruction") != instruction:
                raise ValueError("active_Bridge_task_instruction_does_not_match_config")
            if int(initial.get("timestep", -1)) != 0:
                raise ValueError("Bridge_did_not_start_at_the_configured_initial_state")
            bindings = _read_only_rpc(connection, {"op": "task_semantic_geometry_bindings"})
            status = _read_only_rpc(connection, {"op": "native_task_status"})
            goal = _read_only_rpc(connection, {
                "op": "native_goal_query", "predicate": config["target_predicate"]
            })
            target_context = _read_only_rpc(connection, {
                "op": "task_success_first_context",
                "expected_instruction": instruction,
                "target_predicate": config["target_predicate"],
            })
            controller = _read_only_rpc(connection, {"op": "repair_context"})
            arm_qpos = target_context.get("arm_qpos")
            if arm_qpos is None:
                raise ValueError("read_only_target_context_lacks_live_arm_joint_state")
            fk = _read_only_rpc(connection, {
                "op": "scratch_robot_fk_audit",
                "arm_joint_positions_rad": [arm_qpos],
            })
            collision_scene = _read_only_rpc(connection, {"op": "collision_scene_export"})
            initial_contact_audit = _read_only_rpc(connection, {
                "op": "scratch_robot_path_contact_audit",
                "arm_joint_states_rad": [arm_qpos],
                "gripper_joint_states_rad": [target_context["active_open_gripper_qpos"]],
                "object_id": config["target_object_id"],
            })
            final = _read_only_rpc(connection, {"op": "observe", "include_snapshot_state": True})

            metadata = initial.get("scene_metadata", {})
            expected_metadata = {
                "benchmark": config["benchmark"],
                "level": config["level"],
                "task_id": config["task_id"],
                "init_state_index": config["init_state"],
                "environment_seed": config["seed"],
                "horizon_mode": "standard_benchmark",
            }
            for field, expected in expected_metadata.items():
                actual = metadata.get(field)
                if actual != expected:
                    raise ValueError(f"scene_metadata_mismatch:{field}:{actual}!={expected}")
            if bindings.get("status") != "complete":
                raise ValueError("active_BDDL_semantic_geometry_bindings_are_incomplete")
            if status.get("status") != "complete" or goal.get("status") != "complete":
                if status.get("status") != "complete":
                    raise ValueError(f"native_task_status_is_incomplete:{status.get('error')}")
            if target_context.get("status") != "complete" or target_context.get("read_only") is not True:
                raise ValueError("active_target_physical_context_is_incomplete")
            if target_context.get("native_goal_predicate") != config["target_predicate"]:
                raise ValueError("configured_target_predicate_is_not_the_active_native_goal")
            if fk.get("status") != "complete" or fk.get("physical_state_unchanged") is not True:
                raise ValueError("MuJoCo_scratch_robot_FK_audit_is_incomplete")
            if collision_scene.get("status") != "complete" or collision_scene.get("physical_state_unchanged") is not True:
                raise ValueError("full_collision_scene_export_is_incomplete")
            if collision_scene.get("unsupported_geometries"):
                raise ValueError("MuJoCo_collision_scene_has_unsupported_geometries")
            if (
                initial_contact_audit.get("status") != "complete"
                or initial_contact_audit.get("physical_state_unchanged") is not True
            ):
                raise ValueError("scratch_MuJoCo_robot_contact_audit_is_incomplete_or_mutated_state")
            before_state = initial.get("snapshot_state", {}).get("mujoco_integration_state")
            after_state = final.get("snapshot_state", {}).get("mujoco_integration_state")
            if before_state is None or before_state != after_state:
                raise ValueError("read_only_scene_queries_changed_the_live_integration_state")
            if final.get("scene_metadata") != metadata:
                raise ValueError("scene_metadata_changed_during_read_only_capture")

            root_name = fk.get("root_body_name")
            poses = fk["poses"]
            bodies = poses[0].get("bodies", {}) if len(poses) == 1 else {}
            if root_name not in bodies or "robot0_right_hand" not in bodies:
                raise ValueError("scratch_FK_did_not_return_verified_base_and_planner_tool_bodies")
            base_body = bodies[root_name]
            hand_body = bodies["robot0_right_hand"]
            eef_pose = target_context["eef_pose_world"]
            world_base = _pose_matrix(base_body["position_world_m"], RotationFromQuat(base_body["quaternion_world_wxyz"]))
            world_eef = _pose_matrix(eef_pose["position_m"], eef_pose["rotation"])
            world_planner_tool = _pose_matrix(hand_body["position_world_m"], RotationFromQuat(hand_body["quaternion_world_wxyz"]))
            eef_to_planner_tool = np.linalg.inv(world_eef) @ world_planner_tool
            if not np.isfinite(eef_to_planner_tool).all() or not np.allclose(
                eef_to_planner_tool[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8
            ):
                raise ValueError("live_EEF_to_CuRobo_tool_calibration_is_invalid")
            if controller.get("status") != "available" or not controller.get("robots"):
                raise ValueError("active_robot_controller_context_is_incomplete")
            if controller["robots"][0].get("controller_name") != "OSC_POSE":
                raise ValueError("physical_episode_controller_is_not_OSC_POSE")
            region_binding = bindings.get("bindings", {}).get(config["goal_region_id"])
            if not isinstance(region_binding, dict):
                raise ValueError(f"configured_goal_region_frame_is_not_BDDL_bound:{config['goal_region_id']}")
            region_state_matches = [
                state for state in initial.get("snapshot_state", {}).get("object_states", {}).values()
                if int(state.get("body_id", -1)) == int(region_binding["root_body_id"])
            ]
            if len(region_state_matches) != 1:
                raise ValueError("goal_region_root_pose_is_not_unique_in_live_MuJoCo_state")
            region_state = region_state_matches[0]
            region_quat = np.asarray(region_state["root_quat"], dtype=np.float64)
            region_rotation = RotationFromQuat(region_quat)
            world_goal_region = _pose_matrix(region_state["root_pos"], region_rotation)

            source = {
                "schema": "gs_opd_libero_safety_scene_source.v1",
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "scene_metadata": metadata,
                "initial_observation": {
                    key: value for key, value in initial.items() if key != "images"
                },
                "initial_native_task_status": status,
                "initial_native_goal": goal,
                "semantic_geometry_bindings": bindings,
                "target_context": target_context,
                "controller_context": controller,
                "scratch_robot_fk_audit": fk,
                "collision_scene_export": collision_scene,
                "initial_scratch_robot_contact_audit": initial_contact_audit,
                "read_only_state_equal_after_queries": True,
                "planner_frames": {
                    "world_base_transform_m": world_base.tolist(),
                    "world_eef_transform_m": world_eef.tolist(),
                    "world_panda_hand_transform_m": world_planner_tool.tolist(),
                    "world_goal_region_transform_m": world_goal_region.tolist(),
                    "eef_to_panda_hand_transform": eef_to_planner_tool.tolist(),
                    "planner_tool_frame": "panda_hand",
                    "MuJoCo_panda_hand_body": "robot0_right_hand",
                    "MuJoCo_EEF_frame": "gripper0_eef_site",
                    "base_body_name": root_name,
                    "goal_region_frame_id": config["goal_region_id"],
                    "goal_region_body_id": int(region_binding["root_body_id"]),
                },
            }
            payload_path = output_dir / "scene_source.json.gz"
            with gzip.open(payload_path, "wt", encoding="utf-8") as handle:
                json.dump(_json_safe(source), handle, allow_nan=False, separators=(",", ":"))
            digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
            summary = {
                "schema": "gs_opd_libero_safety_scene_probe_report.v1",
                "status": "pass",
                "scene_source_path": str(payload_path),
                "scene_source_sha256": digest,
                "scene_source_bytes_gzip": payload_path.stat().st_size,
                "config": config,
                "scene_metadata": metadata,
                "bddl_sha256": metadata.get("bddl_sha256"),
                "model_geometry_sha256": collision_scene.get("model_geometry_sha256"),
                "native_goal": goal,
                "native_goal_details_status": goal.get("status"),
                "native_goal_details_error": goal.get("error"),
                "initial_task_success": status.get("task_success"),
                "robot_collision_geom_count": len(collision_scene.get("robot_collision_geometries", [])),
                "scene_collision_geom_count": len(collision_scene.get("scene_collision_geometries", [])),
                "unsupported_collision_geom_count": len(collision_scene.get("unsupported_geometries", [])),
                "initial_robot_contact_count": sum(
                    state.get("contact_count", 0) for state in initial_contact_audit.get("states", [])
                ),
                "initial_illegal_robot_contact_count": initial_contact_audit.get("illegal_contact_count"),
                "target_collision_geom_count": len(target_context.get("target_binding", {}).get("geom_ids", [])),
                "base_body_name": root_name,
                "planner_tool_frame": "panda_hand",
                "mujoco_tool_body": "robot0_right_hand",
                "eef_to_planner_tool_transform": eef_to_planner_tool.tolist(),
                "world_goal_region_transform_m": world_goal_region.tolist(),
                "goal_region_frame_id": config["goal_region_id"],
                "controller_name": controller["robots"][0].get("controller_name"),
                "controller_action_dim": controller["robots"][0].get("action_dim"),
                "queries_are_read_only": True,
                "active_integration_state_unchanged": True,
                "action_count": 0,
                "safety_certification": "not_applicable_no_rollout",
            }
            (output_dir / "scene_probe_report.json").write_text(
                json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            return summary
        finally:
            if connection is not None:
                connection.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def RotationFromQuat(quaternion_wxyz: Any) -> np.ndarray:
    """Convert a MuJoCo WXYZ quaternion through SciPy's XYZW interface."""
    from scipy.spatial.transform import Rotation

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("MuJoCo_quaternion_must_be_a_finite_WXYZ_vector")
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("MuJoCo_quaternion_must_be_unit_WXYZ")
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = _capture(config, args.output_dir.resolve())
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
