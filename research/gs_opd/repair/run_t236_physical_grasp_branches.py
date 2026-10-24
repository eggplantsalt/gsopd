# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Bounded closed-loop t=236 OSC_POSE Pot 2 grasp and lift branches.

Every branch begins from the exact archived Runtime Snapshot. Branches are
isolated and never concatenated. Every physical action records all 25 MuJoCo
substeps, dual-camera observations, live grasp evidence, and a recoverable
Runtime Snapshot. A failed branch stops on the first unsafe or incomplete
monitoring result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from research.gs_opd.repair.automatic_recovery import RecoveryArtifactWriter
from research.gs_opd.repair.automatic_recovery_runtime import (
    AutomaticRecoveryBridgeBackend,
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
)
from research.gs_opd.repair.mujoco_jointspace_planning import (
    validate_finger_only_contact_geoms,
)
from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action


ARCHIVE_ROOT = Path("/opt/data/private/gs_opd/rollouts/l0t2_round0_tsfs_safe_seed_20261024")
DEFAULT_CONFIG = ARCHIVE_ROOT / "tsfs01_episode_config.json"
DEFAULT_SNAPSHOT = ARCHIVE_ROOT / "episode1/snapshots/main_timestep_000236.pkl"
DEFAULT_OUTPUT = ARCHIVE_ROOT / "episode1_t236_relaxed_clearance_physical_branches_20261024"
EXPECTED_MODEL_SHA256 = "1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752"
EXPECTED_T236_INTEGRATION_SHA256 = "a299ea309f78435c868306716bfa403834d20f3599473793b25612e93207c0f8"
TARGET_OBJECT = "moka_pot_2"
STANDARD_OBSTACLES = ("bottle_of_ginger_beer_1", "white_place_box_1")
MAX_BRANCHES = 2
MAX_ACTIONS_PER_BRANCH = 100
EXPECTED_SUBSTEPS = 25

BRANCH_DESIGNS = [
    {"name": "prior_axis2_rot+30_current_eef_short_approach", "step_mm": 2.0, "roll_deg": 0.0, "offset_object_mm": [0, 0, 0], "early_close_mm": 0.0, "lift_tilt_deg": 0.0},
    {"name": "prior_axis2_rot+30_wrist_roll15_current_eef", "step_mm": 1.5, "roll_deg": 15.0, "offset_object_mm": [0, 0, 0], "early_close_mm": 4.0, "lift_tilt_deg": 0.0},
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "bytes_length": len(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _pose_matrix(position: Any, quaternion_wxyz: Any) -> np.ndarray:
    quat = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12:
        raise ValueError("object_pose_quaternion_is_invalid")
    quat = quat / np.linalg.norm(quat)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(np.r_[quat[1:], quat[0]]).as_matrix()
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    return pose


def _quat_wxyz(rotation: Rotation) -> np.ndarray:
    xyzw = rotation.as_quat()
    return np.r_[xyzw[3], xyzw[:3]]


def _runtime_state_hash(data: dict[str, Any]) -> str:
    state = np.asarray(data["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
    return hashlib.sha256(state.tobytes()).hexdigest()


def _target_binding(backend, object_id: str) -> dict[str, Any]:
    binding = backend.bindings[object_id]
    return {
        "object_id": object_id,
        "body_id": int(binding["root_body_id"]),
        "body_name": str(binding["root_body_name"]),
        "geom_ids": [int(value) for value in binding["root_geom_ids"]],
        "geom_names": [str(value) for value in binding["root_geom_names"]],
    }


def _make_policy(backend, target_margin: float, obstacle_margins: dict[str, float]) -> dict[str, Any]:
    policy = json.loads(json.dumps(backend.safety_policy))
    binding = backend.bindings[TARGET_OBJECT]
    finger_ids = sorted(int(value) for value in backend.context_static["gripper_geometry_ids"])
    inventory = backend.bridge.request({"op": "gripper_geometry_inventory"})
    geometries = inventory["grippers"][0]["geometries"]
    left, right = set(), set()
    for item in geometries:
        name = str(item.get("geom_name", ""))
        geom_id = int(item["geom_id"])
        if name.endswith("finger1_collision") or name.endswith("finger1_pad_collision"):
            left.add(geom_id)
        elif name.endswith("finger2_collision") or name.endswith("finger2_pad_collision"):
            right.add(geom_id)
    verified = validate_finger_only_contact_geoms(list(left), list(right), finger_ids)
    if verified != set(finger_ids) or len(verified) != 4:
        raise ValueError("finger_only_contact_allowlist_does_not_equal_four_verified_finger_geoms")
    for group in policy["forbidden_objects"]:
        object_id = group["object_id"]
        group["minimum_clearance_m"] = float(obstacle_margins.get(object_id, 0.010))
    for group in policy["allowed_target_objects"]:
        object_id = group["object_id"]
        group["minimum_clearance_m"] = float(target_margin if object_id == TARGET_OBJECT else 0.010)
    policy["allowed_contact_pairs"] = [
        {"robot_geom_id": finger, "scene_geom_id": int(scene), "minimum_clearance_m": -0.002}
        for finger in sorted(verified)
        for scene in sorted(int(value) for value in binding["geom_ids"])
    ]
    policy["minimum_clearance_m"] = 0.010
    policy["target_minimum_clearance_m"] = float(target_margin)
    policy["clearance_tiers_m"] = {
        "target_object_nonfinger_minimum": float(target_margin),
        "forbidden_objects": {
            str(key): float(value) for key, value in obstacle_margins.items()
        },
        "default_forbidden_object_minimum": 0.010,
        "allowed_finger_target_penetration_maximum": 0.002,
    }
    return policy


def _grasp_state(bridge, backend, object_id: str) -> dict[str, Any]:
    return bridge.request({
        "op": "grasp_state",
        "expected_instruction": backend.context_static["instruction"],
        "carried_binding": _target_binding(backend, object_id),
    })


def _state_object(backend, data: dict[str, Any], object_id: str) -> dict[str, Any]:
    return backend._live_object_state(data, object_id)


def _pose_states(data: dict[str, Any], backend, object_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robot = data["snapshot_state"]["robot_states"][0]
    eef_p = np.asarray(robot["eef_pos"], dtype=np.float64)
    eef_q = np.asarray(robot["eef_quat"], dtype=np.float64)
    obj = _state_object(backend, data, object_id)
    return eef_p, eef_q, _pose_matrix(obj["root_pos"], obj["root_quat"])


def _camera_frame(data: dict[str, Any], key: str) -> np.ndarray:
    image = data.get("images", {}).get(key)
    if not isinstance(image, dict):
        raise ValueError(f"required_dual_camera_observation_missing:{key}")
    return np.frombuffer(image["data"], dtype=np.dtype(image["dtype"])).reshape(image["shape"]).copy()


class BranchRecorder:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.snapshots_dir = self.output_dir / "snapshots"
        self.snapshots_dir.mkdir()
        self.rows: list[dict[str, Any]] = []
        self.agent_frames: list[np.ndarray] = []
        self.wrist_frames: list[np.ndarray] = []
        self.events_path = self.output_dir / "events.jsonl"
        self.substeps_path = self.output_dir / "substep_safety.jsonl"

    def event(self, row: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")

    def save_snapshot(self, label: str, payload: bytes) -> str:
        path = self.snapshots_dir / f"{label}.pkl"
        path.write_bytes(payload)
        return str(path)

    def add_initial_observation(self, data: dict[str, Any]) -> None:
        self.agent_frames.append(_camera_frame(data, "agentview_image"))
        self.wrist_frames.append(_camera_frame(data, "robot0_eye_in_hand_image"))
        (self.output_dir / "initial_observation.json").write_text(
            json.dumps({
                "timestep": data.get("timestep"),
                "state": data.get("state"),
                "snapshot_state": data.get("snapshot_state"),
                "scene_metadata": data.get("scene_metadata"),
            }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def add(self, action: np.ndarray, reply: dict[str, Any], grasp: dict[str, Any], native_status: dict[str, Any], phase: str, backend) -> dict[str, Any]:
        monitor = reply.get("contact_monitoring", {})
        geometry = monitor.get("geometry_monitoring", {})
        substeps = geometry.get("substeps", [])
        if len(substeps) != EXPECTED_SUBSTEPS:
            raise RuntimeError(f"expected_25_substep_safety_records_got_{len(substeps)}")
        eef_p, eef_q, obj_pose = _pose_states(reply, backend, TARGET_OBJECT)
        snapshot = backend.bridge.request({"op": "export_runtime_snapshot"})
        if snapshot.get("status") != "complete" or not isinstance(snapshot.get("payload"), bytes):
            raise RuntimeError("physical_branch_runtime_snapshot_export_failed")
        timestep = int(reply.get("timestep", -1))
        snapshot_path = self.save_snapshot(
            f"timestep_{timestep:06d}", snapshot["payload"]
        )
        min_by_object: dict[str, float] = {}
        for substep in substeps:
            for object_id, detail in substep.get("clearances_by_object", {}).items():
                distance = detail.get("minimum_distance_m") if isinstance(detail, dict) else None
                if distance is not None:
                    min_by_object[object_id] = min(
                        float(distance), min_by_object.get(object_id, math.inf)
                    )
        row = {
            "step_index": len(self.rows),
            "timestep": timestep,
            "phase": phase,
            "action": np.asarray(action, dtype=np.float32).tolist(),
            "eef_position_world_m": eef_p.tolist(),
            "eef_quaternion_wxyz": eef_q.tolist(),
            "pot2_root_position_world_m": obj_pose[:3, 3].tolist(),
            "pot2_root_rotation_world": obj_pose[:3, :3].tolist(),
            "reward": reply.get("reward"),
            "done": reply.get("done"),
            "native_task_success": reply.get("task_success"),
            "native_task_status": native_status,
            "grasp_state": grasp,
            "gripper_command_scalar": float(np.asarray(action, dtype=np.float64)[-1]),
            "gripper_formatted_current_action": grasp.get("gripper_current_action"),
            "gripper_qpos": reply["snapshot_state"]["robot_states"][0].get("gripper_qpos"),
            "arm_joint_qpos": reply["snapshot_state"]["robot_states"][0].get("joint_qpos"),
            "contact_monitoring": monitor,
            "safety_geometry": reply.get("safety_geometry"),
            "contacts": reply.get("contacts"),
            "min_clearance_by_object_m": min_by_object,
            "runtime_snapshot_path": snapshot_path,
            "runtime_snapshot_sha256": hashlib.sha256(snapshot["payload"]).hexdigest(),
        }
        self.rows.append(row)
        self.agent_frames.append(_camera_frame(reply, "agentview_image"))
        self.wrist_frames.append(_camera_frame(reply, "robot0_eye_in_hand_image"))
        with self.substeps_path.open("a", encoding="utf-8") as stream:
            for substep in substeps:
                item = {"action_index": row["step_index"], **substep}
                stream.write(json.dumps(_jsonable(item), ensure_ascii=False, sort_keys=True) + "\n")
        self.event({"event": "physical_action", **row})
        return row

    def save(self) -> None:
        if not self.rows:
            return
        try:
            import cv2
        except ImportError:
            import sys
            video_site_packages = Path("/workspace/zy_workspace/RLinf/.venv/lib/python3.11/site-packages")
            if video_site_packages.is_dir() and str(video_site_packages) not in sys.path:
                sys.path.append(str(video_site_packages))
            import cv2
        agent = np.stack(self.agent_frames)
        wrist = np.stack(self.wrist_frames)
        np.savez_compressed(
            self.output_dir / "observation_action_trace.npz",
            actions=np.asarray([row["action"] for row in self.rows], dtype=np.float32),
            timesteps=np.asarray([row["timestep"] for row in self.rows], dtype=np.int32),
            eef_positions=np.asarray([row["eef_position_world_m"] for row in self.rows], dtype=np.float64),
            eef_quaternions=np.asarray([row["eef_quaternion_wxyz"] for row in self.rows], dtype=np.float64),
            pot2_positions=np.asarray([row["pot2_root_position_world_m"] for row in self.rows], dtype=np.float64),
            agentview_rgb=agent,
            wrist_rgb=wrist,
        )
        for frames, filename in ((agent, "agentview.mp4"), (wrist, "wrist.mp4")):
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(self.output_dir / filename),
                cv2.VideoWriter_fourcc(*"mp4v"),
                20.0,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"could_not_open_video:{filename}")
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        (self.output_dir / "steps.json").write_text(
            json.dumps(_jsonable(self.rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _run_one_branch(bridge, backend, branch_dir: Path, branch_index: int, design: dict[str, Any],
                    source_snapshot: bytes, target_margin: float,
                    obstacle_margins: dict[str, float], maximum_actions: int) -> dict[str, Any]:
    recorder = BranchRecorder(branch_dir)
    branch_id = f"branch_{branch_index:02d}_{design['name']}"
    record: dict[str, Any] = {
        "branch_id": branch_id,
        "design": design,
        "clearance_tiers_m": {
            "moka_pot_2_nonfinger_robot_geom": float(target_margin),
            "forbidden_objects": {
                name: float(obstacle_margins.get(name, 0.010))
                for name in backend.context_static["forbidden_names"]
            },
            "allowed_finger_pot2_max_penetration": 0.002,
        },
        "strict_10mm_safe": False,
        "physical_success": False,
        "stable_grasp": False,
        "lift_30mm": False,
        "safe_exit": False,
        "physical_actions": 0,
        "failure_reason": None,
    }
    policy = _make_policy(backend, target_margin, obstacle_margins)
    record["safety_policy"] = policy
    restored = bridge.request({"op": "restore_runtime_snapshot", "payload": source_snapshot})
    if restored.get("status") != "restored":
        raise RuntimeError(f"branch_t236_snapshot_restore_failed:{restored.get('error')}")
    current = bridge.request({"op": "observe", "include_snapshot_state": True})
    state_sha = _runtime_state_hash(current)
    if int(current.get("timestep", -1)) != 236 or state_sha != EXPECTED_T236_INTEGRATION_SHA256:
        raise RuntimeError(f"branch_start_snapshot_identity_mismatch:t={current.get('timestep')}:sha={state_sha}")
    recorder.event({
        "event": "branch_start_verified",
        "branch_id": branch_id,
        "timestep": 236,
        "runtime_state_sha256": state_sha,
        "model_geometry_sha256": current.get("scene_metadata", {}).get("model_geometry_sha256"),
        "snapshot_sha256": hashlib.sha256(source_snapshot).hexdigest(),
        "clearance_tiers_m": record["clearance_tiers_m"],
        "finger_geom_ids": sorted(int(value) for value in backend.context_static["gripper_geometry_ids"]),
        "allowed_contact_pairs": policy["allowed_contact_pairs"],
    })
    recorder.add_initial_observation(current)
    initial_snapshot = bridge.request({"op": "export_runtime_snapshot"})
    recorder.save_snapshot("branch_start_timestep_000236", initial_snapshot["payload"])

    candidate = next(
        item for item in backend.config_data["task_execution"]["grasp_pose_candidates"]
        if item["candidate_id"] == "tsfs_01_axis2_rot+30"
    )
    object_data = backend._live_object_state(current, TARGET_OBJECT)
    object_world = _pose_matrix(object_data["root_pos"], object_data["root_quat"])
    object_to_eef = np.asarray(candidate["object_to_eef"], dtype=np.float64).copy()
    object_to_eef[:3, :3] = object_to_eef[:3, :3] @ Rotation.from_euler(
        "z", float(design["roll_deg"]), degrees=True
    ).as_matrix()
    object_to_eef[:3, 3] += np.asarray(design["offset_object_mm"], dtype=np.float64) * 0.001
    grasp_pose = object_world @ object_to_eef
    approach_dir_world = object_world[:3, :3] @ np.asarray(
        candidate["approach_direction_object"], dtype=np.float64
    )
    approach_dir_world /= np.linalg.norm(approach_dir_world)
    approach_distance = float(candidate["approach_distance_m"])
    pregrasp_pose = grasp_pose.copy()
    pregrasp_pose[:3, 3] -= approach_dir_world * approach_distance
    start_eef_p, start_eef_q, pot2_start = _pose_states(current, backend, TARGET_OBJECT)
    fields = bridge.request({"op": "repair_context"})["robots"][0]["fields"]
    if fields.get("input_ref_frame") not in (None, "world"):
        raise RuntimeError("active_OSC_POSE_input_is_not_world_frame")
    stable_streak = 0
    last_progress_pose = start_eef_p.copy()
    stagnation = 0
    first_grasp_pose = None
    action_budget = min(MAX_ACTIONS_PER_BRANCH, int(maximum_actions))
    record["initial_gripper_state"] = {
        "gripper_qpos": current["snapshot_state"]["robot_states"][0].get("gripper_qpos"),
        "grasp_state": _grasp_state(bridge, backend, TARGET_OBJECT),
        "joint_limit_monitor_tolerance_rad": 1e-8,
    }

    def do_action(target_pos: np.ndarray, target_quat: np.ndarray, gripper_action: float, phase: str):
        nonlocal current, stable_streak, last_progress_pose, stagnation
        if len(recorder.rows) >= action_budget:
            raise RuntimeError(f"branch_action_budget_{action_budget}_exhausted")
        record["failure_phase"] = phase
        robot = current["snapshot_state"]["robot_states"][0]
        action = world_pose_tracking_action(
            robot["eef_pos"], robot["eef_quat"], target_pos, target_quat, fields,
            gripper_action=gripper_action,
        )
        if action.shape != (7,) or not np.isfinite(action).all():
            raise RuntimeError("OSC_POSE_action_not_finite_7d")
        reply = bridge.request({
            "op": "step_monitored",
            "action": action.tolist(),
            "action_index": len(recorder.rows),
            "safety_geometry_policy": policy,
        })
        grasp = _grasp_state(bridge, backend, TARGET_OBJECT)
        native_status = bridge.request({"op": "native_task_status"})
        row = recorder.add(action, reply, grasp, native_status, phase, backend)
        current = reply
        monitor = reply.get("contact_monitoring", {})
        geometry = monitor.get("geometry_monitoring", {})
        if (
            monitor.get("status") != "complete"
            or monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
            or geometry.get("status") != "complete"
            or len(geometry.get("substeps", [])) != EXPECTED_SUBSTEPS
        ):
            failed_substeps = [
                (substep_index, item)
                for substep_index, item in enumerate(geometry.get("substeps", []))
                if item.get("status") != "complete"
            ]
            if failed_substeps:
                first_index, first_failure = failed_substeps[0]
                record["monitor_failure"] = {
                    "physics_substep_index_zero_based": first_index,
                    **first_failure,
                }
            else:
                record["monitor_failure"] = {
                    "monitor_status": monitor.get("status"),
                    "geometry_status": geometry.get("status"),
                    "captured_substeps": monitor.get("captured_physics_substeps"),
                }
            record["monitor_failure"].update({
                "control_action_index": row["step_index"],
                "control_timestep": row["timestep"],
                "action": row["action"],
                "gripper_command_scalar": row["gripper_command_scalar"],
                "gripper_formatted_current_action": row["gripper_formatted_current_action"],
                "gripper_qpos_after_action": row["gripper_qpos"],
            })
            raise RuntimeError(
                "substep_safety_monitor_failed_closed:"
                + str(geometry.get("status") or monitor.get("status"))
            )
        if reply.get("done") is True:
            raise RuntimeError("native_episode_done_before_local_grasp_lift_exit")
        now = np.asarray(row["eef_position_world_m"], dtype=np.float64)
        if phase in {
            "gripper_limit_preflight_closeward",
            "gripper_limit_preflight_neutral_hold",
            "close_gripper",
        }:
            # These are intentional stationary gripper controls. The gripper's
            # internal command and qpos are the progress variables here.
            stagnation = 0
            last_progress_pose = now
        elif float(np.linalg.norm(now - last_progress_pose)) < 0.00035:
            stagnation += 1
        else:
            stagnation = 0
            last_progress_pose = now
        if stagnation >= 5:
            raise RuntimeError("physical_eef_stagnation_5_control_actions")
        if grasp.get("grasp_confirmed") is True and grasp.get("finger_contacts", {}).get("status") == "pass":
            stable_streak += 1
        else:
            stable_streak = 0
        return row, grasp

    try:
        # t=236 is already at Approach, only 11.27 mm from the archived grasp
        # target. Correct the gripper command in place before any approach
        # motion; the one +1 action moves it inward and the two 0 actions hold
        # the resulting command instead of continuing to drive the open stop.
        preflight_p, preflight_q, _ = _pose_states(current, backend, TARGET_OBJECT)
        preflight_rows = []
        for index, command in enumerate((1.0, 0.0, 0.0)):
            row, _ = do_action(
                preflight_p,
                preflight_q,
                command,
                "gripper_limit_preflight_closeward" if index == 0 else "gripper_limit_preflight_neutral_hold",
            )
            preflight_rows.append({
                "timestep": row["timestep"],
                "gripper_command_scalar": row["gripper_command_scalar"],
                "gripper_formatted_current_action": row["gripper_formatted_current_action"],
                "gripper_qpos": row["gripper_qpos"],
                "joint_limit_substeps_passed": EXPECTED_SUBSTEPS,
            })
        record["gripper_limit_preflight"] = {
            "status": "pass",
            "controls": preflight_rows,
            "format_action_semantics": "PandaGripper adds [-1,+1]*0.01*sign(action) on each low-level control call; measured 25 calls per monitored action, so +1 moved [1,-1] to [0.75,-0.75] and 0 holds",
        }
        recorder.event({"event": "gripper_limit_preflight_pass", **record["gripper_limit_preflight"]})

        # Retarget to the live object after the in-place check. Follow a short
        # interpolated path from the actual EEF to the grasp target; do not
        # backtrack to the old 60.65 mm pregrasp pose and traverse it again.
        live_object = _state_object(backend, current, TARGET_OBJECT)
        object_world = _pose_matrix(live_object["root_pos"], live_object["root_quat"])
        grasp_pose = object_world @ object_to_eef
        approach_dir_world = object_world[:3, :3] @ np.asarray(
            candidate["approach_direction_object"], dtype=np.float64
        )
        approach_dir_world /= np.linalg.norm(approach_dir_world)
        grasp_p = grasp_pose[:3, 3]
        grasp_q = _quat_wxyz(Rotation.from_matrix(grasp_pose[:3, :3]))
        start_eef_p, start_eef_q, _ = _pose_states(current, backend, TARGET_OBJECT)
        start_rotation = Rotation.from_quat(np.r_[start_eef_q[1:], start_eef_q[0]])
        grasp_rotation = Rotation.from_matrix(grasp_pose[:3, :3])
        rotation_path = Slerp([0.0, 1.0], Rotation.from_quat(
            np.stack([start_rotation.as_quat(), grasp_rotation.as_quat()])
        ))
        direct_distance = float(np.linalg.norm(grasp_p - start_eef_p))
        record["approach_retarget"] = {
            "source": "live_EEF_after_gripper_preflight",
            "current_eef_position_world_m": start_eef_p.tolist(),
            "grasp_target_position_world_m": grasp_p.tolist(),
            "direct_remaining_distance_m": direct_distance,
            "historical_pregrasp_distance_m": approach_distance,
            "historical_pregrasp_waypoint_distance_m": float(np.linalg.norm(pregrasp_pose[:3, 3] - start_eef_p)),
        }
        count = max(1, int(math.ceil(direct_distance / (float(design["step_mm"]) * 0.001))))
        for index in range(1, count + 1):
            alpha = index / count
            target_p = start_eef_p + alpha * (grasp_p - start_eef_p)
            target_q = _quat_wxyz(rotation_path([alpha])[0])
            do_action(target_p, target_q, 0.0, "approach_from_current_eef")

        # Keep the existing live bilateral-grasp predicate. Continue its
        # close command in closed loop until three consecutive confirmations
        # or the explicit per-branch action budget is exhausted.
        close_step = 0
        while stable_streak < 3 and len(recorder.rows) < action_budget:
            current_p, _, _ = _pose_states(current, backend, TARGET_OBJECT)
            _, grasp = do_action(current_p, grasp_q, 1.0, "close_gripper")
            close_step += 1
            if grasp.get("grasp_confirmed") is True and grasp.get("finger_contacts", {}).get("status") == "pass":
                if first_grasp_pose is None:
                    first_grasp_pose = _pose_states(current, backend, TARGET_OBJECT)[2]
        record["close_control_actions"] = close_step
        grasp = _grasp_state(bridge, backend, TARGET_OBJECT)
        if grasp.get("grasp_confirmed") is not True or grasp.get("finger_contacts", {}).get("status") != "pass" or stable_streak < 3:
            if len(recorder.rows) >= action_budget:
                raise RuntimeError("branch_action_budget_exhausted_before_stable_bilateral_grasp")
            raise RuntimeError("physical_stable_two_finger_grasp_not_confirmed_for_three_consecutive_actions")
        record["stable_grasp"] = True
        held_start = _pose_states(current, backend, TARGET_OBJECT)[2]
        recorder.event({
            "event": "stable_two_finger_grasp",
            "timestep": current["timestep"],
            "grasp_state": grasp,
            "eef_position_world_m": current["state"]["robot0_eef_pos"],
            "pot2_root_position_world_m": held_start[:3, 3].tolist(),
        })

        lift_tilt = math.radians(float(design["lift_tilt_deg"]))
        lift_vector = np.asarray([math.sin(lift_tilt), 0.0, math.cos(lift_tilt)], dtype=np.float64)
        lift_vector /= np.linalg.norm(lift_vector)
        eef_start = _pose_states(current, backend, TARGET_OBJECT)[0]
        lift_step = min(float(design["step_mm"]) * 0.001, 0.003)
        lift_count = int(math.ceil(0.036 / lift_step))
        for index in range(1, lift_count + 1):
            target_p = eef_start + lift_vector * (0.036 * index / lift_count)
            _, grasp = do_action(target_p, grasp_q, 1.0, "lift")
            if grasp.get("grasp_confirmed") is not True:
                raise RuntimeError("two_finger_grasp_lost_during_lift")
            obj_pose = _pose_states(current, backend, TARGET_OBJECT)[2]
            rise = float(obj_pose[2, 3] - held_start[2, 3])
            if rise >= 0.030:
                record["lift_30mm"] = True
                recorder.event({
                    "event": "pot2_real_lift_at_least_30mm",
                    "timestep": current["timestep"],
                    "pot2_start_position_world_m": held_start[:3, 3].tolist(),
                    "pot2_position_world_m": obj_pose[:3, 3].tolist(),
                    "measured_vertical_rise_m": rise,
                })
                break
        if record["lift_30mm"] is not True:
            raise RuntimeError("pot2_real_vertical_lift_below_30mm")

        eef_after_lift = _pose_states(current, backend, TARGET_OBJECT)[0]
        exit_goal = eef_after_lift - approach_dir_world * 0.012
        exit_count = max(1, int(math.ceil(0.012 / min(0.003, float(design["step_mm"]) * 0.001))))
        for index in range(1, exit_count + 1):
            target_p = eef_after_lift + (exit_goal - eef_after_lift) * index / exit_count
            _, grasp = do_action(target_p, grasp_q, 1.0, "safe_exit")
            if grasp.get("grasp_confirmed") is not True:
                raise RuntimeError("two_finger_grasp_lost_during_safe_exit")
        record["safe_exit"] = True
        record["physical_success"] = True
    except Exception as exc:
        record["failure_reason"] = f"{type(exc).__name__}:{exc}"
    finally:
        try:
            record["native_task_status"] = bridge.request({"op": "native_task_status"})
        except Exception as exc:
            record["native_task_status_error"] = f"{type(exc).__name__}:{exc}"
        record["physical_actions"] = len(recorder.rows)
        by_object: dict[str, float] = {}
        substeps = [
            substep for row in recorder.rows
            for substep in row.get("contact_monitoring", {}).get("geometry_monitoring", {}).get("substeps", [])
        ]
        for row in recorder.rows:
            for key, value in row.get("min_clearance_by_object_m", {}).items():
                by_object[key] = min(float(value), by_object.get(key, math.inf))
        record["minimum_clearance_by_object_m"] = by_object
        record["all_expected_substep_records_present"] = (
            len(substeps) == EXPECTED_SUBSTEPS * len(recorder.rows)
        )
        record["no_forbidden_contacts_or_self_collision"] = all(
            sub.get("status") == "complete"
            and not sub.get("self_contacts")
            and not sub.get("unresolved_contacts")
            and not sub.get("forbidden_object_contacts")
            and not sub.get("unallowed_scene_contacts")
            for sub in substeps
        )
        strict_checked_objects = {
            item["object_id"]
            for group_name in ("forbidden_objects", "allowed_target_objects")
            for item in policy.get(group_name, [])
        }
        strict_measured_clearances = {
            object_id: by_object[object_id]
            for object_id in strict_checked_objects
            if object_id in by_object
        }
        record["strict_10mm_safe"] = bool(
            len(recorder.rows) > 0
            and record["all_expected_substep_records_present"]
            and record["no_forbidden_contacts_or_self_collision"]
            and set(strict_measured_clearances) == strict_checked_objects
            and all(value >= 0.010 for value in strict_measured_clearances.values())
        )
        record["strict_10mm_blockers"] = {
            key: value for key, value in strict_measured_clearances.items() if value < 0.010
        }
        record["strict_10mm_checked_objects"] = sorted(strict_checked_objects)
        recorder.save()
        (branch_dir / "branch_summary.json").write_text(
            json.dumps(_jsonable(record), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return record


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config_data, base_recovery = load_runtime_config(args.config)
    source_snapshot = args.snapshot.read_bytes()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bridge = BridgeProcess(args.output_dir, python=args.python, port=args.bridge_port)
    backend = None
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema": "gs_opd_t236_physical_grasp_branches.v1",
        "source_snapshot": str(args.snapshot),
        "source_snapshot_sha256": hashlib.sha256(source_snapshot).hexdigest(),
        "source_timestep": 236,
        "expected_runtime_state_sha256": EXPECTED_T236_INTEGRATION_SHA256,
        "expected_model_geometry_sha256": EXPECTED_MODEL_SHA256,
        "maximum_physical_branches": min(MAX_BRANCHES, int(args.maximum_branches)),
        "maximum_actions_per_branch": min(MAX_ACTIONS_PER_BRANCH, int(args.maximum_actions)),
        "maximum_actions_total": min(MAX_BRANCHES, int(args.maximum_branches)) * min(MAX_ACTIONS_PER_BRANCH, int(args.maximum_actions)),
        "branches": [],
        "physical_actions_total": 0,
        "obstacle_margin_tier_sequence": [],
        "category": None,
    }
    try:
        bridge.start(
            benchmark=config_data["benchmark"],
            level=int(config_data["level"]),
            task_id=int(config_data["task_id"]),
            init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=min(1800, int(base_recovery.horizon_steps)),
        )
        reset = bridge.request({"op": "reset"})
        model_sha = reset.get("scene_metadata", {}).get("model_geometry_sha256")
        report["native_reset_model_geometry_sha256"] = model_sha
        if model_sha != EXPECTED_MODEL_SHA256 or reset.get("timestep") != 0:
            raise RuntimeError("strict_t236_experiment_native_reset_model_fingerprint_mismatch")
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": source_snapshot})
        if restored.get("status") != "restored":
            raise RuntimeError(f"strict_t236_snapshot_restore_failed:{restored.get('error')}")
        restored_obs = bridge.request({"op": "observe", "include_snapshot_state": True})
        if restored_obs.get("scene_metadata", {}).get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError("restored_t236_model_geometry_sha256_mismatch")
        if int(restored_obs.get("timestep", -1)) != 236 or _runtime_state_hash(restored_obs) != EXPECTED_T236_INTEGRATION_SHA256:
            raise RuntimeError("restored_t236_integration_state_identity_mismatch")
        recovery = replace(
            base_recovery,
            hard_minimum_clearance_m=0.010,
            target_object_clearance_m=0.002,
            object_clearance_m={TARGET_OBJECT: 0.002},
        )
        backend = AutomaticRecoveryBridgeBackend(
            bridge, config_data, recovery, args.output_dir,
            RecoveryArtifactWriter(args.output_dir), mode="runtime-snapshot", student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=Path(config_data["artifacts"]["t10_runtime_snapshot"]),
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
        )
        backend.latest_data = restored_obs
        backend._load_context_inputs(restored_obs)
        report["restore_validation"] = {
            "status": "pass",
            "model_geometry_sha256": model_sha,
            "runtime_state_sha256": _runtime_state_hash(restored_obs),
            "timestep": restored_obs["timestep"],
            "robot_qpos": restored_obs["snapshot_state"]["robot_states"][0]["joint_qpos"],
            "object_states": restored_obs["snapshot_state"].get("object_states"),
            "target_grasp_state_before_action": _grasp_state(bridge, backend, TARGET_OBJECT),
            "controller": bridge.request({"op": "repair_context"}),
            "safety_policy_source": backend.safety_policy["source"],
            "task_phase_source": "branch-local Approach initialized only after strict archived t236 restore",
            "historical_prefix_actions": 226,
            "historical_prefix_substeps": 5650,
        }
        if backend.context_static.get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError("backend_geometry_context_fingerprint_mismatch")
        if sorted(int(value) for value in backend.context_static["gripper_geometry_ids"]) != [76, 77, 79, 80]:
            raise RuntimeError("finger_only_geom_ids_changed_since_verified_archive")

        target_margin = 0.002
        obstacle_margins: dict[str, float] = {}
        designs = BRANCH_DESIGNS[:report["maximum_physical_branches"]]
        for index, design in enumerate(designs, start=1):
            branch_dir = args.output_dir / f"branch_{index:02d}_{design['name']}"
            row = _run_one_branch(
                bridge, backend, branch_dir, index, design, source_snapshot,
                target_margin, obstacle_margins, report["maximum_actions_per_branch"],
            )
            report["branches"].append(row)
            report["physical_actions_total"] += int(row["physical_actions"])
            failure = str(row.get("failure_reason") or "")
            min_by_object = row.get("minimum_clearance_by_object_m", {})
            failed_monitor = row.get("monitor_failure", {})
            failed_objects = {
                str(item.get("object_id"))
                for item in failed_monitor.get("forbidden_contacts", [])
                if isinstance(item, dict) and item.get("object_id") is not None
            }
            target_min = min_by_object.get(TARGET_OBJECT)
            if TARGET_OBJECT in failed_objects and target_min is not None and 0.0 < float(target_min) < target_margin and "substep_safety_monitor_failed_closed" in failure:
                report["target_margin_tier_change"] = {
                    "from_m": target_margin,
                    "to_m": 0.0,
                    "evidence_minimum_m": float(target_min),
                    "reason": "positive nonfinger gap below 2mm; zero tier preserves no-penetration and contact blocking",
                }
                target_margin = 0.0
            if row.get("physical_success") is True:
                report["category"] = "physical_safe_grasp_lift_exit_candidate"
                break
            if row.get("monitor_failure", {}).get("joint_limit_violations"):
                report["category"] = "joint_limit_monitor_failure_stop_no_repeated_branch"
                report["common_failure"] = {
                    "reason": "first physical branch observed a joint-limit monitor failure shared by the remaining route variants",
                    "joint_limit_violations": row["monitor_failure"]["joint_limit_violations"],
                    "remaining_branches_not_run": report["maximum_physical_branches"] - index,
                }
                break
            if "substep_safety_monitor_failed_closed" in failure:
                report["last_failed_safety_monitor"] = failure
        if report["category"] is None:
            if report["physical_actions_total"] == 0:
                report["category"] = "restoration_or_policy_preflight_failure"
            elif any(row.get("stable_grasp") for row in report["branches"]):
                report["category"] = "stable_grasp_without_verified_lift_and_safe_exit"
            elif any("clearance" in str(row.get("failure_reason", "")) for row in report["branches"]):
                report["category"] = "clearance_tier_blocked_before_stable_grasp"
            else:
                report["category"] = "physical_grasp_not_confirmed_within_branch_budget"
        report["elapsed_seconds"] = time.monotonic() - started
        report["strict_10mm_policy_passed"] = any(row["strict_10mm_safe"] for row in report["branches"])
    except Exception as exc:
        report["category"] = "strict_snapshot_restore_or_physical_monitor_preflight_failure"
        report["fatal_error"] = f"{type(exc).__name__}:{exc}"
        report["elapsed_seconds"] = time.monotonic() - started
    finally:
        (args.output_dir / "physical_branch_report.json").write_text(
            json.dumps(_jsonable(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bridge.close()
        if backend is not None:
            try:
                backend.archive.close()
            except Exception:
                pass
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38821)
    parser.add_argument("--maximum-branches", type=int, default=MAX_BRANCHES)
    parser.add_argument("--maximum-actions", type=int, default=MAX_ACTIONS_PER_BRANCH)
    args = parser.parse_args()
    if not 1 <= args.maximum_branches <= MAX_BRANCHES:
        parser.error(f"--maximum-branches must be in 1..{MAX_BRANCHES}")
    if not 1 <= args.maximum_actions <= MAX_ACTIONS_PER_BRANCH:
        parser.error(f"--maximum-actions must be in 1..{MAX_ACTIONS_PER_BRANCH}")
    result = _run(args)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("category") == "physical_safe_grasp_lift_exit_candidate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
