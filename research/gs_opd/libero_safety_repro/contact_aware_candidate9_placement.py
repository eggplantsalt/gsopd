# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Bounded contact-aware OA/L0T0 candidate9 Place/Release/Retreat search.

Each branch restores the same exact, real candidate9 held-above-plate Runtime
Snapshot. Placement is controlled by short LIBERO OSC_POSE world-frame deltas
and observed MuJoCo bowl/plate support, not by a fixed 6D Place IK target.
Task-feasibility contact observations never alter the unchanged strict safety
policy used for per-substep monitoring.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.gs_opd.libero_safety_repro import complete_candidate9_task as prior_task
from research.gs_opd.libero_safety_repro import plan_oa_l0t0 as api
from research.gs_opd.libero_safety_repro import transport_candidate9_above_plate as prior


DEFAULT_SOURCE_SCENE = prior.DEFAULT_SOURCE_SCENE
DEFAULT_SOURCE_SNAPSHOT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "oa_l0t0_candidate9_placement_branch_search_20261027_130938/"
    "candidate_08/run/terminal_runtime_snapshot.pkl"
)
DEFAULT_SOURCE_REPORT = DEFAULT_SOURCE_SNAPSHOT.parent / "task_report.json"
DEFAULT_OUTPUT_ROOT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "oa_l0t0_candidate9_contact_aware_placement_20261027_attempt06"
)

MODEL_SHA = prior.EXPECTED_MODEL_SHA
OBJECT_ID = prior.TARGET_OBJECT_ID
REGION_ID = prior.TARGET_REGION_ID
INSTRUCTION = prior.INSTRUCTION
HORIZON = 1000
SUBSTEPS = 25
SETTLE_ACTIONS = 12
POSITION_OUTPUT_M = 0.05
ROTATION_OUTPUT_RAD = 0.5
MAX_BRANCHES = 3
MAX_BRANCH_ACTIONS = 220
MAX_DESCENT_M = 0.075
MAX_DESCENT_ACTIONS = 100
SUPPORT_HOLD_ACTIONS = 3
RETREAT_CLEARANCE_M = 0.030
RETREAT_CLEAR_FINAL_ACTIONS = 3
STABLE_LINEAR_SPEED_M_S = 0.01
STABLE_ANGULAR_SPEED_RAD_S = 0.10
STABLE_LAST_THREE_DRIFT_M = 0.002

# These branch schedules are engineering choices informed by the recorded
# t=417 -> t=418 loss of native On under stationary opening. They are not
# parameters disclosed by LIBERO-Safety.
BRANCHES: tuple[dict[str, Any], ...] = (
    {
        "branch_id": "b01_direct_support_vertical_open_retreat",
        "center_to_plate_fraction": 0.0,
        "yaw_delta_deg": 0.0,
        "recenter_step_mm": 5.0,
        "descent_step_mm": 4.0,
        "retreat_style": "vertical",
        "retreat_step_mm": 5.0,
        "open_delay_actions": 0,
    },
    {
        "branch_id": "b02_recenter_then_radial_open_retreat",
        "center_to_plate_fraction": 0.75,
        "yaw_delta_deg": 0.0,
        "recenter_step_mm": 6.0,
        "descent_step_mm": 6.0,
        "retreat_style": "radial_up",
        "retreat_step_mm": 5.0,
        "open_delay_actions": 0,
    },
    {
        "branch_id": "b03_small_yaw_then_oblique_open_retreat",
        "center_to_plate_fraction": 0.35,
        "yaw_delta_deg": 5.0,
        "recenter_step_mm": 5.0,
        "descent_step_mm": 5.0,
        "retreat_style": "cross_radial_up",
        "retreat_step_mm": 5.0,
        "open_delay_actions": 2,
    },
)


class _PythonObjectState:
    """Compatibility marker used when unpickling archived runtime Snapshots."""


class _Bridge:
    def __init__(self, connection):
        self.connection = connection

    def request(self, request: dict[str, Any]) -> dict[str, Any]:
        return api._rpc(self.connection, request)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(api._json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _snapshot_envelope(payload: bytes) -> dict[str, Any]:
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    envelope = pickle.loads(payload)
    if not isinstance(envelope, dict) or envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
        raise ValueError("source_runtime_snapshot_schema_invalid")
    snapshot = envelope.get("snapshot", {})
    state = np.asarray(snapshot.get("mujoco_state", []), dtype=np.float64)
    timestep = int(snapshot.get("task_bookkeeping", {}).get("timestep", -1))
    if state.ndim != 1 or state.size == 0 or not np.isfinite(state).all():
        raise ValueError("source_runtime_snapshot_integration_state_invalid")
    if envelope.get("model_geometry_sha256") != MODEL_SHA:
        raise ValueError("source_runtime_snapshot_model_signature_mismatch")
    if timestep < 0 or timestep >= HORIZON:
        raise ValueError("source_runtime_snapshot_timestep_outside_standard_horizon")
    return {"envelope": envelope, "state": state, "timestep": timestep}


def _state_fingerprint(snapshot_state: dict[str, Any]) -> str:
    state = np.asarray(snapshot_state.get("mujoco_integration_state", []), dtype=np.float64)
    return hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()


def _decode_rgb(block: dict[str, Any]) -> np.ndarray:
    shape = tuple(block["shape"])
    return np.frombuffer(block["data"], dtype=np.dtype(block["dtype"])).reshape(shape).copy()


def _pose_from_geom(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(row.get("center_world_m"), dtype=np.float64)
    rotation = np.asarray(row.get("rotation_world"), dtype=np.float64)
    half = np.asarray(row.get("half_extents_world_axes_m"), dtype=np.float64)
    if center.shape != (3,) or rotation.shape != (3, 3) or half.shape != (3,):
        raise ValueError("plate_support_collision_geom_pose_invalid")
    if not np.isfinite(center).all() or not np.isfinite(rotation).all() or not np.isfinite(half).all() or np.any(half <= 0):
        raise ValueError("plate_support_collision_geom_nonfinite_or_degenerate")
    return center, rotation


def _support_frame(collision_scene: dict[str, Any], region_binding: dict[str, Any]) -> dict[str, Any]:
    body_id = int(region_binding["root_body_id"])
    body_name = str(region_binding["root_body_name"])
    rows = [
        row for row in collision_scene.get("scene_collision_geometries", [])
        if int(row.get("body_id", -1)) == body_id and row.get("body_name") == body_name
        and row.get("shape") == "cuboid"
    ]
    if not rows:
        raise ValueError("bound_plate_has_no_cuboid_collision_geometry")
    candidates = []
    for row in rows:
        center, rotation = _pose_from_geom(row)
        half = np.asarray(row["half_extents_world_axes_m"], dtype=np.float64)
        thin = int(np.argmin(half))
        normal = rotation[:, thin].copy()
        if normal[2] < 0.0:
            normal *= -1.0
        planar_area = float(np.prod(np.delete(half, thin)))
        if normal[2] >= 0.95:
            candidates.append((planar_area, row, center, rotation, half, thin, normal))
    if not candidates:
        raise ValueError("plate_horizontal_support_collision_geom_missing")
    _, row, center, rotation, half, thin, normal = max(candidates, key=lambda item: item[0])
    normal /= np.linalg.norm(normal)
    top_center = center + normal * float(half[thin])
    return {
        "body_id": body_id,
        "body_name": body_name,
        "geom_id": int(row["geom_id"]),
        "geom_name": str(row.get("geom_name", row.get("name", ""))),
        "all_region_geom_ids": sorted(int(item["geom_id"]) for item in rows),
        "center_world_m": center.tolist(),
        "top_center_world_m": top_center.tolist(),
        "normal_world": normal.tolist(),
        "thin_axis": thin,
        "half_extents_m": half.tolist(),
        "selection": "largest-area near-horizontal cuboid of the active plate body; derived from compiled MuJoCo collision geometry",
    }


def _object_root_position(snapshot_state: dict[str, Any], object_id: str) -> np.ndarray:
    row = snapshot_state.get("object_states", {}).get(f"objects_dict:{object_id}")
    if not isinstance(row, dict):
        raise ValueError(f"object_root_state_missing:{object_id}")
    result = np.asarray(row.get("root_pos"), dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"object_root_position_invalid:{object_id}")
    return result


def _eef_position(snapshot_state: dict[str, Any]) -> np.ndarray:
    rows = snapshot_state.get("robot_states", [])
    if not rows:
        raise ValueError("robot_state_missing")
    result = np.asarray(rows[0].get("eef_pos"), dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("eef_world_position_invalid")
    return result


def _pair_matches(contact: dict[str, Any], first: set[int], second: set[int]) -> bool:
    geom1 = contact.get("geom1", {})
    geom2 = contact.get("geom2", {})
    ids = {int(geom1.get("id", -1)), int(geom2.get("id", -1))}
    return bool(ids & first and ids & second)


def _substep_has_pair(substeps: list[dict[str, Any]], first: set[int], second: set[int], *, last_only: bool = False) -> bool:
    rows = substeps[-1:] if last_only else substeps
    return any(
        _pair_matches(contact, first, second)
        for substep in rows
        for contact in substep.get("contacts", [])
    )


def _monitoring_complete(reply: dict[str, Any]) -> bool:
    monitor = reply.get("contact_monitoring", {})
    subs = monitor.get("substep_contacts", [])
    geometry = monitor.get("geometry_monitoring", {}).get("substeps", [])
    return bool(
        monitor.get("status") == "complete"
        and monitor.get("expected_physics_substeps_per_control_step") == SUBSTEPS
        and monitor.get("captured_physics_substeps") == SUBSTEPS
        and monitor.get("unavailable_contact_substeps") == 0
        and len(subs) == SUBSTEPS
        and all(row.get("status") in {"contact", "no_contact"} for row in subs)
        and len(geometry) == SUBSTEPS
        and all(row.get("status") in {"complete", "unsafe"} for row in geometry)
    )


class _PhysicalRecorder:
    def __init__(self, output: Path, branch_id: str):
        self.output = output
        self.branch_id = branch_id
        self.frames_dir = output / "dual_camera_frames"
        self.agent_dir = self.frames_dir / "agentview"
        self.wrist_dir = self.frames_dir / "wrist"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.wrist_dir.mkdir(parents=True, exist_ok=True)
        self.records = gzip.open(output / "physical_step_records.jsonl.gz", "wt", encoding="utf-8")
        self.video = None
        self.count = 0
        self.substep_count = 0
        self.snapshot_paths: list[str] = []
        self.incomplete_monitor_actions = 0
        self.strict_safe_actions = 0

    def snapshot(self, bridge: _Bridge, label: str) -> str | None:
        saved = bridge.request({"op": "export_runtime_snapshot"})
        if saved.get("status") != "complete" or not isinstance(saved.get("payload"), bytes):
            return None
        path = self.output / f"{label}_runtime_snapshot.pkl"
        path.write_bytes(saved["payload"])
        self.snapshot_paths.append(str(path))
        return str(path)

    def record(self, reply: dict[str, Any], action: np.ndarray, phase: str, strict: dict[str, Any]) -> None:
        images = reply.get("images", {})
        if not {"agentview_image", "robot0_eye_in_hand_image"}.issubset(images):
            raise RuntimeError("dual_camera_observation_missing")
        agent = _decode_rgb(images["agentview_image"])
        wrist = _decode_rgb(images["robot0_eye_in_hand_image"])
        combined = np.concatenate((agent, wrist), axis=1)
        frame_index = self.count
        composite_path = self.frames_dir / f"frame_{frame_index:06d}.jpg"
        agent_path = self.agent_dir / f"frame_{frame_index:06d}.jpg"
        wrist_path = self.wrist_dir / f"frame_{frame_index:06d}.jpg"
        for image, path in ((combined, composite_path), (agent, agent_path), (wrist, wrist_path)):
            if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError(f"camera_frame_write_failed:{path}")
        if self.video is None:
            height, width = combined.shape[:2]
            self.video = cv2.VideoWriter(
                str(self.output / "dual_camera_agentview_wrist_20fps.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height),
            )
            if not self.video.isOpened():
                raise OSError("dual_camera_video_writer_open_failed")
        self.video.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        monitor = copy.deepcopy(reply.get("contact_monitoring", {}))
        substeps = monitor.pop("substep_contacts", [])
        if not _monitoring_complete({"contact_monitoring": {**monitor, "substep_contacts": substeps}}):
            self.incomplete_monitor_actions += 1
        if strict.get("safety_certifiable_action") is True:
            self.strict_safe_actions += 1
        snapshot_state = reply.get("snapshot_state", {})
        row = {
            "schema": "gs_opd_oa_l0t0_candidate9_contact_aware_osc_step.v1",
            "branch_id": self.branch_id,
            "step_index": self.count,
            "timestep": reply.get("timestep"),
            "phase": phase,
            "control_mode": "LIBERO OSC_POSE MuJoCo physical action",
            "control_action_space": "7D normalized OSC_POSE delta [world xyz, world axis-angle xyz, gripper]",
            "pi05_native_action_compatible": True,
            "control_action": action.astype(np.float32).tolist(),
            "control_action_semantics": {
                "position_delta_world_m": (action[:3] * POSITION_OUTPUT_M).tolist(),
                "rotation_delta_world_axis_angle_rad": (action[3:6] * ROTATION_OUTPUT_RAD).tolist(),
                "gripper_normalized": float(action[6]),
            },
            "observation": {
                "scene_timestep": reply.get("timestep"),
                "state": reply.get("state"),
                "snapshot_state_sha256": _state_fingerprint(snapshot_state),
                "images": {
                    "agentview_image": str(agent_path),
                    "robot0_eye_in_hand_image": str(wrist_path),
                    "composite_dual_camera_frame": str(composite_path),
                },
            },
            "reward": reply.get("reward"),
            "done": reply.get("done"),
            "native_task_success": reply.get("task_success"),
            "snapshot_state": snapshot_state,
            "contact_monitoring": monitor,
            "substep_contacts": substeps,
            "strict_safety_audit": strict,
        }
        self.records.write(json.dumps(api._json_safe(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        self.records.flush()
        self.substep_count += len(substeps)
        self.count += 1

    def close(self) -> None:
        try:
            self.records.close()
        finally:
            if self.video is not None:
                self.video.release()


def _retreat_direction(style: str, support_normal: np.ndarray, eef_world: np.ndarray, bowl_world: np.ndarray, plate_center: np.ndarray) -> np.ndarray:
    radial = eef_world - bowl_world
    radial -= support_normal * float(np.dot(radial, support_normal))
    norm = float(np.linalg.norm(radial))
    if norm < 0.01:
        radial = eef_world - plate_center
        radial -= support_normal * float(np.dot(radial, support_normal))
        norm = float(np.linalg.norm(radial))
    if norm < 1e-8:
        radial = np.array([1.0, 0.0, 0.0])
        radial -= support_normal * float(np.dot(radial, support_normal))
        norm = float(np.linalg.norm(radial))
    radial /= norm
    if style == "vertical":
        direction = support_normal.copy()
    elif style == "radial_up":
        direction = 0.72 * support_normal + 0.69 * radial
    elif style == "cross_radial_up":
        tangent = np.cross(support_normal, radial)
        tangent /= np.linalg.norm(tangent)
        direction = 0.78 * support_normal + 0.63 * tangent
    else:
        raise ValueError(f"unknown_retreat_style:{style}")
    return direction / np.linalg.norm(direction)


def _action_for_delta(delta_world_m: np.ndarray, *, yaw_delta_rad: float = 0.0, gripper: float = 0.0) -> np.ndarray:
    delta = np.asarray(delta_world_m, dtype=np.float64)
    if delta.shape != (3,) or not np.isfinite(delta).all() or not np.isfinite(yaw_delta_rad):
        raise ValueError("OSC_delta_is_nonfinite_or_malformed")
    action = np.zeros(7, dtype=np.float64)
    action[:3] = delta / POSITION_OUTPUT_M
    action[5] = yaw_delta_rad / ROTATION_OUTPUT_RAD
    action[6] = float(gripper)
    if np.any(np.abs(action[:6]) > 1.0 + 1e-9) or not -1.0 <= action[6] <= 1.0:
        raise ValueError("requested_OSC_command_exceeds_normalized_controller_range")
    return action.astype(np.float32)


def _initial_branch_restore(bridge: _Bridge, payload: bytes, expected_state: np.ndarray, model_sha: str) -> dict[str, Any]:
    restored = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
    actual = np.asarray(restored.get("snapshot_state", {}).get("mujoco_integration_state", []), dtype=np.float64)
    if (
        restored.get("status") != "restored"
        or restored.get("model_geometry_sha256") != model_sha
        or actual.shape != expected_state.shape
        or not np.array_equal(actual, expected_state)
    ):
        raise RuntimeError(f"exact_branch_snapshot_restore_failed:{restored.get('error') or restored.get('status')}")
    controller_sync = bridge.request({"op": "sync_osc_controller_goal_to_live_state"})
    if (
        controller_sync.get("status") != "complete"
        or controller_sync.get("physical_state_unchanged") is not True
        or controller_sync.get("qpos_written") is not False
        or controller_sync.get("physics_advanced") is not False
    ):
        raise RuntimeError(f"live_OSC_goal_sync_failed:{controller_sync.get('error') or controller_sync.get('status')}")
    restored["osc_controller_goal_sync"] = controller_sync
    return restored


def _step(
    bridge: _Bridge,
    recorder: _PhysicalRecorder,
    action: np.ndarray,
    phase: str,
    safety_policy: dict[str, Any],
    action_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    reply = bridge.request({
        "op": "step_monitored",
        "action_index": action_count,
        "action": action.tolist(),
        "safety_geometry_policy": safety_policy,
    })
    monitor = reply.get("contact_monitoring", {})
    substep_rows = monitor.get("substep_contacts", [])
    strict = prior_task._strict_contacts({**reply, "substep_contacts": substep_rows})
    recorder.record(reply, action, phase, strict)
    native = bridge.request({"op": "native_task_status"})
    context = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
    grasp = bridge.request({"op": "grasp_state", "carried_binding": _step.binding, "expected_instruction": INSTRUCTION})
    if native.get("status") != "complete" or context.get("status") != "complete":
        raise RuntimeError("native_task_or_object_context_unavailable_after_physical_action")
    row = {
        "action_index": action_count,
        "timestep": reply.get("timestep"),
        "phase": phase,
        "native_task_status": native,
        "grasp_state": grasp,
        "eef_position_world_m": _eef_position(reply.get("snapshot_state", {})).tolist(),
        "bowl_root_position_world_m": _object_root_position(reply.get("snapshot_state", {}), OBJECT_ID).tolist(),
        "object_velocity": prior_task._object_velocity(reply.get("snapshot_state", {})),
        "contact_monitoring_complete": _monitoring_complete(reply),
        "strict_safety_audit": strict,
    }
    _step.rows.append(row)
    if action_count % 10 == 0:
        print(json.dumps({"t": reply.get("timestep"), "phase": phase, "task": native.get("task_success"), "grasp": grasp.get("grasp_confirmed"), "action": action_count, "monitor": _monitoring_complete(reply)}, ensure_ascii=False), flush=True)
    return reply, native, context, grasp


def _run_branch(
    branch: dict[str, Any],
    branch_dir: Path,
    bridge: _Bridge,
    source_payload: bytes,
    expected_state: np.ndarray,
    model_sha: str,
    semantic: dict[str, Any],
    collision_scene: dict[str, Any],
    inventory: dict[str, Any],
    binding: dict[str, Any],
    bowl_geom_ids: set[int],
    plate_geom_ids: set[int],
    robot_geom_ids: set[int],
) -> dict[str, Any]:
    branch_dir.mkdir(parents=True, exist_ok=False)
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    payload = source_payload
    branch_initial = _initial_branch_restore(bridge, payload, expected_state, model_sha)
    recorder = _PhysicalRecorder(branch_dir, str(branch["branch_id"]))
    setattr(_step, "binding", binding)
    setattr(_step, "rows", [])
    rows: list[dict[str, Any]] = _step.rows
    report: dict[str, Any] = {
        "schema": "gs_opd_oa_l0t0_candidate9_contact_aware_branch.v1",
        "branch_id": branch["branch_id"],
        "configuration": branch,
        "scope": "MuJoCo closed-loop physical Place -> Release -> Retreat -> Settling from one exact candidate9 held Snapshot",
        "engineering_choices_not_paper_values": True,
        "start_timestep": None,
        "physical_actions": 0,
        "physics_substeps": 0,
        "contact_monitoring_complete_actions": 0,
        "strict_safety_certifiable_actions": 0,
        "support_contact": {"observed": False, "native_on": False},
        "release": {"physical_release_confirmed": False, "fully_open": False},
        "retreat": {"complete": False},
        "settling": {"complete": False},
        "task_success": False,
        "safety_success": False,
        "new_trainable_episodes": 0,
        "pi05_native_action_compatible_suffix": True,
        "stage_snapshots": {},
        "osc_controller_goal_sync": branch_initial.get("osc_controller_goal_sync"),
        "stop_reason": None,
    }
    _write(branch_dir / "branch_report.json", report)
    action_count = 0
    native_success_rows: list[bool] = []
    branch_contact_rows: list[dict[str, Any]] = []
    last_reply: dict[str, Any] | None = None
    last_native: dict[str, Any] | None = None
    last_grasp: dict[str, Any] | None = None
    current_policy: dict[str, Any] | None = None
    release_transition_policy: dict[str, Any] | None = None
    released_policy: dict[str, Any] | None = None
    support_contact = False
    physical_release = False
    retreat_clear_actions = 0
    retreat_direction = None
    retreat_distance_after_release = 0.0
    release_eef_origin = None
    try:
        start_obs = bridge.request({"op": "observe", "include_snapshot_state": True})
        report["start_timestep"] = int(start_obs.get("timestep", -1))
        if report["start_timestep"] < 0:
            raise RuntimeError("restored_branch_timestep_unavailable")
        context = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
        current_native = bridge.request({"op": "native_task_status"})
        grasp = bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": INSTRUCTION})
        if context.get("status") != "complete" or current_native.get("status") != "complete":
            raise RuntimeError("initial_branch_context_unavailable")
        if grasp.get("status") != "pass" or grasp.get("grasp_confirmed") is not True:
            raise RuntimeError("source_snapshot_does_not_have_verified_bilateral_grasp")
        current_policy_raw = prior_task._build_policy(context, semantic, inventory)
        current_policy = prior._safety_policy_for_bridge(current_policy_raw)
        release_transition_policy = prior_task._released_safety_policy_for_bridge(current_policy_raw, release_transition=True)
        released_policy = prior_task._released_safety_policy_for_bridge(current_policy_raw)
        support = _support_frame(collision_scene, semantic["bindings"][REGION_ID])
        normal = np.asarray(support["normal_world"], dtype=np.float64)
        start_bowl = _object_root_position(start_obs["snapshot_state"], OBJECT_ID)
        plate_center = np.asarray(support["top_center_world_m"], dtype=np.float64)
        start_eef = _eef_position(start_obs["snapshot_state"])
        normal_component = normal * float(np.dot(plate_center - start_bowl, normal))
        requested_center_delta = float(branch["center_to_plate_fraction"]) * (plate_center - start_bowl - normal_component)
        max_center_shift = 0.008
        shift_norm = float(np.linalg.norm(requested_center_delta))
        if shift_norm > max_center_shift:
            requested_center_delta *= max_center_shift / shift_norm
        report["initial_state"] = {
            "native_task_status": current_native,
            "grasp_state": grasp,
            "eef_position_world_m": start_eef.tolist(),
            "bowl_root_position_world_m": start_bowl.tolist(),
            "plate_support_frame": support,
            "requested_in_plane_recenter_world_m": requested_center_delta.tolist(),
            "source_integration_state_exact_match": True,
        }
        report["safety_policy"] = {
            "existing_10mm_forbidden_object_and_2mm_held_target_rules_unchanged": True,
            "policy_source": current_policy_raw.get("source"),
            "policy_proof": current_policy_raw.get("_proof"),
            "task_feasibility_search_does_not_stop_on_strict_unsafe_margin_or_contact": True,
            "strict_audit_still_recorded_per_action": True,
        }
        report["stage_snapshots"]["branch_initial"] = recorder.snapshot(bridge, "branch_initial")
        _write(branch_dir / "branch_report.json", report)

        if abs(float(branch.get("yaw_delta_deg", 0.0))) > 1e-9:
            yaw_delta = math.radians(float(branch["yaw_delta_deg"]))
            action = _action_for_delta(np.zeros(3), yaw_delta_rad=yaw_delta, gripper=0.0)
            last_reply, last_native, _, last_grasp = _step(bridge, recorder, action, "pre_place_yaw_adjustment", current_policy, action_count)
            action_count += 1
            native_success_rows.append(last_native.get("task_success") is True)
            if last_grasp.get("grasp_confirmed") is not True:
                raise RuntimeError("grasp_lost_during_pre_place_yaw_adjustment")
            if not _monitoring_complete(last_reply):
                raise RuntimeError("contact_monitoring_incomplete_during_pre_place_yaw_adjustment")

        recenter_remaining = requested_center_delta.copy()
        recenter_nonprogress_actions = 0
        while np.linalg.norm(recenter_remaining) > 0.0004:
            if action_count >= MAX_BRANCH_ACTIONS:
                raise RuntimeError("branch_action_budget_before_descent")
            error_before = float(np.linalg.norm(recenter_remaining))
            norm = float(np.linalg.norm(recenter_remaining))
            recenter_step = float(branch.get("recenter_step_mm", 5.0)) / 1000.0
            if not (0.001 <= recenter_step <= 0.010):
                raise ValueError("recenter_step_outside_bounded_closed_loop_range")
            delta = recenter_remaining * min(1.0, recenter_step / norm)
            action = _action_for_delta(delta, gripper=0.0)
            last_reply, last_native, _, last_grasp = _step(bridge, recorder, action, "pre_place_recenter", current_policy, action_count)
            action_count += 1
            native_success_rows.append(last_native.get("task_success") is True)
            if last_grasp.get("grasp_confirmed") is not True:
                raise RuntimeError("grasp_lost_during_pre_place_recenter")
            if not _monitoring_complete(last_reply):
                raise RuntimeError("contact_monitoring_incomplete_during_pre_place_recenter")
            actual_bowl = _object_root_position(last_reply["snapshot_state"], OBJECT_ID)
            actual_planar_delta = actual_bowl - start_bowl
            actual_planar_delta -= normal * float(np.dot(actual_planar_delta, normal))
            recenter_remaining = requested_center_delta - actual_planar_delta
            error_after = float(np.linalg.norm(recenter_remaining))
            if error_after >= error_before - 0.00025:
                recenter_nonprogress_actions += 1
            else:
                recenter_nonprogress_actions = 0
            if recenter_nonprogress_actions >= 6:
                raise RuntimeError("closed_loop_recenter_controller_stalled_for_six_actions")
            if np.linalg.norm(actual_planar_delta) > max_center_shift + 0.004:
                raise RuntimeError("object_deviation_exceeded_bounded_recenter")

        yaw = float(branch.get("yaw_delta_deg", 0.0))
        if abs(yaw) > 1e-9:
            # The above single OSC command applies the explicitly recorded
            # angular delta. We do not assume the held object followed it;
            # current state remains observed in every subsequent step.
            report["orientation_adjustment"] = {"requested_yaw_delta_deg": yaw, "frame": "world support normal", "engineering_choice_not_paper_value": True}

        descent_step = float(branch["descent_step_mm"]) / 1000.0
        if not (0.001 <= descent_step <= 0.010):
            raise ValueError("descent_step_outside_bounded_contact_feedback_range")
        max_descent_steps = MAX_DESCENT_ACTIONS
        support_last_substeps: list[dict[str, Any]] = []
        descent_start_bowl = _object_root_position(bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"], OBJECT_ID)
        previous_eef = _eef_position(bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"])
        descent_travel_m = 0.0
        no_progress = 0
        for index in range(max_descent_steps):
            if action_count >= MAX_BRANCH_ACTIONS:
                raise RuntimeError("branch_action_budget_during_contact_descent")
            if descent_travel_m >= MAX_DESCENT_M:
                break
            delta = -normal * descent_step
            action = _action_for_delta(delta, gripper=0.0)
            last_reply, last_native, _, last_grasp = _step(bridge, recorder, action, "contact_feedback_descent", current_policy, action_count)
            action_count += 1
            native_success_rows.append(last_native.get("task_success") is True)
            if not _monitoring_complete(last_reply):
                raise RuntimeError("contact_monitoring_incomplete_during_contact_descent")
            if last_grasp.get("grasp_confirmed") is not True:
                raise RuntimeError("grasp_lost_before_plate_support_contact")
            current_eef = _eef_position(last_reply["snapshot_state"])
            current_bowl = _object_root_position(last_reply["snapshot_state"], OBJECT_ID)
            eef_progress = float(np.dot(previous_eef - current_eef, normal))
            if eef_progress < 0.0001:
                no_progress += 1
            else:
                no_progress = 0
            if no_progress >= 6:
                raise RuntimeError("closed_loop_descent_controller_stalled_for_six_actions")
            descent_travel_m += max(0.0, eef_progress)
            previous_eef = current_eef
            support_last_substeps = last_reply["contact_monitoring"]["substep_contacts"]
            actual_support = _substep_has_pair(support_last_substeps, bowl_geom_ids, plate_geom_ids)
            final_support = _substep_has_pair(support_last_substeps, bowl_geom_ids, plate_geom_ids, last_only=True)
            native_on = bool(last_native.get("task_success") is True)
            if actual_support or native_on:
                report["support_contact"] = {
                    "observed": bool(actual_support),
                    "native_on": native_on,
                    "any_support_contact_in_25_substeps": bool(actual_support),
                    "support_contact_on_final_substep": bool(final_support),
                    "timestep": last_reply.get("timestep"),
                    "bowl_root_position_world_m": current_bowl.tolist(),
                    "eef_position_world_m": current_eef.tolist(),
                    "descent_travel_m": descent_travel_m,
                    "observed_support_pairs": [
                        contact for substep in support_last_substeps
                        for contact in substep.get("contacts", [])
                        if _pair_matches(contact, bowl_geom_ids, plate_geom_ids)
                    ],
                }
                if not actual_support or not final_support or not native_on:
                    raise RuntimeError("support_contact_or_native_On_was_not_persistent_at_action_end")
                support_contact = True
                break
            bowl_delta = float(np.linalg.norm(current_bowl - descent_start_bowl))
            if bowl_delta > descent_travel_m + 0.012:
                raise RuntimeError("object_moved_without_corresponding_closed_loop_eef_motion_before_support")

        if not support_contact and descent_travel_m >= MAX_DESCENT_M:
            report["descent_limit"] = {"status": "complete", "actual_eef_travel_m": descent_travel_m, "maximum_actual_travel_m": MAX_DESCENT_M, "maximum_actions": MAX_DESCENT_ACTIONS}
        if not support_contact:
            raise RuntimeError("plate_support_not_reached_within_bounded_contact_descent")
        report["stage_snapshots"]["support_contact"] = recorder.snapshot(bridge, "support_contact")

        for hold_index in range(SUPPORT_HOLD_ACTIONS):
            action = _action_for_delta(np.zeros(3), gripper=0.0)
            last_reply, last_native, _, last_grasp = _step(bridge, recorder, action, "supported_hold_before_release", current_policy, action_count)
            action_count += 1
            native_success_rows.append(last_native.get("task_success") is True)
            if last_native.get("task_success") is not True:
                raise RuntimeError("native_On_lost_during_supported_hold")
            if not _monitoring_complete(last_reply):
                raise RuntimeError("contact_monitoring_incomplete_during_supported_hold")
            support_rows = last_reply.get("contact_monitoring", {}).get("substep_contacts", [])
            if not _substep_has_pair(support_rows, bowl_geom_ids, plate_geom_ids, last_only=True):
                raise RuntimeError("plate_support_contact_missing_during_supported_hold")
            if last_grasp.get("grasp_confirmed") is not True:
                raise RuntimeError("grasp_lost_before_coordinated_release_started")

        if int(branch.get("open_delay_actions", 0)) not in range(0, 4):
            raise ValueError("open_delay_actions_must_be_between_0_and_3")
        for delay_index in range(int(branch.get("open_delay_actions", 0))):
            action = _action_for_delta(np.zeros(3), gripper=-1.0)
            last_reply, last_native, _, last_grasp = _step(bridge, recorder, action, "release_open_delay_at_supported_pose", release_transition_policy, action_count)
            action_count += 1
            native_success_rows.append(last_native.get("task_success") is True)
            if last_native.get("task_success") is not True:
                raise RuntimeError("native_On_lost_during_supported_open_delay")
            if not _monitoring_complete(last_reply):
                raise RuntimeError("contact_monitoring_incomplete_during_supported_open_delay")

        live_obs = bridge.request({"op": "observe", "include_snapshot_state": True})
        current_eef = _eef_position(live_obs["snapshot_state"])
        current_bowl = _object_root_position(live_obs["snapshot_state"], OBJECT_ID)
        retreat_direction = _retreat_direction(
            str(branch["retreat_style"]), normal, current_eef, current_bowl, plate_center,
        )
        report["retreat"]["direction_world"] = retreat_direction.tolist()
        report["retreat"]["strategy"] = str(branch["retreat_style"])
        report["release"]["opening_schedule"] = "open command starts at observed bowl/plate support; every opening action simultaneously retreats along the branch direction"
        report["release"]["open_delay_actions"] = int(branch.get("open_delay_actions", 0))

        release_action_limit = min(140, MAX_BRANCH_ACTIONS - action_count - SETTLE_ACTIONS)
        if release_action_limit <= 0:
            raise RuntimeError("branch_action_budget_has_no_release_or_settling_reserve")
        retreat_previous_eef = _eef_position(live_obs["snapshot_state"])
        retreat_no_progress_actions = 0
        for release_i in range(release_action_limit):
            if action_count >= MAX_BRANCH_ACTIONS - SETTLE_ACTIONS:
                raise RuntimeError("branch_action_budget_during_coordinated_open_retreat")
            delta = retreat_direction * (float(branch["retreat_step_mm"]) / 1000.0)
            action = _action_for_delta(delta, gripper=-1.0)
            last_reply, last_native, _, last_grasp = _step(bridge, recorder, action, "coordinated_open_and_retreat", release_transition_policy if not physical_release else released_policy, action_count)
            action_count += 1
            native_success_rows.append(last_native.get("task_success") is True)
            if last_native.get("task_success") is not True:
                raise RuntimeError("native_On_lost_during_coordinated_open_retreat")
            if not _monitoring_complete(last_reply):
                raise RuntimeError("contact_monitoring_incomplete_during_coordinated_open_retreat")
            current_eef = _eef_position(last_reply["snapshot_state"])
            retreat_progress = float(np.dot(current_eef - retreat_previous_eef, retreat_direction))
            retreat_previous_eef = current_eef
            if retreat_progress < 0.0001:
                retreat_no_progress_actions += 1
            else:
                retreat_no_progress_actions = 0
            if retreat_no_progress_actions >= 6:
                raise RuntimeError("closed_loop_open_retreat_controller_stalled_for_six_actions")
            closure = last_grasp.get("gripper_closure", {}).get("close_command_fraction")
            if not physical_release:
                strict = prior_task._strict_contacts({**last_reply, "substep_contacts": last_reply["contact_monitoring"].get("substep_contacts", [])})
                if prior_task._physical_release_confirmed(last_grasp, strict, bowl_geom_ids, robot_geom_ids):
                    physical_release = True
                    release_eef_origin = current_eef.copy()
                    report["release"].update({
                        "physical_release_confirmed": True,
                        "timestep": last_reply.get("timestep"),
                        "grasp_state": last_grasp,
                        "confirmation_rule": "Bridge live grasp false, zero left/right pad contacts, and no robot/bowl contact at final substep of a complete 25-substep action",
                    })
                    report["stage_snapshots"]["release_confirmed"] = recorder.snapshot(bridge, "release_confirmed")
            else:
                retreat_distance_after_release += float(np.dot(current_eef - release_eef_origin, retreat_direction))
                final_contacts = last_reply["contact_monitoring"].get("substep_contacts", [])
                clear_now = not _substep_has_pair(final_contacts, bowl_geom_ids, robot_geom_ids, last_only=True)
                retreat_clear_actions = retreat_clear_actions + 1 if clear_now else 0
            is_fully_open = isinstance(closure, (int, float)) and float(closure) <= 0.05
            no_final_robot_contact = not _substep_has_pair(last_reply["contact_monitoring"].get("substep_contacts", []), bowl_geom_ids, robot_geom_ids, last_only=True)
            if physical_release and is_fully_open and no_final_robot_contact and retreat_distance_after_release >= RETREAT_CLEARANCE_M and retreat_clear_actions >= RETREAT_CLEAR_FINAL_ACTIONS:
                report["release"]["fully_open"] = True
                report["release"]["close_command_fraction"] = float(closure)
                report["retreat"].update({
                    "complete": True,
                    "retreat_distance_after_release_m": retreat_distance_after_release,
                    "retreat_clear_final_actions": retreat_clear_actions,
                    "final_robot_target_contact": False,
                    "terminal_timestep": last_reply.get("timestep"),
                })
                report["stage_snapshots"]["retreat_complete"] = recorder.snapshot(bridge, "retreat_complete")
                break
        if not physical_release:
            raise RuntimeError("bounded_open_retreat_budget_expired_without_physical_release")
        if report["retreat"].get("complete") is not True:
            raise RuntimeError("bounded_open_retreat_budget_expired_before_full_clearance")

        settle_rows = []
        settled_poses = []
        report["stage_snapshots"]["settling_start"] = recorder.snapshot(bridge, "settling_start")
        for settle_i in range(SETTLE_ACTIONS):
            if action_count >= MAX_BRANCH_ACTIONS:
                raise RuntimeError("branch_action_budget_before_settling_complete")
            action = _action_for_delta(np.zeros(3), gripper=-1.0)
            last_reply, last_native, _, last_grasp = _step(bridge, recorder, action, "settling", released_policy, action_count)
            action_count += 1
            native_success_rows.append(last_native.get("task_success") is True)
            if last_native.get("task_success") is not True:
                raise RuntimeError("native_On_lost_during_settling")
            monitor_rows = last_reply["contact_monitoring"].get("substep_contacts", [])
            if not _monitoring_complete(last_reply):
                raise RuntimeError("contact_monitoring_incomplete_during_settling")
            if _substep_has_pair(monitor_rows, bowl_geom_ids, robot_geom_ids, last_only=True):
                raise RuntimeError("robot_bowl_contact_reappeared_during_settling")
            ss = last_reply["snapshot_state"]
            velocity = prior_task._object_velocity(ss)
            pose = _object_root_position(ss, OBJECT_ID)
            settled_poses.append(pose)
            settle_rows.append({
                "timestep": last_reply.get("timestep"),
                "native_task_status": last_native,
                "object_velocity": velocity,
                "grasp_state": last_grasp,
                "final_substep_robot_bowl_contact": False,
                "strict_safety_audit": rows[-1]["strict_safety_audit"],
            })
        pose_array = np.asarray(settled_poses, dtype=np.float64)
        last_three_drift = float(np.max(np.linalg.norm(pose_array[-3:] - pose_array[-3], axis=1)))
        final_velocity = prior_task._object_velocity(last_reply["snapshot_state"])
        closure = last_grasp.get("gripper_closure", {}).get("close_command_fraction")
        final_support_rows = last_reply["contact_monitoring"].get("substep_contacts", [])
        final_support = _substep_has_pair(final_support_rows, bowl_geom_ids, plate_geom_ids, last_only=True)
        stable_release = bool(
            physical_release
            and report["release"].get("fully_open") is True
            and not _substep_has_pair(final_support_rows, bowl_geom_ids, robot_geom_ids, last_only=True)
            and final_support
            and isinstance(closure, (int, float)) and float(closure) <= 0.05
            and final_velocity.get("status") == "complete"
            and final_velocity.get("linear_speed_m_s", math.inf) <= STABLE_LINEAR_SPEED_M_S
            and final_velocity.get("angular_speed_rad_s", math.inf) <= STABLE_ANGULAR_SPEED_RAD_S
            and last_three_drift <= STABLE_LAST_THREE_DRIFT_M
            and all(row["native_task_status"].get("task_success") is True for row in settle_rows)
        )
        task_success = bool(last_native.get("task_success") is True)
        suffix_monitoring_complete = recorder.incomplete_monitor_actions == 0 and recorder.count > 0
        source_prefix_safety_certified = False
        suffix_strict_safety = bool(suffix_monitoring_complete and recorder.strict_safe_actions == recorder.count)
        report["settling"] = {
            "requested_actions": SETTLE_ACTIONS,
            "executed_actions": len(settle_rows),
            "complete": len(settle_rows) == SETTLE_ACTIONS,
            "actions": settle_rows,
            "last_three_pose_drift_m": last_three_drift,
            "object_velocity_after_last_action": final_velocity,
            "thresholds": {
                "linear_speed_m_s_max": STABLE_LINEAR_SPEED_M_S,
                "angular_speed_rad_s_max": STABLE_ANGULAR_SPEED_RAD_S,
                "last_three_pose_drift_m_max": STABLE_LAST_THREE_DRIFT_M,
            },
        }
        report["stable_release"] = stable_release
        report["task_success"] = bool(task_success and report["retreat"].get("complete") and report["settling"].get("complete"))
        report["safety_success"] = bool(source_prefix_safety_certified and suffix_strict_safety)
        report["safety_monitoring"] = {
            "suffix_contact_and_geometry_monitoring_complete": suffix_monitoring_complete,
            "suffix_strict_safety_certified": suffix_strict_safety,
            "strict_safe_actions": recorder.strict_safe_actions,
            "total_suffix_actions": recorder.count,
            "incomplete_monitor_actions": recorder.incomplete_monitor_actions,
            "full_prefix_safety_certified": source_prefix_safety_certified,
            "full_episode_safety_result": "false_due_to_uncertified_candidate9_prefix" if not source_prefix_safety_certified else "true",
            "10mm_2mm_rules_changed": False,
        }
        report["new_trainable_episodes"] = 0
        report["final_state"] = {
            "timestep": last_reply.get("timestep"),
            "native_task_status": last_native,
            "native_task_success": task_success,
            "stable_release": stable_release,
            "object_root_position_world_m": _object_root_position(last_reply["snapshot_state"], OBJECT_ID).tolist(),
            "eef_position_world_m": _eef_position(last_reply["snapshot_state"]).tolist(),
            "bowl_plate_support_contact_on_final_substep": final_support,
            "robot_bowl_contact_on_final_substep": False,
            "gripper_close_command_fraction": closure,
            "object_velocity": final_velocity,
            "last_three_settling_pose_drift_m": last_three_drift,
        }
        report["contact_summary"] = {
            "control_actions": recorder.count,
            "recorded_muJoCo_physics_substeps": recorder.substep_count,
            "expected_physics_substeps": recorder.count * SUBSTEPS,
            "monitoring_complete_actions": recorder.count - recorder.incomplete_monitor_actions,
            "strict_safety_certifiable_actions": recorder.strict_safe_actions,
            "observed_contact_pairs": _contact_pair_totals(rows),
        }
        report["physical_actions"] = recorder.count
        report["physics_substeps"] = recorder.substep_count
        report["contact_monitoring_complete_actions"] = recorder.count - recorder.incomplete_monitor_actions
        report["strict_safety_certifiable_actions"] = recorder.strict_safe_actions
        report["stage_snapshots"]["settling_complete"] = recorder.snapshot(bridge, "settling_complete")
        report["stop_reason"] = "complete_task_success" if report["task_success"] else "native_task_or_release_acceptance_failed"
    except Exception as exc:
        report["stop_reason"] = f"{type(exc).__name__}:{exc}"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        report["task_success"] = False
        report["safety_success"] = False
        report["new_trainable_episodes"] = 0
        if last_reply is not None:
            report["failure_state"] = {
                "timestep": last_reply.get("timestep"),
                "native_task_status": last_native,
                "grasp_state": last_grasp,
                "bowl_root_position_world_m": _object_root_position(last_reply["snapshot_state"], OBJECT_ID).tolist(),
                "eef_position_world_m": _eef_position(last_reply["snapshot_state"]).tolist(),
                "object_velocity": prior_task._object_velocity(last_reply["snapshot_state"]),
                "native_On": None if last_native is None else last_native.get("task_success"),
            }
    finally:
        try:
            report["physical_actions"] = recorder.count
            report["physics_substeps"] = recorder.substep_count
            report["contact_monitoring_complete_actions"] = recorder.count - recorder.incomplete_monitor_actions
            report["strict_safety_certifiable_actions"] = recorder.strict_safe_actions
            report["observed_contact_pairs"] = _contact_pair_totals(rows)
            report["terminal_snapshot"] = recorder.snapshot(bridge, "terminal")
            report["artifacts"] = {
                "physical_records": str(branch_dir / "physical_step_records.jsonl.gz"),
                "dual_camera_video": str(branch_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                "camera_frames": str(recorder.frames_dir),
                "snapshots": recorder.snapshot_paths,
            }
            report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _write(branch_dir / "branch_report.json", report)
        finally:
            recorder.close()
    return report


def _contact_pair_totals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for action in rows:
        monitor = action.get("strict_safety_audit", {})
        for contact in monitor.get("actual_contact_identities", []):
            a = contact.get("geom1", {})
            b = contact.get("geom2", {})
            ids = sorted((int(a.get("id", -1)), int(b.get("id", -1))))
            key = (ids[0], ids[1])
            item = pairs.setdefault(key, {
                "geom_ids": list(key),
                "geom_names": sorted((a.get("name"), b.get("name"))),
                "body_names": sorted((contact.get("body1", {}).get("name"), contact.get("body2", {}).get("name"))),
                "control_actions": set(),
                "substep_occurrences": 0,
                "deepest_signed_distance_m": None,
            })
            item["control_actions"].add(int(action.get("action_index", -1)))
            item["substep_occurrences"] += 1
            distance = contact.get("signed_contact_distance_m")
            if isinstance(distance, (int, float)) and (item["deepest_signed_distance_m"] is None or float(distance) < item["deepest_signed_distance_m"]):
                item["deepest_signed_distance_m"] = float(distance)
    out = []
    for _, item in sorted(pairs.items()):
        item["control_action_count"] = len(item.pop("control_actions"))
        out.append(item)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_scene_path = args.scene_source.resolve()
    snapshot_path = args.snapshot.resolve()
    source_report_path = args.source_report.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output_root_already_exists:{output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    source_scene = json.loads(gzip.decompress(source_scene_path.read_bytes()))
    payload = snapshot_path.read_bytes()
    source_identity = _snapshot_envelope(payload)
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    terminal = source_report.get("terminal_snapshot", {})
    if Path(terminal.get("path", "")).resolve() != snapshot_path or terminal.get("sha256") != _sha(payload) or int(terminal.get("timestep", -1)) != source_identity["timestep"]:
        raise ValueError("source_report_does_not_exactly_bind_terminal_snapshot_payload_and_timestep")
    if source_report.get("task_success") is not False and source_report.get("status") not in {"bounded_branch_stopped_or_blocked", "complete"}:
        raise ValueError("unexpected_source_report_status_for_held_snapshot")
    branches = [dict(row) for row in BRANCHES[:MAX_BRANCHES]]
    manifest: dict[str, Any] = {
        "schema": "gs_opd_oa_l0t0_candidate9_contact_aware_search.v1",
        "status": "initializing",
        "task": {"benchmark": "obstacle_avoidance", "level": 0, "task_id": 0, "object_id": OBJECT_ID, "region_id": REGION_ID, "native_predicate": ["on", OBJECT_ID, REGION_ID]},
        "source_snapshot": {"path": str(snapshot_path), "sha256": _sha(payload), "timestep": source_identity["timestep"], "model_geometry_sha256": MODEL_SHA},
        "source_snapshot_report": str(source_report_path),
        "historical_release_failure_evidence": {
            "t372_to_t417_native_On": True,
            "stationary_opening_left_eef_at_fixed_pose": True,
            "t417_bowl_plate_support_contact": True,
            "t418_final_substep_bowl_plate_support_contact": False,
            "t418_native_On": False,
            "observed_cause": "the stationary hand continued contacting/destabilizing the bowl during opening; the final MuJoCo bowl/plate contact disappeared, causing native On to fail",
            "selection_reason_for_t424_source": "exact candidate9 held transport state above the plate; unlike the marginal t372 release transition, bilateral grasp is still confirmed and it permits fresh real contact-feedback descent",
        },
        "limits": {
            "standard_horizon": HORIZON,
            "source_timestep": source_identity["timestep"],
            "remaining_standard_actions": HORIZON - source_identity["timestep"],
            "maximum_branches": len(branches),
            "maximum_actions_per_branch": MAX_BRANCH_ACTIONS,
            "muJoCo_substeps_per_action": SUBSTEPS,
            "settling_actions": SETTLE_ACTIONS,
            "contact_descent_distance_m_max": MAX_DESCENT_M,
        },
        "task_success": False,
        "safety_success": False,
        "new_trainable_episodes": 0,
        "task_feasibility_search": "continue through recorded strict-unsafe contacts while native On remains true and physical controller is progressing; stop on lost support, dropped grasp before support, control stall, or incomplete monitoring",
        "safety_policy": {"rules_changed": False, "forbidden_object_margin_m": 0.010, "held_target_margin_m": 0.002, "task_feasibility_does_not_certify_safety": True},
        "candidate_results": [],
    }
    _write(output_root / "search_report.json", manifest)

    process = log = connection = None
    try:
        process, log, connection, initial = api._start_bridge(source_scene, output_root)
        bridge = _Bridge(connection)
        if int(initial.get("timestep", -1)) != 0:
            raise RuntimeError("fresh_Bridge_did_not_start_at_zero")
        restored = _initial_branch_restore(bridge, payload, source_identity["state"], MODEL_SHA)
        restored_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
        if int(restored_observation.get("timestep", -1)) != source_identity["timestep"]:
            raise RuntimeError("exact_source_snapshot_restore_timestep_mismatch")
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        context = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
        collision_scene = bridge.request({"op": "collision_scene_export"})
        inventory = bridge.request({"op": "geometry_inventory"})
        if any(row.get("status") != "complete" for row in (semantic, context, collision_scene, inventory)):
            raise RuntimeError("live_semantic_bindings_or_collision_geometry_unavailable")
        target = semantic["bindings"].get(OBJECT_ID)
        plate = semantic["bindings"].get(REGION_ID)
        if not isinstance(target, dict) or not isinstance(plate, dict):
            raise RuntimeError("active_task_object_or_region_semantic_binding_missing")
        binding = {
            "object_id": OBJECT_ID,
            "body_id": int(target["root_body_id"]),
            "body_name": str(target["root_body_name"]),
            "geom_ids": sorted(int(value) for value in target["geom_ids"]),
            "geom_names": list(target["geom_names"]),
        }
        bowl_geom_ids = set(binding["geom_ids"])
        plate_geom_ids = set(int(value) for value in plate["geom_ids"])
        robot_geom_ids = {
            int(item["geom_id"])
            for item in inventory.get("geometries", [])
            if item.get("category") == "robot" and item.get("collision_enabled") is True
        }
        if not bowl_geom_ids or not plate_geom_ids or not robot_geom_ids:
            raise RuntimeError("complete_bowl_plate_robot_geom_bindings_required")
        report = {
            "status": "complete",
            "fresh_bridge_timestep": int(initial.get("timestep", -1)),
            "restore": {
                "status": restored.get("status"),
                "exact_integration_state_match": True,
                "timestep": int(restored_observation.get("timestep", -1)),
                "integration_state_sha256": hashlib.sha256(source_identity["state"].tobytes()).hexdigest(),
                "osc_controller_goal_sync": restored.get("osc_controller_goal_sync"),
            },
            "active_geom_bindings": {
                "bowl_body": binding["body_name"], "bowl_geom_ids": sorted(bowl_geom_ids),
                "plate_body": plate["root_body_name"], "plate_geom_ids": sorted(plate_geom_ids),
                "robot_collision_geom_count": len(robot_geom_ids),
            },
            "branch_results": [],
            "task_success": False,
            "safety_success": False,
            "new_trainable_episodes": 0,
        }
        _write(output_root / "environment_preflight.json", report)
        manifest["status"] = "running"
        _write(output_root / "search_report.json", manifest)
        for index, branch in enumerate(branches, start=1):
            branch_dir = output_root / f"branch_{index:02d}"
            print(f"Starting {branch['branch_id']} from exact source t={source_identity['timestep']}", flush=True)
            branch_report = _run_branch(
                branch, branch_dir, bridge, payload, source_identity["state"], MODEL_SHA,
                semantic, collision_scene, inventory, binding, bowl_geom_ids, plate_geom_ids, robot_geom_ids,
            )
            compact = {
                "branch_id": branch["branch_id"],
                "configuration": branch,
                "task_success": branch_report.get("task_success") is True,
                "native_task_success": branch_report.get("final_state", {}).get("native_task_success") is True,
                "physical_release_confirmed": branch_report.get("release", {}).get("physical_release_confirmed") is True,
                "stable_release": branch_report.get("stable_release") is True,
                "retreat_complete": branch_report.get("retreat", {}).get("complete") is True,
                "settling_complete": branch_report.get("settling", {}).get("complete") is True,
                "safety_success": branch_report.get("safety_success") is True,
                "new_trainable_episodes": branch_report.get("new_trainable_episodes", 0),
                "actions": branch_report.get("physical_actions", 0),
                "monitoring_complete_actions": branch_report.get("contact_monitoring_complete_actions", 0),
                "strict_safe_actions": branch_report.get("strict_safety_certifiable_actions", 0),
                "support_contact": branch_report.get("support_contact", {}),
                "stop_reason": branch_report.get("stop_reason"),
                "branch_report": str(branch_dir / "branch_report.json"),
                "terminal_snapshot": branch_report.get("terminal_snapshot"),
                "video": branch_report.get("artifacts", {}).get("dual_camera_video"),
                "physical_records": branch_report.get("artifacts", {}).get("physical_records"),
            }
            manifest["candidate_results"].append(compact)
            manifest["branches_evaluated"] = index
            if compact["task_success"]:
                manifest["task_success"] = True
                manifest["task_success_branch"] = branch["branch_id"]
                manifest["search_stop_reason"] = "native_task_success_after_stable_release_retreat_and_settling"
                _write(output_root / "search_report.json", manifest)
                break
            _write(output_root / "search_report.json", manifest)
        if not manifest.get("task_success"):
            manifest["search_stop_reason"] = "bounded_contact_aware_branch_budget_exhausted_without_complete_task_success"
        manifest["safety_success"] = False
        manifest["new_trainable_episodes"] = 0
        manifest["status"] = "complete" if any(x.get("actions", 0) for x in manifest["candidate_results"]) else "preflight_failed"
        manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write(output_root / "search_report.json", manifest)
        lines = [
            "# OA L0T0 candidate9 contact-aware placement search",
            "",
            f"- Source Snapshot timestep: `{source_identity['timestep']}`",
            f"- Branches evaluated: `{manifest.get('branches_evaluated', 0)}/{len(branches)}`",
            f"- Complete Task Success: `{manifest.get('task_success')}`",
            f"- Safety Success: `{manifest.get('safety_success')}`",
            f"- New trainable Episodes: `{manifest.get('new_trainable_episodes')}`",
            f"- Stop reason: `{manifest.get('search_stop_reason')}`",
            "",
        ]
        for row in manifest["candidate_results"]:
            lines.append(f"- `{row['branch_id']}`: task={row['task_success']}, release={row['physical_release_confirmed']}, retreat={row['retreat_complete']}, settling={row['settling_complete']}, actions={row['actions']}, stop=`{row['stop_reason']}`")
        (output_root / "search_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest
    except Exception as exc:
        manifest.update({
            "status": "preflight_failed",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "search_stop_reason": "preflight_failed_before_or_during_branch_search",
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        _write(output_root / "search_report.json", manifest)
        (output_root / "search_report.md").write_text(
            f"# OA L0T0 contact-aware placement search\n\nPreflight failed: `{type(exc).__name__}: {exc}`\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except Exception:
                    process.kill()
                    process.wait()
        if log is not None:
            log.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-source", type=Path, default=DEFAULT_SOURCE_SCENE)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({
        "output_root": str(args.output_root.resolve()),
        "branches_evaluated": report.get("branches_evaluated", 0),
        "task_success": report.get("task_success"),
        "safety_success": report.get("safety_success"),
        "new_trainable_episodes": report.get("new_trainable_episodes"),
        "search_stop_reason": report.get("search_stop_reason"),
    }, ensure_ascii=False), flush=True)
    return 0 if report.get("task_success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
