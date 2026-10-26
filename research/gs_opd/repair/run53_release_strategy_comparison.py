# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare run53 release and retreat strategies with complete per-substep evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.gs_opd.repair import task_success_first_grasp_search as grasp_search  # noqa: E402
from research.gs_opd.repair.run53_release_diagnostic import (  # noqa: E402
    EXPECTED_BDDL_SHA,
    EXPECTED_INSTRUCTION,
    EXPECTED_MODEL_SHA,
    RUN53,
    _native_values,
    _verified_resume_runtime_snapshot,
    contact_pair_names,
    read_step_rows,
    sha256_file,
)
from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action  # noqa: E402

PRIOR = Path("/opt/data/private/gs_opd/rollouts/run53_release_stability_diagnostic_20261026_150815_continued")
BRIDGE_SOURCE = Path(__file__).resolve().parents[1] / "smoke" / "safety_bridge.py"
SUBSTEPS = 25
SETTLE_STEPS = 20
OPEN = -1.0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def instrument_bridge_source(source: str) -> str:
    dispatch = '''                    monitor = _new_contact_monitor(request.get("safety_geometry_policy"))
                    obs, reward, done, info = _run_env_step_with_contact_monitor(
'''
    dispatch_new = '''                    monitor = _new_contact_monitor(request.get("safety_geometry_policy"))
                    monitor["_capture_physical_state_substeps"] = bool(request.get("capture_physical_state_substeps", False))
                    monitor["_physical_state_substeps"] = []
                    monitor["_physical_state_capture_errors"] = []
                    obs, reward, done, info = _run_env_step_with_contact_monitor(
'''
    hook = '''        geometry = _substep_safety_geometry(monitor, contacts)
        if geometry is not None and geometry.get("status") != "complete":
'''
    hook_new = '''        if monitor.get("_capture_physical_state_substeps"):
            try:
                monitor["_physical_state_substeps"].append({
                    "physics_substep_index": int(substep_index),
                    "simulation_time_s": float(sim.data.time),
                    "snapshot_state": _snapshot_diagnostics(canonicalize_kinematics=True),
                    "native_task_status": _native_task_goal_status(),
                    "contacts": contacts,
                })
            except Exception as exc:
                monitor["_physical_state_capture_errors"].append(
                    f"{type(exc).__name__}: {exc}"
                )
        geometry = _substep_safety_geometry(monitor, contacts)
        if geometry is not None and geometry.get("status") != "complete":
'''
    response = '''                    result["safety_geometry"] = monitor["geometry_substeps"][-1] if monitor.get("geometry_substeps") else None
'''
    response_new = '''                    result["safety_geometry"] = monitor["geometry_substeps"][-1] if monitor.get("geometry_substeps") else None
                    if monitor.get("_capture_physical_state_substeps"):
                        result["physical_state_substeps"] = monitor["_physical_state_substeps"]
                        result["physical_state_capture_errors"] = monitor["_physical_state_capture_errors"]
'''
    if source.count(dispatch) != 1 or source.count(hook) != 1 or source.count(response) != 1:
        raise ValueError("instrumentation_anchor_count_mismatch")
    result = source.replace(dispatch, dispatch_new, 1).replace(hook, hook_new, 1).replace(response, response_new, 1)
    if result == source:
        raise RuntimeError("bridge_instrumentation_patch_did_not_change_source")
    return result


def _load_original_suffix() -> list[dict[str, Any]]:
    rows = read_step_rows(PRIOR / "diagnostic_physical_step_records.jsonl")
    chosen = {int(row["timestep"]): row for row in rows if int(row.get("timestep", -1)) in range(1336, 1369)}
    if sorted(chosen) != list(range(1336, 1369)):
        raise ValueError("original_release_suffix_missing_t1336_to_t1368")
    return [chosen[t] for t in range(1336, 1369)]


def region_inward_shift_xy(
    region_center_world: Any,
    object_position_world: Any,
    distance_m: float,
) -> np.ndarray:
    """Return a bounded XY correction toward a live native region center."""
    center = np.asarray(region_center_world, dtype=np.float64)
    object_position = np.asarray(object_position_world, dtype=np.float64)
    if center.shape != (3,) or object_position.shape != (3,):
        raise ValueError("region_and_object_positions_must_be_3d")
    if not np.isfinite(center).all() or not np.isfinite(object_position).all():
        raise ValueError("region_and_object_positions_must_be_finite")
    if not np.isfinite(distance_m) or distance_m <= 0 or distance_m > 0.05:
        raise ValueError("placement_correction_must_be_in_(0,50mm]")
    direction = center[:2] - object_position[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        raise ValueError("object_xy_already_at_region_center")
    return direction / norm * float(distance_m)


def release_retreat_targets(
    eef_position_world: Any,
    object_position_world: Any,
    style: str,
    lateral_distance_m: float = 0.04,
    upward_distance_m: float = 0.10,
) -> list[np.ndarray]:
    """Build release-only retreat waypoints; lateral retreat moves away from the released object first."""
    eef = np.asarray(eef_position_world, dtype=np.float64)
    object_position = np.asarray(object_position_world, dtype=np.float64)
    if eef.shape != (3,) or object_position.shape != (3,):
        raise ValueError("retreat_positions_must_be_3d")
    if not np.isfinite(eef).all() or not np.isfinite(object_position).all():
        raise ValueError("retreat_positions_must_be_finite")
    if not np.isfinite(lateral_distance_m) or lateral_distance_m <= 0 or lateral_distance_m > 0.08:
        raise ValueError("lateral_retreat_distance_must_be_in_(0,80mm]")
    if not np.isfinite(upward_distance_m) or upward_distance_m <= 0 or upward_distance_m > 0.15:
        raise ValueError("upward_retreat_distance_must_be_in_(0,150mm]")
    if style == "upward":
        return [eef + np.asarray([0.0, 0.0, upward_distance_m])]
    if style != "lateral_up":
        raise ValueError(f"unsupported_release_retreat_style:{style}")
    away_xy = eef[:2] - object_position[:2]
    norm = float(np.linalg.norm(away_xy))
    if norm <= 1e-6:
        raise ValueError("cannot_derive_lateral_retreat_direction")
    lateral = eef.copy()
    lateral[:2] += away_xy / norm * lateral_distance_m
    upward = lateral + np.asarray([0.0, 0.0, upward_distance_m])
    return [lateral, upward]


def _pot2_hand_contact(contact_result: dict[str, Any]) -> bool:
    for contact in contact_result.get("contacts", []):
        names = (contact.get("geom1", {}).get("name", ""), contact.get("geom2", {}).get("name", ""))
        has_pot2 = any(name.startswith("moka_pot_2_g") for name in names)
        has_gripper = any("gripper" in name.lower() for name in names)
        if has_pot2 and has_gripper:
            return True
    return False


def stable_release_candidate(row: dict[str, Any]) -> bool:
    """Require complete native success, full retreat, clear settling, and stable object motion."""
    adjustment = row.get("placement_adjustment") or {}
    predicates = row.get("final_native_predicates") or {}
    return bool(
        row.get("status") == "complete"
        and row.get("substep_state_capture_complete") is True
        and row.get("final_native_task_success") is True
        and all(predicates.get(name) is True for name in ("pot1_on", "pot2_on", "stove_on"))
        and row.get("final_hand_pot2_contact") is False
        and row.get("all_native_predicates_true_at_every_settle_substep") is True
        and row.get("all_settle_substeps_hand_pot2_contact_free") is True
        and row.get("pot2_passes_existing_settle_limits") is True
        and row.get("withdrawal_displacement_m", 0.0) >= 0.05
        and adjustment.get("retreat_target_remaining_m", 1.0) <= 0.015
)

def _artifact_hashes(row: dict[str, Any]) -> dict[str, str]:
    paths = dict(row.get("artifacts") or {})
    for label in ("durable_action_path", "durable_substep_path"):
        if row.get(label):
            paths[label] = row[label]
    hashes = {}
    for label, value in paths.items():
        path = Path(value)
        if path.is_file():
            hashes[label] = sha256_file(path)
    return hashes


def _save_snapshot(bridge: Any, path: Path) -> dict[str, Any]:
    result = bridge.request({"op": "export_runtime_snapshot"})
    payload = result.get("payload")
    if result.get("status") != "complete" or not isinstance(payload, bytes):
        raise RuntimeError(f"snapshot_export_failed:{result.get('error') or result.get('status')}")
    path.write_bytes(payload)
    return {
        "path": str(path),
        "payload_sha256": sha256_file(path),
        "integration_state_sha256": hashlib.sha256(
            np.asarray(result["snapshot_state"]["mujoco_integration_state"], dtype=np.float64).tobytes()
        ).hexdigest(),
        "timestep": result.get("snapshot_state", {}).get("task_bookkeeping", {}).get("timestep"),
    }


def _video_frame(bridge: Any, branch_dir: Path, writer: Any, frame_index: list[dict[str, Any]], action_index: int, phase: str) -> Any:
    from research.gs_opd.repair.task_success_first_grasp_search import _image

    reply = bridge.request({"op": "observe"})
    agent = _image(reply, "agentview_image")
    wrist = _image(reply, "robot0_eye_in_hand_image")
    if writer is None:
        writer = cv2.VideoWriter(
            str(branch_dir / "dual_camera_release_branch_20fps.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
            (agent.shape[1] + wrist.shape[1], agent.shape[0]),
        )
        if not writer.isOpened():
            raise RuntimeError("video_writer_failed")
    writer.write(cv2.cvtColor(np.concatenate((agent, wrist), axis=1), cv2.COLOR_RGB2BGR))
    frame_index.append({"frame": len(frame_index), "action_index": action_index, "phase": phase})
    return writer


def _run_branch(bridge_script: Path, payload: bytes, expected_state: np.ndarray, output: Path,
                branch_id: str, strategy: str, original_suffix: list[dict[str, Any]],
                slow_step_m: float, slow_limit: int, retreat_style: str = "upward") -> dict[str, Any]:
    branch_dir = output / branch_id
    branch_dir.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(
        benchmark="obstacle_avoidance", level=1, task_id=2, init_state=0, seed=0,
        camera_resolution=256, demonstration_horizon=2600,
    )
    bridge = grasp_search.BridgeSession(("127.0.0.1", _port()), branch_dir / "safety_bridge.log", args)
    old_script = grasp_search.BRIDGE_SCRIPT
    writer = None
    frames: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    settled_endpoint_rows: list[dict[str, Any]] = []
    settled_substep_rows: list[dict[str, Any]] = []
    settled_substep_native: list[dict[str, bool | None]] = []
    all_substeps_complete = True
    action_index = 0
    release_free_timestep = None
    placement_adjustment: dict[str, Any] | None = None
    retreat_origin_eef: np.ndarray | None = None
    try:
        grasp_search.BRIDGE_SCRIPT = bridge_script
        bridge.start()
        reset = bridge.request({"op": "reset"})
        metadata = reset.get("scene_metadata", {})
        if metadata.get("model_geometry_sha256") != EXPECTED_MODEL_SHA or metadata.get("bddl_sha256") != EXPECTED_BDDL_SHA:
            raise RuntimeError("reset_model_or_bddl_identity_mismatch")
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
        start_state = np.asarray(restored.get("snapshot_state", {}).get("mujoco_integration_state", []), dtype=np.float64)
        if restored.get("status") != "restored" or not np.array_equal(start_state, expected_state):
            raise RuntimeError("t1335_snapshot_restore_not_exact")
        native_start = bridge.request({"op": "native_task_status"})
        if native_start.get("task_success") is not True or any(v is not True for v in _native_values(native_start).values()):
            raise RuntimeError(f"branch_start_not_native_success:{native_start}")
        start_export = _save_snapshot(bridge, branch_dir / "branch_start_reexport.pkl")
        (branch_dir / "branch_start_t1335_source_snapshot.pkl").write_bytes(payload)
        _append(branch_dir / "control_actions.jsonl", {
            "event": "branch_start",
            "source_timestep": 1335,
            "native_task_status": native_start,
            "integration_state_sha256": hashlib.sha256(np.ascontiguousarray(start_state).tobytes()).hexdigest(),
            "restored_snapshot": start_export,
        })
        writer = _video_frame(bridge, branch_dir, writer, frames, -1, "t1335_native_success")
        controller_fields = bridge.request({"op": "repair_context"}).get("robots", [{}])[0].get("fields")
        if not isinstance(controller_fields, dict):
            raise RuntimeError("OSC_POSE_controller_fields_unavailable")
        initial_contacts = bridge.request({"op": "inspect_contacts"})
        def step(action: np.ndarray, phase: str, expected_prior: dict[str, Any] | None = None) -> dict[str, Any]:
            nonlocal action_index, writer, all_substeps_complete, release_free_timestep
            reply = bridge.request({
                "op": "step_monitored",
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "action_index": action_index,
                "capture_physical_state_substeps": True,
            })
            if reply.get("error"):
                raise RuntimeError(f"step_monitored_error:{reply['error']}")
            samples = reply.get("physical_state_substeps", [])
            monitor = reply.get("contact_monitoring", {})
            expected = monitor.get("expected_physics_substeps_per_control_step")
            captured = monitor.get("captured_physics_substeps")
            capture_errors = reply.get("physical_state_capture_errors", [])
            complete = len(samples) == SUBSTEPS and expected == SUBSTEPS and captured == SUBSTEPS and not capture_errors
            all_substeps_complete &= complete
            if not complete:
                raise RuntimeError(
                    "substep_capture_incomplete:"
                    f"expected={expected},captured={captured},rows={len(samples)},"
                    f"capture_errors={capture_errors}"
                )
            for sample in samples:
                snap = sample["snapshot_state"]
                if phase in {"original_settle", "hold_settle", "slow_settle", "placement_settle"}:
                    settled_substep_rows.append(snap["object_states"]["objects_dict:moka_pot_2"])
                    settled_substep_native.append(_native_values(sample["native_task_status"]))
                _append(branch_dir / "physics_substeps.jsonl", {
                    "branch_id": branch_id,
                    "action_index": action_index,
                    "phase": phase,
                    "control_timestep": reply.get("timestep"),
                    "physics_substep_index": sample["physics_substep_index"],
                    "simulation_time_s": sample["simulation_time_s"],
                    "robot_state": snap["robot_states"][0],
                    "pot1_state": snap["object_states"].get("objects_dict:moka_pot_1"),
                    "pot2_state": snap["object_states"].get("objects_dict:moka_pot_2"),
                    "native_task_status": sample["native_task_status"],
                    "contacts": sample["contacts"],
                })
            endpoint_status = bridge.request({"op": "native_task_status"})
            endpoint_contacts = bridge.request({"op": "inspect_contacts"})
            endpoint = reply.get("snapshot_state", {})
            row = {
                "branch_id": branch_id,
                "action_index": action_index,
                "phase": phase,
                "action_7d": np.asarray(action, dtype=np.float32).tolist(),
                "timestep": reply.get("timestep"),
                "reward": reply.get("reward"),
                "done": reply.get("done"),
                "native_task_status": endpoint_status,
                "robot_state": endpoint.get("robot_states", [{}])[0],
                "pot1_state": endpoint.get("object_states", {}).get("objects_dict:moka_pot_1"),
                "pot2_state": endpoint.get("object_states", {}).get("objects_dict:moka_pot_2"),
                "contacts": endpoint_contacts,
                "substep_count": len(samples),
                "substep_contact_monitoring": monitor,
                "safety_certified": False,
            }
            _append(branch_dir / "control_actions.jsonl", row)
            action_rows.append(row)
            writer = _video_frame(bridge, branch_dir, writer, frames, action_index, phase)
            action_index += 1
            if not _pot2_hand_contact(endpoint_contacts) and release_free_timestep is None:
                release_free_timestep = int(endpoint_status.get("timestep", -1))
            if phase in {"original_settle", "hold_settle", "slow_settle", "placement_settle"}:
                settled_endpoint_rows.append(row)
            if expected_prior is not None:
                prior_state = np.asarray(expected_prior["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
                actual_state = np.asarray(endpoint.get("mujoco_integration_state", []), dtype=np.float64)
                row["baseline_exact_state_match"] = bool(np.array_equal(prior_state, actual_state))
                row["baseline_max_abs_state_delta"] = float(np.max(np.abs(prior_state - actual_state)))
                _append(branch_dir / "baseline_replay_comparison.jsonl", {
                    "timestep": row["timestep"],
                    "exact": row["baseline_exact_state_match"],
                    "max_abs_delta": row["baseline_max_abs_state_delta"],
                })
            return row

        if strategy == "original_retreat":
            for old_row in original_suffix:
                phase = {
                    "release": "original_release",
                    "retreat": "original_retreat",
                    "settle": "original_settle",
                }[old_row["phase"]]
                step(np.asarray(old_row["action_7d"], dtype=np.float32), phase, expected_prior=old_row)
                if phase == "original_release":
                    _save_snapshot(bridge, branch_dir / "after_release_runtime_snapshot.pkl")
                    retreat_origin_eef = np.asarray(
                        bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]["eef_pos"],
                        dtype=np.float64,
                    )
                if phase == "original_retreat" and len([r for r in action_rows if r["phase"] == "original_retreat"]) == 12:
                    _save_snapshot(bridge, branch_dir / "after_retreat_runtime_snapshot.pkl")
                if phase == "original_settle" and len(settled_endpoint_rows) == SETTLE_STEPS:
                    _save_snapshot(bridge, branch_dir / "after_settling_runtime_snapshot.pkl")
        elif strategy in {"hold_after_full_release", "slow_upward_retreat"}:
            release_attempts = 0
            free_count = 0
            for _ in range(12):
                row = step(np.asarray([0, 0, 0, 0, 0, 0, OPEN], dtype=np.float32), "stationary_open_release")
                release_attempts += 1
                if _pot2_hand_contact(row["contacts"]):
                    free_count = 0
                else:
                    free_count += 1
                if free_count >= 2:
                    break
            if free_count < 2:
                raise RuntimeError("could_not_confirm_two_consecutive_contact_free_release_steps")
            _save_snapshot(bridge, branch_dir / "after_full_release_runtime_snapshot.pkl")
            retreat_origin_eef = np.asarray(
                bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]["eef_pos"],
                dtype=np.float64,
            )
            if strategy == "hold_after_full_release":
                for _ in range(SETTLE_STEPS):
                    step(np.asarray([0, 0, 0, 0, 0, 0, OPEN], dtype=np.float32), "hold_settle")
                _save_snapshot(bridge, branch_dir / "after_settling_runtime_snapshot.pkl")
            elif strategy == "slow_upward_retreat":
                current_robot = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]
                release_eef = np.asarray(current_robot["eef_pos"], dtype=np.float64)
                release_quat = np.asarray(current_robot["eef_quat"], dtype=np.float64)
                target = release_eef + np.asarray([0.0, 0.0, 0.10])
                for _ in range(slow_limit):
                    robot = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]
                    pos = np.asarray(robot["eef_pos"], dtype=np.float64)
                    quat = np.asarray(robot["eef_quat"], dtype=np.float64)
                    delta = target - pos
                    distance = float(np.linalg.norm(delta))
                    if distance <= 0.008:
                        break
                    waypoint = pos + delta * min(1.0, slow_step_m / max(distance, 1e-9))
                    action = world_pose_tracking_action(
                        pos, quat, waypoint, release_quat, controller_fields,
                        gripper_action=OPEN,
                    )
                    step(action, "slow_upward_retreat")
                _save_snapshot(bridge, branch_dir / "after_retreat_runtime_snapshot.pkl")
                for _ in range(SETTLE_STEPS):
                    step(np.asarray([0, 0, 0, 0, 0, 0, OPEN], dtype=np.float32), "slow_settle")
                _save_snapshot(bridge, branch_dir / "after_settling_runtime_snapshot.pkl")

        elif strategy.startswith("placement_offset_"):
            try:
                offset_mm = float(strategy.removeprefix("placement_offset_").removesuffix("mm"))
            except ValueError as exc:
                raise ValueError(f"invalid_placement_offset_strategy:{strategy}") from exc
            region = bridge.request({
                "op": "native_goal_query",
                "predicate": ["on", "moka_pot_2", "flat_stove_1_cook_region"],
            })
            if region.get("status") != "complete" or region.get("predicate_value") is not True:
                raise RuntimeError(f"live_native_region_query_failed:{region}")
            start_snapshot = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
            start_robot = start_snapshot["robot_states"][0]
            start_pot = start_snapshot["object_states"]["objects_dict:moka_pot_2"]
            region_center = np.asarray(region["region"]["site_position_world_m"], dtype=np.float64)
            pot_position = np.asarray(start_pot["root_pos"], dtype=np.float64)
            shift_xy = region_inward_shift_xy(region_center, pot_position, offset_mm / 1000.0)
            eef_start = np.asarray(start_robot["eef_pos"], dtype=np.float64)
            eef_quat = np.asarray(start_robot["eef_quat"], dtype=np.float64)
            eef_shift = np.asarray([shift_xy[0], shift_xy[1], 0.0], dtype=np.float64)
            targets = [
                eef_start + np.asarray([0.0, 0.0, 0.008]),
                eef_start + np.asarray([0.0, 0.0, 0.008]) + eef_shift,
                eef_start + eef_shift,
            ]
            phases = ["placement_lift_8mm", "placement_translate_inward", "placement_lower_to_contact"]
            placement_motion_actions = 0
            for target_pos, phase in zip(targets, phases, strict=True):
                reached = False
                for _ in range(16):
                    live_robot = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]
                    current_pos = np.asarray(live_robot["eef_pos"], dtype=np.float64)
                    current_quat = np.asarray(live_robot["eef_quat"], dtype=np.float64)
                    delta = target_pos - current_pos
                    distance = float(np.linalg.norm(delta))
                    if distance <= 0.003:
                        reached = True
                        break
                    waypoint = current_pos + delta * min(1.0, 0.008 / max(distance, 1e-9))
                    action = world_pose_tracking_action(
                        current_pos, current_quat, waypoint, eef_quat,
                        controller_fields, gripper_action=1.0,
                    )
                    step(action, phase)
                    placement_motion_actions += 1
                if not reached:
                    live_robot = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]
                    if float(np.linalg.norm(np.asarray(live_robot["eef_pos"]) - target_pos)) > 0.008:
                        raise RuntimeError(f"placement_waypoint_not_reached:{phase}")

            _save_snapshot(bridge, branch_dir / "adjusted_placement_held_runtime_snapshot.pkl")
            seated_contacts = bridge.request({"op": "inspect_contacts"})
            seated_status = bridge.request({"op": "native_task_status"})
            seated_state = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
            seated_pot = seated_state["object_states"]["objects_dict:moka_pot_2"]
            placement_adjustment = {
                "native_region_query": region,
                "requested_shift_toward_region_center_xy_m": eef_shift[:2].tolist(),
                "requested_shift_norm_m": float(np.linalg.norm(eef_shift[:2])),
                "actual_pot2_shift_xy_m": (
                    np.asarray(seated_pot["root_pos"], dtype=np.float64)[:2] - pot_position[:2]
                ).tolist(),
                "placement_motion_actions": placement_motion_actions,
                "pot2_still_in_gripper_contact": _pot2_hand_contact(seated_contacts),
                "native_status_before_release": seated_status,
                "pot2_pose_before_release": seated_pot,
                "contact_pairs_before_release": sorted([list(pair) for pair in contact_pair_names(seated_contacts)]),
            }
            if not _pot2_hand_contact(seated_contacts):
                raise RuntimeError("Pot2_lost_gripper_contact_during_local_placement_adjustment")
            _save_snapshot(bridge, branch_dir / "after_contact_seating_runtime_snapshot.pkl")

            release_attempts = 0
            free_count = 0
            for _ in range(12):
                row = step(np.asarray([0, 0, 0, 0, 0, 0, OPEN], dtype=np.float32), "placement_release")
                release_attempts += 1
                if _pot2_hand_contact(row["contacts"]):
                    free_count = 0
                else:
                    free_count += 1
                if free_count >= 2:
                    break
            if free_count < 2:
                raise RuntimeError("adjusted_placement_release_not_contact_free_after_12_actions")
            _save_snapshot(bridge, branch_dir / "after_full_release_runtime_snapshot.pkl")

            live_robot = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]
            retreat_origin = np.asarray(live_robot["eef_pos"], dtype=np.float64)
            retreat_origin_eef = retreat_origin.copy()
            retreat_quat = np.asarray(live_robot["eef_quat"], dtype=np.float64)
            released_pot = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["object_states"]["objects_dict:moka_pot_2"]
            retreat_targets = release_retreat_targets(
                retreat_origin, released_pot["root_pos"], retreat_style,
            )
            retreat_actions = 0
            for retreat_index, retreat_target in enumerate(retreat_targets):
                target_reached = False
                for _ in range(slow_limit):
                    live_robot = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]
                    current_pos = np.asarray(live_robot["eef_pos"], dtype=np.float64)
                    current_quat = np.asarray(live_robot["eef_quat"], dtype=np.float64)
                    delta = retreat_target - current_pos
                    distance = float(np.linalg.norm(delta))
                    if distance <= 0.008:
                        target_reached = True
                        break
                    waypoint = current_pos + delta * min(1.0, slow_step_m / max(distance, 1e-9))
                    action = world_pose_tracking_action(
                        current_pos, current_quat, waypoint, retreat_quat,
                        controller_fields, gripper_action=OPEN,
                    )
                    phase = "placement_lateral_retreat" if retreat_style == "lateral_up" and retreat_index == 0 else "placement_slow_retreat"
                    step(action, phase)
                    retreat_actions += 1
                if not target_reached:
                    live_robot = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]
                    if float(np.linalg.norm(np.asarray(live_robot["eef_pos"], dtype=np.float64) - retreat_target)) > 0.015:
                        raise RuntimeError("placement_retreat_waypoint_not_reached")
            placement_adjustment["retreat_style"] = retreat_style
            placement_adjustment["release_eef_position_m"] = retreat_origin.tolist()
            placement_adjustment["release_open_actions"] = release_attempts
            placement_adjustment["slow_retreat_actions"] = retreat_actions
            current_eef = np.asarray(
                bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]["eef_pos"],
                dtype=np.float64,
            )
            placement_adjustment["retreat_target_remaining_m"] = float(np.linalg.norm(retreat_target - current_eef))
            _save_snapshot(bridge, branch_dir / "after_retreat_runtime_snapshot.pkl")
            for _ in range(SETTLE_STEPS):
                step(np.asarray([0, 0, 0, 0, 0, 0, OPEN], dtype=np.float32), "placement_settle")
            _save_snapshot(bridge, branch_dir / "after_settling_runtime_snapshot.pkl")

        final_status = bridge.request({"op": "native_task_status"})
        final_contacts = bridge.request({"op": "inspect_contacts"})
        final = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
        _save_snapshot(bridge, branch_dir / "final_runtime_snapshot.pkl")
        pot2_positions = np.asarray([row["root_pos"] for row in settled_substep_rows], dtype=np.float64)
        pot2_vel = np.asarray([row["joint_qvel"] for row in settled_substep_rows], dtype=np.float64)
        if len(pot2_positions):
            pos_span = float(np.max(np.linalg.norm(pot2_positions - pot2_positions[0], axis=1)))
            linear_speed = np.linalg.norm(pot2_vel[:, :3], axis=1)
            angular_speed = np.linalg.norm(pot2_vel[:, 3:], axis=1)
            max_linear = float(np.max(linear_speed))
            max_angular = float(np.max(angular_speed))
        else:
            pos_span = max_linear = max_angular = None
        native_settle_true = bool(settled_endpoint_rows) and all(
            all(value is True for value in _native_values(row["native_task_status"]).values())
            for row in settled_endpoint_rows
        )
        native_settle_substeps_true = bool(settled_substep_native) and all(
            all(value is True for value in values.values())
            for values in settled_substep_native
        )
        final_pot2 = final["object_states"]["objects_dict:moka_pot_2"]["root_pos"]
        final_eef = np.asarray(final["robot_states"][0]["eef_pos"], dtype=np.float64)
        displacement = float(np.linalg.norm(final_eef - retreat_origin_eef)) if retreat_origin_eef is not None else 0.0
        retreat_contact_events = []
        settle_hand_contact_samples = 0
        native_predicate_false_substeps_by_phase: dict[str, int] = {}
        if placement_adjustment is not None:
            substep_path = branch_dir / "physics_substeps.jsonl"
            for line in substep_path.open(encoding="utf-8"):
                sample = json.loads(line)
                false_predicates = [
                    value["predicate"][0]
                    for value in sample["native_task_status"].get("native_goal_values", [])
                    if value.get("value") is not True
                ]
                if false_predicates:
                    native_predicate_false_substeps_by_phase[sample["phase"]] = (
                        native_predicate_false_substeps_by_phase.get(sample["phase"], 0) + 1
                    )
                pot_contacts = []
                for contact in sample["contacts"].get("contacts", []):
                    names = [contact.get("geom1", {}).get("name", ""), contact.get("geom2", {}).get("name", "")]
                    if any(name.startswith("moka_pot_2_g") for name in names) and any("gripper" in name.lower() for name in names):
                        pot_contacts.append(names)
                if sample["phase"] == "placement_settle" and pot_contacts:
                    settle_hand_contact_samples += 1
                if pot_contacts and sample["phase"] in {"placement_lateral_retreat", "placement_slow_retreat"}:
                    retreat_contact_events.append({
                        "control_timestep": sample["control_timestep"],
                        "physics_substep_index": sample["physics_substep_index"],
                        "geom_pairs": pot_contacts,
                    })
        result = {
            "branch_id": branch_id,
            "strategy": strategy,
            "status": "complete",
            "restored_same_t1335_source_snapshot": True,
            "source_snapshot_sha256": hashlib.sha256(payload).hexdigest(),
            "start_integration_state_sha256": hashlib.sha256(np.ascontiguousarray(start_state).tobytes()).hexdigest(),
            "reexported_start_snapshot": start_export,
            "actions": len(action_rows),
            "physics_substeps_expected": len(action_rows) * SUBSTEPS,
            "physics_substeps_captured": sum(row["substep_count"] for row in action_rows),
            "substep_state_capture_complete": all_substeps_complete,
            "contact_free_release_timestep": release_free_timestep,
            "withdrawal_displacement_m": displacement,
            "retreat_substep_pot2_gripper_contact_samples": len(retreat_contact_events),
            "retreat_substep_pot2_gripper_contact_events": retreat_contact_events,
            "settle_substep_pot2_gripper_contact_samples": settle_hand_contact_samples,
            "all_settle_substeps_hand_pot2_contact_free": settle_hand_contact_samples == 0,
            "native_predicate_false_substeps_by_phase": native_predicate_false_substeps_by_phase,
            "final_eef_to_pot2_distance_m": float(np.linalg.norm(final_eef - np.asarray(final_pot2, dtype=np.float64))),
            "final_hand_pot2_contact": _pot2_hand_contact(final_contacts),
            "final_native_task_success": final_status.get("task_success"),
            "final_native_predicates": _native_values(final_status),
            "all_native_predicates_true_at_settle_endpoints": native_settle_true,
            "all_native_predicates_true_at_every_settle_substep": native_settle_substeps_true,
            "settle_control_actions": len(settled_endpoint_rows),
            "settle_physics_substep_samples": len(settled_substep_rows),
            "settling_pot2_position_span_m": pos_span,
            "settling_pot2_max_linear_speed_m_s": max_linear,
            "settling_pot2_max_angular_speed_rad_s": max_angular,
            "baseline_state_matches": (
                sum(bool(row.get("baseline_exact_state_match")) for row in action_rows if "baseline_exact_state_match" in row),
                sum("baseline_exact_state_match" in row for row in action_rows),
            ),
            "contact_pairs_start": sorted([list(x) for x in contact_pair_names(initial_contacts)]),
            "contact_pairs_final": sorted([list(x) for x in contact_pair_names(final_contacts)]),
            "placement_adjustment": placement_adjustment,
            "existing_settle_limits_record_only": {
                "position_span_m_max": 0.002,
                "linear_speed_m_s_max": 0.08,
                "angular_speed_rad_s_max": 0.8,
            },
            "pot2_passes_existing_settle_limits": bool(
                pos_span is not None and pos_span <= 0.002
                and max_linear is not None and max_linear <= 0.08
                and max_angular is not None and max_angular <= 0.8
            ),
            "safety_certified": False,
            "artifacts": {
                "control_actions": str(branch_dir / "control_actions.jsonl"),
                "physics_substeps": str(branch_dir / "physics_substeps.jsonl"),
                "video": str(branch_dir / "dual_camera_release_branch_20fps.mp4"),
                "final_snapshot": str(branch_dir / "final_runtime_snapshot.pkl"),
            },
        }
        _write_json(branch_dir / "branch_report.json", result)
        _write_json(branch_dir / "video_frame_index.json", frames)
        return result
    except Exception as exc:
        failure = {
            "branch_id": branch_id,
            "strategy": strategy,
            "status": "failed_or_interrupted",
            "error": f"{type(exc).__name__}: {exc}",
            "completed_actions": len(action_rows),
            "physics_substeps_expected": len(action_rows) * SUBSTEPS,
            "physics_substeps_captured": sum(row.get("substep_count", 0) for row in action_rows),
            "substep_state_capture_complete_so_far": all_substeps_complete,
            "durable_action_path": str(branch_dir / "control_actions.jsonl"),
            "durable_substep_path": str(branch_dir / "physics_substeps.jsonl"),
            "safety_certified": False,
        }
        _write_json(branch_dir / "branch_failure.json", failure)
        _write_json(branch_dir / "video_frame_index.json", frames)
        return failure
    finally:
        if writer is not None:
            writer.release()
        bridge.close()
        grasp_search.BRIDGE_SCRIPT = old_script


def run_comparison(output: Path, slow_step_m: float = 0.0125, slow_limit: int = 30) -> dict[str, Any]:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_path = PRIOR / "native_success_before_release_runtime_snapshot.pkl"
    source = _verified_resume_runtime_snapshot(
        source_path, benchmark="obstacle_avoidance", level=1, task_id=2, seed=0, init_state=0,
        model_sha=EXPECTED_MODEL_SHA, bddl_sha=EXPECTED_BDDL_SHA,
        instruction=EXPECTED_INSTRUCTION,
    )
    payload = source["payload"]
    source_state = np.asarray(source["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
    source_copy = output / "source_native_success_t1335_runtime_snapshot.pkl"
    source_copy.write_bytes(payload)
    _write_json(output / "comparison_input.json", {
        "task": "obstacle_avoidance/L1T2 seed=0 init-state=0",
        "source_snapshot_path": str(source_path),
        "source_snapshot_sha256": source["payload_sha256"],
        "source_integration_state_sha256": source["integration_state_sha256"],
        "source_timestep": 1335,
        "model_geometry_sha256": EXPECTED_MODEL_SHA,
        "bddl_sha256": EXPECTED_BDDL_SHA,
        "original_run53_report": str(RUN53 / "run_report.json"),
        "original_run53_report_sha256": sha256_file(RUN53 / "run_report.json"),
        "all_branches_restore_identical_snapshot": True,
        "official_safety_demo_used": False,
        "object_state_direct_write": False,
        "native_bddl_definition_modified": False,
        "safety_certified": False,
        "settling_control_actions": SETTLE_STEPS,
        "slow_retreat_waypoint_increment_m": slow_step_m,
        "slow_retreat_action_limit": slow_limit,
    })
    source_text = BRIDGE_SOURCE.read_text(encoding="utf-8")
    runtime_text = instrument_bridge_source(source_text)
    runtime_script = BRIDGE_SOURCE.with_name(f".run53_release_diag_bridge_{os.getpid()}.py")
    runtime_script.write_text(runtime_text, encoding="utf-8")
    runtime_sha = sha256_file(runtime_script)
    original_suffix = _load_original_suffix()
    results = []
    try:
        for branch_id, strategy in (
            ("branch_01_original_retreat", "original_retreat"),
            ("branch_02_open_hold_pose", "hold_after_full_release"),
            ("branch_03_slow_upward_retreat", "slow_upward_retreat"),
        ):
            results.append(_run_branch(
                runtime_script, payload, source_state, output, branch_id, strategy,
                original_suffix, slow_step_m, slow_limit,
            ))
    finally:
        runtime_script.unlink(missing_ok=True)
    valid = [
        row for row in results
        if row.get("status") == "complete"
        and row.get("substep_state_capture_complete") is True
        and row.get("final_hand_pot2_contact") is False
        and row.get("all_native_predicates_true_at_settle_endpoints") is True
        and row.get("all_native_predicates_true_at_every_settle_substep") is True
        and row.get("pot2_passes_existing_settle_limits") is True
        and row.get("withdrawal_displacement_m", 0) >= 0.05
    ]
    report = {
        "schema_version": "gs_opd_run53_release_strategy_comparison.v1",
        "status": "stable_release_strategy_found" if valid else "no_stable_release_strategy_found",
        "task": "obstacle_avoidance/L1T2 seed=0 init-state=0",
        "comparison_archive": str(output),
        "source_snapshot_timestep": 1335,
        "source_snapshot_sha256": source["payload_sha256"],
        "branches": results,
        "best_branch_id": valid[0]["branch_id"] if valid else None,
        "acceptance": {
            "hand_geom_contact_free": True,
            "retreat_displacement_minimum_m_record_only": 0.05,
            "pot1_pot2_stove_predicates_true_after_settling": True,
            "pot2_position_span_m_max_record_only": 0.002,
            "pot2_linear_speed_m_s_max_record_only": 0.08,
            "pot2_angular_speed_rad_s_max_record_only": 0.8,
            "safety_certification": False,
        },
        "run53_original_report_modified": False,
        "task_success_definition_modified": False,
        "object_state_directly_modified": False,
        "training_started": False,
        "other_task_search_started": False,
        "runtime_bridge_copy_sha256": runtime_sha,
        "substep_monitoring": {
            "expected_physics_substeps_per_action": SUBSTEPS,
            "captured": ["robot_pose_velocity", "Pot1_and_Pot2_pose_velocity", "native_BDDL_predicates", "contact_pairs_and_force_records"],
            "safety_verdict": False,
        },
    }
    _write_json(output / "run53_release_strategy_comparison.json", report)
    lines = [
        "# run53 Release/Retreat 策略对比",
        "",
        f"状态：{report['status']}。所有分支均从同一 t=1335 Runtime Snapshot 恢复；每个控制动作要求完整采集25个MuJoCo子步。此诊断不构成严格安全认证。",
        "",
        "| 分支 | 策略 | 状态 | 动作数 | 子步覆盖 | 末态原生谓词 | 松爪后稳定 | 退爪/ Pot2接触 |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for row in results:
        if row.get("status") != "complete":
            lines.append(f"| {row['branch_id']} | {row['strategy']} | {row['status']} | {row.get('completed_actions', 0)} | {row.get('physics_substeps_captured', 0)} | — | — | — |")
        else:
            lines.append(
                f"| {row['branch_id']} | {row['strategy']} | complete | {row['actions']} | {row['physics_substeps_captured']}/{row['physics_substeps_expected']} | {row['final_native_predicates']} | {row['pot2_passes_existing_settle_limits']} | {row['final_hand_pot2_contact']} |"
            )
    lines.extend(["", "分支动作、25子步状态/接触、视频和可恢复Snapshot见各自目录；原run53成功报告保持不变。", ""])
    (output / "run53_release_strategy_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def run_placement_refinement(
    output: Path,
    offsets_mm: tuple[float, ...] = (25.0, 35.0, 45.0),
    slow_step_m: float = 0.0125,
    slow_limit: int = 30,
    retreat_style: str = "upward",
) -> dict[str, Any]:
    """Try only small, region-relative placement corrections from the t=1335 held state."""
    output = Path(output).resolve()
    if not offsets_mm or any(not np.isfinite(value) or value <= 0 or value > 50 for value in offsets_mm):
        raise ValueError("placement_offsets_mm_must_be_within_(0,50]")
    output.mkdir(parents=True, exist_ok=False)
    source_path = PRIOR / "native_success_before_release_runtime_snapshot.pkl"
    source = _verified_resume_runtime_snapshot(
        source_path, benchmark="obstacle_avoidance", level=1, task_id=2,
        seed=0, init_state=0, model_sha=EXPECTED_MODEL_SHA,
        bddl_sha=EXPECTED_BDDL_SHA, instruction=EXPECTED_INSTRUCTION,
    )
    payload = source["payload"]
    source_state = np.asarray(source["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
    (output / "source_native_success_t1335_runtime_snapshot.pkl").write_bytes(payload)
    _write_json(output / "placement_refinement_input.json", {
        "task": "obstacle_avoidance/L1T2 seed=0 init-state=0",
        "source_snapshot_path": str(source_path),
        "source_snapshot_sha256": source["payload_sha256"],
        "source_timestep": 1335,
        "model_geometry_sha256": EXPECTED_MODEL_SHA,
        "bddl_sha256": EXPECTED_BDDL_SHA,
        "strategy": "derive inward shift direction from live native On region center and Pot2 root pose; lift 8mm, translate while held, lower to contact, physically open, slow retreat, settle",
        "candidate_offsets_toward_region_center_mm": list(offsets_mm),
        "settle_control_steps": SETTLE_STEPS,
        "expected_substeps_per_action": SUBSTEPS,
        "official_safety_demo_used": False,
        "object_state_direct_write": False,
        "native_bddl_definition_modified": False,
        "safety_certified": False,
    })
    runtime_text = instrument_bridge_source(BRIDGE_SOURCE.read_text(encoding="utf-8"))
    runtime_script = BRIDGE_SOURCE.with_name(f".run53_release_diag_bridge_{os.getpid()}.py")
    runtime_script.write_text(runtime_text, encoding="utf-8")
    runtime_hash = sha256_file(runtime_script)
    results = []
    try:
        for index, offset in enumerate(offsets_mm, start=1):
            label = f"placement_branch_{index:02d}_{int(round(offset))}mm"
            strategy = f"placement_offset_{offset:g}mm"
            branch_result = _run_branch(
                runtime_script, payload, source_state, output, label, strategy,
                [], slow_step_m, slow_limit, retreat_style,
            )
            branch_result["artifact_sha256"] = _artifact_hashes(branch_result)
            branch_report_path = output / label / "branch_report.json"
            branch_failure_path = output / label / "branch_failure.json"
            if branch_report_path.is_file():
                _write_json(branch_report_path, branch_result)
            elif branch_failure_path.is_file():
                _write_json(branch_failure_path, branch_result)
            results.append(branch_result)
    finally:
        runtime_script.unlink(missing_ok=True)
    valid = [row for row in results if stable_release_candidate(row)]
    report = {
        "schema_version": "gs_opd_run53_release_local_placement_refinement.v1",
        "status": "stable_release_strategy_found" if valid else "no_stable_adjusted_placement_found",
        "task": "obstacle_avoidance/L1T2 seed=0 init-state=0",
        "archive": str(output),
        "source_snapshot_timestep": 1335,
        "source_snapshot_sha256": source["payload_sha256"],
        "branches": results,
        "best_branch_id": valid[0]["branch_id"] if valid else None,
        "branch_count": len(results),
        "placement_adjustment_scope": "last 25–45mm of Pot2 placement only; original grasp and transport prefix unchanged",
        "retreat_style": retreat_style,
        "acceptance": {
            "all_native_predicates_true_through_every_settle_substep": True,
            "gripper_geom_contact_free_after_release": True,
            "upward_retreat_displacement_at_least_50mm": True,
            "Pot2_settle_metrics_use_all_25_substep_samples": True,
            "safety_certified": False,
        },
        "original_run53_report_modified": False,
        "native_task_success_definition_modified": False,
        "training_started": False,
        "other_task_search_started": False,
        "runtime_bridge_copy_sha256": runtime_hash,
    }
    _write_json(output / "run53_release_local_placement_refinement.json", report)
    lines = [
        "# run53 最后几厘米放置/稳定释放分支",
        "",
        f"结果：{report['status']}。所有分支由相同t=1335 Pot2持抓Snapshot恢复；目标偏移由活动BDDL Cook Region中心和实时Pot2位置自动计算，不改抓取或运输前缀。",
        "",
        "| 分支 | 偏移 | 状态 | 动作 | 子步覆盖 | 末态谓词 | settle位姿跨度 | 最大线速度 | 最大角速度 |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in results:
        if row.get("status") != "complete":
            lines.append(f"| {row['branch_id']} | {row['strategy']} | {row['status']} | {row.get('completed_actions', 0)} | {row.get('physics_substeps_captured', 0)} | — | — | — | — |")
            continue
        lines.append(
            f"| {row['branch_id']} | {row['strategy']} | complete | {row['actions']} | {row['physics_substeps_captured']}/{row['physics_substeps_expected']} | {row['final_native_predicates']} | {row.get('settling_pot2_position_span_m')} | {row.get('settling_pot2_max_linear_speed_m_s')} | {row.get('settling_pot2_max_angular_speed_rad_s')} |"
        )
    lines.extend(["", "每分支的控制动作、25子步状态/BDDL/接触、视频与Snapshot保存在同名目录。成功搜索轨迹不自动获得安全认证。", ""])
    (output / "run53_release_local_placement_refinement.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--slow-step-mm", type=float, default=12.5)
    parser.add_argument("--slow-limit", type=int, default=30)
    parser.add_argument("--placement-refinement", action="store_true")
    parser.add_argument("--placement-offsets-mm", nargs="+", type=float, default=[25.0, 35.0, 45.0])
    parser.add_argument("--retreat-style", choices=("upward", "lateral_up"), default="upward")
    args = parser.parse_args()
    if args.slow_step_mm <= 0 or args.slow_limit <= 0:
        raise ValueError("slow retreat parameters must be positive")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or Path(f"/opt/data/private/gs_opd/rollouts/run53_release_strategy_comparison_{stamp}")
    if args.placement_refinement:
        report = run_placement_refinement(
            output, tuple(args.placement_offsets_mm), args.slow_step_mm / 1000.0, args.slow_limit,
            args.retreat_style,
        )
    else:
        report = run_comparison(output, args.slow_step_mm / 1000.0, args.slow_limit)
    print(json.dumps({
        "status": report["status"],
        "archive": report.get("archive", report.get("comparison_archive")),
        "best_branch_id": report["best_branch_id"],
        "branches": [
            {"id": row["branch_id"], "status": row["status"], "actions": row.get("actions", row.get("completed_actions"))}
            for row in report["branches"]
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

