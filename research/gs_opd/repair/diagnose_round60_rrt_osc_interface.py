# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run one bounded Round60 RRT-to-OSC interface diagnostic prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from research.gs_opd.repair.task_success_first_executor import (
    _placement_ik_seed_portfolio,
    _verified_runtime_snapshot_descriptor,
    _verify_planner_live_state,
)
from research.gs_opd.repair.task_success_first_grasp_search import (
    BridgeSession,
    _build_safety_policy,
    _snapshot_state_vector,
    _state_sha,
)
from research.gs_opd.repair.task_success_first_grasp_search import (
    _advance_pose,
    _compress_eef_route_with_indices,
)
from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor
from research.gs_opd.repair.trajectory_tracking import quaternion_conjugate, quaternion_multiply, quaternion_to_rotation_vector


ROUND60_ROOT = Path(
    "/opt/data/private/gs_opd/rollouts/l2t2_crosslevel_keypose_20261026_round60/"
    "task_success_first_executor_obstacle_avoidance_L2T2_20261026_175558_280342129"
)
ROUND60_REPORT = ROUND60_ROOT / "run_report.json"
ROUND60_BRANCH = ROUND60_ROOT / "expert_native_goal_stage_control/placement_branch_00_route_01_region_layout_20_v1_yaw+180_h380"
INITIAL_T0_ROOT = Path(
    "/opt/data/private/gs_opd/rollouts/task_success_first_cross_level_L2T2_20261026_run22_L0_success_migration/"
    "obstacle_avoidance_L2T2/stage_snapshots/t0"
)
OUTPUT_ROOT = Path("/opt/data/private/gs_opd/diagnostics/round60_rrt_osc_interface_20261026")
MAX_VALIDATION_ACTIONS = 18  # t=2589 -> at most t=2607; no full failed-route replay.


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _rotation_error(target: np.ndarray, actual: np.ndarray) -> float:
    delta = quaternion_multiply(target, quaternion_conjugate(actual))
    return float(np.linalg.norm(quaternion_to_rotation_vector(delta)))


def run() -> dict[str, Any]:
    global OUTPUT_ROOT
    if OUTPUT_ROOT.exists():
        OUTPUT_ROOT = OUTPUT_ROOT.with_name(
            f"{OUTPUT_ROOT.name}_{time.strftime('%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
        )
    report = json.loads(ROUND60_REPORT.read_text(encoding="utf-8"))
    segment = report["segments"][-1]
    start_path = Path(segment["start_snapshot"])
    source_rows_path = Path(segment["artifacts"]["durable_records"])
    source_rows = [json.loads(line) for line in source_rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not source_rows or int(source_rows[0].get("timestep", -1)) != 2590:
        raise ValueError("Round60 source action prefix does not begin at t=2590")
    manifest = json.loads((INITIAL_T0_ROOT / "snapshots/runtime_snapshot_manifest.json").read_text(encoding="utf-8"))
    scene = manifest["scene_metadata"]
    expected_model_sha = str(scene["model_geometry_sha256"])
    expected_bddl_sha = str(scene["bddl_sha256"])
    if report["scope"].get("model_geometry_sha256") != expected_model_sha:
        raise ValueError("Round60 model SHA differs from its t0 manifest")
    verified_snapshot = _verified_runtime_snapshot_descriptor(
        start_path,
        benchmark="obstacle_avoidance", level=2, task_id=2,
        seed=0, init_state=0, model_sha=expected_model_sha,
        bddl_sha=expected_bddl_sha, instruction=str(manifest["instruction"]),
    )
    start_payload = verified_snapshot["payload"]
    base_snapshot_hash = verified_snapshot["payload_sha256"]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    bridge_args = argparse.Namespace(
        benchmark="obstacle_avoidance", level=2, task_id=2, init_state=0, seed=0,
        camera_resolution=256, port=38679, demonstration_horizon=3600,
    )
    bridge = BridgeSession(("127.0.0.1", bridge_args.port), OUTPUT_ROOT / "safety_bridge.log", bridge_args)
    completed_actions = [0]
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        if reset.get("scene_metadata", {}).get("model_geometry_sha256") != expected_model_sha:
            raise RuntimeError("fresh_L2T2_bridge_model_SHA_mismatch")
        if reset.get("scene_metadata", {}).get("bddl_sha256") != expected_bddl_sha:
            raise RuntimeError("fresh_L2T2_bridge_BDDL_SHA_mismatch")
        restore = bridge.request({"op": "restore_runtime_snapshot", "payload": start_payload})
        expected_snapshot_state = restore.get("snapshot_state", {})
        if restore.get("status") != "restored" or restore.get("model_geometry_sha256") != expected_model_sha:
            raise RuntimeError(f"exact_Round60_snapshot_restore_failed:{restore.get('error')}")
        start_state = _snapshot_state_vector(expected_snapshot_state)
        start_state_sha = _state_sha(start_state)
        exact_start = bridge.request({"op": "observe", "include_snapshot_state": True})
        if not np.array_equal(_snapshot_state_vector(exact_start["snapshot_state"]), start_state):
            raise RuntimeError("Round60_start_Snapshot_integration_state_not_bit_exact")
        if exact_start.get("timestep") != 2589 or exact_start.get("instruction") != manifest["instruction"]:
            raise RuntimeError("Round60_start_timestep_or_instruction_mismatch")

        native = bridge.request({"op": "native_task_status"})
        predicate = ["on", "moka_pot_2", "flat_stove_1_cook_region"]
        if native.get("timestep") != 2589 or not any(
            item.get("predicate") == predicate and item.get("value") is False
            for item in native.get("native_goal_values", [])
        ):
            raise RuntimeError("Round60_native_goal_state_does_not_match_expected_partial_prefix")
        geometry = bridge.request({"op": "geometry_inventory"})
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        context = bridge.request({
            "op": "task_success_first_context",
            "expected_instruction": manifest["instruction"],
            "target_predicate": predicate,
        })
        gripper = bridge.request({"op": "gripper_geometry_inventory"})
        repair = bridge.request({"op": "repair_context"})
        if any(item.get("status") != "complete" for item in (geometry, semantic, context, gripper)):
            raise RuntimeError("Round60_active_semantic_or_controller_binding_unavailable")
        if not isinstance(repair.get("robots"), list) or not repair["robots"]:
            raise RuntimeError("Round60_active_OSC_controller_fields_unavailable")
        target_id = str(context["target_object_id"])
        target_binding_row = semantic["bindings"][target_id]
        target_binding = {
            "object_id": target_id,
            "body_id": int(target_binding_row["root_body_id"]),
            "body_name": str(target_binding_row["root_body_name"]),
            "geom_ids": sorted(int(value) for value in target_binding_row["root_geom_ids"]),
            "geom_names": list(target_binding_row["root_geom_names"]),
        }
        policy = _build_safety_policy(context, semantic, geometry, gripper)
        planning_policy = dict(policy)
        planning_policy["carried_object"] = {
            "object_id": target_id,
            "grasp_binding": target_binding,
            "allowed_contact_pairs": [],
            "allowed_support_contact_pairs": [],
        }
        controller_fields = repair["robots"][0]["fields"]
        _advance_pose.controller_fields = controller_fields
        initial_export = bridge.request({"op": "export_runtime_snapshot"})
        if initial_export.get("status") != "complete":
            raise RuntimeError("restored_Round60_start_snapshot_could_not_be_reexported")
        (OUTPUT_ROOT / "start_t2589_runtime_snapshot.pkl").write_bytes(initial_export["payload"])

        preplace = segment["keypose_transfer"]["retargeted_keyposes"][0]["eef_pose_world"]
        goal_pose = np.r_[preplace["position_m"], preplace["quaternion_wxyz"]].astype(np.float64)
        planner_result = bridge.request({
            "op": "task_success_first_rrt",
            "expected_instruction": manifest["instruction"],
            "model_geometry_sha256": expected_model_sha,
            "start_state_sha256": start_state_sha,
            "goal_pose_world": {
                "position_world_m": goal_pose[:3].tolist(),
                "quaternion_wxyz": goal_pose[3:].tolist(),
            },
            "safety_geometry_policy": planning_policy,
            "planner_config": {
                "step_rad": 0.22, "connect_max_steps": 16, "goal_bias": 0.18,
                "max_iterations": 2000, "max_config_checks": 8000,
                "max_candidate_routes": 1, "search_time_budget_s": 20.0,
                "shortcut_attempts": 12, "minimum_clearance_m": 0.0,
                "require_safe_configurations": False, "seed": 84202,
            },
            "ik_seed_qpos": _placement_ik_seed_portfolio(exact_start["snapshot_state"]["robot_states"][0]),
        })
        after_plan = bridge.request({"op": "observe", "include_snapshot_state": True})
        live_check = _verify_planner_live_state(
            planner_result,
            _state_sha(_snapshot_state_vector(after_plan["snapshot_state"])),
            start_state_sha,
        )
        if not live_check["live_state_unchanged"]:
            raise RuntimeError("RRT_planner_changed_live_Physics_state")
        if planner_result.get("status") != "offline_feasible":
            result = {
                "status": "no_physical_actions_RRT_not_feasible",
                "planner": planner_result,
                "live_state_verification": live_check,
                "exact_start_snapshot_restored": True,
                "physical_actions": 0,
            }
            _write_json(OUTPUT_ROOT / "diagnostic_report.json", result)
            return result

        qpos_path = np.asarray(planner_result["qpos_path"], dtype=np.float64)
        raw_eef = np.asarray(planner_result["eef_reference_poses_world"], dtype=np.float64)
        if qpos_path.shape != raw_eef.shape or qpos_path.ndim != 2 or qpos_path.shape[1] != 7:
            raise RuntimeError("RRT_joint_and_EEF_reference_paths_are_not_row_aligned")
        compressed, source_indices = _compress_eef_route_with_indices(raw_eef)
        route_archive = {
            "schema_version": "gs_opd_rrt_osc_route_archive.v1",
            "keypose": "preplace",
            "model_geometry_sha256": expected_model_sha,
            "start_state_sha256": start_state_sha,
            "goal_eef_pose_world": goal_pose.tolist(),
            "planner_seed": 84202,
            "planner_status": planner_result["status"],
            "planning_seconds": planner_result.get("planning_seconds"),
            "qpos_path_rad": qpos_path.tolist(),
            "raw_eef_reference_poses_world": raw_eef.tolist(),
            "compressed_waypoints": [
                {
                    "compressed_index": index,
                    "source_path_index": source_index,
                    "eef_pose_world": compressed[index].tolist(),
                    "rrt_reference_qpos_rad": qpos_path[source_index].tolist(),
                }
                for index, source_index in enumerate(source_indices)
            ],
            "compression": {
                "translation_threshold_m": 0.020,
                "rotation_threshold_rad": 0.10,
                "compressed_source_path_indices": source_indices,
            },
            "safety_certified": False,
        }
        route_path = OUTPUT_ROOT / "round60_rrt_osc_route.json"
        _write_json(route_path, route_archive)

        goal_position = goal_pose[:3]
        best_error = float("inf")
        best_snapshot_path = OUTPUT_ROOT / "best_progress_runtime_snapshot.pkl"
        frames_dir = OUTPUT_ROOT / "dual_camera_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        durable_records_path = OUTPUT_ROOT / "physical_step_records.jsonl"

        def persist_step(record: dict[str, Any], frame: dict[str, Any]) -> None:
            nonlocal best_error
            completed_actions[0] += 1
            with durable_records_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
                handle.flush()
            combined = np.concatenate((frame["agentview_rgb"], frame["wrist_rgb"]), axis=1)
            image_path = frames_dir / f"frame_{int(frame['step_index']):04d}.jpg"
            if not cv2.imwrite(str(image_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)):
                raise RuntimeError("diagnostic_dual_camera_frame_write_failed")
            state = record["snapshot_state"]["robot_states"][0]
            position_error = float(np.linalg.norm(np.asarray(state["eef_pos"]) - goal_position))
            rotation_error = _rotation_error(goal_pose[3:], np.asarray(state["eef_quat"]))
            metric = position_error + 0.05 * rotation_error
            if metric < best_error:
                best_error = metric
                saved = bridge.request({"op": "export_runtime_snapshot"})
                if saved.get("status") == "complete":
                    best_snapshot_path.write_bytes(saved["payload"])

        records: list[dict[str, Any]] = []
        frames: list[dict[str, Any]] = []
        counter = [0]
        monitor = GraspStabilityMonitor(required_samples=3)
        execution_start_time = time.monotonic()
        end_reason = "route_completed_before_action_cap"
        try:
            for waypoint_index, source_index in enumerate(source_indices):
                if counter[0] >= MAX_VALIDATION_ACTIONS:
                    end_reason = "strict_18_action_validation_cap"
                    break
                context_row = {
                    "rrt_archive_path": str(route_path),
                    "rrt_route_sha256": _sha_bytes(route_path.read_bytes()),
                    "keypose": "preplace",
                    "compressed_waypoint_index": waypoint_index,
                    "rrt_source_path_index": source_index,
                    "rrt_reference_eef_pose_world": raw_eef[source_index].tolist(),
                    "rrt_reference_qpos_rad": qpos_path[source_index].tolist(),
                }
                remaining = MAX_VALIDATION_ACTIONS - counter[0]
                _advance_pose(
                    bridge, compressed[waypoint_index, :3], compressed[waypoint_index, 3:],
                    "diagnostic_transport_preplace", "round60_interface_diagnostic", counter,
                    remaining, target_binding, manifest["instruction"], policy, monitor,
                    records, frames,
                    step_callback=persist_step,
                    stagnation_window_actions=24,
                    position_tolerance_m=0.010,
                    rotation_tolerance_rad=0.10,
                    planning_tracking=context_row,
                )
            if counter[0] >= MAX_VALIDATION_ACTIONS:
                end_reason = "strict_18_action_validation_cap"
        except TimeoutError:
            end_reason = "strict_18_action_validation_cap"

        physical_seconds = time.monotonic() - execution_start_time
        terminal = bridge.request({"op": "export_runtime_snapshot"})
        if terminal.get("status") != "complete":
            raise RuntimeError("terminal_diagnostic_snapshot_export_failed")
        terminal_path = OUTPUT_ROOT / "terminal_validation_runtime_snapshot.pkl"
        terminal_path.write_bytes(terminal["payload"])
        native_after = bridge.request({"op": "native_task_status"})
        output_video = OUTPUT_ROOT / "dual_camera_agentview_wrist_20fps.mp4"
        if frames:
            height, width = frames[0]["agentview_rgb"].shape[:2]
            writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (2 * width, height))
            for frame in frames:
                combined = np.concatenate((frame["agentview_rgb"], frame["wrist_rgb"]), axis=1)
                writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            writer.release()

        route_joint_errors = [row.get("planning_tracking", {}).get("rrt_reference_joint_error_rad") for row in records]
        reprojected = []
        all_hand_pot_contacts = []
        joint_limit_violation_count = 0
        max_action_abs = 0.0
        for row in records:
            tracking = row.get("planning_tracking", {})
            actual_q = np.asarray(tracking.get("actual_joint_qpos_after_rad", []), dtype=np.float64)
            nearest_index = int(np.argmin(np.linalg.norm(qpos_path - actual_q[None, :], axis=1))) if actual_q.shape == (7,) else None
            actual_eef = np.asarray(row["snapshot_state"]["robot_states"][0]["eef_pos"], dtype=np.float64)
            before_eef = np.asarray(tracking.get("observed_eef_pose_before_action_world", [])[:3], dtype=np.float64)
            requested_delta = np.asarray(tracking.get("requested_translation_delta_world_m", []), dtype=np.float64)
            actual_delta = actual_eef - before_eef if before_eef.shape == (3,) else np.zeros(3)
            direction_cosine = None
            if requested_delta.shape == (3,) and np.linalg.norm(requested_delta) > 1e-9 and np.linalg.norm(actual_delta) > 1e-9:
                direction_cosine = float(np.dot(requested_delta, actual_delta) / (np.linalg.norm(requested_delta) * np.linalg.norm(actual_delta)))
            nearest_eef_index = int(np.argmin(np.linalg.norm(raw_eef[:, :3] - actual_eef[None, :], axis=1)))
            action = np.asarray(row.get("action_7d", []), dtype=np.float64)
            if action.shape == (7,):
                max_action_abs = max(max_action_abs, float(np.max(np.abs(action))))
            for substep in row.get("contact_monitoring", {}).get("geometry_monitoring", {}).get("substeps", []):
                joint_limit_violation_count += len(substep.get("joint_limit_violations", []))
                for contact in substep.get("forbidden_contacts", []):
                    geom1 = contact.get("geom1", {}).get("name", "")
                    geom2 = contact.get("geom2", {}).get("name", "")
                    if {geom1, geom2} == {"gripper0_hand_collision", "moka_pot_2_g14"}:
                        all_hand_pot_contacts.append(contact)
            reprojected.append({
                "timestep": row.get("timestep"),
                "active_rrt_path_index": tracking.get("rrt_source_path_index"),
                "nearest_rrt_qpos_path_index": nearest_index,
                "nearest_rrt_qpos_distance_rad": (
                    float(np.linalg.norm(actual_q - qpos_path[nearest_index])) if nearest_index is not None else None
                ),
                "nearest_rrt_eef_path_index": nearest_eef_index,
                "nearest_rrt_eef_path_distance_m": float(np.linalg.norm(actual_eef - raw_eef[nearest_eef_index, :3])),
                "active_waypoint_reference_error_rad": tracking.get("rrt_reference_joint_error_rad"),
                "command_target_eef_pose_world": tracking.get("osc_command_target_eef_pose_world"),
                "active_subgoal_eef_pose_world": tracking.get("active_subgoal_eef_pose_world"),
                "requested_translation_delta_world_m": requested_delta.tolist() if requested_delta.shape == (3,) else None,
                "actual_translation_delta_world_m": actual_delta.tolist(),
                "translation_direction_cosine": direction_cosine,
                "actual_qvel_rad_s": row["snapshot_state"]["robot_states"][0].get("joint_qvel"),
            })
        contact_normal_forces = [
            abs(float(row["force_and_torque"][0]))
            for row in all_hand_pot_contacts
            if row.get("force_status") == "available"
            and isinstance(row.get("force_and_torque"), list)
            and len(row["force_and_torque"]) >= 3
        ]
        first_axis_opposition = None
        for item in reprojected:
            requested = item.get("requested_translation_delta_world_m")
            actual = item.get("actual_translation_delta_world_m")
            if requested is None:
                continue
            for axis, request_value, actual_value in zip("xyz", requested, actual):
                if abs(request_value) >= 0.001 and abs(actual_value) >= 0.001 and request_value * actual_value < 0:
                    first_axis_opposition = {
                        "timestep": item["timestep"], "axis": axis,
                        "requested_delta_m": request_value, "actual_delta_m": actual_value,
                    }
                    break
            if first_axis_opposition is not None:
                break
        result = {
            "schema_version": "gs_opd_round60_rrt_osc_interface_diagnostic.v1",
            "status": "bounded_physical_validation_complete",
            "scope": {
                "task": "obstacle_avoidance/L2T2",
                "start_timestep": 2589,
                "terminal_timestep": native_after.get("timestep"),
                "model_geometry_sha256": expected_model_sha,
                "bddl_sha256": expected_bddl_sha,
                "start_snapshot_path": str(start_path),
                "start_snapshot_sha256": base_snapshot_hash,
                "start_integration_state_sha256": start_state_sha,
                "instruction": manifest["instruction"],
                "demo_only_horizon_extension": True,
            },
            "physical_budget": {
                "max_control_actions": MAX_VALIDATION_ACTIONS,
                "executed_control_actions": len(records),
                "captured_physics_substeps": sum(
                    row.get("contact_monitoring", {}).get("captured_physics_substeps", 0) for row in records
                ),
                "end_reason": end_reason,
                "physics_execution_seconds": physical_seconds,
                "full_failed_route_replayed": False,
                "official_demonstrations_used": False,
            },
            "planner": {
                "status": planner_result.get("status"),
                "planning_seconds": planner_result.get("planning_seconds"),
                "configuration_checks": planner_result.get("config_checks"),
                "iterations": planner_result.get("iterations"),
                "seed": 84202,
                "controller_configuration": controller_fields,
                "path_rows": len(qpos_path),
                "compressed_waypoint_count": len(compressed),
                "compression_source_indices": source_indices,
                "joint_and_eef_paths_row_aligned": True,
                "route_archive": str(route_path),
                "route_archive_sha256": _sha_bytes(route_path.read_bytes()),
                "live_state_verification": live_check,
            },
            "tracking_metrics": {
                "per_step": reprojected,
                "max_active_waypoint_qpos_error_rad": max(
                    (float(value) for value in route_joint_errors if value is not None), default=None,
                ),
                "nearest_path_qpos_deviation_rad_range": [
                    min((item["nearest_rrt_qpos_distance_rad"] for item in reprojected if item["nearest_rrt_qpos_distance_rad"] is not None), default=None),
                    max((item["nearest_rrt_qpos_distance_rad"] for item in reprojected if item["nearest_rrt_qpos_distance_rad"] is not None), default=None),
                ],
                "nearest_path_eef_deviation_m_range": [
                    min(item["nearest_rrt_eef_path_distance_m"] for item in reprojected),
                    max(item["nearest_rrt_eef_path_distance_m"] for item in reprojected),
                ],
                "first_axis_opposition": first_axis_opposition,
                "minimum_translation_direction_cosine": min(
                    (item["translation_direction_cosine"] for item in reprojected if item["translation_direction_cosine"] is not None), default=None,
                ),
                "first_step_where_actual_translation_opposed_osc_request": next((
                    row.get("timestep") for row in records
                    if row.get("planning_tracking", {}).get("requested_translation_delta_world_m") is not None
                    and float(np.dot(
                        np.asarray(row["planning_tracking"]["requested_translation_delta_world_m"]),
                        np.asarray(row["snapshot_state"]["robot_states"][0]["eef_pos"])
                        - np.asarray(row["planning_tracking"]["observed_eef_pose_before_action_world"][:3]),
                    )) < 0.0
                ), None),
                "first_original_Round60_reversal_timestep": 2607,
                "joint_limit_violation_samples": joint_limit_violation_count,
                "maximum_abs_normalized_7d_action": max_action_abs,
            },
            "contacts": {
                "substep_monitoring_complete_for_all_steps": all(
                    row.get("substep_monitoring_complete") is True for row in records
                ),
                "nonpad_hand_pot2_contact": {
                    "sample_count": len(all_hand_pot_contacts),
                    "geom_pair": ["gripper0_hand_collision (Geom 74)", "moka_pot_2_g14 (Geom 117)"],
                    "body_pair": ["gripper0_right_gripper (Body 12)", "moka_pot_2_main (Body 24)"],
                    "signed_contact_distance_m_range": [
                        min((float(row["signed_contact_distance_m"]) for row in all_hand_pot_contacts), default=None),
                        max((float(row["signed_contact_distance_m"]) for row in all_hand_pot_contacts), default=None),
                    ],
                    "normal_force_contact_frame_N_range": [
                        min(contact_normal_forces, default=None), max(contact_normal_forces, default=None),
                    ],
                    "all_samples_have_force": len(contact_normal_forces) == len(all_hand_pot_contacts),
                    "source_records_contain_full_per_substep_contact_rows": True,
                },
                "full_step_records_path": str(OUTPUT_ROOT / "physical_step_records.jsonl"),
            },
            "native_task_status_after": native_after,
            "best_snapshot": str(best_snapshot_path) if best_snapshot_path.is_file() else None,
            "terminal_snapshot": str(terminal_path),
            "artifacts": {
                "records": str(OUTPUT_ROOT / "physical_step_records.jsonl"),
                "frames": str(frames_dir),
                "video": str(output_video) if output_video.is_file() else None,
                "route_archive": str(route_path),
            },
            "safety_certified": False,
            "task_success_candidate": False,
        }
        _write_json(OUTPUT_ROOT / "diagnostic_report.json", result)
        return result
    except Exception as exc:
        _write_json(OUTPUT_ROOT / "diagnostic_error.json", {
            "status": "diagnostic_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "start_snapshot": str(start_path),
            "physical_actions_completed_before_error": completed_actions[0],
        })
        raise
    finally:
        bridge.close()


def main() -> int:
    result = run()
    print(json.dumps({
        "status": result.get("status"),
        "archive": str(OUTPUT_ROOT),
        "physical_actions": result.get("physical_budget", {}).get("executed_control_actions"),
        "planner_status": result.get("planner", {}).get("status"),
        "failure_reason": result.get("failure_reason"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
