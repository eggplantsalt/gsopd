# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Continue the already executed t0->t1 prefix from its exact runtime snapshot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from research.gs_opd.repair.automatic_recovery import RecoveryArtifactWriter
from research.gs_opd.repair.automatic_recovery_runtime import (
    AutomaticRecoveryBridgeBackend,
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
)
from research.gs_opd.repair.replay_episode9_t10_to_t249_current_safety import (
    EXPECTED_BDDL_SHA256,
    EXPECTED_MODEL_SHA256,
    RAW_STUDENT_DIR,
    T0_TO_T10_TRACE,
    T10_SNAPSHOT,
    T10_VERIFICATION,
    T249_CHAIN_POLICY_REPORT,
    _canonical_sha256,
    _camera_frame,
    _grasp_state,
    _integration_hash,
    _jsonable,
    _make_current_policy,
    _max_abs,
    _native_substep_pass,
    _operational_substep_pass,
    _robot_contact_audit,
    _runtime_state_comparison,
    _snapshot_runtime_fields,
    _write_video,
)
from research.gs_opd.repair.replay_raw_pi05_t0_to_t10_current_safety import (
    DEFAULT_CONFIG,
    _margins,
    _sha256,
)


ROLLOUT_ROOT = Path("/opt/data/private/gs_opd/rollouts")
PRIOR_T0_T1_DIR = ROLLOUT_ROOT / "l0t2_round0_raw_pi05_t0_to_t10_current_safety_20261025_attempt2"
EXPECTED_TOTAL_ACTIONS = 10
EXPECTED_SUBSTEPS = 25


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _robot_ids(bridge: BridgeProcess) -> set[int]:
    inventory = bridge.request({"op": "geometry_inventory"})
    if inventory.get("status") != "complete":
        raise ValueError("full_robot_geometry_inventory_incomplete")
    return {
        int(item["geom_id"])
        for item in inventory.get("geometries", [])
        if item.get("category") == "robot" and item.get("collision_enabled") is True
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROLLOUT_ROOT / f"l0t2_round0_raw_pi05_t1_to_t10_current_safety_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir()

    prior_report_path = PRIOR_T0_T1_DIR / "t0_to_t10_current_safety_report.json"
    prior_report = json.loads(prior_report_path.read_text())
    if prior_report.get("physical_action_count") != 1 or prior_report.get("physical_substep_count") != 25:
        raise ValueError("prior_t0_t1_archive_does_not_contain_exactly_one_complete_physical_action")
    if prior_report.get("initialization", {}).get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("prior_t0_t1_model_fingerprint_mismatch")
    prior_action_records = _jsonl_rows(PRIOR_T0_T1_DIR / "action_records.jsonl")
    prior_substeps = _jsonl_rows(PRIOR_T0_T1_DIR / "substep_safety.jsonl")
    if len(prior_action_records) != 1 or prior_action_records[0].get("timestep") != 1:
        raise ValueError("prior_archive_action_row_is_not_t0_to_t1")
    if not prior_action_records[0].get("native_safety_pass") or not prior_action_records[0].get("operational_safety_pass"):
        raise ValueError("prior_t0_to_t1_action_did_not_pass_current_safety")
    if len(prior_substeps) != EXPECTED_SUBSTEPS or not all(
        item.get("native_safety_pass") and item.get("operational_safety_pass") for item in prior_substeps
    ):
        raise ValueError("prior_t0_to_t1_substep_evidence_is_incomplete_or_failed")

    prior_t1_snapshot_path = PRIOR_T0_T1_DIR / "snapshots/timestep_000001.pkl"
    prior_t1_snapshot = prior_t1_snapshot_path.read_bytes()
    prior_t1_runtime = _snapshot_runtime_fields(prior_t1_snapshot)
    prior_npz = np.load(PRIOR_T0_T1_DIR / "observation_action_trace.npz", allow_pickle=False)
    prior_action = np.asarray(prior_npz["actions"], dtype=np.float32)
    prior_post_rgb_agent = np.asarray(prior_npz["post_agentview"], dtype=np.uint8)
    prior_post_rgb_wrist = np.asarray(prior_npz["post_eye_in_hand"], dtype=np.uint8)
    prior_npz.close()

    raw_trace_path = RAW_STUDENT_DIR / "rollout_trace.npz"
    with np.load(raw_trace_path, allow_pickle=False) as raw, np.load(T0_TO_T10_TRACE, allow_pickle=False) as archived:
        all_raw_actions = np.asarray(raw["executed_actions"][:EXPECTED_TOTAL_ACTIONS], dtype=np.float32)
        all_archived_actions = np.asarray(archived["actions"], dtype=np.float32)
        if all_raw_actions.shape != (EXPECTED_TOTAL_ACTIONS, 7) or all_archived_actions.shape != (EXPECTED_TOTAL_ACTIONS, 7):
            raise ValueError("t0_to_t10_action_archive_shape_mismatch")
        if not np.array_equal(all_raw_actions, all_archived_actions):
            raise ValueError("t0_to_t10_action_source_mismatch")
        actions = all_archived_actions[1:]
        if not np.array_equal(prior_action, all_archived_actions[:1]):
            raise ValueError("saved_t0_to_t1_action_differs_from_original_pi05_prefix")

    config_data, recovery = load_runtime_config(args.config)
    frozen_policy = json.loads(T249_CHAIN_POLICY_REPORT.read_text()).get("policy")
    if not isinstance(frozen_policy, dict):
        raise ValueError("frozen_current_object_level_policy_missing")
    report: dict[str, Any] = {
        "schema": "gs_opd_round0_raw_pi05_t1_to_t10_current_safety_continuation.v1",
        "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "exact_restore_of_previously_recorded_t1_runtime_snapshot_then_raw_pi05_actions_1_through_9",
        "later_than_t10_actions_executed": False,
        "prior_action_t0_to_t1": {
            "archive": str(PRIOR_T0_T1_DIR),
            "report_sha256": _sha256(prior_report_path),
            "t1_snapshot_sha256": hashlib.sha256(prior_t1_snapshot).hexdigest(),
            "action_record": prior_action_records[0],
            "substep_count": len(prior_substeps),
        },
        "source_actions": {
            "raw_trace": str(raw_trace_path),
            "raw_trace_sha256": _sha256(raw_trace_path),
            "archived_action_trace": str(T0_TO_T10_TRACE),
            "archived_action_trace_sha256": _sha256(T0_TO_T10_TRACE),
            "all_first_ten_actions_exact_match": True,
            "continued_action_indices_0based": [1, 9],
        },
        "physical_action_count": 0,
        "physical_substep_count": 0,
        "outcome": None,
        "failure": None,
    }
    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    rows: list[dict[str, Any]] = []
    substep_rows: list[dict[str, Any]] = []
    pre_agent: list[np.ndarray] = []
    post_agent: list[np.ndarray] = []
    pre_wrist: list[np.ndarray] = []
    post_wrist: list[np.ndarray] = []
    pre_state: list[list[float]] = []
    post_integration_states: list[list[float]] = []
    action_log = None
    substep_log = None
    start_payload: bytes | None = None
    last_payload: bytes | None = None
    backend = None

    try:
        bridge.start(
            benchmark=config_data["benchmark"], level=int(config_data["level"]),
            task_id=int(config_data["task_id"]), init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]), demonstration_horizon=1000,
        )
        first_compile = bridge.request({"op": "observe", "include_snapshot_state": True})
        native_reset = bridge.request({"op": "reset"})
        reset_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
        if reset_observation.get("scene_metadata", {}).get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError("bridge_reset_model_hash_incompatible_with_saved_t1_snapshot")
        if int(reset_observation.get("timestep", -1)) != 0:
            raise RuntimeError("bridge_reset_did_not_return_to_t0_before_t1_restore")
        backend = AutomaticRecoveryBridgeBackend(
            bridge, config_data, recovery, output_dir, RecoveryArtifactWriter(output_dir),
            mode="episode", student=None,
            archived_trace_path=raw_trace_path,
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=T10_SNAPSHOT,
            t10_verification_path=T10_VERIFICATION,
        )
        backend.latest_data = reset_observation
        backend._load_context_inputs(reset_observation)
        policy = _make_current_policy(backend, frozen_policy)
        margins = _margins(policy)
        robot_ids = _robot_ids(bridge)
        finger_ids = set(int(x) for x in backend.context_static["gripper_geometry_ids"])
        target = backend.bindings["moka_pot_2"]
        restore = bridge.request({"op": "restore_runtime_snapshot", "payload": prior_t1_snapshot})
        if restore.get("status") != "restored":
            raise RuntimeError(f"t1_runtime_snapshot_restore_failed:{restore.get('error')}")
        current = bridge.request({"op": "observe", "include_snapshot_state": True})
        actual_t1_state = current.get("snapshot_state", {})
        t1_state_error = _max_abs(actual_t1_state.get("mujoco_integration_state"), prior_t1_runtime.get("integration_state"))
        reexport = bridge.request({"op": "export_runtime_snapshot"})
        if reexport.get("status") != "complete":
            raise RuntimeError("restored_t1_snapshot_reexport_failed")
        actual_t1_runtime = _snapshot_runtime_fields(reexport["payload"])
        runtime_compare = _runtime_state_comparison(prior_t1_runtime, actual_t1_runtime)
        report["t1_restore_validation"] = {
            "first_compile_model_geometry_sha256": first_compile.get("scene_metadata", {}).get("model_geometry_sha256"),
            "original_rollout_reset_rpc_timestep": native_reset.get("timestep"),
            "model_geometry_sha256_after_reset": current.get("scene_metadata", {}).get("model_geometry_sha256"),
            "restore_status": restore.get("status"),
            "restored_timestep": current.get("timestep"),
            "integration_state_max_abs_error_vs_prior_t1": t1_state_error,
            "integration_state_exact": t1_state_error == 0.0,
            "controller_gripper_buffers_and_task_bookkeeping": runtime_compare,
            "reexported_snapshot_byte_identical_to_source": reexport["payload"] == prior_t1_snapshot,
        }
        if int(current.get("timestep", -1)) != 1 or t1_state_error != 0.0 or runtime_compare["status"] != "pass":
            raise RuntimeError("t1_snapshot_restore_state_not_exact;no_following_action_executed")

        geometry = bridge.request({
            "op": "observe", "include_snapshot_state": True, "safety_geometry_policy": policy,
        }).get("safety_geometry", {})
        contacts = bridge.request({"op": "inspect_contacts"})
        contact_audit = _robot_contact_audit(
            {"contact_identities": geometry.get("contact_identities", [])}, robot_ids, finger_ids,
            int(target["root_body_id"]), set(int(x) for x in target["root_geom_ids"]),
        )
        native_preflight = _native_substep_pass(geometry, contact_audit)
        operational_preflight = _operational_substep_pass(geometry, margins)
        report["t1_preflight_after_restore"] = {
            "geometry_status": geometry.get("status"),
            "native_safety_pass": native_preflight,
            "operational_safety_pass": operational_preflight,
            "geometry_clearances_by_object_m": geometry.get("clearances_by_object"),
            "robot_contact_audit": contact_audit,
            "grasp_state": _grasp_state(bridge, backend),
            "native_task_status": bridge.request({"op": "native_task_status"}),
            "prior_t1_substep_policy_pass": True,
            "observed_contacts": contacts.get("contacts", []),
        }
        if geometry.get("status") != "complete" or not native_preflight or not operational_preflight:
            raise RuntimeError("restored_t1_state_current_safety_preflight_failed")
        start_export = bridge.request({"op": "export_runtime_snapshot"})
        if start_export.get("status") != "complete":
            raise RuntimeError("t1_continuation_start_snapshot_export_failed")
        start_payload = start_export["payload"]
        (snapshots_dir / "start_timestep_000001.pkl").write_bytes(start_payload)
        action_log = (output_dir / "action_records.jsonl").open("w")
        substep_log = (output_dir / "substep_safety.jsonl").open("w")
        action_log.write(json.dumps(_jsonable(prior_action_records[0]), sort_keys=True, allow_nan=False) + "\n")
        action_log.flush()
        for prior_substep in prior_substeps:
            substep_log.write(json.dumps(_jsonable(prior_substep), sort_keys=True, allow_nan=False) + "\n")
        substep_log.flush()

        for action_offset, action in enumerate(actions, start=1):
            pre_timestep = action_offset
            post_timestep = action_offset + 1
            before = bridge.request({"op": "observe", "include_snapshot_state": True})
            if int(before.get("timestep", -1)) != pre_timestep:
                raise RuntimeError(f"resume_pre_action_timestep_mismatch:{action_offset}")
            before_state = before.get("snapshot_state", {})
            reply = bridge.request({
                "op": "step_monitored", "action": action.tolist(),
                "action_index": action_offset, "safety_geometry_policy": policy,
            })
            if int(reply.get("timestep", -1)) != post_timestep:
                raise RuntimeError(f"resume_post_action_timestep_mismatch:{action_offset}")
            _dump(output_dir / f"action_{post_timestep:06d}_raw_response.json", {
                "timestep": post_timestep,
                "action_index_0based": action_offset,
                "action": action,
                "reward": reply.get("reward"),
                "done": reply.get("done"),
                "task_success": reply.get("task_success"),
                "snapshot_state": reply.get("snapshot_state"),
                "contact_monitoring": reply.get("contact_monitoring"),
                "contacts": reply.get("contacts"),
                "safety_geometry": reply.get("safety_geometry"),
            })
            np.savez_compressed(
                output_dir / f"action_{post_timestep:06d}_observation.npz",
                action=np.asarray(action, dtype=np.float32),
                pre_agentview=_camera_frame(before, "agentview_image"),
                post_agentview=_camera_frame(reply, "agentview_image"),
                pre_eye_in_hand=_camera_frame(before, "robot0_eye_in_hand_image"),
                post_eye_in_hand=_camera_frame(reply, "robot0_eye_in_hand_image"),
                pre_integration_state=np.asarray(before.get("snapshot_state", {}).get("mujoco_integration_state"), dtype=np.float64),
                post_integration_state=np.asarray(reply.get("snapshot_state", {}).get("mujoco_integration_state"), dtype=np.float64),
                pre_robot0_eef_pos=np.asarray(before.get("state", {}).get("robot0_eef_pos"), dtype=np.float64),
                post_robot0_eef_pos=np.asarray(reply.get("state", {}).get("robot0_eef_pos"), dtype=np.float64),
                pre_robot0_eef_quat=np.asarray(before.get("state", {}).get("robot0_eef_quat"), dtype=np.float64),
                post_robot0_eef_quat=np.asarray(reply.get("state", {}).get("robot0_eef_quat"), dtype=np.float64),
                pre_robot0_gripper_qpos=np.asarray(before.get("state", {}).get("robot0_gripper_qpos"), dtype=np.float64),
                post_robot0_gripper_qpos=np.asarray(reply.get("state", {}).get("robot0_gripper_qpos"), dtype=np.float64),
            )
            monitor = reply.get("contact_monitoring", {})
            geometry_monitor = monitor.get("geometry_monitoring", {})
            substeps = geometry_monitor.get("substeps", [])
            complete = (
                monitor.get("status") == "complete"
                and geometry_monitor.get("status") in {"complete", "unsafe"}
                and len(substeps) == EXPECTED_SUBSTEPS
                and monitor.get("captured_physics_substeps") == EXPECTED_SUBSTEPS
                and all(item.get("status") in {"complete", "unsafe"} for item in substeps)
            )
            action_native = bool(complete)
            action_operational = bool(complete)
            strict10 = bool(complete)
            min_clearance: dict[str, float] = {}
            first_failure = None
            action_substeps = []
            for substep_index, substep in enumerate(substeps):
                audit = _robot_contact_audit(
                    substep, robot_ids, finger_ids,
                    int(target["root_body_id"]), set(int(x) for x in target["root_geom_ids"]),
                )
                native_ok = _native_substep_pass(substep, audit)
                operational_ok = _operational_substep_pass(substep, margins)
                distances = []
                for object_id, detail in substep.get("clearances_by_object", {}).items():
                    distance = detail.get("minimum_distance_m") if isinstance(detail, dict) else None
                    if distance is not None:
                        distance = float(distance)
                        min_clearance[object_id] = min(min_clearance.get(object_id, math.inf), distance)
                        distances.append(distance)
                for pair, value in substep.get("allowed_contact_clearances_m", {}).items():
                    distance = float(value)
                    min_clearance[f"allowed_pair:{pair}"] = min(min_clearance.get(f"allowed_pair:{pair}", math.inf), distance)
                    distances.append(distance)
                strict_ok = bool(distances) and min(distances) >= 0.010
                action_native &= native_ok
                action_operational &= operational_ok
                strict10 &= strict_ok
                subrow = {
                    "timestep": post_timestep,
                    "action_index_0based": action_offset,
                    "physics_substep_index_0based": substep_index,
                    "native_safety_pass": native_ok,
                    "operational_safety_pass": operational_ok,
                    "strict_10mm_additional_metric_pass": strict_ok,
                    "robot_contact_audit": audit,
                    "geometry": substep,
                }
                substep_rows.append(subrow)
                action_substeps.append(subrow)
                if first_failure is None and (not native_ok or not operational_ok):
                    first_failure = {
                        "physics_substep_index_0based": substep_index,
                        "native_safety_pass": native_ok,
                        "operational_safety_pass": operational_ok,
                        "geometry": substep,
                        "robot_contact_audit": audit,
                    }
            post_snapshot_state = reply.get("snapshot_state", {})
            post_native = bridge.request({"op": "native_task_status"})
            held, grasp_complete, grasp_state = backend._grasp_states()
            record = {
                "timestep": post_timestep,
                "action_index_0based": action_offset,
                "action_source": "raw_pi05_executed_actions_exact_archive",
                "action": action.tolist(),
                "pre_integration_state_sha256": _integration_hash(before_state),
                "post_integration_state_sha256": _integration_hash(post_snapshot_state),
                "pre_robot_states": before_state.get("robot_states"),
                "post_robot_states": post_snapshot_state.get("robot_states"),
                "pre_object_states": before_state.get("object_states"),
                "post_object_states": post_snapshot_state.get("object_states"),
                "grasp_inventory_complete": grasp_complete,
                "grasp_states_post": grasp_state,
                "held_object_post": held,
                "native_task_status_post": post_native,
                "reward": float(reply.get("reward", 0.0)),
                "done": bool(reply.get("done", False)),
                "contact_monitoring_status": monitor.get("status"),
                "captured_physics_substeps": monitor.get("captured_physics_substeps"),
                "native_safety_pass": bool(complete and action_native),
                "operational_safety_pass": bool(complete and action_operational),
                "strict_10mm_additional_metric_pass": bool(complete and strict10),
                "minimum_clearance_by_object_m": min_clearance,
                "first_failure_substep": first_failure,
                "substep_rows": action_substeps,
            }
            rows.append(record)
            report["physical_action_count"] += 1
            report["physical_substep_count"] += len(substeps)
            action_log.write(json.dumps(_jsonable(record), sort_keys=True, allow_nan=False) + "\n")
            action_log.flush()
            for item in action_substeps:
                substep_log.write(json.dumps(_jsonable(item), sort_keys=True, allow_nan=False) + "\n")
            substep_log.flush()
            pre_agent.append(_camera_frame(before, "agentview_image"))
            post_agent.append(_camera_frame(reply, "agentview_image"))
            pre_wrist.append(_camera_frame(before, "robot0_eye_in_hand_image"))
            post_wrist.append(_camera_frame(reply, "robot0_eye_in_hand_image"))
            pre_state.append(np.asarray(before_state["mujoco_integration_state"], dtype=np.float64).tolist())
            post_integration_states.append(np.asarray(post_snapshot_state["mujoco_integration_state"], dtype=np.float64).tolist())
            snap = bridge.request({"op": "export_runtime_snapshot"})
            if snap.get("status") != "complete":
                raise RuntimeError(f"snapshot_export_failed_at_t{post_timestep}")
            last_payload = snap["payload"]
            (snapshots_dir / f"timestep_{post_timestep:06d}.pkl").write_bytes(last_payload)
            if not (complete and action_native and action_operational):
                report["outcome"] = "stopped_at_first_current_safety_failure"
                report["failure"] = {"timestep": post_timestep, "first_failure_substep": first_failure}
                break
            if bool(reply.get("done", False)):
                report["outcome"] = "unexpected_done_before_t10"
                report["failure"] = {"timestep": post_timestep, "done": True}
                break

        if last_payload is None:
            last_payload = start_payload
        terminal = bridge.request({"op": "observe", "include_snapshot_state": True})
        terminal_runtime = _snapshot_runtime_fields(last_payload)
        source_t10_runtime = _snapshot_runtime_fields(T10_SNAPSHOT.read_bytes())
        final_integration_error = _max_abs(
            terminal.get("snapshot_state", {}).get("mujoco_integration_state"),
            source_t10_runtime.get("integration_state"),
        )
        terminal_runtime_comparison = _runtime_state_comparison(source_t10_runtime, terminal_runtime)
        report["terminal_comparison"] = {
            "timestep": int(terminal.get("timestep", -1)),
            "t10_source_integration_state_max_abs_error": final_integration_error,
            "t10_source_integration_state_exact": final_integration_error == 0.0,
            "source_t10_runtime_components": terminal_runtime_comparison,
            "terminal_snapshot_sha256": hashlib.sha256(last_payload).hexdigest(),
            "t10_source_snapshot_sha256": _sha256(T10_SNAPSHOT),
            "native_task_status": bridge.request({"op": "native_task_status"}),
            "grasp_state": _grasp_state(bridge, backend),
        }
        actions_total = 1 + len(rows)
        substeps_total = len(prior_substeps) + len(substep_rows)
        exact_t10 = (
            int(terminal.get("timestep", -1)) == 10
            and final_integration_error == 0.0
            and terminal_runtime_comparison["status"] == "pass"
        )
        prefix_pass = actions_total == 10 and substeps_total == 250 and exact_t10 and all(
            row.get("native_safety_pass") and row.get("operational_safety_pass")
            for row in prior_action_records + rows
        ) and all(
            row.get("native_safety_pass") and row.get("operational_safety_pass")
            for row in prior_substeps + substep_rows
        )
        if report["outcome"] is None:
            report["outcome"] = "t0_to_t10_continuous_snapshot_resumed_current_safety_pass" if prefix_pass else "t0_to_t10_acceptance_failed"
        report["acceptance"] = {
            "actions_total_t0_to_t10": actions_total,
            "physical_substeps_total": substeps_total,
            "all_250_current_policy_substeps_complete": substeps_total == 250 and all(
                item.get("geometry", {}).get("status") in {"complete", "unsafe"}
                for item in prior_substeps + substep_rows
            ),
            "all_substeps_native_and_operational_pass": prefix_pass,
            "t0_to_t1_exact_runtime_snapshot_resume_verified": bool(
                report.get("t1_restore_validation", {}).get("integration_state_exact")
                and report.get("t1_restore_validation", {}).get("controller_gripper_buffers_and_task_bookkeeping", {}).get("status") == "pass"
            ),
            "t10_full_runtime_state_exact_to_episode9_source": exact_t10,
            "prefix_t0_to_t10_pass": prefix_pass,
        }
        report["strict_metric_summary"] = {
            "strict_10mm_is_additional_only": True,
            "strict_10mm_action_passes_across_full_prefix": all(
                row.get("strict_10mm_additional_metric_pass") for row in prior_action_records + rows
            ),
            "operational_action_passes_across_full_prefix": all(
                row.get("operational_safety_pass") for row in prior_action_records + rows
            ),
            "strict_10mm_does_not_override_operational_result": True,
        }
        (snapshots_dir / "start_timestep_000001.pkl").write_bytes(start_payload)
        end_step = int(terminal.get("timestep", 1))
        (snapshots_dir / f"end_timestep_{end_step:06d}.pkl").write_bytes(last_payload)
    except Exception as exc:
        report["outcome"] = report.get("outcome") or "failed_or_stopped_with_evidence"
        report["failure"] = report.get("failure") or {
            "exception_type": type(exc).__name__, "message": str(exc),
            "completed_actions": int(report.get("physical_action_count", 0)),
        }
        report["exception_traceback"] = __import__("traceback").format_exc()
    finally:
        if rows:
            prev = np.load(PRIOR_T0_T1_DIR / "observation_action_trace.npz", allow_pickle=False)
            combined_pre_agent = np.concatenate([prev["pre_agentview"], np.asarray(pre_agent, dtype=np.uint8)])
            combined_post_agent = np.concatenate([prev["post_agentview"], np.asarray(post_agent, dtype=np.uint8)])
            combined_pre_wrist = np.concatenate([prev["pre_eye_in_hand"], np.asarray(pre_wrist, dtype=np.uint8)])
            combined_post_wrist = np.concatenate([prev["post_eye_in_hand"], np.asarray(post_wrist, dtype=np.uint8)])
            combined_pre_int = np.concatenate([prev["pre_integration_state"], np.asarray(pre_state, dtype=np.float64)])
            combined_post_int = np.concatenate([prev["post_integration_state"], np.asarray(post_integration_states, dtype=np.float64)])
            prev.close()
            full_actions = np.concatenate([prior_action, actions[:len(rows)]], axis=0)
            np.savez_compressed(
                output_dir / "observation_action_trace.npz",
                actions=full_actions,
                timestep=np.arange(1, len(full_actions) + 1, dtype=np.int64),
                pre_integration_state=combined_pre_int,
                post_integration_state=combined_post_int,
                pre_agentview=combined_pre_agent,
                post_agentview=combined_post_agent,
                pre_eye_in_hand=combined_pre_wrist,
                post_eye_in_hand=combined_post_wrist,
                metadata_json=np.asarray(json.dumps({
                    "schema": "gs_opd_round0_raw_pi05_t0_to_t10_current_safety.v1",
                    "physical_segment_stitch": "one_t1_RuntimeSnapshot_restore_exactly_validated_before_actions_1_to_9",
                    "raw_action_source_exact": True,
                    "action_count": len(full_actions),
                }, sort_keys=True)),
            )
            _write_video(output_dir / "agentview.mp4", [combined_pre_agent[0]] + list(combined_post_agent))
            _write_video(output_dir / "wrist.mp4", [combined_pre_wrist[0]] + list(combined_post_wrist))
            (snapshots_dir / "start_timestep_000000.pkl").write_bytes(
                (PRIOR_T0_T1_DIR / "snapshots/start_timestep_000000.pkl").read_bytes()
            )
            (snapshots_dir / "timestep_000001.pkl").write_bytes(prior_t1_snapshot)
        report["archive_dir"] = str(output_dir)
        report["physical_wall_seconds"] = time.monotonic() - started
        _dump(output_dir / "t0_to_t10_current_safety_report.json", report)
        if action_log is not None:
            action_log.close()
        if substep_log is not None:
            substep_log.close()
        bridge.close()
        checks = {}
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.name != "sha256_manifest.json":
                checks[str(path.relative_to(output_dir))] = _sha256(path)
        _dump(output_dir / "sha256_manifest.json", checks)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38785)
    report = run(parser.parse_args())
    print(json.dumps({
        "outcome": report.get("outcome"),
        "physical_action_count_continued": report.get("physical_action_count"),
        "physical_substep_count_continued": report.get("physical_substep_count"),
        "acceptance": report.get("acceptance"),
        "archive_dir": report.get("archive_dir"),
        "failure": report.get("failure"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("outcome") == "t0_to_t10_continuous_snapshot_resumed_current_safety_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
