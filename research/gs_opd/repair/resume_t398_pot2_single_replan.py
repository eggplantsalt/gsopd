# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Continue the safe t359→t398 route prefix after its first frozen tracking stop.

This entry point restores the exact durable t398 Snapshot, makes exactly one
existing RRT-Connect request from that state, and uses the unused portion of
the original 450-control-action authorization. It does not replay the failed
route prefix or start Pot 1.
"""

from __future__ import annotations

import argparse
import __main__
import json
import math
import pickle
import signal
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from research.gs_opd.repair.automatic_recovery import (
    AutomaticSafetyRecoveryEngine,
    RecoveryArtifactWriter,
)
from research.gs_opd.repair.automatic_recovery_runtime import (
    AutomaticRecoveryBridgeBackend,
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
    sha256_file,
)
from research.gs_opd.repair.run_t329_native_task_suffix import (
    ARCHIVED_ROUTE_DIR,
    CONFIG_PATH,
    EXPECTED_SUBSTEPS,
    MODEL_SHA256,
    OBJECT_ID,
    SOURCE_REPORT_PATH,
    DurableTaskSuffixBackend,
    _all_robot_contacts_finger_target,
    _canonical_sha256,
    _load_archived_route,
    _suffix_acceptance,
    _write_json,
)
from research.gs_opd.repair.resume_t359_pot2_placement import (
    LEFT_FINGER_GEOMS,
    RIGHT_FINGER_GEOMS,
    T359ResumeBackend,
    _PythonObjectState,
    _contact_sides,
    _normalized_state_hash,
)


FIRST_RUN = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_round0_t359_pot2_placement_20261024_213308"
)
FIRST_RUN_ACTIONS = 39
FIRST_STOP_TIMESTEP = 398
ORIGINAL_ACTION_LIMIT = 450
REMAINING_ACTION_LIMIT = ORIGINAL_ACTION_LIMIT - FIRST_RUN_ACTIONS
MAX_TASK_TIMESTEP = 1800
STANDARD_HORIZON = 1000
EXPECTED_FROZEN_TRACKING_STOP_RAD = 0.18


class T398SingleReplanBackend(T359ResumeBackend):
    """Reuse normal task control and permit one fresh existing RRT call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner_calls = 0

    def plan_rrt_connect(self, observation, goal, context, config):
        if self.planner_calls >= 1:
            raise RuntimeError("t398_single_existing_rrt_call_budget_exhausted")
        self.planner_calls += 1
        # Call the existing native Bridge-backed RRT-Connect implementation.
        # No cached t329 route is reused after its measured joint drift stop.
        return AutomaticRecoveryBridgeBackend.plan_rrt_connect(
            self, observation, goal, context, config
        )


def _parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=FIRST_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38744)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def _read_scalar_json(archive: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    value = archive[key]
    if value.shape != ():
        raise ValueError(f"t398_durable_journal_{key}_is_not_scalar")
    raw = value.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"t398_durable_journal_{key}_is_not_json_text")
    decoded = json.loads(raw)
    # DurableTaskSuffixBackend JSON-encodes already-serialized *_json fields
    # once more when writing the one-action journal.
    if isinstance(decoded, str):
        decoded = json.loads(decoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"t398_durable_journal_{key}_did_not_decode_to_an_object")
    return decoded


def _load_t398_source(source_run: Path) -> dict[str, Any]:
    source_run = Path(source_run)
    result = json.loads((source_run / "suffix_result.json").read_text(encoding="utf-8"))
    if result.get("physical_actions") != FIRST_RUN_ACTIONS:
        raise ValueError("t398_source_action_count_does_not_match_prior_stop_record")
    if result.get("final_timestep") != FIRST_STOP_TIMESTEP:
        raise ValueError("t398_source_terminal_timestep_mismatch")
    if result.get("suffix_native_safety") is not True or result.get(
        "suffix_operational_safety"
    ) is not True:
        raise ValueError("t398_source_safe_prefix_attestation_failed")
    if result.get("reason") != (
        "rrt_connect_failed:RuntimeError:resume_route_replan_not_authorized_without_new_evidence"
    ):
        raise ValueError("t398_source_did_not_stop_at_the_expected_archived_route_deviation")

    index_path = source_run / "durable_action_journal" / "index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    if len(rows) != FIRST_RUN_ACTIONS or rows[-1].get("timestep") != FIRST_STOP_TIMESTEP:
        raise ValueError("t398_source_durable_action_journal_is_not_complete")
    last = rows[-1]
    if last.get("task_phase") != "transport":
        raise ValueError("t398_source_terminal_phase_is_not_transport")
    snapshot_path = Path(last["runtime_snapshot_path"])
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    journal_path = Path(last["journal_path"])
    if sha256_file(journal_path) != last.get("journal_sha256"):
        raise ValueError("t398_source_terminal_action_journal_hash_mismatch")
    with np.load(journal_path, allow_pickle=False) as journal:
        physical = _read_scalar_json(journal, "post_physical_state_json")
        native_status = _read_scalar_json(journal, "post_native_task_status_json")
        contact_monitor = _read_scalar_json(journal, "post_contact_monitoring_json")
    if int(native_status.get("timestep", -1)) != FIRST_STOP_TIMESTEP:
        raise ValueError("t398_terminal_native_record_timestep_mismatch")
    if native_status.get("task_success") is not False:
        raise ValueError("t398_native_task_status_is_not_pending")
    if contact_monitor.get("status") != "complete":
        raise ValueError("t398_terminal_action_substep_monitor_is_incomplete")
    substeps = contact_monitor.get("geometry_monitoring", {}).get("substeps", [])
    if len(substeps) != EXPECTED_SUBSTEPS or any(
        item.get("status") != "complete"
        or any(item.get(key) for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
        ))
        for item in substeps
    ):
        raise ValueError("t398_terminal_action_did_not_pass_all_25_safety_substeps")

    envelope = pickle.loads(snapshot_path.read_bytes())
    if envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
        raise ValueError("t398_runtime_snapshot_schema_mismatch")
    if envelope.get("model_geometry_sha256") != MODEL_SHA256:
        raise ValueError("t398_runtime_snapshot_model_fingerprint_mismatch")
    snapshot = envelope.get("snapshot")
    if not isinstance(snapshot, dict) or int(
        snapshot.get("task_bookkeeping", {}).get("timestep", -1)
    ) != FIRST_STOP_TIMESTEP:
        raise ValueError("t398_runtime_snapshot_task_timestep_mismatch")
    integration = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    expected_integration = np.asarray(
        physical.get("mujoco_integration_state"),
        dtype=np.float64,
    )
    integration_sha = __import__("hashlib").sha256(
        np.ascontiguousarray(integration).tobytes()
    ).hexdigest()
    expected_integration_sha = __import__("hashlib").sha256(
        np.ascontiguousarray(expected_integration).tobytes()
    ).hexdigest()
    if integration_sha != expected_integration_sha:
        raise ValueError("t398_snapshot_integration_state_differs_from_terminal_action_record")

    robot = physical["robot_states"][0]
    object_state = physical["object_states"][f"objects_dict:{OBJECT_ID}"]
    robot_runtime = snapshot["robots"][0]
    if not robot_runtime.get("has_gripper_current_action"):
        raise ValueError("t398_snapshot_has_no_native_gripper_internal_action")
    attestation = {
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(snapshot_path),
        "source_snapshot_sha256": sha256_file(snapshot_path),
        "source_action_record": str(journal_path),
        "source_action_record_sha256": last["journal_sha256"],
        "timestep": FIRST_STOP_TIMESTEP,
        "model_geometry_sha256": MODEL_SHA256,
        "object_id": OBJECT_ID,
        "arm_joint_qpos": robot["joint_qpos"],
        "eef_position_world_m": robot["eef_pos"],
        "eef_quaternion_wxyz": robot["eef_quat"],
        "gripper_qpos_m": robot["gripper_qpos"],
        "object_position_world_m": object_state["root_pos"],
        "object_quaternion_wxyz": object_state["root_quat"],
        "restored_gripper_current_action": np.asarray(
            robot_runtime["gripper_current_action"], dtype=np.float64
        ).tolist(),
        "integration_state_sha256": integration_sha,
        "runtime_controller_state_sha256": _normalized_state_hash(
            robot_runtime["controller"]
        ),
        "source_t359_run": str(source_run),
        "source_t359_actions": FIRST_RUN_ACTIONS,
    }
    policy = result["suffix_acceptance"]["policy"]
    if _canonical_sha256(policy) != "ee2498380261f31ffd2fd83cc3386f97b26f7feeae7929a3887d47b14698db2d":
        raise ValueError("t398_frozen_operational_safety_policy_fingerprint_mismatch")
    if result["suffix_acceptance"].get("operational_joint_limit_violations") != 0:
        raise ValueError("t398_source_prefix_has_operational_joint_limit_violations")
    if result["suffix_acceptance"].get("observed_substeps") != FIRST_RUN_ACTIONS * EXPECTED_SUBSTEPS:
        raise ValueError("t398_source_prefix_substep_coverage_is_incomplete")

    return {
        "source_run": str(source_run),
        "source_result": result,
        "source_index": rows,
        "last_action": last,
        "snapshot_path": snapshot_path,
        "snapshot": snapshot,
        "attestation": attestation,
        "policy": policy,
        "native_task_status": native_status,
        "terminal_physical_state": physical,
        "integration_state_sha256": integration_sha,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    __main__._PythonObjectState = _PythonObjectState
    source = _load_t398_source(args.source_run)
    if args.audit_only:
        return {
            "status": "t398_snapshot_source_verified_for_one_bounded_replan",
            "source_run": source["source_run"],
            "source_snapshot": source["attestation"],
            "actions_already_executed_in_authorized_segment": FIRST_RUN_ACTIONS,
            "remaining_authorized_actions": REMAINING_ACTION_LIMIT,
            "previous_suffix_safety": {
                "native": source["source_result"].get("suffix_native_safety"),
                "operational": source["source_result"].get("suffix_operational_safety"),
                "strict_10mm": source["source_result"].get("suffix_strict_10mm"),
            },
        }

    source_report = json.loads(SOURCE_REPORT_PATH.read_text(encoding="utf-8"))
    config_data, recovery = load_runtime_config(CONFIG_PATH)
    config_data = json.loads(json.dumps(config_data))
    config_data["candidate_search"]["enabled"] = False
    policy = source["policy"]
    object_clearances = {
        str(item["object_id"]): float(item["minimum_clearance_m"])
        for item in policy["allowed_target_objects"]
    }
    recovery = replace(
        recovery,
        hard_minimum_clearance_m=float(policy["minimum_clearance_m"]),
        target_object_clearance_m=float(policy["minimum_clearance_m"]),
        object_clearance_m=object_clearances,
        execution_step_budget=FIRST_STOP_TIMESTEP + REMAINING_ACTION_LIMIT,
        horizon_steps=MAX_TASK_TIMESTEP,
        standard_horizon_steps=STANDARD_HORIZON,
        allow_horizon_extension=True,
        # The native RRT search remains capped at 360 s. This separate outer
        # allowance prevents result packaging from invalidating a completed
        # route, as happened when the prior wall time was 360.055 s.
        planning_time_budget_s=420.0,
        maximum_replans=1,
        maximum_replans_per_phase=1,
    )
    recovery.validate()
    config_data["recovery"] = {
        **vars(recovery),
        "object_clearance_m": dict(recovery.object_clearance_m),
    }

    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(
        f"/opt/data/private/gs_opd/rollouts/l0t2_round0_t398_pot2_single_replan_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    route = _load_archived_route(ARCHIVED_ROUTE_DIR)
    _write_json(output_dir / "t398_source_audit.json", {
        "source_run": source["source_run"],
        "source_attestation": source["attestation"],
        "previous_stop": {
            "reason": source["source_result"].get("reason"),
            "final_timestep": FIRST_STOP_TIMESTEP,
            "physical_actions": FIRST_RUN_ACTIONS,
            "native_safety": source["source_result"].get("suffix_native_safety"),
            "operational_safety": source["source_result"].get("suffix_operational_safety"),
            "strict_10mm": source["source_result"].get("suffix_strict_10mm"),
        },
        "planner_policy": {
            "tool": "existing_native_bridge_rrt_connect",
            "maximum_calls": 1,
            "rrt_search_time_budget_s": config_data["rrt_connect"]["search_time_budget_s"],
            "outer_result_packaging_budget_s": recovery.planning_time_budget_s,
            "safety_thresholds_changed": False,
            "grasp_search_repeated": False,
        },
        "physical_budget": {
            "actions_already_executed": FIRST_RUN_ACTIONS,
            "remaining_action_budget": REMAINING_ACTION_LIMIT,
            "combined_maximum": ORIGINAL_ACTION_LIMIT,
            "absolute_timestep_limit": recovery.execution_step_budget,
        },
    })
    _write_json(output_dir / "continuation_chain_start.json", {
        "same_logical_episode": True,
        "previous_archive": source["source_run"],
        "previous_terminal_timestep": FIRST_STOP_TIMESTEP,
        "previous_terminal_snapshot": source["attestation"]["source_snapshot_path"],
        "previous_terminal_snapshot_sha256": source["attestation"]["source_snapshot_sha256"],
        "restored_integration_state_sha256": source["integration_state_sha256"],
        "no_action_replay": True,
        "no_qpos_edit": True,
    })

    artifacts = RecoveryArtifactWriter(output_dir)
    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend: T398SingleReplanBackend | None = None
    engine: AutomaticSafetyRecoveryEngine | None = None
    result_json: dict[str, Any] | None = None
    original_handlers = {}
    video_released = False

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal_{signum}_received_after_durable_action_boundary")

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.signal(signum, interrupt)
    started = time.monotonic()
    try:
        bridge.start(
            benchmark=config_data["benchmark"], level=int(config_data["level"]),
            task_id=int(config_data["task_id"]), init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]), demonstration_horizon=MAX_TASK_TIMESTEP,
        )
        backend = T398SingleReplanBackend(
            bridge,
            config_data,
            recovery,
            output_dir,
            artifacts,
            mode="task-runtime-snapshot",
            student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=source["snapshot_path"],
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
            runtime_snapshot_attestation=source["attestation"],
            journal_dir=output_dir / "durable_action_journal",
            cached_route=route,
            resume_sources={},
        )
        restored = backend.restore_start_state()
        if int(restored.get("timestep", -1)) != FIRST_STOP_TIMESTEP:
            raise RuntimeError("strict_t398_restore_timestep_mismatch")
        if restored.get("scene_metadata", {}).get("model_geometry_sha256") != MODEL_SHA256:
            raise RuntimeError("strict_t398_restore_model_fingerprint_mismatch")
        state_hash = __import__("hashlib").sha256(np.ascontiguousarray(
            np.asarray(restored["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
        ).tobytes()).hexdigest()
        if state_hash != source["integration_state_sha256"]:
            raise RuntimeError("strict_t398_restore_integration_state_mismatch")
        internal = bridge.request({"op": "export_runtime_snapshot"})
        if internal.get("status") != "complete":
            raise RuntimeError("strict_t398_restore_runtime_internal_export_unavailable")
        envelope = pickle.loads(internal["payload"])
        internal_hash = _normalized_state_hash(envelope.get("snapshot", {}))
        expected_internal_hash = _normalized_state_hash(source["snapshot"])
        if internal_hash != expected_internal_hash:
            raise RuntimeError("strict_t398_restore_controller_or_gripper_internal_state_mismatch")

        native = bridge.request({"op": "native_task_status"})
        if native.get("status") != "complete" or native.get("task_success") is not False:
            raise RuntimeError("t398_native_task_state_is_not_pending_and_known")
        preflight = bridge.request({"op": "observe", "include_snapshot_state": True})
        backend._load_context_inputs(preflight)
        backend.latest_data = preflight
        backend.safety_policy["joint_limit_tolerance_policy"] = policy[
            "joint_limit_tolerance_policy"
        ]
        backend.safety_policy["source"] = policy["source"]
        backend.latest_data["native_task_status"] = native
        contacts = bridge.request({"op": "inspect_contacts"})
        backend.latest_data["contacts"] = contacts
        context = backend.current_context()
        if context.phase != "transport" or context.held_object != OBJECT_ID:
            raise RuntimeError("t398_live_task_phase_or_held_object_mismatch")
        if _canonical_sha256(backend.safety_policy) != _canonical_sha256(policy):
            active = backend.safety_policy
            differing = sorted(
                key for key in set(active) | set(policy)
                if _canonical_sha256({key: active.get(key)})
                != _canonical_sha256({key: policy.get(key)})
            )
            raise RuntimeError(
                "t398_live_operational_safety_policy_fingerprint_mismatch:"
                f"active={_canonical_sha256(active)};expected={_canonical_sha256(policy)};"
                f"differing_top_level_fields={differing}"
            )
        grasp = context.metadata.get("held_object_state_evidence", {}).get(OBJECT_ID, {})
        if (
            grasp.get("grasp_confirmed") is not True
            or grasp.get("finger_contacts", {}).get("left") is not True
            or grasp.get("finger_contacts", {}).get("right") is not True
            or not _all_robot_contacts_finger_target(contacts, backend)
        ):
            raise RuntimeError("t398_live_finger_only_bilateral_grasp_preflight_failed")
        geometry_data = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        geometry = geometry_data.get("safety_geometry")
        if not isinstance(geometry, dict) or geometry.get("status") != "complete":
            raise RuntimeError("t398_live_operational_safety_geometry_preflight_incomplete")
        backend.latest_data.update(geometry_data)
        backend.latest_data["native_task_status"] = native
        backend.latest_data["contacts"] = contacts
        observation = backend._observation(backend.latest_data)
        context = backend.current_context()
        if context.phase != "transport" or context.held_object != OBJECT_ID:
            raise RuntimeError("t398_transport_context_changed_during_zero_action_preflight")
        goal = backend.resolve_goal(observation, context)
        if goal.phase != "transport":
            raise RuntimeError("t398_current_goal_is_not_transport")

        preflight_report = {
            "status": "strict_t398_restore_and_live_hold_preflight_passed",
            "restored_timestep": FIRST_STOP_TIMESTEP,
            "model_geometry_sha256": MODEL_SHA256,
            "integration_state_sha256": state_hash,
            "runtime_internal_snapshot_sha256_expected": expected_internal_hash,
            "runtime_internal_snapshot_sha256_restored": internal_hash,
            "held_object": context.held_object,
            "phase": context.phase,
            "pot2_grasp_confirmed": grasp.get("grasp_confirmed"),
            "bilateral_finger_contacts": grasp.get("finger_contacts"),
            "live_robot_contact_count": len(contacts.get("contacts", [])),
            "safety_policy_sha256": _canonical_sha256(backend.safety_policy),
            "remaining_action_budget": REMAINING_ACTION_LIMIT,
            "no_old_route_replayed": True,
            "one_existing_rrt_call_authorized_by_measured_route_deviation": True,
        }
        _write_json(output_dir / "t398_resume_preflight.json", preflight_report)
        artifacts.event({"event": "t398_strict_restore_preflight_passed", **preflight_report})
        backend.initialize_trajectory()

        engine = AutomaticSafetyRecoveryEngine(backend, recovery, artifacts=artifacts)
        engine_result = engine.run()
        result_json = dict(engine_result.__dict__)
        acceptance = _suffix_acceptance(backend.trajectory, backend.safety_policy)
        terminal = backend.observe()
        native_terminal = terminal.raw.get("native_task_status")
        on_value = next((
            item.get("value") for item in (native_terminal or {}).get("native_goal_values", [])
            if tuple(str(token).lower() for token in item.get("predicate", []))
            == ("on", OBJECT_ID, "flat_stove_1_cook_region")
        ), None)
        milestone = backend._milestone_status
        total_actions = FIRST_RUN_ACTIONS + backend.transition_count
        if total_actions > ORIGINAL_ACTION_LIMIT:
            raise RuntimeError("combined_t359_t398_action_budget_exceeded")
        result_json.update({
            "status": engine_result.status,
            "reason": engine_result.reason,
            "physical_actions_this_continuation": backend.transition_count,
            "physical_actions_prior_t359_to_t398": FIRST_RUN_ACTIONS,
            "physical_actions_combined_authorized_segment": total_actions,
            "combined_action_limit": ORIGINAL_ACTION_LIMIT,
            "planner_calls": backend.planner_calls,
            "native_task_status": native_terminal,
            "native_task_success": terminal.task_success,
            "pot2_native_on_predicate": on_value,
            "pot2_local_placement_success": bool(milestone and on_value is True),
            "pot2_milestone_evidence": milestone,
            "initial_timestep": FIRST_STOP_TIMESTEP,
            "final_timestep": terminal.timestep,
            "suffix_native_safety": acceptance["native_safety"],
            "suffix_operational_safety": acceptance["operational_safety"],
            "suffix_strict_10mm": acceptance["strict_10mm"],
            "suffix_acceptance": acceptance,
            "complete_task_success": terminal.task_success is True,
            "complete_round0_seed": False,
            "pot1_phase_started": any(
                event.get("event") == "task_phase_transition"
                and event.get("to_phase") == "second_object_recovery"
                for event in engine._events
            ),
            "source_t359_archive": str(args.source_run),
            "source_t398_snapshot_sha256": source["attestation"]["source_snapshot_sha256"],
            "source_t398_integration_state_sha256": source["integration_state_sha256"],
            "planner_search_calls": backend.planner_calls,
            "failure_category": (
                "none_pot2_local_milestone_reached"
                if milestone and on_value is True
                else "existing_rrt_plan_or_preview_failed"
                if engine_result.reason and (
                    "rrt_connect_failed" in engine_result.reason
                    or "isolated_prefix" in engine_result.reason
                )
                else "route_or_task_stage_budget_or_progress_stop"
            ),
        })
        return result_json
    except BaseException as exc:
        result_json = {
            "status": "interrupted_or_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_timestep": int(backend.latest_data.get("timestep", -1))
                if backend is not None and backend.latest_data else None,
            "physical_transitions_recorded_this_continuation": backend.transition_count
                if backend is not None else 0,
            "physical_actions_prior_t359_to_t398": FIRST_RUN_ACTIONS,
            "physical_action_limit_combined": ORIGINAL_ACTION_LIMIT,
            "planner_calls": backend.planner_calls if backend is not None else 0,
            "preflight_report": str(output_dir / "t398_resume_preflight.json"),
            "complete_round0_seed": False,
        }
        return result_json
    finally:
        if backend is not None and hasattr(backend, "trajectory"):
            try:
                backend.finalize(result_json or {"complete_task_success": False})
                video_released = True
            except Exception as exc:
                (output_dir / "finalize_error.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                )
        if backend is not None and not video_released:
            for writer in backend._video_writers.values():
                writer.release()
        bridge.close()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        if result_json is not None:
            result_json["output_dir"] = str(output_dir)
            result_json["elapsed_seconds"] = time.monotonic() - started
            _write_json(output_dir / "suffix_result.json", result_json)
            _write_json(output_dir / "continuation_chain.json", {
                "same_logical_episode": True,
                "segment_1_archive": str(args.source_run),
                "segment_1_start_timestep": 359,
                "segment_1_terminal_timestep": FIRST_STOP_TIMESTEP,
                "segment_1_physical_actions": FIRST_RUN_ACTIONS,
                "segment_2_archive": str(output_dir),
                "segment_2_start_timestep": FIRST_STOP_TIMESTEP,
                "segment_2_terminal_timestep": result_json.get("final_timestep"),
                "segment_2_physical_actions": result_json.get(
                    "physical_actions_this_continuation",
                    result_json.get("physical_transitions_recorded_this_continuation", 0),
                ),
                "total_physical_actions": result_json.get(
                    "physical_actions_combined_authorized_segment"
                ),
                "action_limit": ORIGINAL_ACTION_LIMIT,
                "state_link_snapshot_sha256": source["attestation"]["source_snapshot_sha256"],
                "state_link_integration_state_sha256": source["integration_state_sha256"],
                "complete_round0_seed": False,
            })


def main() -> int:
    args = _parser()
    try:
        result = run(args)
    except Exception as exc:
        result = {"status": "source_audit_failed", "failure_reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if args.audit_only:
        return 0 if result.get("status") == "t398_snapshot_source_verified_for_one_bounded_replan" else 2
    return 0 if result.get("pot2_local_placement_success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
