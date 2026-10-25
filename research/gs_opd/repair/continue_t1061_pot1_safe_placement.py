# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Continue the verified t1061 Pot 1 grasp through safe native placement.

The historical successful placement is used only through the existing task
goal configuration and its high transport-clearance prior. A bounded RRT
route is generated from the strictly restored live state when the controller
requires one. Every OSC_POSE action is certified in an isolated MuJoCo branch
from the current real state before it is committed.
"""

from __future__ import annotations

import __main__
import argparse
import hashlib
import json
import math
import os
import pickle
import signal
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from research.gs_opd.repair.automatic_recovery import JointSpacePlan, RecoveryArtifactWriter
from research.gs_opd.repair.automatic_recovery_runtime import (
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
    sha256_file,
)
from research.gs_opd.repair.continue_t794_pot1_execution_aware_pregrasp import (
    MODEL_SHA256,
    _normalized_state_sha256,
)
from research.gs_opd.repair.continue_t1019_pot1_execution_aware_grasp_lift import (
    EXPECTED_SUBSTEPS,
    _check_substeps,
    _native_values,
    _object_state,
)
from research.gs_opd.repair.run_t329_native_task_suffix import (
    CONFIG_PATH,
    DurableTaskSuffixBackend,
    _atomic_npz,
    _canonical_sha256,
    _jsonable,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/l0t1042_pot1_grasp_lift_20261025_032402"
)
SOURCE_SNAPSHOT = SOURCE_ARCHIVE / "snapshots/main_timestep_001061.pkl"
SOURCE_TRACE = SOURCE_ARCHIVE / "observation_action_trace.npz"
SOURCE_CONFIG = SOURCE_ARCHIVE / "run_config.json"
SOURCE_RESULT = SOURCE_ARCHIVE / "suffix_result.json"
SOURCE_JOURNAL = SOURCE_ARCHIVE / "durable_action_journal/index.jsonl"
HISTORICAL_SUCCESS = Path(
    "/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_final_placement_demo_20261022_attempt3/evidence/success_acceptance_report.json"
)
INITIAL_TIMESTEP = 1061
MAX_DEMONSTRATION_HORIZON = 1800
STANDARD_HORIZON = 1000
RISK_TRIGGER_M = 0.050
RISK_REARM_M = 0.060
OBJECT_ID = "moka_pot_1"
POT2_ID = "moka_pot_2"
POT1_ON = ("on", OBJECT_ID, "flat_stove_1_cook_region")
POT2_ON = ("on", POT2_ID, "flat_stove_1_cook_region")
STOVE_ON = ("turnon", "flat_stove_1")

setattr(__main__, "_PythonObjectState", __import__(
    "research.gs_opd.repair.continue_t794_pot1_execution_aware_pregrasp",
    fromlist=["_PythonObjectState"],
)._PythonObjectState)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _record_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _source_attestation() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path in (SOURCE_SNAPSHOT, SOURCE_TRACE, SOURCE_CONFIG, SOURCE_RESULT, SOURCE_JOURNAL):
        if not path.is_file():
            raise FileNotFoundError(f"t1061_source_artifact_missing:{path}")
    source_config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    prior_result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    frozen_policy = source_config.get("frozen_operational_safety_policy")
    if (
        prior_result.get("final_timestep") != INITIAL_TIMESTEP
        or prior_result.get("final_phase") != "transport"
        or prior_result.get("Pot1_held_object") != OBJECT_ID
        or prior_result.get("Pot1_grasp_state_verified") is not True
        or prior_result.get("Pot1_lift_rise_from_t1019_m", 0.0) < 0.030
        or prior_result.get("Pot2_On_preserved") is not True
        or not isinstance(frozen_policy, dict)
    ):
        raise RuntimeError("t1061_source_report_does_not_attest_the_required_held_transport_state")
    source_bytes = SOURCE_SNAPSHOT.read_bytes()
    envelope = pickle.loads(source_bytes)
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
    ):
        raise RuntimeError("t1061_snapshot_schema_or_compiled_model_fingerprint_mismatch")
    snapshot = envelope.get("snapshot", {})
    if int(snapshot.get("task_bookkeeping", {}).get("timestep", -1)) != INITIAL_TIMESTEP:
        raise RuntimeError("t1061_snapshot_internal_timestep_mismatch")
    integration = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    if integration.shape != (317,) or not np.isfinite(integration).all():
        raise RuntimeError("t1061_snapshot_Integration_State_invalid")
    robots = snapshot.get("robots", [])
    if len(robots) != 1 or not robots[0].get("has_gripper_current_action"):
        raise RuntimeError("t1061_snapshot_robot_or_gripper_internal_state_missing")

    with np.load(SOURCE_TRACE, allow_pickle=False) as trace:
        if (
            trace["executed_actions"].shape != (19, 7)
            or trace["step_timesteps"].shape != (19,)
            or not np.array_equal(trace["step_timesteps"], np.arange(1043, 1062))
            or len(trace["post_physical_state_json"]) != 19
        ):
            raise RuntimeError("t1042_t1061_physical_OA_segment_is_not_a_contiguous_19_action_prefix")
        terminal_physical = json.loads(str(trace["post_physical_state_json"][-1]))
        terminal_native = json.loads(str(trace["post_native_task_status_json"][-1]))
        terminal_monitor = json.loads(str(trace["post_contact_monitoring_json"][-1]))
        if not np.array_equal(
            np.asarray(terminal_physical.get("mujoco_integration_state"), dtype=np.float64),
            integration,
        ):
            raise RuntimeError("t1061_runtime_snapshot_and_terminal_OA_Integration_State_differ")
        native_map = _native_values(terminal_native)
        if native_map != {POT1_ON: False, POT2_ON: True, STOVE_ON: True}:
            raise RuntimeError(f"t1061_archived_native_predicates_mismatch:{native_map}")
        if terminal_native.get("task_success") is not False or terminal_native.get("done") is not False:
            raise RuntimeError("t1061_source_is_not_a_pending_native_task_state")
        if (
            terminal_monitor.get("status") != "complete"
            or terminal_monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
            or len(terminal_monitor.get("geometry_monitoring", {}).get("substeps", [])) != EXPECTED_SUBSTEPS
        ):
            raise RuntimeError("t1061_terminal_OA_action_lacks_complete_25_substep_audit")
        camera_keys = ("post_agentview_image", "post_robot0_eye_in_hand_image")
        if any(key not in trace or trace[key].shape[0] != 19 or not trace[key][-1].size for key in camera_keys):
            raise RuntimeError("t1061_dual_camera_OA_evidence_is_incomplete")
        # Re-audit each action with the exact policy archived by the source run.
        for index in range(19):
            monitor = json.loads(str(trace["post_contact_monitoring_json"][index]))
            substeps = monitor.get("geometry_monitoring", {}).get("substeps", [])
            if (
                monitor.get("status") != "complete"
                or monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
                or len(substeps) != EXPECTED_SUBSTEPS
            ):
                raise RuntimeError(f"t{int(trace['step_timesteps'][index])}_source_substep_audit_incomplete")
            _check_substeps(substeps, frozen_policy)

    journal_rows = [
        json.loads(line) for line in SOURCE_JOURNAL.read_text(encoding="utf-8").splitlines() if line
    ]
    terminal_rows = [row for row in journal_rows if int(row.get("timestep", -1)) == INITIAL_TIMESTEP]
    if len(journal_rows) != 19 or len(terminal_rows) != 1:
        raise RuntimeError("t1061_durable_journal_action_count_or_terminal_row_mismatch")
    terminal = terminal_rows[0]
    if (
        Path(terminal.get("runtime_snapshot_path", "")) != SOURCE_SNAPSHOT
        or _sha(Path(terminal["journal_path"]).read_bytes()) != terminal.get("journal_sha256")
        or terminal.get("task_phase") != "lift"
    ):
        raise RuntimeError("t1061_snapshot_not_attested_by_terminal_durable_action")
    for index, row in enumerate(journal_rows):
        if int(row.get("timestep", -1)) != 1043 + index:
            raise RuntimeError("t1061_durable_action_journal_is_not_contiguous")

    state = terminal_physical
    robot = state["robot_states"][0]
    pot = _object_state(state, OBJECT_ID)
    pot2 = _object_state(state, POT2_ID)
    attestation = {
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(SOURCE_SNAPSHOT),
        "source_snapshot_sha256": _sha(source_bytes),
        "source_action_trace_sha256": sha256_file(SOURCE_TRACE),
        "source_model_geometry_sha256": MODEL_SHA256,
        "timestep": INITIAL_TIMESTEP,
        "model_geometry_sha256": MODEL_SHA256,
        "integration_state_sha256": _array_sha(integration.astype(np.float64)),
        "arm_joint_qpos": robot["joint_qpos"],
        "eef_position_world_m": robot["eef_pos"],
        "eef_quaternion_wxyz": robot["eef_quat"],
        "gripper_qpos_m": robot["gripper_qpos"],
        "object_id": OBJECT_ID,
        "object_position_world_m": pot["root_pos"],
        "object_quaternion_wxyz": pot["root_quat"],
        "pot2_position_world_m": pot2["root_pos"],
        "pot2_quaternion_wxyz": pot2["root_quat"],
        "native_status": terminal_native,
        "prior_source_result": str(SOURCE_RESULT),
        "prior_actions_fully_substep_audited_in_this_entry": 19,
    }
    return attestation, frozen_policy, source_config


def _committed_action_resume_attestation(
    run_dir: Path,
    base_attestation: dict[str, Any],
    frozen_policy: dict[str, Any],
    source_config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Validate a durable same-Episode tail before restoring its terminal state."""
    run_dir = run_dir.resolve()
    trace_path = run_dir / "observation_action_trace.npz"
    result_path = run_dir / "suffix_result.json"
    route_path = run_dir / "reused_transport_route.npz"
    if not route_path.is_file():
        route_path = run_dir / "transport_route.npz"
    journal_path = run_dir / "durable_action_journal/index.jsonl"
    cert_path = run_dir / "execution_certification.jsonl"
    for path in (trace_path, result_path, route_path, journal_path, cert_path):
        if not path.is_file():
            raise FileNotFoundError(f"resume_artifact_missing:{path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    action_count = int(result.get("physical_actions_committed", 0))
    if action_count < 1:
        raise RuntimeError("resume_run_has_no_committed_physical_actions")
    known_reporting_failures = (
        "prediction_commit_observable_state_mismatch",
        "live_object_state_not_unique",
        "name 'prediction_step' is not defined",
        "existing_RRT_route_ended_outside_live_goal_tolerance",
        "predicted_branch_released_Pot1_without_native_On",
    )
    if not any(tag in result.get("failure_reason", "") for tag in known_reporting_failures):
        raise RuntimeError("resume_run_failure_is_not_one_of_the_audited_reporting_interface_errors")
    cert_rows = [json.loads(line) for line in cert_path.read_text(encoding="utf-8").splitlines() if line]
    action_certs = [row for row in cert_rows if row.get("event") == "action_certified_for_commit"]
    physical_commits = [row for row in cert_rows if row.get("event") == "physical_action_committed_and_audited"]
    if len(action_certs) != action_count:
        raise RuntimeError("resume_run_action_certification_count_does_not_match_committed_steps")
    if len(physical_commits) not in ({0, 1} if action_count == 1 else {action_count}):
        raise RuntimeError("resume_run_physical_commit_audit_count_is_incomplete")
    for row in physical_commits:
        audit = row.get("actual_substep_audit", {})
        compare = row.get("physics_prediction_commit", {})
        if (
            audit.get("substeps") != EXPECTED_SUBSTEPS
            or audit.get("failures") != []
            or compare.get("Integration_State_match_atol_1e-9") is not True
            or compare.get("all_native_predicates_match") is not True
            or compare.get("native_task_success_match") is not True
            or compare.get("contact_geom_pair_set_match") is not True
            or compare.get("reward_match") is not True
            or compare.get("done_match") is not True
        ):
            raise RuntimeError("resume_run_contains_a_failed_or_incomplete_committed_step_audit")
    cert = action_certs[-1]
    with np.load(trace_path, allow_pickle=False) as trace:
        initial_timestep = int(result.get("initial_timestep", -1))
        expected_timesteps = np.arange(initial_timestep + 1, initial_timestep + action_count + 1)
        if (
            trace["step_timesteps"].shape != (action_count,)
            or not np.array_equal(trace["step_timesteps"], expected_timesteps)
            or trace["executed_actions"].shape != (action_count, 7)
        ):
            raise RuntimeError("resume_run_OA_timesteps_are_not_a_contiguous_committed_tail")
        resume_timestep = int(trace["step_timesteps"][-1])
        if (
            result.get("failure_timestep") != resume_timestep
            or any(key not in trace or trace[key].shape[0] != action_count
                   for key in ("post_agentview_image", "post_robot0_eye_in_hand_image"))
        ):
            raise RuntimeError("resume_run_OA_or_dual_camera_record_is_not_one_contiguous_action")
        actual_native = json.loads(str(trace["post_native_task_status_json"][-1]))
        actual_monitor = json.loads(str(trace["post_contact_monitoring_json"][-1]))
        physical = json.loads(str(trace["post_physical_state_json"][-1]))
        action = np.asarray(trace["executed_actions"][-1], dtype=np.float64)
    if cert.get("timestep") != resume_timestep - 1:
        raise RuntimeError("resume_terminal_action_certification_timestep_mismatch")
    if cert.get("action") != action.tolist():
        raise RuntimeError("resume_terminal_certified_and_committed_actions_differ")
    snapshot_path = run_dir / f"snapshots/main_timestep_{resume_timestep:06d}.pkl"
    predicted_path = run_dir / f"predicted_branch_snapshots/predicted_timestep_{resume_timestep:06d}.pkl"
    for path in (snapshot_path, predicted_path):
        if not path.is_file():
            raise FileNotFoundError(f"resume_action_snapshot_missing:{path}")
    predicted_native = cert.get("predicted_native_task_status")
    if not isinstance(predicted_native, dict) or predicted_native != actual_native:
        raise RuntimeError("resume_run_predicted_and_committed_native_status_are_not_identical")
    if _native_values(actual_native) != {POT1_ON: False, POT2_ON: True, STOVE_ON: True}:
        raise RuntimeError(f"resume_t{resume_timestep}_native_predicates_are_not_the_expected_pending_Pot1_state")
    if actual_native.get("task_success") is not False or actual_native.get("done") is not False:
        raise RuntimeError(f"resume_t{resume_timestep}_native_status_is_terminal_or_unknown")
    if actual_monitor.get("status") != "complete" or actual_monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS:
        raise RuntimeError(f"resume_t{resume_timestep}_committed_action_substep_monitor_is_incomplete")
    actual_substeps = actual_monitor.get("geometry_monitoring", {}).get("substeps", [])
    if len(actual_substeps) != EXPECTED_SUBSTEPS:
        raise RuntimeError(f"resume_t{resume_timestep}_committed_action_does_not_have_25_geometry_substeps")
    if action_count == 1:
        _check_substeps(actual_substeps, frozen_policy)

    with open(snapshot_path, "rb") as stream:
        actual_envelope = pickle.load(stream)
    with open(predicted_path, "rb") as stream:
        predicted_envelope = pickle.load(stream)
    for label, envelope in (("actual", actual_envelope), ("predicted", predicted_envelope)):
        if envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1" or envelope.get("model_geometry_sha256") != MODEL_SHA256:
            raise RuntimeError(f"resume_t1062_{label}_snapshot_schema_or_model_fingerprint_mismatch")
        if int(envelope.get("snapshot", {}).get("task_bookkeeping", {}).get("timestep", -1)) != resume_timestep:
            raise RuntimeError(f"resume_t{resume_timestep}_{label}_snapshot_timestep_mismatch")
    actual_i = np.asarray(actual_envelope["snapshot"].get("mujoco_state"), dtype=np.float64)
    predicted_i = np.asarray(predicted_envelope["snapshot"].get("mujoco_state"), dtype=np.float64)
    trace_i = np.asarray(physical.get("mujoco_integration_state"), dtype=np.float64)
    if (
        actual_i.shape != (317,) or not np.isfinite(actual_i).all()
        or not np.array_equal(actual_i, predicted_i)
        or not np.array_equal(actual_i, trace_i)
    ):
        raise RuntimeError(f"resume_t{resume_timestep}_committed_and_predicted_Integration_State_mismatch")
    with np.load(route_path, allow_pickle=False) as route:
        route_qpos = np.asarray(route["qpos_path"], dtype=np.float64)
        route_eef = np.asarray(route["eef_reference_poses_world"], dtype=np.float64)
        route_meta = json.loads(str(route["metadata_json"]))
    qpos_actual = np.asarray(physical["robot_states"][0]["joint_qpos"], dtype=np.float64)
    eef_actual = np.asarray(physical["robot_states"][0]["eef_pos"], dtype=np.float64)
    quat_actual = np.asarray(physical["robot_states"][0]["eef_quat"], dtype=np.float64)
    if route_qpos.shape[0] < 2 or route_qpos.shape[1:] != (7,) or route_eef.shape != (len(route_qpos), 7):
        raise RuntimeError("resume_route_shape_or_length_mismatch")
    # Resume only after the last archived route point has a physical commit
    # record. Cursor remains diagnostic if a new live-state plan is needed.
    raw_route_index = cert.get("route_diagnostics", {}).get("route_index")
    expected_route_index = (
        int(raw_route_index) if isinstance(raw_route_index, int) and not isinstance(raw_route_index, bool)
        else None
    )
    if expected_route_index is not None and not 1 <= expected_route_index < len(route_qpos):
        raise RuntimeError("resume_committed_transport_route_cursor_is_invalid")
    eef_position_error = None
    eef_orientation_error = None
    qpos_error = None
    if expected_route_index is not None:
        eef_position_error = float(np.linalg.norm(route_eef[expected_route_index, :3] - eef_actual))
        quat_ref = route_eef[expected_route_index, 3:] / np.linalg.norm(route_eef[expected_route_index, 3:])
        quat_now = quat_actual / np.linalg.norm(quat_actual)
        eef_orientation_error = float(2.0 * math.acos(float(np.clip(abs(np.dot(quat_ref, quat_now)), 0.0, 1.0))))
        qpos_error = float(np.max(np.abs(route_qpos[expected_route_index] - qpos_actual)))
    eef_limit = float(source_config.get("recovery", {}).get("maximum_eef_tracking_error_m", 0.04))
    ori_limit = float(source_config.get("recovery", {}).get("maximum_eef_orientation_error_rad", 0.2))
    if action_count == 1 and expected_route_index is not None and (eef_position_error > eef_limit or eef_orientation_error > ori_limit):
        raise RuntimeError(f"resume_t{resume_timestep}_live_state_does_not_match_committed_transport_route_waypoint")
    original_start_snapshot = Path()
    ancestor = run_dir
    for _ in range(8):
        for candidate in ancestor.glob("source_t*.pkl"):
            candidate_envelope = pickle.loads(candidate.read_bytes())
            candidate_i = np.asarray(candidate_envelope.get("snapshot", {}).get("mujoco_state"), dtype=np.float64)
            if candidate_i.shape == (317,) and _array_sha(candidate_i) == route_meta.get("start_integration_sha256"):
                original_start_snapshot = candidate
                break
        if original_start_snapshot.is_file():
            break
        config_path = ancestor / "run_config.json"
        if not config_path.is_file():
            break
        previous_archive = Path(json.loads(config_path.read_text(encoding="utf-8")).get("source_archive", ""))
        if not str(previous_archive) or previous_archive == ancestor:
            break
        ancestor = previous_archive
    if not original_start_snapshot.is_file():
        raise FileNotFoundError("resume_transport_route_original_start_snapshot_missing")
    original_start_envelope = pickle.loads(original_start_snapshot.read_bytes())
    route_start_i = np.asarray(original_start_envelope.get("snapshot", {}).get("mujoco_state"), dtype=np.float64)
    if _array_sha(route_start_i) != route_meta.get("start_integration_sha256"):
        raise RuntimeError("resume_transport_route_start_fingerprint_does_not_match_original_t1061_state")
    journal_rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line]
    expected_journal_steps = list(range(initial_timestep + 1, resume_timestep + 1))
    if len(journal_rows) != action_count or [int(row.get("timestep", -1)) for row in journal_rows] != expected_journal_steps:
        raise RuntimeError(f"resume_t{resume_timestep}_durable_action_journal_is_not_contiguous")
    for journal in journal_rows:
        journal_npz = Path(journal.get("journal_path", ""))
        if not journal_npz.is_file() or _sha(journal_npz.read_bytes()) != journal.get("journal_sha256"):
            raise RuntimeError(f"resume_durable_action_record_hash_mismatch_t{journal.get('timestep')}")
    last_journal_path = Path(journal_rows[-1]["journal_path"])
    with np.load(last_journal_path, allow_pickle=False) as latest_journal:
        journal_physical = json.loads(json.loads(latest_journal["post_physical_state_json"].item()))
    if journal_physical != physical:
        raise RuntimeError("resume_terminal_OA_and_durable_journal_physical_states_differ")

    phase_events_path = run_dir / "task_phase_events.jsonl"
    phase_events = [
        json.loads(line) for line in phase_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if phase_events_path.is_file() else []
    terminal_phase_events = [
        row for row in phase_events
        if row.get("event") == "task_phase_transition_observed"
        and int(row.get("timestep", -1)) == resume_timestep
    ]
    terminal_phase_event = terminal_phase_events[-1] if terminal_phase_events else None
    if terminal_phase_event is not None:
        event_native = terminal_phase_event.get("native_task_status", {})
        if (
            terminal_phase_event.get("from_phase") != "precise_placement"
            or terminal_phase_event.get("to_phase") != "release"
            or terminal_phase_event.get("held_object") != OBJECT_ID
            or _native_values(event_native) != {POT1_ON: False, POT2_ON: True, STOVE_ON: True}
        ):
            raise RuntimeError("resume_terminal_phase_event_does_not_attest_the_observed_Pot1_release_transition")
    # The resumed multi-action tail must start exactly from the previous
    # committed runtime snapshot, with no cross-snapshot state splice.
    source_step = run_dir / f"source_t{initial_timestep}.pkl"
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    previous_run = Path(run_config.get("source_archive", ""))
    if action_count > 1:
        previous_terminal = previous_run / f"snapshots/main_timestep_{initial_timestep:06d}.pkl"
        if not source_step.is_file() or not previous_terminal.is_file() or _sha(source_step.read_bytes()) != _sha(previous_terminal.read_bytes()):
            raise RuntimeError("resume_contiguous_tail_start_snapshot_differs_from_previous_episode_terminal")
    if action_count > 1 and len(physical_commits) != action_count:
        raise RuntimeError("resume_multi_action_tail_lacks_per_action_physical_safety_audits")

    pot1 = _object_state(physical, OBJECT_ID)
    pot2 = _object_state(physical, POT2_ID)
    attestation = {
        **base_attestation,
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(snapshot_path),
        "source_snapshot_sha256": _sha(snapshot_path.read_bytes()),
        "source_action_trace_sha256": sha256_file(trace_path),
        "source_model_geometry_sha256": MODEL_SHA256,
        "timestep": resume_timestep,
        "integration_state_sha256": _array_sha(actual_i),
        "arm_joint_qpos": qpos_actual.tolist(),
        "eef_position_world_m": eef_actual.tolist(),
        "eef_quaternion_wxyz": quat_actual.tolist(),
        "gripper_qpos_m": physical["robot_states"][0]["gripper_qpos"],
        "object_position_world_m": pot1["root_pos"],
        "object_quaternion_wxyz": pot1["root_quat"],
        "pot2_position_world_m": pot2["root_pos"],
        "pot2_quaternion_wxyz": pot2["root_quat"],
        "native_status": actual_native,
        "prior_source_result": str(result_path),
        "prior_actions_fully_substep_audited_in_this_entry": action_count,
        "committed_route_cursor": expected_route_index,
        "route_original_start_snapshot_path": str(original_start_snapshot),
        "route_waypoint_match": {
            "route_cursor": expected_route_index,
            "eef_position_error_m": eef_position_error,
            "eef_orientation_error_rad": eef_orientation_error,
            "qpos_reference_error_rad_diagnostic_only": qpos_error,
            "qpos_deviation_is_not_a_safety_blocker": True,
        },
        "terminal_observed_phase_transition": terminal_phase_event,
        "previous_failure_was_an_audited_recording_schema_error": True,
    }
    return snapshot_path, attestation


def _predicate_map(status: dict[str, Any]) -> dict[tuple[str, ...], bool]:
    return {
        tuple(str(part).lower() for part in item.get("predicate", [])): bool(item["value"])
        for item in status.get("native_goal_values", [])
        if isinstance(item, dict) and isinstance(item.get("value"), bool)
    }


def _contacts_signature(contacts: dict[str, Any]) -> list[tuple[int, int]]:
    pairs = []
    for item in contacts.get("contacts", []):
        a = int(item.get("geom1", {}).get("id", -1))
        b = int(item.get("geom2", {}).get("id", -1))
        if a < 0 or b < 0:
            raise RuntimeError("live_contact_record_has_no_geom_identity")
        pairs.append(tuple(sorted((a, b))))
    return sorted(set(pairs))


def _robot_contact_rows(contacts: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in contacts.get("contacts", []):
        body1 = str(item.get("body1", {}).get("name", ""))
        body2 = str(item.get("body2", {}).get("name", ""))
        if body1.startswith(("robot0_", "gripper0_")) or body2.startswith(("robot0_", "gripper0_")):
            rows.append(item)
    return rows


def _assert_robot_contacts(backend: DurableTaskSuffixBackend, contacts: dict[str, Any], phase: str) -> None:
    finger_ids = set(int(x) for x in backend.context_static["gripper_geometry_ids"])
    if len(finger_ids) != 4:
        raise RuntimeError(f"active_Panda_finger_geom_inventory_not_exactly_four:{sorted(finger_ids)}")
    target_ids = set(int(x) for x in backend.bindings[OBJECT_ID]["geom_ids"])
    invalid = []
    rows = _robot_contact_rows(contacts)
    if phase in {"settle", "verify_task"} and rows:
        raise RuntimeError(f"{phase}_requires_no_robot_contact_after_withdrawal")
    for contact in rows:
        a = int(contact.get("geom1", {}).get("id", -1))
        b = int(contact.get("geom2", {}).get("id", -1))
        if not ((a in finger_ids and b in target_ids) or (b in finger_ids and a in target_ids)):
            invalid.append({"geom1": contact.get("geom1"), "geom2": contact.get("geom2")})
            continue
        depth = contact.get("signed_contact_distance_m")
        if depth is None or not np.isfinite(float(depth)) or float(depth) < -0.002 - 1e-9:
            invalid.append({"geom1": contact.get("geom1"), "geom2": contact.get("geom2"), "depth_m": depth})
    if invalid:
        raise RuntimeError(f"{phase}_contact_is_not_whitelisted_four_finger_to_Pot1:{invalid[:8]}")


def _policy_thresholds(policy: dict[str, Any]) -> dict[str, float]:
    thresholds = {}
    for group_name in ("forbidden_objects", "allowed_target_objects", "other_scene_objects"):
        for row in policy.get(group_name, []):
            thresholds[str(row["object_id"])] = float(row["minimum_clearance_m"])
    return thresholds


def _audit_substeps(
    substeps: list[dict[str, Any]], policy: dict[str, Any], *, held: bool, phase: str
) -> dict[str, Any]:
    minima: dict[str, float] = {}
    required = _policy_thresholds(policy)
    failures = []
    if len(substeps) != EXPECTED_SUBSTEPS:
        raise RuntimeError(f"{phase}_expected_25_substeps_got_{len(substeps)}")
    for index, row in enumerate(substeps):
        if row.get("status") != "complete":
            failures.append({"substep": index, "kind": "geometry_status", "status": row.get("status")})
        for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
        ):
            if row.get(key):
                failures.append({"substep": index, "kind": key, "count": len(row[key])})
        clearances = row.get("clearances_by_object", {})
        for object_id, minimum in required.items():
            record = clearances.get(object_id)
            value = record.get("minimum_distance_m") if isinstance(record, dict) else None
            if value is None or not np.isfinite(float(value)):
                failures.append({"substep": index, "kind": "clearance_missing", "object_id": object_id})
                continue
            distance = float(value)
            minima[object_id] = min(distance, minima.get(object_id, math.inf))
            if distance < minimum - 1e-9:
                failures.append({
                    "substep": index, "kind": "robot_operational_margin", "object_id": object_id,
                    "distance_m": distance, "required_m": minimum,
                })
        carried = row.get("carried_clearances_by_object", {})
        if held:
            for object_id, record in carried.items():
                value = record.get("minimum_distance_m") if isinstance(record, dict) else record
                if value is None or not np.isfinite(float(value)):
                    failures.append({"substep": index, "kind": "carried_clearance_missing", "object_id": object_id})
                    continue
                distance = float(value)
                key = f"carried:{object_id}"
                minima[key] = min(distance, minima.get(key, math.inf))
                required_margin = next(
                    (float(row["minimum_clearance_m"]) for row in policy.get("forbidden_objects", [])
                     if str(row["object_id"]) == str(object_id)),
                    next((float(row["minimum_clearance_m"]) for row in policy.get("allowed_target_objects", [])
                          if str(row["object_id"]) == str(object_id)), 0.0),
                )
                # Object contact with the active stove support is permitted
                # only through the controller's explicit phase-scoped pairs.
                support_allowed = (
                    phase in {"precise_placement", "release"}
                    and object_id == "flat_stove_1"
                    and bool(policy.get("carried_object", {}).get("allowed_contact_pairs"))
                )
                if not support_allowed and distance < required_margin - 1e-9:
                    failures.append({
                        "substep": index, "kind": "carried_object_operational_margin",
                        "object_id": object_id, "distance_m": distance,
                        "required_m": required_margin,
                    })
        for pair, value in row.get("allowed_contact_clearances_m", {}).items():
            distance = float(value)
            minima[f"allowed_pair:{pair}"] = min(distance, minima.get(f"allowed_pair:{pair}", math.inf))
            if distance < -0.002 - 1e-9:
                failures.append({
                    "substep": index, "kind": "allowed_pair_penetration_limit",
                    "pair": pair, "distance_m": distance, "minimum_m": -0.002,
                })
    if failures:
        raise RuntimeError(f"{phase}_frozen_operational_substep_policy_failed:" + json.dumps(failures[:16], sort_keys=True))
    return {"substeps": len(substeps), "minimum_clearance_by_object_m": minima, "failures": []}


def _static_policy(policy: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in policy.items() if key not in {"allowed_contact_pairs", "carried_object"}}


def _assert_phase_contact_policy(backend: DurableTaskSuffixBackend, phase: str) -> dict[str, Any]:
    policy = backend.safety_policy
    target_ids = set(int(x) for x in backend.bindings[OBJECT_ID]["geom_ids"])
    fingers = set(int(x) for x in backend.context_static["gripper_geometry_ids"])
    pairs = {
        (int(row["robot_geom_id"]), int(row["scene_geom_id"])): float(row["minimum_clearance_m"])
        for row in policy.get("allowed_contact_pairs", [])
    }
    contact_active = phase in {
        "transport", "precise_placement", "release", "withdrawal"
    }
    expected = (
        {(finger, geom): -0.002 for finger in fingers for geom in target_ids}
        if contact_active else {}
    )
    if pairs != expected or len(fingers) != 4:
        raise RuntimeError(f"{phase}_live_finger_contact_allowlist_is_not_exactly_four_finger_geoms_to_Pot1")
    carried = policy.get("carried_object")
    if phase in {"transport", "precise_placement", "release"}:
        if not isinstance(carried, dict) or carried.get("object_id") != OBJECT_ID:
            raise RuntimeError(f"{phase}_live_carried_object_binding_is_not_Pot1")
    scoped_support = []
    if phase in {"precise_placement", "release"}:
        fixture = backend.bindings.get(backend.current_region_parent)
        if not isinstance(fixture, dict):
            raise RuntimeError("Pot1_placement_has_no_native_fixture_binding")
        support_ids = set(int(x) for x in backend.task_support_geom_ids) & set(int(x) for x in fixture["geom_ids"])
        expected_support = {
            (int(carried_geom), int(support_geom))
            for carried_geom in backend.bindings[OBJECT_ID]["geom_ids"]
            for support_geom in support_ids
        }
        actual_support = {
            (int(row["carried_geom_id"]), int(row["scene_geom_id"]))
            for row in carried.get("allowed_contact_pairs", [])
        }
        if not support_ids or actual_support != expected_support or any(
            float(row["minimum_clearance_m"]) != -0.002
            for row in carried.get("allowed_contact_pairs", [])
        ):
            raise RuntimeError("Pot1_support_contact_permission_is_not_exactly_native_burner_support")
        scoped_support = sorted([list(pair) for pair in actual_support])
    elif isinstance(carried, dict) and carried.get("allowed_contact_pairs"):
        raise RuntimeError(f"{phase}_unexpected_carried_support_contact_permission")
    return {
        "phase": phase,
        "finger_geom_ids": sorted(fingers),
        "target_geom_ids": sorted(target_ids),
        "finger_target_pair_count": len(expected),
        "support_pairs": scoped_support,
        "palm_and_other_robot_geoms_remain_constrained": True,
    }


def _native_on(status: dict[str, Any], key: tuple[str, ...]) -> bool | None:
    values = _predicate_map(status)
    return values.get(key)


def _runtime_snapshot(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError("predicted_terminal_runtime_snapshot_missing")
    envelope = pickle.loads(payload)
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
        or not isinstance(envelope.get("snapshot"), dict)
    ):
        raise RuntimeError("predicted_terminal_runtime_snapshot_schema_or_model_mismatch")
    return envelope["snapshot"]


def _inspect_predicted_terminal(
    bridge: BridgeProcess,
    backend: DurableTaskSuffixBackend,
    source_snapshot_id: str,
    source_integration: np.ndarray,
    payload: bytes,
    phase: str,
) -> dict[str, Any]:
    predicted_snapshot = _runtime_snapshot(payload)
    try:
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
        if restored.get("status") != "restored":
            raise RuntimeError(f"predicted_terminal_snapshot_restore_failed:{restored.get('error')}")
        native = bridge.request({"op": "native_task_status"})
        contacts = bridge.request({"op": "inspect_contacts"})
        held, grasp_verified, grasp_records = backend._grasp_states()
        backend._refresh_task_contact_policy(phase, OBJECT_ID, held)
        geometry = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        }).get("safety_geometry", {})
        if native.get("status") != "complete" or _native_on(native, POT2_ON) is not True:
            raise RuntimeError("predicted_branch_lost_or_unknown_Pot2_native_On")
        if held is not None and held != OBJECT_ID:
            raise RuntimeError(f"predicted_branch_reports_wrong_held_object:{held}")
        if held == OBJECT_ID:
            record = grasp_records.get(OBJECT_ID, {})
            fingers = record.get("finger_contacts", {})
            if not record.get("grasp_confirmed") or not fingers.get("left") or not fingers.get("right"):
                raise RuntimeError("predicted_branch_Pot1_grasp_not_bilateral_stable")
        if phase in {"transport", "precise_placement"} and held != OBJECT_ID:
            raise RuntimeError(f"predicted_branch_lost_Pot1_during_{phase}")
        if phase in {"withdrawal", "settle", "verify_task"} and held is not None:
            raise RuntimeError(f"predicted_branch_regrasped_or_retained_Pot1_during_{phase}")
        if held is None and phase in {"release", "withdrawal", "settle", "verify_task"}:
            if _native_on(native, POT1_ON) is not True:
                raise RuntimeError("predicted_branch_released_Pot1_without_native_On")
        _assert_robot_contacts(backend, contacts, phase)
        if geometry.get("status") != "complete" or any(geometry.get(key) for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
        )):
            raise RuntimeError("predicted_branch_terminal_state_fails_native_or_operational_geometry")
        live_state = geometry.get("snapshot_state")
        # Some bridge versions omit state in the geometry response; the export
        # below still binds this inspection to the exact predicted snapshot.
        predicted_integration = np.asarray(predicted_snapshot.get("mujoco_state"), dtype=np.float64)
        if predicted_integration.shape != source_integration.shape:
            raise RuntimeError("predicted_Integration_State_shape_mismatch")
        return {
            "native_task_status": native,
            "held_object": held,
            "grasp_verified": grasp_verified,
            "grasp_evidence": grasp_records.get(OBJECT_ID),
            "contact_geom_pairs": _contacts_signature(contacts),
            "terminal_geometry": geometry,
            "predicted_integration_state_sha256": _array_sha(predicted_integration),
            "predicted_live_state_available": isinstance(live_state, dict),
        }
    finally:
        restored = bridge.request({"op": "restore", "snapshot_id": source_snapshot_id})
        if restored.get("restored_snapshot_id") != source_snapshot_id:
            raise RuntimeError(f"formal_source_state_restore_after_preview_failed:{restored.get('error')}")
        source = bridge.request({"op": "observe", "include_snapshot_state": True})
        actual_integration = np.asarray(
            source.get("snapshot_state", {}).get("mujoco_integration_state"), dtype=np.float64
        )
        if actual_integration.shape != source_integration.shape or not np.array_equal(
            actual_integration, source_integration
        ):
            raise RuntimeError("formal_Integration_State_changed_during_preview_terminal_inspection")
        source_held, _, _ = backend._grasp_states()
        backend._refresh_task_contact_policy(phase, OBJECT_ID, source_held)
        backend.latest_data = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        backend.latest_data["native_task_status"] = bridge.request({"op": "native_task_status"})
        backend.latest_data["contacts"] = bridge.request({"op": "inspect_contacts"})


def _compare_prediction_commit(
    predicted_payload: bytes,
    actual_payload: bytes,
    predicted_status: dict[str, Any],
    actual_data: dict[str, Any],
    predicted_step: dict[str, Any],
    predicted_contacts: dict[str, Any],
    actual_contacts: dict[str, Any],
) -> dict[str, Any]:
    predicted = _runtime_snapshot(predicted_payload)
    actual = _runtime_snapshot(actual_payload)
    pred_i = np.asarray(predicted.get("mujoco_state"), dtype=np.float64)
    act_i = np.asarray(actual.get("mujoco_state"), dtype=np.float64)
    if pred_i.shape != act_i.shape:
        raise RuntimeError("prediction_commit_Integration_State_shape_mismatch")
    integration_max = float(np.max(np.abs(pred_i - act_i)))
    integration_match = bool(np.allclose(pred_i, act_i, rtol=0.0, atol=1e-9))
    if not integration_match:
        raise RuntimeError(f"prediction_commit_physical_Integration_State_mismatch:{integration_max}")
    actual_status = actual_data.get("native_task_status", {})
    pred_native_match = _predicate_map(predicted_status) == _predicate_map(actual_status)
    task_success_match = predicted_status.get("task_success") == actual_status.get("task_success")
    contacts_match = _contacts_signature(predicted_contacts) == _contacts_signature(actual_contacts)
    reward_match = math.isclose(float(predicted_step.get("reward", math.nan)), float(actual_data.get("reward", math.nan)), rel_tol=0.0, abs_tol=1e-9)
    done_match = bool(predicted_step.get("done")) == bool(actual_data.get("done"))
    if not (pred_native_match and task_success_match and contacts_match and reward_match and done_match):
        raise RuntimeError(
            "prediction_commit_observable_state_mismatch:"
            f"native={pred_native_match};success={task_success_match};contacts={contacts_match};"
            f"reward={reward_match};done={done_match}"
        )
    predicted_observation = predicted_step.get("observation", {})
    predicted_physical = (
        predicted_observation.get("snapshot_state")
        if isinstance(predicted_observation, dict) else None
    )
    if not isinstance(predicted_physical, dict):
        predicted_physical = predicted_step.get("state")
    actual_physical = actual_data.get("snapshot_state")
    if not isinstance(predicted_physical, dict) or not isinstance(actual_physical, dict):
        raise RuntimeError("prediction_commit_physical_diagnostic_state_missing")
    diagnostics = {}
    for object_id in (OBJECT_ID, POT2_ID):
        pstate = _object_state(predicted_physical, object_id)
        astate = _object_state(actual_physical, object_id)
        for key in ("root_pos", "root_quat", "joint_qvel"):
            p = np.asarray(pstate.get(key), dtype=np.float64)
            a = np.asarray(astate.get(key), dtype=np.float64)
            if p.shape != a.shape or not np.isfinite(p).all() or not np.isfinite(a).all():
                raise RuntimeError(f"prediction_commit_{object_id}_{key}_invalid")
            error = float(np.max(np.abs(p - a))) if p.size else 0.0
            if key == "joint_qvel" and error > 1e-8:
                raise RuntimeError(f"prediction_commit_{object_id}_velocity_mismatch:{error}")
            if key != "joint_qvel" and error > 1e-8:
                diagnostics[f"{object_id}_{key}"] = {
                    "max_abs_component_error": error,
                    "classification": "derived_pose_difference_diagnostic_only",
                }
    return {
        "Integration_State_match_atol_1e-9": True,
        "Integration_State_max_abs_error": integration_max,
        "all_native_predicates_match": pred_native_match,
        "native_task_success_match": task_success_match,
        "contact_geom_pair_set_match": contacts_match,
        "reward_match": reward_match,
        "done_match": done_match,
        "small_derived_pose_differences_diagnostic_only": diagnostics,
    }


class Pot1PlacementBackend(DurableTaskSuffixBackend):
    """Keep the held Pot1 closed until native support is observed."""

    def task_expert_action(self, observation, context):
        action = super().task_expert_action(observation, context)
        if action is not None:
            action = np.asarray(action, dtype=np.float32).copy()
            held, _verified, _records = self._grasp_states()
            native_on = _native_on(self.current_native_task_status or {}, POT1_ON)
            if context.phase == "release" and held == OBJECT_ID and native_on is False:
                # EEF pose tolerance can be met while the object has not yet
                # reached the native cook region. Continue the configured
                # OSC placement target with the verified grasp closed; the
                # existing controller will select open only after native On.
                action[6] = float(self.config_data["task_execution"].get("close_gripper_action", 1.0))
                self._current_action_source = "task_expert_osc_pose_hold_Pot1_until_native_On"
                self.current_forecast = action[None, :].copy()
            elif context.phase == "release" and held is None and native_on is False:
                # Once opened, keep the arm stationary while the supported
                # object settles and the native predicate catches up. This
                # avoids chasing a released object with the grasp-relative
                # placement transform.
                action[:6] = 0.0
                action[6] = 0.0
                self._current_action_source = "task_expert_osc_pose_hold_stationary_pending_native_On"
                self.current_forecast = action[None, :].copy()
            elif context.phase in {"release", "withdrawal", "settle", "verify_task"} and held is None:
                action[6] = 0.0
                self._current_action_source = "task_expert_osc_pose_neutral_gripper_after_release"
                self.current_forecast = action[None, :].copy()
        return action


def _parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38761)
    parser.add_argument("--reuse-transport-route", type=Path)
    parser.add_argument(
        "--resume-from-run", type=Path,
        help="Resume the same Episode from an already audited physical action commit.",
    )
    parser.add_argument(
        "--resume-observed-phase", action="store_true",
        help="Reconstruct the terminal phase only from a matching same-Episode observed transition event; counters and anchors are initialized from the live restored state.",
    )
    parser.add_argument(
        "--replan-from-resume", action="store_true",
        help="After an archived route endpoint misses the live phase goal, make one evidence-justified existing RRT call from the restored current state.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    global SOURCE_ARCHIVE, SOURCE_SNAPSHOT, SOURCE_TRACE, SOURCE_RESULT, SOURCE_JOURNAL, INITIAL_TIMESTEP
    base_attestation, frozen_policy, source_config = _source_attestation()
    resume_cursor = 0
    if args.resume_from_run is not None:
        resume_dir = args.resume_from_run.resolve()
        SOURCE_ARCHIVE = resume_dir
        SOURCE_SNAPSHOT = resume_dir / "snapshots/main_timestep_001062.pkl"
        SOURCE_TRACE = resume_dir / "observation_action_trace.npz"
        SOURCE_RESULT = resume_dir / "suffix_result.json"
        SOURCE_JOURNAL = resume_dir / "durable_action_journal/index.jsonl"
        SOURCE_SNAPSHOT, attestation = _committed_action_resume_attestation(
            resume_dir, base_attestation, frozen_policy, source_config
        )
        INITIAL_TIMESTEP = int(attestation["timestep"])
        cursor_value = attestation.get("committed_route_cursor")
        resume_cursor = int(cursor_value) if cursor_value is not None else 0
        if args.resume_observed_phase:
            phase_event = attestation.get("terminal_observed_phase_transition")
            if (
                not isinstance(phase_event, dict)
                or phase_event.get("to_phase") != "release"
                or phase_event.get("held_object") != OBJECT_ID
            ):
                raise RuntimeError("resume_observed_phase_lacks_matching_durable_same_episode_transition")
            if args.replan_from_resume:
                raise RuntimeError("recorded_release_phase_resume_must_not_replan_transport")
            args.reuse_transport_route = None
        if args.replan_from_resume:
            prior_result = json.loads((resume_dir / "suffix_result.json").read_text(encoding="utf-8"))
            if "existing_RRT_route_ended_outside_live_goal_tolerance" not in prior_result.get("failure_reason", ""):
                raise RuntimeError("resume_replan_is_only_authorized_after_an_audited_route_endpoint_miss")
            with np.load(resume_dir / "reused_transport_route.npz", allow_pickle=False) as route:
                route_len = len(route["qpos_path"])
            if resume_cursor != route_len - 1:
                raise RuntimeError("resume_replan_requires_the_previous_route_terminal_cursor")
            args.reuse_transport_route = None
        elif not args.resume_observed_phase and args.reuse_transport_route is None:
            args.reuse_transport_route = resume_dir / "reused_transport_route.npz"
        if (
            not args.resume_observed_phase
            and args.reuse_transport_route is not None
            and args.reuse_transport_route.resolve() != (resume_dir / "reused_transport_route.npz").resolve()
        ):
            raise RuntimeError("t1062_resume_must_reuse_the_exact_route_that_certified_its_committed_action")
    else:
        attestation = base_attestation
    config_data, recovery = load_runtime_config(args.config)
    config_data = json.loads(json.dumps(config_data))
    config_data["candidate_search"]["enabled"] = False
    target_clearances = {
        str(row["object_id"]): float(row["minimum_clearance_m"])
        for row in frozen_policy.get("allowed_target_objects", [])
    }
    recovery = recovery.__class__(
        **{
            **recovery.__dict__,
            "hard_minimum_clearance_m": float(frozen_policy["minimum_clearance_m"]),
            "target_object_clearance_m": min(target_clearances.values()),
            "object_clearance_m": target_clearances,
            "execution_step_budget": MAX_DEMONSTRATION_HORIZON - INITIAL_TIMESTEP,
            "horizon_steps": MAX_DEMONSTRATION_HORIZON,
            "standard_horizon_steps": STANDARD_HORIZON,
            "allow_horizon_extension": True,
        }
    )
    recovery.validate()
    config_data["recovery"] = _jsonable(recovery.__dict__)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        f"/opt/data/private/gs_opd/rollouts/l0t1061_pot1_safe_placement_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "snapshots").mkdir()
    (output_dir / "predicted_branch_snapshots").mkdir()
    _write_json(output_dir / "run_config.json", {
        "entrypoint": str(Path(__file__).resolve()),
        "source_archive": str(SOURCE_ARCHIVE),
        "source_snapshot_attestation": attestation,
        "historical_success_structure_prior": {
            "report": str(HISTORICAL_SUCCESS),
            "historical_success_report_sha256": sha256_file(HISTORICAL_SUCCESS),
            "t854_to_t1093": "high transport bypass, stove-side entry, configured final target, release and settle are priors only; historical actions are not replayed",
            "pot1_target_pose_in_native_region": next(
                row for row in config_data["task_execution"]["objects"] if row["object_id"] == OBJECT_ID
            ),
            "transport_clearance_m": config_data["task_execution"].get("transport_clearance_m"),
        },
        "mode": f"same_episode_state_level_continuation_from_t{INITIAL_TIMESTEP}",
        "task_controller": "existing_TaskPhaseController_adopts_live_verified_Pot1_grasp",
        "controller": "native_OSC_POSE",
        "rolling_physical_branch_authorizes_every_action": True,
        "maximum_task_timestep": MAX_DEMONSTRATION_HORIZON,
        "standard_benchmark_horizon": STANDARD_HORIZON,
        "remaining_action_budget": MAX_DEMONSTRATION_HORIZON - INITIAL_TIMESTEP,
        "per_action_physics_substeps": EXPECTED_SUBSTEPS,
        "RRT_policy": "existing_bounded_RRT_only_when_entering_a_new_rrt_connect_phase;one_search_per_phase",
        "reused_transport_route": str(args.reuse_transport_route) if args.reuse_transport_route else None,
        "resume_source_run": str(args.resume_from_run.resolve()) if args.resume_from_run else None,
        "resume_replan_from_live_state": bool(args.replan_from_resume),
        "resume_observed_phase_from_same_episode_event": bool(args.resume_observed_phase),
        "resume_previous_route_cursor": resume_cursor if args.resume_from_run else None,
        "qpos_reference_mode": "diagnostic_only_for_RRT_OSC_execution",
        "qpos_deviation_limit_rad_logged_only": recovery.maximum_joint_path_deviation_rad,
        "risk_hysteresis_m": [RISK_TRIGGER_M, RISK_REARM_M],
        "risk_warning_is_diagnostic_not_a_standalone_blocker": True,
        "safety_thresholds_changed": False,
        "frozen_operational_policy": frozen_policy,
        "frozen_policy_sha256": _canonical_sha256(frozen_policy),
        "lora_training_started": False,
        "complete_round0_seed": False,
    })
    Path(output_dir / f"source_t{INITIAL_TIMESTEP}.pkl").write_bytes(SOURCE_SNAPSHOT.read_bytes())
    if args.resume_from_run is not None:
        route_origin = Path(attestation["route_original_start_snapshot_path"])
        (output_dir / "source_t1061.pkl").write_bytes(route_origin.read_bytes())
    artifacts = RecoveryArtifactWriter(output_dir)
    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend: Pot1PlacementBackend | None = None
    original_handlers = {}
    result: dict[str, Any] = {
        "status": "preflight_not_started",
        "initial_timestep": INITIAL_TIMESTEP,
        "physical_actions_committed": 0,
        "rrt_calls": 0,
        "complete_round0_seed": False,
    }
    started = time.monotonic()

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal_{signum}_received_at_durable_action_boundary")

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.signal(signum, interrupt)
    try:
        bridge.start(
            benchmark=config_data["benchmark"], level=int(config_data["level"]),
            task_id=int(config_data["task_id"]), init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=MAX_DEMONSTRATION_HORIZON,
        )
        backend = Pot1PlacementBackend(
            bridge, config_data, recovery, output_dir, artifacts,
            mode="task-runtime-snapshot", student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=SOURCE_SNAPSHOT,
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
            runtime_snapshot_attestation=attestation,
            journal_dir=output_dir / "durable_action_journal",
        )
        restored = backend.restore_start_state()
        if int(restored.get("timestep", -1)) != INITIAL_TIMESTEP:
            raise RuntimeError("strict_t1061_restore_timestep_mismatch")
        if restored.get("scene_metadata", {}).get("model_geometry_sha256") != MODEL_SHA256:
            raise RuntimeError("strict_t1061_restore_model_fingerprint_mismatch")
        integration = np.asarray(restored["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
        if _array_sha(integration) != attestation["integration_state_sha256"]:
            raise RuntimeError("strict_t1061_restore_Integration_State_mismatch")
        exported = bridge.request({"op": "export_runtime_snapshot"})
        if exported.get("status") != "complete":
            raise RuntimeError("strict_t1061_runtime_controller_export_missing")
        runtime = pickle.loads(exported["payload"]).get("snapshot", {})
        if _normalized_state_sha256(runtime) != _normalized_state_sha256(pickle.loads(SOURCE_SNAPSHOT.read_bytes())["snapshot"]):
            raise RuntimeError("strict_t1061_robot_gripper_or_controller_internal_state_mismatch")

        backend._load_context_inputs(restored)
        if _canonical_sha256(_static_policy(backend.safety_policy)) != _canonical_sha256(_static_policy(frozen_policy)):
            raise RuntimeError("t1061_frozen_static_operational_policy_mismatch")
        preflight = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        preflight["native_task_status"] = bridge.request({"op": "native_task_status"})
        preflight["contacts"] = bridge.request({"op": "inspect_contacts"})
        backend.latest_data = preflight
        context = backend.current_context()
        if context.phase != "transport" or context.held_object != OBJECT_ID:
            raise RuntimeError(f"t1061_live_task_adoption_mismatch:{context.phase}:{context.held_object}")
        backend.handoff_to_task_expert(backend._observation(preflight), context)
        context = backend.current_context()
        # `current_context()` adopts the restored bilateral grasp and binds
        # the phase-scoped four-finger-to-Pot1 permission. Recompute the
        # read-only geometry only after that live-derived policy is active;
        # the initial observation above intentionally did not authorize any
        # action and must not be used as the held-object verdict.
        preflight = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        preflight["native_task_status"] = bridge.request({"op": "native_task_status"})
        preflight["contacts"] = bridge.request({"op": "inspect_contacts"})
        backend.latest_data = preflight
        context = backend.current_context()
        phase_resume_event = None
        if args.resume_observed_phase:
            phase_resume_event = attestation.get("terminal_observed_phase_transition")
            if (
                not isinstance(phase_resume_event, dict)
                or int(phase_resume_event.get("timestep", -1)) != INITIAL_TIMESTEP
                or phase_resume_event.get("from_phase") != "precise_placement"
                or phase_resume_event.get("to_phase") != "release"
                or phase_resume_event.get("held_object") != OBJECT_ID
                or _native_values(phase_resume_event.get("native_task_status", {}))
                    != {POT1_ON: False, POT2_ON: True, STOVE_ON: True}
            ):
                raise RuntimeError("live_phase_resume_event_no_longer_matches_t1453_native_or_grasp_evidence")
            held_now, grasp_now, grasp_records_now = backend._grasp_states()
            native_now = backend.current_native_task_status or {}
            if (
                held_now != OBJECT_ID
                or not grasp_now
                or not grasp_records_now.get(OBJECT_ID, {}).get("finger_contacts", {}).get("left")
                or not grasp_records_now.get(OBJECT_ID, {}).get("finger_contacts", {}).get("right")
                or _native_values(native_now) != {POT1_ON: False, POT2_ON: True, STOVE_ON: True}
            ):
                raise RuntimeError("live_phase_resume_requires_the_attested_held_Pot1_pending_native_On_state")
            # Reconstruct only the latest phase transition that was durably
            # observed in this exact Episode. No historical phase counters are
            # restored; the fresh phase starts at zero and its anchor is the
            # current, strictly restored EEF pose.
            backend.task_controller.phase = "release"
            backend.task_controller._phase_steps = 0
            backend.task_controller._last_timestep = -1
            live_eef_position, live_eef_quaternion = backend._live_eef_pose(preflight)
            backend.task_phase_anchor = (live_eef_position.copy(), live_eef_quaternion.copy())
            backend.last_task_phase = "release"
            context = backend.current_context()
            if context.phase != "release" or context.held_object != OBJECT_ID:
                raise RuntimeError(f"durable_release_phase_reconstruction_did_not_hold:{context.phase}:{context.held_object}")
            preflight = bridge.request({
                "op": "observe", "include_snapshot_state": True,
                "safety_geometry_policy": backend.safety_policy,
            })
            preflight["native_task_status"] = bridge.request({"op": "native_task_status"})
            preflight["contacts"] = bridge.request({"op": "inspect_contacts"})
            backend.latest_data = preflight
            context = backend.current_context()
            if context.phase != "release" or context.held_object != OBJECT_ID:
                raise RuntimeError("release_phase_changed_during_read_only_live_policy_revalidation")
            if not np.array_equal(
                np.asarray(preflight["snapshot_state"]["mujoco_integration_state"], dtype=np.float64), integration
            ):
                raise RuntimeError("phase_reconstruction_changed_restored_physical_Integration_State")
            _record_jsonl(output_dir / "task_phase_events.jsonl", {
                "event": "task_phase_reconstructed_from_same_episode_observed_transition",
                "timestep": INITIAL_TIMESTEP,
                "source_event": phase_resume_event,
                "phase": "release",
                "controller_counters_replayed": False,
                "fresh_phase_counter_initial_value": 0,
                "phase_anchor_source": "strictly_restored_live_EEF_at_terminal_transition_timestep",
                "native_status_rechecked_live": preflight["native_task_status"],
                "no_physical_action_taken_during_reconstruction": True,
            })
        phase_policy = _assert_phase_contact_policy(backend, context.phase)
        grasp = context.metadata.get("held_object_state_evidence", {}).get(OBJECT_ID, {})
        fingers = grasp.get("finger_contacts", {})
        if not grasp.get("grasp_confirmed") or not fingers.get("left") or not fingers.get("right"):
            raise RuntimeError("t1061_strict_restore_does_not_have_verified_bilateral_Pot1_grasp")
        _assert_robot_contacts(backend, preflight["contacts"], context.phase)
        live_geometry = preflight.get("safety_geometry", {})
        if live_geometry.get("status") != "complete" or any(live_geometry.get(key) for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
        )):
            raise RuntimeError("t1061_live_preflight_fails_native_or_operational_safety")
        native = preflight["native_task_status"]
        if _native_values(native) != {POT1_ON: False, POT2_ON: True, STOVE_ON: True}:
            raise RuntimeError("t1061_live_native_predicates_do_not_match_expected_pending_Pot1_state")
        if not np.array_equal(
            np.asarray(preflight["snapshot_state"]["mujoco_integration_state"], dtype=np.float64), integration
        ):
            raise RuntimeError("t1061_read_only_preflight_changed_physical_Integration_State")
        backend.latest_data = preflight
        backend.initialize_trajectory()
        pot1 = _object_state(preflight["snapshot_state"], OBJECT_ID)
        pot2 = _object_state(preflight["snapshot_state"], POT2_ID)
        _write_json(output_dir / "strict_restore_preflight.json", {
            "status": "strict_t1061_restore_and_live_operational_preflight_passed",
            "source_snapshot_sha256": attestation["source_snapshot_sha256"],
            "model_geometry_sha256": MODEL_SHA256,
            "Integration_State_sha256": attestation["integration_state_sha256"],
            "runtime_controller_gripper_state_hash_match": True,
            "task_phase": context.phase,
            "held_object": context.held_object,
            "Pot1_grasp_evidence": grasp,
            "Pot1_root_position_world_m": pot1["root_pos"],
            "Pot2_root_position_world_m": pot2["root_pos"],
            "native_task_status": native,
            "safety_policy_static_fields_unchanged": True,
            "phase_contact_policy": phase_policy,
            "live_safety_geometry": live_geometry,
            "prior_prefix_current_policy_fully_reaudited": False,
        })
        _record_jsonl(output_dir / "task_phase_events.jsonl", {
            "event": "live_task_controller_adopted_from_runtime_snapshot",
            "timestep": INITIAL_TIMESTEP,
            "phase": context.phase,
            "held_object": context.held_object,
            "controller_counters_replayed": False,
            "native_task_status": native,
            "prior_successful_demo_used_as_structure_only": str(HISTORICAL_SUCCESS),
        })
        if args.replan_from_resume:
            previous_route = args.resume_from_run.resolve() / "reused_transport_route.npz"
            with np.load(previous_route, allow_pickle=False) as archived_route:
                previous_eef = np.asarray(archived_route["eef_reference_poses_world"], dtype=np.float64)
            live_eef = np.asarray(preflight["snapshot_state"]["robot_states"][0]["eef_pos"], dtype=np.float64)
            live_eef_quat = np.asarray(preflight["snapshot_state"]["robot_states"][0]["eef_quat"], dtype=np.float64)
            terminal_target = backend._task_stage_target(preflight, context.phase)
            terminal_pos_error, terminal_ori_error = backend._pose_error(preflight, terminal_target)
            old_route_pos_error = float(np.linalg.norm(live_eef - previous_eef[-1, :3]))
            old_route_quat = previous_eef[-1, 3:] / np.linalg.norm(previous_eef[-1, 3:])
            live_eef_quat = live_eef_quat / np.linalg.norm(live_eef_quat)
            old_route_ori_error = 2.0 * math.acos(float(np.clip(abs(np.dot(old_route_quat, live_eef_quat)), 0.0, 1.0)))
            target_tolerances = backend.task_controller.pose_goal_tolerances(context.phase)
            _record_jsonl(output_dir / "task_phase_events.jsonl", {
                "event": "previous_transport_route_terminal_missed_recomputed_live_grasp_target",
                "timestep": INITIAL_TIMESTEP,
                "previous_route_cursor": resume_cursor,
                "previous_route_points": len(previous_eef),
                "previous_route_endpoint_live_EEF_residual_m_rad": [old_route_pos_error, old_route_ori_error],
                "current_transport_target_error_m_rad": [terminal_pos_error, terminal_ori_error],
                "current_transport_target_tolerances_m_rad": list(target_tolerances),
                "new_RRT_reason": "current_object_to_EEF_grasp_transform_drifted_from_archived_route_endpoint_orientation",
                "new_RRT_calls_authorized": 1,
                "safety_thresholds_or_controller_changed": False,
                "all_historical_actions_replayed": False,
            })

        result.update({
            "status": "executing_Pot1_safe_placement_suffix",
            "preflight_passed": True,
            "frozen_policy_sha256": _canonical_sha256(frozen_policy),
            "maximum_total_timestep": MAX_DEMONSTRATION_HORIZON,
        })
        risk_armed = float(live_geometry.get("minimum_forbidden_clearance_m") or 0.0) >= RISK_REARM_M
        route: JointSpacePlan | None = None
        route_phase: str | None = None
        route_qpos = np.empty((0, 7), dtype=np.float64)
        route_eef = np.empty((0, 7), dtype=np.float64)
        planned_phases: set[str] = set()
        progress_window: list[float] = []
        previous_phase = context.phase
        terminal_reason: str | None = None
        while True:
            current_t = int(backend.latest_data.get("timestep", -1))
            if current_t >= MAX_DEMONSTRATION_HORIZON:
                terminal_reason = "demonstration_only_horizon_1800_reached"
                break
            # Confirm every stage from the current committed physical state.
            observation = backend.observe()
            current = backend.latest_data
            context = backend.current_context()
            if context.phase != previous_phase:
                _record_jsonl(output_dir / "task_phase_events.jsonl", {
                    "event": "task_phase_transition_observed",
                    "timestep": observation.timestep,
                    "from_phase": previous_phase,
                    "to_phase": context.phase,
                    "held_object": context.held_object,
                    "native_task_status": backend.current_native_task_status,
                })
                previous_phase = context.phase
                route = None
                route_phase = None
            native = backend.current_native_task_status or {}
            native_map = _native_values(native)
            if native.get("status") != "complete" or native_map.get(POT2_ON) is not True:
                raise RuntimeError(f"live_Pot2_On_lost_or_unknown_at_t{observation.timestep}")
            if native_map.get(STOVE_ON) is not True:
                raise RuntimeError(f"live_stove_Turnon_lost_or_unknown_at_t{observation.timestep}")
            if native_map.get(POT1_ON) is True and context.held_object == OBJECT_ID and context.phase not in {"release", "withdrawal", "settle", "verify_task"}:
                raise RuntimeError(f"Pot1_native_On_became_true_before_release_phase:{context.phase}")
            phase_policy = _assert_phase_contact_policy(backend, context.phase)
            if context.phase in {"transport", "precise_placement"} and context.held_object != OBJECT_ID:
                raise RuntimeError(f"Pot1_live_grasp_lost_before_release:{context.phase}")
            if context.phase in {"withdrawal", "settle", "verify_task"}:
                if context.held_object is not None or native_map.get(POT1_ON) is not True:
                    raise RuntimeError("Pot1_release_or_native_On_lost_after_placement")
            if native.get("task_success") is True:
                # The controller still proceeds through the configured stable
                # placement verification if the simulator has not terminated.
                result["native_task_success_first_observed_timestep"] = observation.timestep
            if native.get("task_success") is True and observation.reward >= 1.0 and observation.done:
                result["full_native_success"] = True
                result["full_native_success_timestep"] = observation.timestep
                live_values = _native_values(native)
                placement_released = context.held_object is None
                all_required_native_values = (
                    live_values.get(POT1_ON) is True
                    and live_values.get(POT2_ON) is True
                    and live_values.get(STOVE_ON) is True
                )
                if all_required_native_values and placement_released:
                    terminal_reason = "full_native_success_reward_done_after_verified_release"
                    break
                if context.phase == "verify_task":
                    terminal_reason = "full_native_success_and_task_phase_verification"
                    break

            live_geom = current.get("safety_geometry", {})
            if live_geom.get("status") != "complete" or any(live_geom.get(key) for key in (
                "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
                "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
            )):
                raise RuntimeError(f"pre_action_live_operational_safety_failed_at_t{observation.timestep}")
            live_margin = live_geom.get("minimum_forbidden_clearance_m")
            if live_margin is None or not np.isfinite(float(live_margin)):
                raise RuntimeError("pre_action_live_forbidden_clearance_unknown")
            if not risk_armed and float(live_margin) >= RISK_REARM_M:
                risk_armed = True
            elif risk_armed and float(live_margin) <= RISK_TRIGGER_M:
                risk_armed = False
                _record_jsonl(output_dir / "risk_boundary_events.jsonl", {
                    "event": "risk_margin_50mm_crossed_diagnostic_only",
                    "timestep": observation.timestep,
                    "minimum_forbidden_clearance_m": float(live_margin),
                    "hysteresis_m": [RISK_TRIGGER_M, RISK_REARM_M],
                    "motion_blocked_by_warning": False,
                })

            action: np.ndarray
            prediction_step: dict[str, Any]
            predicted_payload: bytes
            predicted_native: dict[str, Any]
            predicted_contacts: dict[str, Any]
            preview_audit: dict[str, Any]
            source_snapshot = bridge.request({"op": "snapshot"})
            source_snapshot_id = source_snapshot.get("snapshot_id")
            source_integration = np.asarray(
                source_snapshot.get("snapshot_state", {}).get("mujoco_integration_state"),
                dtype=np.float64,
            )
            live_integration = np.asarray(current["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
            if not isinstance(source_snapshot_id, str) or not np.array_equal(source_integration, live_integration):
                raise RuntimeError("pre_action_snapshot_does_not_match_current_real_timestep")

            if context.phase in {"transport", "withdrawal"}:
                if route is None or route_phase != context.phase:
                    if context.phase in planned_phases:
                        raise RuntimeError(f"duplicate_RRT_replan_for_phase_forbidden:{context.phase}")
                    goal = backend.resolve_goal(observation, context)
                    reuse_path = args.reuse_transport_route if context.phase == "transport" else None
                    if reuse_path is not None:
                        if not reuse_path.is_file():
                            raise FileNotFoundError(f"archived_transport_route_missing:{reuse_path}")
                        with np.load(reuse_path, allow_pickle=False) as archived:
                            route_qpos = np.asarray(archived["qpos_path"], dtype=np.float64)
                            route_eef = np.asarray(archived["eef_reference_poses_world"], dtype=np.float64)
                            route_meta = json.loads(str(archived["metadata_json"]))
                        live_state_hash = _array_sha(np.asarray(
                            current["snapshot_state"]["mujoco_integration_state"], dtype=np.float64
                        ))
                        expected_route_cursor = resume_cursor if args.resume_from_run is not None else 0
                        if expected_route_cursor:
                            original_start_path = Path(attestation["route_original_start_snapshot_path"])
                            original_start = pickle.loads(original_start_path.read_bytes())
                            original_start_hash = _array_sha(np.asarray(
                                original_start["snapshot"]["mujoco_state"], dtype=np.float64
                            ))
                            expected_start_hash = original_start_hash
                        else:
                            expected_start_hash = live_state_hash
                        if (
                            route_meta.get("planner_status") != "offline_feasible"
                            or route_meta.get("phase") != "transport"
                            or route_meta.get("start_integration_sha256") != expected_start_hash
                            or route_meta.get("model_geometry_sha256") != MODEL_SHA256
                            or route_meta.get("frozen_policy_sha256") != _canonical_sha256(backend.safety_policy)
                        ):
                            raise RuntimeError("reused_transport_route_start_model_or_dynamic_policy_fingerprint_mismatch")
                        robot_now = current["snapshot_state"]["robot_states"][0]
                        if (
                            route_qpos.ndim != 2 or route_qpos.shape[1:] != (7,)
                            or route_eef.shape != (len(route_qpos), 7) or len(route_qpos) < 2
                            or expected_route_cursor < 0 or expected_route_cursor >= len(route_qpos) - 1
                        ):
                            raise RuntimeError("reused_transport_route_shape_or_resume_cursor_invalid")
                        cursor_position_error = float(np.linalg.norm(
                            route_eef[expected_route_cursor, :3]
                            - np.asarray(robot_now["eef_pos"], dtype=np.float64)
                        ))
                        cursor_quat_ref = route_eef[expected_route_cursor, 3:] / np.linalg.norm(
                            route_eef[expected_route_cursor, 3:]
                        )
                        cursor_quat_now = np.asarray(robot_now["eef_quat"], dtype=np.float64)
                        cursor_quat_now /= np.linalg.norm(cursor_quat_now)
                        cursor_orientation_error = 2.0 * math.acos(float(np.clip(abs(np.dot(
                            cursor_quat_ref, cursor_quat_now
                        )), 0.0, 1.0)))
                        if (
                            cursor_position_error > recovery.maximum_eef_tracking_error_m
                            or cursor_orientation_error > recovery.maximum_eef_orientation_error_rad
                        ):
                            raise RuntimeError("reused_transport_route_resume_cursor_does_not_match_live_EEF")
                        cursor_qpos_deviation = float(np.max(np.abs(
                            route_qpos[expected_route_cursor] - observation.qpos
                        )))
                        if (
                            args.resume_from_run is not None
                            and attestation.get("committed_route_cursor") != expected_route_cursor
                        ):
                            raise RuntimeError("resume_route_cursor_does_not_match_already_committed_route_action")
                        pos_tol, ori_tol = backend.task_controller.pose_goal_tolerances("transport")
                        endpoint_position_error = float(np.linalg.norm(route_eef[-1, :3] - np.asarray(goal.position_world_m)))
                        endpoint_orientation_error = 2.0 * math.acos(float(np.clip(abs(np.dot(
                            route_eef[-1, 3:] / np.linalg.norm(route_eef[-1, 3:]),
                            np.asarray(goal.quaternion_wxyz) / np.linalg.norm(goal.quaternion_wxyz),
                        )), 0.0, 1.0)))
                        if endpoint_position_error > pos_tol or endpoint_orientation_error > ori_tol:
                            raise RuntimeError("reused_transport_route_endpoint_does_not_match_current_native_goal")
                        route = JointSpacePlan(
                            qpos_path=route_qpos,
                            minimum_forbidden_clearance_m=float(route_meta["minimum_forbidden_clearance_m"]),
                            planner_status="offline_feasible",
                            planner_source=str(route_meta["planner_source"]),
                            planning_seconds=0.0,
                            diagnostics={"route_reused_from": str(reuse_path), "route_metadata": route_meta},
                        )
                        backend._plan_poses = route_eef.copy()
                        backend.plan_cursor = expected_route_cursor
                        backend.plan_counter += 1
                        planned_phases.add(context.phase)
                        result["reused_RRT_calls"] = int(result.get("reused_RRT_calls", 0)) + 1
                        shutil.copyfile(reuse_path, output_dir / "reused_transport_route.npz")
                        _record_jsonl(output_dir / "task_phase_events.jsonl", {
                            "event": "previously_completed_transport_RRT_route_reused_after_packaging_failure",
                            "timestep": observation.timestep,
                            "phase": context.phase,
                            "route_source": str(reuse_path),
                            "start_integration_sha256": live_state_hash,
                            "model_geometry_sha256": MODEL_SHA256,
                            "dynamic_policy_sha256": _canonical_sha256(backend.safety_policy),
                            "endpoint_error_m_rad": [endpoint_position_error, endpoint_orientation_error],
                            "resume_route_cursor": expected_route_cursor,
                            "live_cursor_EEF_error_m_rad": [cursor_position_error, cursor_orientation_error],
                            "live_cursor_qpos_deviation_rad_diagnostic_only": cursor_qpos_deviation,
                            "planner_search_repeated": False,
                        })
                    else:
                        started_plan = time.monotonic()
                        route = backend.plan_rrt_connect(observation, goal, context, recovery)
                        planned_phases.add(context.phase)
                        result["rrt_calls"] = int(result.get("rrt_calls", 0)) + 1
                        route_qpos = np.asarray(route.qpos_path, dtype=np.float64)
                        route_eef = np.asarray(backend._plan_poses, dtype=np.float64)
                    route_phase = context.phase
                    if route_qpos.ndim != 2 or route_qpos.shape[1:] != (7,) or route_eef.shape != (len(route_qpos), 7) or len(route_qpos) < 2:
                        raise RuntimeError(f"{context.phase}_RRT_route_payload_malformed")
                    if reuse_path is None:
                        _atomic_npz(output_dir / f"{context.phase}_route.npz", {
                            "qpos_path": route_qpos,
                            "eef_reference_poses_world": route_eef,
                            "metadata_json": np.asarray(json.dumps({
                                "status": route.planner_status,
                                "phase": context.phase,
                                "source_snapshot_sha256": _sha(SOURCE_SNAPSHOT.read_bytes()) if context.phase == "transport" else _array_sha(source_integration),
                                "start_integration_sha256": _array_sha(source_integration),
                                "model_geometry_sha256": MODEL_SHA256,
                                "frozen_policy_sha256": _canonical_sha256(backend.safety_policy),
                                "planner_status": route.planner_status,
                                "planner_source": route.planner_source,
                                "planner_diagnostics": _jsonable(route.diagnostics),
                                "minimum_forbidden_clearance_m": route.minimum_forbidden_clearance_m,
                                "route_points": len(route_qpos),
                                "plan_elapsed_seconds_including_bridge": time.monotonic() - started_plan,
                                "historical_actions_replayed": False,
                                "qpos_reference_mode": "diagnostic_only",
                            }, ensure_ascii=False, sort_keys=True)),
                        })
                        _record_jsonl(output_dir / "task_phase_events.jsonl", {
                            "event": "existing_bounded_rrt_route_created_from_live_state",
                            "timestep": observation.timestep,
                            "phase": context.phase,
                            "goal": {
                                "goal_id": goal.goal_id,
                                "phase": goal.phase,
                                "source": goal.source,
                                "position_world_m": list(goal.position_world_m),
                                "quaternion_wxyz": list(goal.quaternion_wxyz),
                                "allowed_target_contacts": list(goal.allowed_target_contacts),
                            },
                            "route_points": len(route_qpos),
                            "planning_seconds": route.planning_seconds,
                            "minimum_clearance_m": route.minimum_forbidden_clearance_m,
                            "historical_route_used_as_structure_only": True,
                        })
                index = int(backend.plan_cursor)
                if index >= len(route_qpos) - 1:
                    reached = backend.is_recovery_goal_reached(observation, backend.resolve_goal(observation, context))
                    if reached:
                        # Force a fresh context update on the next loop to
                        # adopt precise_placement/settle from live evidence.
                        backend.latest_data["timestep"] = observation.timestep
                        context = backend.current_context()
                        route = None
                        route_phase = None
                        continue
                    raise RuntimeError(f"{context.phase}_existing_RRT_route_ended_outside_live_goal_tolerance")
                response = bridge.request({
                    "op": "osc_prefix_probe",
                    "snapshot_id": source_snapshot_id,
                    "current_state_sha256": _array_sha(source_integration.astype(np.float64)),
                    # The path prefix before `index` is already committed in
                    # this same Episode. Pass the remaining route beginning at
                    # its exact live-matched cursor to avoid revalidating old
                    # FK samples; no reference points or physical actions are
                    # skipped.
                    "jointspace_path_qpos": route_qpos[index:].tolist(),
                    "eef_reference_poses_world": route_eef[index:].tolist(),
                    "path_start_index": 0,
                    "max_control_steps": 1,
                    "gripper_action": 0.0,
                    "risk_trigger_distance_m": RISK_TRIGGER_M,
                    "risk_rearm_distance_m": RISK_REARM_M,
                    "risk_crossing_armed": bool(risk_armed),
                    "maximum_joint_path_deviation_rad": recovery.maximum_joint_path_deviation_rad,
                    "maximum_eef_tracking_error_m": recovery.maximum_eef_tracking_error_m,
                    "maximum_eef_orientation_error_rad": recovery.maximum_eef_orientation_error_rad,
                    "qpos_reference_mode": "diagnostic_only",
                    "current_minimum_forbidden_clearance_m": float(live_margin),
                    "safety_geometry_policy": backend.safety_policy,
                })
                if response.get("main_state_restored_verified") is not True or not response.get("complete") or not response.get("safe"):
                    raise RuntimeError(f"{context.phase}_isolated_RRT_OSC_branch_not_safe_or_restored:{response.get('failure_reason') or response.get('stop_reason')}")
                if response.get("contact_monitoring", {}).get("status") != "complete" or response.get("contact_monitoring", {}).get("captured_physics_substeps") != EXPECTED_SUBSTEPS:
                    raise RuntimeError(f"{context.phase}_RRT_OSC_preview_lacks_25_substeps")
                if response.get("qpos_reference_mode") != "diagnostic_only" or len(response.get("steps", [])) != 1:
                    raise RuntimeError("RRT_OSC_preview_did_not_certify_exactly_one_diagnostic_only_action")
                prediction_step = response["steps"][0]
                _record_actual = prediction_step
                substeps = prediction_step.get("safety_geometry_substeps", [])
                preview_audit = _audit_substeps(substeps, backend.safety_policy, held=context.held_object == OBJECT_ID, phase=context.phase)
                if float(prediction_step.get("eef_tracking_error_m", math.inf)) > recovery.maximum_eef_tracking_error_m or float(prediction_step.get("eef_orientation_error_rad", math.inf)) > recovery.maximum_eef_orientation_error_rad:
                    raise RuntimeError(f"{context.phase}_physical_branch_RRT_EEF_tracking_error_exceeded_frozen_limit")
                stop_reason = response.get("stop_reason")
                if stop_reason not in {None, "narrow_channel_entry_replan"}:
                    raise RuntimeError(f"{context.phase}_RRT_OSC_preview_physical_progress_failed:{stop_reason}")
                action = np.asarray(prediction_step["action"], dtype=np.float32)
                predicted_payload = prediction_step.get("terminal_runtime_snapshot")
                predicted_native = {}
                predicted_contacts = prediction_step.get("contacts", {})
                if not isinstance(predicted_payload, bytes):
                    raise RuntimeError("RRT_OSC_preview_terminal_snapshot_missing")
                preview_info = _inspect_predicted_terminal(
                    bridge, backend, source_snapshot_id, source_integration,
                    predicted_payload, context.phase,
                )
                predicted_native = preview_info["native_task_status"]
                route_diag = {
                    "route_index": index + 1,
                    "reference_qpos": route_qpos[index + 1].tolist(),
                    "predicted_qpos": prediction_step.get("actual_qpos"),
                    "qpos_reference_deviation_rad_diagnostic_only": prediction_step.get("joint_path_deviation_rad"),
                    "frozen_qpos_reference_limit_rad_logged_only": recovery.maximum_joint_path_deviation_rad,
                    "reference_eef_pose": route_eef[index + 1].tolist(),
                    "predicted_eef_tracking_error_m": prediction_step.get("eef_tracking_error_m"),
                    "predicted_eef_orientation_error_rad": prediction_step.get("eef_orientation_error_rad"),
                    "risk_probe_stop_reason_diagnostic": stop_reason,
                }
                backend._current_action_source = "safety_recovery_osc_pose"
            else:
                action = backend.task_expert_action(observation, context)
                if action is None:
                    raise RuntimeError(f"existing_TaskPhaseController_produced_no_native_OSC_action:{context.phase}")
                action = np.asarray(action, dtype=np.float32)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise RuntimeError(f"{context.phase}_TaskPhaseController_action_is_not_finite_native_7D")
                branch = bridge.request({
                    "op": "branch_rollout",
                    "snapshot_id": source_snapshot_id,
                    "candidates": [{"candidate_id": f"t1061_{context.phase}_{observation.timestep}", "actions": [action.tolist()]}],
                    "safety_geometry_policy": backend.safety_policy,
                    "capture_runtime_snapshots": True,
                })
                if branch.get("main_state_restored") is not True or len(branch.get("candidates", [])) != 1:
                    raise RuntimeError(f"{context.phase}_OSC_branch_did_not_restore_or_return_one_candidate")
                candidate = branch["candidates"][0]
                if candidate.get("status") != "ok" or candidate.get("executed_steps") != 1 or len(candidate.get("steps", [])) != 1:
                    raise RuntimeError(f"{context.phase}_OSC_physical_branch_incomplete:{candidate.get('error')}")
                prediction_step = candidate["steps"][0]
                monitor = candidate.get("contact_monitoring", {})
                geom = monitor.get("geometry_monitoring", {})
                if monitor.get("status") != "complete" or monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS or geom.get("status") != "complete" or len(geom.get("substeps", [])) != EXPECTED_SUBSTEPS:
                    raise RuntimeError(f"{context.phase}_OSC_branch_lacks_complete_25_substep_geometry")
                preview_audit = _audit_substeps(
                    geom["substeps"], backend.safety_policy,
                    held=context.held_object == OBJECT_ID, phase=context.phase,
                )
                predicted_contacts = prediction_step.get("contacts", {})
                _assert_robot_contacts(backend, predicted_contacts, context.phase)
                predicted_payload = prediction_step.get("terminal_runtime_snapshot")
                preview_info = _inspect_predicted_terminal(
                    bridge, backend, source_snapshot_id, source_integration,
                    predicted_payload, context.phase,
                )
                predicted_native = preview_info["native_task_status"]
                route_diag = {
                    "route_index": None,
                    "reference_qpos": None,
                    "predicted_qpos": prediction_step.get("state", {}).get("robot_states", [{}])[0].get("joint_qpos"),
                    "qpos_reference_deviation_rad_diagnostic_only": None,
                    "frozen_qpos_reference_limit_rad_logged_only": recovery.maximum_joint_path_deviation_rad,
                    "reference_eef_pose": None,
                    "predicted_eef_tracking_error_m": None,
                    "predicted_eef_orientation_error_rad": None,
                    "risk_probe_stop_reason_diagnostic": None,
                }
                backend._current_action_source = getattr(backend, "_current_action_source", "task_expert_osc_pose")

            pre_state = current["snapshot_state"]
            pre_pot1 = _object_state(pre_state, OBJECT_ID)
            pre_pot2 = _object_state(pre_state, POT2_ID)
            decision_t = observation.timestep
            # Store each terminal preview Snapshot before the physical commit.
            preview_snapshot_path = output_dir / "predicted_branch_snapshots" / f"predicted_timestep_{decision_t + 1:06d}.pkl"
            preview_snapshot_path.write_bytes(predicted_payload)
            action_record = {
                "event": "action_certified_for_commit",
                "timestep": decision_t,
                "phase": context.phase,
                "held_object": context.held_object,
                "action": action.tolist(),
                "gripper_action": float(action[6]),
                "phase_contact_policy": phase_policy,
                "predicted_substep_audit": preview_audit,
                "predicted_native_task_status": predicted_native,
                "predicted_grasp_evidence": preview_info.get("grasp_evidence"),
                "prediction_snapshot": str(preview_snapshot_path),
                "route_diagnostics": route_diag,
            }
            _record_jsonl(output_dir / "execution_certification.jsonl", action_record)
            backend.execute_action(action, context)
            actual = backend.latest_data
            if actual is None or int(actual.get("timestep", -1)) != decision_t + 1:
                raise RuntimeError("committed_OSC_action_did_not_advance_exactly_one_episode_timestep")
            actual_native = actual.get("native_task_status", {})
            actual_contacts = bridge.request({"op": "inspect_contacts"})
            actual_geometry = actual.get("contact_monitoring", {}).get("geometry_monitoring", {})
            actual_substeps = actual_geometry.get("substeps", [])
            if actual.get("contact_monitoring", {}).get("status") != "complete" or actual.get("contact_monitoring", {}).get("captured_physics_substeps") != EXPECTED_SUBSTEPS or len(actual_substeps) != EXPECTED_SUBSTEPS:
                raise RuntimeError("committed_action_lacks_complete_25_substep_monitor")
            actual_audit = _audit_substeps(actual_substeps, backend.safety_policy, held=context.held_object == OBJECT_ID, phase=context.phase)
            _assert_robot_contacts(backend, actual_contacts, context.phase)
            actual_held, actual_grasp_verified, actual_grasps = backend._grasp_states()
            if context.phase in {"transport", "precise_placement"} and actual_held != OBJECT_ID:
                raise RuntimeError(f"committed_action_lost_Pot1_grasp_in_{context.phase}")
            if context.phase == "release" and actual_held is None and _native_on(actual_native, POT1_ON) is not True:
                raise RuntimeError("committed_release_dropped_Pot1_without_native_On")
            if context.phase in {"withdrawal", "settle", "verify_task"} and (actual_held is not None or _native_on(actual_native, POT1_ON) is not True):
                raise RuntimeError(f"committed_post_release_state_invalid:{actual_held}:{_native_on(actual_native, POT1_ON)}")
            if _native_on(actual_native, POT2_ON) is not True or _native_on(actual_native, STOVE_ON) is not True:
                raise RuntimeError("committed_action_lost_Pot2_On_or_Stove_Turnon")
            actual_export = bridge.request({"op": "export_runtime_snapshot"})
            if actual_export.get("status") != "complete":
                raise RuntimeError("committed_runtime_snapshot_export_failed")
            actual_snapshot_payload = actual_export["payload"]
            predicted_contacts_for_compare = predicted_contacts
            physics_compare = _compare_prediction_commit(
                predicted_payload, actual_snapshot_payload, predicted_native,
                actual, prediction_step, predicted_contacts_for_compare,
                actual_contacts,
            )
            post_state = actual["snapshot_state"]
            post_pot1 = _object_state(post_state, OBJECT_ID)
            post_pot2 = _object_state(post_state, POT2_ID)
            pot2_displacement = float(np.linalg.norm(
                np.asarray(post_pot2["root_pos"], dtype=np.float64)
                - np.asarray(attestation["pot2_position_world_m"], dtype=np.float64)
            ))
            if _native_on(actual_native, POT2_ON) is not True:
                raise RuntimeError("Pot2_native_On_false_after_committed_action")
            if context.phase in {"transport", "precise_placement", "release"} and actual_held == OBJECT_ID:
                grasp_record = actual_grasps.get(OBJECT_ID, {})
                if not grasp_record.get("grasp_confirmed") or not grasp_record.get("finger_contacts", {}).get("left") or not grasp_record.get("finger_contacts", {}).get("right"):
                    raise RuntimeError(f"committed_action_Pot1_grasp_is_not_bilateral:{context.phase}")
            # Record progress using the live stage target; this is diagnostic
            # for waypoints and a fail-closed test only over a full frozen
            # stagnation window, not on a single small OSC step.
            goal_before = np.asarray(context.goals[0].position_world_m, dtype=np.float64)
            eef_after = np.asarray(post_state["robot_states"][0]["eef_pos"], dtype=np.float64)
            residual_after = float(np.linalg.norm(goal_before - eef_after))
            progress_window.append(float(np.linalg.norm(
                eef_after - np.asarray(pre_state["robot_states"][0]["eef_pos"], dtype=np.float64)
            )))
            if len(progress_window) > recovery.stagnation_window_steps:
                progress_window.pop(0)
            _record_jsonl(output_dir / "execution_certification.jsonl", {
                "event": "physical_action_committed_and_audited",
                "timestep": int(actual["timestep"]),
                "phase": context.phase,
                "action": action.tolist(),
                "actual_substep_audit": actual_audit,
                "physics_prediction_commit": physics_compare,
                "Pot1_root_position_world_m": post_pot1["root_pos"],
                "Pot1_root_quaternion_wxyz": post_pot1["root_quat"],
                "Pot2_root_position_world_m": post_pot2["root_pos"],
                "Pot2_displacement_from_t1061_m": pot2_displacement,
                "Pot1_grasp_state": actual_grasps.get(OBJECT_ID),
                "native_task_status": actual_native,
                "reward": actual.get("reward"),
                "done": actual.get("done"),
                "phase_target_residual_after_m_diagnostic": residual_after,
                "risk_warning_only": float(actual_geometry.get("minimum_clearance_m") or 1.0) <= RISK_TRIGGER_M,
                "path_cursor": backend.plan_cursor,
                "route_diagnostics": route_diag,
            })
            result["physical_actions_committed"] = backend.transition_count
            result["final_timestep"] = int(actual["timestep"])
            result["current_phase"] = context.phase
            result["native_task_success"] = actual_native.get("task_success")
            result["Pot1_On"] = _native_on(actual_native, POT1_ON)
            result["Pot2_On"] = _native_on(actual_native, POT2_ON)
            result["Stove_Turnon"] = _native_on(actual_native, STOVE_ON)
            result["reward"] = actual.get("reward")
            result["done"] = actual.get("done")
            result["Pot2_displacement_from_t1061_m"] = pot2_displacement

            # Refresh the phase state only after the committed physical state
            # and all 25 actual substeps have passed.
            next_context = backend.current_context()
            if next_context.phase != context.phase:
                _record_jsonl(output_dir / "task_phase_events.jsonl", {
                    "event": "task_phase_transition_observed",
                    "timestep": int(actual["timestep"]),
                    "from_phase": context.phase,
                    "to_phase": next_context.phase,
                    "held_object": next_context.held_object,
                    "native_task_status": backend.current_native_task_status,
                })
                route = None
                route_phase = None
                previous_phase = next_context.phase
            if actual.get("task_success") is True and float(actual.get("reward", 0.0)) >= 1.0 and actual.get("done") is True:
                actual_values = _native_values(actual_native)
                placement_released = actual_held is None
                all_required_native_values = (
                    actual_values.get(POT1_ON) is True
                    and actual_values.get(POT2_ON) is True
                    and actual_values.get(STOVE_ON) is True
                )
                if all_required_native_values and placement_released:
                    terminal_reason = "full_native_success_reward_done_after_verified_release"
                    result["full_native_success"] = True
                    result["full_native_success_timestep"] = int(actual["timestep"])
                    break
                # If the task is already terminal but the active object is
                # still grasped, keep the configured release phase active so
                # the native success state cannot be mistaken for a completed
                # placement before the physical gripper release is observed.
                if next_context.phase == "verify_task":
                    terminal_reason = "full_native_task_success_reward_done_and_stable_phase_verified"
                    break
            if len(progress_window) == recovery.stagnation_window_steps and sum(progress_window) < recovery.minimum_progress_m and next_context.phase in {"transport", "precise_placement", "withdrawal"}:
                raise RuntimeError(f"physical_stagnation_fail_closed:{next_context.phase}:{sum(progress_window)}m")
            if actual.get("done") is True and actual.get("task_success") is not True:
                raise RuntimeError("environment_done_without_native_task_success")

        terminal = backend.observe()
        final_native = terminal.raw.get("native_task_status", {})
        final_values = _native_values(final_native)
        final_context = backend.current_context()
        final_reward = result.get("reward")
        final_done = result.get("done")
        full_success = bool(
            final_values.get(POT1_ON) is True
            and final_values.get(POT2_ON) is True
            and final_values.get(STOVE_ON) is True
            and final_native.get("task_success") is True
            and final_reward is not None and float(final_reward) >= 1.0
            and final_done is True
        )
        result.update({
            "status": "complete_native_task_success_safe_suffix" if full_success else "stopped_or_incomplete_safe_suffix",
            "stop_reason": terminal_reason or "loop_stopped_without_success",
            "final_timestep": terminal.timestep,
            "physical_actions_committed": backend.transition_count,
            "final_phase": final_context.phase,
            "final_native_task_status": final_native,
            "Pot1_On": final_values.get(POT1_ON),
            "Pot2_On": final_values.get(POT2_ON),
            "Stove_Turnon": final_values.get(STOVE_ON),
            "task_success": final_native.get("task_success"),
            "reward": final_reward,
            "done": final_done,
            "standard_1000_step_success": bool(full_success and terminal.timestep <= STANDARD_HORIZON),
            "demonstration_1800_step_success": bool(full_success and terminal.timestep <= MAX_DEMONSTRATION_HORIZON),
            "verified_same_episode_safe_success_suffix": bool(full_success and backend.transition_count > 0),
            "complete_round0_seed": False,
            "early_t10_prefix_current_full_safety_replay_still_required": True,
            "rrt_calls_by_phase": sorted(planned_phases),
            "qpos_reference_mode": "diagnostic_only_for_OSC_route_execution",
            "risk_hysteresis_m": [RISK_TRIGGER_M, RISK_REARM_M],
            "elapsed_seconds": time.monotonic() - started,
        })
        return result
    except BaseException as exc:
        current = backend.latest_data if backend is not None else None
        result.update({
            "status": "interrupted_or_failed_fail_closed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_timestep": int(current.get("timestep", -1)) if current else None,
            "failure_phase": getattr(backend.task_controller, "phase", None) if backend else None,
            "physical_actions_committed": backend.transition_count if backend else 0,
            "complete_round0_seed": False,
            "task_success": False,
        })
        return result
    finally:
        if backend is not None:
            try:
                backend.finalize({
                    "complete_task_success": bool(result.get("task_success")),
                    "complete_safety_success": bool(result.get("physical_actions_committed", 0) > 0 and not result.get("status", "").endswith("failed_fail_closed")),
                    "local_recovery_success": bool(result.get("task_success")),
                    "standard_horizon_success": bool(result.get("standard_1000_step_success")),
                    "extended_horizon": bool(result.get("demonstration_1800_step_success")),
                    **result,
                })
            except Exception as exc:
                (output_dir / "finalize_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            for writer in backend._video_writers.values():
                writer.release()
        bridge.close()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        result["output_dir"] = str(output_dir)
        result["elapsed_seconds"] = time.monotonic() - started
        _write_json(output_dir / "suffix_result.json", result)
        if backend is not None:
            try:
                backend.trajectory.save(steps=backend.transition_count, chunks=backend.transition_count)
            except Exception:
                pass


def main() -> int:
    args = _parser()
    result = run(args)
    print(json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.get("task_success") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
