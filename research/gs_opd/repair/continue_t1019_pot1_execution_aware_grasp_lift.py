# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Continue Pot 1 from t1019 through stable grasp and a verified 30 mm lift.

TaskPhaseController generates each native OSC_POSE action. The Safety Bridge's
isolated MuJoCo prefix probe certifies that exact 7D action from the exact live
Runtime Snapshot before it is committed. This is a local suffix, not a complete
Round 0 seed, and it does not invoke a planner.
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
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from scipy.spatial.transform import Rotation

from research.gs_opd.repair.automatic_recovery import RecoveryArtifactWriter
from research.gs_opd.repair.automatic_recovery_runtime import (
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
)
from research.gs_opd.repair.continue_t794_pot1_execution_aware_pregrasp import (
    EXPECTED_SUBSTEPS,
    MODEL_SHA256,
    POT1_ON,
    POT2_ON,
    STOVE_ON,
    _PythonObjectState,
    _array_sha256,
    _check_substeps,
    _native_values,
    _normalized_state_sha256,
    _quat_error_rad,
    _validate_native_stage,
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
    "/opt/data/private/gs_opd/rollouts/"
    "l0t796_pot1_execution_aware_pregrasp_20261025_021845"
)
SOURCE_SNAPSHOT = SOURCE_ARCHIVE / "snapshots/main_timestep_001019.pkl"
SOURCE_TRACE = SOURCE_ARCHIVE / "observation_action_trace.npz"
SOURCE_EVENTS = SOURCE_ARCHIVE / "recovery_events.jsonl"
SOURCE_RESULT = SOURCE_ARCHIVE / "suffix_result.json"
SOURCE_CONFIG = SOURCE_ARCHIVE / "run_config.json"
SOURCE_JOURNAL_INDEX = SOURCE_ARCHIVE / "durable_action_journal/index.jsonl"
INITIAL_TIMESTEP = 1019
MAX_DEMONSTRATION_HORIZON = 1800
STANDARD_HORIZON = 1000
EXPECTED_OBJECT_ORDER = ["moka_pot_2", "moka_pot_1"]
RISK_TRIGGER_M = 0.050
RISK_REARM_M = 0.060
EXPECTED_FIRST_ACTION: list[float] | None = None
RUN_POLICY_METADATA: dict[str, Any] = {}

setattr(__main__, "_PythonObjectState", _PythonObjectState)


def _object_state(physical: dict[str, Any], object_id: str) -> dict[str, Any]:
    rows = [
        value for key, value in physical.get("object_states", {}).items()
        if str(key).endswith(":" + object_id)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"live_object_state_not_unique:{object_id}:{len(rows)}")
    return rows[0]


def _robot_contact_rows(contacts: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for contact in contacts.get("contacts", []):
        body1, body2 = contact.get("body1", {}), contact.get("body2", {})
        if str(body1.get("name", "")).startswith(("robot0_", "gripper0_")) or str(
            body2.get("name", "")
        ).startswith(("robot0_", "gripper0_")):
            rows.append(contact)
    return rows


def _validate_finger_only_contacts(
    contacts: dict[str, Any], *, finger_geom_ids: set[int], target_geom_ids: set[int], context: str
) -> list[dict[str, Any]]:
    rows = _robot_contact_rows(contacts)
    invalid = []
    for row in rows:
        geom1 = int(row.get("geom1", {}).get("id", -1))
        geom2 = int(row.get("geom2", {}).get("id", -1))
        if not (
            (geom1 in finger_geom_ids and geom2 in target_geom_ids)
            or (geom2 in finger_geom_ids and geom1 in target_geom_ids)
        ):
            invalid.append({
                "geom1": row.get("geom1"), "geom2": row.get("geom2"),
                "body1": row.get("body1"), "body2": row.get("body2"),
            })
            continue
        penetration = row.get("signed_contact_distance_m")
        if penetration is None or not np.isfinite(float(penetration)):
            invalid.append({
                "geom1": row.get("geom1"), "geom2": row.get("geom2"),
                "reason": "finger_target_contact_depth_missing",
            })
        elif float(penetration) < -0.002 - 1e-9:
            invalid.append({
                "geom1": row.get("geom1"), "geom2": row.get("geom2"),
                "signed_contact_distance_m": float(penetration),
                "allowed_minimum_m": -0.002,
            })
    if invalid:
        raise RuntimeError(f"{context}_nonfinger_or_non_target_robot_contact:{invalid[:8]}")
    return rows


def _check_allowed_contact_clearances(
    substeps: list[dict[str, Any]], safety_policy: dict[str, Any]
) -> dict[str, Any]:
    rules = {
        (int(row["robot_geom_id"]), int(row["scene_geom_id"])): float(
            row["minimum_clearance_m"]
        )
        for row in safety_policy.get("allowed_contact_pairs", [])
    }
    minima: dict[str, float] = {}
    failures = []
    for index, substep in enumerate(substeps):
        observed = substep.get("allowed_contact_clearances_m", {})
        for (robot_geom, scene_geom), required in rules.items():
            key = f"{robot_geom}:{scene_geom}"
            value = observed.get(key)
            if value is None or not np.isfinite(float(value)):
                failures.append({"substep": index, "pair": key, "reason": "missing_clearance"})
                continue
            distance = float(value)
            minima[key] = min(distance, minima.get(key, math.inf))
            if distance < required - 1e-9:
                failures.append({
                    "substep": index, "pair": key, "distance_m": distance,
                    "required_m": required,
                })
    if failures:
        raise RuntimeError("allowed_finger_contact_clearance_failed:" + json.dumps(failures[:12]))
    return {"checked_pairs": len(rules), "minimum_by_geom_pair_m": minima, "failures": []}


def _source_attestation() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    prior = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    if (
        result.get("status") != "pregrasp_reached"
        or result.get("final_timestep") != INITIAL_TIMESTEP
        or result.get("final_phase") != "approach"
        or result.get("native_safety_and_operational_safety") is not True
        or result.get("Pot2_On_preserved") is not True
    ):
        raise ValueError("archived_t1019_suffix_does_not_attest_the_required_approach_state")
    if not SOURCE_SNAPSHOT.is_file() or not SOURCE_TRACE.is_file() or not SOURCE_EVENTS.is_file():
        raise FileNotFoundError("t1019_snapshot_or_its_action_evidence_is_missing")

    snapshot_sha = hashlib.sha256(SOURCE_SNAPSHOT.read_bytes()).hexdigest()
    with SOURCE_JOURNAL_INDEX.open("r", encoding="utf-8") as stream:
        journal_rows = [json.loads(line) for line in stream if line.strip()]
    matching = [row for row in journal_rows if int(row.get("timestep", -1)) == INITIAL_TIMESTEP]
    if len(matching) != 1 or matching[0].get("runtime_snapshot_path") != str(SOURCE_SNAPSHOT):
        raise ValueError("t1019_snapshot_is_not_the_durable_action_journal_terminal_state")
    if hashlib.sha256(Path(matching[0]["journal_path"]).read_bytes()).hexdigest() != matching[0].get(
        "journal_sha256"
    ):
        raise ValueError("t1019_durable_action_record_hash_mismatch")
    envelope = pickle.loads(SOURCE_SNAPSHOT.read_bytes())
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
    ):
        raise ValueError("t1019_snapshot_schema_or_model_fingerprint_mismatch")
    snapshot = envelope.get("snapshot")
    if not isinstance(snapshot, dict) or int(
        snapshot.get("task_bookkeeping", {}).get("timestep", -1)
    ) != INITIAL_TIMESTEP:
        raise ValueError("t1019_snapshot_internal_timestep_mismatch")
    integration = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    if integration.shape != (317,) or not np.isfinite(integration).all():
        raise ValueError("t1019_snapshot_Integration_State_invalid")
    robots = snapshot.get("robots", [])
    if len(robots) != 1 or not robots[0].get("has_gripper_current_action"):
        raise ValueError("t1019_snapshot_robot_or_gripper_runtime_state_missing")

    with np.load(SOURCE_TRACE, allow_pickle=False) as trace:
        if (
            trace["executed_actions"].shape != (223, 7)
            or int(trace["step_timesteps"][-1]) != INITIAL_TIMESTEP
            or len(trace["post_physical_state_json"]) != 223
        ):
            raise ValueError("t1019_OA_terminal_boundary_is_malformed")
        physical = json.loads(str(trace["post_physical_state_json"][-1]))
        native = json.loads(str(trace["post_native_task_status_json"][-1]))
        monitor = json.loads(str(trace["post_contact_monitoring_json"][-1]))
        if not np.array_equal(
            np.asarray(physical.get("mujoco_integration_state"), dtype=np.float64), integration
        ):
            raise ValueError("t1019_snapshot_Integration_State_differs_from_terminal_OA")
        if (
            monitor.get("status") != "complete"
            or monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
            or monitor.get("geometry_monitoring", {}).get("status") != "complete"
        ):
            raise ValueError("t1019_terminal_action_lacks_complete_25_substep_evidence")
        if not trace["post_agentview_image"][-1].size or not trace[
            "post_robot0_eye_in_hand_image"
        ][-1].size:
            raise ValueError("t1019_terminal_dual_camera_observation_missing")

    model_hash = hashlib.sha256(integration.tobytes()).hexdigest()
    if matching[0].get("timestep") != INITIAL_TIMESTEP:
        raise ValueError("t1019_journal_timestep_mismatch")
    if _native_values(native) != {POT1_ON: False, POT2_ON: True, STOVE_ON: True}:
        raise ValueError("t1019_native_predicates_do_not_match_Pot1_approach_state")
    if native.get("task_success") is not False or native.get("done") is not False:
        raise ValueError("t1019_native_task_unexpectedly_terminated")

    events = [json.loads(line) for line in SOURCE_EVENTS.read_text(encoding="utf-8").splitlines() if line]
    transitions = [
        event for event in events
        if event.get("event") == "task_phase_transition"
        and int(event.get("timestep", -1)) == INITIAL_TIMESTEP
        and event.get("to_phase") == "approach"
        and event.get("object_id") == "moka_pot_1"
    ]
    if len(transitions) != 1:
        raise ValueError("t1019_approach_transition_event_is_not_unique")

    physical_robot = physical["robot_states"][0]
    pot1 = _object_state(physical, "moka_pot_1")
    pot2 = _object_state(physical, "moka_pot_2")
    attestation = {
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(SOURCE_SNAPSHOT),
        "source_snapshot_sha256": snapshot_sha,
        "source_action_record": str(SOURCE_TRACE),
        "source_action_record_sha256": hashlib.sha256(SOURCE_TRACE.read_bytes()).hexdigest(),
        "timestep": INITIAL_TIMESTEP,
        "model_geometry_sha256": MODEL_SHA256,
        "object_id": "moka_pot_1",
        "arm_joint_qpos": physical_robot["joint_qpos"],
        "eef_position_world_m": physical_robot["eef_pos"],
        "eef_quaternion_wxyz": physical_robot["eef_quat"],
        "gripper_qpos_m": physical_robot["gripper_qpos"],
        "object_position_world_m": pot1["root_pos"],
        "object_quaternion_wxyz": pot1["root_quat"],
        "restored_gripper_current_action": robots[0].get("gripper_current_action"),
        "integration_state_sha256": hashlib.sha256(integration.tobytes()).hexdigest(),
        "runtime_snapshot_normalized_sha256": _normalized_state_sha256(snapshot),
        "source_pot2_position_world_m": pot2["root_pos"],
        "source_pot2_quaternion_wxyz": pot2["root_quat"],
        "native_status": native,
        "phase_transition_event": transitions[0],
        "safety_policy_sha256": prior.get("frozen_operational_safety_policy_sha256"),
    }
    source = {
        "snapshot": snapshot,
        "integration_state": integration,
        "physical": physical,
        "native": native,
        "monitor": monitor,
        "policy": prior.get("frozen_operational_safety_policy"),
        "phase_event": transitions[0],
    }
    if not isinstance(source["policy"], dict):
        raise ValueError("t1019_source_frozen_operational_safety_policy_missing")
    return attestation, source, prior


def _assert_static_policy_unchanged(actual: dict[str, Any], frozen: dict[str, Any]) -> None:
    dynamic_keys = {"allowed_contact_pairs", "carried_object"}
    actual_static = {key: value for key, value in actual.items() if key not in dynamic_keys}
    frozen_static = {key: value for key, value in frozen.items() if key not in dynamic_keys}
    if _canonical_sha256(actual_static) != _canonical_sha256(frozen_static):
        raise RuntimeError("restored_static_operational_safety_policy_changed")
    forbidden = {
        str(item["object_id"]): float(item["minimum_clearance_m"])
        for item in actual.get("forbidden_objects", [])
    }
    if forbidden.get("bottle_of_ginger_beer_1") != 0.010 or forbidden.get(
        "white_place_box_1"
    ) != 0.010:
        raise RuntimeError("Ginger_Beer_or_Place_Box_hard_10mm_clearance_changed")


def _clearance_minimum(substeps: list[dict[str, Any]], object_ids: list[str]) -> float | None:
    values = []
    for row in substeps:
        for object_id in object_ids:
            record = row.get("clearances_by_object", {}).get(object_id)
            value = record.get("minimum_distance_m") if isinstance(record, dict) else None
            if value is not None:
                values.append(float(value))
        for object_id in object_ids:
            record = row.get("carried_clearances_by_object", {}).get(object_id)
            value = record.get("minimum_distance_m") if isinstance(record, dict) else None
            if value is not None:
                values.append(float(value))
    return min(values) if values else None


def _action_reference_pose(action: np.ndarray, live_robot: dict[str, Any], fields: dict[str, Any]):
    input_min = np.asarray(fields["input_min"], dtype=np.float64)
    input_max = np.asarray(fields["input_max"], dtype=np.float64)
    output_min = np.asarray(fields["output_min"], dtype=np.float64)
    output_max = np.asarray(fields["output_max"], dtype=np.float64)
    if any(row.shape != (6,) for row in (input_min, input_max, output_min, output_max)):
        raise RuntimeError("active_native_OSC_POSE_scale_fields_are_malformed")
    if np.any(input_max <= input_min) or np.any(output_max <= output_min):
        raise RuntimeError("active_native_OSC_POSE_scale_ranges_are_invalid")
    world_delta = output_min + (action[:6].astype(np.float64) - input_min) * (
        (output_max - output_min) / (input_max - input_min)
    )
    current_pos = np.asarray(live_robot["eef_pos"], dtype=np.float64)
    current_q = np.asarray(live_robot["eef_quat"], dtype=np.float64)
    current_rotation = Rotation.from_quat(current_q[[1, 2, 3, 0]])
    target_rotation = Rotation.from_rotvec(world_delta[3:]) * current_rotation
    target_xyzw = target_rotation.as_quat()
    return np.concatenate((current_pos + world_delta[:3], target_xyzw[[3, 0, 1, 2]]))


def _inspect_predicted_action(
    bridge: BridgeProcess,
    backend: DurableTaskSuffixBackend,
    snapshot_id: str,
    source_integration: np.ndarray,
    payload: bytes,
    expected_phase: str,
    safety_policy: dict[str, Any],
    finger_geom_ids: set[int],
    target_geom_ids: set[int],
) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError("isolated_action_preview_has_no_terminal_Runtime_Snapshot")
    envelope = pickle.loads(payload)
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
    ):
        raise RuntimeError("isolated_action_preview_model_or_snapshot_schema_mismatch")
    expected_snapshot = envelope.get("snapshot", {})
    expected_integration = np.asarray(expected_snapshot.get("mujoco_state"), dtype=np.float64)
    if expected_integration.shape != source_integration.shape:
        raise RuntimeError("predicted_Integration_State_shape_mismatch")
    predicted_native = {}
    predicted_contacts = {}
    predicted_grasp = {}
    predicted_geometry = {}
    try:
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
        if restored.get("status") != "restored":
            raise RuntimeError(f"predicted_action_Snapshot_restore_failed:{restored.get('error')}")
        native = bridge.request({"op": "native_task_status"})
        _validate_native_stage(native, context=f"predicted_{expected_phase}")
        predicted_native = _native_values(native)
        predicted_contacts = bridge.request({"op": "inspect_contacts"})
        held, grasp_verified, grasp_records = backend._grasp_states()
        predicted_grasp = grasp_records.get("moka_pot_1", {})
        if held not in {None, "moka_pot_1"}:
            raise RuntimeError(f"predicted_wrong_object_held:{held}")
        if expected_phase == "lift" and (
            held != "moka_pot_1"
            or not predicted_grasp.get("grasp_confirmed")
            or not grasp_verified
        ):
            raise RuntimeError("predicted_lift_state_lost_stable_Pot1_grasp")
        _validate_finger_only_contacts(
            predicted_contacts, finger_geom_ids=finger_geom_ids,
            target_geom_ids=target_geom_ids, context=f"predicted_{expected_phase}",
        )
        backend._refresh_task_contact_policy(expected_phase, "moka_pot_1", held)
        observed = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": safety_policy,
        })
        predicted_geometry = observed.get("safety_geometry", {})
        if predicted_geometry.get("status") != "complete":
            raise RuntimeError("predicted_terminal_operational_geometry_incomplete")
        for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
        ):
            if predicted_geometry.get(key):
                raise RuntimeError(f"predicted_terminal_safety_violation:{key}")
        live_integration = np.asarray(
            observed.get("snapshot_state", {}).get("mujoco_integration_state"),
            dtype=np.float64,
        )
        if not np.array_equal(live_integration, expected_integration):
            raise RuntimeError("predicted_runtime_snapshot_differs_from_restored_terminal_state")
        if predicted_native.get(POT2_ON) is not True:
            raise RuntimeError("predicted_branch_lost_Pot2_native_On")
        return {
            "timestep": int(observed.get("timestep", -1)),
            "native_status": native,
            "native_predicates": {"pot1_on": predicted_native[POT1_ON], "pot2_on": predicted_native[POT2_ON], "stove_turnon": predicted_native[STOVE_ON]},
            "held_object": held,
            "grasp_state": predicted_grasp,
            "contact_geom_pairs": [
                sorted((int(row["geom1"]["id"]), int(row["geom2"]["id"])))
                for row in predicted_contacts.get("contacts", [])
            ],
            "predicted_geometry": predicted_geometry,
            "integration_state_sha256": _array_sha256(expected_integration),
        }
    finally:
        restored_source = bridge.request({"op": "restore", "snapshot_id": snapshot_id})
        if restored_source.get("restored_snapshot_id") != snapshot_id:
            raise RuntimeError(f"source_Snapshot_restore_after_preview_failed:{restored_source.get('error')}")
        live = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": safety_policy,
        })
        actual = np.asarray(live.get("snapshot_state", {}).get("mujoco_integration_state"), dtype=np.float64)
        if not np.array_equal(actual, source_integration):
            raise RuntimeError("source_Integration_State_not_exactly_restored_after_preview")
        backend.latest_data = live
        backend.latest_data["native_task_status"] = bridge.request({"op": "native_task_status"})
        backend.latest_data["contacts"] = bridge.request({"op": "inspect_contacts"})
        source_held, _source_grasp_verified, _source_grasp_records = backend._grasp_states()
        if source_held not in {None, "moka_pot_1"}:
            raise RuntimeError(f"restored_source_wrong_object_held_after_preview:{source_held}")
        backend._refresh_task_contact_policy(expected_phase, "moka_pot_1", source_held)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38754)
    return parser.parse_args()


def run(args) -> dict[str, Any]:
    attestation, source, prior_config = _source_attestation()
    frozen_policy = source["policy"]
    config_data, recovery = load_runtime_config(args.config)
    config_data = json.loads(json.dumps(config_data))
    config_data["candidate_search"]["enabled"] = False
    target_clearances = {
        str(item["object_id"]): float(item["minimum_clearance_m"])
        for item in frozen_policy.get("allowed_target_objects", [])
    }
    recovery = recovery.__class__(
        **{
            **recovery.__dict__,
            "object_clearance_m": target_clearances,
            "execution_step_budget": MAX_DEMONSTRATION_HORIZON,
            "horizon_steps": MAX_DEMONSTRATION_HORIZON,
            "standard_horizon_steps": STANDARD_HORIZON,
            "allow_horizon_extension": True,
        }
    )
    recovery.validate()
    config_data["recovery"] = _jsonable(recovery.__dict__)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(
        f"/opt/data/private/gs_opd/rollouts/l0t{INITIAL_TIMESTEP}_pot1_grasp_lift_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    run_config = {
        "entrypoint": RUN_POLICY_METADATA.get("entrypoint", str(Path(__file__).resolve())),
        "implementation_module": str(Path(__file__).resolve()),
        "source_archive": str(SOURCE_ARCHIVE),
        "source_snapshot": str(SOURCE_SNAPSHOT),
        "source_attestation": attestation,
        "source_frozen_safety_policy_sha256": prior_config.get("frozen_operational_safety_policy_sha256"),
        "mode": "strict_runtime_snapshot_taskphase_native_action_rolling_certification",
        "resume_timestep": INITIAL_TIMESTEP,
        "resume_snapshot": str(SOURCE_SNAPSHOT),
        "phases_targeted": ["approach", "close_gripper", "stable_grasp", "lift"],
        "per_action_physics_substeps": EXPECTED_SUBSTEPS,
        "risk_warning_50mm_diagnostic_only": True,
        "risk_trigger_m": RISK_TRIGGER_M,
        "risk_rearm_m": RISK_REARM_M,
        "qpos_reference_mode": "diagnostic_only",
        "maximum_total_task_timestep": MAX_DEMONSTRATION_HORIZON,
        "standard_benchmark_horizon": STANDARD_HORIZON,
        "new_RRT_calls_authorized": 0,
        "LoRA_training_started": False,
        "safety_policy_source": frozen_policy.get("source"),
        "frozen_operational_safety_policy": frozen_policy,
        "policy_reuse_metadata": RUN_POLICY_METADATA,
    }
    _write_json(output_dir / "run_config.json", run_config)
    _atomic_npz(output_dir / f"source_runtime_state_t{INITIAL_TIMESTEP:06d}.npz", {
        "integration_state": source["integration_state"],
        "source_snapshot_sha256": np.asarray(attestation["source_snapshot_sha256"]),
        "source_integration_state_sha256": np.asarray(attestation["integration_state_sha256"]),
    })

    artifacts = RecoveryArtifactWriter(output_dir)
    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend = None
    result: dict[str, Any] = {"status": "preflight_not_completed", "physical_actions_committed": 0}
    original_handlers = {}
    started = time.monotonic()
    action_events = (output_dir / "rolling_action_certificates.jsonl").open("a", encoding="utf-8")

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal_{signum}_received_after_durable_action_boundary")

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.signal(signum, interrupt)
    try:
        bridge.start(
            benchmark=config_data["benchmark"], level=int(config_data["level"]),
            task_id=int(config_data["task_id"]), init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=MAX_DEMONSTRATION_HORIZON,
        )
        backend = DurableTaskSuffixBackend(
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
            raise RuntimeError("strict_t1019_restore_timestep_mismatch")
        if restored.get("scene_metadata", {}).get("model_geometry_sha256") != MODEL_SHA256:
            raise RuntimeError("strict_t1019_restore_model_fingerprint_mismatch")
        restored_state = np.asarray(
            restored.get("snapshot_state", {}).get("mujoco_integration_state"), dtype=np.float64
        )
        if not np.array_equal(restored_state, source["integration_state"]):
            raise RuntimeError("strict_t1019_restore_Integration_State_mismatch")
        exported = bridge.request({"op": "export_runtime_snapshot"})
        if exported.get("status") != "complete":
            raise RuntimeError("strict_t1019_controller_runtime_export_failed")
        restored_snapshot = pickle.loads(exported["payload"]).get("snapshot", {})
        restored_runtime_hash = _normalized_state_sha256(restored_snapshot)
        if restored_runtime_hash != attestation["runtime_snapshot_normalized_sha256"]:
            raise RuntimeError("strict_t1019_robot_gripper_controller_state_mismatch")

        backend.latest_data = restored
        backend._load_context_inputs(restored)
        if _canonical_sha256(backend.safety_policy) != _canonical_sha256(frozen_policy):
            raise RuntimeError("t1019_restored_safety_policy_does_not_match_frozen_archive_policy")
        if [str(row["object_id"]) for row in backend.task_controller.objects] != EXPECTED_OBJECT_ORDER:
            raise RuntimeError("native_task_phase_object_order_changed")

        native = bridge.request({"op": "native_task_status"})
        native_values = _validate_native_stage(native, context="strict_t1019_restore")
        held, grasp_verified, grasp_records = backend._grasp_states()
        if held is not None:
            raise RuntimeError(f"t1019_unexpected_object_held_before_Pot1_approach:{held}")

        # The t1019 archived phase-transition event and live pregrasp pose
        # establish approach. Internal counters are fresh; none are replayed.
        controller = backend.task_controller
        controller.object_index = 1
        controller.phase = "approach"
        controller._expert_handoff = True
        controller._phase_steps = 0
        controller._last_timestep = -1
        controller._grasp_last_timestep = -1
        controller._grasp_streak = 0
        controller._lift_start = None
        controller._lift_eef_start = None
        controller._settle_samples.clear()
        controller._settle_last_timestep = -1
        backend.task_expert_active = False
        backend.last_task_phase = None
        backend.task_phase_anchor = None
        backend.confirmed_placed_objects.clear()

        preflight = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        preflight["native_task_status"] = native
        preflight["contacts"] = bridge.request({"op": "inspect_contacts"})
        backend.latest_data = preflight
        context = backend.current_context()
        context.validate()
        if context.phase != "approach" or context.held_object is not None:
            raise RuntimeError(f"t1019_live_phase_reconstruction_mismatch:{context.phase}:{context.held_object}")
        target = context.goals[0]
        _assert_static_policy_unchanged(backend.safety_policy, frozen_policy)
        finger_ids = set(int(value) for value in backend.context_static["gripper_geometry_ids"])
        pot1_ids = set(int(value) for value in backend.bindings["moka_pot_1"]["geom_ids"])
        allowed_pairs = {
            (int(row["robot_geom_id"]), int(row["scene_geom_id"]))
            for row in backend.safety_policy.get("allowed_contact_pairs", [])
        }
        expected_pairs = {(finger, target_geom) for finger in finger_ids for target_geom in pot1_ids}
        if allowed_pairs != expected_pairs or len(finger_ids) != 4:
            raise RuntimeError("t1019_approach_contact_policy_is_not_exact_four_finger_to_Pot1")
        if grasp_records.get("moka_pot_1", {}).get("grasp_confirmed") is True:
            raise RuntimeError("t1019_source_unexpectedly_reports_Pot1_held")

        # Rebuild the controller phase from the archived transition plus the
        # current native/live target; the Python controller counters are new.
        backend.handoff_to_task_expert(backend._observation(preflight), context)
        context = backend.current_context()
        if context.phase != "approach":
            raise RuntimeError(f"t1019_handoff_changed_phase_unexpectedly:{context.phase}")
        bridge_contacts = bridge.request({"op": "inspect_contacts"})
        source_contacts = _validate_finger_only_contacts(
            bridge_contacts, finger_geom_ids=finger_ids, target_geom_ids=pot1_ids,
            context="strict_t1019_source",
        )
        preflight = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        preflight["native_task_status"] = native
        preflight["contacts"] = bridge_contacts
        geometry = preflight.get("safety_geometry", {})
        if geometry.get("status") != "complete":
            raise RuntimeError("t1019_live_operational_geometry_incomplete")
        for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
        ):
            if geometry.get(key):
                raise RuntimeError(f"t1019_live_safety_preflight_not_clean:{key}")
        if not np.array_equal(
            np.asarray(preflight["snapshot_state"]["mujoco_integration_state"], dtype=np.float64),
            source["integration_state"],
        ):
            raise RuntimeError("t1019_readonly_preflight_changed_physical_state")
        backend.latest_data = preflight
        backend.initialize_trajectory()
        backend._save_video_frame(preflight)
        source_clearance = geometry.get("minimum_forbidden_clearance_m")
        if source_clearance is None or not np.isfinite(float(source_clearance)):
            raise RuntimeError("t1019_forbidden_clearance_unavailable")
        risk_armed = float(source_clearance) >= RISK_REARM_M
        attested = {
            "status": "strict_t1019_restore_and_approach_preflight_passed",
            "source_snapshot_sha256": attestation["source_snapshot_sha256"],
            "source_integration_state_sha256": attestation["integration_state_sha256"],
            "source_runtime_controller_hash": restored_runtime_hash,
            "model_geometry_sha256": MODEL_SHA256,
            "native_predicates": native_values,
            "task_phase": context.phase,
            "task_phase_counters_replayed": False,
            "task_target_source": target.source,
            "task_target_position_world_m": list(target.position_world_m),
            "task_target_quaternion_wxyz": list(target.quaternion_wxyz),
            "task_goal_position_error_m": context.metadata.get("phase_goal_position_error_m"),
            "task_goal_orientation_error_rad": context.metadata.get("phase_goal_orientation_error_rad"),
            "Pot1_grasp": grasp_records.get("moka_pot_1"),
            "Pot2_On": True,
            "finger_contact_geom_ids": sorted(finger_ids),
            "Pot1_geom_ids": sorted(pot1_ids),
            "allowed_contact_pairs_exact": True,
            "source_robot_contacts": source_contacts,
            "live_geometry": geometry,
            "frozen_static_policy_unchanged": True,
            "risk_warning_50mm_is_diagnostic_only": True,
        }
        _write_json(output_dir / "strict_restore_preflight.json", attested)
        artifacts.event({"event": "strict_t1019_approach_restore_preflight_passed", **attested})

        initial_object = _object_state(preflight["snapshot_state"], "moka_pot_1")
        initial_pot1_z = float(np.asarray(initial_object["root_pos"], dtype=np.float64)[2])
        phase_errors: list[dict[str, Any]] = []
        risk_events: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        phase_progress_history: list[tuple[str, float, float]] = []
        stop_reason = None
        last_failure = None

        while True:
            current_t = int(backend.latest_data.get("timestep", -1))
            if current_t >= MAX_DEMONSTRATION_HORIZON:
                stop_reason = "demonstration_horizon_1800_reached_before_stable_lift"
                break

            # Re-query state and safety geometry before every candidate action.
            prior_data = backend.latest_data
            prior_integration = np.asarray(
                prior_data.get("snapshot_state", {}).get("mujoco_integration_state"),
                dtype=np.float64,
            )
            live = bridge.request({
                "op": "observe", "include_snapshot_state": True,
                "safety_geometry_policy": backend.safety_policy,
            })
            live_integration = np.asarray(
                live.get("snapshot_state", {}).get("mujoco_integration_state"), dtype=np.float64
            )
            if int(live.get("timestep", -1)) != current_t or not np.array_equal(
                live_integration, prior_integration
            ):
                raise RuntimeError("pre_action_live_snapshot_does_not_match_last_committed_state")
            live_native = bridge.request({"op": "native_task_status"})
            _validate_native_stage(live_native, context="pre_action")
            live["native_task_status"] = live_native
            live["contacts"] = bridge.request({"op": "inspect_contacts"})
            backend.latest_data = live
            context = backend.current_context()
            context.validate()
            if context.phase == "transport":
                lift_start = controller._lift_start
                current_object = _object_state(live["snapshot_state"], "moka_pot_1")
                rise = (
                    float(np.asarray(current_object["root_pos"], dtype=np.float64)[2] - lift_start[2])
                    if lift_start is not None else 0.0
                )
                held_now, verified_now, grasp_now = backend._grasp_states()
                if rise < 0.030 or held_now != "moka_pot_1" or not grasp_now[
                    "moka_pot_1"
                ].get("grasp_confirmed"):
                    raise RuntimeError("controller_transport_transition_lacks_30mm_verified_lift")
                stop_reason = "pot1_stable_finger_only_grasp_and_30mm_lift_verified"
                records.append({
                    "event": "lift_milestone_verified",
                    "timestep": current_t,
                    "rise_from_controller_lift_start_m": rise,
                    "rise_from_t1019_m": float(np.asarray(current_object["root_pos"])[2] - initial_pot1_z),
                    "held_object": held_now,
                    "grasp_state": grasp_now["moka_pot_1"],
                    "grasp_state_verified": verified_now,
                })
                break
            if context.phase not in {"approach", "close_gripper", "stable_grasp", "lift"}:
                stop_reason = f"unexpected_phase_before_lift_success:{context.phase}"
                break
            live_geom = live.get("safety_geometry", {})
            if live_geom.get("status") != "complete":
                raise RuntimeError("pre_action_live_operational_geometry_incomplete")
            for key in (
                "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
                "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
            ):
                if live_geom.get(key):
                    raise RuntimeError(f"pre_action_live_safety_not_clean:{key}")
            _validate_finger_only_contacts(
                live["contacts"], finger_geom_ids=finger_ids,
                target_geom_ids=pot1_ids, context=f"pre_action_t{current_t}",
            )
            margin = live_geom.get("minimum_forbidden_clearance_m")
            if margin is None or not np.isfinite(float(margin)):
                raise RuntimeError("pre_action_forbidden_clearance_unknown")
            if not risk_armed and float(margin) >= RISK_REARM_M:
                risk_armed = True
            warning_now = float(margin) <= RISK_TRIGGER_M
            if warning_now and risk_armed:
                risk_armed = False
                risk_events.append({"timestep": current_t, "minimum_forbidden_clearance_m": float(margin)})

            observation = backend._observation(live)
            action = backend.task_expert_action(observation, context)
            if action is None:
                raise RuntimeError(f"task_phase_controller_produced_no_native_action:{context.phase}")
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError("task_phase_controller_action_is_not_finite_native_7d")
            if current_t == INITIAL_TIMESTEP and EXPECTED_FIRST_ACTION is not None:
                expected_first = np.asarray(EXPECTED_FIRST_ACTION, dtype=np.float32)
                if expected_first.shape != (7,) or not np.allclose(
                    action, expected_first, rtol=0.0, atol=1e-7
                ):
                    raise RuntimeError(
                        "first_resumed_TaskPhaseController_action_differs_from_preapproved_t1042_action"
                    )

            # Exact OSC output scaling maps this task-controller action back to
            # a diagnostic EEF reference. The action itself is the previewed
            # input; EEF/qpos reference deviations never authorize or reject it.
            repair = bridge.request({"op": "repair_context"})
            if repair.get("status") != "available" or not repair.get("robots"):
                raise RuntimeError("active_OSC_controller_fields_unavailable")
            robot_fields = repair["robots"][0]["fields"]
            robot_state = live["snapshot_state"]["robot_states"][0]
            current_pose = np.concatenate((
                np.asarray(robot_state["eef_pos"], dtype=np.float64),
                np.asarray(robot_state["eef_quat"], dtype=np.float64),
            ))
            reference_pose = _action_reference_pose(action, robot_state, robot_fields)
            poses = np.stack((current_pose, reference_pose), axis=0)
            qpos = np.asarray(robot_state["joint_qpos"], dtype=np.float64)
            qpos_path = np.stack((qpos, qpos), axis=0)
            snapshot_response = bridge.request({"op": "snapshot"})
            snapshot_id = snapshot_response.get("snapshot_id")
            snap_state = np.asarray(
                snapshot_response.get("snapshot_state", {}).get("mujoco_integration_state"),
                dtype=np.float64,
            )
            if not isinstance(snapshot_id, str) or not np.array_equal(snap_state, live_integration):
                raise RuntimeError("pre_action_snapshot_does_not_match_fresh_live_state")
            preview = bridge.request({
                "op": "osc_prefix_probe",
                "snapshot_id": snapshot_id,
                "current_state_sha256": _array_sha256(live_integration),
                "jointspace_path_qpos": qpos_path.tolist(),
                "eef_reference_poses_world": poses.tolist(),
                "path_start_index": 0,
                "max_control_steps": 1,
                "gripper_action": float(action[6]),
                "native_action_prefix": [action.tolist()],
                "risk_trigger_distance_m": RISK_TRIGGER_M,
                "risk_rearm_distance_m": RISK_REARM_M,
                "risk_crossing_armed": risk_armed,
                "maximum_joint_path_deviation_rad": recovery.maximum_joint_path_deviation_rad,
                "maximum_eef_tracking_error_m": recovery.maximum_eef_tracking_error_m,
                "maximum_eef_orientation_error_rad": recovery.maximum_eef_orientation_error_rad,
                "qpos_reference_mode": "diagnostic_only",
                "current_minimum_forbidden_clearance_m": float(margin),
                "safety_geometry_policy": backend.safety_policy,
            })
            if (
                preview.get("complete") is not True
                or preview.get("safe") is not True
                or preview.get("main_state_restored_verified") is not True
                or len(preview.get("steps", [])) != 1
                or preview.get("action_source") != "explicit_native_7d_action_prefix"
                or preview.get("stop_reason") not in {
                    None, "narrow_channel_entry_replan", "motion_stagnation",
                }
            ):
                raise RuntimeError("current_state_native_action_preview_failed:" + json.dumps({
                    key: preview.get(key) for key in (
                        "status", "complete", "safe", "stop_reason", "failure_reason",
                        "main_state_restored_verified", "action_source",
                    )
                }, sort_keys=True))
            branch_step = preview["steps"][0]
            substep_audit = _check_substeps(
                branch_step.get("safety_geometry_substeps", []), backend.safety_policy
            )
            substep_audit["allowed_contact_clearances"] = _check_allowed_contact_clearances(
                branch_step.get("safety_geometry_substeps", []), backend.safety_policy
            )
            predicted_path = output_dir / "branch_snapshots" / f"preview_timestep_{current_t + 1:06d}.pkl"
            predicted_path.parent.mkdir(parents=True, exist_ok=True)
            predicted_path.write_bytes(branch_step["terminal_runtime_snapshot"])
            predicted = _inspect_predicted_action(
                bridge, backend, snapshot_id, live_integration,
                branch_step["terminal_runtime_snapshot"], context.phase,
                backend.safety_policy, finger_ids, pot1_ids,
            )
            if not np.array_equal(
                np.asarray(backend.latest_data["snapshot_state"]["mujoco_integration_state"], dtype=np.float64),
                live_integration,
            ):
                raise RuntimeError("source_state_not_restored_before_action_commit")
            certificate = {
                "source_timestep": current_t,
                "target_timestep": current_t + 1,
                "phase": context.phase,
                "source_integration_state_sha256": _array_sha256(live_integration),
                "native_action": action.tolist(),
                "gripper_command": float(action[6]),
                "task_controller_target_source": context.target_source,
                "task_goal_position_error_m": context.metadata.get("phase_goal_position_error_m"),
                "task_goal_orientation_error_rad": context.metadata.get("phase_goal_orientation_error_rad"),
                "qpos_reference_mode": "diagnostic_only",
                "qpos_reference_limit_rad": recovery.maximum_joint_path_deviation_rad,
                "eef_reference_pose_diagnostic": reference_pose.tolist(),
                "risk_margin_before_m": float(margin),
                "risk_warning_only": warning_now,
                "preview_stop_reason_diagnostic": preview.get("stop_reason"),
                "preview_step": {
                    key: branch_step.get(key) for key in (
                        "joint_path_deviation_rad", "eef_tracking_error_m",
                        "eef_orientation_error_rad", "reward", "done", "safety_geometry_substeps",
                    )
                },
                "preview_contact_monitoring": preview.get("contact_monitoring"),
                "preview_substep_audit": substep_audit,
                "predicted_state": predicted,
                "predicted_snapshot": str(predicted_path),
                "committed": False,
            }
            action_events.write(json.dumps(_jsonable(certificate), ensure_ascii=False, sort_keys=True) + "\n")
            action_events.flush()
            os.fsync(action_events.fileno())

            # Commit only the exact action certified above; any next action is
            # generated from a newly observed actual state and recertified.
            current = bridge.request({
                "op": "observe", "include_snapshot_state": True,
                "safety_geometry_policy": backend.safety_policy,
            })
            current_integration = np.asarray(
                current.get("snapshot_state", {}).get("mujoco_integration_state"), dtype=np.float64
            )
            if not np.array_equal(current_integration, live_integration):
                raise RuntimeError("physical_state_changed_between_action_preview_and_commit")
            current["native_task_status"] = bridge.request({"op": "native_task_status"})
            current["contacts"] = bridge.request({"op": "inspect_contacts"})
            backend.latest_data = current
            backend._current_action_source = "task_expert_osc_pose"
            committed_observation = backend.execute_action(action, context)
            committed_step = backend.trajectory.steps[-1]
            monitor = json.loads(committed_step["contact_monitoring_json"])
            actual_substeps = monitor.get("geometry_monitoring", {}).get("substeps", [])
            actual_audit = _check_substeps(actual_substeps, backend.safety_policy)
            actual_audit["allowed_contact_clearances"] = _check_allowed_contact_clearances(
                actual_substeps, backend.safety_policy
            )
            if (
                monitor.get("status") != "complete"
                or int(monitor.get("captured_physics_substeps", -1)) != EXPECTED_SUBSTEPS
                or monitor.get("geometry_monitoring", {}).get("status") != "complete"
            ):
                raise RuntimeError("committed_action_missing_complete_25_substep_monitoring")
            actual_native = bridge.request({"op": "native_task_status"})
            _validate_native_stage(actual_native, context="committed_action")
            actual_contacts = bridge.request({"op": "inspect_contacts"})
            actual_contact_rows = _validate_finger_only_contacts(
                actual_contacts, finger_geom_ids=finger_ids,
                target_geom_ids=pot1_ids, context=f"committed_t{committed_observation.timestep}",
            )
            held_after, grasp_verified_after, grasp_records_after = backend._grasp_states()
            if held_after not in {None, "moka_pot_1"}:
                raise RuntimeError(f"wrong_object_held_after_commit:{held_after}")
            if context.phase == "lift" and (
                held_after != "moka_pot_1"
                or not grasp_records_after["moka_pot_1"].get("grasp_confirmed")
                or not grasp_verified_after
            ):
                raise RuntimeError("physical_Pot1_grasp_lost_during_lift")
            if committed_observation.done or committed_observation.task_success is True:
                raise RuntimeError("unexpected_native_episode_termination_before_requested_lift_milestone")
            actual_native_values = _native_values(actual_native)
            if actual_native_values[POT2_ON] is not True:
                raise RuntimeError("physical_commit_lost_Pot2_On")
            actual_state = committed_observation.raw["snapshot_state"]
            actual_integration = np.asarray(actual_state["mujoco_integration_state"], dtype=np.float64)
            predicted_integration = np.asarray(
                pickle.loads(branch_step["terminal_runtime_snapshot"])["snapshot"]["mujoco_state"],
                dtype=np.float64,
            )
            delta = actual_integration - predicted_integration
            actual_robot = actual_state["robot_states"][0]
            actual_object = _object_state(actual_state, "moka_pot_1")
            row = {
                "timestep": int(committed_observation.timestep),
                "phase": context.phase,
                "native_action": action.tolist(),
                "gripper_command": float(action[6]),
                "native_predicates": {"pot1_on": actual_native_values[POT1_ON], "pot2_on": actual_native_values[POT2_ON], "stove_turnon": actual_native_values[STOVE_ON]},
                "held_object": held_after,
                "grasp_state": grasp_records_after.get("moka_pot_1"),
                "robot_contact_count": len(actual_contact_rows),
                "contact_identities": actual_contact_rows,
                "substep_safety": actual_audit,
                "pot1_root_position_world_m": actual_object["root_pos"],
                "pot1_eef_position_world_m": actual_robot["eef_pos"],
                "gripper_qpos_m": actual_robot["gripper_qpos"],
                "qpos_reference_deviation_rad_diagnostic_only": branch_step.get("joint_path_deviation_rad"),
                "eef_reference_position_error_m_diagnostic_only": branch_step.get("eef_tracking_error_m"),
                "prediction_commit_Integration_State_delta_max_abs_diagnostic_only": float(np.max(np.abs(delta))),
            }
            if context.phase == "lift" and controller._lift_start is not None:
                row["pot1_rise_from_lift_anchor_m"] = float(
                    np.asarray(actual_object["root_pos"], dtype=np.float64)[2] - controller._lift_start[2]
                )
            records.append(row)
            certificate["committed"] = True
            certificate["actual_commit"] = row
            action_events.write(json.dumps(_jsonable(certificate), ensure_ascii=False, sort_keys=True) + "\n")
            action_events.flush()
            os.fsync(action_events.fileno())

            # Update the existing TaskPhaseController from actual physical and
            # native evidence. Its phase counters were reset, not fabricated.
            post_context = backend.current_context()
            post_context.validate()
            phase_errors.append({
                "timestep": int(committed_observation.timestep),
                "phase_before_action": context.phase,
                "phase_after_observed_transition": post_context.phase,
                "position_error_m": post_context.metadata.get("phase_goal_position_error_m"),
                "orientation_error_rad": post_context.metadata.get("phase_goal_orientation_error_rad"),
                "held_object": post_context.held_object,
                "grasp_streak": int(controller._grasp_streak),
                "phase_steps": int(controller._phase_steps),
            })
            metric = None
            if post_context.phase == "approach":
                metric = float(post_context.metadata.get("phase_goal_position_error_m", math.inf))
            elif post_context.phase in {"close_gripper", "stable_grasp"}:
                metric = float(controller._grasp_streak)
            elif post_context.phase == "lift" and controller._lift_start is not None:
                metric = float(
                    np.asarray(_object_state(backend.latest_data["snapshot_state"], "moka_pot_1")["root_pos"])[2]
                    - controller._lift_start[2]
                )
            if metric is not None:
                phase_progress_history.append((post_context.phase, metric, float(committed_observation.timestep)))
                phase_progress_history = [
                    item for item in phase_progress_history
                    if item[0] == post_context.phase and committed_observation.timestep - item[2] <= 12
                ]
                if len(phase_progress_history) >= 12:
                    first, last = phase_progress_history[0][1], phase_progress_history[-1][1]
                    if post_context.phase == "approach" and first - last < 0.002:
                        stop_reason = "approach_actual_goal_error_stagnation"
                        last_failure = {"phase": "approach", "window_actions": 12, "position_error_reduction_m": first - last}
                        break
                    if post_context.phase == "lift" and last - first < 0.002:
                        stop_reason = "lift_actual_object_rise_stagnation"
                        last_failure = {"phase": "lift", "window_actions": 12, "object_rise_progress_m": last - first}
                        break
            if post_context.phase == "transport":
                obj = _object_state(backend.latest_data["snapshot_state"], "moka_pot_1")
                rise = float(np.asarray(obj["root_pos"], dtype=np.float64)[2] - controller._lift_start[2])
                held_final, verified_final, grasp_final = backend._grasp_states()
                if rise < 0.030 or held_final != "moka_pot_1" or not grasp_final[
                    "moka_pot_1"
                ].get("grasp_confirmed") or not verified_final:
                    raise RuntimeError("transport_phase_transition_without_verified_30mm_stable_lift")
                stop_reason = "pot1_stable_finger_only_grasp_and_30mm_lift_verified"
                records.append({
                    "event": "lift_milestone_verified",
                    "timestep": int(committed_observation.timestep),
                    "rise_from_controller_lift_start_m": rise,
                    "rise_from_t1019_m": float(
                        np.asarray(obj["root_pos"], dtype=np.float64)[2] - initial_pot1_z
                    ),
                    "held_object": held_final,
                    "grasp_state": grasp_final["moka_pot_1"],
                    "grasp_state_verified": verified_final,
                })
                break

        final_observation = backend.observe()
        final_native = final_observation.raw.get("native_task_status") or bridge.request({"op": "native_task_status"})
        final_stage = _validate_native_stage(final_native, context="terminal")
        safety_rows = []
        clearance_minima: dict[str, float] = {}
        for step in backend.trajectory.steps:
            monitor = json.loads(step["contact_monitoring_json"])
            substeps = monitor.get("geometry_monitoring", {}).get("substeps", [])
            audit = _check_substeps(substeps, backend.safety_policy)
            safety_rows.append(monitor.get("status") == "complete" and len(substeps) == EXPECTED_SUBSTEPS)
            for row in substeps:
                for name, record in row.get("clearances_by_object", {}).items():
                    value = record.get("minimum_distance_m") if isinstance(record, dict) else None
                    if value is not None:
                        clearance_minima[str(name)] = min(float(value), clearance_minima.get(str(name), math.inf))
                for name, record in row.get("carried_clearances_by_object", {}).items():
                    value = record.get("minimum_distance_m") if isinstance(record, dict) else None
                    if value is not None:
                        key = "carried:" + str(name)
                        clearance_minima[key] = min(float(value), clearance_minima.get(key, math.inf))
        summary = {
            "actions": backend.transition_count,
            "physical_substeps": sum(len(json.loads(step["contact_monitoring_json"]).get("geometry_monitoring", {}).get("substeps", [])) for step in backend.trajectory.steps),
            "all_substep_coverage_and_safety_pass": bool(safety_rows) and all(safety_rows),
            "minimum_clearance_by_object_m": clearance_minima,
            "strict_10mm_all_recorded_clearances": bool(clearance_minima) and all(value >= 0.010 for value in clearance_minima.values()),
        }
        final_physical = final_observation.raw["snapshot_state"]
        final_pot1 = _object_state(final_physical, "moka_pot_1")
        final_held, final_grasp_verified, final_grasp = backend._grasp_states()
        lift_event = next((row for row in reversed(records) if row.get("event") == "lift_milestone_verified"), None)
        success = stop_reason == "pot1_stable_finger_only_grasp_and_30mm_lift_verified"
        result = {
            "status": "pot1_stable_grasp_and_lift_verified" if success else "stopped_before_lift_milestone",
            "initial_timestep": INITIAL_TIMESTEP,
            "final_timestep": int(final_observation.timestep),
            "physical_actions_committed": backend.transition_count,
            "final_phase": controller.phase,
            "final_stop_reason": stop_reason,
            "failure_detail": last_failure,
            "final_native_task_status": final_native,
            "final_native_stage": final_stage,
            "Pot2_On_preserved": final_stage["pot2_on"],
            "Pot1_On": final_stage["pot1_on"],
            "Pot1_held_object": final_held,
            "Pot1_grasp_state": final_grasp.get("moka_pot_1"),
            "Pot1_grasp_state_verified": final_grasp_verified,
            "Pot1_root_position_world_m": final_pot1["root_pos"],
            "Pot1_lift_rise_from_t1019_m": float(
                np.asarray(final_pot1["root_pos"], dtype=np.float64)[2] - initial_pot1_z
            ),
            "verified_lift_milestone": lift_event,
            "task_success": bool(final_native.get("task_success")),
            "reward": float(final_native.get("reward", 0.0)),
            "done": bool(final_native.get("done", False)),
            "safety_acceptance": summary,
            "native_safety_and_operational_safety": summary["all_substep_coverage_and_safety_pass"],
            "risk_warnings_are_diagnostic_only": True,
            "risk_warning_events": risk_events,
            "phase_transitions_and_progress": phase_errors,
            "new_RRT_calls": 0,
            "complete_round0_seed": False,
            "LoRA_training_started": False,
            "standard_1000_step_complete_task_success": bool(final_native.get("task_success")) and int(final_observation.timestep) <= STANDARD_HORIZON,
            "demonstration_1800_step_complete_task_success": bool(final_native.get("task_success")) and int(final_observation.timestep) <= MAX_DEMONSTRATION_HORIZON,
            "output_dir": str(output_dir),
        }
        _write_json(output_dir / "suffix_result.json", result)
    except BaseException as exc:
        result = {
            "status": "failed_closed_or_interrupted",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_timestep": int(backend.latest_data.get("timestep", -1)) if backend and backend.latest_data else None,
            "phase": getattr(getattr(backend, "task_controller", None), "phase", None),
            "physical_actions_committed": backend.transition_count if backend else 0,
            "strict_restore_completed": bool(backend and backend.latest_data),
            "complete_round0_seed": False,
            "new_RRT_calls": 0,
            "LoRA_training_started": False,
            "output_dir": str(output_dir),
        }
        _write_json(output_dir / "failure_report.json", result)
    finally:
        action_events.close()
        if backend is not None:
            try:
                backend.finalize({
                    "complete_task_success": False,
                    "complete_safety_success": False,
                    "local_recovery_success": result.get("status") == "pot1_stable_grasp_and_lift_verified",
                    "standard_horizon_success": bool(result.get("standard_1000_step_complete_task_success")),
                    "extended_horizon": bool(result.get("demonstration_1800_step_complete_task_success")),
                    "result": result,
                })
            except Exception as exc:
                (output_dir / "finalize_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                try:
                    backend.trajectory.save(steps=backend.transition_count, chunks=backend.transition_count)
                except Exception as save_exc:
                    (output_dir / "trajectory_save_error.txt").write_text(f"{type(save_exc).__name__}: {save_exc}\n", encoding="utf-8")
            for writer in backend._video_writers.values():
                writer.release()
        bridge.close()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        result["elapsed_seconds"] = time.monotonic() - started
        result["output_dir"] = str(output_dir)
        _write_json(output_dir / "suffix_result.json", result)
        _write_json(output_dir / "execution_manifest.json", {
            "entrypoint": RUN_POLICY_METADATA.get("entrypoint", str(Path(__file__).resolve())),
            "implementation_module": str(Path(__file__).resolve()),
            "source_snapshot": str(SOURCE_SNAPSHOT),
            "source_snapshot_sha256": attestation["source_snapshot_sha256"],
            "new_RRT_calls": 0,
            "physical_actions_committed": backend.transition_count if backend else 0,
            "result_status": result.get("status"),
            "complete_round0_seed": False,
        })
    return result


def main() -> int:
    args = _parser()
    result = run(args)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "pot1_stable_grasp_and_lift_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
