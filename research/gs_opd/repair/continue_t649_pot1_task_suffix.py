# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Resume L0T2 from the audited t=649 state and execute the Pot 1 suffix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import signal
import time
from dataclasses import asdict, replace
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
    DurableTaskSuffixBackend,
    _jsonable,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "research/gs_opd/configs/l0t2_task_success_first_safety_reconstruction.json"
SOURCE_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_execution_aware_pot2_release_continue_t642_20261025"
)
SOURCE_AUDIT = SOURCE_ARCHIVE / "posthoc_acceptance_audit.json"
SOURCE_SNAPSHOT = SOURCE_ARCHIVE / "snapshots/main_timestep_000649.pkl"
SOURCE_TRACE = SOURCE_ARCHIVE / "observation_action_trace.npz"
SOURCE_POLICY = SOURCE_ARCHIVE / "safety_policy.json"
MODEL_SHA256 = "1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752"
INITIAL_TIMESTEP = 649
MAX_TASK_TIMESTEP = 1800
STANDARD_HORIZON = 1000
EXPECTED_SUBSTEPS = 25


class _PythonObjectState:
    """Snapshot pickle helper class emitted by the Safety Bridge as __main__."""

    def __init__(self, module=None, qualname=None, attributes=None):
        self.module = module
        self.qualname = qualname
        self.attributes = attributes or {}


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _normalize_snapshot_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _normalize_snapshot_value(value.item())
    if isinstance(value, _PythonObjectState) or (
        value.__class__.__name__ == "_PythonObjectState"
        and hasattr(value, "attributes")
    ):
        return {
            "kind": "python_object_state",
            "module": getattr(value, "module", None),
            "qualname": getattr(value, "qualname", None),
            "attributes": _normalize_snapshot_value(getattr(value, "attributes", {})),
        }
    if isinstance(value, dict):
        return {
            str(key): _normalize_snapshot_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize_snapshot_value(item) for item in value]
    if isinstance(value, bytes):
        return {"kind": "bytes", "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, float):
        return {"kind": "float", "hex": value.hex()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return {"kind": "repr", "value": repr(value)}


def _normalized_state_sha256(value: Any) -> str:
    payload = json.dumps(
        _normalize_snapshot_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _native_values(status: dict[str, Any]) -> dict[tuple[str, ...], bool]:
    if status.get("status") != "complete":
        raise RuntimeError(f"native_task_status_incomplete:{status.get('error')}")
    return {
        tuple(str(token).lower() for token in row.get("predicate", [])): bool(row["value"])
        for row in status.get("native_goal_values", [])
        if isinstance(row, dict) and isinstance(row.get("value"), bool)
    }


def _source_attestation() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    if audit.get("terminal_snapshot") != str(Path("snapshots/main_timestep_000649.pkl")):
        raise ValueError("t649_audit_terminal_snapshot_reference_mismatch")
    snapshot_sha = sha256_file(SOURCE_SNAPSHOT)
    if snapshot_sha != audit.get("terminal_snapshot_sha256"):
        raise ValueError("t649_runtime_snapshot_file_hash_mismatch")
    if audit.get("model_geometry_sha256") != MODEL_SHA256:
        raise ValueError("t649_archived_model_geometry_fingerprint_mismatch")
    if audit.get("terminal_timestep") != INITIAL_TIMESTEP:
        raise ValueError("t649_archive_terminal_timestep_mismatch")
    if audit.get("native_pot2_on") is not True or audit.get("native_pot1_on") is not False:
        raise ValueError("t649_archive_native_object_predicates_are_not_expected")
    if audit.get("pot2_placement_and_settle_milestone", {}).get("settle_samples") != 5:
        raise ValueError("t649_archive_lacks_five_sample_Pot2_settle_milestone")

    envelope = pickle.loads(SOURCE_SNAPSHOT.read_bytes())
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
    ):
        raise ValueError("t649_runtime_snapshot_schema_or_model_mismatch")
    snapshot = envelope.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("t649_runtime_snapshot_payload_missing")
    integration = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    if integration.shape != (317,) or not np.isfinite(integration).all():
        raise ValueError("t649_snapshot_Integration_State_shape_or_value_invalid")
    if int(snapshot.get("task_bookkeeping", {}).get("timestep", -1)) != INITIAL_TIMESTEP:
        raise ValueError("t649_snapshot_internal_timestep_mismatch")

    with np.load(SOURCE_TRACE, allow_pickle=False) as trace:
        if len(trace["executed_actions"]) != 7 or int(trace["step_timesteps"][-1]) != INITIAL_TIMESTEP:
            raise ValueError("t642_to_t649_terminal_action_trace_is_incomplete")
        physical = json.loads(str(trace["post_physical_state_json"][-1]))
        native = json.loads(str(trace["post_native_task_status_json"][-1]))
        if int(trace["step_timesteps"][-1]) != INITIAL_TIMESTEP:
            raise ValueError("t649_physical_action_record_timestep_mismatch")
    recorded_integration = np.asarray(physical.get("mujoco_integration_state"), dtype=np.float64)
    if recorded_integration.shape != integration.shape or not np.array_equal(
        recorded_integration, integration
    ):
        raise ValueError("t649_runtime_snapshot_Integration_State_differs_from_action_trace")
    predicates = _native_values(native)
    expected_predicates = {
        ("on", "moka_pot_1", "flat_stove_1_cook_region"): False,
        ("on", "moka_pot_2", "flat_stove_1_cook_region"): True,
        ("turnon", "flat_stove_1"): True,
    }
    if predicates != expected_predicates or native.get("task_success") is not False:
        raise ValueError("t649_action_trace_native_status_does_not_match_audited_successor_state")

    robot = physical["robot_states"][0]
    pot1 = physical["object_states"]["objects_dict:moka_pot_1"]
    pot2 = physical["object_states"]["objects_dict:moka_pot_2"]
    robots = snapshot.get("robots", [])
    if len(robots) != 1 or not robots[0].get("has_gripper_current_action"):
        raise ValueError("t649_native_robot_or_gripper_runtime_state_missing")
    attestation = {
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(SOURCE_SNAPSHOT),
        "source_snapshot_sha256": snapshot_sha,
        "source_action_record": str(SOURCE_TRACE),
        "timestep": INITIAL_TIMESTEP,
        "model_geometry_sha256": MODEL_SHA256,
        "object_id": "moka_pot_1",
        "arm_joint_qpos": robot["joint_qpos"],
        "eef_position_world_m": robot["eef_pos"],
        "eef_quaternion_wxyz": robot["eef_quat"],
        "gripper_qpos_m": robot["gripper_qpos"],
        "object_position_world_m": pot1["root_pos"],
        "object_quaternion_wxyz": pot1["root_quat"],
        "restored_gripper_current_action": np.asarray(
            robots[0]["gripper_current_action"], dtype=np.float64
        ).tolist(),
        "integration_state_sha256": _array_sha256(integration),
        "runtime_snapshot_normalized_sha256": _normalized_state_sha256(snapshot),
    }
    source = {
        "audit": audit,
        "snapshot": snapshot,
        "integration_state": integration,
        "integration_state_sha256": attestation["integration_state_sha256"],
        "runtime_snapshot_normalized_sha256": attestation[
            "runtime_snapshot_normalized_sha256"
        ],
        "native_status": native,
        "pot1_state": pot1,
        "pot2_state": pot2,
        "policy": json.loads(SOURCE_POLICY.read_text(encoding="utf-8")),
    }
    return attestation, source, physical


def _safety_acceptance(trajectory, policy: dict[str, Any]) -> dict[str, Any]:
    actions = list(trajectory.steps)
    native_ok: bool | None = True if actions else None
    operational_ok: bool | None = True if actions else None
    strict_10mm: bool | None = True if actions else None
    pot2_on_preserved: bool | None = True if actions else None
    substeps_seen = 0
    clearance_minima: dict[str, float] = {}
    joint_limit_violations = 0
    native_task_history = []
    for action in actions:
        raw = action.get("contact_monitoring_json")
        monitor = json.loads(raw) if isinstance(raw, str) else {}
        geometry = monitor.get("geometry_monitoring", {})
        substeps = geometry.get("substeps", [])
        substeps_seen += len(substeps)
        complete = (
            monitor.get("status") == "complete"
            and geometry.get("status") == "complete"
            and int(monitor.get("captured_physics_substeps", -1)) == EXPECTED_SUBSTEPS
            and len(substeps) == EXPECTED_SUBSTEPS
        )
        if not complete:
            native_ok = operational_ok = False
        for substep in substeps:
            if substep.get("status") != "complete":
                native_ok = operational_ok = False
            if any(substep.get(key) for key in (
                "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
                "unallowed_scene_contacts", "unresolved_contacts",
            )):
                native_ok = False
            violations = substep.get("joint_limit_violations", [])
            joint_limit_violations += len(violations)
            if violations:
                operational_ok = False
            for name, detail in substep.get("clearances_by_object", {}).items():
                distance = detail.get("minimum_distance_m") if isinstance(detail, dict) else None
                if distance is None:
                    continue
                value = float(distance)
                clearance_minima[str(name)] = min(
                    value, clearance_minima.get(str(name), float("inf"))
                )
                if value < 0.010:
                    strict_10mm = False
            for name, value in substep.get("carried_clearances_by_object", {}).items():
                distance = value.get("minimum_distance_m") if isinstance(value, dict) else value
                if distance is None:
                    continue
                value = float(distance)
                label = f"carried:{name}"
                clearance_minima[label] = min(value, clearance_minima.get(label, float("inf")))
                if value < 0.010:
                    strict_10mm = False
        native_raw = action.get("native_task_status_json")
        status = json.loads(native_raw) if isinstance(native_raw, str) else {}
        vals = _native_values(status) if status.get("status") == "complete" else {}
        p1 = vals.get(("on", "moka_pot_1", "flat_stove_1_cook_region"))
        p2 = vals.get(("on", "moka_pot_2", "flat_stove_1_cook_region"))
        turnon = vals.get(("turnon", "flat_stove_1"))
        native_task_history.append({
            "timestep": status.get("timestep"),
            "pot1_on": p1,
            "pot2_on": p2,
            "stove_turnon": turnon,
            "task_success": status.get("task_success"),
            "reward": status.get("reward"),
            "done": status.get("done"),
        })
        if p2 is not True or turnon is not True:
            pot2_on_preserved = False
    return {
        "native_safety": native_ok,
        "operational_safety": operational_ok,
        "strict_10mm_supplementary": strict_10mm,
        "pot2_native_on_preserved_each_action": pot2_on_preserved,
        "physical_actions": len(actions),
        "expected_substeps": len(actions) * EXPECTED_SUBSTEPS,
        "observed_substeps": substeps_seen,
        "substep_coverage_complete": (
            substeps_seen == len(actions) * EXPECTED_SUBSTEPS if actions else None
        ),
        "operational_joint_limit_violations": joint_limit_violations,
        "minimum_clearance_by_group_m": clearance_minima,
        "operational_policy": policy,
        "native_task_history": native_task_history,
    }


class Pot1ContinuationBackend(DurableTaskSuffixBackend):
    """Existing recovery backend with durable one-action logging."""

    def preview_prefix(
        self, observation, plan, context, config, *, risk_boundary_armed=True
    ):
        # The t=649 gripper is already open. Keep it neutral in isolated RRT
        # branches through withdrawal, approach, and settle; close only when
        # the existing controller enters close_gripper.
        previous = self.config_data.get("open_gripper_action", -1.0)
        if context.held_object is None:
            self.config_data["open_gripper_action"] = 0.0
        try:
            return super().preview_prefix(
                observation,
                plan,
                context,
                config,
                risk_boundary_armed=risk_boundary_armed,
            )
        finally:
            self.config_data["open_gripper_action"] = previous

    def task_expert_action(self, observation, context):
        action = super().task_expert_action(observation, context)
        if (
            action is not None
            and context.held_object is None
            and context.phase in {
                "pregrasp", "approach", "withdrawal", "settle", "verify_task"
            }
        ):
            action = np.asarray(action, dtype=np.float32).copy()
            action[6] = 0.0
            self._current_action_source = "task_expert_osc_pose_neutral_open_gripper_hold"
            self.current_forecast = action[None, :].copy()
        return action


def _parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38649)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    attestation, source, source_physical = _source_attestation()
    config_data, recovery = load_runtime_config(args.config)
    config_data = json.loads(json.dumps(config_data))
    config_data["candidate_search"]["enabled"] = False
    source_policy = source["policy"]
    object_clearances = {
        str(item["object_id"]): float(item["minimum_clearance_m"])
        for key in ("forbidden_objects", "allowed_target_objects")
        for item in source_policy.get(key, [])
    }
    recovery = replace(
        recovery,
        hard_minimum_clearance_m=float(source_policy["minimum_clearance_m"]),
        target_object_clearance_m=float(source_policy["minimum_clearance_m"]),
        object_clearance_m=object_clearances,
        execution_step_budget=MAX_TASK_TIMESTEP,
        horizon_steps=MAX_TASK_TIMESTEP,
        standard_horizon_steps=STANDARD_HORIZON,
        allow_horizon_extension=True,
    )
    recovery.validate()
    config_data["recovery"] = asdict(recovery)
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(
        f"/opt/data/private/gs_opd/rollouts/l0t2_t649_pot1_task_suffix_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "run_config.json", {
        "entrypoint": str(Path(__file__).resolve()),
        "mode": "task-runtime-snapshot",
        "config_path": str(args.config),
        "source_archive": str(SOURCE_ARCHIVE),
        "source_snapshot_attestation": attestation,
        "frozen_policy_source": "t649_operational_policy_thresholds_and_native_BDDL_constraints",
        "operational_policy_thresholds_unchanged": True,
        "task_phase_reconstruction": "fresh_TaskPhaseController_transitions_from_live_native_Pot2_On_true_and_Pot1_On_false",
        "no_historical_phase_counters_or_controller_state_adopted": True,
        "candidate_search_disabled": True,
        "existing_bounded_rrt_enabled_for_unplanned_second_object_recovery_and_transport": True,
        "max_episode_timestep": MAX_TASK_TIMESTEP,
        "standard_benchmark_horizon": STANDARD_HORIZON,
        "safety_monitoring_substeps_per_action": EXPECTED_SUBSTEPS,
        "recovery_config": asdict(recovery),
    })
    artifacts = RecoveryArtifactWriter(output_dir)
    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend: Pot1ContinuationBackend | None = None
    engine: AutomaticSafetyRecoveryEngine | None = None
    result_json: dict[str, Any] | None = None
    started = time.monotonic()
    original_handlers = {}

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal_{signum}_received_after_durable_action_boundary")

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.signal(signum, interrupt)
    try:
        bridge.start(
            benchmark=config_data["benchmark"],
            level=int(config_data["level"]),
            task_id=int(config_data["task_id"]),
            init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=MAX_TASK_TIMESTEP,
        )
        backend = Pot1ContinuationBackend(
            bridge,
            config_data,
            recovery,
            output_dir,
            artifacts,
            mode="task-runtime-snapshot",
            student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=SOURCE_SNAPSHOT,
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
            runtime_snapshot_attestation=attestation,
            journal_dir=output_dir / "durable_action_journal",
        )
        restored = backend.restore_start_state()
        if int(restored.get("timestep", -1)) != INITIAL_TIMESTEP:
            raise RuntimeError("strict_t649_snapshot_restore_timestep_mismatch")
        if restored.get("scene_metadata", {}).get("model_geometry_sha256") != MODEL_SHA256:
            raise RuntimeError("strict_t649_snapshot_restore_model_fingerprint_mismatch")
        restored_integration = np.asarray(
            restored.get("snapshot_state", {}).get("mujoco_integration_state"),
            dtype=np.float64,
        )
        if not np.array_equal(restored_integration, source["integration_state"]):
            raise RuntimeError("strict_t649_snapshot_restore_Integration_State_mismatch")

        restored_export = bridge.request({"op": "export_runtime_snapshot"})
        if restored_export.get("status") != "complete":
            raise RuntimeError("strict_t649_runtime_controller_state_export_failed")
        restored_envelope = pickle.loads(restored_export["payload"])
        runtime_hash = _normalized_state_sha256(restored_envelope.get("snapshot", {}))
        if runtime_hash != source["runtime_snapshot_normalized_sha256"]:
            raise RuntimeError("strict_t649_robot_controller_gripper_or_integrator_state_mismatch")

        backend.latest_data = restored
        backend._load_context_inputs(restored)
        native_before = bridge.request({"op": "native_task_status"})
        native_before_values = _native_values(native_before)
        expected_native = {
            ("on", "moka_pot_1", "flat_stove_1_cook_region"): False,
            ("on", "moka_pot_2", "flat_stove_1_cook_region"): True,
            ("turnon", "flat_stove_1"): True,
        }
        if native_before_values != expected_native or native_before.get("task_success") is not False:
            raise RuntimeError("t649_live_native_task_predicates_do_not_match_source_audit")
        held_object, _grasp_verified, grasp_records = backend._grasp_states()
        if held_object is not None:
            raise RuntimeError(f"t649_expected_released_state_but_grasp_monitor_reports:{held_object}")
        live_contacts = bridge.request({"op": "inspect_contacts"})
        if live_contacts.get("status") not in {"contact", "no_contact"}:
            raise RuntimeError("t649_live_contact_inventory_is_unavailable")
        finger_ids = set(int(item) for item in backend.context_static["gripper_geometry_ids"])
        pot2_binding = backend.bindings["moka_pot_2"]
        pot2_body = int(pot2_binding["root_body_id"])
        pot2_geoms = set(int(item) for item in pot2_binding["geom_ids"])
        robot_contact_rows = []
        residual_pot2_finger_contact = False
        for contact in live_contacts.get("contacts", []):
            body1, body2 = contact.get("body1", {}), contact.get("body2", {})
            geom1, geom2 = contact.get("geom1", {}), contact.get("geom2", {})
            robot1 = str(body1.get("name", "")).startswith(("robot0_", "gripper0_"))
            robot2 = str(body2.get("name", "")).startswith(("robot0_", "gripper0_"))
            if not robot1 and not robot2:
                continue
            robot_body, scene_body = (body1, body2) if robot1 else (body2, body1)
            robot_geom, scene_geom = (geom1, geom2) if robot1 else (geom2, geom1)
            row = {
                "robot_body": robot_body,
                "robot_geom": robot_geom,
                "scene_body": scene_body,
                "scene_geom": scene_geom,
                "signed_contact_distance_m": contact.get("signed_contact_distance_m"),
            }
            robot_contact_rows.append(row)
            if (
                int(robot_geom.get("id", -1)) in finger_ids
                and int(scene_body.get("id", -1)) == pot2_body
                and int(scene_geom.get("id", -1)) in pot2_geoms
            ):
                residual_pot2_finger_contact = True
            else:
                raise RuntimeError(
                    "t649_residual_robot_contact_is_not_finger_only_to_already_placed_Pot2"
                )
        if residual_pot2_finger_contact:
            robot_state = restored["snapshot_state"]["robot_states"][0]
            backend.task_controller.adopt_released_object_withdrawal(
                "moka_pot_2",
                satisfied_object_ids=("moka_pot_2",),
                held_object=held_object,
            )
            backend.task_phase_anchor = (
                np.asarray(robot_state["eef_pos"], dtype=np.float64),
                np.asarray(robot_state["eef_quat"], dtype=np.float64),
            )
            backend.last_task_phase = "withdrawal"
            backend.task_expert_active = True
        context = backend.current_context()
        context.validate()
        if context.phase not in {"withdrawal", "second_object_recovery"} or context.held_object is not None:
            raise RuntimeError(
                f"t649_native_predicate_phase_reconstruction_mismatch:phase={context.phase},held={context.held_object}"
            )
        if context.phase == "second_object_recovery" and context.metadata.get("task_object") != "moka_pot_1":
            raise RuntimeError("t649_second_object_phase_did_not_select_Pot1")
        if context.phase == "withdrawal" and context.metadata.get("task_object") != "moka_pot_2":
            raise RuntimeError("t649_remaining_finger_contact_did_not_adopt_Pot2_withdrawal")
        if backend.safety_policy.get("minimum_clearance_m") != source_policy.get("minimum_clearance_m"):
            raise RuntimeError("t649_live_operational_global_clearance_policy_mismatch")
        for group_key in ("forbidden_objects", "allowed_target_objects"):
            expected = {
                row["object_id"]: float(row["minimum_clearance_m"])
                for row in source_policy.get(group_key, [])
            }
            actual = {
                row["object_id"]: float(row["minimum_clearance_m"])
                for row in backend.safety_policy.get(group_key, [])
            }
            if actual != expected:
                raise RuntimeError(f"t649_live_operational_clearance_policy_mismatch:{group_key}")

        # Obtain a live, policy-scoped safety/contact read before any action.
        preflight = bridge.request({
            "op": "observe",
            "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        native_status = bridge.request({"op": "native_task_status"})
        preflight["native_task_status"] = native_status
        backend.latest_data = preflight
        if preflight.get("timestep") != INITIAL_TIMESTEP:
            raise RuntimeError("t649_preflight_physical_timestep_changed_before_action")
        preflight_state = np.asarray(
            preflight.get("snapshot_state", {}).get("mujoco_integration_state"),
            dtype=np.float64,
        )
        if not np.array_equal(preflight_state, source["integration_state"]):
            raise RuntimeError("t649_read_only_preflight_changed_Integration_State")
        geometry = preflight.get("safety_geometry", {})
        _write_json(output_dir / "strict_restore_preflight.json", {
            "status": "live_safety_preflight_complete" if geometry.get("status") == "complete" else "live_safety_preflight_unsafe_or_incomplete",
            "source_snapshot": str(SOURCE_SNAPSHOT),
            "source_snapshot_sha256": attestation["source_snapshot_sha256"],
            "model_geometry_sha256": MODEL_SHA256,
            "timestep": INITIAL_TIMESTEP,
            "integration_state_exact_match": True,
            "runtime_internal_state_exact_match": True,
            "task_phase": context.phase,
            "task_object": context.metadata.get("task_object"),
            "initial_Pot2_withdrawal_required": residual_pot2_finger_contact,
            "robot_contact_identities": robot_contact_rows,
            "native_predicates": native_status,
            "operational_safety_preflight": geometry,
            "active_safety_policy": backend.safety_policy,
            "physical_actions_before_preflight": 0,
        })
        if geometry.get("status") != "complete":
            raise RuntimeError(f"t649_current_state_fails_frozen_Operational_Safety:{geometry.get('status')}")
        if preflight.get("contact_monitoring") is not None and preflight[
            "contact_monitoring"
        ].get("status") != "complete":
            raise RuntimeError("t649_current_contact_monitoring_is_incomplete")
        live_values = _native_values(native_status)
        if (
            live_values.get(("on", "moka_pot_2", "flat_stove_1_cook_region")) is not True
            or live_values.get(("on", "moka_pot_1", "flat_stove_1_cook_region")) is not False
            or live_values.get(("turnon", "flat_stove_1")) is not True
            or native_status.get("task_success") is not False
        ):
            raise RuntimeError("t649_live_native_predicates_changed_or_are_unknown")
        pot2_live = next(
            row for key, row in preflight["snapshot_state"]["object_states"].items()
            if str(key).endswith(":moka_pot_2")
        )
        pot2_qvel = np.asarray(pot2_live.get("joint_qvel"), dtype=np.float64)
        if pot2_qvel.shape != (6,) or np.linalg.norm(pot2_qvel[:3]) > 0.05 or np.linalg.norm(
            pot2_qvel[3:]
        ) > 0.5:
            raise RuntimeError("t649_restored_Pot2_is_not_physically_stable")

        _write_json(output_dir / "strict_restore_preflight.json", {
            "status": "strict_t649_runtime_restore_and_native_stage_adoption_passed",
            "source_snapshot": str(SOURCE_SNAPSHOT),
            "source_snapshot_sha256": attestation["source_snapshot_sha256"],
            "model_geometry_sha256": MODEL_SHA256,
            "timestep": INITIAL_TIMESTEP,
            "integration_state_sha256": _array_sha256(restored_integration),
            "integration_state_exact_match": True,
            "runtime_snapshot_normalized_sha256": runtime_hash,
            "runtime_internal_state_exact_match": True,
            "t649_Pot2_source_settle_samples": 5,
            "restored_Pot2_linear_speed_m_s": float(np.linalg.norm(pot2_qvel[:3])),
            "restored_Pot2_angular_speed_rad_s": float(np.linalg.norm(pot2_qvel[3:])),
            "native_predicates": native_status,
            "task_phase": context.phase,
            "task_object": context.metadata.get("task_object"),
            "held_object": context.held_object,
            "phase_reconstructed_from_native_predicates_and_live_contact_identity": True,
            "initial_Pot2_withdrawal_required": residual_pot2_finger_contact,
            "initial_robot_contact_identities": robot_contact_rows,
            "live_grasp_records": grasp_records,
            "historical_task_phase_counters_adopted": False,
            "operational_safety_preflight": geometry,
            "operational_policy_thresholds_match_t649": True,
            "physical_actions_before_preflight": 0,
        })
        artifacts.event({
            "event": "t649_strict_restore_preflight_passed",
            "timestep": INITIAL_TIMESTEP,
            "task_phase": context.phase,
            "task_object": context.metadata.get("task_object"),
            "held_object": context.held_object,
            "native_task_status": native_status,
            "initial_Pot2_withdrawal_required": residual_pot2_finger_contact,
            "integration_state_sha256": _array_sha256(restored_integration),
            "runtime_snapshot_normalized_sha256": runtime_hash,
            "safety_policy": backend.safety_policy,
        })
        backend.latest_data = preflight
        backend.initialize_trajectory()
        backend._save_video_frame(preflight)

        engine = AutomaticSafetyRecoveryEngine(backend, recovery, artifacts=artifacts)
        engine_result = engine.run()
        result_json = _jsonable(asdict(engine_result))
        terminal = backend.observe()
        terminal_native = terminal.raw.get("native_task_status") or {}
        terminal_values = _native_values(terminal_native)
        safety = _safety_acceptance(backend.trajectory, backend.safety_policy)
        task_success = terminal.task_success is True and terminal_native.get("task_success") is True
        final_timestep = int(terminal.timestep)
        standard_success = bool(task_success and final_timestep <= STANDARD_HORIZON)
        demo_success = bool(task_success and final_timestep <= MAX_TASK_TIMESTEP)
        result_json.update({
            "status": "complete_task_success" if task_success else "stopped_without_complete_task_success",
            "initial_timestep": INITIAL_TIMESTEP,
            "final_timestep": final_timestep,
            "physical_actions_committed": backend.transition_count,
            "maximum_total_episode_timestep": MAX_TASK_TIMESTEP,
            "native_task_status": terminal_native,
            "native_task_success": task_success,
            "reward": terminal.reward,
            "done": terminal.done,
            "pot1_on": terminal_values.get(("on", "moka_pot_1", "flat_stove_1_cook_region")),
            "pot2_on": terminal_values.get(("on", "moka_pot_2", "flat_stove_1_cook_region")),
            "stove_turnon": terminal_values.get(("turnon", "flat_stove_1")),
            "standard_horizon_complete_task_success": standard_success,
            "demonstration_horizon_complete_task_success": demo_success,
            "suffix_native_safety": safety["native_safety"],
            "suffix_operational_safety": safety["operational_safety"],
            "suffix_strict_10mm_supplementary": safety["strict_10mm_supplementary"],
            "pot2_native_on_preserved_each_action": safety["pot2_native_on_preserved_each_action"],
            "safety_acceptance": safety,
            "verified_safe_success_suffix": bool(
                task_success
                and safety["native_safety"] is True
                and safety["operational_safety"] is True
                and safety["pot2_native_on_preserved_each_action"] is True
            ),
            "complete_round0_seed": False,
            "t10_to_t249_early_prefix_still_requires_unified_continuous_reaudit": True,
            "lora_training_started": False,
            "planner_search_scope": "only_existing_RRT_for_unplanned_Pot1_recovery_and_transport_phases",
            "rrt_calls": backend.plan_attempt,
            "failure_phase": engine_result.phase if not task_success else None,
            "failure_reason": engine_result.reason if not task_success else None,
        })
        _write_json(output_dir / "suffix_result.json", result_json)
        _write_json(output_dir / "posthoc_acceptance_audit.json", {
            "schema": "gs_opd_t649_pot1_task_suffix_audit.v1",
            "source_snapshot": str(SOURCE_SNAPSHOT),
            "source_snapshot_sha256": attestation["source_snapshot_sha256"],
            "source_model_geometry_sha256": MODEL_SHA256,
            "source_integration_state_sha256": source["integration_state_sha256"],
            "terminal_timestep": final_timestep,
            "native_task_status": terminal_native,
            "task_success": task_success,
            "reward": terminal.reward,
            "done": terminal.done,
            "standard_horizon_complete_task_success": standard_success,
            "demonstration_horizon_complete_task_success": demo_success,
            "native_safety": safety["native_safety"],
            "operational_safety": safety["operational_safety"],
            "strict_10mm_supplementary": safety["strict_10mm_supplementary"],
            "physical_actions": backend.transition_count,
            "physical_substeps": safety["observed_substeps"],
            "pot2_native_on_preserved_each_action": safety["pot2_native_on_preserved_each_action"],
            "safety_acceptance": safety,
            "verified_safe_success_suffix": result_json["verified_safe_success_suffix"],
            "complete_round0_seed": False,
        })
        return result_json
    except BaseException as exc:
        failure_timestep = (
            int(backend.latest_data.get("timestep", -1))
            if backend is not None and backend.latest_data else None
        )
        failure_phase = getattr(engine, "_phase", None) if engine else None
        result_json = {
            "status": "interrupted_or_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_timestep": failure_timestep,
            "failure_phase": failure_phase,
            "physical_transitions_recorded": backend.transition_count if backend else 0,
            "task_success": False,
            "complete_round0_seed": False,
            "strict_restore_was_attested": bool(backend is not None and backend.latest_data),
            "output_dir": str(output_dir),
        }
        _write_json(output_dir / "failure_report.json", result_json)
        return result_json
    finally:
        if backend is not None:
            try:
                backend.finalize(result_json or {"complete_task_success": False})
            except Exception as finalize_exc:
                (output_dir / "finalize_error.txt").write_text(
                    f"{type(finalize_exc).__name__}: {finalize_exc}\n", encoding="utf-8"
                )
                try:
                    backend.trajectory.save(
                        steps=backend.transition_count,
                        chunks=backend.transition_count,
                    )
                except Exception as save_exc:
                    (output_dir / "trajectory_save_error.txt").write_text(
                        f"{type(save_exc).__name__}: {save_exc}\n", encoding="utf-8"
                    )
            for writer in backend._video_writers.values():
                writer.release()
        bridge.close()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        if result_json is not None:
            result_json["output_dir"] = str(output_dir)
            result_json["elapsed_seconds"] = time.monotonic() - started
            _write_json(output_dir / "suffix_result.json", result_json)


def main() -> int:
    args = _parser()
    result = run(args)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "complete_task_success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
