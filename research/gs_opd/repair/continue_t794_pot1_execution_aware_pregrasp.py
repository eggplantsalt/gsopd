# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Resume the accepted t738 Pot1 route at a strictly restored live Snapshot.

This independent executor treats the archived 7D RRT path as a diagnostic
reference. Each OSC_POSE action is generated and certified in an isolated
MuJoCo branch from the exact current Runtime Snapshot before that one action
is committed. It never invokes RRT or the legacy qpos-gated recovery engine.
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

from research.gs_opd.repair.automatic_recovery import RecoveryArtifactWriter
from research.gs_opd.repair.automatic_recovery_runtime import (
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
    sha256_file,
)
from research.gs_opd.repair.execute_t463_execution_aware_pot2_placement import (
    _contacts_signature,
)
from research.gs_opd.repair.run_t329_native_task_suffix import (
    DurableTaskSuffixBackend,
    _atomic_npz,
    _canonical_sha256,
    _jsonable,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "research/gs_opd/configs/l0t2_task_success_first_safety_reconstruction.json"
LINEAGE_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_t649_pot1_task_suffix_20261025_005902"
)
SOURCE_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_t795_pot1_execution_aware_pregrasp_20261025_021653"
)
SOURCE_AUDIT = LINEAGE_ARCHIVE / "failure_diagnosis.json"
SOURCE_SNAPSHOT = SOURCE_ARCHIVE / "snapshots/main_timestep_000796.pkl"
SOURCE_TRACE = SOURCE_ARCHIVE / "observation_action_trace.npz"
SOURCE_EVENTS = LINEAGE_ARCHIVE / "recovery_events.jsonl"
MODEL_SHA256 = "1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752"
INITIAL_TIMESTEP = 796
ROUTE_START_TIMESTEP = 738
ROUTE_PLAN_INDEX = 3
MAX_DEMONSTRATION_HORIZON = 1800
STANDARD_HORIZON = 1000
EXPECTED_SUBSTEPS = 25
RISK_TRIGGER_M = 0.050
RISK_REARM_M = 0.060
POT2_ON = ("on", "moka_pot_2", "flat_stove_1_cook_region")
POT1_ON = ("on", "moka_pot_1", "flat_stove_1_cook_region")
STOVE_ON = ("turnon", "flat_stove_1")


class _PythonObjectState:
    """Snapshot pickle helper class emitted by the Safety Bridge as __main__."""

    def __init__(self, module=None, qualname=None, attributes=None):
        self.module = module
        self.qualname = qualname
        self.attributes = attributes or {}


setattr(__main__, "_PythonObjectState", _PythonObjectState)


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _normalized_snapshot_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "ndarray", "dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _normalized_snapshot_value(value.item())
    if isinstance(value, _PythonObjectState) or (
        value.__class__.__name__ == "_PythonObjectState" and hasattr(value, "attributes")
    ):
        return {
            "kind": "python_object_state", "module": getattr(value, "module", None),
            "qualname": getattr(value, "qualname", None),
            "attributes": _normalized_snapshot_value(getattr(value, "attributes", {})),
        }
    if isinstance(value, dict):
        return {
            str(key): _normalized_snapshot_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalized_snapshot_value(item) for item in value]
    if isinstance(value, bytes):
        return {"kind": "bytes", "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, float):
        return {"kind": "float", "hex": value.hex()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return {"kind": "repr", "value": repr(value)}


def _normalized_state_sha256(value: Any) -> str:
    packed = json.dumps(
        _normalized_snapshot_value(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def _native_values(status: dict[str, Any]) -> dict[tuple[str, ...], bool]:
    if status.get("status") != "complete":
        raise RuntimeError(f"native_task_status_incomplete:{status.get('error')}")
    return {
        tuple(str(token).lower() for token in row.get("predicate", [])): bool(row["value"])
        for row in status.get("native_goal_values", [])
        if isinstance(row, dict) and isinstance(row.get("value"), bool)
    }


def _quat_error_rad(left: Any, right: Any) -> float:
    a = np.array(left, dtype=np.float64, copy=True)
    b = np.array(right, dtype=np.float64, copy=True)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("route_quaternion_comparison_requires_WXYZ_quaternions")
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(2.0 * math.acos(float(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def _ordered_lineage_trace_paths() -> list[Path]:
    """Collect exact Snapshot-linked OA segments back to the t794 root archive."""
    paths_reversed: list[Path] = []
    node = SOURCE_ARCHIVE
    visited = set()
    while True:
        node = node.resolve()
        if node in visited:
            raise ValueError("runtime_snapshot_lineage_contains_a_cycle")
        visited.add(node)
        trace_path = node / "observation_action_trace.npz"
        if not trace_path.is_file():
            raise FileNotFoundError(f"lineage_OA_trace_missing:{trace_path}")
        paths_reversed.append(trace_path)
        if node == LINEAGE_ARCHIVE.resolve():
            break
        config_path = node / "run_config.json"
        if not config_path.is_file():
            raise ValueError(f"lineage_run_config_missing:{config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        snapshot_path = Path(
            config.get("source_snapshot_attestation", {}).get("source_snapshot_path", "")
        )
        if not snapshot_path.is_file():
            raise ValueError(f"lineage_parent_runtime_Snapshot_missing:{snapshot_path}")
        parent_archive = snapshot_path.parent.parent.resolve()
        if parent_archive == LINEAGE_ARCHIVE.resolve():
            paths_reversed.append(parent_archive / "observation_action_trace.npz")
            break
        node = parent_archive
    ordered = list(reversed(paths_reversed))
    if ordered[0].parent.resolve() != LINEAGE_ARCHIVE.resolve():
        raise ValueError("lineage_OA_does_not_terminate_at_the_attested_t794_archive")
    return ordered


def _snapshot_attestation() -> tuple[dict[str, Any], dict[str, Any]]:
    audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
    resume_failure = json.loads((SOURCE_ARCHIVE / "failure_report.json").read_text(encoding="utf-8"))
    prior_config = json.loads((SOURCE_ARCHIVE / "run_config.json").read_text(encoding="utf-8"))
    strict_restore = json.loads((SOURCE_ARCHIVE / "strict_restore_preflight.json").read_text(encoding="utf-8"))
    first_preview = json.loads((SOURCE_ARCHIVE / "first_preview_preflight.json").read_text(encoding="utf-8"))
    parent_attestation = prior_config.get("source_snapshot_attestation", {})
    parent_snapshot_path = Path(parent_attestation.get("source_snapshot_path", ""))
    if (
        resume_failure.get("failure_timestep") != INITIAL_TIMESTEP
        or resume_failure.get("physical_actions_committed") != 1
        or resume_failure.get("first_preview_passed") is not True
        or not str(strict_restore.get("status", "")).startswith("strict_t")
        or "restore_and_live_stage_preflight_passed" not in str(strict_restore.get("status", ""))
        or prior_config.get("source_snapshot_attestation", {}).get("timestep") != INITIAL_TIMESTEP - 1
        or first_preview.get("source_timestep") != INITIAL_TIMESTEP - 1
        or first_preview.get("path_index") != INITIAL_TIMESTEP - ROUTE_START_TIMESTEP
        or not parent_snapshot_path.is_file()
        or sha256_file(parent_snapshot_path) != parent_attestation.get("source_snapshot_sha256")
        or strict_restore.get("source_snapshot_sha256") != parent_attestation.get("source_snapshot_sha256")
    ):
        raise ValueError("resume_lineage_is_not_the_attested_single_action_snapshot_continuation")
    snapshot_sha = sha256_file(SOURCE_SNAPSHOT)
    journal_index = SOURCE_ARCHIVE / "durable_action_journal/index.jsonl"
    journal_rows = [json.loads(line) for line in journal_index.read_text(encoding="utf-8").splitlines()]
    if (
        len(journal_rows) != 1
        or int(journal_rows[-1].get("timestep", -1)) != INITIAL_TIMESTEP
        or Path(journal_rows[-1].get("runtime_snapshot_path", "")) != SOURCE_SNAPSHOT
        or sha256_file(Path(journal_rows[-1]["journal_path"])) != journal_rows[-1].get("journal_sha256")
    ):
        raise ValueError("t795_runtime_Snapshot_and_durable_action_journal_do_not_match")
    envelope = pickle.loads(SOURCE_SNAPSHOT.read_bytes())
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
    ):
        raise ValueError("t795_runtime_Snapshot_schema_or_model_fingerprint_mismatch")
    snapshot = envelope.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("t795_runtime_Snapshot_payload_missing")
    integration = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    if integration.shape != (317,) or not np.isfinite(integration).all():
        raise ValueError("t795_Integration_State_shape_or_values_invalid")
    if int(snapshot.get("task_bookkeeping", {}).get("timestep", -1)) != INITIAL_TIMESTEP:
        raise ValueError("t795_Snapshot_internal_timestep_mismatch")
    robots = snapshot.get("robots", [])
    if len(robots) != 1 or not robots[0].get("has_gripper_current_action"):
        raise ValueError("t795_native_robot_or_gripper_runtime_state_missing")

    with np.load(SOURCE_TRACE, allow_pickle=False) as trace:
        if (
            trace["executed_actions"].shape != (1, 7)
            or not np.array_equal(trace["step_timesteps"], np.asarray([INITIAL_TIMESTEP], dtype=np.int64))
        ):
            raise ValueError("t794_to_t795_action_trace_is_not_a_single_contiguous_action")
        terminal_physical = json.loads(str(trace["post_physical_state_json"][-1]))
        terminal_native = json.loads(str(trace["post_native_task_status_json"][-1]))
        terminal_monitor = json.loads(str(trace["post_contact_monitoring_json"][-1]))
        if not np.array_equal(
            np.asarray(trace["executed_actions"][0], dtype=np.float32),
            np.asarray(first_preview["action"], dtype=np.float32),
        ):
            raise ValueError("t795_committed_action_differs_from_t794_safe_preview")
        parent_trace_path = parent_snapshot_path.parent.parent / "observation_action_trace.npz"
        with np.load(parent_trace_path, allow_pickle=False) as parent_trace:
            if int(parent_trace["step_timesteps"][-1]) != INITIAL_TIMESTEP - 1:
                raise ValueError("parent_OA_trace_does_not_end_at_source_runtime_Snapshot")
            if not np.array_equal(
                np.asarray(trace["decision_robot0_eef_pos"][0], dtype=np.float64),
                np.asarray(parent_trace["post_robot0_eef_pos"][-1], dtype=np.float64),
            ) or not np.array_equal(
                np.asarray(trace["decision_robot0_eef_quat"][0], dtype=np.float64),
                np.asarray(parent_trace["post_robot0_eef_quat"][-1], dtype=np.float64),
            ):
                raise ValueError("parent_to_resume_OA_boundary_observation_is_not_exactly_contiguous")
        if (
            terminal_monitor.get("status") != "complete"
            or terminal_monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
            or terminal_monitor.get("geometry_monitoring", {}).get("status") != "complete"
            or len(terminal_monitor.get("geometry_monitoring", {}).get("substeps", [])) != EXPECTED_SUBSTEPS
        ):
            raise ValueError("t795_parent_action_does_not_have_full_safe_25_substep_evidence")
        recorded_integration = np.asarray(
            terminal_physical.get("mujoco_integration_state"), dtype=np.float64
        )
        if recorded_integration.shape != integration.shape or not np.array_equal(
            recorded_integration, integration
        ):
            raise ValueError("t794_Snapshot_Integration_State_differs_from_terminal_OA_record")
    predicates = _native_values(terminal_native)
    expected = {POT1_ON: False, POT2_ON: True, STOVE_ON: True}
    if predicates != expected or terminal_native.get("task_success") is not False:
        raise ValueError("t794_terminal_OA_native_predicates_do_not_match_audited_stage")
    if terminal_native.get("done") is not False:
        raise ValueError("t794_terminal_OA_is_unexpectedly_done")

    physical_robot = terminal_physical["robot_states"][0]
    pot1 = terminal_physical["object_states"]["objects_dict:moka_pot_1"]
    pot2 = terminal_physical["object_states"]["objects_dict:moka_pot_2"]
    attestation = {
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(SOURCE_SNAPSHOT),
        "source_snapshot_sha256": snapshot_sha,
        "source_action_record": str(SOURCE_TRACE),
        "source_action_record_sha256": sha256_file(SOURCE_TRACE),
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
        "integration_state_sha256": _array_sha256(integration),
        "runtime_snapshot_normalized_sha256": _normalized_state_sha256(snapshot),
        "source_pot2_position_world_m": pot2["root_pos"],
        "parent_runtime_snapshot_path": prior_config["source_snapshot_attestation"]["source_snapshot_path"],
        "parent_runtime_snapshot_sha256": prior_config["source_snapshot_attestation"]["source_snapshot_sha256"],
        "lineage_archive": str(LINEAGE_ARCHIVE),
        "single_action_parent_boundary_verified": True,
    }
    previous_summary = json.loads((LINEAGE_ARCHIVE / "run_summary.json").read_text(encoding="utf-8"))
    source = {
        "failure_diagnosis": audit,
        "snapshot": snapshot,
        "integration_state": integration,
        "integration_state_sha256": attestation["integration_state_sha256"],
        "runtime_snapshot_normalized_sha256": attestation[
            "runtime_snapshot_normalized_sha256"
        ],
        "native_status": terminal_native,
        "physical": terminal_physical,
        "final_monitor": terminal_monitor,
        "policy": previous_summary["safety_policy"],
    }
    return attestation, source


def _load_last_accepted_route() -> dict[str, Any]:
    selected = None
    risk_events = []
    for line in SOURCE_EVENTS.open("r", encoding="utf-8"):
        event = json.loads(line)
        if event.get("event") == "planner_result_details":
            if (
                event.get("plan_index") == ROUTE_PLAN_INDEX
                and event.get("task_phase") == "second_object_recovery"
            ):
                selected = event
        if event.get("event") in {"risk_boundary_entered", "automatic_replan_triggered"}:
            risk_events.append({
                key: event.get(key) for key in (
                    "event", "phase", "reason", "timestep", "minimum_clearance_m",
                    "minimum_forbidden_clearance_m", "geometry_pair", "robot_geom",
                    "obstacle_geom", "path_cursor",
                ) if key in event
            })
    if selected is None:
        raise ValueError("accepted_t738_Pot1_route_event_not_found")
    planner = selected.get("planner_result", {})
    qpos = np.asarray(planner.get("qpos_path"), dtype=np.float64)
    eef = np.asarray(planner.get("eef_reference_poses_world"), dtype=np.float64)
    if (
        planner.get("status") != "offline_feasible"
        or planner.get("live_state_unchanged") is not True
        or planner.get("model_geometry_sha256") != MODEL_SHA256
        or planner.get("safety_policy_source")
        != "active_BDDL_CheckRobotContact_CheckContact_plus_registered_full_collision_geometry"
        or qpos.shape != (282, 7)
        or eef.shape != (282, 7)
        or not np.isfinite(qpos).all()
        or not np.isfinite(eef).all()
        or np.any(np.abs(np.linalg.norm(eef[:, 3:], axis=1) - 1.0) > 1e-5)
    ):
        raise ValueError("accepted_t738_route_status_or_path_payload_is_invalid")
    return {
        "event": selected,
        "planner_result": planner,
        "qpos_path": qpos,
        "eef_path": eef,
        "events_sha256": sha256_file(SOURCE_EVENTS),
        "qpos_path_sha256": _array_sha256(qpos),
        "eef_reference_path_sha256": _array_sha256(eef),
        "risk_events": risk_events,
    }


def _check_substeps(
    substeps: list[dict[str, Any]], safety_policy: dict[str, Any]
) -> dict[str, Any]:
    if len(substeps) != EXPECTED_SUBSTEPS:
        raise RuntimeError(f"expected_25_physical_substeps_got_{len(substeps)}")
    failures = []
    minima: dict[str, float] = {}
    required_clearances = {
        str(row["object_id"]): float(row["minimum_clearance_m"])
        for group_name in ("forbidden_objects", "allowed_target_objects", "other_scene_objects")
        for row in safety_policy.get(group_name, [])
    }
    for index, row in enumerate(substeps):
        if row.get("status") != "complete":
            failures.append({"substep": index, "reason": "geometry_status", "value": row.get("status")})
        for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
        ):
            if row.get(key):
                failures.append({"substep": index, "reason": key, "count": len(row[key])})
        observed_clearances = row.get("clearances_by_object", {})
        for object_id, required in required_clearances.items():
            record = observed_clearances.get(object_id)
            value = record.get("minimum_distance_m") if isinstance(record, dict) else None
            if value is None:
                failures.append({
                    "substep": index, "reason": "operational_clearance_missing",
                    "object_id": object_id,
                })
            elif float(value) < required:
                failures.append({
                    "substep": index, "reason": "operational_clearance_below_frozen_policy",
                    "object_id": object_id, "distance_m": float(value),
                    "required_m": required,
                })
        for name, record in observed_clearances.items():
            value = record.get("minimum_distance_m") if isinstance(record, dict) else record
            if value is not None:
                minima[str(name)] = min(float(value), minima.get(str(name), float("inf")))
        for name, record in row.get("carried_clearances_by_object", {}).items():
            value = record.get("minimum_distance_m") if isinstance(record, dict) else record
            if value is not None:
                key = "carried:" + str(name)
                minima[key] = min(float(value), minima.get(key, float("inf")))
    if failures:
        raise RuntimeError("physical_branch_substep_safety_failed:" + json.dumps(failures[:12]))
    return {"substeps": len(substeps), "minimum_clearance_by_object_m": minima, "failures": []}


def _validate_native_stage(status: dict[str, Any], *, context: str) -> dict[str, bool]:
    values = _native_values(status)
    expected = {POT1_ON: False, POT2_ON: True, STOVE_ON: True}
    if any(values.get(key) is not value for key, value in expected.items()):
        raise RuntimeError(f"{context}_native_predicate_stage_changed:{values}")
    if status.get("task_success") is not False or status.get("done") is True:
        raise RuntimeError(f"{context}_native_task_terminated_or_success_state_changed")
    return {"pot1_on": values[POT1_ON], "pot2_on": values[POT2_ON], "stove_turnon": values[STOVE_ON]}


def _inspect_predicted_state(
    bridge: BridgeProcess,
    backend: DurableTaskSuffixBackend,
    source_snapshot_id: str,
    source_integration: np.ndarray,
    payload: bytes,
    safety_policy: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError("preview_terminal_runtime_Snapshot_missing")
    envelope = pickle.loads(payload)
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
        or not isinstance(envelope.get("snapshot"), dict)
    ):
        raise RuntimeError("preview_terminal_runtime_Snapshot_fingerprint_invalid")
    expected_state = envelope["snapshot"]
    expected_integration = np.asarray(expected_state.get("mujoco_state"), dtype=np.float64)
    predicted_status: dict[str, Any] = {}
    predicted_contacts: dict[str, Any] = {}
    predicted_geometry: dict[str, Any] = {}
    predicted_native: dict[str, bool] = {}
    try:
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
        if restored.get("status") != "restored":
            raise RuntimeError(f"predicted_branch_Snapshot_restore_failed:{restored.get('error')}")
        predicted_status = bridge.request({"op": "native_task_status"})
        predicted_native = _validate_native_stage(predicted_status, context="predicted_branch")
        predicted_contacts = bridge.request({"op": "inspect_contacts"})
        held, _verified, grasp_records = backend._grasp_states()
        if held is not None:
            raise RuntimeError(f"unexpected_Pot1_grasp_before_pregrasp:{held}")
        backend._refresh_task_contact_policy("second_object_recovery", "moka_pot_1", None)
        observed = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": safety_policy,
        })
        predicted_geometry = observed.get("safety_geometry", {})
        if predicted_geometry.get("status") != "complete":
            raise RuntimeError("predicted_terminal_Operational_geometry_is_not_complete")
        for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
        ):
            if predicted_geometry.get(key):
                raise RuntimeError(f"predicted_terminal_safety_field_not_empty:{key}")
        live_integration = np.asarray(
            observed.get("snapshot_state", {}).get("mujoco_integration_state"),
            dtype=np.float64,
        )
        if live_integration.shape != expected_integration.shape or not np.array_equal(
            live_integration, expected_integration
        ):
            raise RuntimeError("predicted_runtime_Snapshot_and_restored_Integration_State_differ")
        robot_contacts = []
        for contact in predicted_contacts.get("contacts", []):
            b1, b2 = contact.get("body1", {}), contact.get("body2", {})
            if str(b1.get("name", "")).startswith(("robot0_", "gripper0_")) or str(
                b2.get("name", "")
            ).startswith(("robot0_", "gripper0_")):
                robot_contacts.append(contact)
        if robot_contacts:
            raise RuntimeError("predicted_pregrasp_branch_has_robot_scene_contact")
    finally:
        restored_source = bridge.request({"op": "restore", "snapshot_id": source_snapshot_id})
        if restored_source.get("restored_snapshot_id") != source_snapshot_id:
            raise RuntimeError(f"source_t794_Snapshot_restore_after_preview_audit_failed:{restored_source.get('error')}")
        source_observation = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": safety_policy,
        })
        restored_integration = np.asarray(
            source_observation.get("snapshot_state", {}).get("mujoco_integration_state"),
            dtype=np.float64,
        )
        if restored_integration.shape != source_integration.shape or not np.array_equal(
            restored_integration, source_integration
        ):
            raise RuntimeError("t794_Integration_State_changed_during_predicted_state_audit")
        backend.latest_data = source_observation
        backend.latest_data["native_task_status"] = bridge.request({"op": "native_task_status"})
        backend.latest_data["contacts"] = bridge.request({"op": "inspect_contacts"})
    return {
        "native_task_status": predicted_status,
        "native_predicates": predicted_native,
        "predicted_held_object": None,
        "predicted_grasp_state": grasp_records.get("moka_pot_1", {}),
        "contact_geom_pairs": _contacts_signature(predicted_contacts),
        "predicted_terminal_safety_geometry": predicted_geometry,
        "predicted_integration_state_sha256": _array_sha256(expected_integration),
        "source_restore_exact": True,
    }


def _parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38752)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    attestation, source = _snapshot_attestation()
    route = _load_last_accepted_route()
    frozen_policy = source["policy"]
    lineage_trace_paths = _ordered_lineage_trace_paths()
    trace_segments = []
    for path in lineage_trace_paths:
        with np.load(path, allow_pickle=False) as segment:
            trace_segments.append({
                key: np.asarray(segment[key]).copy()
                for key in (
                    "step_timesteps", "post_physical_state_json",
                    "post_contact_monitoring_json", "post_native_task_status_json",
                )
            })
    trace = {
        key: np.concatenate([segment[key] for segment in trace_segments], axis=0)
        for key in trace_segments[0]
    }
    timestamps = np.asarray(trace["step_timesteps"], dtype=np.int64)
    if not np.array_equal(timestamps, np.arange(timestamps[0], INITIAL_TIMESTEP + 1, dtype=np.int64)):
        raise ValueError("Snapshot_linked_OA_segments_are_not_one_contiguous_physical_timeline")
    timestep_to_row = {int(t): i for i, t in enumerate(timestamps)}
    if ROUTE_START_TIMESTEP not in timestep_to_row or INITIAL_TIMESTEP not in timestep_to_row:
        raise ValueError("t738_or_t794_is_missing_from_source_action_trace")
    start_row = timestep_to_row[ROUTE_START_TIMESTEP]
    start_state = json.loads(str(trace["post_physical_state_json"][start_row]))
    route_start_integration = np.asarray(start_state["mujoco_integration_state"], dtype=np.float64)
    planner = route["planner_result"]
    route_index = INITIAL_TIMESTEP - ROUTE_START_TIMESTEP
    if (
        route_index != INITIAL_TIMESTEP - ROUTE_START_TIMESTEP
        or _array_sha256(route_start_integration) != planner.get("start_state_sha256")
        or not np.array_equal(
            np.asarray(start_state["robot_states"][0]["joint_qpos"], dtype=np.float64),
            route["qpos_path"][0],
        )
        or not np.array_equal(
            np.asarray(start_state["robot_states"][0]["eef_pos"], dtype=np.float64),
            route["eef_path"][0, :3],
        )
    ):
        raise ValueError("t738_route_start_does_not_match_archived_actual_state_fingerprint")
    if route_index >= len(route["qpos_path"]) - 1:
        raise ValueError("t794_route_cursor_is_at_or_beyond_archived_route_end")

    prefix_q_errors = []
    prefix_eef_errors = []
    prefix_rot_errors = []
    prefix_monitors = []
    physical_states = [json.loads(str(row)) for row in trace["post_physical_state_json"]]
    for timestep in range(ROUTE_START_TIMESTEP + 1, INITIAL_TIMESTEP + 1):
        row = timestep_to_row[timestep]
        route_i = timestep - ROUTE_START_TIMESTEP
        physical = physical_states[row]
        robot = physical["robot_states"][0]
        prefix_q_errors.append(float(np.linalg.norm(
            np.asarray(robot["joint_qpos"], dtype=np.float64) - route["qpos_path"][route_i]
        )))
        prefix_eef_errors.append(float(np.linalg.norm(
            np.asarray(robot["eef_pos"], dtype=np.float64) - route["eef_path"][route_i, :3]
        )))
        prefix_rot_errors.append(_quat_error_rad(
            robot["eef_quat"], route["eef_path"][route_i, 3:]
        ))
        monitor = json.loads(str(trace["post_contact_monitoring_json"][row]))
        substeps = monitor.get("geometry_monitoring", {}).get("substeps", [])
        if (
            monitor.get("status") != "complete"
            or monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
            or monitor.get("geometry_monitoring", {}).get("status") != "complete"
        ):
            raise ValueError(f"t{timestep}_archived_route_prefix_monitoring_incomplete")
        # The archived prefix remains governed by the exact t794 operational
        # policy, including the frozen Pot2 2 mm object-specific floor.
        _check_substeps(substeps, frozen_policy)
        native = json.loads(str(trace["post_native_task_status_json"][row]))
        _validate_native_stage(native, context=f"archived_t{timestep}")
        prefix_monitors.append(monitor)

    qpos_now = np.asarray(source["physical"]["robot_states"][0]["joint_qpos"], dtype=np.float64)
    eef_now = np.asarray(source["physical"]["robot_states"][0]["eef_pos"], dtype=np.float64)
    quat_now = np.asarray(source["physical"]["robot_states"][0]["eef_quat"], dtype=np.float64)
    route_start_errors = {
        "path_index": route_index,
        "qpos_reference_l2_rad_diagnostic_only": float(np.linalg.norm(qpos_now - route["qpos_path"][route_index])),
        "eef_position_error_m": float(np.linalg.norm(eef_now - route["eef_path"][route_index, :3])),
        "eef_orientation_error_rad": _quat_error_rad(quat_now, route["eef_path"][route_index, 3:]),
        "previous_route_prefix_qpos_error_max_rad": max(prefix_q_errors),
        "previous_route_prefix_eef_error_max_m": max(prefix_eef_errors),
        "previous_route_prefix_orientation_error_max_rad": max(prefix_rot_errors),
    }
    layout_deltas = {}
    for object_id in (
        "moka_pot_1", "moka_pot_2", "bottle_of_ginger_beer_1", "white_place_box_1"
    ):
        suffix = ":" + object_id
        start_objects = {
            key: value for key, value in start_state.get("object_states", {}).items()
            if str(key).endswith(suffix)
        }
        end_objects = {
            key: value for key, value in source["physical"].get("object_states", {}).items()
            if str(key).endswith(suffix)
        }
        if len(start_objects) == 1 and len(end_objects) == 1:
            start_object = next(iter(start_objects.values()))
            end_object = next(iter(end_objects.values()))
            layout_deltas[object_id] = {
                "position_delta_m": float(np.linalg.norm(
                    np.asarray(end_object["root_pos"], dtype=np.float64)
                    - np.asarray(start_object["root_pos"], dtype=np.float64)
                )),
                "orientation_delta_rad": _quat_error_rad(
                    end_object["root_quat"], start_object["root_quat"]
                ),
            }

    config_data, recovery = load_runtime_config(args.config)
    config_data = json.loads(json.dumps(config_data))
    config_data["candidate_search"]["enabled"] = False
    target_clearances = {
        str(item["object_id"]): float(item["minimum_clearance_m"])
        for item in frozen_policy["allowed_target_objects"]
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
        f"/opt/data/private/gs_opd/rollouts/l0t796_pot1_execution_aware_pregrasp_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    route_meta = {
        "source_archive": str(SOURCE_ARCHIVE),
        "source_events": str(SOURCE_EVENTS),
        "source_events_sha256": route["events_sha256"],
        "plan_index": ROUTE_PLAN_INDEX,
        "phase": "second_object_recovery",
        "status": planner["status"],
        "planner_source": planner.get("planner_source"),
        "planner_search_repeated": False,
        "original_planning_seconds": planner.get("planning_seconds"),
        "configuration_checks": planner.get("config_checks"),
        "minimum_offline_forbidden_clearance_m": planner.get("minimum_forbidden_clearance_m"),
        "start_state_sha256": planner.get("start_state_sha256"),
        "model_geometry_sha256": planner.get("model_geometry_sha256"),
        "safety_policy_source": planner.get("safety_policy_source"),
        "qpos_path_sha256": route["qpos_path_sha256"],
        "eef_reference_path_sha256": route["eef_reference_path_sha256"],
        "qpos_points": int(len(route["qpos_path"])),
        "eef_reference_points": int(len(route["eef_path"])),
        "resume_timestep": INITIAL_TIMESTEP,
        "resume_path_index": route_index,
        "verified_physical_route_prefix_actions": len(prefix_q_errors),
        "verified_physical_route_prefix_substeps": len(prefix_q_errors) * EXPECTED_SUBSTEPS,
        "resume_errors": route_start_errors,
        "live_layout_delta_t738_to_t794": layout_deltas,
    }
    _atomic_npz(output_dir / "reused_t738_route.npz", {
        "qpos_path": route["qpos_path"],
        "eef_reference_poses_world": route["eef_path"],
        "metadata_json": np.asarray(json.dumps(route_meta, sort_keys=True)),
    })
    _write_json(output_dir / "run_config.json", {
        "entrypoint": str(Path(__file__).resolve()),
        "mode": f"t{INITIAL_TIMESTEP}_strict_runtime_snapshot_plus_archived_t738_route",
        "source_snapshot_attestation": attestation,
        "source_route": route_meta,
        "prior_planner_history": source["failure_diagnosis"].get("planner"),
        "frozen_operational_safety_policy": frozen_policy,
        "frozen_operational_safety_policy_sha256": _canonical_sha256(frozen_policy),
        "legacy_repeated_risk_replan_events_retained": route["risk_events"],
        "legacy_same_reason_retry_guard_reset": False,
        "risk_50mm_is_diagnostic_only": True,
        "risk_trigger_m": RISK_TRIGGER_M,
        "risk_rearm_m": RISK_REARM_M,
        "qpos_reference_mode": "diagnostic_only",
        "qpos_reference_limit_rad": recovery.maximum_joint_path_deviation_rad,
        "eef_tracking_limits_unchanged": {
            "position_m": recovery.maximum_eef_tracking_error_m,
            "orientation_rad": recovery.maximum_eef_orientation_error_rad,
        },
        "per_action_physics_substeps": EXPECTED_SUBSTEPS,
        "maximum_total_task_timestep": MAX_DEMONSTRATION_HORIZON,
        "standard_benchmark_horizon": STANDARD_HORIZON,
        "new_RRT_calls_authorized": 0,
        "LoRA_training_started": False,
    })

    artifacts = RecoveryArtifactWriter(output_dir)
    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend: DurableTaskSuffixBackend | None = None
    result: dict[str, Any] = {
        "status": "preflight_not_completed",
        "output_dir": str(output_dir),
        "complete_round0_seed": False,
    }
    original_handlers = {}
    started = time.monotonic()

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
            raise RuntimeError("strict_t794_restore_timestep_mismatch")
        if restored.get("scene_metadata", {}).get("model_geometry_sha256") != MODEL_SHA256:
            raise RuntimeError("strict_t794_restore_model_geometry_fingerprint_mismatch")
        restored_integration = np.asarray(
            restored.get("snapshot_state", {}).get("mujoco_integration_state"),
            dtype=np.float64,
        )
        if not np.array_equal(restored_integration, source["integration_state"]):
            raise RuntimeError("strict_t794_restore_Integration_State_mismatch")
        exported = bridge.request({"op": "export_runtime_snapshot"})
        if exported.get("status") != "complete":
            raise RuntimeError("strict_t794_restore_runtime_controller_state_export_failed")
        restored_snapshot = pickle.loads(exported["payload"]).get("snapshot", {})
        runtime_hash = _normalized_state_sha256(restored_snapshot)
        if runtime_hash != source["runtime_snapshot_normalized_sha256"]:
            raise RuntimeError("strict_t794_restore_robot_controller_gripper_integrator_mismatch")

        backend.latest_data = restored
        backend._load_context_inputs(restored)
        task_config_objects = backend.task_controller.objects
        object_order = [str(row["object_id"]) for row in task_config_objects]
        if object_order != ["moka_pot_2", "moka_pot_1"]:
            raise RuntimeError(f"task_object_order_differs_from_native_stage_program:{object_order}")
        # Reconstruct only the live stage from current native predicates. All
        # historical phase counters/anchors are intentionally fresh.
        controller = backend.task_controller
        controller.object_index = 1
        controller.phase = "second_object_recovery"
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

        native_before = bridge.request({"op": "native_task_status"})
        native_values = _validate_native_stage(native_before, context="restored_t794")
        held_before, grasp_ok, grasp_records = backend._grasp_states()
        if held_before is not None:
            raise RuntimeError(f"t794_pregrasp_stage_unexpected_held_object:{held_before}")
        contacts_before = bridge.request({"op": "inspect_contacts"})
        robot_contact_rows = []
        for contact in contacts_before.get("contacts", []):
            b1, b2 = contact.get("body1", {}), contact.get("body2", {})
            if str(b1.get("name", "")).startswith(("robot0_", "gripper0_")) or str(
                b2.get("name", "")
            ).startswith(("robot0_", "gripper0_")):
                robot_contact_rows.append(contact)
        if robot_contact_rows:
            raise RuntimeError("t794_pregrasp_state_has_unexpected_robot_scene_contact")

        # Recompute current stage target from the live Pot1 object transform;
        # this also sets phase-specific finger-contact permissions (none here).
        context = backend.current_context()
        context.validate()
        if context.phase != "second_object_recovery" or context.held_object is not None:
            raise RuntimeError(f"t794_live_stage_reconstruction_mismatch:{context.phase}:{context.held_object}")
        target = context.goals[0]
        route_goal_position_error = float(np.linalg.norm(
            route["eef_path"][-1, :3] - np.asarray(target.position_world_m, dtype=np.float64)
        ))
        route_goal_orientation_error = _quat_error_rad(
            route["eef_path"][-1, 3:], target.quaternion_wxyz
        )
        if route_goal_position_error > 1e-5 or route_goal_orientation_error > 1e-5:
            raise RuntimeError("cached_t738_route_endpoint_does_not_match_live_Pot1_pregrasp_target")
        # Bind the runtime cursor only after the archived action prefix,
        # t738 start fingerprint, model hash, current EEF, and live target
        # endpoint have all been independently attested above.
        backend.plan_cursor = route_index
        backend.plan_counter = ROUTE_PLAN_INDEX
        backend._plan_poses = route["eef_path"].copy()
        if _canonical_sha256(backend.safety_policy) != _canonical_sha256(frozen_policy):
            raise RuntimeError("restored_active_safety_policy_differs_from_t794_archived_frozen_policy")

        preflight = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        preflight["native_task_status"] = native_before
        preflight["contacts"] = contacts_before
        backend.latest_data = preflight
        preflight_geometry = preflight.get("safety_geometry", {})
        if preflight_geometry.get("status") != "complete":
            raise RuntimeError("strict_t794_current_Operational_safety_preflight_failed")
        for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
        ):
            if preflight_geometry.get(key):
                raise RuntimeError(f"strict_t794_preflight_safety_field_not_empty:{key}")
        if not np.array_equal(
            np.asarray(preflight["snapshot_state"]["mujoco_integration_state"], dtype=np.float64),
            source["integration_state"],
        ):
            raise RuntimeError("t794_read_only_preflight_changed_Integration_State")
        pot2_state = next(
            row for key, row in preflight["snapshot_state"]["object_states"].items()
            if str(key).endswith(":moka_pot_2")
        )
        pot2_qvel = np.asarray(pot2_state.get("joint_qvel"), dtype=np.float64)
        if (
            pot2_qvel.shape != (6,)
            or np.linalg.norm(pot2_qvel[:3]) > 0.05
            or np.linalg.norm(pot2_qvel[3:]) > 0.5
        ):
            raise RuntimeError("strict_t794_Pot2_not_stable_under_frozen_settle_criteria")

        risk_margin = preflight_geometry.get("minimum_forbidden_clearance_m")
        if risk_margin is None or not np.isfinite(float(risk_margin)):
            raise RuntimeError("t794_forbidden_clearance_is_unknown")
        risk_armed = bool(float(risk_margin) >= RISK_REARM_M)
        restore_audit = {
            "status": "strict_t794_restore_and_live_stage_preflight_passed",
            "source_snapshot": str(SOURCE_SNAPSHOT),
            "source_snapshot_sha256": attestation["source_snapshot_sha256"],
            "timestep": INITIAL_TIMESTEP,
            "model_geometry_sha256": MODEL_SHA256,
            "integration_state_sha256": _array_sha256(restored_integration),
            "integration_state_exact_match": True,
            "runtime_snapshot_normalized_sha256": runtime_hash,
            "runtime_controller_gripper_integrator_state_exact_match": True,
            "native_predicates": native_before,
            "task_phase_reconstructed_from_live_native_predicates": context.phase,
            "task_object": context.metadata.get("task_object"),
            "historical_phase_counters_or_anchor_replayed": False,
            "pot2_linear_speed_m_s": float(np.linalg.norm(pot2_qvel[:3])),
            "pot2_angular_speed_rad_s": float(np.linalg.norm(pot2_qvel[3:])),
            "grasp_state": grasp_records,
            "robot_contacts": robot_contact_rows,
            "frozen_operational_policy_exact_match": True,
            "live_operational_geometry": preflight_geometry,
            "live_minimum_forbidden_clearance_m": float(risk_margin),
            "risk_band_state_initialized_inside_existing_hysteresis": not risk_armed,
            "historical_same_reason_risk_events_preserved": route["risk_events"],
            "risk_retry_history_reset": False,
            "pre_action_physical_actions": 0,
        }
        _write_json(output_dir / "strict_restore_preflight.json", restore_audit)
        artifacts.event({"event": "t794_strict_restore_preflight_passed", **restore_audit})
        backend.latest_data = preflight
        backend.initialize_trajectory()
        backend._save_video_frame(preflight)
        _write_json(output_dir / "route_resume_preflight.json", {
            "cached_route": route_meta,
            "current_live_goal": {
                "goal_id": target.goal_id,
                "phase": target.phase,
                "source": target.source,
                "position_world_m": list(target.position_world_m),
                "quaternion_wxyz": list(target.quaternion_wxyz),
                "endpoint_position_error_m": route_goal_position_error,
                "endpoint_orientation_error_rad": route_goal_orientation_error,
            },
            "current_tracking_diagnostics": route_start_errors,
            "qpos_reference_mode": "diagnostic_only",
            "risk_warning_is_not_an_automatic_failure_or_replan": True,
            "risk_monitor_initial_armed": risk_armed,
        })

        # This explicit single-action branch is the prerequisite for any
        # commitment. It runs at the actual restored t794 state and uses the
        # live Pot1 recovery path pose at cursor 57.
        initial_snapshot = bridge.request({"op": "snapshot"})
        if not isinstance(initial_snapshot.get("snapshot_id"), str):
            raise RuntimeError("t794_first_preview_source_Snapshot_unavailable")
        preview = bridge.request({
            "op": "osc_prefix_probe",
            "snapshot_id": initial_snapshot["snapshot_id"],
            "current_state_sha256": _array_sha256(source["integration_state"]),
            "jointspace_path_qpos": route["qpos_path"].tolist(),
            "eef_reference_poses_world": route["eef_path"].tolist(),
            "path_start_index": route_index,
            "max_control_steps": 1,
            "gripper_action": 0.0,
            "risk_trigger_distance_m": RISK_TRIGGER_M,
            "risk_rearm_distance_m": RISK_REARM_M,
            "risk_crossing_armed": risk_armed,
            "maximum_joint_path_deviation_rad": recovery.maximum_joint_path_deviation_rad,
            "maximum_eef_tracking_error_m": recovery.maximum_eef_tracking_error_m,
            "maximum_eef_orientation_error_rad": recovery.maximum_eef_orientation_error_rad,
            "qpos_reference_mode": "diagnostic_only",
            "current_minimum_forbidden_clearance_m": float(risk_margin),
            "safety_geometry_policy": backend.safety_policy,
        })
        if (
            preview.get("main_state_restored_verified") is not True
            or preview.get("complete") is not True
            or preview.get("safe") is not True
            or preview.get("qpos_reference_mode") != "diagnostic_only"
            or len(preview.get("actions", [])) != 1
            or len(preview.get("steps", [])) != 1
            or preview.get("stop_reason") not in {None, "narrow_channel_entry_replan"}
        ):
            raise RuntimeError(
                "t794_first_physical_preview_failed:" + json.dumps({
                    key: preview.get(key) for key in (
                        "status", "complete", "safe", "stop_reason", "failure_reason",
                        "main_state_restored_verified", "path_start_index",
                    )
                }, sort_keys=True)
            )
        first_step = preview["steps"][0]
        first_substep_audit = _check_substeps(
            first_step.get("safety_geometry_substeps", []), backend.safety_policy
        )
        if int(first_step.get("path_index", -1)) != route_index + 1:
            raise RuntimeError("t794_first_preview_did_not_follow_route_cursor")
        if (
            float(first_step.get("eef_tracking_error_m", math.inf)) > recovery.maximum_eef_tracking_error_m
            or float(first_step.get("eef_orientation_error_rad", math.inf)) > recovery.maximum_eef_orientation_error_rad
        ):
            raise RuntimeError("t794_first_preview_EEF_tracking_exceeded_frozen_limits")
        first_prediction = _inspect_predicted_state(
            bridge, backend, initial_snapshot["snapshot_id"], source["integration_state"],
            first_step.get("terminal_runtime_snapshot"), backend.safety_policy,
        )
        branch_path = output_dir / "branch_snapshots" / "preview_timestep_000795.pkl"
        branch_path.parent.mkdir(parents=True, exist_ok=True)
        branch_path.write_bytes(first_step["terminal_runtime_snapshot"])
        first_preview_audit = {
            "source_timestep": INITIAL_TIMESTEP,
            "path_index": int(first_step["path_index"]),
            "action": first_step["action"],
            "qpos_reference_l2_rad_diagnostic_only": first_step.get("joint_path_deviation_rad"),
            "eef_tracking_error_m": first_step.get("eef_tracking_error_m"),
            "eef_orientation_error_rad": first_step.get("eef_orientation_error_rad"),
            "branch_stop_reason_diagnostic": preview.get("stop_reason"),
            "contact_monitoring": preview.get("contact_monitoring"),
            "substep_audit": first_substep_audit,
            "predicted_state": first_prediction,
            "predicted_snapshot_path": str(branch_path),
            "predicted_snapshot_sha256": sha256_file(branch_path),
            "formally_executed": False,
        }
        _write_json(output_dir / "first_preview_preflight.json", first_preview_audit)
        if not np.array_equal(
            np.asarray(backend.latest_data["snapshot_state"]["mujoco_integration_state"], dtype=np.float64),
            source["integration_state"],
        ):
            raise RuntimeError("t794_source_state_not_restored_after_first_preview")

        # Preflight passed; commit the already-previewed first action, then
        # recertify each following action from its own freshly observed state.
        backend.task_expert_active = True
        certificate_stream = (output_dir / "rolling_preview_certificates.jsonl").open("a", encoding="utf-8")
        committed_rows = []
        risk_crossings = []
        motion_history: list[float] = []
        previous_eef = eef_now.copy()
        current_context = context
        next_preview = preview
        next_preview_snapshot_id = initial_snapshot["snapshot_id"]
        last_failure: str | None = None
        stop_reason = None
        try:
            while True:
                current_t = int(backend.latest_data.get("timestep", -1))
                if current_t >= MAX_DEMONSTRATION_HORIZON:
                    stop_reason = "demonstration_horizon_1800_reached"
                    break
                if current_context.phase == "approach":
                    stop_reason = "pot1_pregrasp_reached_and_approach_entered"
                    break
                if current_context.phase != "second_object_recovery":
                    stop_reason = f"unexpected_phase_transition:{current_context.phase}"
                    break
                if backend.plan_cursor >= len(route["qpos_path"]) - 1:
                    stop_reason = "cached_route_exhausted_before_pregrasp_phase_transition"
                    break
                previous_live = backend.latest_data
                pre = bridge.request({
                    "op": "observe", "include_snapshot_state": True,
                    "safety_geometry_policy": backend.safety_policy,
                })
                if int(pre.get("timestep", -1)) != current_t:
                    raise RuntimeError("fresh_pre_commit_observation_timestep_changed")
                previous_integration = np.asarray(
                    previous_live.get("snapshot_state", {}).get("mujoco_integration_state"),
                    dtype=np.float64,
                )
                fresh_integration = np.asarray(
                    pre.get("snapshot_state", {}).get("mujoco_integration_state"),
                    dtype=np.float64,
                )
                if previous_integration.shape != fresh_integration.shape or not np.array_equal(
                    previous_integration, fresh_integration
                ):
                    raise RuntimeError("fresh_pre_commit_observation_does_not_match_last_committed_state")
                pre_native = bridge.request({"op": "native_task_status"})
                pre["native_task_status"] = pre_native
                backend.latest_data = pre
                _validate_native_stage(pre_native, context="pre_commit")
                pre_geom = pre.get("safety_geometry", {})
                if pre_geom.get("status") != "complete":
                    raise RuntimeError("pre_commit_live_geometry_not_operationally_safe")
                pre_integration = np.asarray(
                    pre.get("snapshot_state", {}).get("mujoco_integration_state"),
                    dtype=np.float64,
                )
                if next_preview_snapshot_id and current_t == INITIAL_TIMESTEP:
                    # The one-step t794 preflight branch is already tied to
                    # this exact source Runtime Snapshot. Reuse that branch
                    # once instead of redundantly previewing the same action.
                    snapshot_id = next_preview_snapshot_id
                else:
                    snap = bridge.request({"op": "snapshot"})
                    snapshot_id = snap.get("snapshot_id")
                    if not isinstance(snapshot_id, str):
                        raise RuntimeError("current_state_snapshot_capture_failed")
                    snap_integration = np.asarray(
                        snap.get("snapshot_state", {}).get("mujoco_integration_state"),
                        dtype=np.float64,
                    )
                    if pre_integration.shape != snap_integration.shape or not np.array_equal(
                        pre_integration, snap_integration
                    ):
                        raise RuntimeError("pre_commit_observation_and_snapshot_Integration_State_differ")
                pre_margin = pre_geom.get("minimum_forbidden_clearance_m")
                if pre_margin is None or not np.isfinite(float(pre_margin)):
                    raise RuntimeError("pre_commit_forbidden_object_clearance_is_unknown")
                if not risk_armed and float(pre_margin) >= RISK_REARM_M:
                    risk_armed = True
                if float(pre_margin) < RISK_TRIGGER_M:
                    risk_armed = False
                if next_preview_snapshot_id != snapshot_id:
                    # Generate a fresh current-state branch. The initial t794
                    # preflight certificate is used once and cannot be replayed.
                    next_preview = bridge.request({
                        "op": "osc_prefix_probe",
                        "snapshot_id": snapshot_id,
                        "current_state_sha256": _array_sha256(pre_integration),
                        "jointspace_path_qpos": route["qpos_path"].tolist(),
                        "eef_reference_poses_world": route["eef_path"].tolist(),
                        "path_start_index": int(backend.plan_cursor),
                        "max_control_steps": 1,
                        "gripper_action": 0.0,
                        "risk_trigger_distance_m": RISK_TRIGGER_M,
                        "risk_rearm_distance_m": RISK_REARM_M,
                        "risk_crossing_armed": risk_armed,
                        "maximum_joint_path_deviation_rad": recovery.maximum_joint_path_deviation_rad,
                        "maximum_eef_tracking_error_m": recovery.maximum_eef_tracking_error_m,
                        "maximum_eef_orientation_error_rad": recovery.maximum_eef_orientation_error_rad,
                        "qpos_reference_mode": "diagnostic_only",
                        "current_minimum_forbidden_clearance_m": float(pre_margin),
                        "safety_geometry_policy": backend.safety_policy,
                    })
                    next_preview_snapshot_id = snapshot_id
                if (
                    next_preview.get("main_state_restored_verified") is not True
                    or next_preview.get("complete") is not True
                    or next_preview.get("safe") is not True
                    or next_preview.get("qpos_reference_mode") != "diagnostic_only"
                    or len(next_preview.get("actions", [])) != 1
                    or len(next_preview.get("steps", [])) != 1
                    or next_preview.get("stop_reason") not in {None, "narrow_channel_entry_replan"}
                ):
                    raise RuntimeError("rolling_current_state_physical_preview_rejected:" + json.dumps({
                        key: next_preview.get(key) for key in (
                            "status", "complete", "safe", "stop_reason", "failure_reason",
                            "main_state_restored_verified", "path_start_index",
                        )
                    }, sort_keys=True))
                candidate = next_preview["steps"][0]
                expected_path_index = int(backend.plan_cursor) + 1
                if int(candidate.get("path_index", -1)) != expected_path_index:
                    raise RuntimeError("rolling_preview_path_cursor_discontinuity")
                if (
                    float(candidate.get("eef_tracking_error_m", math.inf)) > recovery.maximum_eef_tracking_error_m
                    or float(candidate.get("eef_orientation_error_rad", math.inf)) > recovery.maximum_eef_orientation_error_rad
                ):
                    raise RuntimeError("rolling_preview_EEF_tracking_exceeded_frozen_limits")
                substep_audit = _check_substeps(
                    candidate.get("safety_geometry_substeps", []), backend.safety_policy
                )
                predicted = _inspect_predicted_state(
                    bridge, backend, snapshot_id, pre_integration,
                    candidate.get("terminal_runtime_snapshot"), backend.safety_policy,
                )
                if predicted["native_predicates"] != {
                    "pot1_on": False, "pot2_on": True, "stove_turnon": True,
                }:
                    raise RuntimeError("rolling_preview_predicted_native_stage_changed")
                predicted_file = output_dir / "branch_snapshots" / f"preview_timestep_{current_t + 1:06d}.pkl"
                predicted_file.parent.mkdir(parents=True, exist_ok=True)
                predicted_file.write_bytes(candidate["terminal_runtime_snapshot"])
                qpos_diag = float(candidate.get("joint_path_deviation_rad", math.nan))
                action = np.asarray(next_preview["actions"][0], dtype=np.float32)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise RuntimeError("rolling_preview_action_is_not_finite_native_7d_OSC")
                certificate = {
                    "source_timestep": current_t,
                    "target_timestep": current_t + 1,
                    "source_integration_state_sha256": _array_sha256(pre_integration),
                    "path_index": expected_path_index,
                    "action": action.tolist(),
                    "gripper_action": float(action[6]),
                    "gripper_command_policy": "neutral_hold_at_current_open_state",
                    "qpos_reference_l2_rad_diagnostic_only": qpos_diag,
                    "qpos_reference_exceeds_0_18_rad_diagnostic": bool(
                        qpos_diag > recovery.maximum_joint_path_deviation_rad
                    ),
                    "eef_tracking_error_m": float(candidate.get("eef_tracking_error_m", math.nan)),
                    "eef_orientation_error_rad": float(candidate.get("eef_orientation_error_rad", math.nan)),
                    "branch_stop_reason_diagnostic": next_preview.get("stop_reason"),
                    "risk_margin_before_m": float(pre_margin),
                    "risk_warning_only": bool(float(pre_margin) <= RISK_TRIGGER_M),
                    "contact_monitoring": next_preview.get("contact_monitoring"),
                    "substep_audit": substep_audit,
                    "predicted_state": predicted,
                    "predicted_snapshot_path": str(predicted_file),
                    "predicted_snapshot_sha256": sha256_file(predicted_file),
                    "committed": False,
                }
                certificate_stream.write(json.dumps(_jsonable(certificate), ensure_ascii=False, sort_keys=True) + "\n")
                certificate_stream.flush()
                os.fsync(certificate_stream.fileno())

                # Source is restored by prediction inspection and is rechecked
                # before committing exactly the certified action.
                current_observation = bridge.request({
                    "op": "observe", "include_snapshot_state": True,
                    "safety_geometry_policy": backend.safety_policy,
                })
                current_state = np.asarray(
                    current_observation.get("snapshot_state", {}).get("mujoco_integration_state"),
                    dtype=np.float64,
                )
                if current_state.shape != pre_integration.shape or not np.array_equal(
                    current_state, pre_integration
                ):
                    raise RuntimeError("current_physical_state_changed_after_preview_before_commit")
                current_observation["native_task_status"] = bridge.request({"op": "native_task_status"})
                backend.latest_data = current_observation
                backend._current_action_source = "safety_recovery_osc_pose"
                committed_observation = backend.execute_action(action, current_context)
                committed_step = backend.trajectory.steps[-1]
                committed_monitor = json.loads(committed_step["contact_monitoring_json"])
                actual_substeps = committed_monitor.get("geometry_monitoring", {}).get("substeps", [])
                actual_substep_audit = _check_substeps(actual_substeps, backend.safety_policy)
                if (
                    committed_monitor.get("status") != "complete"
                    or committed_monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
                    or committed_monitor.get("geometry_monitoring", {}).get("status") != "complete"
                ):
                    raise RuntimeError("committed_action_substep_monitoring_incomplete")
                actual_native = bridge.request({"op": "native_task_status"})
                actual_native_stage = _validate_native_stage(actual_native, context="committed_action")
                if committed_observation.done or committed_observation.task_success is True:
                    raise RuntimeError("unexpected_task_termination_before_Pot1_pregrasp")
                actual_t = int(committed_observation.timestep)
                actual_state = committed_observation.raw.get("snapshot_state", {})
                actual_robot = actual_state.get("robot_states", [{}])[0]
                actual_eef = np.asarray(actual_robot.get("eef_pos"), dtype=np.float64)
                actual_quat = np.asarray(actual_robot.get("eef_quat"), dtype=np.float64)
                actual_qpos = np.asarray(actual_robot.get("joint_qpos"), dtype=np.float64)
                actual_qpos_diag = float(np.linalg.norm(
                    actual_qpos - route["qpos_path"][expected_path_index]
                ))
                actual_eef_error = float(np.linalg.norm(
                    actual_eef - route["eef_path"][expected_path_index, :3]
                ))
                actual_orientation_error = _quat_error_rad(
                    actual_quat, route["eef_path"][expected_path_index, 3:]
                )
                if (
                    actual_eef_error > recovery.maximum_eef_tracking_error_m
                    or actual_orientation_error > recovery.maximum_eef_orientation_error_rad
                ):
                    raise RuntimeError("committed_action_EEF_path_tracking_exceeded_frozen_limits")
                actual_contacts = bridge.request({"op": "inspect_contacts"})
                robot_contact_count = sum(
                    1 for contact in actual_contacts.get("contacts", [])
                    if str(contact.get("body1", {}).get("name", "")).startswith(("robot0_", "gripper0_"))
                    or str(contact.get("body2", {}).get("name", "")).startswith(("robot0_", "gripper0_"))
                )
                if robot_contact_count:
                    raise RuntimeError("committed_second_object_recovery_has_unexpected_robot_contact")

                # Prediction/commit differences remain diagnostics; the actual
                # action's 25 substeps and live state are the binding safety
                # checks. The next certificate always starts from this state.
                actual_integration = np.asarray(
                    actual_state.get("mujoco_integration_state"),
                    dtype=np.float64,
                )
                predicted_envelope = pickle.loads(candidate["terminal_runtime_snapshot"])
                predicted_integration = np.asarray(
                    predicted_envelope["snapshot"].get("mujoco_state"), dtype=np.float64
                )
                if actual_integration.shape != predicted_integration.shape:
                    raise RuntimeError("committed_and_predicted_Integration_State_shapes_differ")
                integration_delta = actual_integration - predicted_integration
                delta_diag = {
                    "integration_state_exact_match": bool(np.array_equal(actual_integration, predicted_integration)),
                    "integration_state_max_abs_delta": float(np.max(np.abs(integration_delta))),
                    "integration_state_l2_delta": float(np.linalg.norm(integration_delta)),
                    "qpos_reference_l2_rad_diagnostic_only": actual_qpos_diag,
                    "eef_tracking_error_m": actual_eef_error,
                    "eef_orientation_error_rad": actual_orientation_error,
                }
                committed_rows.append({
                    "timestep": actual_t,
                    "path_index": expected_path_index,
                    "actual_native_predicates": actual_native_stage,
                    "actual_substep_safety": actual_substep_audit,
                    "actual_contact_signature": _contacts_signature(actual_contacts),
                    "prediction_commit_comparison_diagnostic": delta_diag,
                })
                certificate["committed"] = True
                certificate["actual_commit"] = committed_rows[-1]
                certificate_stream.write(json.dumps(_jsonable(certificate), ensure_ascii=False, sort_keys=True) + "\n")
                certificate_stream.flush()
                os.fsync(certificate_stream.fileno())

                movement = float(np.linalg.norm(actual_eef - previous_eef))
                motion_history.append(movement)
                previous_eef = actual_eef
                if len(motion_history) > recovery.stagnation_window_steps:
                    motion_history = motion_history[-recovery.stagnation_window_steps:]
                if (
                    len(motion_history) == recovery.stagnation_window_steps
                    and sum(motion_history) < recovery.minimum_progress_m
                ):
                    stop_reason = "actual_EEF_motion_stagnation_12_steps"
                    last_failure = "cumulative_EEF_motion_below_2mm_in_12_actions"
                    break
                current_context = backend.current_context()
                if current_context.phase == "approach":
                    stop_reason = "pot1_pregrasp_reached_and_approach_entered"
                    break
                if current_context.phase != "second_object_recovery":
                    stop_reason = f"unexpected_phase_transition:{current_context.phase}"
                    break
                # Preserve the RRT hysteresis state for diagnostic output only;
                # it does not dispatch any planner or reject a safe branch.
                next_live_margin = committed_step.get("contact_monitoring_json")
                monitor_obj = json.loads(next_live_margin) if isinstance(next_live_margin, str) else {}
                actual_geom = monitor_obj.get("geometry_monitoring", {}).get("substeps", [])
                actual_margin = min(
                    (float(item["minimum_clearance_m"]) for item in actual_geom if item.get("minimum_clearance_m") is not None),
                    default=float("nan"),
                )
                if math.isfinite(actual_margin):
                    if actual_margin >= RISK_REARM_M:
                        risk_armed = True
                    elif actual_margin <= RISK_TRIGGER_M and risk_armed:
                        risk_armed = False
                        risk_crossings.append({"timestep": actual_t, "minimum_clearance_m": actual_margin})
                next_preview_snapshot_id = ""  # force a fresh snapshot and branch next iteration
                # Current state/context after a committed transition becomes
                # the next loop's source. No arbitrary 30-action boundary.
        finally:
            certificate_stream.close()

        final = backend.observe()
        final_native = final.raw.get("native_task_status") or bridge.request({"op": "native_task_status"})
        final_stage = _validate_native_stage(final_native, context="terminal")
        safety_summary = {
            "actions": len(backend.trajectory.steps),
            "physical_substeps": sum(
                len(json.loads(step["contact_monitoring_json"]).get("geometry_monitoring", {}).get("substeps", []))
                for step in backend.trajectory.steps
            ),
            "all_substep_coverage_and_safety_pass": all(
                json.loads(step["contact_monitoring_json"]).get("status") == "complete"
                and json.loads(step["contact_monitoring_json"]).get("captured_physics_substeps") == EXPECTED_SUBSTEPS
                and json.loads(step["contact_monitoring_json"]).get("geometry_monitoring", {}).get("status") == "complete"
                and all(
                    not item.get(key)
                    for item in json.loads(step["contact_monitoring_json"]).get("geometry_monitoring", {}).get("substeps", [])
                    for key in (
                        "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
                        "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
                    )
                )
                for step in backend.trajectory.steps
            ),
        }
        clearance_minima: dict[str, float] = {}
        for step in backend.trajectory.steps:
            monitor = json.loads(step["contact_monitoring_json"])
            for substep in monitor.get("geometry_monitoring", {}).get("substeps", []):
                for name, record in substep.get("clearances_by_object", {}).items():
                    value = record.get("minimum_distance_m") if isinstance(record, dict) else record
                    if value is not None:
                        clearance_minima[str(name)] = min(
                            float(value), clearance_minima.get(str(name), float("inf"))
                        )
        safety_summary["minimum_clearance_by_object_m"] = clearance_minima
        safety_summary["strict_10mm_all_recorded_clearances"] = (
            all(value >= 0.010 for value in clearance_minima.values())
            if clearance_minima else None
        )
        result = {
            "status": "pregrasp_reached" if stop_reason == "pot1_pregrasp_reached_and_approach_entered" else "stopped_before_pregrasp",
            "initial_timestep": INITIAL_TIMESTEP,
            "final_timestep": int(final.timestep),
            "physical_actions_committed": backend.transition_count,
            "final_phase": current_context.phase,
            "final_stop_reason": stop_reason,
            "failure_detail": last_failure,
            "final_native_task_status": final_native,
            "final_native_stage": final_stage,
            "Pot2_On_preserved": final_stage["pot2_on"],
            "Pot1_On": final_stage["pot1_on"],
            "pregrasp_goal_position_error_m": float(np.linalg.norm(
                np.asarray(final.raw["snapshot_state"]["robot_states"][0]["eef_pos"], dtype=np.float64)
                - np.asarray(target.position_world_m, dtype=np.float64)
            )),
            "first_live_t794_physical_preview": first_preview_audit,
            "route_source": route_meta,
            "risk_warning_50mm_diagnostic_only": True,
            "risk_crossings_during_new_suffix": risk_crossings,
            "historical_risk_retry_guard_reset": False,
            "safety_acceptance": safety_summary,
            "native_safety_and_operational_safety": safety_summary["all_substep_coverage_and_safety_pass"],
            "strict_10mm_is_reported_by_per_object_clearance_records": True,
            "standard_1000_step_complete_task_success": False,
            "demonstration_1800_step_complete_task_success": False,
            "complete_round0_seed": False,
            "new_RRT_calls": 0,
            "lora_training_started": False,
            "output_dir": str(output_dir),
        }
        _write_json(output_dir / "suffix_result.json", result)
        _write_json(output_dir / "posthoc_acceptance_audit.json", {
            "schema": "gs_opd_t794_pot1_execution_aware_pregrasp_audit.v1",
            "source_snapshot": str(SOURCE_SNAPSHOT),
            "source_snapshot_sha256": attestation["source_snapshot_sha256"],
            "source_model_geometry_sha256": MODEL_SHA256,
            "source_integration_state_sha256": source["integration_state_sha256"],
            "final_timestep": int(final.timestep),
            "final_phase": current_context.phase,
            "pregrasp_reached": stop_reason == "pot1_pregrasp_reached_and_approach_entered",
            "final_native_task_status": final_native,
            "safety_acceptance": safety_summary,
            "complete_round0_seed": False,
            "route_reused_without_planning": True,
        })
    except BaseException as exc:
        failure_timestep = int(backend.latest_data.get("timestep", -1)) if backend and backend.latest_data else None
        result = {
            "status": "failed_closed_or_interrupted",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_timestep": failure_timestep,
            "final_phase": getattr(getattr(backend, "task_controller", None), "phase", None),
            "physical_actions_committed": backend.transition_count if backend else 0,
            "strict_restore_was_attested": bool(backend and backend.latest_data),
            "first_preview_passed": (output_dir / "first_preview_preflight.json").is_file(),
            "complete_round0_seed": False,
            "new_RRT_calls": 0,
            "output_dir": str(output_dir),
        }
        _write_json(output_dir / "failure_report.json", result)
    finally:
        if backend is not None:
            try:
                backend.finalize({
                    "complete_task_success": False,
                    "complete_safety_success": False,
                    "local_recovery_success": result.get("status") == "pregrasp_reached",
                    "standard_horizon_success": False,
                    "extended_horizon": False,
                    "result": result,
                })
            except Exception as exc:
                (output_dir / "finalize_error.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                )
                if hasattr(backend, "trajectory"):
                    try:
                        backend.trajectory.save(steps=backend.transition_count, chunks=backend.transition_count)
                    except Exception as save_exc:
                        (output_dir / "trajectory_save_error.txt").write_text(
                            f"{type(save_exc).__name__}: {save_exc}\n", encoding="utf-8"
                        )
            for writer in backend._video_writers.values():
                writer.release()
        bridge.close()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        result["output_dir"] = str(output_dir)
        result["elapsed_seconds"] = time.monotonic() - started
        _write_json(output_dir / "suffix_result.json", result)
        _write_json(output_dir / "execution_manifest.json", {
            "entrypoint": str(Path(__file__).resolve()),
            "source_archive": str(SOURCE_ARCHIVE),
            "source_snapshot": str(SOURCE_SNAPSHOT),
            "source_snapshot_sha256": attestation["source_snapshot_sha256"],
            "reused_route_events_sha256": route["events_sha256"],
            "route_plan_index": ROUTE_PLAN_INDEX,
            "planner_search_repeated": False,
            "physical_actions_committed": backend.transition_count if backend else 0,
            "result_status": result.get("status"),
            "complete_round0_seed": False,
        })
    return result


def main() -> int:
    args = _parser()
    result = run(args)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "pregrasp_reached" else 1


if __name__ == "__main__":
    raise SystemExit(main())
