# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Execute a bounded, physically certified Pot 2 suffix from the t463 Snapshot.

The archived 193-point RRT route is treated only as an EEF reference. Its 7D
qpos deviation is retained in the record but never authorizes or rejects an
action. Every committed action must first pass an isolated MuJoCo rollout from
the exact live Integration State, then match that rollout after commitment.
This entry point never calls an RRT planner.
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
from research.gs_opd.repair.diagnose_t463_osc_path_reference import (
    FORBIDDEN_CLEARANCE_M,
    JOINT_REFERENCE_LIMIT_RAD,
    MODEL_SHA256,
    OBJECT_ID,
    RISK_REARM_M,
    RISK_TRIGGER_M,
    T398_TO_T463_RUN,
    T463_CURSOR,
    T463_TIMESTEP,
    _normalized_state_hash,
    _read_journal,
    _source_463,
    _verify_geometry_substeps,
)
from research.gs_opd.repair.resume_t359_pot2_placement import (
    T359ResumeBackend,
    _PythonObjectState,
    _contact_sides,
)
from research.gs_opd.repair.run_t329_native_task_suffix import (
    CONFIG_PATH,
    EXPECTED_SUBSTEPS,
    _all_robot_contacts_finger_target,
    _atomic_npz,
    _canonical_sha256,
    _jsonable,
    _suffix_acceptance,
    _write_json,
)


MAX_FORMAL_ACTIONS = 450
MAX_ROLLING_PREVIEW_ACTIONS = 8
MAX_DEMONSTRATION_HORIZON = 1800
STANDARD_HORIZON = 1000
POT2_NONFINGER_CLEARANCE_M = 0.002
EEF_POSITION_NUMERIC_DIAGNOSTIC_TOLERANCE_M = 0.0001
EXPECTED_ROUTE_POINTS = 193
EXPECTED_ROUTE_QPOS_SHA256 = "a57e1ef5243098ed9003537ca701b5043ddf749be10554c647942f070ac25831"
EXPECTED_ROUTE_EEF_SHA256 = "28ea80890d4fcab0be21fcfab5c7a76419b005e88de2c240a2d81c2800d6b418"
POT2_ON_PREDICATE = ("on", OBJECT_ID, "flat_stove_1_cook_region")


class T463ExecutionBackend(T359ResumeBackend):
    """Reuse durable journaling and existing Pot 2 post-release acceptance."""

    def plan_rrt_connect(self, observation, goal, context, config):
        raise RuntimeError("t463_execution_aware_suffix_forbids_any_RRT_call")


def _sha(value: Any) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value, dtype=np.float64)).tobytes()
    ).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _native_on(status: dict[str, Any]) -> bool | None:
    matches = [
        item.get("value")
        for item in status.get("native_goal_values", [])
        if tuple(str(token).lower() for token in item.get("predicate", []))
        == POT2_ON_PREDICATE
    ]
    if len(matches) != 1 or not isinstance(matches[0], bool):
        return None
    return matches[0]


def _robot_objects(snapshot_state: dict[str, Any]) -> dict[str, Any]:
    objects = snapshot_state.get("object_states", {})
    matches = [
        value for key, value in objects.items()
        if str(key).endswith(":" + OBJECT_ID)
    ]
    if len(matches) != 1:
        raise RuntimeError("predicted_or_actual_Pot2_state_is_not_unique")
    return matches[0]


def _contacts_signature(contacts: dict[str, Any]) -> list[tuple[int, int]]:
    pairs = []
    for item in contacts.get("contacts", []):
        first, second = item.get("geom1", {}), item.get("geom2", {})
        a, b = int(first.get("id", -1)), int(second.get("id", -1))
        if a < 0 or b < 0:
            raise RuntimeError("contact_record_has_no_native_Geom_identity")
        pairs.append(tuple(sorted((a, b))))
    return sorted(set(pairs))


def _verify_substep_policy(
    substeps: list[dict[str, Any]], *, held: bool
) -> tuple[bool, dict[str, Any]]:
    if len(substeps) != EXPECTED_SUBSTEPS:
        return False, {"failures": [{"reason": "expected_25_substeps", "observed": len(substeps)}]}
    failures: list[dict[str, Any]] = []
    minima: dict[str, float | None] = {
        OBJECT_ID: None,
        "bottle_of_ginger_beer_1": None,
        "white_place_box_1": None,
        f"carried:{OBJECT_ID}->bottle_of_ginger_beer_1": None,
        f"carried:{OBJECT_ID}->white_place_box_1": None,
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
        clearances = row.get("clearances_by_object", {})
        for object_id, required in (
            (OBJECT_ID, POT2_NONFINGER_CLEARANCE_M),
            ("bottle_of_ginger_beer_1", FORBIDDEN_CLEARANCE_M),
            ("white_place_box_1", FORBIDDEN_CLEARANCE_M),
        ):
            value = clearances.get(object_id, {}).get("minimum_distance_m")
            if value is None or float(value) < required:
                failures.append({
                    "substep": index, "reason": "robot_object_clearance_below_policy",
                    "object_id": object_id, "distance_m": value, "required_m": required,
                })
            elif minima[object_id] is None or float(value) < minima[object_id]:
                minima[object_id] = float(value)
        if held:
            carried = row.get("carried_clearances_by_object", {})
            for object_id in ("bottle_of_ginger_beer_1", "white_place_box_1"):
                key = f"carried:{OBJECT_ID}->{object_id}"
                value = carried.get(object_id, {}).get("minimum_distance_m")
                if value is None or float(value) < FORBIDDEN_CLEARANCE_M:
                    failures.append({
                        "substep": index,
                        "reason": "carried_object_clearance_below_policy",
                        "object_id": object_id, "distance_m": value,
                        "required_m": FORBIDDEN_CLEARANCE_M,
                    })
                elif minima[key] is None or float(value) < minima[key]:
                    minima[key] = float(value)
    return not failures, {"minimum_clearance_by_object_m": minima, "failures": failures}


def _finger_grasp_evidence(backend: T463ExecutionBackend, contacts: dict[str, Any]) -> dict[str, Any]:
    binding = backend.bindings[OBJECT_ID]
    left, right, robot_contacts = _contact_sides(
        contacts, set(int(item) for item in binding["geom_ids"])
    )
    identity_ok = _all_robot_contacts_finger_target(contacts, backend)
    return {
        "left_finger_contact": bool(left),
        "right_finger_contact": bool(right),
        "all_robot_contacts_are_whitelisted_finger_to_pot2": bool(identity_ok),
        "robot_contact_count": len(robot_contacts),
        "contact_geom_pairs": _contacts_signature(contacts),
        "grasp_confirmed": bool(left and right and identity_ok),
    }


def _validate_live_phase_policy(
    backend: T463ExecutionBackend,
    archived_policy: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    """Keep archived hard policy fixed while checking existing placement pairs."""
    active = json.loads(json.dumps(backend.safety_policy))
    expected = json.loads(json.dumps(archived_policy))
    placement_pairs: list[dict[str, Any]] = []
    if phase in {"precise_placement", "release"}:
        parent = backend.current_region_parent
        fixture = backend.bindings.get(parent) if parent else None
        if not isinstance(fixture, dict) or not fixture.get("geom_ids"):
            raise RuntimeError("placement_phase_has_no_live_native_fixture_binding")
        support_ids = sorted(
            int(geom_id) for geom_id in backend.task_support_geom_ids
            if int(geom_id) in set(int(value) for value in fixture["geom_ids"])
        )
        if not support_ids:
            raise RuntimeError("placement_phase_has_no_configured_native_fixture_support_Geom")
        object_ids = sorted(int(value) for value in backend.bindings[OBJECT_ID]["geom_ids"])
        placement_pairs = [
            {"carried_geom_id": object_geom, "scene_geom_id": support_geom,
             "minimum_clearance_m": -0.002}
            for object_geom in object_ids for support_geom in support_ids
        ]
        actual_pairs = active.get("carried_object", {}).get("allowed_contact_pairs", [])
        if _canonical_sha256(actual_pairs) != _canonical_sha256(placement_pairs):
            raise RuntimeError("phase_controller_placement_contact_pairs_differ_from_explicit_fixture_pairs")
        active["carried_object"]["allowed_contact_pairs"] = expected.get(
            "carried_object", {}
        ).get("allowed_contact_pairs", [])
    active_hash = _canonical_sha256(active)
    expected_hash = _canonical_sha256(expected)
    if active_hash != expected_hash:
        differing = sorted(
            key for key in set(active) | set(expected)
            if _canonical_sha256({key: active.get(key)})
            != _canonical_sha256({key: expected.get(key)})
        )
        raise RuntimeError(
            "live_phase_policy_changed_frozen_safety_fields:"
            + ",".join(differing)
        )
    return {
        "frozen_policy_sha256": expected_hash,
        "active_phase_policy_sha256": _canonical_sha256(backend.safety_policy),
        "phase": phase,
        "phase_scoped_carried_support_pairs": placement_pairs,
        "frozen_thresholds_and_forbidden_pairs_unchanged": True,
    }


def _runtime_payload_snapshot(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError("isolated_preview_terminal_runtime_snapshot_missing")
    envelope = pickle.loads(payload)
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
        or not isinstance(envelope.get("snapshot"), dict)
    ):
        raise RuntimeError("isolated_preview_terminal_runtime_snapshot_fingerprint_invalid")
    return envelope


def _inspect_prediction(
    bridge: BridgeProcess,
    backend: T463ExecutionBackend,
    source_snapshot_id: str,
    source_integration: np.ndarray,
    payload: bytes,
    safety_policy: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    envelope = _runtime_payload_snapshot(payload)
    terminal_snapshot = envelope["snapshot"]
    expected_integration = np.asarray(terminal_snapshot.get("mujoco_state"), dtype=np.float64)
    if expected_integration.ndim != 1 or expected_integration.size == 0:
        raise RuntimeError("predicted_runtime_snapshot_has_no_complete_MuJoCo_Integration_State")
    try:
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
        if restored.get("status") != "restored":
            raise RuntimeError(f"predicted_branch_snapshot_restore_failed:{restored.get('error')}")
        predicted_status = bridge.request({"op": "native_task_status"})
        predicted_contacts = bridge.request({"op": "inspect_contacts"})
        predicted_grasp, _, grasp_records = backend._grasp_states()
        # Contact and carried-object permissions are derived from the restored
        # physical grasp at the predicted endpoint. Do not leave the source
        # state's carried-object binding active after a legitimate release.
        backend._refresh_task_contact_policy(phase, OBJECT_ID, predicted_grasp)
        predicted_live = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        on_value = _native_on(predicted_status)
        if predicted_status.get("status") != "complete" or on_value is None:
            raise RuntimeError("predicted_branch_native_task_predicate_unknown")
        if predicted_live.get("safety_geometry", {}).get("status") != "complete":
            raise RuntimeError("predicted_branch_terminal_geometry_status_incomplete_or_unsafe")
        terminal_geometry = predicted_live["safety_geometry"]
        if any(terminal_geometry.get(key) for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
        )):
            raise RuntimeError("predicted_branch_terminal_state_fails_Native_or_Operational_safety")
        if not _finger_grasp_evidence(backend, predicted_contacts).get(
            "all_robot_contacts_are_whitelisted_finger_to_pot2"
        ):
            raise RuntimeError("predicted_branch_has_a_nonfinger_or_non_Pot2_robot_contact")
        terminal_observation = predicted_live.get("snapshot_state")
        if not isinstance(terminal_observation, dict):
            raise RuntimeError("predicted_branch_terminal_snapshot_diagnostics_missing")
        predicted_integration = np.asarray(
            terminal_observation.get("mujoco_integration_state"), dtype=np.float64
        )
        if predicted_integration.shape != expected_integration.shape or not np.allclose(
            predicted_integration, expected_integration, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("predicted_runtime_snapshot_and_restored_Integration_State_disagree")
    finally:
        returned = bridge.request({"op": "restore", "snapshot_id": source_snapshot_id})
        if returned.get("restored_snapshot_id") != source_snapshot_id:
            raise RuntimeError(f"source_state_restore_after_prediction_inspection_failed:{returned.get('error')}")
        restored_observation = bridge.request({
            "op": "observe", "include_snapshot_state": True,
        })
        actual_source = np.asarray(
            restored_observation.get("snapshot_state", {}).get("mujoco_integration_state"),
            dtype=np.float64,
        )
        if actual_source.shape != source_integration.shape or not np.array_equal(
            actual_source, source_integration
        ):
            raise RuntimeError("source_Integration_State_changed_during_prediction_inspection")
        source_held, _, _ = backend._grasp_states()
        backend._refresh_task_contact_policy(phase, OBJECT_ID, source_held)
        restored_observation = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        # Keep the backend's cached physical data aligned with the exact
        # restored source state even when predicted-state validation fails.
        backend.latest_data = restored_observation
        backend.latest_data["native_task_status"] = bridge.request({"op": "native_task_status"})
        backend.latest_data["contacts"] = bridge.request({"op": "inspect_contacts"})
    return {
        "native_task_status": predicted_status,
        "pot2_on": on_value,
        "predicted_held_object": predicted_grasp,
        "predicted_grasp_evidence": grasp_records.get(OBJECT_ID, {}),
        "predicted_contact_geom_pairs": _contacts_signature(predicted_contacts),
        "predicted_current_geometry": predicted_live.get("safety_geometry"),
        "predicted_integration_state_sha256": _sha(predicted_integration),
        "source_restore_verified": True,
        "contact_policy_rebound_to_predicted_grasp": True,
    }


def _compare_prediction_to_commit(
    predicted_payload: bytes,
    actual_data: dict[str, Any],
    predicted_status: dict[str, Any],
    predicted_step: dict[str, Any],
    actual_runtime_payload: bytes,
    predicted_contacts: dict[str, Any],
    actual_contacts: dict[str, Any],
) -> dict[str, Any]:
    predicted_snapshot = _runtime_payload_snapshot(predicted_payload)["snapshot"]
    actual_snapshot = _runtime_payload_snapshot(actual_runtime_payload)["snapshot"]
    predicted_state = np.asarray(predicted_snapshot.get("mujoco_state"), dtype=np.float64)
    actual_state = np.asarray(actual_snapshot.get("mujoco_state"), dtype=np.float64)
    if predicted_state.shape != actual_state.shape:
        raise RuntimeError("prediction_commit_Integration_State_shape_mismatch")
    max_abs = float(np.max(np.abs(predicted_state - actual_state)))
    integration_match = bool(np.allclose(predicted_state, actual_state, rtol=0.0, atol=1e-9))
    predicted_status_value = _native_on(predicted_status)
    actual_status = actual_data.get("native_task_status", {})
    actual_status_value = _native_on(actual_status)
    if predicted_status_value is None or actual_status_value is None:
        raise RuntimeError("prediction_commit_native_Pot2_On_predicate_is_unknown")
    def predicate_map(status: dict[str, Any]) -> dict[tuple[str, ...], bool]:
        return {
            tuple(str(token).lower() for token in item.get("predicate", [])): bool(item["value"])
            for item in status.get("native_goal_values", [])
            if isinstance(item, dict) and isinstance(item.get("value"), bool)
        }
    native_match = (
        predicted_status_value == actual_status_value
        and predicate_map(predicted_status) == predicate_map(actual_status)
        and predicted_status.get("task_success") == actual_status.get("task_success")
    )
    predicted_pairs = _contacts_signature(predicted_contacts)
    actual_pairs = _contacts_signature(actual_contacts)
    contact_match = predicted_pairs == actual_pairs
    reward_match = math.isclose(
        float(predicted_step.get("reward", math.nan)),
        float(actual_data.get("reward", math.nan)), rel_tol=0.0, abs_tol=1e-9,
    )
    done_match = bool(predicted_step.get("done")) == bool(actual_data.get("done"))
    if not (integration_match and native_match and contact_match and reward_match and done_match):
        raise RuntimeError(
            "prediction_commit_state_mismatch:"
            f"integration_max_abs={max_abs};native={native_match};contacts={contact_match};"
            f"reward={reward_match};done={done_match}"
        )
    return {
        "integration_state_match_atol_1e-9": integration_match,
        "integration_state_max_abs_error": max_abs,
        "native_pot2_on_match": native_match,
        "all_native_goal_predicates_match": native_match,
        "contact_geom_pair_set_match": contact_match,
        "reward_match": reward_match,
        "done_match": done_match,
        "predicted_contact_geom_pairs": predicted_pairs,
        "actual_contact_geom_pairs": actual_pairs,
        "predicted_native_pot2_on": predicted_status_value,
        "actual_native_pot2_on": actual_status_value,
    }


def _verify_one_branch_step(
    step: dict[str, Any], *, held_expected: bool, phase: str, backend: T463ExecutionBackend
) -> dict[str, Any]:
    substeps = step.get("safety_geometry_substeps", [])
    policy_ok, policy_report = _verify_substep_policy(substeps, held=held_expected)
    if not policy_ok:
        raise RuntimeError("isolated_action_branch_failed_frozen_substep_policy:" + json.dumps(policy_report, sort_keys=True))
    if len(substeps) != EXPECTED_SUBSTEPS or any(row.get("status") != "complete" for row in substeps):
        raise RuntimeError("isolated_action_branch_did_not_monitor_all_25_complete_substeps")
    grasp = _finger_grasp_evidence(backend, step.get("contacts", {}))
    if not grasp["all_robot_contacts_are_whitelisted_finger_to_pot2"]:
        raise RuntimeError("isolated_action_branch_contact_identity_is_not_finger_only")
    if held_expected and not grasp["grasp_confirmed"]:
        raise RuntimeError(f"isolated_action_branch_lost_bilateral_Pot2_grasp_in_{phase}")
    if bool(step.get("done")):
        raise RuntimeError("isolated_action_branch_would_terminate_environment")
    return {"safety": policy_report, "grasp": grasp}


def _parse_route(route_path: Path, policy_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    with np.load(route_path, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos_path"], dtype=np.float64)
        poses = np.asarray(archive["eef_reference_poses_world"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if qpos.shape != (EXPECTED_ROUTE_POINTS, 7) or poses.shape != (EXPECTED_ROUTE_POINTS, 7):
        raise RuntimeError("archived_t398_route_does_not_match_193_point_7D_contract")
    if not np.isfinite(qpos).all() or not np.isfinite(poses).all():
        raise RuntimeError("archived_t398_route_contains_nonfinite_values")
    qhash, ehash = _array_hash(qpos), _array_hash(poses)
    if qhash != EXPECTED_ROUTE_QPOS_SHA256 or ehash != EXPECTED_ROUTE_EEF_SHA256:
        raise RuntimeError("archived_t398_route_array_fingerprint_mismatch")
    if metadata.get("model_geometry_sha256") != MODEL_SHA256:
        raise RuntimeError("archived_t398_route_model_fingerprint_mismatch")
    if metadata.get("qpos_path_sha256_float64_c_order") != qhash:
        raise RuntimeError("archived_t398_route_metadata_qpos_hash_mismatch")
    if metadata.get("eef_reference_sha256_float64_c_order") != ehash:
        raise RuntimeError("archived_t398_route_metadata_EEF_hash_mismatch")
    if _canonical_sha256(policy) != metadata.get("safety_policy_sha256"):
        raise RuntimeError("archived_t398_route_safety_policy_fingerprint_mismatch")
    if metadata.get("planner_status") != "offline_feasible" or metadata.get("planner_operation") != "rrt_connect":
        raise RuntimeError("archived_route_lacks_original_offline_feasibility_provenance")
    return qpos, poses, metadata, policy


def _source_from_prior_segment(prior_dir: Path) -> dict[str, Any]:
    """Strictly attest an exact saved terminal state from this same suffix."""
    prior_dir = Path(prior_dir)
    if (prior_dir / "prediction_commit_mismatch_stop.json").is_file():
        return _source_from_prediction_mismatch_segment(prior_dir)
    result_path = prior_dir / "suffix_result.json"
    index_path = prior_dir / "durable_action_journal" / "index.jsonl"
    cert_path = prior_dir / "execution_certification.jsonl"
    if not result_path.is_file() or not index_path.is_file() or not cert_path.is_file():
        raise FileNotFoundError("prior_segment_result_index_or_certification_log_missing")
    prior_result = json.loads(result_path.read_text(encoding="utf-8"))
    prior_config_path = prior_dir / "run_config.json"
    if not prior_config_path.is_file():
        raise FileNotFoundError("prior_segment_run_config_missing_for_episode_budget_attestation")
    prior_config = json.loads(prior_config_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    certs = [json.loads(line) for line in cert_path.read_text().splitlines() if line.strip()]
    prior_actions = int(prior_result.get("physical_actions_committed", -1))
    if prior_actions <= 0 or len(rows) != prior_actions or len(certs) != prior_actions:
        raise RuntimeError("prior_segment_action_or_certificate_count_is_not_complete")
    segment_parent_actions = int(prior_config.get("source_action_count_from_t463", -1))
    cumulative_actions = int(prior_result.get("cumulative_physical_actions_from_t463", -1))
    if (
        segment_parent_actions < 0
        or int(prior_result.get("prior_physical_actions_from_t463", -1)) != segment_parent_actions
        or cumulative_actions != segment_parent_actions + prior_actions
        or int(prior_config.get("maximum_cumulative_formal_actions_from_t463", -1)) != MAX_FORMAL_ACTIONS
    ):
        raise RuntimeError("prior_segment_cumulative_episode_action_budget_attestation_mismatch")
    if prior_result.get("failure_timestep") != rows[-1].get("timestep"):
        raise RuntimeError("prior_segment_failure_state_does_not_match_last_durable_action")
    for index, cert in enumerate(certs):
        compare = cert.get("prediction_commit_runtime", {})
        if (
            not compare.get("integration_state_match_atol_1e-9")
            or not compare.get("all_native_goal_predicates_match")
            or not compare.get("contact_geom_pair_set_match")
            or len(cert.get("actual_safety_substeps", [])) != EXPECTED_SUBSTEPS
            or not cert.get("actual_grasp", {}).get("all_robot_contacts_are_whitelisted_finger_to_pot2")
        ):
            raise RuntimeError(f"prior_segment_action_certificate_failed_at_index_{index}")
        if any(
            substep.get("status") != "complete"
            or any(substep.get(key) for key in (
                "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
                "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
            ))
            for substep in cert["actual_safety_substeps"]
        ):
            raise RuntimeError(f"prior_segment_substep_safety_failed_at_index_{index}")
    last = rows[-1]
    journal_path = Path(last["journal_path"])
    if sha256_file(journal_path) != last.get("journal_sha256"):
        raise RuntimeError("prior_segment_terminal_action_journal_hash_mismatch")
    data = _read_journal(journal_path)
    if data["monitor"].get("status") != "complete":
        raise RuntimeError("prior_segment_terminal_action_monitor_incomplete")
    substeps = data["monitor"].get("geometry_monitoring", {}).get("substeps", [])
    if len(substeps) != EXPECTED_SUBSTEPS or any(row.get("status") != "complete" for row in substeps):
        raise RuntimeError("prior_segment_terminal_action_did_not_have_25_complete_substeps")
    snapshot_path = Path(last["runtime_snapshot_path"])
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    envelope = pickle.loads(snapshot_path.read_bytes())
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
    ):
        raise RuntimeError("prior_segment_terminal_runtime_snapshot_model_fingerprint_mismatch")
    snapshot = envelope.get("snapshot")
    timestep = int(last.get("timestep", -1))
    if not isinstance(snapshot, dict) or int(snapshot.get("task_bookkeeping", {}).get("timestep", -1)) != timestep:
        raise RuntimeError("prior_segment_terminal_runtime_snapshot_timestep_mismatch")
    physical = data["physical"]
    integration = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    physical_integration = np.asarray(physical.get("mujoco_integration_state"), dtype=np.float64)
    if integration.shape != physical_integration.shape or not np.array_equal(integration, physical_integration):
        raise RuntimeError("prior_segment_terminal_runtime_snapshot_Integration_State_mismatch")
    robot = physical["robot_states"][0]
    obj = physical["object_states"][f"objects_dict:{OBJECT_ID}"]
    runtime_robot = snapshot.get("robots", [{}])[0]
    if not runtime_robot.get("has_gripper_current_action"):
        raise RuntimeError("prior_segment_terminal_native_gripper_state_missing")
    attestation = {
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(snapshot_path),
        "source_snapshot_sha256": sha256_file(snapshot_path),
        "source_action_record": str(journal_path),
        "source_action_record_sha256": last["journal_sha256"],
        "timestep": timestep,
        "model_geometry_sha256": MODEL_SHA256,
        "object_id": OBJECT_ID,
        "arm_joint_qpos": robot["joint_qpos"],
        "eef_position_world_m": robot["eef_pos"],
        "eef_quaternion_wxyz": robot["eef_quat"],
        "gripper_qpos_m": robot["gripper_qpos"],
        "object_position_world_m": obj["root_pos"],
        "object_quaternion_wxyz": obj["root_quat"],
        "restored_gripper_current_action": runtime_robot["gripper_current_action"],
        "integration_state_sha256": _sha(integration),
        "controller_initial_joint_rad": runtime_robot.get("controller", {}).get("initial_joint"),
        "prior_segment_path": str(prior_dir),
        "prior_segment_actions": cumulative_actions,
        "prior_segment_native_operational_prefix_verified": True,
    }
    route_cursor = int(certs[-1].get("path_index") or T463_CURSOR)
    terminal_release_event = None
    phase_events_path = prior_dir / "task_phase_events.jsonl"
    if phase_events_path.is_file():
        phase_events = [
            json.loads(line) for line in phase_events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        terminal_events = [
            event for event in phase_events
            if event.get("event") == "task_phase_transition_observed"
            and int(event.get("timestep", -1)) == timestep
        ]
        if terminal_events:
            event = terminal_events[-1]
            if (
                event.get("from_phase") == "precise_placement"
                and event.get("to_phase") == "release"
                and event.get("held_object") == OBJECT_ID
                and event.get("native_pot2_on") is False
            ):
                terminal_release_event = event
    return {
        "source_run": str(prior_dir),
        "snapshot_path": snapshot_path,
        "snapshot": snapshot,
        "attestation": attestation,
        "integration_sha256": _sha(integration),
        "prior_actions": cumulative_actions,
        "route_cursor": route_cursor,
        "prior_result": prior_result,
        "terminal_release_event": terminal_release_event,
    }


def _source_from_prediction_mismatch_segment(prior_dir: Path) -> dict[str, Any]:
    """Attest an actual endpoint after a derived-EEF-only numeric stop.

    The earlier 126-action segment is revalidated through the normal strict
    certificate path. The single t590 action is accepted as a new source only
    after verifying its durable action/snapshot hashes and all 25 actual
    Native/Operational substeps. The former prediction mismatch stays attached
    to the source attestation as diagnostic evidence. The live Bridge restore,
    model/runtime checks, task adoption and grasp checks still run before any
    action is permitted.
    """
    prior_dir = Path(prior_dir)
    stop_path = prior_dir / "prediction_commit_mismatch_stop.json"
    if stop_path.is_file():
        stop_probe = json.loads(stop_path.read_text(encoding="utf-8"))
        stop_timestep = int(stop_probe.get("failure_timestep", stop_probe.get("timestep", -1)))
        if stop_timestep != 590:
            return _source_from_single_eef_diagnostic_segment(prior_dir)
    result_path = prior_dir / "suffix_result.json"
    config_path = prior_dir / "run_config.json"
    stop_path = prior_dir / "prediction_commit_mismatch_stop.json"
    index_path = prior_dir / "durable_action_journal" / "index.jsonl"
    if not all(path.is_file() for path in (result_path, config_path, stop_path, index_path)):
        raise FileNotFoundError("t590_numeric_mismatch_resume_evidence_missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    chain_path = prior_dir / "continuation_chain.json"
    if not chain_path.is_file():
        raise FileNotFoundError("t590_resume_continuation_chain_missing")
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    parent_dir = Path(config.get("same_episode_continuation_from") or "")
    if not parent_dir.is_dir():
        raise RuntimeError("t590_resume_parent_segment_missing")
    parent = _source_from_prior_segment(parent_dir)
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (
        stop.get("status") != "stopped_fail_closed_on_prediction_commit_mismatch"
        or stop.get("exception") != result.get("failure_reason")
        or not str(stop.get("exception", "")).startswith("RuntimeError: prediction_commit_eef_pos_mismatch:")
        or int(stop.get("timestep", -1)) != 590
        or float(stop.get("observed", math.inf)) > EEF_POSITION_NUMERIC_DIAGNOSTIC_TOLERANCE_M
        or stop.get("metric") != "max_abs_component_error_m"
    ):
        raise RuntimeError("t590_resume_source_is_not_the_attested_small_EEF_numeric_mismatch")
    if (
        len(rows) != 1
        or int(result.get("initial_timestep", -1)) != 589
        or int(result.get("failure_timestep", -1)) != 590
        or int(result.get("prior_physical_actions_from_t463", -1)) != 126
        or int(result.get("physical_actions_committed", -1)) != 1
        or int(result.get("cumulative_physical_actions_from_t463", -1)) != 127
        or int(result.get("route_cursor", -1)) != 191
        or int(parent.get("prior_actions", -1)) != 126
        or int(parent.get("attestation", {}).get("timestep", -1)) != 589
    ):
        raise RuntimeError("t590_resume_action_or_parent_chain_counts_do_not_match")
    parent_snapshot_path = Path(parent["attestation"]["source_snapshot_path"])
    source_589_copy = prior_dir / "source_t589.pkl"
    if (
        not source_589_copy.is_file()
        or sha256_file(source_589_copy) != sha256_file(parent_snapshot_path)
    ):
        raise RuntimeError("t590_resume_t589_source_snapshot_is_not_the_exact_verified_parent_endpoint")
    row = rows[0]
    journal_path = Path(row["journal_path"])
    if sha256_file(journal_path) != row.get("journal_sha256"):
        raise RuntimeError("t590_resume_action_journal_hash_mismatch")
    data = _read_journal(journal_path)
    monitor = data["monitor"]
    substeps = monitor.get("geometry_monitoring", {}).get("substeps", [])
    if (
        int(row.get("timestep", -1)) != 590
        or row.get("task_phase") != "precise_placement"
        or monitor.get("status") != "complete"
        or monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
        or monitor.get("geometry_monitoring", {}).get("status") != "complete"
        or len(substeps) != EXPECTED_SUBSTEPS
    ):
        raise RuntimeError("t590_resume_action_monitor_is_incomplete")
    safe, safety_audit = _verify_substep_policy(substeps, held=True)
    if not safe or any(
        substep.get(key)
        for substep in substeps
        for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
            "strict_joint_limit_violations",
        )
    ):
        raise RuntimeError("t590_resume_actual_substeps_fail_frozen_native_operational_policy")
    native = data["native"]
    if (
        native.get("status") != "complete"
        or int(native.get("timestep", -1)) != 590
        or _native_on(native) is not False
        or native.get("task_success") is not False
        or native.get("done") is not False
    ):
        raise RuntimeError("t590_resume_native_task_state_is_not_a_live_pending_placement")
    snapshot_path = Path(row["runtime_snapshot_path"])
    expected_snapshot_hash = (
        chain.get("episode_chain", {}).get("timestep_590", {}).get("runtime_snapshot_sha256")
    )
    if not snapshot_path.is_file() or sha256_file(snapshot_path) != expected_snapshot_hash:
        raise RuntimeError("t590_resume_actual_runtime_snapshot_hash_mismatch")
    envelope = pickle.loads(snapshot_path.read_bytes())
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
    ):
        raise RuntimeError("t590_resume_actual_runtime_snapshot_model_fingerprint_mismatch")
    snapshot = envelope.get("snapshot")
    if not isinstance(snapshot, dict) or int(snapshot.get("task_bookkeeping", {}).get("timestep", -1)) != 590:
        raise RuntimeError("t590_resume_actual_runtime_snapshot_timestep_mismatch")
    integration = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    physical_integration = np.asarray(data["physical"].get("mujoco_integration_state"), dtype=np.float64)
    if integration.shape != physical_integration.shape or not np.array_equal(integration, physical_integration):
        raise RuntimeError("t590_resume_actual_snapshot_and_action_journal_integration_state_mismatch")
    physical = data["physical"]
    robot = physical.get("robot_states", [{}])[0]
    object_rows = [
        value for key, value in physical.get("object_states", {}).items()
        if str(key).endswith(":" + OBJECT_ID)
    ]
    runtime_robots = snapshot.get("robots", [])
    if len(object_rows) != 1 or len(runtime_robots) != 1 or not runtime_robots[0].get("has_gripper_current_action"):
        raise RuntimeError("t590_resume_actual_snapshot_robot_gripper_or_Pot2_state_missing")
    attestation = {
        # Keep the runtime bridge's established attestation contract while
        # preserving the provenance of this independently audited t590 source.
        "status": "verified_source_action_snapshot",
        "source_attestation_kind": "actual_t590_after_small_EEF_prediction_mismatch",
        "source_snapshot_path": str(snapshot_path),
        "source_snapshot_sha256": sha256_file(snapshot_path),
        "source_action_record": str(journal_path),
        "source_action_record_sha256": row["journal_sha256"],
        "timestep": 590,
        "model_geometry_sha256": MODEL_SHA256,
        "object_id": OBJECT_ID,
        "arm_joint_qpos": robot["joint_qpos"],
        "eef_position_world_m": robot["eef_pos"],
        "eef_quaternion_wxyz": robot["eef_quat"],
        "gripper_qpos_m": robot["gripper_qpos"],
        "object_position_world_m": object_rows[0]["root_pos"],
        "object_quaternion_wxyz": object_rows[0]["root_quat"],
        "restored_gripper_current_action": runtime_robots[0]["gripper_current_action"],
        "integration_state_sha256": _sha(integration),
        "controller_initial_joint_rad": runtime_robots[0].get("controller", {}).get("initial_joint"),
        "prior_segment_path": str(prior_dir),
        "prior_segment_actions": 127,
        "prior_segment_native_operational_prefix_verified": True,
        "prior_numeric_mismatch_diagnostic_only": {
            "component": "derived_robot0_eef_pos",
            "max_abs_component_error_m": float(stop["observed"]),
            "source_reason": stop["exception"],
            "current_action_replayed": False,
        },
        "prior_segment_actual_safety_audit": safety_audit,
    }
    return {
        "source_run": str(prior_dir),
        "snapshot_path": snapshot_path,
        "snapshot": snapshot,
        "attestation": attestation,
        "integration_sha256": _sha(integration),
        "prior_actions": 127,
        "route_cursor": 191,
        "prior_result": result,
    }


def _source_from_single_eef_diagnostic_segment(prior_dir: Path) -> dict[str, Any]:
    """Resume from a fully journaled one-action endpoint stopped on EEF derivation.

    Acceptance requires the actual and predicted runtime Snapshot bytes and
    Integration State to match exactly, plus complete recorded Native and
    Operational monitoring. Only deterministic EEF/object root transforms
    derived from that exact state become diagnostics; joint/gripper state,
    velocity, contacts, predicates and safety remain strict.
    """
    prior_dir = Path(prior_dir)
    result_path = prior_dir / "suffix_result.json"
    config_path = prior_dir / "run_config.json"
    stop_path = prior_dir / "prediction_commit_mismatch_stop.json"
    index_path = prior_dir / "durable_action_journal" / "index.jsonl"
    if not all(path.is_file() for path in (result_path, config_path, stop_path, index_path)):
        raise FileNotFoundError("numeric_EEF_endpoint_resume_evidence_missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    record_path = prior_dir / "prediction_commit_records" / f"prediction_t{int(stop.get('failure_timestep', -1)):06d}.json"
    if not record_path.is_file():
        raise FileNotFoundError("numeric_EEF_endpoint_prediction_record_missing")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    parent_dir = Path(config.get("same_episode_continuation_from") or "")
    if not parent_dir.is_dir():
        raise RuntimeError("numeric_EEF_endpoint_parent_segment_missing")
    parent = _source_from_prior_segment(parent_dir)
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    timestep = int(stop.get("failure_timestep", -1))
    pre_timestep = timestep - 1
    stop_reason = str(stop.get("failure_reason", ""))
    if (
        stop.get("status") != "stopped_fail_closed_on_prediction_commit_mismatch"
        or not any(stop_reason.startswith(prefix) for prefix in (
            "RuntimeError: prediction_commit_eef_pos_mismatch:",
            "RuntimeError: prediction_commit_eef_quat_mismatch:",
            "RuntimeError: prediction_commit_Pot2_root_pos_mismatch:",
            "RuntimeError: prediction_commit_Pot2_root_quat_mismatch:",
        ))
        or record.get("failure_reason") != stop.get("failure_reason")
        or record.get("status") != "physical_commit_stopped_before_certification"
        or int(record.get("pre_timestep", -1)) != pre_timestep
        or int(record.get("predicted_timestep", -1)) != timestep
        or record.get("qpos_reference_mode") != "diagnostic_only"
        or len(rows) != 1
        or int(result.get("initial_timestep", -1)) != pre_timestep
        or int(result.get("failure_timestep", -1)) != timestep
        or int(result.get("physical_actions_committed", -1)) != 1
        or int(result.get("prior_physical_actions_from_t463", -1)) != int(parent.get("prior_actions", -2))
        or int(result.get("cumulative_physical_actions_from_t463", -1)) != int(parent.get("prior_actions", -2)) + 1
        or int(parent.get("attestation", {}).get("timestep", -1)) != pre_timestep
        or int(config.get("route_cursor_start", -1)) != int(parent.get("route_cursor", -2))
    ):
        raise RuntimeError("numeric_EEF_endpoint_action_or_parent_chain_is_inconsistent")
    parent_snapshot = Path(parent["attestation"]["source_snapshot_path"])
    parent_copy = prior_dir / f"source_t{pre_timestep}.pkl"
    if not parent_copy.is_file() or sha256_file(parent_copy) != sha256_file(parent_snapshot):
        raise RuntimeError("numeric_EEF_endpoint_parent_snapshot_is_not_exact")
    row = rows[0]
    journal_path = Path(row["journal_path"])
    if sha256_file(journal_path) != row.get("journal_sha256") or int(row.get("timestep", -1)) != timestep:
        raise RuntimeError("numeric_EEF_endpoint_action_journal_hash_or_timestep_mismatch")
    data = _read_journal(journal_path)
    monitor = data.get("monitor", {})
    geometry = monitor.get("geometry_monitoring", {})
    substeps = geometry.get("substeps", [])
    safe, safety_audit = _verify_substep_policy(substeps, held=True)
    if (
        monitor.get("status") != "complete"
        or monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
        or geometry.get("status") != "complete"
        or len(substeps) != EXPECTED_SUBSTEPS
        or not safe
    ):
        raise RuntimeError("numeric_EEF_endpoint_actual_substep_safety_not_verified")
    native = data.get("native", {})
    if (
        native.get("status") != "complete"
        or int(native.get("timestep", -1)) != timestep
        or _native_on(native) is not False
        or native.get("task_success") is not False
        or native.get("done") is not False
    ):
        raise RuntimeError("numeric_EEF_endpoint_native_task_state_is_not_pending_Pot2_placement")
    snapshot_path = Path(row["runtime_snapshot_path"])
    predicted_path = Path(record.get("expected_runtime_snapshot_path") or "")
    expected_hash = record.get("expected_runtime_snapshot_sha256")
    if (
        not snapshot_path.is_file()
        or not predicted_path.is_file()
        or sha256_file(snapshot_path) != expected_hash
        or sha256_file(predicted_path) != expected_hash
        or sha256_file(snapshot_path) != sha256_file(predicted_path)
    ):
        raise RuntimeError("numeric_EEF_endpoint_actual_predicted_runtime_snapshot_bytes_differ")
    envelope = pickle.loads(snapshot_path.read_bytes())
    predicted_envelope = pickle.loads(predicted_path.read_bytes())
    snapshot = envelope.get("snapshot")
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
        or predicted_envelope.get("model_geometry_sha256") != MODEL_SHA256
        or not isinstance(snapshot, dict)
        or int(snapshot.get("task_bookkeeping", {}).get("timestep", -1)) != timestep
    ):
        raise RuntimeError("numeric_EEF_endpoint_runtime_snapshot_fingerprint_or_timestep_mismatch")
    integration = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    journal_integration = np.asarray(data.get("physical", {}).get("mujoco_integration_state"), dtype=np.float64)
    predicted_integration = np.asarray(predicted_envelope.get("snapshot", {}).get("mujoco_state"), dtype=np.float64)
    if (
        integration.shape != journal_integration.shape
        or integration.shape != predicted_integration.shape
        or not np.array_equal(integration, journal_integration)
        or not np.array_equal(integration, predicted_integration)
    ):
        raise RuntimeError("numeric_EEF_endpoint_actual_predicted_Integration_State_mismatch")
    actual_signature = _contacts_signature(data.get("contacts", {}))
    expected_signature = [tuple(int(x) for x in pair) for pair in record.get("expected_contact_signature", [])]
    if actual_signature != expected_signature:
        raise RuntimeError("numeric_EEF_endpoint_actual_predicted_contact_identity_mismatch")
    robot_rows = data.get("physical", {}).get("robot_states", [])
    object_rows = [
        value for key, value in data.get("physical", {}).get("object_states", {}).items()
        if str(key).endswith(":" + OBJECT_ID)
    ]
    runtime_robots = snapshot.get("robots", [])
    if len(robot_rows) != 1 or len(object_rows) != 1 or len(runtime_robots) != 1:
        raise RuntimeError("numeric_EEF_endpoint_robot_or_Pot2_state_missing")
    if not runtime_robots[0].get("has_gripper_current_action"):
        raise RuntimeError("numeric_EEF_endpoint_native_gripper_state_missing")
    attestation = {
        "status": "verified_source_action_snapshot",
        "source_attestation_kind": "actual_endpoint_after_derived_pose_numeric_difference",
        "source_snapshot_path": str(snapshot_path),
        "source_snapshot_sha256": sha256_file(snapshot_path),
        "source_action_record": str(journal_path),
        "source_action_record_sha256": row["journal_sha256"],
        "timestep": timestep,
        "model_geometry_sha256": MODEL_SHA256,
        "object_id": OBJECT_ID,
        "arm_joint_qpos": robot_rows[0]["joint_qpos"],
        "eef_position_world_m": robot_rows[0]["eef_pos"],
        "eef_quaternion_wxyz": robot_rows[0]["eef_quat"],
        "gripper_qpos_m": robot_rows[0]["gripper_qpos"],
        "object_position_world_m": object_rows[0]["root_pos"],
        "object_quaternion_wxyz": object_rows[0]["root_quat"],
        "restored_gripper_current_action": runtime_robots[0]["gripper_current_action"],
        "integration_state_sha256": _sha(integration),
        "controller_initial_joint_rad": runtime_robots[0].get("controller", {}).get("initial_joint"),
        "prior_segment_path": str(prior_dir),
        "prior_segment_actions": int(result["cumulative_physical_actions_from_t463"]),
        "prior_segment_native_operational_prefix_verified": True,
        "prior_numeric_mismatch_diagnostic_only": {
            "component": next((name for prefix, name in (
                ("prediction_commit_eef_pos_mismatch:", "derived_robot0_eef_pos"),
                ("prediction_commit_eef_quat_mismatch:", "derived_robot0_eef_quat"),
                ("prediction_commit_Pot2_root_pos_mismatch:", "derived_Pot2_root_pos"),
                ("prediction_commit_Pot2_root_quat_mismatch:", "derived_Pot2_root_quat"),
            ) if prefix in stop_reason), "unknown"),
            "exception": stop_reason,
            "actual_predicted_runtime_snapshot_sha256_equal": True,
            "actual_predicted_Integration_State_exact": True,
            "current_action_replayed": False,
        },
        "prior_segment_actual_safety_audit": safety_audit,
    }
    return {
        "source_run": str(prior_dir),
        "snapshot_path": snapshot_path,
        "snapshot": snapshot,
        "attestation": attestation,
        "integration_sha256": _sha(integration),
        "prior_actions": int(result["cumulative_physical_actions_from_t463"]),
        "route_cursor": int(config["route_cursor_start"]),
        "prior_result": result,
    }


def _record_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _snapshot_id_and_current(bridge: BridgeProcess, backend: T463ExecutionBackend) -> tuple[str, dict[str, Any]]:
    observed = bridge.request({
        "op": "observe", "include_snapshot_state": True,
        "safety_geometry_policy": backend.safety_policy,
    })
    if observed.get("safety_geometry", {}).get("status") != "complete":
        raise RuntimeError("current_live_state_operational_geometry_is_not_complete")
    native = bridge.request({"op": "native_task_status"})
    contacts = bridge.request({"op": "inspect_contacts"})
    observed["native_task_status"] = native
    observed["contacts"] = contacts
    backend.latest_data = observed
    snapshot = bridge.request({"op": "snapshot"})
    if not isinstance(snapshot.get("snapshot_id"), str):
        raise RuntimeError("current_live_state_Snapshot_capture_failed")
    recorded = np.asarray(
        observed["snapshot_state"]["mujoco_integration_state"], dtype=np.float64
    )
    snapshot_state = np.asarray(
        snapshot.get("snapshot_state", {}).get("mujoco_integration_state"), dtype=np.float64
    )
    if recorded.shape != snapshot_state.shape or not np.array_equal(recorded, snapshot_state):
        raise RuntimeError("observed_Integration_State_and_captured_Snapshot_do_not_match")
    return snapshot["snapshot_id"], observed


def _preview_route_segment(
    bridge: BridgeProcess,
    backend: T463ExecutionBackend,
    context,
    current: dict[str, Any],
    source_snapshot_id: str,
    qpos_path: np.ndarray,
    eef_path: np.ndarray,
    risk_armed: bool,
) -> dict[str, Any]:
    index = int(backend.plan_cursor)
    if index >= len(qpos_path) - 1:
        raise RuntimeError("cached_t398_route_exhausted_before_live_transport_goal")
    integration = np.asarray(current["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
    response = bridge.request({
        "op": "osc_prefix_probe",
        "snapshot_id": source_snapshot_id,
        "current_state_sha256": _sha(integration),
        "jointspace_path_qpos": qpos_path.tolist(),
        "eef_reference_poses_world": eef_path.tolist(),
        "path_start_index": index,
        # Commit at most the first action from a certified route segment. The
        # next candidate is always generated from the newly observed real
        # state, so no stale multi-action prediction is executed after drift.
        "max_control_steps": min(1, len(qpos_path) - 1 - index),
        "gripper_action": 0.0,
        "risk_trigger_distance_m": RISK_TRIGGER_M,
        "risk_rearm_distance_m": RISK_REARM_M,
        "risk_crossing_armed": bool(risk_armed),
        "maximum_joint_path_deviation_rad": JOINT_REFERENCE_LIMIT_RAD,
        "maximum_eef_tracking_error_m": float(backend.config.maximum_eef_tracking_error_m),
        "maximum_eef_orientation_error_rad": float(backend.config.maximum_eef_orientation_error_rad),
        "qpos_reference_mode": "diagnostic_only",
        "current_minimum_forbidden_clearance_m": current["safety_geometry"].get("minimum_forbidden_clearance_m"),
        "safety_geometry_policy": backend.safety_policy,
    })
    if response.get("main_state_restored_verified") is not True:
        raise RuntimeError("route_preview_failed_to_restore_formal_state")
    if not response.get("complete") or not response.get("actions") or not response.get("steps"):
        raise RuntimeError("route_preview_did_not_produce_a_complete_safe_action_prefix")
    if response.get("qpos_reference_mode") != "diagnostic_only":
        raise RuntimeError("formal_route_preview_did_not_use_isolated_diagnostic_only_mode")
    if response.get("stop_reason") not in {None, "narrow_channel_entry_replan"}:
        raise RuntimeError(f"route_preview_stopped_for_tracking_or_progress:{response.get('stop_reason')}")
    if response.get("contact_monitoring", {}).get("status") != "complete":
        raise RuntimeError("route_preview_substep_monitoring_is_not_complete")
    if response.get("contact_monitoring", {}).get("geometry_monitoring", {}).get("status") != "complete":
        raise RuntimeError("route_preview_geometry_monitoring_is_not_complete")
    if not response.get("safe"):
        raise RuntimeError("route_preview_native_or_operational_safety_verdict_not_safe")
    if len(response["actions"]) != len(response["steps"]) or len(response["actions"]) > MAX_ROLLING_PREVIEW_ACTIONS:
        raise RuntimeError("route_preview_action_step_count_mismatch_or_limit_exceeded")
    if len(response["actions"]) != 1:
        raise RuntimeError("route_preview_must_certify_exactly_one_action_from_current_live_state")
    predicted: list[dict[str, Any]] = []
    for offset, step in enumerate(response["steps"]):
        expected_index = index + offset + 1
        if int(step.get("path_index", -1)) != expected_index:
            raise RuntimeError("route_preview_path_cursor_discontinuity")
        if float(step.get("eef_tracking_error_m", math.inf)) > backend.config.maximum_eef_tracking_error_m:
            raise RuntimeError("route_preview_EEF_position_tracking_exceeded_frozen_limit")
        if float(step.get("eef_orientation_error_rad", math.inf)) > backend.config.maximum_eef_orientation_error_rad:
            raise RuntimeError("route_preview_EEF_orientation_tracking_exceeded_frozen_limit")
        audit = _verify_one_branch_step(
            step, held_expected=True, phase=context.phase, backend=backend
        )
        native_and_live = _inspect_prediction(
            bridge, backend, source_snapshot_id, integration,
            step.get("terminal_runtime_snapshot"), backend.safety_policy,
            context.phase,
        )
        if native_and_live.get("predicted_held_object") != OBJECT_ID:
            raise RuntimeError("route_preview_live_grasp_state_did_not_confirm_Pot2")
        predicted.append({
            "path_index": expected_index,
            "action": step["action"],
            "step": step,
            "audit": audit,
            "native_and_live": native_and_live,
        })
    restored_id, restored = _snapshot_id_and_current(bridge, backend)
    del restored_id
    restored_integration = np.asarray(
        restored["snapshot_state"]["mujoco_integration_state"], dtype=np.float64
    )
    if not np.array_equal(restored_integration, integration):
        raise RuntimeError("formal_state_changed_after_route_preview_and_prediction_audit")
    return {"response": response, "predicted": predicted}


def _preview_expert_action(
    bridge: BridgeProcess,
    backend: T463ExecutionBackend,
    context,
    observation,
    current: dict[str, Any],
    source_snapshot_id: str,
    action: np.ndarray,
) -> dict[str, Any]:
    integration = np.asarray(current["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
    branch = bridge.request({
        "op": "branch_rollout",
        "snapshot_id": source_snapshot_id,
        "candidates": [{"candidate_id": f"t463_{context.phase}_{observation.timestep}", "actions": [action.tolist()]}],
        "safety_geometry_policy": backend.safety_policy,
        "capture_runtime_snapshots": True,
    })
    if branch.get("main_state_restored") is not True or len(branch.get("candidates", [])) != 1:
        raise RuntimeError("expert_action_branch_did_not_restore_or_return_one_candidate")
    candidate = branch["candidates"][0]
    if candidate.get("status") != "ok" or candidate.get("executed_steps") != 1 or len(candidate.get("steps", [])) != 1:
        raise RuntimeError(f"expert_action_branch_incomplete:{candidate.get('error')}")
    step = candidate["steps"][0]
    monitor = candidate.get("contact_monitoring", {})
    geometry = monitor.get("geometry_monitoring", {})
    if (
        monitor.get("status") != "complete"
        or monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
        or geometry.get("status") != "complete"
        or len(geometry.get("substeps", [])) != EXPECTED_SUBSTEPS
    ):
        raise RuntimeError("expert_action_branch_failed_25_substep_contact_or_geometry_coverage")
    held_expected = context.held_object == OBJECT_ID
    policy_ok, policy_audit = _verify_substep_policy(geometry["substeps"], held=held_expected)
    if not policy_ok:
        raise RuntimeError("expert_action_branch_failed_frozen_substep_policy:" + json.dumps(policy_audit, sort_keys=True))
    contacts = step.get("contacts", {})
    grasp = _finger_grasp_evidence(backend, contacts)
    if not grasp["all_robot_contacts_are_whitelisted_finger_to_pot2"]:
        raise RuntimeError("expert_action_branch_has_nonfinger_or_non_Pot2_robot_contact")
    if held_expected and context.phase != "release" and not grasp["grasp_confirmed"]:
        raise RuntimeError("expert_action_branch_lost_bilateral_Pot2_grasp")
    if bool(step.get("done")):
        raise RuntimeError("expert_action_branch_would_terminate_environment")
    payload = step.get("terminal_runtime_snapshot")
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError("expert_action_branch_did_not_export_terminal_runtime_snapshot")
    prediction = _inspect_prediction(
        bridge, backend, source_snapshot_id, integration, payload,
        backend.safety_policy, context.phase,
    )
    if context.phase == "release":
        if prediction.get("predicted_held_object") not in {None, OBJECT_ID}:
            raise RuntimeError("release_branch_reports_an_unexpected_held_object")
        # The controller enters release only after the precise-placement pose
        # goal is reached. Physical support contact is what makes the native
        # On predicate true, so the certified release transition may precede
        # that predicate. This action is still gated by the full branch checks
        # and the exact live source-state comparison below.
    if context.phase in {"withdrawal", "settle"} and (
        prediction.get("predicted_held_object") is not None
        or prediction.get("pot2_on") is not True
    ):
        raise RuntimeError("post_release_branch_lacks_unheld_native_On_state")
    phase_target = backend._task_stage_target(current, context.phase)
    before_robot = current["snapshot_state"]["robot_states"][0]
    before_eef = np.asarray(before_robot["eef_pos"], dtype=np.float64)
    before_quat = np.asarray(before_robot["eef_quat"], dtype=np.float64)
    after_state = step.get("state", {})
    after_robots = after_state.get("robot_states", [])
    if len(after_robots) == 1:
        after_eef = np.asarray(after_robots[0].get("eef_pos"), dtype=np.float64)
        after_quat = np.asarray(after_robots[0].get("eef_quat"), dtype=np.float64)
    else:
        after_obs_state = step.get("observation", {}).get("state", {})
        after_eef = np.asarray(after_obs_state.get("robot0_eef_pos"), dtype=np.float64)
        after_quat = np.asarray(after_obs_state.get("robot0_eef_quat"), dtype=np.float64)
    goal_error_before = float(np.linalg.norm(before_eef - phase_target["position"]))
    goal_error_after = float(np.linalg.norm(after_eef - phase_target["position"]))
    target_quat = np.asarray(phase_target["quaternion"], dtype=np.float64)
    orientation_error_before = 2.0 * math.acos(float(np.clip(
        abs(np.dot(before_quat / np.linalg.norm(before_quat), target_quat / np.linalg.norm(target_quat))),
        0.0, 1.0,
    )))
    orientation_error_after = 2.0 * math.acos(float(np.clip(
        abs(np.dot(after_quat / np.linalg.norm(after_quat), target_quat / np.linalg.norm(target_quat))),
        0.0, 1.0,
    )))
    if not np.isfinite([
        goal_error_before, goal_error_after,
        orientation_error_before, orientation_error_after,
    ]).all():
        raise RuntimeError("expert_action_branch_EEF_tracking_error_is_not_finite")
    return {
        "step": step,
        "payload": payload,
        "native_and_live": prediction,
        "substep_audit": policy_audit,
        "predicted_substeps": geometry["substeps"],
        "grasp": grasp,
        "phase_target_position_world_m": phase_target["position"].tolist(),
        "phase_goal_position_error_before_m": goal_error_before,
        "phase_goal_position_error_predicted_after_m": goal_error_after,
        "phase_goal_progress_m": goal_error_before - goal_error_after,
        "phase_goal_orientation_error_before_rad": orientation_error_before,
        "phase_goal_orientation_error_predicted_after_rad": orientation_error_after,
        "phase_goal_orientation_progress_rad": orientation_error_before - orientation_error_after,
    }


def _compare_branch_physics(predicted_step: dict[str, Any], actual_data: dict[str, Any]) -> dict[str, Any]:
    expected = predicted_step.get("observation", {}).get("snapshot_state")
    if not isinstance(expected, dict):
        expected = predicted_step.get("state", {})
    actual = actual_data.get("snapshot_state", {})
    exp_robots, act_robots = expected.get("robot_states", []), actual.get("robot_states", [])
    if len(exp_robots) != 1 or len(act_robots) != 1:
        raise RuntimeError("prediction_commit_robot_state_inventory_mismatch")
    errors: dict[str, float] = {}
    diagnostic_only_differences: dict[str, dict[str, Any]] = {}
    for key in ("joint_qpos", "eef_pos", "eef_quat", "gripper_qpos"):
        a = np.asarray(exp_robots[0].get(key), dtype=np.float64)
        b = np.asarray(act_robots[0].get(key), dtype=np.float64)
        if a.shape != b.shape:
            raise RuntimeError(f"prediction_commit_{key}_shape_mismatch")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"prediction_commit_{key}_nonfinite")
        errors[key] = float(np.max(np.abs(a - b))) if a.size else 0.0
        if key in {"eef_pos", "eef_quat"} and errors[key] > 1e-8:
            diagnostic_only_differences[key] = {
                "max_abs_component_error": errors[key],
                "classification": f"derived_EEF_{'position' if key == 'eef_pos' else 'quaternion'}_difference_diagnostic_only",
                "physical_state_gate": "joint_qpos_object_state_and_exact_post_action_Integration_State",
            }
        elif key not in {"eef_pos", "eef_quat"} and errors[key] > 1e-8:
            raise RuntimeError(f"prediction_commit_{key}_mismatch:{errors[key]}")
    expected_pot = _robot_objects(expected)
    actual_pot = _robot_objects(actual)
    for key in ("root_pos", "root_quat", "joint_qvel"):
        a = np.asarray(expected_pot.get(key), dtype=np.float64)
        b = np.asarray(actual_pot.get(key), dtype=np.float64)
        if a.shape != b.shape:
            raise RuntimeError(f"prediction_commit_Pot2_{key}_shape_mismatch")
        errors[f"pot2_{key}"] = float(np.max(np.abs(a - b))) if a.size else 0.0
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"prediction_commit_Pot2_{key}_nonfinite")
        if key in {"root_pos", "root_quat"} and errors[f"pot2_{key}"] > 1e-8:
            diagnostic_only_differences[f"pot2_{key}"] = {
                "max_abs_component_error": errors[f"pot2_{key}"],
                "classification": f"derived_Pot2_{'root_position' if key == 'root_pos' else 'root_quaternion'}_difference_diagnostic_only",
                "physical_state_gate": "exact_object_joint_state_and_exact_post_action_Integration_State",
            }
        elif key == "joint_qvel" and errors[f"pot2_{key}"] > 1e-8:
            raise RuntimeError(f"prediction_commit_Pot2_{key}_mismatch:{errors[f'pot2_{key}']}")
    return {
        "physical_state_component_max_abs_errors": errors,
        "diagnostic_only_numeric_differences": diagnostic_only_differences,
        "eef_position_comparison": "diagnostic_only; physical authorization requires exact Integration State and other state/safety checks",
        "non_eef_component_tolerance_unchanged": 1e-8,
    }


def _record_actual_route_reference(actual: dict[str, Any], qpos: np.ndarray, pose: np.ndarray, index: int) -> dict[str, Any]:
    robot = actual["snapshot_state"]["robot_states"][0]
    actual_q = np.asarray(robot["joint_qpos"], dtype=np.float64)
    actual_eef = np.asarray(robot["eef_pos"], dtype=np.float64)
    actual_quat = np.asarray(robot["eef_quat"], dtype=np.float64)
    q_delta = actual_q - qpos[index]
    dot = float(np.clip(abs(np.dot(actual_quat / np.linalg.norm(actual_quat), pose[index, 3:])), 0.0, 1.0))
    eef_rot = 2.0 * math.acos(dot)
    nearest = int(np.argmin(np.linalg.norm(qpos - actual_q[None, :], axis=1)))
    return {
        "path_index": index,
        "actual_qpos_rad": actual_q.tolist(),
        "reference_qpos_rad": qpos[index].tolist(),
        "per_joint_qpos_error_actual_minus_reference_rad": q_delta.tolist(),
        "per_joint_abs_qpos_error_rad": np.abs(q_delta).tolist(),
        "qpos_reference_l2_error_rad_diagnostic_only": float(np.linalg.norm(q_delta)),
        "exceeds_frozen_0_18_rad_diagnostic_only": bool(np.linalg.norm(q_delta) > JOINT_REFERENCE_LIMIT_RAD),
        "nearest_qpos_route_index_diagnostic_only": nearest,
        "actual_eef_position_world_m": actual_eef.tolist(),
        "eef_reference_pose_world": pose[index].tolist(),
        "eef_position_error_m": float(np.linalg.norm(actual_eef - pose[index, :3])),
        "eef_orientation_error_rad": eef_rot,
    }


def _parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=T398_TO_T463_RUN)
    parser.add_argument("--route-npz", type=Path)
    parser.add_argument("--safety-policy", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-output", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38749)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    __main__._PythonObjectState = _PythonObjectState
    source_run = Path(args.source_run)
    route_path = Path(args.route_npz) if args.route_npz else source_run / "replanned_route_t398.npz"
    policy_path = Path(args.safety_policy) if args.safety_policy else source_run / "replanned_route_safety_policy.json"
    qpos_path, eef_path, route_metadata, archived_policy = _parse_route(route_path, policy_path)
    prior_segment = Path(args.resume_from_output) if args.resume_from_output else None
    source = _source_from_prior_segment(prior_segment) if prior_segment else _source_463()
    if prior_segment is None and source["source_run"] != str(source_run):
        raise RuntimeError("strict_t463_source_snapshot_is_not_from_the_requested_source_run")
    start_timestep = int(source["attestation"]["timestep"])
    prior_actions = int(source.get("prior_actions", 0))
    remaining_action_budget = MAX_FORMAL_ACTIONS - prior_actions
    if remaining_action_budget <= 0:
        raise RuntimeError("cumulative_t463_450_action_budget_is_already_exhausted")
    route_cursor_start = int(source.get("route_cursor", T463_CURSOR))
    attestation = source["attestation"]
    config_data, recovery = load_runtime_config(CONFIG_PATH)
    config_data = json.loads(json.dumps(config_data))
    config_data["candidate_search"]["enabled"] = False
    # Rebuild RecoveryConfig from the exact archived object-specific policy.
    # The task JSON's generic defaults are not authoritative for this already
    # validated t398 route (Pot 2=2 mm, forbidden obstacles=10 mm).
    config_data["recovery"]["hard_minimum_clearance_m"] = float(
        archived_policy["minimum_clearance_m"]
    )
    config_data["recovery"]["object_clearance_m"] = {
        str(row["object_id"]): float(row["minimum_clearance_m"])
        for row in archived_policy["allowed_target_objects"]
    }
    config_data["recovery"]["target_object_clearance_m"] = min(
        config_data["recovery"]["object_clearance_m"].values()
    )
    config_data["recovery"]["execution_step_budget"] = remaining_action_budget
    config_data["recovery"]["horizon_steps"] = MAX_DEMONSTRATION_HORIZON
    config_data["recovery"]["standard_horizon_steps"] = STANDARD_HORIZON
    config_data["recovery"]["allow_horizon_extension"] = True
    recovery = recovery.__class__(**config_data["recovery"])
    recovery.validate()
    if recovery.maximum_joint_path_deviation_rad != JOINT_REFERENCE_LIMIT_RAD:
        raise RuntimeError("formal_frozen_qpos_reference_threshold_changed")
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    default_output = (
        f"/opt/data/private/gs_opd/rollouts/l0t2_execution_aware_pot2_resume_t{start_timestep}_{timestamp}"
        if prior_segment else
        f"/opt/data/private/gs_opd/rollouts/l0t2_round0_t463_execution_aware_pot2_placement_{timestamp}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(default_output)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "snapshots").mkdir()
    (output_dir / "predicted_branch_snapshots").mkdir()
    _write_json(output_dir / "run_config.json", {
        "entrypoint": str(Path(__file__).resolve()),
        "source_snapshot": attestation,
        "same_episode_continuation_from": str(prior_segment) if prior_segment else None,
        "source_action_count_from_t463": prior_actions,
        "remaining_cumulative_action_budget": remaining_action_budget,
        "route_npz": str(route_path),
        "route_npz_sha256": sha256_file(route_path),
        "route_qpos_hash": EXPECTED_ROUTE_QPOS_SHA256,
        "route_eef_hash": EXPECTED_ROUTE_EEF_SHA256,
        "route_metadata": route_metadata,
        "route_cursor_start": route_cursor_start,
        "qpos_reference_mode": "diagnostic_only_in_this_independent_formal_entrypoint",
        "frozen_qpos_reference_deviation_rad": JOINT_REFERENCE_LIMIT_RAD,
        "qpos_reference_authorizes_motion": False,
        "per_action_physical_branch_authorizes_motion": True,
        "maximum_rolling_preview_actions": MAX_ROLLING_PREVIEW_ACTIONS,
        "maximum_formal_actions_this_segment": remaining_action_budget,
        "maximum_cumulative_formal_actions_from_t463": MAX_FORMAL_ACTIONS,
        "maximum_task_timestep": MAX_DEMONSTRATION_HORIZON,
        "standard_benchmark_horizon": STANDARD_HORIZON,
        "safety_thresholds_changed": False,
        "rrt_calls_authorized": 0,
        "safety_policy_sha256": _canonical_sha256(archived_policy),
        "safety_policy": archived_policy,
        "task_phase_controller": "fresh_state_adoption_from_live_Pot2_grasp_and_native_predicates",
    })
    Path(output_dir / f"source_t{start_timestep}.pkl").write_bytes(Path(attestation["source_snapshot_path"]).read_bytes())
    Path(output_dir / "reused_route_t398.npz").write_bytes(route_path.read_bytes())
    _write_json(output_dir / "safety_policy.json", archived_policy)

    artifacts = RecoveryArtifactWriter(output_dir)
    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend: T463ExecutionBackend | None = None
    original_handlers = {}
    result: dict[str, Any] = {
        "status": "preflight_not_started",
        "physical_actions_committed": 0,
        "rrt_calls": 0,
        "complete_round0_seed": False,
        "verified_safe_success_suffix": False,
        "initial_timestep": start_timestep,
        "prior_physical_actions_from_t463": prior_actions,
        "remaining_cumulative_action_budget": remaining_action_budget,
        "maximum_physical_actions_from_t463": MAX_FORMAL_ACTIONS,
        "maximum_preview_actions": MAX_ROLLING_PREVIEW_ACTIONS,
    }
    pending_action_certificate: dict[str, Any] | None = None

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal_{signum}_received_at_t463_action_boundary")

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.signal(signum, interrupt)
    started = time.monotonic()
    try:
        bridge.start(
            benchmark=config_data["benchmark"], level=int(config_data["level"]),
            task_id=int(config_data["task_id"]), init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=MAX_DEMONSTRATION_HORIZON,
        )
        cached_route_stub = {"planner_result": {"planner_search_repeated": False}}
        backend = T463ExecutionBackend(
            bridge, config_data, recovery, output_dir, artifacts,
            mode="task-runtime-snapshot", student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=Path(attestation["source_snapshot_path"]),
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
            runtime_snapshot_attestation=attestation,
            journal_dir=output_dir / "durable_action_journal",
            resume_sources={}, cached_route=cached_route_stub,
        )
        restored = backend.restore_start_state()
        if int(restored.get("timestep", -1)) != start_timestep:
            raise RuntimeError("strict_resume_snapshot_timestep_mismatch")
        if restored.get("scene_metadata", {}).get("model_geometry_sha256") != MODEL_SHA256:
            raise RuntimeError("strict_t463_restore_model_fingerprint_mismatch")
        integration = np.asarray(restored["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
        if _sha(integration) != attestation["integration_state_sha256"]:
            raise RuntimeError("strict_t463_restore_Integration_State_fingerprint_mismatch")
        source_runtime = pickle.loads(Path(attestation["source_snapshot_path"]).read_bytes())["snapshot"]
        live_runtime_result = bridge.request({"op": "export_runtime_snapshot"})
        if live_runtime_result.get("status") != "complete":
            raise RuntimeError("resume_snapshot_runtime_internal_state_export_missing")
        live_runtime = pickle.loads(live_runtime_result["payload"])["snapshot"]
        if _normalized_state_hash(live_runtime) != _normalized_state_hash(source_runtime):
            raise RuntimeError("strict_resume_snapshot_controller_or_gripper_state_mismatch")

        preflight = bridge.request({"op": "observe", "include_snapshot_state": True})
        backend._load_context_inputs(preflight)
        backend.latest_data = preflight
        backend.safety_policy["joint_limit_tolerance_policy"] = archived_policy["joint_limit_tolerance_policy"]
        backend.safety_policy["source"] = archived_policy["source"]
        preflight["native_task_status"] = bridge.request({"op": "native_task_status"})
        preflight["contacts"] = bridge.request({"op": "inspect_contacts"})
        backend.latest_data = preflight
        context = backend.current_context()
        if context.held_object != OBJECT_ID or context.phase not in {"transport", "precise_placement"}:
            raise RuntimeError(f"live_task_adoption_mismatch:{context.phase}:{context.held_object}")
        resume_release_event = source.get("terminal_release_event")
        if resume_release_event is not None:
            if (
                int(resume_release_event.get("timestep", -1)) != start_timestep
                or resume_release_event.get("from_phase") != "precise_placement"
                or resume_release_event.get("to_phase") != "release"
                or resume_release_event.get("held_object") != OBJECT_ID
                or resume_release_event.get("native_pot2_on") is not False
            ):
                raise RuntimeError("archived_release_phase_transition_does_not_match_restored_source_state")
            # The simulator Snapshot omits Python TaskPhaseController
            # counters. Re-adopt only the phase proven by this exact
            # same-timestep durable event and the live grasp/native checks.
            backend.task_controller.phase = "release"
            backend.task_controller._expert_handoff = True
            backend.task_controller._phase_steps = 0
            backend.task_controller._last_timestep = -1
            backend.task_expert_active = True
            context = backend.current_context()
            if context.phase != "release" or context.held_object != OBJECT_ID:
                raise RuntimeError("verified_release_phase_re_adoption_failed_live_state_check")
            _record_jsonl(output_dir / "task_phase_events.jsonl", {
                "event": "task_phase_resume_adopted_from_verified_transition",
                "timestep": start_timestep,
                "from_phase": "precise_placement",
                "phase": "release",
                "held_object": OBJECT_ID,
                "native_pot2_on": _native_on(preflight["native_task_status"]),
                "source_event": resume_release_event,
                "controller_counters_reinitialized": True,
            })
        backend.safety_policy["source"] = archived_policy["source"]
        phase_policy_attestation = _validate_live_phase_policy(
            backend, archived_policy, context.phase
        )
        backend.task_expert_active = True
        grasp = context.metadata.get("held_object_state_evidence", {}).get(OBJECT_ID, {})
        contact_evidence = _finger_grasp_evidence(backend, preflight["contacts"])
        if (
            grasp.get("grasp_confirmed") is not True
            or grasp.get("finger_contacts", {}).get("left") is not True
            or grasp.get("finger_contacts", {}).get("right") is not True
            or not contact_evidence["grasp_confirmed"]
        ):
            raise RuntimeError("t463_restored_state_is_not_a_live_bilateral_finger_only_Pot2_grasp")
        geometry = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        }).get("safety_geometry", {})
        if geometry.get("status") != "complete" or any(
            geometry.get(key) for key in (
                "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
                "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
            )
        ):
            raise RuntimeError("t463_restored_state_fails_frozen_Native_or_Operational_safety")
        backend.latest_data = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        backend.latest_data["native_task_status"] = bridge.request({"op": "native_task_status"})
        backend.latest_data["contacts"] = bridge.request({"op": "inspect_contacts"})
        observation = backend._observation(backend.latest_data)
        context = backend.current_context()
        phase_policy_attestation = _validate_live_phase_policy(
            backend, archived_policy, context.phase
        )
        endpoint_position_error = None
        endpoint_orientation_error = None
        position_tolerance = None
        orientation_tolerance = None
        if context.phase == "transport":
            goal = backend.resolve_goal(observation, context)
            position_tolerance, orientation_tolerance = backend.task_controller.pose_goal_tolerances("transport")
            endpoint_position_error = float(np.linalg.norm(eef_path[-1, :3] - np.asarray(goal.position_world_m)))
            endpoint_orientation_error = 2.0 * math.acos(float(np.clip(
                abs(np.dot(eef_path[-1, 3:] / np.linalg.norm(eef_path[-1, 3:]),
                           np.asarray(goal.quaternion_wxyz) / np.linalg.norm(goal.quaternion_wxyz))), 0.0, 1.0
            )))
            if endpoint_position_error > position_tolerance or endpoint_orientation_error > orientation_tolerance:
                raise RuntimeError("archived_route_endpoint_is_outside_live_transport_goal_tolerance")
        elif context.phase not in {"precise_placement", "release"}:
            raise RuntimeError(f"restored_task_phase_is_not_supported_for_t463_suffix:{context.phase}")
        qpos_now = observation.qpos.copy()
        route_cursor_error = float(np.linalg.norm(qpos_now - qpos_path[route_cursor_start]))
        eef_cursor_error = float(np.linalg.norm(observation.eef_position_world_m - eef_path[route_cursor_start, :3]))
        if start_timestep == T463_TIMESTEP and abs(route_cursor_error - 0.18204520604490024) > 1e-5:
            raise RuntimeError(f"t463_route_qpos_diagnostic_reproduction_mismatch:{route_cursor_error}")
        backend.plan_cursor = route_cursor_start
        backend._plan_poses = eef_path.copy()
        backend.plan_counter = 1
        backend.initialize_trajectory()
        start_report = {
            "status": "strict_snapshot_restore_and_frozen_policy_preflight_passed",
            "timestep": start_timestep,
            "model_geometry_sha256": MODEL_SHA256,
            "integration_state_sha256": _sha(integration),
            "runtime_state_normalized_sha256": _normalized_state_hash(live_runtime),
            "safety_policy_sha256": _canonical_sha256(backend.safety_policy),
            "phase_policy_attestation": phase_policy_attestation,
            "task_phase": context.phase,
            "held_object": context.held_object,
            "live_grasp_evidence": grasp,
            "live_contact_evidence": contact_evidence,
            "operational_geometry": geometry,
            "route_cursor": route_cursor_start,
            "qpos_cursor_error_rad_diagnostic_only": route_cursor_error,
            "qpos_threshold_rad_unmodified": JOINT_REFERENCE_LIMIT_RAD,
            "eef_cursor_position_error_m": eef_cursor_error,
            "route_endpoint_live_transport_error_m_rad": [endpoint_position_error, endpoint_orientation_error],
            "transport_pose_tolerances_m_rad": [position_tolerance, orientation_tolerance],
            "same_episode_snapshot_resume": bool(prior_segment),
            "prior_action_count_from_t463": prior_actions,
            "actions_committed_before_preflight": 0,
        }
        _write_json(output_dir / "strict_restore_preflight.json", start_report)
        _record_jsonl(output_dir / "task_phase_events.jsonl", {
            "event": "t463_live_state_task_controller_adopted",
            "timestep": start_timestep, "phase": context.phase,
            "held_object": context.held_object,
        })
        result.update({
            "status": "executing_execution_aware_suffix",
            "preflight": start_report,
            "route_points": len(qpos_path),
            "route_cursor": backend.plan_cursor,
            "route_planner_search_repeated": False,
        })

        risk_armed = float(geometry.get("minimum_forbidden_clearance_m") or 0.0) >= RISK_REARM_M
        travel_window: list[float] = []
        previous_phase = context.phase
        stopped_reason = None
        while backend.transition_count < remaining_action_budget:
            if backend.latest_data is None:
                raise RuntimeError("formal_runner_has_no_live_state")
            if int(backend.latest_data.get("timestep", -1)) >= MAX_DEMONSTRATION_HORIZON:
                stopped_reason = "demonstration_horizon_1800_reached"
                break
            observation = backend.observe()
            current = backend.latest_data
            context = backend.current_context()
            if context.phase != previous_phase:
                _record_jsonl(output_dir / "task_phase_events.jsonl", {
                    "event": "task_phase_transition_observed",
                    "timestep": observation.timestep,
                    "from_phase": previous_phase, "to_phase": context.phase,
                    "held_object": context.held_object,
                    "native_pot2_on": _native_on(current.get("native_task_status", {})),
                })
                previous_phase = context.phase

            native = current.get("native_task_status", {})
            if native.get("status") != "complete" or _native_on(native) is None:
                raise RuntimeError("live_native_Pot2_On_predicate_is_unknown")
            if native.get("task_success") is True and _native_on(native) is not True:
                raise RuntimeError("native_task_success_report_conflicts_with_Pot2_On_predicate")
            if context.phase in {"transport", "precise_placement"} and context.held_object != OBJECT_ID:
                raise RuntimeError(f"live_Pot2_grasp_lost_before_release:{context.phase}")

            source_snapshot_id, current = _snapshot_id_and_current(bridge, backend)
            pre_integration = np.asarray(current["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
            action_predictions: list[dict[str, Any]] = []
            if context.phase == "transport":
                if backend.plan_cursor >= len(qpos_path) - 1:
                    # Re-evaluate the existing phase goal before declaring the
                    # archived route exhausted; never generate another route.
                    live_goal = backend.resolve_goal(observation, context)
                    reached = backend.is_recovery_goal_reached(observation, live_goal)
                    if not reached:
                        raise RuntimeError("cached_t398_route_ended_before_live_Pot2_transport_goal")
                    continue
                segment = _preview_route_segment(
                    bridge, backend, context, current, source_snapshot_id,
                    qpos_path, eef_path, risk_armed,
                )
                action_predictions = segment["predicted"]
                backend.artifacts.event({
                    "event": "t463_execution_aware_route_segment_certified",
                    "timestep": observation.timestep,
                    "path_start_index": backend.plan_cursor,
                    "certified_actions": len(action_predictions),
                    "qpos_reference_mode": "diagnostic_only",
                    "stop_reason": segment["response"].get("stop_reason"),
                })
            else:
                action = backend.task_expert_action(observation, context)
                if action is None:
                    raise RuntimeError(
                        f"task_phase_requires_RRT_but_RRT_is_prohibited:{context.phase}"
                    )
                action = np.asarray(action, dtype=np.float32)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise RuntimeError("existing_TaskPhaseController_action_is_not_finite_7D")
                pred = _preview_expert_action(
                    bridge, backend, context, observation, current,
                    source_snapshot_id, action,
                )
                action_predictions = [{"path_index": None, "action": action.tolist(), "expert": pred}]

            # The branch has restored the active physical state. Verify again
            # before submitting the certified action prefix.
            pre_commit = bridge.request({
                "op": "observe", "include_snapshot_state": True,
                "safety_geometry_policy": backend.safety_policy,
            })
            pre_commit_integration = np.asarray(
                pre_commit.get("snapshot_state", {}).get("mujoco_integration_state"),
                dtype=np.float64,
            )
            if pre_commit_integration.shape != pre_integration.shape or not np.array_equal(
                pre_commit_integration, pre_integration
            ):
                raise RuntimeError("active_physical_state_changed_between_branch_certificate_and_commit")
            if not action_predictions:
                raise RuntimeError("no_physically_certified_action_available_to_commit")

            for predicted in action_predictions:
                if backend.transition_count >= remaining_action_budget:
                    stopped_reason = "authorized_450_action_limit_reached"
                    break
                action = np.asarray(predicted["action"], dtype=np.float32)
                phase_before = context.phase
                held_expected_before_action = context.held_object == OBJECT_ID
                expected_snapshot_payload = (
                    predicted["step"].get("terminal_runtime_snapshot")
                    if phase_before == "transport"
                    else predicted["expert"]["payload"]
                )
                expected_predicted_status = (
                    predicted["native_and_live"]["native_task_status"]
                    if phase_before == "transport"
                    else predicted["expert"]["native_and_live"]["native_task_status"]
                )
                expected_contacts = (
                    predicted["step"].get("contacts", {})
                    if phase_before == "transport"
                    else predicted["expert"]["step"].get("contacts", {})
                )
                expected_raw_step = predicted.get("step") if phase_before == "transport" else predicted["expert"]["step"]
                expected_pre_state = pre_integration if not backend.transition_count else np.asarray(
                    backend.latest_data["snapshot_state"]["mujoco_integration_state"], dtype=np.float64
                )
                live_before = bridge.request({"op": "observe", "include_snapshot_state": True})
                live_before_state = np.asarray(live_before["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
                if not np.array_equal(live_before_state, expected_pre_state):
                    raise RuntimeError("rolling_segment_actual_pre_state_does_not_match_prior_prediction")
                predicted_timestep = int(live_before.get("timestep", -1)) + 1
                pred_path = output_dir / "predicted_branch_snapshots" / f"predicted_timestep_{predicted_timestep:06d}.pkl"
                # Make the full certified branch state durable before the matching
                # real action is submitted. If commit comparison fails, the
                # prediction and actual journal must both survive for diagnosis.
                pred_path.write_bytes(expected_snapshot_payload)
                pending_action_certificate = {
                    "schema": "gs_opd_prediction_commit_record.v1",
                    "status": "awaiting_physical_commit",
                    "pre_timestep": int(live_before.get("timestep", -1)),
                    "predicted_timestep": predicted_timestep,
                    "phase": phase_before,
                    "path_index": predicted.get("path_index"),
                    "action": action.tolist(),
                    "qpos_reference_mode": "diagnostic_only",
                    "qpos_reference_threshold_rad_unchanged": JOINT_REFERENCE_LIMIT_RAD,
                    "expected_runtime_snapshot_path": str(pred_path),
                    "expected_runtime_snapshot_sha256": hashlib.sha256(expected_snapshot_payload).hexdigest(),
                    "expected_native_task_status": expected_predicted_status,
                    "expected_contact_signature": _contacts_signature(expected_contacts),
                    "expected_substep_policy_audit": (
                        predicted.get("audit")
                        if phase_before == "transport"
                        else predicted["expert"].get("substep_audit")
                    ),
                    "comparison_limits": {
                        "integration_state_atol": 1e-9,
                        "robot_and_object_state_max_abs": 1e-8,
                        "contact_pair_set_exact": True,
                        "native_goal_predicates_exact": True,
                        "modified_for_this_run": False,
                    },
                    "actual_runtime_snapshot_path": None,
                    "actual_state": None,
                }
                prediction_record_path = output_dir / "prediction_commit_records" / f"prediction_t{predicted_timestep:06d}.json"
                _write_json(prediction_record_path, pending_action_certificate)
                pending_action_certificate["record_path"] = str(prediction_record_path)
                if phase_before == "transport":
                    backend._current_action_source = "safety_recovery_osc_pose"
                backend.current_forecast = action[None, :].copy()
                committed_observation = backend.execute_action(action, context)
                actual = backend.latest_data
                if actual is None:
                    raise RuntimeError("committed_action_has_no_actual_Bridge_response")
                monitor = actual.get("contact_monitoring", {})
                geometry_monitor = monitor.get("geometry_monitoring", {})
                actual_substeps = geometry_monitor.get("substeps", [])
                actual_state = actual.get("snapshot_state", {})
                actual_robot_states = actual_state.get("robot_states", [])
                actual_objects = actual_state.get("object_states", {})
                actual_pot2_rows = [
                    value for key, value in actual_objects.items()
                    if str(key).endswith(":" + OBJECT_ID)
                ]
                pending_action_certificate["actual_state"] = {
                    "timestep": int(actual.get("timestep", -1)),
                    "robot0": actual_robot_states[0] if len(actual_robot_states) == 1 else None,
                    "pot2": actual_pot2_rows[0] if len(actual_pot2_rows) == 1 else None,
                    "native_task_status": actual.get("native_task_status"),
                    "reward": actual.get("reward"),
                    "done": actual.get("done"),
                    "contact_monitoring_status": monitor.get("status"),
                    "captured_physics_substeps": monitor.get("captured_physics_substeps"),
                    "geometry_monitoring_status": geometry_monitor.get("status"),
                    "geometry_substeps": len(actual_substeps),
                    "contacts": _contacts_signature(actual.get("contacts", {})),
                    "per_step_safety_summary": _verify_substep_policy(
                        actual_substeps, held=held_expected_before_action
                    )[1] if len(actual_substeps) == EXPECTED_SUBSTEPS else None,
                }
                pending_action_certificate["actual_runtime_snapshot_path"] = actual.get("snapshot_path")
                _write_json(prediction_record_path, pending_action_certificate)
                held_expected = held_expected_before_action
                actual_safe, actual_audit = _verify_substep_policy(actual_substeps, held=held_expected)
                if (
                    monitor.get("status") != "complete"
                    or monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
                    or geometry_monitor.get("status") != "complete"
                    or not actual_safe
                ):
                    raise RuntimeError("committed_action_failed_full_25_substep_Native_or_Operational_safety")
                actual_contacts = actual.get("contacts", {})
                actual_grasp = _finger_grasp_evidence(backend, actual_contacts)
                if not actual_grasp["all_robot_contacts_are_whitelisted_finger_to_pot2"]:
                    raise RuntimeError("committed_action_created_nonfinger_or_non_Pot2_robot_contact")
                if held_expected and phase_before != "release" and not actual_grasp["grasp_confirmed"]:
                    raise RuntimeError(f"committed_action_lost_bilateral_Pot2_grasp:{phase_before}")
                runtime_actual = bridge.request({"op": "export_runtime_snapshot"})
                if runtime_actual.get("status") != "complete":
                    raise RuntimeError("post_commit_runtime_snapshot_export_failed")
                commit_physics = _compare_branch_physics(expected_raw_step, actual)
                commit_full = _compare_prediction_to_commit(
                    expected_snapshot_payload, actual, expected_predicted_status,
                    expected_raw_step, runtime_actual["payload"], expected_contacts, actual_contacts,
                )
                pending_action_certificate["status"] = "prediction_commit_verified"
                pending_action_certificate["prediction_commit_physics"] = commit_physics
                pending_action_certificate["prediction_commit_runtime"] = commit_full
                _write_json(prediction_record_path, pending_action_certificate)
                if phase_before == "transport":
                    ref = _record_actual_route_reference(
                        actual, qpos_path, eef_path, int(predicted["path_index"])
                    )
                else:
                    ref = None
                record = {
                    "event": "action_prediction_commit_verified",
                    "pre_timestep": int(live_before.get("timestep", -1)),
                    "timestep": int(actual.get("timestep", -1)),
                    "phase_before": phase_before,
                    "phase_after_pending_update": None,
                    "action": action.tolist(),
                    "path_index": predicted.get("path_index"),
                    "qpos_reference": ref,
                    "predicted_native_task_status": expected_predicted_status,
                    "actual_native_task_status": actual.get("native_task_status"),
                    "predicted_safety_substeps": (
                        predicted["step"].get("safety_geometry_substeps")
                        if phase_before == "transport"
                        else predicted["expert"].get("predicted_substeps")
                    ),
                    "actual_safety_substeps": actual_substeps,
                    "actual_safety_summary": actual_audit,
                    "predicted_grasp": (
                        predicted.get("audit", {}).get("grasp")
                        if phase_before == "transport"
                        else predicted["expert"].get("grasp")
                    ),
                    "actual_grasp": actual_grasp,
                    "prediction_commit_physics": commit_physics,
                    "prediction_commit_runtime": commit_full,
                    "actual_reward": actual.get("reward"),
                    "actual_done": actual.get("done"),
                    "pot2_native_on": _native_on(actual.get("native_task_status", {})),
                    "runtime_snapshot_path": actual.get("snapshot_path"),
                }
                if ref is not None:
                    _record_jsonl(output_dir / "qpos_reference_diagnostics.jsonl", {
                        "timestep": int(actual.get("timestep", -1)), **ref,
                    })

                eef_before = np.asarray(live_before["snapshot_state"]["robot_states"][0]["eef_pos"], dtype=np.float64)
                eef_after = np.asarray(actual["snapshot_state"]["robot_states"][0]["eef_pos"], dtype=np.float64)
                travel_window.append(float(np.linalg.norm(eef_after - eef_before)))
                if len(travel_window) > recovery.stagnation_window_steps:
                    travel_window.pop(0)
                clearance = actual.get("safety_geometry", {}).get("minimum_forbidden_clearance_m")
                risk_event = None
                if clearance is not None:
                    if not risk_armed and float(clearance) >= RISK_REARM_M:
                        risk_armed = True
                        risk_event = "risk_boundary_rearmed_at_60mm"
                    elif risk_armed and float(clearance) <= RISK_TRIGGER_M:
                        risk_armed = False
                        risk_event = "risk_boundary_entered_at_50mm"

                next_observation = backend._observation(actual)
                old_phase = phase_before
                context = backend.current_context()
                if context.phase != old_phase:
                    record["phase_after_pending_update"] = context.phase
                    _record_jsonl(output_dir / "task_phase_events.jsonl", {
                        "event": "task_phase_transition_observed",
                        "timestep": next_observation.timestep,
                        "from_phase": old_phase, "to_phase": context.phase,
                        "held_object": context.held_object,
                        "native_pot2_on": _native_on(actual.get("native_task_status", {})),
                    })
                    previous_phase = context.phase
                else:
                    record["phase_after_pending_update"] = context.phase
                _record_jsonl(output_dir / "execution_certification.jsonl", record)
                pending_action_certificate = None
                if risk_event:
                    _record_jsonl(output_dir / "risk_boundary_events.jsonl", {
                        "event": risk_event, "timestep": int(actual.get("timestep", -1)),
                        "clearance_m": clearance, "hysteresis_m": [RISK_TRIGGER_M, RISK_REARM_M],
                        "rrt_launched": False,
                    })
                result["physical_actions_committed"] = backend.transition_count
                result["final_timestep"] = int(actual.get("timestep", -1))
                result["current_phase"] = context.phase
                result["path_cursor"] = backend.plan_cursor

                milestone = backend.task_milestone_reached(next_observation)
                if milestone:
                    result["pot2_local_milestone"] = milestone
                    stopped_reason = "pot2_native_on_and_settled"
                    break
                if actual.get("task_success") is True:
                    result["unexpected_full_native_success"] = True
                    stopped_reason = "native_task_success"
                    break
                if actual.get("done") is True:
                    stopped_reason = "environment_done"
                    break
                if (
                    len(travel_window) >= recovery.stagnation_window_steps
                    and sum(travel_window) < recovery.minimum_progress_m
                    and context.phase in {"transport", "precise_placement"}
                ):
                    stopped_reason = f"physical_stagnation_fail_closed:{context.phase}"
                    raise RuntimeError(stopped_reason)
                if context.phase != phase_before:
                    # The remaining controls in a transport prefix were
                    # generated under the prior stage. Re-preview from the
                    # new physical state under precise placement.
                    break
            if stopped_reason in {"pot2_native_on_and_settled", "native_task_success", "environment_done", "authorized_450_action_limit_reached"}:
                break
        if stopped_reason is None:
            stopped_reason = "authorized_450_action_limit_reached"
        result["stop_reason"] = stopped_reason
        result["status"] = "completed_Pot2_local_safe_suffix" if stopped_reason == "pot2_native_on_and_settled" else "stopped_fail_closed"
        terminal = backend.observe()
        terminal_context = backend.current_context()
        terminal_native = terminal.raw.get("native_task_status", {})
        on_value = _native_on(terminal_native)
        suffix = _suffix_acceptance(backend.trajectory, backend.safety_policy)
        result.update({
            "initial_timestep": start_timestep,
            "final_timestep": terminal.timestep,
            "physical_actions_committed": backend.transition_count,
            "cumulative_physical_actions_from_t463": prior_actions + backend.transition_count,
            "prior_segment_verified_under_current_native_operational_monitor": (
                source["attestation"].get("prior_segment_native_operational_prefix_verified")
                if prior_segment else None
            ),
            "current_phase": terminal_context.phase,
            "held_object": terminal_context.held_object,
            "native_task_status": terminal_native,
            "pot2_native_on": on_value,
            "native_task_success": terminal.task_success,
            "pot2_local_placement_success": bool(on_value is True and result.get("pot2_local_milestone")),
            "suffix_native_safety": suffix.get("native_safety"),
            "suffix_operational_safety": suffix.get("operational_safety"),
            "suffix_strict_10mm": suffix.get("strict_10mm"),
            "suffix_acceptance": suffix,
            "verified_safe_success_suffix": bool(
                on_value is True and result.get("pot2_local_milestone")
                and suffix.get("native_safety") is True
                and suffix.get("operational_safety") is True
                and (
                    prior_segment is None
                    or source["attestation"].get("prior_segment_native_operational_prefix_verified") is True
                )
            ),
            "complete_round0_seed": False,
            "pot2_local_milestone_within_standard_horizon": bool(
                on_value is True and result.get("pot2_local_milestone")
                and terminal.timestep <= STANDARD_HORIZON
            ),
            "pot2_local_milestone_within_demonstration_horizon": bool(
                on_value is True and result.get("pot2_local_milestone")
                and terminal.timestep <= MAX_DEMONSTRATION_HORIZON
            ),
            "standard_horizon_success": bool(
                terminal.task_success is True and terminal.timestep <= STANDARD_HORIZON
            ),
            "demonstration_horizon_success": bool(
                terminal.task_success is True and terminal.timestep <= MAX_DEMONSTRATION_HORIZON
            ),
            "route_path_points": len(qpos_path),
            "route_final_cursor": backend.plan_cursor,
            "route_qpos_path_sha256": EXPECTED_ROUTE_QPOS_SHA256,
            "route_eef_path_sha256": EXPECTED_ROUTE_EEF_SHA256,
            "planner_search_repeated": False,
            "rrt_calls": 0,
            "risk_hysteresis_m": [RISK_TRIGGER_M, RISK_REARM_M],
            "qpos_reference_mode": "diagnostic_only",
            "qpos_reference_threshold_rad_unchanged": JOINT_REFERENCE_LIMIT_RAD,
            "source_prefix_safety_under_current_policy": False,
            "same_episode_snapshot_continuation_from": str(prior_segment) if prior_segment else None,
            "required_for_complete_round0_seed": "replay_and_audit_one_continuous_safe_episode_from_verified_raw_t10_snapshot",
            "outputs": {
                "observation_action_trace": str(output_dir / "observation_action_trace.npz"),
                "execution_certification": str(output_dir / "execution_certification.jsonl"),
                "qpos_reference_diagnostics": str(output_dir / "qpos_reference_diagnostics.jsonl"),
                "snapshots": str(output_dir / "snapshots"),
                "predicted_branch_snapshots": str(output_dir / "predicted_branch_snapshots"),
                "agentview_video": str(output_dir / "agentview.mp4"),
                "wrist_video": str(output_dir / "wrist.mp4"),
            },
        })
        return result
    except BaseException as exc:
        if pending_action_certificate is not None:
            pending_action_certificate["status"] = "physical_commit_stopped_before_certification"
            pending_action_certificate["failure_reason"] = f"{type(exc).__name__}: {exc}"
            pending_action_certificate["failure_timestep"] = (
                int(backend.latest_data.get("timestep", -1))
                if backend and backend.latest_data else None
            )
            record_path = pending_action_certificate.get("record_path")
            if record_path:
                _write_json(Path(record_path), pending_action_certificate)
            if str(exc).startswith(("prediction_commit_", "RuntimeError: prediction_commit_")):
                _write_json(output_dir / "prediction_commit_mismatch_stop.json", {
                    "status": "stopped_fail_closed_on_prediction_commit_mismatch",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "failure_timestep": pending_action_certificate.get("failure_timestep"),
                    "pending_prediction_record": record_path,
                    "predicted_runtime_snapshot_path": pending_action_certificate.get("expected_runtime_snapshot_path"),
                    "predicted_runtime_snapshot_sha256": pending_action_certificate.get("expected_runtime_snapshot_sha256"),
                    "actual_runtime_snapshot_path": pending_action_certificate.get("actual_runtime_snapshot_path"),
                    "actual_state": pending_action_certificate.get("actual_state"),
                    "further_actions_committed": False,
                    "qpos_reference_threshold_rad_unchanged": JOINT_REFERENCE_LIMIT_RAD,
                    "safety_thresholds_changed": False,
                    "rrt_calls": 0,
                })
        result.update({
            "status": "interrupted_or_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "physical_actions_committed": backend.transition_count if backend else 0,
            "cumulative_physical_actions_from_t463": prior_actions + (backend.transition_count if backend else 0),
            "failure_timestep": int(backend.latest_data.get("timestep", -1)) if backend and backend.latest_data else None,
            "failure_phase": getattr(backend.task_controller, "phase", None) if backend and backend.task_controller else None,
            "complete_round0_seed": False,
            "verified_safe_success_suffix": False,
        })
        return result
    finally:
        if backend is not None:
            try:
                acceptance = _suffix_acceptance(backend.trajectory, backend.safety_policy) if hasattr(backend, "trajectory") else None
                backend.finalize({
                    "complete_task_success": False,
                    "complete_safety_success": bool(result.get("verified_safe_success_suffix")),
                    "local_recovery_success": bool(result.get("pot2_local_placement_success")),
                    "standard_horizon_success": bool(result.get("standard_horizon_success")),
                    "extended_horizon": int(result.get("final_timestep", start_timestep)) > STANDARD_HORIZON,
                    "result": result,
                    "suffix_acceptance": acceptance,
                })
            except Exception as exc:
                (output_dir / "finalize_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        bridge.close()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        result["output_dir"] = str(output_dir)
        result["elapsed_seconds"] = time.monotonic() - started
        _write_json(output_dir / "suffix_result.json", result)


def main() -> int:
    args = _parser()
    result = run(args)
    print(json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.get("verified_safe_success_suffix") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
