# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Continue Pot 1 grasp/lift from t1042 with the verified Pot 2 target margin.

The first native OSC action must match the previously rejected t1042 action,
which is independently re-previewed under the object-specific policy before
this entrypoint is allowed to commit any physical action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.repair import continue_t1019_pot1_execution_aware_grasp_lift as runner


REPO_ROOT = Path(__file__).resolve().parents[3]
EXECUTION_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/l0t1019_pot1_grasp_lift_20261025_030744"
)
PREFIX_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/l0t796_pot1_execution_aware_pregrasp_20261025_021845"
)
T1042_SNAPSHOT = EXECUTION_ARCHIVE / "snapshots/main_timestep_001042.pkl"
TRACE = EXECUTION_ARCHIVE / "observation_action_trace.npz"
JOURNAL_INDEX = EXECUTION_ARCHIVE / "durable_action_journal/index.jsonl"
EXECUTION_RESULT = EXECUTION_ARCHIVE / "suffix_result.json"
EXECUTION_FAILURE = EXECUTION_ARCHIVE / "failure_report.json"
PREVIEW_GLOB = "l0t1042_pot1_policy_reentry_preview_*"
MODEL_SHA256 = runner.MODEL_SHA256
TIMESTEP = 1042
EXPECTED_TIMESTEPS = np.arange(1020, 1043, dtype=np.int64)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _policy_with_pot2_target_margin(prior: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = json.loads(json.dumps(prior["frozen_operational_safety_policy"]))
    targets = {str(row["object_id"]): row for row in policy["allowed_target_objects"]}
    forbidden = {str(row["object_id"]): float(row["minimum_clearance_m"])
                 for row in policy["forbidden_objects"]}
    if targets.get("moka_pot_1", {}).get("minimum_clearance_m") != 0.010:
        raise RuntimeError("source_Pot1_target_margin_is_not_the_archived_10mm_value")
    if targets.get("moka_pot_2", {}).get("minimum_clearance_m") != 0.002:
        raise RuntimeError("source_Pot2_target_margin_is_not_the_verified_2mm_value")
    if forbidden != {
        "bottle_of_ginger_beer_1": 0.010,
        "white_place_box_1": 0.010,
    }:
        raise RuntimeError("source_forbidden_object_margins_differ_from_frozen_10mm_policy")
    targets["moka_pot_1"]["minimum_clearance_m"] = float(
        targets["moka_pot_2"]["minimum_clearance_m"]
    )
    metadata = {
        "entrypoint": str(Path(__file__).resolve()),
        "policy_derivation": "Pot1_nonfinger_target_clearance_inherits_verified_Pot2_object_specific_clearance",
        "source_policy_sha256": prior.get("frozen_operational_safety_policy_sha256"),
        "Pot1_nonfinger_minimum_clearance_m": 0.002,
        "Pot2_nonfinger_minimum_clearance_m": 0.002,
        "GingerBeer_minimum_clearance_m": 0.010,
        "PlaceBox_minimum_clearance_m": 0.010,
        "global_hard_minimum_clearance_m": float(policy["minimum_clearance_m"]),
        "only_four_verified_finger_geoms_may_contact_Pot1": True,
        "allowed_finger_penetration_m": 0.002,
        "palm_or_other_robot_penetration_allowed": False,
        "self_collision_joint_limit_and_native_BDDL_rules_unchanged": True,
        "millimeter_threshold_search": False,
    }
    return policy, metadata


def _select_passed_preview(snapshot_sha: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(Path("/opt/data/private/gs_opd/rollouts").glob(PREVIEW_GLOB))
    for archive in reversed(candidates):
        report_path = archive / "rejected_action_preview_audit.json"
        manifest_path = archive / "policy_reentry_preview_manifest.json"
        if not report_path.is_file() or not manifest_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            report.get("safe") is True
            and report.get("physical_action_committed") is False
            and report.get("source_timestep") == TIMESTEP
            and report.get("source_snapshot_sha256") == snapshot_sha
            and manifest.get("physical_action_committed") is False
            and manifest.get("policy_adjustment", {}).get("applied") is True
        ):
            return archive, report
    raise RuntimeError("no_safe_isolated_t1042_preview_under_Pot2_object_margin_policy")


def _attest_t1042() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not all(path.is_file() for path in (
        T1042_SNAPSHOT, TRACE, JOURNAL_INDEX, EXECUTION_RESULT,
        EXECUTION_FAILURE, PREFIX_ARCHIVE / "run_config.json",
        PREFIX_ARCHIVE / "observation_action_trace.npz",
        PREFIX_ARCHIVE / "recovery_events.jsonl",
    )):
        raise FileNotFoundError("t1042_snapshot_or_contiguous_episode_evidence_missing")

    prior = json.loads((PREFIX_ARCHIVE / "run_config.json").read_text(encoding="utf-8"))
    policy, policy_metadata = _policy_with_pot2_target_margin(prior)
    snapshot_bytes = T1042_SNAPSHOT.read_bytes()
    envelope = pickle.loads(snapshot_bytes)
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != MODEL_SHA256
    ):
        raise RuntimeError("t1042_snapshot_schema_or_model_fingerprint_mismatch")
    snapshot = envelope["snapshot"]
    integration = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    if integration.shape != (317,) or not np.isfinite(integration).all():
        raise RuntimeError("t1042_Integration_State_invalid")
    if int(snapshot.get("task_bookkeeping", {}).get("timestep", -1)) != TIMESTEP:
        raise RuntimeError("t1042_snapshot_internal_timestep_mismatch")
    robots = snapshot.get("robots", [])
    if len(robots) != 1 or not robots[0].get("has_gripper_current_action"):
        raise RuntimeError("t1042_robot_or_gripper_runtime_state_missing")

    result = json.loads(EXECUTION_RESULT.read_text(encoding="utf-8"))
    failure = json.loads(EXECUTION_FAILURE.read_text(encoding="utf-8"))
    if (
        result.get("failure_timestep") != TIMESTEP
        or result.get("physical_actions_committed") != 23
        or failure.get("failure_timestep") != TIMESTEP
        or failure.get("physical_actions_committed") != 23
    ):
        raise RuntimeError("t1042_is_not_the_terminal_state_after_the_attested_23_action_prefix")

    with np.load(TRACE, allow_pickle=False) as trace, np.load(
        PREFIX_ARCHIVE / "observation_action_trace.npz", allow_pickle=False
    ) as prefix:
        if (
            trace["executed_actions"].shape != (23, 7)
            or not np.array_equal(trace["step_timesteps"], EXPECTED_TIMESTEPS)
            or int(prefix["step_timesteps"][-1]) != 1019
            or len(trace["post_physical_state_json"]) != 23
        ):
            raise RuntimeError("t1019_to_t1042_OA_prefix_is_not_contiguous")
        if not np.array_equal(
            np.asarray(trace["decision_robot0_eef_pos"][0], dtype=np.float64),
            np.asarray(prefix["post_robot0_eef_pos"][-1], dtype=np.float64),
        ) or not np.array_equal(
            np.asarray(trace["decision_robot0_eef_quat"][0], dtype=np.float64),
            np.asarray(prefix["post_robot0_eef_quat"][-1], dtype=np.float64),
        ):
            raise RuntimeError("t1019_to_t1020_observation_boundary_mismatch")
        physical = json.loads(str(trace["post_physical_state_json"][-1]))
        native = json.loads(str(trace["post_native_task_status_json"][-1]))
        terminal_monitor = json.loads(str(trace["post_contact_monitoring_json"][-1]))
        if not np.array_equal(
            np.asarray(physical.get("mujoco_integration_state"), dtype=np.float64), integration
        ):
            raise RuntimeError("t1042_snapshot_and_terminal_OA_Integration_State_mismatch")
        camera_shape_ok = all(
            trace[key].shape[0] == 23 and trace[key].size > 0
            for key in (
                "decision_agentview_image", "post_agentview_image",
                "decision_robot0_eye_in_hand_image", "post_robot0_eye_in_hand_image",
            )
        )
        if not camera_shape_ok:
            raise RuntimeError("t1042_prefix_dual_camera_OA_evidence_missing")
        for index in range(23):
            monitor = json.loads(str(trace["post_contact_monitoring_json"][index]))
            substeps = monitor.get("geometry_monitoring", {}).get("substeps", [])
            if (
                monitor.get("status") != "complete"
                or monitor.get("captured_physics_substeps") != runner.EXPECTED_SUBSTEPS
                or monitor.get("geometry_monitoring", {}).get("status") != "complete"
                or len(substeps) != runner.EXPECTED_SUBSTEPS
            ):
                raise RuntimeError(f"t{int(trace['step_timesteps'][index])}_substep_audit_incomplete")
            runner._check_substeps(substeps, policy)
            row_native = json.loads(str(trace["post_native_task_status_json"][index]))
            runner._validate_native_stage(row_native, context=f"archived_t{int(trace['step_timesteps'][index])}")
        if terminal_monitor.get("status") != "complete":
            raise RuntimeError("t1042_terminal_action_monitor_is_incomplete")

    journal_rows = [
        json.loads(line) for line in JOURNAL_INDEX.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(journal_rows) != 23:
        raise RuntimeError("t1042_durable_action_journal_does_not_have_23_rows")
    terminal_rows = [row for row in journal_rows if int(row.get("timestep", -1)) == TIMESTEP]
    if len(terminal_rows) != 1:
        raise RuntimeError("t1042_durable_terminal_journal_row_missing")
    terminal_row = terminal_rows[0]
    journal_path = Path(terminal_row["journal_path"])
    if (
        Path(terminal_row.get("runtime_snapshot_path", "")) != T1042_SNAPSHOT
        or _sha256(journal_path.read_bytes()) != terminal_row.get("journal_sha256")
    ):
        raise RuntimeError("t1042_snapshot_and_durable_action_journal_mismatch")
    for index, row in enumerate(journal_rows):
        if int(row.get("timestep", -1)) != int(EXPECTED_TIMESTEPS[index]):
            raise RuntimeError("t1042_action_journal_timestep_sequence_mismatch")

    snapshot_sha = _sha256(snapshot_bytes)
    preview_archive, preview = _select_passed_preview(snapshot_sha)
    preview_action = np.asarray(preview.get("action"), dtype=np.float32)
    if (
        preview_action.shape != (7,)
        or not np.allclose(preview_action, runner.EXPECTED_FIRST_ACTION or preview_action, rtol=0.0, atol=1e-7)
    ):
        raise RuntimeError("t1042_passed_preview_action_is_malformed")
    preview_manifest = json.loads(
        (preview_archive / "policy_reentry_preview_manifest.json").read_text(encoding="utf-8")
    )
    if preview_manifest.get("policy_adjustment", {}).get("Pot1_nonfinger_minimum_clearance_m") != 0.002:
        raise RuntimeError("t1042_preview_did_not_use_exact_Pot2_target_clearance")

    native_values = runner._native_values(native)
    if native_values != {runner.POT1_ON: False, runner.POT2_ON: True, runner.STOVE_ON: True}:
        raise RuntimeError("t1042_native_predicates_do_not_match_Pot1_approach_stage")
    if native.get("task_success") is not False or native.get("done") is not False:
        raise RuntimeError("t1042_native_episode_unexpectedly_terminated")

    events = [
        json.loads(line)
        for line in (PREFIX_ARCHIVE / "recovery_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    transitions = [
        row for row in events
        if row.get("event") == "task_phase_transition"
        and int(row.get("timestep", -1)) == 1019
        and row.get("to_phase") == "approach"
        and row.get("object_id") == "moka_pot_1"
    ]
    if len(transitions) != 1:
        raise RuntimeError("t1019_approach_phase_transition_evidence_missing")
    robot = physical["robot_states"][0]
    pot1 = runner._object_state(physical, "moka_pot_1")
    pot2 = runner._object_state(physical, "moka_pot_2")
    attestation = {
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(T1042_SNAPSHOT),
        "source_snapshot_sha256": snapshot_sha,
        "source_action_record": str(TRACE),
        "source_action_record_sha256": _sha256(TRACE.read_bytes()),
        "timestep": TIMESTEP,
        "model_geometry_sha256": MODEL_SHA256,
        "object_id": "moka_pot_1",
        "arm_joint_qpos": robot["joint_qpos"],
        "eef_position_world_m": robot["eef_pos"],
        "eef_quaternion_wxyz": robot["eef_quat"],
        "gripper_qpos_m": robot["gripper_qpos"],
        "restored_gripper_current_action": robots[0].get("gripper_current_action"),
        "integration_state_sha256": hashlib.sha256(integration.tobytes()).hexdigest(),
        "runtime_snapshot_normalized_sha256": runner._normalized_state_sha256(snapshot),
        "object_position_world_m": pot1["root_pos"],
        "object_quaternion_wxyz": pot1["root_quat"],
        "source_pot2_position_world_m": pot2["root_pos"],
        "native_status": native,
        "phase_transition_event": transitions[0],
        "preapproved_action_preview_archive": str(preview_archive),
        "preapproved_native_action": preview_action.tolist(),
        "policy_adjustment": policy_metadata,
    }
    source = {
        "snapshot": snapshot,
        "integration_state": integration,
        "physical": physical,
        "native": native,
        "monitor": terminal_monitor,
        "policy": policy,
        "phase_event": transitions[0],
    }
    runner.EXPECTED_FIRST_ACTION = preview_action.tolist()
    runner.RUN_POLICY_METADATA = policy_metadata
    return attestation, source, prior


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=runner.CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=runner.DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38756)
    return parser.parse_args()


def main() -> int:
    runner.SOURCE_ARCHIVE = EXECUTION_ARCHIVE
    runner.SOURCE_SNAPSHOT = T1042_SNAPSHOT
    runner.SOURCE_TRACE = TRACE
    runner.SOURCE_EVENTS = PREFIX_ARCHIVE / "recovery_events.jsonl"
    runner.SOURCE_RESULT = EXECUTION_RESULT
    runner.SOURCE_CONFIG = PREFIX_ARCHIVE / "run_config.json"
    runner.SOURCE_JOURNAL_INDEX = JOURNAL_INDEX
    runner.INITIAL_TIMESTEP = TIMESTEP
    runner._source_attestation = _attest_t1042
    result = runner.run(_parse_args())
    print(json.dumps(runner._jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "pot1_stable_grasp_and_lift_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
