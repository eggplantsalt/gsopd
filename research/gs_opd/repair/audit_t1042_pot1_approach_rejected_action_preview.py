# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Strictly restore t1042 and diagnose, without committing, its rejected action."""

from __future__ import annotations

import __main__
import argparse
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from research.gs_opd.repair.automatic_recovery import RecoveryArtifactWriter
from research.gs_opd.repair.automatic_recovery_runtime import (
    BridgeProcess, DEFAULT_ENV_PYTHON, load_runtime_config,
)
from research.gs_opd.repair.continue_t1019_pot1_execution_aware_grasp_lift import (
    EXPECTED_OBJECT_ORDER,
    MAX_DEMONSTRATION_HORIZON,
    MODEL_SHA256,
    RISK_REARM_M,
    RISK_TRIGGER_M,
    SOURCE_ARCHIVE as T1019_ARCHIVE,
    _PythonObjectState,
    _action_reference_pose,
    _array_sha256,
    _native_values,
    _normalized_state_sha256,
    _validate_native_stage,
)
from research.gs_opd.repair.run_t329_native_task_suffix import (
    CONFIG_PATH, DurableTaskSuffixBackend, _canonical_sha256, _jsonable, _write_json,
)


FAILED_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/l0t1019_pot1_grasp_lift_20261025_030744"
)
TIMESTEP = 1042
SNAPSHOT = FAILED_ARCHIVE / "snapshots/main_timestep_001042.pkl"
TRACE = FAILED_ARCHIVE / "observation_action_trace.npz"
RUN_CONFIG = FAILED_ARCHIVE / "run_config.json"
PREVIOUSLY_REJECTED_ACTION = np.asarray([
    -0.04992363601922989,
    0.04706912115216255,
    -0.18786199390888214,
    0.0009311616304330528,
    -0.00413983128964901,
    -0.0003281764511484653,
    -1.0,
], dtype=np.float32)

setattr(__main__, "_PythonObjectState", _PythonObjectState)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-pot2-target-policy", action="store_true",
        help="Set Pot 1 non-finger clearance to the verified Pot 2 target-object margin, preserving all other rules.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38755)
    args = parser.parse_args()
    envelope = pickle.loads(SNAPSHOT.read_bytes())
    if envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1" or envelope.get(
        "model_geometry_sha256"
    ) != MODEL_SHA256:
        raise RuntimeError("t1042_snapshot_model_or_schema_mismatch")
    snapshot = envelope["snapshot"]
    integration = np.asarray(snapshot["mujoco_state"], dtype=np.float64)
    if integration.shape != (317,) or int(snapshot["task_bookkeeping"]["timestep"]) != TIMESTEP:
        raise RuntimeError("t1042_snapshot_integration_or_timestep_mismatch")
    with np.load(TRACE, allow_pickle=False) as trace:
        if int(trace["step_timesteps"][-1]) != TIMESTEP:
            raise RuntimeError("t1042_OA_does_not_terminate_at_snapshot")
        physical = json.loads(str(trace["post_physical_state_json"][-1]))
        native_source = json.loads(str(trace["post_native_task_status_json"][-1]))
    if not np.array_equal(np.asarray(physical["mujoco_integration_state"], dtype=np.float64), integration):
        raise RuntimeError("t1042_snapshot_and_OA_integration_state_mismatch")
    if _native_values(native_source) != {
        ("on", "moka_pot_1", "flat_stove_1_cook_region"): False,
        ("on", "moka_pot_2", "flat_stove_1_cook_region"): True,
        ("turnon", "flat_stove_1"): True,
    }:
        raise RuntimeError("t1042_native_predicate_mismatch")

    prior = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    policy = prior["frozen_operational_safety_policy"]
    policy = json.loads(json.dumps(policy))
    policy_adjustment = {"applied": False}
    if args.reuse_pot2_target_policy:
        target_rows = {
            str(row["object_id"]): row
            for row in policy.get("allowed_target_objects", [])
        }
        if set(("moka_pot_1", "moka_pot_2")) - set(target_rows):
            raise RuntimeError("prior_policy_lacks_Pot1_or_Pot2_target_margin")
        if target_rows["moka_pot_1"].get("minimum_clearance_m") != 0.010:
            raise RuntimeError("historical_Pot1_target_margin_is_not_expected_10mm_baseline")
        pot2_margin = float(target_rows["moka_pot_2"]["minimum_clearance_m"])
        if pot2_margin != 0.002:
            raise RuntimeError("verified_Pot2_target_margin_is_not_2mm")
        target_rows["moka_pot_1"]["minimum_clearance_m"] = pot2_margin
        policy_adjustment = {
            "applied": True,
            "source": "same_object_specific_nonfinger_target_clearance_as_verified_Pot2",
            "Pot1_nonfinger_minimum_clearance_m": pot2_margin,
            "Pot2_nonfinger_minimum_clearance_m": pot2_margin,
            "GingerBeer_minimum_clearance_m": 0.010,
            "PlaceBox_minimum_clearance_m": 0.010,
            "only_four_verified_finger_geoms_may_contact_Pot1": True,
            "allowed_finger_penetration_m": 0.002,
            "palm_or_other_robot_penetration_allowed": False,
        }
    config_data, recovery = load_runtime_config(CONFIG_PATH)
    config_data = json.loads(json.dumps(config_data))
    config_data["candidate_search"]["enabled"] = False
    target_clearances = {
        str(item["object_id"]): float(item["minimum_clearance_m"])
        for item in policy["allowed_target_objects"]
    }
    target_clearances = {
        str(item["object_id"]): float(item["minimum_clearance_m"])
        for item in policy.get("allowed_target_objects", [])
    }
    recovery = recovery.__class__(
        **{
            **recovery.__dict__,
            "hard_minimum_clearance_m": float(policy["minimum_clearance_m"]),
            "target_object_clearance_m": float(policy["minimum_clearance_m"]),
            "object_clearance_m": target_clearances,
            "execution_step_budget": MAX_DEMONSTRATION_HORIZON,
            "horizon_steps": MAX_DEMONSTRATION_HORIZON,
            "allow_horizon_extension": True,
        }
    )
    config_data["recovery"] = _jsonable(recovery.__dict__)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    out = args.output_dir or Path(
        f"/opt/data/private/gs_opd/rollouts/l0t1042_pot1_policy_reentry_preview_{stamp}"
        if args.reuse_pot2_target_policy
        else f"/opt/data/private/gs_opd/rollouts/l0t1042_rejected_approach_preview_audit_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=False)
    attestation = {
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(SNAPSHOT),
        "source_snapshot_sha256": hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        "timestep": TIMESTEP,
        "model_geometry_sha256": MODEL_SHA256,
        "object_id": "moka_pot_1",
        "arm_joint_qpos": physical["robot_states"][0]["joint_qpos"],
        "eef_position_world_m": physical["robot_states"][0]["eef_pos"],
        "eef_quaternion_wxyz": physical["robot_states"][0]["eef_quat"],
        "gripper_qpos_m": physical["robot_states"][0]["gripper_qpos"],
        "object_position_world_m": physical["object_states"]["objects_dict:moka_pot_1"]["root_pos"],
        "object_quaternion_wxyz": physical["object_states"]["objects_dict:moka_pot_1"]["root_quat"],
        "integration_state_sha256": _array_sha256(integration),
        "runtime_snapshot_normalized_sha256": _normalized_state_sha256(snapshot),
    }
    artifacts = RecoveryArtifactWriter(out)
    bridge = BridgeProcess(out, python=args.python, port=args.bridge_port)
    backend = None
    try:
        bridge.start(
            benchmark=config_data["benchmark"], level=int(config_data["level"]),
            task_id=int(config_data["task_id"]), init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=MAX_DEMONSTRATION_HORIZON,
        )
        backend = DurableTaskSuffixBackend(
            bridge, config_data, recovery, out, artifacts,
            mode="task-runtime-snapshot", student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=SNAPSHOT,
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
            runtime_snapshot_attestation=attestation,
            journal_dir=out / "durable_action_journal",
        )
        restored = backend.restore_start_state()
        restored_state = np.asarray(restored["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
        if int(restored["timestep"]) != TIMESTEP or not np.array_equal(restored_state, integration):
            raise RuntimeError("strict_t1042_restore_failed")
        exported = bridge.request({"op": "export_runtime_snapshot"})
        if _normalized_state_sha256(pickle.loads(exported["payload"])["snapshot"]) != attestation[
            "runtime_snapshot_normalized_sha256"
        ]:
            raise RuntimeError("t1042_controller_runtime_state_restore_mismatch")
        backend.latest_data = restored
        backend._load_context_inputs(restored)
        if _canonical_sha256(backend.safety_policy) != _canonical_sha256(policy):
            # Compare static policy; phase-specific finger pairs are rebound below.
            dynamic = {"allowed_contact_pairs", "carried_object"}
            left = {k: v for k, v in backend.safety_policy.items() if k not in dynamic}
            right = {k: v for k, v in policy.items() if k not in dynamic}
            if _canonical_sha256(left) != _canonical_sha256(right):
                raise RuntimeError("t1042_static_safety_policy_restore_mismatch")
        ctl = backend.task_controller
        if [str(row["object_id"]) for row in ctl.objects] != EXPECTED_OBJECT_ORDER:
            raise RuntimeError("t1042_task_object_order_mismatch")
        ctl.object_index = 1
        ctl.phase = "approach"
        ctl._expert_handoff = True
        # The 23 committed approach actions are evidenced in this archive;
        # one approach sample was also counted at t1019 in the prior runner.
        ctl._phase_steps = 24
        ctl._last_timestep = TIMESTEP
        ctl._grasp_last_timestep = -1
        ctl._grasp_streak = 0
        ctl._lift_start = None
        ctl._lift_eef_start = None
        backend.task_expert_active = True
        backend.last_task_phase = "approach"
        backend.task_phase_anchor = None
        backend.confirmed_placed_objects.clear()
        live = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        live["native_task_status"] = bridge.request({"op": "native_task_status"})
        live["contacts"] = bridge.request({"op": "inspect_contacts"})
        backend.latest_data = live
        context = backend.current_context()
        if context.phase != "approach":
            raise RuntimeError(f"t1042_reconstructed_phase_changed:{context.phase}")
        finger_ids = set(int(value) for value in backend.context_static["gripper_geometry_ids"])
        pot1_ids = set(int(value) for value in backend.bindings["moka_pot_1"]["geom_ids"])
        allowed_pairs = {
            (int(row["robot_geom_id"]), int(row["scene_geom_id"]))
            for row in backend.safety_policy.get("allowed_contact_pairs", [])
        }
        expected_pairs = {(finger, target_geom) for finger in finger_ids for target_geom in pot1_ids}
        if len(finger_ids) != 4 or allowed_pairs != expected_pairs:
            raise RuntimeError("active_Pot1_contact_allowlist_is_not_exact_four_fingers_by_target_geoms")
        if any(
            not np.isclose(float(row.get("minimum_clearance_m", 1)), -0.002, rtol=0.0, atol=1e-12)
            for row in backend.safety_policy.get("allowed_contact_pairs", [])
        ):
            raise RuntimeError("active_finger_contact_penetration_limit_is_not_exactly_2mm")
        obs = backend._observation(live)
        action = np.asarray(backend.task_expert_action(obs, context), dtype=np.float32)
        if not np.allclose(action, PREVIOUSLY_REJECTED_ACTION, rtol=0.0, atol=1e-7):
            raise RuntimeError("reconstructed_T1042_action_differs_from_previously_rejected_action")
        robot = live["snapshot_state"]["robot_states"][0]
        repair = bridge.request({"op": "repair_context"})
        reference = _action_reference_pose(action, robot, repair["robots"][0]["fields"])
        poses = np.stack((np.concatenate((robot["eef_pos"], robot["eef_quat"])), reference))
        qpos = np.asarray(robot["joint_qpos"], dtype=np.float64)
        sid = bridge.request({"op": "snapshot"})["snapshot_id"]
        margin = float(live["safety_geometry"]["minimum_forbidden_clearance_m"])
        preview = bridge.request({
            "op": "osc_prefix_probe", "snapshot_id": sid,
            "current_state_sha256": _array_sha256(integration),
            "jointspace_path_qpos": np.stack((qpos, qpos)).tolist(),
            "eef_reference_poses_world": poses.tolist(),
            "path_start_index": 0, "max_control_steps": 1,
            "gripper_action": float(action[6]),
            "native_action_prefix": [action.tolist()],
            "risk_trigger_distance_m": RISK_TRIGGER_M,
            "risk_rearm_distance_m": RISK_REARM_M,
            "risk_crossing_armed": margin >= RISK_REARM_M,
            "maximum_joint_path_deviation_rad": recovery.maximum_joint_path_deviation_rad,
            "maximum_eef_tracking_error_m": recovery.maximum_eef_tracking_error_m,
            "maximum_eef_orientation_error_rad": recovery.maximum_eef_orientation_error_rad,
            "qpos_reference_mode": "diagnostic_only",
            "current_minimum_forbidden_clearance_m": margin,
            "safety_geometry_policy": backend.safety_policy,
        })
        compact_steps = []
        for step in preview.get("steps", []):
            substeps = step.get("safety_geometry_substeps", [])
            failing_substeps = []
            for index, sub in enumerate(substeps):
                if sub.get("status") != "complete" or any(sub.get(key) for key in (
                    "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
                    "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
                )):
                    failing_substeps.append({
                        "substep": index, "status": sub.get("status"),
                        "forbidden_contacts": sub.get("forbidden_contacts"),
                        "forbidden_object_contacts": sub.get("forbidden_object_contacts"),
                        "self_contacts": sub.get("self_contacts"),
                        "unallowed_scene_contacts": sub.get("unallowed_scene_contacts"),
                        "unresolved_contacts": sub.get("unresolved_contacts"),
                        "joint_limit_violations": sub.get("joint_limit_violations"),
                        "clearances_by_object": sub.get("clearances_by_object"),
                        "contact_identities": sub.get("contact_identities"),
                    })
            terminal = step.get("terminal_runtime_snapshot")
            if isinstance(terminal, bytes):
                branch = out / "rejected_terminal_snapshot.pkl"
                branch.write_bytes(terminal)
                step["terminal_runtime_snapshot"] = str(branch)
            compact_steps.append({
                "path_index": step.get("path_index"),
                "action": step.get("action"),
                "joint_path_deviation_rad_diagnostic_only": step.get("joint_path_deviation_rad"),
                "eef_tracking_error_m_diagnostic_only": step.get("eef_tracking_error_m"),
                "eef_orientation_error_rad_diagnostic_only": step.get("eef_orientation_error_rad"),
                "failing_substeps": failing_substeps,
                "all_substep_records": substeps,
            })
        report = {
            "status": preview.get("status"),
            "safe": preview.get("safe"),
            "stop_reason": preview.get("stop_reason"),
            "failure_reason": preview.get("failure_reason"),
            "main_state_restored_verified": preview.get("main_state_restored_verified"),
            "source_timestep": TIMESTEP,
            "source_snapshot_sha256": attestation["source_snapshot_sha256"],
            "source_integration_state_sha256": attestation["integration_state_sha256"],
            "phase": context.phase,
            "action": action.tolist(),
            "gripper_command": float(action[6]),
            "object_specific_policy_adjustment": policy_adjustment,
            "safety_policy_effective": backend.safety_policy,
            "finger_geom_ids": sorted(finger_ids),
            "Pot1_geom_ids": sorted(pot1_ids),
            "allowed_contact_pairs_exact_four_fingers": True,
            "goal_error_position_m": context.metadata.get("phase_goal_position_error_m"),
            "goal_error_orientation_rad": context.metadata.get("phase_goal_orientation_error_rad"),
            "source_live_geometry": live.get("safety_geometry"),
            "steps": compact_steps,
            "physical_action_committed": False,
            "new_RRT_calls": 0,
            "output_dir": str(out),
        }
        _write_json(out / "policy_reentry_preview_manifest.json", {
            "source_snapshot": str(SNAPSHOT),
            "source_snapshot_sha256": attestation["source_snapshot_sha256"],
            "source_policy_sha256": prior.get("frozen_operational_safety_policy_sha256"),
            "policy_adjustment": policy_adjustment,
            "candidate_action_must_match": PREVIOUSLY_REJECTED_ACTION.tolist(),
            "physical_action_committed": False,
            "new_RRT_calls": 0,
        })
        _write_json(out / "rejected_action_preview_audit.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
