#!/usr/bin/env python3
"""Short, audited release probe from an exact held-object Runtime Snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.gs_opd.repair import task_success_first_grasp_search as search
from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor
from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--resume-snapshot", type=Path, required=True)
    p.add_argument("--t0-manifest", type=Path, required=True)
    p.add_argument("--parent-run-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--target-predicate", nargs=3, required=True)
    p.add_argument("--model-sha", required=True)
    p.add_argument("--bddl-sha", required=True)
    p.add_argument("--demonstration-horizon", type=int, default=1000)
    p.add_argument("--maximum-actions", type=int, default=10)
    p.add_argument("--minimum-release-actions", type=int, default=1)
    p.add_argument("--gripper-action", type=float, default=-1.0,
                   help="real gripper command during the arm-hold probe; negative opens, positive closes")
    p.add_argument("--allow-satisfied-target-goal", action="store_true")
    p.add_argument("--control-mode", choices=("eef_osc", "joint_hold"), default="eef_osc")
    p.add_argument(
        "--continue-partial-release-contact", action="store_true",
        help=(
            "continue an already-open joint-hold release from a Snapshot with "
            "no verified grasp but remaining finger-pad contact"
        ),
    )
    p.add_argument(
        "--allow-unilateral-contact", action="store_true",
        help=(
            "allow a release probe when the exact source Snapshot has a live, "
            "verified contact on at least one finger pad but not a bilateral grasp"
        ),
    )
    a = p.parse_args()
    if not 1 <= a.maximum_actions <= 250:
        raise ValueError("gravity_release_probe_is_limited_to_1_through_250_actions")
    if not np.isfinite(a.gripper_action) or not -1.0 <= a.gripper_action <= 1.0:
        raise ValueError("gripper_action_must_be_finite_and_in_minus1_to_plus1")
    if not 1 <= a.minimum_release_actions <= a.maximum_actions:
        raise ValueError("minimum_release_actions_must_fit_the_release_budget")
    if not 1000 <= a.demonstration_horizon <= search.MAX_DEMONSTRATION_HORIZON:
        raise ValueError("gravity_release_probe_horizon_is_outside_supported_range")
    resume = search._verified_resume_runtime_snapshot(
        a.resume_snapshot, benchmark=a.benchmark, level=a.level,
        task_id=a.task_id, seed=0, init_state=0,
        model_sha=a.model_sha, bddl_sha=a.bddl_sha, instruction=a.instruction,
    )
    t0_manifest = json.loads(a.t0_manifest.read_text(encoding="utf-8"))
    t0_scene = t0_manifest.get("scene_metadata", {})
    identity = {
        "benchmark": a.benchmark, "level": a.level, "task_id": a.task_id,
        "environment_seed": 0, "init_state_index": 0,
        "model_geometry_sha256": a.model_sha, "bddl_sha256": a.bddl_sha,
        "instruction": a.instruction,
    }
    for key, expected in identity.items():
        if t0_scene.get(key) != expected:
            raise ValueError(f"t0_manifest_identity_mismatch:{key}:{t0_scene.get(key)}")
    parent_report = json.loads(a.parent_run_report.read_text(encoding="utf-8"))
    parent_terminal = parent_report.get("terminal_snapshot", {})
    if (
        Path(str(parent_terminal.get("path", ""))).resolve() != a.resume_snapshot.resolve()
        or parent_terminal.get("payload_sha256") != resume["payload_sha256"]
    ):
        raise ValueError("parent_report_terminal_snapshot_does_not_match_resume_snapshot")
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "source_resume_runtime_snapshot.pkl").write_bytes(resume["payload"])
    (a.output / "source_snapshot_manifest.json").write_text(
        json.dumps({
            "schema_version": "gs_opd_runtime_snapshot_manifest.v1",
            "scene_metadata": resume["scene_metadata"],
            "snapshot": {
                "path": str(a.resume_snapshot.resolve()),
                "payload_sha256": resume["payload_sha256"],
                "integration_state_sha256": resume["integration_state_sha256"],
            },
            "t0_manifest": str(a.t0_manifest.resolve()),
            "t0_manifest_sha256": hashlib.sha256(a.t0_manifest.read_bytes()).hexdigest(),
            "parent_run_report": str(a.parent_run_report.resolve()),
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    bridge_args = SimpleNamespace(
        benchmark=a.benchmark, level=a.level, task_id=a.task_id, init_state=0,
        seed=0, camera_resolution=256,
        demonstration_horizon=a.demonstration_horizon, port=port,
    )
    bridge = search.BridgeSession(
        ("127.0.0.1", port), a.output / "safety_bridge.log", bridge_args,
    )
    records: list[dict] = []
    frames: list[dict] = []
    report = {
        "schema_version": (
            "gs_opd_snapshot_gripper_hold_probe.v1"
            if a.gripper_action > 0.5 else
            "gs_opd_snapshot_gravity_release_probe.v1"
        ),
        "status": "in_progress",
        "hypothesis": (
            "a real close-gripper command establishes bilateral grasp from the measured two-side contact pose"
            if a.gripper_action > 0.5 else
            "a real open-gripper action lets gravity and existing physical contacts move the object while the arm holds its measured pose"
        ),
        "identity": {
            "benchmark": a.benchmark, "level": a.level, "task_id": a.task_id,
            "seed": 0, "init_state": 0, "model_geometry_sha256": a.model_sha,
            "bddl_sha256": a.bddl_sha, "native_goal_predicate": a.target_predicate,
        },
        "source_snapshot": {
            "path": str(a.resume_snapshot.resolve()),
            "payload_sha256": resume["payload_sha256"],
            "integration_state_sha256": resume["integration_state_sha256"],
            "bridge_horizon": resume["scene_metadata"].get("bridge_horizon"),
        },
        "t0_manifest": str(a.t0_manifest.resolve()),
        "t0_manifest_sha256": hashlib.sha256(a.t0_manifest.read_bytes()).hexdigest(),
        "parent_run_report": str(a.parent_run_report.resolve()),
        "demonstration_horizon": a.demonstration_horizon,
        "configured_maximum_actions": a.maximum_actions,
        "safety_certified": False,
        "started_at_epoch": time.time(),
    }
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        if reset.get("scene_metadata", {}).get("model_geometry_sha256") != a.model_sha:
            raise RuntimeError("fresh_reset_model_fingerprint_mismatch")
        restored = search._restore_initial(
            bridge, resume["payload"],
            resume["snapshot_state"]["mujoco_integration_state"], a.model_sha,
        )
        obs = bridge.request({"op": "observe", "include_snapshot_state": True})
        if not np.array_equal(
            search._snapshot_state_vector(obs.get("snapshot_state", {})),
            resume["snapshot_state"]["mujoco_integration_state"],
        ):
            raise RuntimeError("resume_snapshot_integration_state_mismatch")
        native = bridge.request({"op": "native_task_status"})
        if native.get("status") != "complete" or native.get("task_success") is not False:
            raise RuntimeError(f"resume_native_status_invalid:{native}")
        if native.get("timestep", a.demonstration_horizon) + a.maximum_actions >= a.demonstration_horizon:
            raise ValueError("release_probe_would_reach_demonstration_horizon")
        context = bridge.request({
            "op": "task_success_first_context",
            "expected_instruction": a.instruction,
            "target_predicate": a.target_predicate,
        })
        if (context.get("status") != "complete"
                or (context.get("native_goal_value") is not False
                    and not (a.allow_satisfied_target_goal and context.get("native_goal_value") is True))):
            raise RuntimeError(
                f"active_native_goal_context_invalid:status={context.get('status')};"
                f"goal_value={context.get('native_goal_value')}"
            )
        binding = dict(context["target_binding"])
        binding["body_id"] = int(binding["root_body_id"])
        binding["body_name"] = str(binding["root_body_name"])
        controller = bridge.request({"op": "repair_context"})
        if controller.get("status") not in {"available", "complete"}:
            raise RuntimeError(f"active_controller_context_invalid:{controller}")
        fields = controller["robots"][0]["fields"]
        search.configure_stage_a_contact_only_substep_audit(True)
        initial_grasp = bridge.request({
            "op": "grasp_state", "carried_binding": binding,
            "expected_instruction": a.instruction,
        })
        finger_contacts = initial_grasp.get("finger_contacts", {})
        bilateral_grasp = bool(
            initial_grasp.get("grasp_confirmed") is True
            and finger_contacts.get("left") is True
            and finger_contacts.get("right") is True
        )
        unilateral_contact = bool(
            finger_contacts.get("status") == "pass"
            and (finger_contacts.get("left") is True or finger_contacts.get("right") is True)
        )
        partial_release_continuation = bool(
            a.continue_partial_release_contact
            and a.control_mode == "joint_hold"
            and context.get("native_goal_value") is True
            and initial_grasp.get("grasp_confirmed") is False
            # A one-sided pad contact is reported with status="fail" by the
            # bilateral grasp validator. For an explicitly authorized
            # continuation, accept that measured partial-release state while
            # still requiring a real matched finger contact and a satisfied
            # native In predicate.
            and finger_contacts.get("status") in {"pass", "fail"}
            and (
                (
                    (finger_contacts.get("left") is True or finger_contacts.get("right") is True)
                    and int(finger_contacts.get("contact_count", 0)) > 0
                )
                or (
                    finger_contacts.get("left") is False
                    and finger_contacts.get("right") is False
                    and int(finger_contacts.get("contact_count", 0)) == 0
                    and float(initial_grasp.get("gripper_closure", {}).get(
                        "close_command_fraction", 1.0
                    )) <= 0.05
                )
            )
        )
        if not bilateral_grasp and not (a.allow_unilateral_contact and unilateral_contact) and not partial_release_continuation:
            raise RuntimeError(
                "release_probe_requires_verified_live_bilateral_grasp_or_explicit_unilateral_contact"
            )
        initial_goal = bridge.request({"op": "native_goal_query", "predicate": a.target_predicate})
        initial_object = search._object_state(
            obs["snapshot_state"], int(binding["body_id"]),
        )
        start_snapshot = bridge.request({"op": "export_runtime_snapshot"})
        if start_snapshot.get("status") != "complete" or not isinstance(start_snapshot.get("payload"), bytes):
            raise RuntimeError("branch_start_snapshot_export_failed")
        (a.output / "branch_start_runtime_snapshot.pkl").write_bytes(start_snapshot["payload"])
        report.update({
            "resume_native_task_status": native,
            "initial_native_goal_query": initial_goal,
            "initial_target_root_pose": initial_object,
            "initial_eef_pose": obs["snapshot_state"]["robot_states"][0],
            "initial_grasp_or_contact_evidence": initial_grasp,
            "initial_verified_bilateral_grasp": bilateral_grasp,
            "unilateral_contact_release_enabled": bool(a.allow_unilateral_contact),
            "partial_release_continuation_authorized": partial_release_continuation,
            "controller_fields_source": "active Safety Bridge repair_context",
            "action_profile": (
                f"mass-matrix PD holds source arm qpos while gripper receives command ({a.gripper_action:+.3f}); no direct state writes"
                if a.control_mode == "joint_hold" else
                f"OSC holds observed live EEF pose while gripper receives command ({a.gripper_action:+.3f}); no object/joint state writes"
            ),
        })
        counter = [int(native["timestep"])]
        monitor = GraspStabilityMonitor(required_samples=3)
        monitor.update(initial_grasp, rollout_step=counter[0])
        hold_joint_qpos = np.asarray(obs["snapshot_state"]["robot_states"][0]["joint_qpos"], dtype=np.float64)
        search._record_joint_trajectory_step.controller_fields = fields
        report["release_control_mode"] = a.control_mode
        consecutive_verified_release_observations = 0
        consecutive_verified_grasp_observations = 0
        consecutive_verified_grasp_observations = 0
        for _ in range(a.maximum_actions):
            live = bridge.request({"op": "observe", "include_snapshot_state": True})
            robot = live["snapshot_state"]["robot_states"][0]
            pos = np.asarray(robot["eef_pos"], dtype=np.float64)
            quat = np.asarray(robot["eef_quat"], dtype=np.float64)
            action = world_pose_tracking_action(
                pos, quat, pos.copy(), quat.copy(), fields, gripper_action=a.gripper_action,
            )
            if a.control_mode == "joint_hold":
                command_label = "close" if a.gripper_action > 0.5 else "release"
                search._record_joint_trajectory_step(
                    bridge, hold_joint_qpos,
                    f"snapshot_joint_hold_gripper_{command_label}_probe",
                    f"native_bddl_region_joint_hold_gripper_{command_label}", len(records),
                    binding, a.instruction, {}, monitor, records, frames,
                    gripper_action=a.gripper_action, raise_on_done=False,
                )
            else:
                search._record_step(
                    bridge, action,
                    "snapshot_gravity_release_probe" if a.gripper_action < -0.5
                    else "snapshot_gravity_gripper_command_probe",
                    "native_bddl_region_gravity_release" if a.gripper_action < -0.5
                    else "native_bddl_region_gripper_command", len(records),
                    binding, a.instruction, {}, monitor, records, frames,
                    raise_on_done=False,
                )
            counter[0] += 1
            after_goal = bridge.request({"op": "native_goal_query", "predicate": a.target_predicate})
            after_native = bridge.request({"op": "native_task_status"})
            last = records[-1]
            state = last.get("snapshot_state", {})
            obj = search._object_state(state, int(binding["body_id"]))
            velocity = np.asarray(obj.get("joint_qvel", []), dtype=np.float64)
            grasp_after = last.get("grasp_state", {})
            fingers_after = grasp_after.get("finger_contacts", {})
            no_finger_pad_contact = bool(
                fingers_after.get("status") in {"pass", "fail"}
                and fingers_after.get("left") is False
                and fingers_after.get("right") is False
                and int(fingers_after.get("contact_count", 0)) == 0
            )
            verified_unheld = bool(
                grasp_after.get("grasp_confirmed") is False and no_finger_pad_contact
            )
            object_speed = float(np.linalg.norm(velocity)) if velocity.size else None
            if (after_goal.get("predicate_value") is True and verified_unheld
                    and object_speed is not None and object_speed < 0.01):
                consecutive_verified_release_observations += 1
            else:
                consecutive_verified_release_observations = 0
            verified_bilateral_grasp = bool(
                grasp_after.get("grasp_confirmed") is True
                and fingers_after.get("left") is True
                and fingers_after.get("right") is True
            )
            if a.gripper_action > 0.5 and verified_bilateral_grasp:
                consecutive_verified_grasp_observations += 1
            else:
                consecutive_verified_grasp_observations = 0
            report.setdefault("steps", []).append({
                "timestep": after_native.get("timestep"),
                "action_7d": last["action_7d"],
                "joint_target_qpos_rad": last.get("joint_target_qpos_rad"),
                "object_root_pose": obj,
                "object_root_speed": object_speed,
                "native_goal_query": after_goal,
                "native_task_status": after_native,
                "grasp_state": last.get("grasp_state"),
                "verified_bilateral_grasp": verified_bilateral_grasp,
                "consecutive_verified_grasp_observations": consecutive_verified_grasp_observations,
                "finger_pad_contact_free": no_finger_pad_contact,
                "consecutive_verified_release_observations": consecutive_verified_release_observations,
                "captured_contact_substeps": last.get("contact_monitoring", {}).get("captured_physics_substeps"),
            })
            if after_native.get("task_success") is True:
                report["status"] = "native_task_success"
                break
            if consecutive_verified_grasp_observations >= 3:
                report["status"] = "verified_bilateral_grasp_candidate"
                break
            if (after_goal.get("predicate_value") is True
                    and len(records) >= a.minimum_release_actions
                    and consecutive_verified_release_observations >= 5):
                report["status"] = "native_goal_preserved_after_verified_release"
                break
            if after_native.get("done") is True:
                report["early_stop_reason"] = "environment_done_without_native_task_success"
                break
            if (verified_unheld and after_goal.get("predicate_value") is not True
                    and len(report["steps"]) >= max(4, a.minimum_release_actions)):
                recent = report["steps"][-4:]
                dz = abs(float(recent[-1]["object_root_pose"]["root_pos"][2]) - float(recent[0]["object_root_pose"]["root_pos"][2]))
                speed = recent[-1].get("object_root_speed")
                if dz < 0.002 and speed is not None and speed < 0.01:
                    report["early_stop_reason"] = "released_object_settled_without_native_goal_progress"
                    break
        report["actions_executed"] = len(records)
        report["final_native_task_status"] = bridge.request({"op": "native_task_status"})
        report["final_native_goal_query"] = bridge.request({
            "op": "native_goal_query", "predicate": a.target_predicate,
        })
        terminal = bridge.request({"op": "export_runtime_snapshot"})
        if terminal.get("status") == "complete" and isinstance(terminal.get("payload"), bytes):
            (a.output / "terminal_runtime_snapshot.pkl").write_bytes(terminal["payload"])
            report["terminal_snapshot"] = {
                "path": str(a.output / "terminal_runtime_snapshot.pkl"),
                "payload_sha256": hashlib.sha256(terminal["payload"]).hexdigest(),
            }
        report["action_record_archive"] = search._write_recording(a.output, records, frames)
        if report["final_native_task_status"].get("task_success") is True:
            report["status"] = "native_task_success"
        elif a.gripper_action > 0.5:
            if report["status"] != "verified_bilateral_grasp_candidate":
                report["status"] = "gripper_command_probe_incomplete"
        elif report["status"] not in {
            "native_goal_satisfied_by_physical_release",
            "native_goal_preserved_after_verified_release",
        }:
            report["status"] = "bounded_release_probe_failed"
        report["physics_audit"] = {
            "muJoCo_actions": len(records),
            "captured_contact_substeps": sum(
                int(row.get("contact_monitoring", {}).get("captured_physics_substeps", 0))
                for row in records
            ),
            "all_contact_substeps_complete": all(
                row.get("substep_contact_monitoring_complete") is True for row in records
            ),
            "safety_certified": False,
        }
    except Exception as exc:
        report["status"] = "probe_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        try:
            report["final_native_task_status"] = bridge.request({"op": "native_task_status"})
            terminal = bridge.request({"op": "export_runtime_snapshot"})
            if terminal.get("status") == "complete" and isinstance(terminal.get("payload"), bytes):
                (a.output / "terminal_runtime_snapshot.pkl").write_bytes(terminal["payload"])
            search._write_recording(a.output, records, frames)
        except Exception as archive_exc:
            report["archive_error"] = f"{type(archive_exc).__name__}: {archive_exc}"
        raise
    finally:
        report["finished_at_epoch"] = time.time()
        report["wall_time_seconds"] = report["finished_at_epoch"] - report["started_at_epoch"]
        (a.output / "run_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2,
                       default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)) + "\n",
            encoding="utf-8",
        )
        bridge.close()
    print(json.dumps({
        "status": report["status"], "actions": report.get("actions_executed"),
        "goal_value": report.get("final_native_goal_query", {}).get("predicate_value"),
        "task_success": report.get("final_native_task_status", {}).get("task_success"),
        "output": str(a.output),
    }, ensure_ascii=False))
    return 0 if report["status"] in {
        "native_goal_satisfied_by_physical_release",
        "native_goal_preserved_after_verified_release",
        "native_task_success",
        "verified_bilateral_grasp_candidate",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
