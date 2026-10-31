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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--target-predicate", nargs=3, required=True)
    p.add_argument("--model-sha", required=True)
    p.add_argument("--bddl-sha", required=True)
    p.add_argument("--maximum-actions", type=int, default=10)
    a = p.parse_args()
    if not 1 <= a.maximum_actions <= 10:
        raise ValueError("gravity_release_probe_is_limited_to_1_through_10_actions")
    resume = search._verified_resume_runtime_snapshot(
        a.resume_snapshot, benchmark=a.benchmark, level=a.level,
        task_id=a.task_id, seed=0, init_state=0,
        model_sha=a.model_sha, bddl_sha=a.bddl_sha, instruction=a.instruction,
    )
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "source_resume_runtime_snapshot.pkl").write_bytes(resume["payload"])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    bridge_args = SimpleNamespace(
        benchmark=a.benchmark, level=a.level, task_id=a.task_id, init_state=0,
        seed=0, camera_resolution=256, demonstration_horizon=1000, port=port,
    )
    bridge = search.BridgeSession(
        ("127.0.0.1", port), a.output / "safety_bridge.log", bridge_args,
    )
    records: list[dict] = []
    frames: list[dict] = []
    report = {
        "schema_version": "gs_opd_snapshot_gravity_release_probe.v1",
        "status": "in_progress",
        "hypothesis": "the physically held mug is horizontally over the microwave opening; a normal gripper release and gravity may place it into the native BDDL In region",
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
        if native.get("timestep", 1000) + a.maximum_actions >= 1000:
            raise ValueError("release_probe_would_reach_standard_horizon")
        context = bridge.request({
            "op": "task_success_first_context",
            "expected_instruction": a.instruction,
            "target_predicate": a.target_predicate,
        })
        if context.get("status") != "complete" or context.get("native_goal_value") is not False:
            raise RuntimeError(f"active_native_goal_context_invalid:{context}")
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
        if not (
            initial_grasp.get("grasp_confirmed") is True
            and initial_grasp.get("finger_contacts", {}).get("left") is True
            and initial_grasp.get("finger_contacts", {}).get("right") is True
        ):
            raise RuntimeError("release_probe_requires_verified_live_bilateral_grasp")
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
            "initial_verified_grasp": initial_grasp,
            "controller_fields_source": "active Safety Bridge repair_context",
            "action_profile": "OSC holds observed live EEF pose while gripper receives normal open command (-1); no object/joint state writes",
        })
        counter = [int(native["timestep"])]
        monitor = GraspStabilityMonitor(required_samples=3)
        monitor.update(initial_grasp, rollout_step=counter[0])
        for _ in range(a.maximum_actions):
            live = bridge.request({"op": "observe", "include_snapshot_state": True})
            robot = live["snapshot_state"]["robot_states"][0]
            pos = np.asarray(robot["eef_pos"], dtype=np.float64)
            quat = np.asarray(robot["eef_quat"], dtype=np.float64)
            action = world_pose_tracking_action(
                pos, quat, pos.copy(), quat.copy(), fields, gripper_action=-1.0,
            )
            search._record_step(
                bridge, action, "snapshot_gravity_release_probe",
                "native_microwave_region_gravity_release", counter[0] + 1,
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
            report.setdefault("steps", []).append({
                "timestep": after_native.get("timestep"),
                "action_7d": last["action_7d"],
                "object_root_pose": obj,
                "object_root_speed": float(np.linalg.norm(velocity)) if velocity.size else None,
                "native_goal_query": after_goal,
                "native_task_status": after_native,
                "grasp_state": last.get("grasp_state"),
                "captured_contact_substeps": last.get("contact_monitoring", {}).get("captured_physics_substeps"),
            })
            if after_goal.get("value") is True or after_native.get("task_success") is True:
                report["status"] = "native_goal_satisfied_by_physical_release"
                break
            if after_native.get("done") is True:
                report["early_stop_reason"] = "environment_done_without_native_task_success"
                break
            if len(report["steps"]) >= 4:
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
        elif report["status"] != "native_goal_satisfied_by_physical_release":
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
        "goal_value": report.get("final_native_goal_query", {}).get("value"),
        "task_success": report.get("final_native_task_status", {}).get("task_success"),
        "output": str(a.output),
    }, ensure_ascii=False))
    return 0 if report["status"] in {"native_goal_satisfied_by_physical_release", "native_task_success"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
