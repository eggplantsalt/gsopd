#!/usr/bin/env python3
"""Physically clear an existing carried-grasp robot/scene contact from a verified Snapshot.

The script derives a short retreat direction from live MuJoCo contacts and moves
the held object with ordinary OSC actions. It never edits robot or object state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from research.gs_opd.repair import task_success_first_grasp_search as search
from research.gs_opd.repair.task_success_first_executor import (
    _robot_scene_contact_escape_direction,
)
from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-snapshot", type=Path, required=True)
    parser.add_argument("--t0-snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--target-predicate", nargs=3, required=True)
    parser.add_argument("--model-sha", required=True)
    parser.add_argument("--bddl-sha", required=True)
    parser.add_argument("--escape-distance-m", type=float, default=None)
    parser.add_argument(
        "--vertical-lift-m", type=float, default=None,
        help="physically lift the held object vertically to clear an overhead robot/fixture contact",
    )
    parser.add_argument("--maximum-actions", type=int, default=160)
    parser.add_argument("--demonstration-horizon", type=int, default=5000)
    parser.add_argument("--parent-run-report", type=Path)
    args = parser.parse_args()
    if args.escape_distance_m is not None and args.vertical_lift_m is not None:
        parser.error("--escape-distance-m and --vertical-lift-m are mutually exclusive")
    if args.vertical_lift_m is not None and not 0.005 <= args.vertical_lift_m <= 0.12:
        parser.error("--vertical-lift-m must be in 0.005..0.12")

    manifest_path = args.t0_snapshot_root / "snapshots/runtime_snapshot_manifest.json"
    initial_snapshot_path = args.t0_snapshot_root / "snapshots/initial_runtime_snapshot.pkl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene = manifest.get("scene_metadata", {})
    identity = {
        "benchmark": args.benchmark,
        "level": args.level,
        "task_id": args.task_id,
        "seed": 0,
        "init_state": 0,
        "model_geometry_sha256": args.model_sha,
        "bddl_sha256": args.bddl_sha,
        "instruction": args.instruction,
    }
    for key, expected in {
        "benchmark": args.benchmark,
        "level": args.level,
        "task_id": args.task_id,
        "environment_seed": 0,
        "init_state_index": 0,
        "model_geometry_sha256": args.model_sha,
        "bddl_sha256": args.bddl_sha,
        "instruction": args.instruction,
    }.items():
        if scene.get(key) != expected:
            raise ValueError(f"t0_manifest_identity_mismatch:{key}:{scene.get(key)}")
    resume = search._verified_resume_runtime_snapshot(
        args.resume_snapshot,
        benchmark=args.benchmark,
        level=args.level,
        task_id=args.task_id,
        seed=0,
        init_state=0,
        model_sha=args.model_sha,
        bddl_sha=args.bddl_sha,
        instruction=args.instruction,
    )
    initial = search._verified_resume_runtime_snapshot(
        initial_snapshot_path,
        benchmark=args.benchmark,
        level=args.level,
        task_id=args.task_id,
        seed=0,
        init_state=0,
        model_sha=args.model_sha,
        bddl_sha=args.bddl_sha,
        instruction=args.instruction,
    )
    if not initial_snapshot_path.is_file():
        raise FileNotFoundError("source_t0_runtime_snapshot_missing")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source_resume_runtime_snapshot.pkl").write_bytes(resume["payload"])
    (args.output / "source_snapshot_manifest.json").write_text(
        json.dumps({
            "schema_version": "gs_opd_contact_escape_snapshot_manifest.v1",
            "identity": identity,
            "t0_snapshot_root": str(args.t0_snapshot_root.resolve()),
            "t0_manifest_sha256": _sha(manifest_path.read_bytes()),
            "source_snapshot": str(args.resume_snapshot.resolve()),
            "source_snapshot_sha256": resume["payload_sha256"],
            "source_integration_state_sha256": resume["integration_state_sha256"],
            "parent_run_report": str(args.parent_run_report.resolve()) if args.parent_run_report else None,
            "actions_from_parent_replayed": False,
            "safety_certified": False,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    port = _port()
    bridge_args = SimpleNamespace(
        benchmark=args.benchmark,
        level=args.level,
        task_id=args.task_id,
        init_state=0,
        seed=0,
        camera_resolution=256,
        demonstration_horizon=args.demonstration_horizon,
        port=port,
    )
    bridge = search.BridgeSession(
        ("127.0.0.1", port), args.output / "safety_bridge.log", bridge_args,
    )
    search.configure_stage_a_contact_only_substep_audit(True)
    rows: list[dict] = []
    frames: list[dict] = []
    monitor = GraspStabilityMonitor(required_samples=3)
    records_path = args.output / "durable_physical_step_records.jsonl"
    frames_dir = args.output / "durable_dual_camera_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "gs_opd_verified_robot_scene_contact_escape.v1",
        "status": "in_progress",
        "identity": identity,
        "source_snapshot": str(args.resume_snapshot.resolve()),
        "source_snapshot_sha256": resume["payload_sha256"],
        "maximum_actions": args.maximum_actions,
        "demonstration_horizon": args.demonstration_horizon,
        "hypothesis": (
            "a real contact-derived retreat or vertical lift clears robot/scene contact "
            "before re-planning the held-object route"
        ),
        "direct_robot_or_object_state_write": False,
        "physical_actions_are_MuJoCo_steps": True,
        "safety_certified": False,
        "started_at_epoch": time.time(),
    }
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        if reset.get("scene_metadata", {}).get("model_geometry_sha256") != args.model_sha:
            raise RuntimeError("reset_model_geometry_mismatch")
        if reset.get("scene_metadata", {}).get("bddl_sha256") != args.bddl_sha:
            raise RuntimeError("reset_BDDL_mismatch")
        search._restore_initial(
            bridge,
            initial["payload"],
            initial["snapshot_state"]["mujoco_integration_state"],
            args.model_sha,
        )
        # Restore the exact partial-task Runtime Snapshot after the standard
        # t0 reset path, then verify the complete integration-state bytes.
        restored = bridge.request({
            "op": "restore_runtime_snapshot",
            "payload": resume["payload"],
        })
        observed = bridge.request({"op": "observe", "include_snapshot_state": True})
        actual = search._snapshot_state_vector(observed.get("snapshot_state", {}))
        expected = search._snapshot_state_vector(resume["snapshot_state"])
        if (
            restored.get("status") != "restored"
            or restored.get("model_geometry_sha256") != args.model_sha
            or actual.shape != expected.shape
            or not np.array_equal(actual, expected)
        ):
            raise RuntimeError("exact_partial_snapshot_restore_verification_failed")
        initial_native = bridge.request({"op": "native_task_status"})
        if initial_native.get("status") != "complete" or initial_native.get("task_success") is True:
            raise RuntimeError("snapshot_native_task_state_unexpected")
        context = bridge.request({
            "op": "task_success_first_context",
            "expected_instruction": args.instruction,
            "target_predicate": args.target_predicate,
        })
        if (
            context.get("status") != "complete"
            or context.get("native_goal_predicate") != args.target_predicate
            or context.get("model_geometry_sha256") != args.model_sha
        ):
            raise RuntimeError("active_native_target_context_mismatch")
        target_binding = dict(context["target_binding"])
        target_binding["body_id"] = int(target_binding["root_body_id"])
        target_binding["body_name"] = str(target_binding["root_body_name"])
        repair = bridge.request({"op": "repair_context"})
        controller_fields = repair["robots"][0]["fields"]
        search._advance_pose.controller_fields = controller_fields
        policy_path = args.parent_run_report.parent / "cold_start_safety_policy.json" if args.parent_run_report else None
        policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path and policy_path.is_file() else {}
        before_state = observed["snapshot_state"]
        before_obj = search._object_state(before_state, int(target_binding["body_id"]))
        before_pos = np.asarray(before_obj["root_pos"], dtype=np.float64)
        before_grasp = bridge.request({
            "op": "grasp_state",
            "carried_binding": target_binding,
            "expected_instruction": args.instruction,
        })
        if (
            before_grasp.get("grasp_confirmed") is not True
            or before_grasp.get("finger_contacts", {}).get("left") is not True
            or before_grasp.get("finger_contacts", {}).get("right") is not True
        ):
            raise RuntimeError("source_snapshot_does_not_retain_bilateral_grasp")
        contacts = bridge.request({"op": "inspect_contacts"})
        retreat = _robot_scene_contact_escape_direction(
            contacts,
            before_state,
            target_body_id=int(target_binding["body_id"]),
            eef_position_world_m=before_state["robot_states"][0]["eef_pos"],
        )
        report["initial_native_task_status"] = initial_native
        report["initial_grasp_state"] = before_grasp
        report["initial_contact_status"] = contacts.get("status")
        report["measured_contact_escape"] = retreat
        if retreat is None:
            report["status"] = "no_robot_scene_contact_to_escape"
            report["physical_actions"] = 0
            final = bridge.request({"op": "export_runtime_snapshot"})
            if final.get("status") == "complete" and isinstance(final.get("payload"), bytes):
                path = args.output / "contact_escape_terminal_runtime_snapshot.pkl"
                path.write_bytes(final["payload"])
                report["terminal_snapshot"] = str(path)
                report["terminal_snapshot_sha256"] = _sha(final["payload"])
            return_code = 0
        else:
            if args.vertical_lift_m is not None:
                distance = float(args.vertical_lift_m)
                direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
                escape_mode = "vertical_lift_for_overhead_robot_fixture_contact"
            elif args.escape_distance_m is not None:
                distance = float(args.escape_distance_m)
                direction = np.asarray(retreat["direction_world"], dtype=np.float64)
                direction[2] = 0.0
                escape_mode = "measured_horizontal_contact_radial_retreat"
            else:
                distance = float(np.clip(
                    retreat["contact_radial_extent_m"] + 0.06,
                    0.09,
                    0.16,
                ))
                direction = np.asarray(retreat["direction_world"], dtype=np.float64)
                direction[2] = 0.0
                escape_mode = "measured_horizontal_contact_radial_retreat"
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                raise RuntimeError("contact_escape_has_no_valid_direction")
            direction /= norm
            obj_world = np.eye(4, dtype=np.float64)
            obj_world[:3, :3] = Rotation.from_quat(
                np.asarray(before_obj["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
            ).as_matrix()
            obj_world[:3, 3] = before_pos
            robot = before_state["robot_states"][0]
            eef_world = np.eye(4, dtype=np.float64)
            eef_world[:3, :3] = Rotation.from_quat(
                np.asarray(robot["eef_quat"], dtype=np.float64)[[1, 2, 3, 0]]
            ).as_matrix()
            eef_world[:3, 3] = robot["eef_pos"]
            object_to_eef = np.linalg.inv(obj_world) @ eef_world
            target_object_world = obj_world.copy()
            target_object_world[:3, 3] += direction * distance
            target_eef_world = target_object_world @ object_to_eef
            target_eef_quat_wxyz = Rotation.from_matrix(
                target_eef_world[:3, :3],
            ).as_quat()[[3, 0, 1, 2]]
            start_export = bridge.request({"op": "export_runtime_snapshot"})
            if start_export.get("status") != "complete" or not isinstance(start_export.get("payload"), bytes):
                raise RuntimeError("contact_escape_start_snapshot_export_failed")
            (args.output / "contact_escape_start_runtime_snapshot.pkl").write_bytes(start_export["payload"])

            def persist(record: dict, frame: dict) -> None:
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
                    handle.write("\n")
                    handle.flush()
                combined = np.concatenate((frame["agentview_rgb"], frame["wrist_rgb"]), axis=1)
                cv2.imwrite(str(frames_dir / f"frame_{int(frame['step_index']):04d}.jpg"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            action_counter = [0]
            reached = search._advance_pose(
                bridge,
                target_eef_world[:3, 3],
                target_eef_quat_wxyz,
                "verified_robot_scene_contact_normal_escape",
                "stage_a_contact_escape",
                action_counter,
                args.maximum_actions,
                target_binding,
                args.instruction,
                policy,
                monitor,
                rows,
                frames,
                step_callback=persist,
                stagnation_window_actions=16,
                position_tolerance_m=0.008,
                rotation_tolerance_rad=0.06,
                max_translation_step_m=0.008,
                planning_tracking={
                    "tracking_mode": "real_OSC_contact_escape_before_held_route_continuation",
                    "escape_mode": escape_mode,
                    "source_contact": retreat,
                    "escape_direction_world": direction.tolist(),
                    "escape_distance_m": distance,
                    "target_object_pose_world": target_object_world.tolist(),
                    "target_eef_pose_world": np.r_[target_eef_world[:3, 3], target_eef_quat_wxyz].tolist(),
                    "object_or_robot_state_written_directly": False,
                    "safety_certified": False,
                },
                absolute_action_limit=None,
            )
            after = bridge.request({"op": "observe", "include_snapshot_state": True})
            after_state = after["snapshot_state"]
            after_obj = search._object_state(after_state, int(target_binding["body_id"]))
            after_native = bridge.request({"op": "native_task_status"})
            after_grasp = bridge.request({
                "op": "grasp_state",
                "carried_binding": target_binding,
                "expected_instruction": args.instruction,
            })
            final_contacts = bridge.request({"op": "inspect_contacts"})
            final_escape = _robot_scene_contact_escape_direction(
                final_contacts,
                after_state,
                target_body_id=int(target_binding["body_id"]),
                eef_position_world_m=after_state["robot_states"][0]["eef_pos"],
            )
            terminal = bridge.request({"op": "export_runtime_snapshot"})
            terminal_path = args.output / "contact_escape_terminal_runtime_snapshot.pkl"
            if terminal.get("status") != "complete" or not isinstance(terminal.get("payload"), bytes):
                raise RuntimeError("contact_escape_terminal_snapshot_export_failed")
            terminal_path.write_bytes(terminal["payload"])
            report.update({
                "status": "contact_escape_completed" if reached else "contact_escape_stopped_without_pose_convergence",
                "physical_actions": len(rows),
                "escape_distance_m": distance,
                "escape_direction_world": direction.tolist(),
                "escape_mode": escape_mode,
                "target_pose_reached": bool(reached),
                "object_position_before_m": before_pos.tolist(),
                "object_position_after_m": list(after_obj["root_pos"]),
                "object_displacement_m": float(np.linalg.norm(np.asarray(after_obj["root_pos"], dtype=np.float64) - before_pos)),
                "final_grasp_state": after_grasp,
                "final_robot_scene_contact_escape": final_escape,
                "final_native_task_status": after_native,
                "native_task_success": after_native.get("task_success") is True,
                "terminal_snapshot": str(terminal_path),
                "terminal_snapshot_sha256": _sha(terminal["payload"]),
                "records": str(records_path),
                "frames": str(frames_dir),
                "object_or_robot_state_written_directly": False,
            })
            return_code = 0 if (
                reached
                and after_grasp.get("grasp_confirmed") is True
                and after_grasp.get("finger_contacts", {}).get("left") is True
                and after_grasp.get("finger_contacts", {}).get("right") is True
                and after_native.get("status") == "complete"
            ) else 2
        report["archive"] = str(args.output)
        (args.output / "run_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return return_code
    except Exception as exc:
        report["status"] = "contact_escape_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["physical_actions"] = len(rows)
        report["archive"] = str(args.output)
        try:
            current_native = bridge.request({"op": "native_task_status"})
            current_snapshot = bridge.request({"op": "export_runtime_snapshot"})
            report["failure_native_task_status"] = current_native
            if current_snapshot.get("status") == "complete" and isinstance(
                current_snapshot.get("payload"), bytes,
            ):
                terminal = args.output / "contact_escape_failed_runtime_snapshot.pkl"
                terminal.write_bytes(current_snapshot["payload"])
                report["terminal_snapshot"] = str(terminal)
                report["terminal_snapshot_sha256"] = _sha(current_snapshot["payload"])
        except Exception as snapshot_exc:
            report["terminal_snapshot_error"] = f"{type(snapshot_exc).__name__}: {snapshot_exc}"
        if rows and frames:
            search._write_recording(args.output, rows, frames)
        (args.output / "run_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
