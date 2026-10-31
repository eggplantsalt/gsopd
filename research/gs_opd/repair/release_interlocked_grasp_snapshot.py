#!/usr/bin/env python3
"""Physically release an interlocked held object and archive an exact branch Snapshot.

This Stage A diagnostic only performs ordinary OSC gripper and pose actions in
MuJoCo. It never writes robot or object state directly. The output Snapshot is
intended for a same-episode cold-start continuation in the same archive root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.repair.task_success_first_grasp_search import (
    BridgeSession,
    _PythonObjectState,
    _contact_substep_monitoring_complete,
    _snapshot_state_vector,
    _state_sha,
    _verified_resume_runtime_snapshot,
    configure_stage_a_contact_only_substep_audit,
)
from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _target_state(snapshot_state: dict[str, Any], body_id: int) -> dict[str, Any]:
    matches = [
        row for row in snapshot_state.get("object_states", {}).values()
        if int(row.get("body_id", -1)) == int(body_id)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"target_body_resolved_to_{len(matches)}_runtime_objects")
    return matches[0]


def _contact_pairs(reply: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in reply.get("contacts", {}).get("contacts", []):
        geom1 = row.get("geom1", {})
        geom2 = row.get("geom2", {})
        if any(token in str(geom1.get("name", "")) + str(geom2.get("name", ""))
               for token in ("alphabet_soup_1", "ketchup_1", "robot0_", "gripper0_")):
            result.append({
                "geom1": geom1.get("name"), "geom2": geom2.get("name"),
                "body1": row.get("body1", {}).get("name"),
                "body2": row.get("body2", {}).get("name"),
                "position_world_m": row.get("position"),
                "signed_distance_m": row.get("signed_contact_distance_m"),
                "normal_geom1_to_geom2": row.get("normal_geom1_to_geom2"),
            })
    return result


def _append_record(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.source_snapshot.resolve()
    source_root = args.snapshot_root.resolve()
    manifest = json.loads((source_root / "snapshots/runtime_snapshot_manifest.json").read_text())
    meta = manifest.get("scene_metadata", {})
    expected_t0 = source_root / "snapshots/initial_runtime_snapshot.pkl"
    from research.gs_opd.repair.task_success_first_grasp_search import _PythonObjectState as state_type
    import pickle
    import sys
    setattr(sys.modules["__main__"], "_PythonObjectState", state_type)
    t0_envelope = pickle.loads(expected_t0.read_bytes())
    model_sha = str(t0_envelope.get("model_geometry_sha256"))
    bddl_sha = str(meta.get("bddl_sha256"))
    source_descriptor = _verified_resume_runtime_snapshot(
        source,
        benchmark=args.benchmark,
        level=args.level,
        task_id=args.task_id,
        seed=args.seed,
        init_state=args.init_state,
        model_sha=model_sha,
        bddl_sha=bddl_sha,
        instruction=str(meta.get("instruction")),
    )
    configure_stage_a_contact_only_substep_audit(True)
    bridge_args = argparse.Namespace(
        benchmark=args.benchmark, level=args.level, task_id=args.task_id,
        init_state=args.init_state, seed=args.seed,
        camera_resolution=args.camera_resolution,
        demonstration_horizon=args.demonstration_horizon,
    )
    bridge = BridgeSession(("127.0.0.1", _port()), output / "safety_bridge.log", bridge_args)
    records_path = output / "physical_step_records.jsonl"
    rows: list[dict[str, Any]] = []
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        if reset.get("scene_metadata", {}).get("model_geometry_sha256") != model_sha:
            raise RuntimeError("fresh_reset_model_fingerprint_mismatch")
        if reset.get("scene_metadata", {}).get("bddl_sha256") != bddl_sha:
            raise RuntimeError("fresh_reset_BDDL_fingerprint_mismatch")
        source_state = _snapshot_state_vector(source_descriptor["snapshot_state"])
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": source_descriptor["payload"]})
        restored_state = _snapshot_state_vector(restored.get("snapshot_state", {}))
        if (restored.get("status") != "restored" or restored.get("model_geometry_sha256") != model_sha
                or not np.array_equal(source_state, restored_state)):
            raise RuntimeError("source_snapshot_exact_restore_failed")
        context = bridge.request({
            "op": "task_success_first_context",
            "expected_instruction": str(meta["instruction"]),
            "target_predicate": ["in", "alphabet_soup_1", "basket_1_contain_region"],
        })
        if context.get("status") != "complete" or context.get("native_goal_value") is not False:
            raise RuntimeError("active_soup_native_goal_context_unavailable_or_already_true")
        native_start = bridge.request({"op": "native_task_status"})
        if native_start.get("task_success") is True:
            raise RuntimeError("source_snapshot_already_has_native_task_success")
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        target_semantic = semantic.get("bindings", {}).get(str(context["target_object_id"]))
        if not isinstance(target_semantic, dict):
            raise RuntimeError("target_semantic_geometry_binding_unavailable")
        target_binding = {
            "object_id": str(context["target_object_id"]),
            "body_id": int(target_semantic["root_body_id"]),
            "body_name": str(target_semantic["root_body_name"]),
            "geom_ids": sorted(int(value) for value in target_semantic["root_geom_ids"]),
            "geom_names": list(target_semantic["root_geom_names"]),
        }
        instruction = str(meta["instruction"])
        grasp = bridge.request({
            "op": "grasp_state", "carried_binding": target_binding,
            "expected_instruction": instruction,
        })
        if not (grasp.get("grasp_confirmed") is True
                and grasp.get("finger_contacts", {}).get("left") is True
                and grasp.get("finger_contacts", {}).get("right") is True):
            raise RuntimeError("source_snapshot_does_not_retain_verified_bilateral_target_grasp")
        inventory = bridge.request({"op": "repair_context"})
        controller_fields = inventory["robots"][0]["fields"]
        observation = bridge.request({"op": "observe", "include_snapshot_state": True})
        current_state = observation["snapshot_state"]
        initial_eef = current_state["robot_states"][0]
        retreat_target = np.asarray(initial_eef["eef_pos"], dtype=np.float64) + np.asarray([0.0, -args.retreat_y_m, 0.0])
        hold_quat = np.asarray(initial_eef["eef_quat"], dtype=np.float64)
        initial_object = _target_state(current_state, target_binding["body_id"])
        initial_ketchup = next(
            (row for row in current_state.get("object_states", {}).values()
             if str(row.get("body_name", "")) == "ketchup_1_main"), None,
        )
        _write_json(output / "source_verification.json", {
            "source_snapshot": str(source),
            "source_snapshot_sha256": source_descriptor["payload_sha256"],
            "source_integration_state_sha256": source_descriptor["integration_state_sha256"],
            "fresh_reset_model_BDDL_match": True,
            "exact_restore": True,
            "source_timestep": int(source_state[0]),
            "native_task_status": native_start,
            "native_target_predicate": context["native_goal_predicate"],
            "initial_bilateral_grasp": grasp,
            "initial_eef_position_world_m": initial_eef["eef_pos"],
            "initial_target_root_position_world_m": initial_object.get("root_pos"),
            "initial_ketchup_root_position_world_m": initial_ketchup.get("root_pos") if initial_ketchup else None,
            "retreat_hypothesis": "release target on its physical support, then move open EEF -Y away from the bottle interlock",
            "retreat_target_eef_position_world_m": retreat_target.tolist(),
            "robot_or_object_state_directly_written": False,
            "official_demos_or_success_actions_used": False,
            "safety_certified": False,
        })

        def step(phase: str, action: np.ndarray, index: int) -> dict[str, Any]:
            reply = bridge.request({
                "op": "step_monitored",
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "action_index": int(index),
                "include_grasp_state": True,
                "grasp_binding": target_binding,
                "expected_instruction": instruction,
                "safety_geometry_policy": None,
            })
            status = bridge.request({"op": "native_task_status"})
            snapshot_state = reply.get("snapshot_state", {})
            robot = snapshot_state.get("robot_states", [{}])[0]
            target = _target_state(snapshot_state, target_binding["body_id"])
            ketchup = next(
                (row for row in snapshot_state.get("object_states", {}).values()
                 if str(row.get("body_name", "")) == "ketchup_1_main"), None,
            )
            monitoring = reply.get("contact_monitoring", {})
            record = {
                "phase": phase,
                "action_index": int(index),
                "action_7d": np.asarray(action, dtype=np.float32).tolist(),
                "timestep": reply.get("timestep"),
                "reward": reply.get("reward"), "done": reply.get("done"),
                "task_success": reply.get("task_success"),
                "native_task_status": status,
                "eef_position_world_m": robot.get("eef_pos"),
                "eef_quaternion_wxyz": robot.get("eef_quat"),
                "target_root_position_world_m": target.get("root_pos"),
                "target_root_quaternion_wxyz": target.get("root_quat"),
                "ketchup_root_position_world_m": ketchup.get("root_pos") if ketchup else None,
                "grasp_state": reply.get("grasp_state"),
                "contacts_of_interest": _contact_pairs(reply),
                "contact_monitoring": monitoring,
                "substep_contact_monitoring_complete": _contact_substep_monitoring_complete(
                    monitoring, require_geometry=False,
                ),
                "snapshot_state": snapshot_state,
                "state_write": False,
                "safety_certified": False,
            }
            _append_record(records_path, record)
            rows.append(record)
            if reply.get("done"):
                raise RuntimeError("environment_terminated_during_interlocked_release_branch")
            return record

        # Open before moving the wrist so the intended object is no longer
        # mechanically attached to the robot while it escapes the movable bottle.
        for i in range(args.open_actions):
            step("physical_gripper_open", np.asarray([0, 0, 0, 0, 0, 0, -1.0]), i)

        latest = bridge.request({"op": "observe", "include_snapshot_state": True})
        release_state = _snapshot_state_vector(latest["snapshot_state"])
        released_grasp = bridge.request({
            "op": "grasp_state", "carried_binding": target_binding,
            "expected_instruction": instruction,
        })
        if released_grasp.get("grasp_confirmed") is True:
            raise RuntimeError("MuJoCo_gripper_open_actions_did_not_release_target")
        current_eef = latest["snapshot_state"]["robot_states"][0]
        retreat_actions = 0
        for j in range(args.retreat_actions):
            latest = bridge.request({"op": "observe", "include_snapshot_state": True})
            robot = latest["snapshot_state"]["robot_states"][0]
            current_position = np.asarray(robot["eef_pos"], dtype=np.float64)
            remaining = retreat_target - current_position
            remaining_norm = float(np.linalg.norm(remaining))
            if remaining_norm <= args.position_tolerance_m:
                break
            # OSC accepts a bounded normalized delta per action. Use a
            # receding 8 mm physical waypoint instead of sending the whole
            # 100 mm retreat in one controller command.
            local_target = current_position + remaining * min(
                1.0, args.max_translation_step_m / remaining_norm,
            )
            action = world_pose_tracking_action(
                robot["eef_pos"], robot["eef_quat"], local_target, hold_quat,
                controller_fields, gripper_action=-1.0,
            )
            step("open_gripper_lateral_escape_minus_y", action, args.open_actions + j)
            retreat_actions += 1
            post = rows[-1]
            error = float(np.linalg.norm(retreat_target - np.asarray(post["eef_position_world_m"], dtype=np.float64)))
            if error <= args.position_tolerance_m:
                break

        # A short settling interval confirms the object remains governed by
        # MuJoCo contact/gravity and records the native BDDL state at every step.
        for k in range(args.settle_actions):
            step("released_object_physical_settle", np.asarray([0, 0, 0, 0, 0, 0, -1.0]), args.open_actions + retreat_actions + k)

        final_export = bridge.request({"op": "export_runtime_snapshot"})
        if final_export.get("status") != "complete" or not isinstance(final_export.get("payload"), bytes):
            raise RuntimeError("released_branch_runtime_snapshot_export_failed")
        final_state = _snapshot_state_vector(final_export.get("snapshot_state", {}))
        if rows and not np.array_equal(final_state, _snapshot_state_vector(rows[-1]["snapshot_state"])):
            raise RuntimeError("released_branch_snapshot_does_not_match_last_physical_action")
        final_grasp = bridge.request({
            "op": "grasp_state", "carried_binding": target_binding,
            "expected_instruction": instruction,
        })
        final_native = bridge.request({"op": "native_task_status"})
        if final_grasp.get("grasp_confirmed") is True:
            raise RuntimeError("target_remained_attached_after_release_and_retreat")
        if final_native.get("task_success") is True:
            raise RuntimeError("unexpected_native_task_success_during_diagnostic_release")
        snapshot_path = output / "released_regrasp_start_runtime_snapshot.pkl"
        snapshot_path.write_bytes(final_export["payload"])
        report = {
            "schema_version": "gs_opd_interlocked_grasp_release.v1",
            "status": "released_snapshot_ready_for_same_episode_regrasp",
            "task": {"benchmark": args.benchmark, "level": args.level, "task_id": args.task_id,
                     "seed": args.seed, "init_state": args.init_state},
            "source_snapshot": str(source),
            "source_snapshot_sha256": source_descriptor["payload_sha256"],
            "released_snapshot": str(snapshot_path),
            "released_snapshot_sha256": _sha(final_export["payload"]),
            "released_integration_state_sha256": _state_sha(final_state),
            "action_start_timestep": int(native_start.get("timestep", -1)) + 1,
            # The first integration-state element is MuJoCo simulation time,
            # not the task's discrete BDDL action index.
            "action_end_timestep": int(final_native.get("timestep", -1)),
            "actions_executed": len(rows),
            "contact_substeps_expected": len(rows) * 25,
            "contact_substep_rows_complete": sum(
                bool(row["substep_contact_monitoring_complete"]) for row in rows
            ),
            "final_native_task_status": final_native,
            "final_target_grasp_state": final_grasp,
            "final_target_root_position_world_m": rows[-1]["target_root_position_world_m"] if rows else None,
            "final_ketchup_root_position_world_m": rows[-1]["ketchup_root_position_world_m"] if rows else None,
            "final_eef_position_world_m": rows[-1]["eef_position_world_m"] if rows else None,
            "records": str(records_path),
            "robot_or_object_state_directly_written": False,
            "official_demos_or_success_actions_used": False,
            "safety_certified": False,
        }
        _write_json(output / "run_report.json", report)
        return report
    except Exception as exc:
        report = {
            "schema_version": "gs_opd_interlocked_grasp_release.v1",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "source_snapshot": str(source),
            "actions_executed": len(rows),
            "records": str(records_path),
            "safety_certified": False,
        }
        _write_json(output / "run_report.json", report)
        raise
    finally:
        bridge.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="obstacle_avoidance_human")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--demonstration-horizon", type=int, default=1800)
    parser.add_argument("--open-actions", type=int, default=8)
    parser.add_argument("--retreat-actions", type=int, default=40)
    parser.add_argument("--retreat-y-m", type=float, default=0.10)
    parser.add_argument("--max-translation-step-m", type=float, default=0.008)
    parser.add_argument("--settle-actions", type=int, default=4)
    parser.add_argument("--position-tolerance-m", type=float, default=0.006)
    args = parser.parse_args()
    if not 1 <= args.open_actions <= 16 or not 1 <= args.retreat_actions <= 100:
        parser.error("open/retreat action budgets are outside supported bounds")
    if (not 0.01 <= args.retreat_y_m <= 0.20 or not 0 <= args.settle_actions <= 32
            or not 0.001 <= args.max_translation_step_m <= 0.02):
        parser.error("retreat displacement/settling budget is outside supported bounds")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
