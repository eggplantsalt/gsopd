#!/usr/bin/env python3
"""Compare short real-MuJoCo escape motions from one exact contact Snapshot.

Each probe is a mutually exclusive branch restored from the identical
Runtime Snapshot. Directions are generated in Cartesian tool pose space, and
all outcomes are ranked from live MuJoCo contacts and target displacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

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


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _body(snapshot_state: dict[str, Any], body_name: str) -> dict[str, Any]:
    rows = [row for row in snapshot_state.get("object_states", {}).values()
            if str(row.get("body_name", "")) == body_name]
    if len(rows) != 1:
        raise RuntimeError(f"runtime_body_{body_name}_resolved_to_{len(rows)}_rows")
    return rows[0]


def _robot_ketchup_contacts(reply: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for contact in reply.get("contacts", []):
        geom1 = contact.get("geom1", {})
        geom2 = contact.get("geom2", {})
        name1, name2 = str(geom1.get("name", "")), str(geom2.get("name", ""))
        if "ketchup_1" not in name1 + name2 or not any(
            token in name1 + name2 for token in ("robot0_", "gripper0_")
        ):
            continue
        output.append({
            "geom1": name1, "geom2": name2,
            "signed_contact_distance_m": float(contact.get("signed_contact_distance_m", 0.0)),
            "normal_geom1_to_geom2": contact.get("normal_geom1_to_geom2"),
            "position_world_m": contact.get("position"),
        })
    return output


def _score(contacts: list[dict[str, Any]]) -> tuple[int, float]:
    return len(contacts), float(sum(max(0.0, -row["signed_contact_distance_m"]) for row in contacts))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.source_snapshot.resolve()
    source_root = args.snapshot_root.resolve()
    manifest = json.loads((source_root / "snapshots/runtime_snapshot_manifest.json").read_text())
    scene = manifest["scene_metadata"]
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    t0 = pickle.loads((source_root / "snapshots/initial_runtime_snapshot.pkl").read_bytes())
    model_sha = str(t0.get("model_geometry_sha256"))
    bddl_sha = str(scene.get("bddl_sha256"))
    descriptor = _verified_resume_runtime_snapshot(
        source, benchmark=args.benchmark, level=args.level, task_id=args.task_id,
        seed=args.seed, init_state=args.init_state, model_sha=model_sha,
        bddl_sha=bddl_sha, instruction=str(scene["instruction"]),
    )
    configure_stage_a_contact_only_substep_audit(True)
    bridge_args = argparse.Namespace(
        benchmark=args.benchmark, level=args.level, task_id=args.task_id,
        init_state=args.init_state, seed=args.seed,
        camera_resolution=args.camera_resolution,
        demonstration_horizon=args.demonstration_horizon,
    )
    bridge = BridgeSession(("127.0.0.1", _port()), output / "safety_bridge.log", bridge_args)
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        if (reset.get("scene_metadata", {}).get("model_geometry_sha256") != model_sha
                or reset.get("scene_metadata", {}).get("bddl_sha256") != bddl_sha):
            raise RuntimeError("probe_scene_identity_differs_from_source_snapshot")
        source_state = _snapshot_state_vector(descriptor["snapshot_state"])
        native_source = None
        branches: list[dict[str, Any]] = []
        controller = bridge.request({"op": "repair_context"})
        controller_fields = controller["robots"][0]["fields"]
        direction_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
        for axis_name, axis in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
            vector = np.asarray(axis, dtype=np.float64)
            direction_rows.append((f"translation_{axis_name}_plus", vector * args.translation_step_m, np.zeros(3)))
            direction_rows.append((f"translation_{axis_name}_minus", -vector * args.translation_step_m, np.zeros(3)))
        for axis_name, axis in (("roll_x", [1, 0, 0]), ("pitch_y", [0, 1, 0]), ("yaw_z", [0, 0, 1])):
            vector = np.asarray(axis, dtype=np.float64)
            direction_rows.append((f"rotation_{axis_name}_plus", np.zeros(3), vector * args.rotation_step_rad))
            direction_rows.append((f"rotation_{axis_name}_minus", np.zeros(3), -vector * args.rotation_step_rad))

        for branch_index, (name, translation, rotation_world) in enumerate(direction_rows):
            restored = bridge.request({"op": "restore_runtime_snapshot", "payload": descriptor["payload"]})
            restored_state = _snapshot_state_vector(restored.get("snapshot_state", {}))
            if (restored.get("status") != "restored" or restored.get("model_geometry_sha256") != model_sha
                    or not np.array_equal(restored_state, source_state)):
                raise RuntimeError(f"probe_branch_{name}_exact_restore_failed")
            native_before = bridge.request({"op": "native_task_status"})
            if native_before.get("task_success") is True:
                raise RuntimeError("contact_escape_probe_source_already_has_native_task_success")
            if native_source is None:
                native_source = native_before
            start_contacts_reply = bridge.request({"op": "inspect_contacts"})
            start_contacts = _robot_ketchup_contacts(start_contacts_reply)
            observed = bridge.request({"op": "observe", "include_snapshot_state": True})
            snapshot_state = observed["snapshot_state"]
            robot = snapshot_state["robot_states"][0]
            eef_pos = np.asarray(robot["eef_pos"], dtype=np.float64)
            eef_quat = np.asarray(robot["eef_quat"], dtype=np.float64)
            target_pos = eef_pos + translation
            delta_rot = Rotation.from_rotvec(rotation_world)
            current_rot = Rotation.from_quat(eef_quat[[1, 2, 3, 0]])
            target_rot = delta_rot * current_rot
            target_quat_xyzw = target_rot.as_quat()
            target_quat = target_quat_xyzw[[3, 0, 1, 2]]
            action = world_pose_tracking_action(
                eef_pos, eef_quat, target_pos, target_quat,
                controller_fields, gripper_action=-1.0,
            ).astype(np.float32)
            step = bridge.request({
                "op": "step_monitored", "action": action.tolist(),
                "action_index": branch_index,
                "include_grasp_state": False,
                "safety_geometry_policy": None,
            })
            if step.get("done"):
                raise RuntimeError(f"probe_branch_{name}_terminated_episode")
            native = bridge.request({"op": "native_task_status"})
            if native_source is None:
                raise RuntimeError("contact_escape_probe_source_native_status_missing")
            contacts_reply = bridge.request({"op": "inspect_contacts"})
            contacts = _robot_ketchup_contacts(contacts_reply)
            contact_count, total_penetration = _score(contacts)
            new_state = step.get("snapshot_state", {})
            after_robot = new_state.get("robot_states", [{}])[0]
            soup = _body(new_state, "alphabet_soup_1_main")
            ketchup = _body(new_state, "ketchup_1_main")
            source_soup = _body(snapshot_state, "alphabet_soup_1_main")
            source_ketchup = _body(snapshot_state, "ketchup_1_main")
            export = bridge.request({"op": "export_runtime_snapshot"})
            exported_state = _snapshot_state_vector(export.get("snapshot_state", {}))
            step_state = _snapshot_state_vector(new_state)
            if (export.get("status") != "complete" or export.get("model_geometry_sha256") != model_sha
                    or not np.array_equal(exported_state, step_state)):
                raise RuntimeError(f"probe_branch_{name}_terminal_snapshot_mismatch")
            branch_dir = output / f"branch_{branch_index:02d}_{name}"
            branch_dir.mkdir()
            snapshot_path = branch_dir / "terminal_runtime_snapshot.pkl"
            snapshot_path.write_bytes(export["payload"])
            record = {
                "schema_version": "gs_opd_contact_escape_probe_action.v1",
                "branch": name,
                "branch_index": branch_index,
                "source_snapshot": str(source),
                "source_snapshot_sha256": descriptor["payload_sha256"],
                "source_integration_state_sha256": descriptor["integration_state_sha256"],
                "source_timestep": int(native_source.get("timestep", -1)),
                "target_eef_translation_m": translation.tolist(),
                "target_eef_rotation_world_rad": rotation_world.tolist(),
                "action_7d": action.tolist(),
                "terminal_timestep": int(native.get("timestep", -1)),
                "native_task_status": native,
                "robot_start_contacts_with_ketchup": start_contacts,
                "robot_ketchup_contact_count_after": contact_count,
                "robot_ketchup_total_penetration_m_after": total_penetration,
                "robot_ketchup_contacts_after": contacts,
                "target_displacement_m": float(np.linalg.norm(
                    np.asarray(soup["root_pos"], dtype=np.float64)
                    - np.asarray(source_soup["root_pos"], dtype=np.float64)
                )),
                "ketchup_displacement_m": float(np.linalg.norm(
                    np.asarray(ketchup["root_pos"], dtype=np.float64)
                    - np.asarray(source_ketchup["root_pos"], dtype=np.float64)
                )),
                "eef_displacement_m": float(np.linalg.norm(
                    np.asarray(after_robot["eef_pos"], dtype=np.float64) - eef_pos
                )),
                "contact_monitoring": step.get("contact_monitoring"),
                "substep_contact_monitoring_complete": _contact_substep_monitoring_complete(
                    step.get("contact_monitoring", {}), require_geometry=False,
                ),
                "terminal_runtime_snapshot": str(snapshot_path),
                "terminal_runtime_snapshot_sha256": _sha(export["payload"]),
                "terminal_integration_state_sha256": _state_sha(exported_state),
                "direct_state_write": False,
                "safety_certified": False,
            }
            _write_json(branch_dir / "probe_result.json", record)
            branches.append(record)

        branches.sort(key=lambda row: (
            row["robot_ketchup_contact_count_after"],
            row["robot_ketchup_total_penetration_m_after"],
            row["target_displacement_m"],
        ))
        report = {
            "schema_version": "gs_opd_interlocked_contact_escape_probe.v1",
            "status": "all_directional_probes_complete",
            "task": {"benchmark": args.benchmark, "level": args.level,
                     "task_id": args.task_id, "seed": args.seed, "init_state": args.init_state},
            "source_snapshot": str(source),
            "source_snapshot_sha256": descriptor["payload_sha256"],
            "source_integration_state_sha256": descriptor["integration_state_sha256"],
            "source_native_task_status": native_source,
            "directional_physical_probe_count": len(branches),
            "complete_contact_substep_probe_count": sum(
                row["substep_contact_monitoring_complete"] for row in branches
            ),
            "best_branch_by_contact_clearance_then_target_displacement": branches[0],
            "all_branches": branches,
            "branching_method": "each OSC probe restored the identical source Runtime Snapshot; sibling branch actions are not stitched",
            "official_demos_or_success_actions_used": False,
            "direct_state_write": False,
            "safety_certified": False,
        }
        _write_json(output / "probe_report.json", report)
        return report
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
    parser.add_argument("--translation-step-m", type=float, default=0.008)
    parser.add_argument("--rotation-step-rad", type=float, default=0.12)
    args = parser.parse_args()
    if not 0.001 <= args.translation_step_m <= 0.02 or not 0.01 <= args.rotation_step_rad <= 0.35:
        parser.error("probe displacement outside validated OSC bounds")
    report = run(args)
    print(json.dumps({"status": report["status"], "best_branch": report["best_branch_by_contact_clearance_then_target_displacement"]["branch"],
                      "contact_count": report["best_branch_by_contact_clearance_then_target_displacement"]["robot_ketchup_contact_count_after"],
                      "archive": str(args.output_dir.resolve() / "probe_report.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
