# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Bounded object-local grasp screening using existing CuRobo and MuJoCo APIs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from research.gs_opd.libero_safety_repro.keyposes import (
    inverse_transform,
    validate_transform,
)
from research.gs_opd.libero_safety_repro.scene_probe import _json_safe, _read_only_rpc


def local_candidates(reference: np.ndarray, tool_offsets_m: list) -> list[np.ndarray]:
    """Apply configured offsets in the reference tool frame, retaining orientation."""
    reference = validate_transform(reference)
    offsets = np.asarray(tool_offsets_m, dtype=float)
    if offsets.ndim != 2 or offsets.shape[1:] != (3,) or not np.isfinite(offsets).all():
        raise ValueError("finite N by 3 tool offsets required")
    if not 1 <= len(offsets) <= 12:
        raise ValueError("one to twelve local candidates required")
    result = []
    for offset in offsets:
        delta = np.eye(4)
        delta[:3, 3] = offset
        result.append(reference @ delta)
    return result


def first_bilateral_sample(audit: dict, pad_sides: dict) -> dict:
    """Require a contact-free/allowed prefix up to the first bilateral contact."""
    if audit.get("status") != "complete" or not audit.get("physical_state_unchanged"):
        return {"accepted": False, "reason": "incomplete_scratch_audit"}
    for index, state in enumerate(audit["states"]):
        if state["illegal_contact_count"]:
            return {
                "accepted": False,
                "reason": "illegal_contact_before_bilateral",
                "sample_index": index,
            }
        contacted = {
            side
            for c in state["contacts"]
            if c["allowed"]
            for gid in c["geom_ids"]
            for side in [pad_sides.get(gid)]
            if side is not None
        }
        if contacted == {"left", "right"}:
            return {
                "accepted": True,
                "reason": "static_bilateral_geometry_only",
                "sample_index": index,
                "gripper_joint_positions_m": state["gripper_joint_positions_rad"],
            }
    return {
        "accepted": False,
        "reason": "no_bilateral_contact_in_bounded_opening_range",
    }


def main() -> None:
    import torch
    from curobo.types import JointState

    from research.gs_opd.libero_safety_repro import plan_oa_l0t0 as planner_api

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-source", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = json.loads(gzip.decompress(args.scene_source.read_bytes()))
    historical = json.loads(args.reference_report.read_text())
    branch = next(
        b
        for b in historical["physical_branches"]
        if b.get("verified_object_to_eef_at_grasp")
    )
    reference = validate_transform(branch["verified_object_to_eef_at_grasp"])
    offsets = [
        [x, 0.0, -z] for z in [0.0, 0.004, 0.008, 0.012] for x in [0.0, -0.0015, 0.0015]
    ]
    candidates = local_candidates(reference, offsets)
    report = {
        "schema": "libero_safety_local_grasp_screen.v1",
        "status": "running",
        "budget": {
            "expert_pose_candidates": 12,
            "planned_approach_routes": 2,
            "physical_branches": 1,
        },
        "source_scene": str(args.scene_source),
        "reference": {
            "report": str(args.reference_report),
            "sha256": hashlib.sha256(args.reference_report.read_bytes()).hexdigest(),
            "branch": branch["candidate_id"],
            "T_object_eef_measured": reference.tolist(),
            "source_task_success": False,
            "source_safe_grasp": False,
            "source_actions_replayed": False,
            "use": "measured sparse orientation/position reference; current geometry must independently pass",
        },
        "engineering_choices": {
            "tool_offsets_m": offsets,
            "pregrasp_retraction_m": 0.06,
            "jaw_sampling": "0.25 mm increments down to 0.75 mm per jaw; never force zero",
            "not_paper_parameters": True,
        },
        "candidates": [],
        "routes": [],
        "physical_actions": 0,
    }
    report_path = args.output_dir / "grasp_search_report.json"

    def save():
        report_path.write_text(json.dumps(_json_safe(report), indent=2) + "\n")

    save()
    process = log = bridge = None
    try:
        process, log, bridge, initial = planner_api._start_bridge(
            source, args.output_dir
        )
        base = validate_transform(source["planner_frames"]["world_base_transform_m"])
        obj = planner_api._matrix_from_position_rotation(
            source["target_context"]["target_root_pose_world"]["position_m"],
            source["target_context"]["target_root_pose_world"]["rotation"],
        )
        eef_hand = validate_transform(
            source["planner_frames"]["eef_to_panda_hand_transform"]
        )
        jaw_open = source["target_context"]["active_open_gripper_qpos"]
        planner, device, dtype, initial_joint_state, initial_q, planner_config = (
            planner_api._make_planner(
                source, base, "open", 20261027, gripper_qpos_m=jaw_open
            )
        )
        report["planner_configuration"] = planner_config
        pads = {
            g["geom_id"]: side
            for side, rows in source["target_context"][
                "fingertip_pad_geometries"
            ].items()
            for g in rows
        }
        jaw_samples = [jaw_open] + [
            [float(x), -float(x)] for x in np.arange(0.020, 0.00074, -0.00025)
        ]
        accepted = []
        for index, candidate in enumerate(candidates):
            target = inverse_transform(base) @ obj @ candidate @ eef_hand
            ik = planner.ik_solver.solve_pose(
                planner_api._goal_pose(target, device, dtype),
                current_state=initial_joint_state,
                return_seeds=8,
            )
            success = planner_api._tensor(ik.success).reshape(-1)
            positions = planner_api._tensor(ik.js_solution.position)
            solutions = positions.reshape(-1, positions.shape[-1])[:, :7]
            valid = [q for q, ok in zip(solutions, success) if ok]
            row = {
                "candidate_index": index,
                "tool_offset_m": offsets[index],
                "T_object_eef": candidate.tolist(),
                "T_base_panda_hand": target.tolist(),
                "ik_success": success.tolist(),
                "ik_solutions_rad": solutions.tolist(),
            }
            report["candidates"].append(row)
            if not valid:
                row["rejection_reason"] = "no_collision_aware_open_jaw_IK"
                save()
                continue
            q = min(valid, key=lambda q: np.linalg.norm(q - initial_q))
            audit = _read_only_rpc(
                bridge,
                {
                    "op": "scratch_robot_path_contact_audit",
                    "object_id": source["config"]["target_object_id"],
                    "arm_joint_states_rad": [q.tolist()] * len(jaw_samples),
                    "gripper_joint_states_rad": jaw_samples,
                    "fingertip_contact_scope": "all_states",
                },
            )
            acceptance = first_bilateral_sample(audit, pads)
            row.update(
                {
                    "selected_IK_q_rad": q.tolist(),
                    "static_closure_audit": audit,
                    "acceptance": acceptance,
                }
            )
            if acceptance["accepted"]:
                accepted.append(index)
            else:
                row["rejection_reason"] = acceptance["reason"]
            save()
        report["static_geometry_accepted_candidates"] = accepted
        for index in accepted[:2]:
            candidate = candidates[index]
            pre = candidate.copy()
            pre[:3, 3] -= 0.06 * candidate[:3, 2]
            route = {"candidate_index": index, "stages": [], "accepted": False}
            report["routes"].append(route)
            state = initial_joint_state
            for stage, pose in [("pre_grasp", pre), ("grasp", candidate)]:
                target = inverse_transform(base) @ obj @ pose @ eef_hand
                ik_row, plan_row, path, endpoint = planner_api._plan_stage(
                    planner, device, dtype, state, target, grasp_goal=False
                )
                entry = {
                    "stage": stage,
                    "target_T_base_panda_hand": target.tolist(),
                    "ik": ik_row,
                    "curobo": plan_row,
                }
                route["stages"].append(entry)
                if path is None or not plan_row["success"]:
                    route["rejection_reason"] = f"{stage}:CuRobo_no_route"
                    break
                # Audit intermediate joint configurations as well as returned samples.
                dense = []
                for a, b in zip(path[:-1], path[1:]):
                    n = max(1, int(np.ceil(np.max(np.abs(b - a)) / 0.01)))
                    dense.extend(a + (b - a) * t for t in np.arange(n) / n)
                dense.append(path[-1])
                audit = planner_api._open_bridge_audit(
                    bridge,
                    np.asarray(dense),
                    jaw_open,
                    source["config"]["target_object_id"],
                    scope="none",
                )
                entry["exact_open_jaw_path_audit"] = audit
                if (
                    audit.get("status") != "complete"
                    or not audit.get("physical_state_unchanged")
                    or audit.get("illegal_contact_count") != 0
                ):
                    route["rejection_reason"] = f"{stage}:exact_MuJoCo_path_contact"
                    break
                if stage == "grasp":
                    audit = _read_only_rpc(
                        bridge,
                        {
                            "op": "scratch_robot_path_contact_audit",
                            "object_id": source["config"]["target_object_id"],
                            "arm_joint_states_rad": [path[-1].tolist()]
                            * len(jaw_samples),
                            "gripper_joint_states_rad": jaw_samples,
                            "fingertip_contact_scope": "all_states",
                        },
                    )
                    entry["closure_audit_at_planned_endpoint"] = audit
                    entry["closure_acceptance"] = first_bilateral_sample(audit, pads)
                    route["accepted"] = entry["closure_acceptance"]["accepted"]
                state = JointState.from_position(
                    torch.as_tensor(path[-1:, :], device=device, dtype=dtype),
                    joint_names=planner.kinematics.joint_names,
                )
            save()
        report["status"] = (
            "approach_route_available"
            if any(r["accepted"] for r in report["routes"])
            else "blocked"
        )
        final = _read_only_rpc(
            bridge, {"op": "observe", "include_snapshot_state": True}
        )
        report["live_state_unchanged"] = np.array_equal(
            initial["snapshot_state"]["mujoco_integration_state"],
            final["snapshot_state"]["mujoco_integration_state"],
        )
        report["timestep"] = final["timestep"]
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        if bridge is not None:
            bridge.close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
                process.wait()
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
