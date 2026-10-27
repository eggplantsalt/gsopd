# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Plan bounded open-jaw approaches; physical grasp remains unproven."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from curobo.types import JointState

from research.gs_opd.libero_safety_repro import plan_oa_l0t0 as api
from research.gs_opd.libero_safety_repro.keyposes import inverse_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-report", type=Path, required=True)
    parser.add_argument("--candidate", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) > 2 or len(set(args.candidate)) != len(args.candidate):
        raise ValueError("at most two distinct screened candidates")
    screen = json.loads(args.screen_report.read_text())
    source = json.loads(gzip.decompress(Path(screen["source_scene"]).read_bytes()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "source_screen_report": str(args.screen_report),
        "source_scene": screen["source_scene"],
        "status": "running",
        "routes": [],
        "physical_actions": 0,
        "acceptance_scope": "open-jaw approach only; closure must use live state and per-step safety gates",
    }
    process, log, bridge, initial = api._start_bridge(source, args.output_dir)
    try:
        base = np.asarray(source["planner_frames"]["world_base_transform_m"])
        obj = api._matrix_from_position_rotation(
            source["target_context"]["target_root_pose_world"]["position_m"],
            source["target_context"]["target_root_pose_world"]["rotation"],
        )
        eef_hand = np.asarray(source["planner_frames"]["eef_to_panda_hand_transform"])
        jaw = source["target_context"]["active_open_gripper_qpos"]
        planner, device, dtype, start, _, config = api._make_planner(
            source, base, "open", 20261027, gripper_qpos_m=jaw
        )
        report["planner_configuration"] = config
        for index in args.candidate:
            candidate = next(
                c for c in screen["candidates"] if c["candidate_index"] == index
            )
            grasp = np.asarray(candidate["T_object_eef"])
            pre = grasp.copy()
            pre[:3, 3] -= 0.06 * grasp[:3, 2]
            route = {
                "candidate_index": index,
                "T_object_eef": grasp.tolist(),
                "stages": [],
                "accepted": False,
            }
            report["routes"].append(route)
            state = start
            for phase, pose in [("pre_grasp", pre), ("grasp", grasp)]:
                target = inverse_transform(base) @ obj @ pose @ eef_hand
                ik, plan, path, endpoint = api._plan_stage(
                    planner, device, dtype, state, target, grasp_goal=False
                )
                row = {
                    "phase": phase,
                    "target_T_base_panda_hand": target.tolist(),
                    "ik": ik,
                    "curobo": plan,
                }
                route["stages"].append(row)
                if path is None or not plan["success"]:
                    route["rejection_reason"] = f"{phase}:CuRobo_no_path"
                    break
                dense = []
                for a, b in zip(path[:-1], path[1:]):
                    n = max(1, int(np.ceil(np.max(np.abs(b - a)) / 0.01)))
                    dense.extend(a + (b - a) * t for t in np.arange(n) / n)
                dense.append(path[-1])
                row["audits"] = []
                for jaw_q in [jaw, [0.02, -0.02]]:
                    audit = api._open_bridge_audit(
                        bridge,
                        np.asarray(dense),
                        jaw_q,
                        source["config"]["target_object_id"],
                        scope="none",
                    )
                    row["audits"].append(audit)
                if any(
                    a.get("status") != "complete"
                    or not a.get("physical_state_unchanged")
                    or a.get("illegal_contact_count") != 0
                    for a in row["audits"]
                ):
                    route["rejection_reason"] = f"{phase}:exact_contact_audit_failed"
                    break
                row["endpoint_error"] = api._pose_error(target, endpoint)
                state = JointState.from_position(
                    torch.as_tensor(path[-1:, :], device=device, dtype=dtype),
                    joint_names=planner.kinematics.joint_names,
                )
            else:
                route["accepted"] = True
            (args.output_dir / "approach_report.json").write_text(
                json.dumps(api._json_safe(report), indent=2) + "\n"
            )
            if route["accepted"]:
                break
        report["status"] = (
            "approach_available"
            if any(r["accepted"] for r in report["routes"])
            else "blocked"
        )
        final = api._rpc(bridge, {"op": "observe", "include_snapshot_state": True})
        report["live_state_unchanged"] = np.array_equal(
            initial["snapshot_state"]["mujoco_integration_state"],
            final["snapshot_state"]["mujoco_integration_state"],
        )
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.output_dir / "approach_report.json").write_text(
            json.dumps(api._json_safe(report), indent=2) + "\n"
        )
        bridge.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except Exception:
            process.kill()
            process.wait()
        log.close()


if __name__ == "__main__":
    main()
