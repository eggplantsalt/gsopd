# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Separate contact manifold points, unique geom pairs, and static samples."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from research.gs_opd.libero_safety_repro.scene_probe import _json_safe, _read_only_rpc


def aggregate_pairs(states: list[dict], source: dict) -> list[dict]:
    """Resolve identities against the archived compiled model; retain every distance."""
    inventory = source["collision_scene_export"]["geometry_inventory"]
    if inventory["status"] != "complete":
        raise ValueError("complete compiled geometry inventory required")
    geoms = {g["geom_id"]: g for g in inventory["geometries"]}
    pads = {
        g["geom_id"]: side
        for side, group in source["target_context"]["fingertip_pad_geometries"].items()
        for g in group
    }
    target = {
        g["geom_id"] for g in source["target_context"]["target_collision_geometries"]
    }
    pairs = {}
    for index, state in enumerate(states):
        for contact in state["contacts"]:
            ids = contact["geom_ids"]
            for i, gid in enumerate(ids):
                if (
                    geoms[gid]["geom_name"] != contact["geom_names"][i]
                    or geoms[gid]["body_id"] != contact["body_ids"][i]
                ):
                    raise ValueError("contact identity differs from compiled model")
            key = tuple(sorted(ids))
            if key not in pairs:
                identities = []
                for gid in key:
                    g = geoms[gid]
                    identities.append(
                        {
                            **g,
                            "finger_pad_side": pads.get(gid),
                            "is_target": gid in target,
                            "robot_link_mujoco": g["body_name"]
                            if g["category"] == "robot"
                            else None,
                            "curobo_link": None,
                        }
                    )
                pairs[key] = {
                    "geom_ids": list(key),
                    "identities": identities,
                    "robot_self_collision": all(
                        g["category"] == "robot" for g in identities
                    ),
                    "records": [],
                }
            pairs[key]["records"].append(
                {
                    "sample_index": index,
                    "gripper_joint_positions_m": state.get(
                        "gripper_joint_positions_rad"
                    ),
                    "signed_distance_m": contact["distance_m"],
                    "allowed": contact["allowed"],
                }
            )
    for pair in pairs.values():
        records = pair["records"]
        pair.update(
            {
                "contact_record_count": len(records),
                "sample_count": len({r["sample_index"] for r in records}),
                "illegal_record_count": sum(not r["allowed"] for r in records),
                "minimum_signed_distance_m": min(
                    r["signed_distance_m"] for r in records
                ),
                "deepest_penetration_m": max(
                    0.0, -min(r["signed_distance_m"] for r in records)
                ),
            }
        )
    return [pairs[key] for key in sorted(pairs)]


def archived_diagnosis(route: dict, source: dict) -> list[dict]:
    """Read the original single forced-zero-joint sample for each bundle."""
    rows = []
    for bundle in route["route_bundles"]:
        grasp = next(s for s in bundle["stages"] if s["stage"] == "grasp")
        audit = grasp["closed_gripper_contact_check"]
        pairs = aggregate_pairs(
            [
                {
                    "gripper_joint_positions_rad": [0.0, 0.0],
                    "contacts": audit["contacts"],
                }
            ],
            source,
        )
        rows.append(
            {
                "bundle_seed": bundle["bundle_seed"],
                "selected_grasp_variant_id": grasp["selected_variant_id"],
                "sample_count": 1,
                "measurement": "scratch qpos assignment and mj_forward; no motor integration",
                "jaw_joint_unit": "meter (prismatic); legacy RPC suffix _rad is inaccurate",
                "illegal_contact_record_count": audit["illegal_contact_count"],
                "illegal_unique_pair_count": sum(
                    p["illegal_record_count"] > 0 for p in pairs
                ),
                "pairs": pairs,
            }
        )
    return rows


def live_controls(route: dict, source: dict, output: Path) -> dict:
    """Compare static jaw openings at fixed arm poses without changing live physics."""
    from research.gs_opd.libero_safety_repro.plan_oa_l0t0 import _start_bridge

    process, log, connection, initial = _start_bridge(source, output)
    result = {"physical_actions": 0, "controls": []}
    try:
        original = np.asarray(source["target_context"]["active_open_gripper_qpos"])
        # A small deterministic geometry diagnostic, not simulated closure dynamics.
        jaw = [original.tolist()] + [
            [x, -x] for x in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.00025, 0.0]
        ]
        poses = [("initial_arm_away_from_bowl", source["target_context"]["arm_qpos"])]
        for bundle in route["route_bundles"]:
            grasp = next(s for s in bundle["stages"] if s["stage"] == "grasp")
            poses.append(
                (
                    str(bundle["bundle_seed"]),
                    grasp["curobo"]["valid_joint_trajectory_rad"][-1],
                )
            )
        for label, q in poses:
            for mode in (
                ["original_scene"]
                if label.startswith("initial")
                else ["original_scene", "scratch_target_displaced_control"]
            ):
                request = {
                    "op": "scratch_robot_path_contact_audit",
                    "object_id": source["target_context"]["target_object_id"],
                    "arm_joint_states_rad": [q] * len(jaw),
                    "gripper_joint_states_rad": jaw,
                    "fingertip_contact_scope": "all_states",
                }
                if mode != "original_scene":
                    pose = np.eye(4)
                    pose[:3, :3] = source["target_context"]["target_root_pose_world"][
                        "rotation"
                    ]
                    pose[:3, 3] = source["target_context"]["target_root_pose_world"][
                        "position_m"
                    ]
                    pose[2, 3] += 2.0
                    request["T_world_object_m"] = pose.tolist()
                audit = _read_only_rpc(connection, request)
                if audit.get("status") != "complete" or not audit.get(
                    "physical_state_unchanged"
                ):
                    raise RuntimeError(
                        f"incomplete read-only diagnostic: {audit.get('error')}"
                    )
                result["controls"].append(
                    {
                        "arm_pose": label,
                        "mode": mode,
                        "request": request,
                        "audit": audit,
                        "pairs": aggregate_pairs(audit["states"], source),
                    }
                )
        final = _read_only_rpc(
            connection, {"op": "observe", "include_snapshot_state": True}
        )
        result["live_integration_state_unchanged"] = np.array_equal(
            initial["snapshot_state"]["mujoco_integration_state"],
            final["snapshot_state"]["mujoco_integration_state"],
        )
        result["final_timestep"] = final["timestep"]
        result["native_task_success"] = final["task_success"]
        if not result["live_integration_state_unchanged"]:
            raise RuntimeError("read-only diagnostic changed live integration state")
    finally:
        connection.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except Exception:
            process.kill()
            process.wait()
        log.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-report", type=Path, required=True)
    parser.add_argument("--scene-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live-controls", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    route = json.loads(args.route_report.read_text())
    source = json.loads(gzip.decompress(args.scene_source.read_bytes()))
    report = {
        "schema": "libero_safety_grasp_contact_diagnosis.v1",
        "route_report": str(args.route_report),
        "route_sha256": hashlib.sha256(args.route_report.read_bytes()).hexdigest(),
        "scene_source": str(args.scene_source),
        "scene_source_sha256": hashlib.sha256(
            args.scene_source.read_bytes()
        ).hexdigest(),
        "model_geometry_sha256": source["scene_metadata"]["model_geometry_sha256"],
        "archived_bundles": archived_diagnosis(route, source),
        "physics_claim": "static configurations only; no grasp success or safety certification",
    }
    path = args.output_dir / "contact_diagnosis.json"
    path.write_text(json.dumps(_json_safe(report), indent=2) + "\n")
    if args.live_controls:
        report["live_controls"] = live_controls(route, source, args.output_dir)
        path.write_text(json.dumps(_json_safe(report), indent=2) + "\n")


if __name__ == "__main__":
    main()
