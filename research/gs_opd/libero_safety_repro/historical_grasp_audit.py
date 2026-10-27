# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Extract sparse grasp evidence from an own rollout without replaying actions."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from research.gs_opd.libero_safety_repro.execute_grasp import classify_contacts
from research.gs_opd.libero_safety_repro.keyposes import (
    inverse_transform,
    transform_from_pose,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--scene-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(gzip.decompress(args.scene_source.read_bytes()))
    geoms = {
        g["geom_id"]: g
        for g in source["collision_scene_export"]["geometry_inventory"]["geometries"]
    }
    pads = {
        g["geom_id"]
        for rows in source["target_context"]["fingertip_pad_geometries"].values()
        for g in rows
    }
    target = {
        g["geom_id"] for g in source["target_context"]["target_collision_geometries"]
    }
    report = json.loads(args.evaluation_report.read_text())
    with np.load(args.npz, allow_pickle=False) as archive:
        contacts = archive["post_contacts_json"]
        states = archive["post_physical_state_json"]
        rows, sparse = [], []
        longest = run = 0
        first_bilateral = None
        for i, encoded in enumerate(contacts):
            check = classify_contacts(
                json.loads(str(encoded)), geoms, pads, target, True
            )
            bilateral = set(check["pad_ids"]) == pads
            eligible = bilateral and check["complete"] and not check["illegal"]
            run = run + 1 if eligible else 0
            longest = max(longest, run)
            if bilateral and first_bilateral is None:
                first_bilateral = i
            rows.append(
                {
                    "frame_index": i,
                    "bilateral_pads": bilateral,
                    "identity_complete": check["complete"],
                    "illegal_contact_record_count": len(check["illegal"]),
                    "illegal_pairs": sorted(
                        {
                            tuple(sorted((c["geom1"]["id"], c["geom2"]["id"])))
                            for c in check["illegal"]
                        }
                    ),
                }
            )
            if i == first_bilateral or eligible:
                state = json.loads(str(states[i]))
                robot = state["robot_states"][0]
                obj = state["object_states"][
                    f"objects_dict:{source['config']['target_object_id']}"
                ]
                relation = inverse_transform(
                    transform_from_pose(obj["root_pos"], obj["root_quat"])
                ) @ transform_from_pose(robot["eef_pos"], robot["eef_quat"])
                sparse.append(
                    {
                        "frame_index": i,
                        "timestep_after_action": i + 1,
                        "role": "isolated_clean_bilateral_sample"
                        if eligible
                        else "first_bilateral_with_illegal_contacts",
                        "T_object_eef": relation.tolist(),
                        "robot_state": robot,
                        "object_state": obj,
                        "contacts": json.loads(str(encoded)),
                        "safety_certified": False,
                    }
                )
        output = {
            "npz": str(args.npz),
            "npz_sha256": hashlib.file_digest(
                args.npz.open("rb"), "sha256"
            ).hexdigest(),
            "evaluation_report": str(args.evaluation_report),
            "task": report["task"],
            "native_success_reported": report["metrics"]["task_success"],
            "frame_count": len(rows),
            "first_bilateral_frame": first_bilateral,
            "longest_clean_bilateral_post_state_run": longest,
            "clean_bilateral_frames": [
                r["frame_index"]
                for r in rows
                if r["bilateral_pads"]
                and r["identity_complete"]
                and r["illegal_contact_record_count"] == 0
            ],
            "stable_pad_only_prefix_found": longest >= 5,
            "audit_scope": "post-action contacts only; sufficient to reject nonpad evidence, not certify substep safety",
            "pose_convention": "MuJoCo snapshot EEF/root quaternions WXYZ; object-local SE(3), meters",
            "source_actions_replayed": False,
            "rows": rows,
            "sparse_references": sparse,
        }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
