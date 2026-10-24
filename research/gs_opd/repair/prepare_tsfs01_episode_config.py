# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepare a one-candidate L0T2 run from the archived TSFS-01 prior."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_CONFIG = REPO_ROOT / "research/gs_opd/configs/l0t2_task_success_first_safety_reconstruction.json"
DEFAULT_SEARCH_REPORT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "task_success_first_safety_reconstruction_l0t2_20261023_attempt1/"
    "episode9/task_success_first_candidate_search.json"
)
DEFAULT_T10_SNAPSHOT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_round1_safety_recovery_20261023_attempt1/evidence/"
    "raw_pi05_t10_runtime_snapshot.pkl"
)
DEFAULT_TRACE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "obstacle_avoidance_L0T2_base_pi05_rollout_400_20261022_attempt2/"
    "rollout_trace.npz"
)
DEFAULT_TEMPLATE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_round1_jointspace_rrt_connect_20261023_attempt1/source/"
    "source_grasp_template.json"
)
DEFAULT_OUTPUT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_round0_tsfs_safe_seed_20261024/tsfs01_episode_config.json"
)
EXPECTED_CANDIDATE = "tsfs_01_axis2_rot+30"
EXPECTED_MODEL = "1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752"


class _PythonObjectState:
    """Compatibility placeholder for Bridge helper values in its Snapshot."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--search-report", type=Path, default=DEFAULT_SEARCH_REPORT)
    parser.add_argument("--t10-snapshot", type=Path, default=DEFAULT_T10_SNAPSHOT)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def main() -> int:
    args = _arguments()
    search = _load_json(args.search_report)
    if search.get("selected_candidate_id") != EXPECTED_CANDIDATE:
        raise ValueError("archived_search_did_not_select_tsfs_01_axis2_rot_plus_30")
    if search.get("source_snapshot_timestep") != 10:
        raise ValueError("archived_candidate_search_did_not_start_from_t10")
    route = search.get("selected_route_summary", {})
    if (
        route.get("status") != "offline_feasible"
        or route.get("live_state_unchanged") is not True
        or float(route.get("minimum_forbidden_clearance_m", 0.0)) < 0.01
    ):
        raise ValueError("archived_tsfs_01_route_preflight_is_not_safe_and_feasible")
    screen_rows = search.get("offline_screen", {}).get("candidates", [])
    screen = next(
        (row for row in screen_rows if row.get("candidate_id") == EXPECTED_CANDIDATE),
        None,
    )
    if not isinstance(screen, dict) or screen.get("status") != "feasible":
        if not isinstance(screen, dict) or screen.get("grasp_check", {}).get("status") != "pass":
            raise ValueError("archived_tsfs_01_candidate_did_not_pass_offline_screen")

    payload = args.t10_snapshot.read_bytes()
    snapshot_envelope = pickle.loads(payload)
    if snapshot_envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
        raise ValueError("t10_runtime_snapshot_schema_is_not_supported")
    if snapshot_envelope.get("model_geometry_sha256") != EXPECTED_MODEL:
        raise ValueError("t10_snapshot_model_geometry_hash_does_not_match_expected")
    task_bookkeeping = snapshot_envelope["snapshot"].get("task_bookkeeping", {})
    if int(task_bookkeeping.get("timestep", -1)) != 10:
        raise ValueError("archived_t10_snapshot_does_not_contain_timestep_10")
    t10_state_hash = hashlib.sha256(
        np.asarray(
            snapshot_envelope["snapshot"]["mujoco_state"], dtype=np.float64
        ).tobytes()
    ).hexdigest()
    if t10_state_hash != search.get("start_state_sha256"):
        raise ValueError("archived_candidate_search_start_state_differs_from_t10_snapshot")

    with np.load(args.trace, allow_pickle=False) as archive:
        states = archive["post_physical_state_json"]
        stable = json.loads(str(states[140]))
        approach = json.loads(str(states[117]))
        stable_qpos = np.asarray(stable["robot_states"][0]["joint_qpos"], dtype=np.float64)
        approach_qpos = np.asarray(approach["robot_states"][0]["joint_qpos"], dtype=np.float64)

    template = _load_json(args.template)
    rebase = search.get("historical_grasp_rebase", {})
    base_transform = np.asarray(
        rebase.get("reexpressed_object_to_EEF"), dtype=np.float64
    )
    if base_transform.shape != (4, 4):
        raise ValueError("archived_historical_grasp_rebase_is_malformed")

    # Recreate only the already selected +30 degree candidate (index 1),
    # without re-screening the 24-candidate set.
    rotvec = np.asarray([0.0, 0.0, np.deg2rad(30.0)], dtype=np.float64)
    delta = Rotation.from_rotvec(rotvec).as_matrix()
    transform = base_transform.copy()
    transform[:3, :3] = base_transform[:3, :3] @ delta
    direction = np.asarray(template["approach_direction_object"], dtype=np.float64)
    direction = delta.T @ direction
    direction /= np.linalg.norm(direction)
    seed_delta = np.asarray([0.0, -0.35, 0.0, 0.35, 0.0, -0.20, 0.65])
    alternate_seed = stable_qpos + seed_delta
    alternate_seed = approach_qpos + 0.65 * (alternate_seed - stable_qpos)
    candidate = {
        "candidate_id": EXPECTED_CANDIDATE,
        "object_to_eef": transform.tolist(),
        "approach_direction_object": direction.tolist(),
        "approach_distance_m": float(template["approach_distance_m"]),
        "ik_seed_qpos": [stable_qpos.tolist(), alternate_seed.tolist()],
        "preferred_ik_seed_qpos": screen["pregrasp_qpos"],
        "pregrasp_qpos": screen["pregrasp_qpos"],
        "rotation_delta_object_rad": rotvec.tolist(),
        "translation_offset_object_m": [0.0, 0.0, 0.0],
        "search_family": "axis2_rot+30",
        "source_prior": "verified_bilateral_historical_grasp_transform",
        "offline_screen_status": "archived_feasible_candidate_not_new_execution_evidence",
    }

    config = _load_json(args.base_config)
    config = copy.deepcopy(config)
    config["task_execution"]["grasp_pose_candidates"] = [candidate]
    config["candidate_search"] = {
        "enabled": False,
        "reason": "reuse_archived_tsfs_01_candidate_without_repeating_24_candidate_search",
    }
    config["reused_candidate_provenance"] = {
        "candidate_id": EXPECTED_CANDIDATE,
        "search_report": str(args.search_report),
        "search_report_sha256": hashlib.sha256(args.search_report.read_bytes()).hexdigest(),
        "t10_snapshot": str(args.t10_snapshot),
        "t10_snapshot_sha256": hashlib.sha256(payload).hexdigest(),
        "t10_mujoco_state_sha256": t10_state_hash,
        "archived_search_start_state_matches_t10": True,
        "archived_offline_screen_status": screen.get("status"),
        "archived_pregrasp_qpos": screen["pregrasp_qpos"],
        "archived_grasp_qpos": screen.get("grasp_qpos"),
        "archived_exit80_qpos": screen.get("exit80_qpos"),
        "archived_minimum_forbidden_clearance_m": screen.get("minimum_forbidden_clearance_m"),
        "archived_preflight_minimum_clearance_m": route["minimum_forbidden_clearance_m"],
        "archived_preflight_config_checks": route.get("config_checks"),
        "archived_preflight_planning_seconds": route.get("planning_seconds"),
        "prior_candidate_count": int(search.get("candidate_count", 0)),
        "prior_candidate_set_rescreened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "prepared",
        "candidate_id": EXPECTED_CANDIDATE,
        "t10_state_sha256": t10_state_hash,
        "grasp_transform": candidate["object_to_eef"],
        "archived_minimum_forbidden_clearance_m": route["minimum_forbidden_clearance_m"],
        "candidate_set_rescreened": False,
        "config_path": str(args.output),
        "config_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
