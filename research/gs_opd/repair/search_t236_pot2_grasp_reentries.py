# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bounded t=236 Pot 2 grasp, lift, exit and RRT-reentry candidate search.

All candidate kinematics run in MuJoCo scratch data. The live Bridge is reset
and restored from the archived t=236 Snapshot before every search; this script
does not commit a physical Episode action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from research.gs_opd.repair.automatic_recovery import RecoveryArtifactWriter
from research.gs_opd.repair.automatic_recovery_runtime import (
    AutomaticRecoveryBridgeBackend,
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
    sha256_file,
)
from research.gs_opd.repair.mujoco_jointspace_planning import (
    validate_finger_only_contact_geoms,
)


ARCHIVE_ROOT = Path(
    "/opt/data/private/gs_opd/rollouts/l0t2_round0_tsfs_safe_seed_20261024"
)
DEFAULT_CONFIG = ARCHIVE_ROOT / "tsfs01_episode_config.json"
DEFAULT_T236 = ARCHIVE_ROOT / "episode1/snapshots/main_timestep_000236.pkl"
DEFAULT_OUTPUT = ARCHIVE_ROOT / "episode1_t236_grasp_search_20261024"


def _pose_matrix(position: Any, quaternion_wxyz: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        np.r_[np.asarray(quaternion_wxyz, dtype=np.float64)[1:],
              np.asarray(quaternion_wxyz, dtype=np.float64)[0]]
    ).as_matrix()
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _records_for_object(backend, timestep: int, object_id: str) -> dict[str, Any]:
    index = timestep - 1
    record = json.loads(str(backend.archive["post_physical_state_json"][index]))
    matches = [
        value for key, value in record["object_states"].items()
        if key.endswith(":" + object_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"historical_object_pose_not_unique:{timestep}:{object_id}")
    return {"record": record, "object": matches[0]}


def _candidate_specs(base_transform: np.ndarray, current_q: np.ndarray,
                     approach_q: np.ndarray, stable_q: np.ndarray) -> list[dict[str, Any]]:
    # Compound wrist rotations and independently varied object-frame approach
    # directions are intentionally distinct from the archived 24-candidate
    # axis-rotation/near-offset batch.
    wrist_rotations_deg = (
        (45.0, 45.0, 45.0),
        (-45.0, 45.0, -45.0),
        (90.0, 45.0, 0.0),
        (0.0, -45.0, 90.0),
    )
    approach_directions = (
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    )
    offsets = (
        (0.0, 0.0, 0.0),
        (0.0, 0.012, 0.0),
        (-0.012, 0.0, 0.0),
        (0.008, -0.008, 0.0),
    )
    nullspace_deltas = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, -0.30, 0.0, 0.35, 0.0, -0.25, 0.65),
        (0.0, 0.30, 0.0, -0.35, 0.0, 0.25, -0.65),
        (0.0, 0.0, 0.40, 0.0, -0.50, 0.0, 0.35),
    )
    candidates = []
    for wrist_index, wrist_deg in enumerate(wrist_rotations_deg):
        delta_rotation = Rotation.from_euler("xyz", wrist_deg, degrees=True).as_matrix()
        for direction_index, direction in enumerate(approach_directions):
            candidate_index = wrist_index * len(approach_directions) + direction_index
            transform = base_transform.copy()
            transform[:3, :3] = base_transform[:3, :3] @ delta_rotation
            transform[:3, 3] += np.asarray(offsets[wrist_index], dtype=np.float64)
            delta = np.asarray(nullspace_deltas[wrist_index], dtype=np.float64)
            alternate = current_q + delta
            candidates.append({
                "candidate_id": (
                    f"t236_v2_wrist{wrist_index}_approach{direction_index}"
                ),
                "object_to_eef": transform.tolist(),
                "approach_direction_object": list(direction),
                "approach_distance_m": 0.0606,
                "ik_seed_qpos": [
                    current_q.tolist(),
                    approach_q.tolist(),
                    stable_q.tolist(),
                    alternate.tolist(),
                ],
                "search_family": "compound_wrist_x_cardinal_object_approach",
                "wrist_rotation_euler_xyz_deg": list(wrist_deg),
                "translation_offset_object_m": list(offsets[wrist_index]),
                "redundancy_seed_delta_rad": delta.tolist(),
                "source_prior": "archived_verified_historical_Pot2_object_to_EEF_transform",
            })
    return candidates


def _strict_screen_policy(backend: AutomaticRecoveryBridgeBackend,
                          target_object_id: str) -> dict[str, Any]:
    target_geom_ids = set(int(value) for value in backend.bindings[target_object_id]["geom_ids"])
    finger_geom_ids = set(int(value) for value in backend.context_static["gripper_geometry_ids"])
    validate_finger_only_contact_geoms(
        list(finger_geom_ids & {76, 77}),
        list(finger_geom_ids & {79, 80}),
        list(finger_geom_ids),
    )
    policy = json.loads(json.dumps(backend.safety_policy))
    pairs = [
        {
            "robot_geom_id": robot_geom,
            "scene_geom_id": scene_geom,
            "minimum_clearance_m": -0.002,
        }
        for robot_geom in sorted(finger_geom_ids)
        for scene_geom in sorted(target_geom_ids)
    ]
    policy["allowed_contact_pairs"] = pairs
    return policy


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config_data, recovery_config = load_runtime_config(args.config)
    if not args.snapshot.is_file():
        raise FileNotFoundError(f"t236_snapshot_not_found:{args.snapshot}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_dir = args.output_dir
    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend = None
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema": "gs_opd_t236_pot2_grasp_reentry_search.v1",
        "source_snapshot": str(args.snapshot),
        "source_snapshot_sha256": sha256_file(args.snapshot),
        "source_timestep": 236,
        "physical_episode_actions_committed": 0,
        "maximum_candidates": args.maximum_candidates,
        "maximum_rrt_preflights": args.maximum_rrt_preflights,
    }
    try:
        bridge.start(
            benchmark=config_data["benchmark"],
            level=int(config_data["level"]),
            task_id=int(config_data["task_id"]),
            init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=int(recovery_config.horizon_steps),
        )
        backend = AutomaticRecoveryBridgeBackend(
            bridge,
            config_data,
            recovery_config,
            output_dir,
            RecoveryArtifactWriter(output_dir),
            mode="runtime-snapshot",
            student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=Path(config_data["artifacts"]["t10_runtime_snapshot"]),
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
        )

        reset = bridge.request({"op": "reset"})
        model_sha = reset.get("scene_metadata", {}).get("model_geometry_sha256")
        expected_sha = config_data.get("expected_model_geometry_sha256")
        if model_sha != expected_sha or reset.get("timestep") != 0:
            raise RuntimeError("native_reset_fingerprint_or_timestep_mismatch")
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": args.snapshot.read_bytes()})
        if restored.get("status") != "restored":
            raise RuntimeError(f"t236_snapshot_restore_failed:{restored.get('error')}")
        data = bridge.request({"op": "observe", "include_snapshot_state": True})
        if int(data.get("timestep", -1)) != 236:
            raise RuntimeError(f"restored_snapshot_timestep_mismatch:{data.get('timestep')}")
        if data.get("scene_metadata", {}).get("model_geometry_sha256") != expected_sha:
            raise RuntimeError("restored_snapshot_compiled_geometry_fingerprint_mismatch")
        backend.latest_data = data
        backend._load_context_inputs(data)
        object_id = "moka_pot_2"
        binding = backend.bindings[object_id]
        screen_policy = _strict_screen_policy(backend, object_id)
        state_hash = hashlib.sha256(
            np.asarray(data["snapshot_state"]["mujoco_integration_state"], dtype=np.float64).tobytes()
        ).hexdigest()
        live_robot = data["snapshot_state"]["robot_states"][0]
        current_q = np.asarray(live_robot["joint_qpos"], dtype=np.float64)
        approach_record = _records_for_object(backend, 118, object_id)["record"]
        stable_record = _records_for_object(backend, 141, object_id)["record"]
        historical_clean = _records_for_object(backend, 10, object_id)["object"]
        historical_stable = _records_for_object(backend, 141, object_id)["object"]
        historical_approach_q = np.asarray(
            approach_record["robot_states"][0]["joint_qpos"], dtype=np.float64
        )
        historical_stable_q = np.asarray(
            stable_record["robot_states"][0]["joint_qpos"], dtype=np.float64
        )
        current_object = backend._live_object_state(data, object_id)
        current_object_world = _pose_matrix(
            current_object["root_pos"], current_object["root_quat"]
        )
        historical_clean_world = _pose_matrix(
            historical_clean["root_pos"], historical_clean["root_quat"]
        )
        historical_stable_world = _pose_matrix(
            historical_stable["root_pos"], historical_stable["root_quat"]
        )
        stable_motion_prior = historical_stable_world @ np.linalg.inv(historical_clean_world)
        predicted_stable_world = stable_motion_prior @ current_object_world
        base_transform = np.asarray(backend.grasp_template["object_to_eef"], dtype=np.float64)
        candidates = _candidate_specs(
            base_transform,
            current_q,
            historical_approach_q,
            historical_stable_q,
        )[: args.maximum_candidates]
        if len(candidates) != args.maximum_candidates:
            raise ValueError("candidate_generator_did_not_meet_requested_bounded_count")

        stable_gripper = np.asarray(
            stable_record["robot_states"][0]["gripper_qpos"], dtype=np.float64
        ).tolist()
        screened = bridge.request({
            "op": "screen_grasp_candidates",
            "expected_instruction": backend.context_static["instruction"],
            "model_geometry_sha256": backend.context_static["model_geometry_sha256"],
            "start_state_sha256": state_hash,
            "safety_geometry_policy": screen_policy,
            "target_object_id": object_id,
            "target_root_body_id": int(binding["root_body_id"]),
            "clearance_m": 0.010,
            "closed_gripper_qpos": stable_gripper,
            "predicted_stable_object_pose_world": predicted_stable_world.tolist(),
            "candidates": candidates,
        })
        if screened.get("status") != "complete" or screened.get("live_state_unchanged") is not True:
            raise RuntimeError(
                f"scratch_screen_incomplete_or_mutated_live_state:{screened.get('failure_reason') or screened.get('status')}"
            )
        candidate_by_id = {item["candidate_id"]: item for item in candidates}
        screen_by_id = {item["candidate_id"]: item for item in screened.get("candidates", [])}

        route_config = dict(config_data.get("rrt_connect", {}))
        route_config.update({
            "max_candidate_routes": min(3, int(route_config.get("max_candidate_routes", 3))),
            "max_iterations": min(700, int(route_config.get("max_iterations", 2000))),
            "max_config_checks": min(9000, int(route_config.get("max_config_checks", 25000))),
            "search_time_budget_s": min(50.0, float(route_config.get("search_time_budget_s", 360.0))),
            "shortcut_attempts": min(80, int(route_config.get("shortcut_attempts", 250))),
            "minimum_clearance_m": 0.010,
        })
        route_rows = []
        for candidate_id in screened.get("ranked_candidate_ids", [])[: args.maximum_rrt_preflights]:
            candidate = candidate_by_id[candidate_id]
            screen = screen_by_id[candidate_id]
            object_eef = np.asarray(candidate["object_to_eef"], dtype=np.float64)
            eef_world = current_object_world @ object_eef
            approach_world = current_object_world[:3, :3] @ np.asarray(
                candidate["approach_direction_object"], dtype=np.float64
            )
            pregrasp_position = eef_world[:3, 3] - approach_world * float(
                candidate["approach_distance_m"]
            )
            quaternion_xyzw = Rotation.from_matrix(eef_world[:3, :3]).as_quat()
            quaternion_wxyz = np.r_[quaternion_xyzw[3], quaternion_xyzw[:3]]
            route = bridge.request({
                "op": "rrt_connect",
                "expected_instruction": backend.context_static["instruction"],
                "start_state_sha256": state_hash,
                "model_geometry_sha256": backend.context_static["model_geometry_sha256"],
                "goal_pose_world": {
                    "position_world_m": pregrasp_position.tolist(),
                    "quaternion_wxyz": quaternion_wxyz.tolist(),
                },
                "safety_geometry_policy": screen_policy,
                "ik_seed_qpos": [screen["pregrasp_qpos"], *candidate["ik_seed_qpos"]],
                "ik_selection_seed_qpos": screen["pregrasp_qpos"],
                "planner_config": route_config,
            })
            route_rows.append({
                "candidate_id": candidate_id,
                "status": route.get("status"),
                "failure_reason": route.get("failure_reason"),
                "planning_seconds": route.get("planning_seconds"),
                "minimum_forbidden_clearance_m": route.get("minimum_forbidden_clearance_m"),
                "config_checks": route.get("config_checks"),
                "route_candidates": route.get("route_candidates"),
                "live_state_unchanged": route.get("live_state_unchanged"),
                "qpos_path": route.get("qpos_path"),
                "eef_reference_poses_world": route.get("eef_reference_poses_world"),
            })

        result.update({
            "model_geometry_sha256": model_sha,
            "restored_timestep": int(data["timestep"]),
            "start_state_sha256": state_hash,
            "finger_only_allowlist_geom_ids": sorted(
                int(value) for value in backend.context_static["gripper_geometry_ids"]
            ),
            "target_clearance_policy_m": 0.010,
            "allowed_finger_penetration_m": 0.002,
            "historical_stable_motion_prior_world": stable_motion_prior.tolist(),
            "predicted_stable_target_pose_world": predicted_stable_world.tolist(),
            "candidate_count": len(candidates),
            "candidate_ids": [item["candidate_id"] for item in candidates],
            "offline_screen": screened,
            "rrt_route_preflights": route_rows,
            "rrt_route_preflight_budget": route_config,
            "offline_complete_feasible_count": sum(
                item.get("status") == "offline_feasible_candidate"
                for item in screened.get("candidates", [])
            ),
            "route_reachable_count": sum(
                row.get("status") == "offline_feasible" and row.get("live_state_unchanged") is True
                for row in route_rows
            ),
            "elapsed_search_seconds": time.monotonic() - started,
        })
        (output_dir / "t236_pot2_grasp_search.json").write_text(
            json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result
    finally:
        if backend is not None:
            for writer in backend._video_writers.values():
                writer.release()
        bridge.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_T236)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38872)
    parser.add_argument("--maximum-candidates", type=int, choices=(24,), default=24)
    parser.add_argument("--maximum-rrt-preflights", type=int, choices=range(1, 25), default=8)
    args = parser.parse_args()
    result = _run(args)
    print(json.dumps({
        "status": "complete",
        "candidate_count": result.get("candidate_count"),
        "offline_complete_feasible_count": result.get("offline_complete_feasible_count"),
        "route_reachable_count": result.get("route_reachable_count"),
        "elapsed_search_seconds": result.get("elapsed_search_seconds"),
        "report": str(args.output_dir / "t236_pot2_grasp_search.json"),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
