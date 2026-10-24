# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0.

"""Generate CuRobo proposals and re-screen them in exact t=236 MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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

from research.gs_opd.repair.search_t236_pot2_grasp_reentries import (
    _jsonable,
    _pose_matrix,
    _records_for_object,
    _strict_screen_policy,
)


ARCHIVE_ROOT = Path(
    "/opt/data/private/gs_opd/rollouts/l0t2_round0_tsfs_safe_seed_20261024"
)
DEFAULT_CONFIG = ARCHIVE_ROOT / "tsfs01_episode_config.json"
DEFAULT_T236 = ARCHIVE_ROOT / "episode1/snapshots/main_timestep_000236.pkl"
DEFAULT_PRIOR_REPORT = (
    ARCHIVE_ROOT / "episode1_t236_grasp_search_20261024/t236_pot2_grasp_search.json"
)
DEFAULT_OUTPUT = ARCHIVE_ROOT / "episode1_t236_curobo_search_20261024c"
DEFAULT_CUROBO_PYTHON = Path("/workspace/gsopd_curobo_eval/.venv/bin/python")
HELPER = Path(__file__).with_name("curobo_t236_plan_grasps.py")


def _candidate_specs(base_transform: np.ndarray) -> list[dict[str, Any]]:
    directions = (
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    )
    rolls = (0.0, 45.0, 90.0, 135.0)
    base_r = base_transform[:3, :3]
    candidates = []
    for direction_index, direction_raw in enumerate(directions):
        direction = np.asarray(direction_raw, dtype=np.float64)
        z_axis = direction / np.linalg.norm(direction)
        y_axis = base_r[:, 1] - z_axis * float(np.dot(base_r[:, 1], z_axis))
        if np.linalg.norm(y_axis) < 1e-8:
            y_axis = base_r[:, 0] - z_axis * float(np.dot(base_r[:, 0], z_axis))
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        for roll_index, roll_deg in enumerate(rolls):
            roll = np.deg2rad(roll_deg)
            x_rot = np.cos(roll) * x_axis + np.sin(roll) * y_axis
            y_rot = -np.sin(roll) * x_axis + np.cos(roll) * y_axis
            transform = base_transform.copy()
            transform[:3, :3] = np.column_stack((x_rot, y_rot, z_axis))
            candidates.append({
                "candidate_id": f"t236_curobo_dir{direction_index}_roll{roll_index}",
                "object_to_eef": transform.tolist(),
                "approach_direction_object": direction.tolist(),
                "approach_distance_m": 0.0606,
                "search_family": "six_cardinal_approaches_x_four_tool_axis_rolls",
                "approach_axis_in_tool_frame": "+z",
                "roll_about_approach_axis_deg": roll_deg,
            })
    return candidates


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    input_path = args.output_dir / "curobo_input.json"
    curobo_path = args.output_dir / "curobo_proposals.json"
    config_data, recovery_config = load_runtime_config(args.config)
    prior = json.loads(args.prior_report.read_text(encoding="utf-8"))
    if not args.snapshot.is_file():
        raise FileNotFoundError(f"t236_snapshot_not_found:{args.snapshot}")
    if not args.curobo_python.is_file() or not HELPER.is_file():
        raise FileNotFoundError("isolated_curobo_python_or_helper_missing")

    bridge_artifacts = args.output_dir / "bridge_artifacts"
    bridge_artifacts.mkdir(parents=True, exist_ok=True)
    bridge = BridgeProcess(bridge_artifacts, python=args.python,
                           port=args.bridge_port)
    backend = None
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema": "gs_opd_t236_curobo_exact_mujoco_rescreen.v1",
        "source_snapshot": str(args.snapshot),
        "source_snapshot_sha256": sha256_file(args.snapshot),
        "source_timestep": 236,
        "physical_episode_actions_committed": 0,
        "curobo_proposal_generator": str(HELPER),
        "curobo_interpreter": str(args.curobo_python),
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
            bridge, config_data, recovery_config, args.output_dir / "bridge_artifacts",
            RecoveryArtifactWriter(args.output_dir / "bridge_artifacts"),
            mode="runtime-snapshot", student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=Path(config_data["artifacts"]["t10_runtime_snapshot"]),
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
        )
        reset = bridge.request({"op": "reset"})
        expected_sha = config_data.get("expected_model_geometry_sha256")
        reset_sha = reset.get("scene_metadata", {}).get("model_geometry_sha256")
        if reset_sha != expected_sha or int(reset.get("timestep", -1)) != 0:
            raise RuntimeError("native_reset_fingerprint_or_timestep_mismatch")
        restored = bridge.request({
            "op": "restore_runtime_snapshot", "payload": args.snapshot.read_bytes()
        })
        if restored.get("status") != "restored":
            raise RuntimeError(f"t236_restore_failed:{restored.get('error')}")
        live = bridge.request({"op": "observe", "include_snapshot_state": True})
        if int(live.get("timestep", -1)) != 236:
            raise RuntimeError(f"restored_timestep_mismatch:{live.get('timestep')}")
        if live.get("scene_metadata", {}).get("model_geometry_sha256") != expected_sha:
            raise RuntimeError("restored_model_geometry_fingerprint_mismatch")
        backend.latest_data = live
        backend._load_context_inputs(live)

        robot = live["snapshot_state"]["robot_states"][0]
        current_q = np.asarray(robot["joint_qpos"], dtype=np.float64)
        eef_position, eef_quat = backend._live_eef_pose(live)
        current_object = backend._live_object_state(live, "moka_pot_2")
        current_object_world = _pose_matrix(
            current_object["root_pos"], current_object["root_quat"]
        )
        base_transform = np.asarray(backend.grasp_template["object_to_eef"], dtype=np.float64)
        candidates = _candidate_specs(base_transform)
        phase_rows = []
        for candidate in candidates:
            object_eef = np.asarray(candidate["object_to_eef"], dtype=np.float64)
            grasp_world = current_object_world @ object_eef
            direction_world = current_object_world[:3, :3] @ np.asarray(
                candidate["approach_direction_object"], dtype=np.float64
            )
            pregrasp_world = grasp_world.copy()
            pregrasp_world[:3, 3] -= direction_world * candidate["approach_distance_m"]
            phase_rows.append({
                "candidate_id": candidate["candidate_id"],
                "eef_grasp_pose_world": grasp_world.tolist(),
                "eef_pregrasp_pose_world": pregrasp_world.tolist(),
                "approach_direction_world": direction_world.tolist(),
            })
        input_data = {
            "current_arm_qpos": current_q.tolist(),
            "live_eef_position_world_m": eef_position.tolist(),
            "live_eef_quaternion_world_wxyz": eef_quat.tolist(),
            "candidates": phase_rows,
        }
        input_path.write_text(json.dumps(input_data, indent=2) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [str(args.curobo_python), str(HELPER), "--input", str(input_path),
             "--output", str(curobo_path)],
            check=False, capture_output=True, text=True,
        )
        report["curobo_process"] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr[-12000:],
        }
        if completed.returncode != 0 or not curobo_path.is_file():
            raise RuntimeError("curobo_proposal_generation_failed")
        curobo = json.loads(curobo_path.read_text(encoding="utf-8"))

        candidate_output = {item["candidate_id"]: item for item in curobo["candidates"]}
        historical_approach = _records_for_object(backend, 118, "moka_pot_2")["record"]["robot_states"][0]["joint_qpos"]
        stable_record = _records_for_object(backend, 141, "moka_pot_2")["record"]
        historical_stable = stable_record["robot_states"][0]["joint_qpos"]
        stable_gripper = stable_record["robot_states"][0]["gripper_qpos"]
        for candidate in candidates:
            proposal = candidate_output[candidate["candidate_id"]]
            generated = proposal["ik_qpos_by_phase"]
            seeds = [generated[key] for key in ("pregrasp", "grasp", "lift30", "lift80")]
            seeds = [np.asarray(value, dtype=np.float64).tolist() for value in seeds if value is not None]
            seeds.extend([historical_approach, historical_stable, current_q.tolist()])
            candidate["ik_seed_qpos"] = seeds
            candidate["curobo_ik_success_by_phase"] = proposal["ik_success_by_phase"]
            candidate["curobo_selected_for_joint_trajectory"] = proposal[
                "selected_by_curobo_grasp_trajopt"
            ]

        target_ids = set(int(value) for value in backend.bindings["moka_pot_2"]["geom_ids"])
        finger_ids = set(int(value) for value in backend.context_static["gripper_geometry_ids"])
        policy = _strict_screen_policy(backend, "moka_pot_2")
        state_hash = hashlib.sha256(
            np.asarray(live["snapshot_state"]["mujoco_integration_state"],
                       dtype=np.float64).tobytes()
        ).hexdigest()
        screened = bridge.request({
            "op": "screen_grasp_candidates",
            "expected_instruction": backend.context_static["instruction"],
            "model_geometry_sha256": expected_sha,
            "start_state_sha256": state_hash,
            "safety_geometry_policy": policy,
            "target_object_id": "moka_pot_2",
            "target_root_body_id": int(backend.bindings["moka_pot_2"]["root_body_id"]),
            "clearance_m": 0.010,
            "closed_gripper_qpos": stable_gripper,
            "predicted_stable_object_pose_world": prior["predicted_stable_target_pose_world"],
            "candidates": candidates,
        })
        if screened.get("status") != "complete" or screened.get("live_state_unchanged") is not True:
            raise RuntimeError(
                f"exact_mujoco_scratch_rescreen_failed:{screened.get('failure_reason') or screened.get('status')}"
            )
        candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
        screen_by_id = {item["candidate_id"]: item for item in screened["candidates"]}
        rrt_cfg = dict(config_data.get("rrt_connect", {}))
        rrt_cfg.update({
            "max_candidate_routes": min(3, int(rrt_cfg.get("max_candidate_routes", 3))),
            "max_iterations": min(700, int(rrt_cfg.get("max_iterations", 2000))),
            "max_config_checks": min(9000, int(rrt_cfg.get("max_config_checks", 25000))),
            "search_time_budget_s": min(50.0, float(rrt_cfg.get("search_time_budget_s", 360.0))),
            "shortcut_attempts": min(80, int(rrt_cfg.get("shortcut_attempts", 250))),
            "minimum_clearance_m": 0.010,
        })
        rrt_rows = []
        for candidate_id in screened.get("ranked_candidate_ids", [])[:8]:
            candidate = candidate_by_id[candidate_id]
            pose_world = current_object_world @ np.asarray(candidate["object_to_eef"], dtype=np.float64)
            direction_world = current_object_world[:3, :3] @ np.asarray(
                candidate["approach_direction_object"], dtype=np.float64
            )
            pregrasp = pose_world.copy()
            pregrasp[:3, 3] -= direction_world * candidate["approach_distance_m"]
            quat_xyzw = Rotation.from_matrix(pregrasp[:3, :3]).as_quat()
            result = bridge.request({
                "op": "rrt_connect",
                "expected_instruction": backend.context_static["instruction"],
                "start_state_sha256": state_hash,
                "model_geometry_sha256": expected_sha,
                "goal_pose_world": {
                    "position_world_m": pregrasp[:3, 3].tolist(),
                    "quaternion_wxyz": [float(quat_xyzw[3]), *quat_xyzw[:3].tolist()],
                },
                "safety_geometry_policy": policy,
                "ik_seed_qpos": [screen_by_id[candidate_id]["pregrasp_qpos"],
                                 *candidate["ik_seed_qpos"]],
                "ik_selection_seed_qpos": screen_by_id[candidate_id]["pregrasp_qpos"],
                "planner_config": rrt_cfg,
            })
            rrt_rows.append({
                "candidate_id": candidate_id,
                "status": result.get("status"),
                "failure_reason": result.get("failure_reason"),
                "planning_seconds": result.get("planning_seconds"),
                "config_checks": result.get("config_checks"),
                "minimum_forbidden_clearance_m": result.get("minimum_forbidden_clearance_m"),
                "route_candidates": result.get("route_candidates"),
                "live_state_unchanged": result.get("live_state_unchanged"),
                "qpos_path": result.get("qpos_path"),
                "eef_reference_poses_world": result.get("eef_reference_poses_world"),
            })
        report.update({
            "model_geometry_sha256": expected_sha,
            "restored_timestep": int(live["timestep"]),
            "start_state_sha256": state_hash,
            "finger_only_allowlist_geom_ids": sorted(finger_ids),
            "target_clearance_policy_m": 0.010,
            "allowed_finger_penetration_m": 0.002,
            "curobo": curobo,
            "candidate_count": len(candidates),
            "candidate_specs_with_ik_seeds": candidates,
            "exact_mujoco_screen": screened,
            "exact_mujoco_offline_feasible_count": int(screened.get("feasible_count", 0)),
            "rrt_connect_route_preflights": rrt_rows,
            "rrt_reachable_count": sum(
                item.get("status") == "offline_feasible"
                and item.get("live_state_unchanged") is True for item in rrt_rows
            ),
            "physical_branch_count": 0,
            "elapsed_seconds": time.monotonic() - started,
        })
        (args.output_dir / "t236_curobo_exact_mujoco_report.json").write_text(
            json.dumps(_jsonable(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        if backend is not None:
            for writer in backend._video_writers.values():
                writer.release()
        bridge.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_T236)
    parser.add_argument("--prior-report", type=Path, default=DEFAULT_PRIOR_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--curobo-python", type=Path, default=DEFAULT_CUROBO_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38874)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "status": "complete",
        "curobo_plan_grasp_success": result["curobo"]["plan_grasp_success"],
        "curobo_ik_success_count": result["curobo"]["ik_success_count"],
        "mujoco_feasible_count": result["exact_mujoco_offline_feasible_count"],
        "rrt_reachable_count": result["rrt_reachable_count"],
        "elapsed_seconds": result["elapsed_seconds"],
        "report": str(args.output_dir / "t236_curobo_exact_mujoco_report.json"),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
