# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0.

"""CuRobo-only multi-goal IK and approach/grasp/lift proposal generator.

This helper runs inside the isolated Python 3.11/CUDA environment. Its output is
never a safety certificate: the caller must revalidate every proposal against
the exact archived t=236 MuJoCo model and finger-only policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, JointState, Pose


def _pose_matrix(position: list[float], quaternion_wxyz: list[float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_quat(
        [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]]
    ).as_matrix()
    out[:3, 3] = np.asarray(position, dtype=np.float64)
    return out


def _matrix_pose(matrix: np.ndarray) -> tuple[list[float], list[float]]:
    quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return matrix[:3, 3].tolist(), [float(quat_xyzw[3]), *quat_xyzw[:3].tolist()]


def _traj_summary(result, stage: str) -> dict:
    trajectory = getattr(result, f"{stage}_interpolated_trajectory", None)
    if trajectory is None:
        return {"waypoints": 0, "q_path": None}
    q = trajectory.position.detach().float().cpu().numpy().reshape(-1, trajectory.position.shape[-1])
    return {"waypoints": int(len(q)), "q_path": q[:, :7].tolist()}


def run(input_path: Path, output_path: Path) -> dict:
    started = time.monotonic()
    source = json.loads(input_path.read_text(encoding="utf-8"))
    device = torch.device("cuda:0")
    dtype = torch.float32
    current_q = np.asarray(source["current_arm_qpos"], dtype=np.float64)
    if current_q.shape != (7,):
        raise ValueError(f"expected_7d_start_qpos:{current_q.shape}")

    # Calibrate only the rigid world-to-base transform from the live MuJoCo EEF
    # pose and CuRobo's Franka FK at the same seven joint values.
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
    q_tensor = torch.as_tensor(current_q[None, :], device=device, dtype=dtype)
    current_fk = kin.compute_kinematics(
        JointState.from_position(q_tensor, joint_names=kin.joint_names)
    ).tool_poses.get_link_pose("panda_hand")
    curobo_base_eef = _pose_matrix(
        current_fk.position[0].detach().cpu().numpy().tolist(),
        current_fk.quaternion[0].detach().cpu().numpy().tolist(),
    )
    live_eef_world = _pose_matrix(
        source["live_eef_position_world_m"], source["live_eef_quaternion_world_wxyz"]
    )
    world_base = live_eef_world @ np.linalg.inv(curobo_base_eef)
    base_world = np.linalg.inv(world_base)

    candidates = source["candidates"]
    frames = ["panda_hand"]
    phase_rows = []
    grasp_positions, grasp_quats = [], []
    for candidate in candidates:
        grasp_world = np.asarray(candidate["eef_grasp_pose_world"], dtype=np.float64)
        pregrasp_world = np.asarray(candidate["eef_pregrasp_pose_world"], dtype=np.float64)
        lift30_world = grasp_world.copy()
        lift30_world[:3, 3] += np.array([0.0, 0.0, 0.03])
        lift80_world = grasp_world.copy()
        lift80_world[:3, 3] += np.array([0.0, 0.0, 0.08])
        targets = [pregrasp_world, grasp_world, lift30_world, lift80_world]
        phase = []
        for target in targets:
            local = base_world @ target
            position, quaternion = _matrix_pose(local)
            phase.append((position, quaternion))
        phase_rows.append(phase)
        grasp_positions.append(phase[1][0])
        grasp_quats.append(phase[1][1])

    # Public collision-aware batched Franka IK. CuRobo self-collision is on;
    # scene obstacles are not treated as authoritative here and are checked by
    # the exact MuJoCo backend after this proposal stage.
    ik_positions = []
    ik_quats = []
    ik_owner = []
    for index, phases in enumerate(phase_rows):
        for phase_index, (pos, quat) in enumerate(phases):
            ik_positions.append(pos)
            ik_quats.append(quat)
            ik_owner.append((index, phase_index))
    ik_cfg = InverseKinematicsCfg.create(
        robot="franka.yml",
        num_seeds=32,
        max_batch_size=len(ik_positions),
        self_collision_check=True,
    )
    ik_solver = InverseKinematics(ik_cfg)
    ik_pose = Pose(
        position=torch.as_tensor(ik_positions, device=device, dtype=dtype),
        quaternion=torch.as_tensor(ik_quats, device=device, dtype=dtype),
        normalize_rotation=True,
    )
    ik_result = ik_solver.solve_pose(
        GoalToolPose.from_poses({ik_solver.tool_frames[0]: ik_pose}, num_goalset=1)
    )
    q_solutions: list[list[list[float] | None]] = [[None] * 4 for _ in candidates]
    ik_success = [[False] * 4 for _ in candidates]
    ik_position_error = [[None] * 4 for _ in candidates]
    raw_success = ik_result.success.detach().cpu().numpy().reshape(-1)
    raw_q = ik_result.js_solution.position.detach().float().cpu().numpy()
    raw_q = raw_q.reshape(len(ik_positions), -1, raw_q.shape[-1])[:, 0, :]
    raw_pos_err = ik_result.position_error.detach().cpu().numpy().reshape(-1)
    for flat_index, (candidate_index, phase_index) in enumerate(ik_owner):
        ik_success[candidate_index][phase_index] = bool(raw_success[flat_index])
        ik_position_error[candidate_index][phase_index] = float(raw_pos_err[flat_index])
        if raw_success[flat_index]:
            q_solutions[candidate_index][phase_index] = raw_q[flat_index, :7].tolist()

    # Native CuRobo joint-space optimizer interface: one 24-pose goalset and
    # plan_grasp's ordered approach -> grasp -> lift optimization. The exact
    # MuJoCo check below remains the only eligibility gate for physical tests.
    planner_cfg = MotionPlannerCfg.create(
        robot="franka.yml",
        self_collision_check=True,
        max_goalset=len(candidates),
        max_batch_size=1,
        num_ik_seeds=16,
        num_trajopt_seeds=4,
        use_cuda_graph=False,
        interpolation_dt=0.04,
        interpolation_buffer_size=800,
        position_tolerance=0.005,
        orientation_tolerance=0.05,
        random_seed=236,
    )
    planner = MotionPlanner(planner_cfg)
    start = JointState.from_position(
        torch.as_tensor(current_q[None, :], device=device, dtype=dtype),
        joint_names=planner.joint_names,
    )
    goals = GoalToolPose(
        tool_frames=frames,
        position=torch.as_tensor(grasp_positions, device=device, dtype=dtype)
        .reshape(1, 1, 1, len(candidates), 3),
        quaternion=torch.as_tensor(grasp_quats, device=device, dtype=dtype)
        .reshape(1, 1, 1, len(candidates), 4),
    )
    plan = planner.plan_grasp(
        grasp_poses=goals,
        current_state=start,
        grasp_approach_axis="z",
        grasp_approach_offset=-0.0606,
        grasp_approach_in_tool_frame=True,
        grasp_lift_axis="z",
        grasp_lift_offset=0.080,
        grasp_lift_in_tool_frame=False,
        plan_approach_to_grasp=True,
        plan_grasp_to_lift=True,
    )

    selected = None
    if getattr(plan, "goalset_index", None) is not None:
        selected_tensor = plan.goalset_index.detach().cpu().reshape(-1)
        selected = int(selected_tensor[0].item()) if selected_tensor.numel() else None
    candidate_outputs = []
    for index, candidate in enumerate(candidates):
        candidate_outputs.append({
            "candidate_id": candidate["candidate_id"],
            "ik_success_by_phase": dict(zip(("pregrasp", "grasp", "lift30", "lift80"), ik_success[index])),
            "ik_position_error_m_by_phase": dict(zip(("pregrasp", "grasp", "lift30", "lift80"), ik_position_error[index])),
            "ik_qpos_by_phase": dict(zip(("pregrasp", "grasp", "lift30", "lift80"), q_solutions[index])),
            "selected_by_curobo_grasp_trajopt": index == selected,
        })
    success = bool(getattr(plan, "success", torch.tensor([False])).any().item())
    out = {
        "schema": "gs_opd_curobo_t236_grasp_proposals.v1",
        "curobo_version": "v2_source_main",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "world_base_transform": world_base.tolist(),
        "calibration_source": "live_t236_MuJoCo_EEF_pose_plus_CuRobo_Franka_FK_at_identical_arm_q",
        "ik_scene_model": "Franka self-collision model; no exact LIBERO scene mesh",
        "ik_goal_count": len(ik_positions),
        "ik_success_count": int(raw_success.sum()),
        "plan_grasp_success": success,
        "plan_grasp_status": str(getattr(plan, "status", "")),
        "plan_grasp_selected_candidate_index": selected,
        "plan_grasp_stage_success": {
            "approach": bool(getattr(plan, "approach_success", torch.tensor([False])).any().item()),
            "grasp": bool(getattr(plan, "grasp_success", torch.tensor([False])).any().item()),
            "lift": bool(getattr(plan, "lift_success", torch.tensor([False])).any().item()),
        },
        "plan_grasp_stage_trajectories": {
            name: _traj_summary(plan, name) for name in ("approach", "grasp", "lift")
        },
        "candidates": candidate_outputs,
        "elapsed_seconds": time.monotonic() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({
        "plan_grasp_success": result["plan_grasp_success"],
        "ik_success_count": result["ik_success_count"],
        "selected_candidate_index": result["plan_grasp_selected_candidate_index"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output": str(args.output),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
