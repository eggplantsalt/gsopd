# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0.

"""Plan Human L0T2 transport goals in a MuJoCo-derived CuRobo sphere world."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, Sphere
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose


def _pose_matrix(position, quaternion_wxyz):
    out = np.eye(4, dtype=np.float64)
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    out[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    out[:3, 3] = np.asarray(position, dtype=np.float64)
    return out


def _matrix_pose(matrix):
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return matrix[:3, 3].tolist(), [float(q[3]), *q[:3].tolist()]


def _run(source: dict) -> dict:
    started = time.monotonic()
    device = torch.device("cuda:0")
    dtype = torch.float32
    current_q = np.asarray(source["current_arm_qpos_rad"], dtype=np.float64)
    if current_q.shape != (7,) or not np.isfinite(current_q).all():
        raise ValueError("current_arm_qpos_rad_must_be_finite_7D")

    kin = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
    q_tensor = torch.as_tensor(current_q[None, :], device=device, dtype=dtype)
    current_state = JointState.from_position(q_tensor, joint_names=kin.joint_names)
    current_fk = kin.compute_kinematics(current_state).tool_poses.get_link_pose("panda_hand")
    curobo_base_eef = _pose_matrix(
        current_fk.position[0].detach().cpu().numpy(),
        current_fk.quaternion[0].detach().cpu().numpy(),
    )
    live_eef_world = _pose_matrix(
        source["live_eef_position_world_m"], source["live_eef_quaternion_world_wxyz"]
    )
    world_base = live_eef_world @ np.linalg.inv(curobo_base_eef)
    base_world = np.linalg.inv(world_base)

    spheres, cuboids, meshes = [], [], []
    for row in source["obstacle_geometries_world"]:
        if row["shape"] == "mesh":
            vertices_world = np.asarray(row["vertices_world_m"], dtype=np.float64)
            faces = np.asarray(row["faces"], dtype=np.int64)
            if vertices_world.ndim != 2 or vertices_world.shape[1] != 3 or not np.isfinite(vertices_world).all():
                raise ValueError("invalid_MuJoCo_compiled_collision_mesh_vertices")
            if faces.ndim != 1 or faces.size == 0 or faces.size % 3 or np.any(faces < 0) or np.any(faces >= len(vertices_world)):
                raise ValueError("invalid_MuJoCo_compiled_collision_mesh_faces")
            vertices_base = vertices_world @ base_world[:3, :3].T + base_world[:3, 3][None, :]
            meshes.append(
                Mesh(
                    name=str(row["name"]),
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    vertices=vertices_base.tolist(),
                    faces=faces.tolist(),
                )
            )
            continue
        center = np.r_[np.asarray(row["center_world_m"], dtype=np.float64), 1.0]
        center_base = (base_world @ center)[:3]
        if not np.isfinite(center_base).all():
            raise ValueError("invalid_MuJoCo_derived_obstacle_center")
        if row["shape"] == "sphere":
            radius = float(row["radius_m"])
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("invalid_MuJoCo_derived_obstacle_sphere")
            spheres.append(
                Sphere(
                    name=str(row["name"]),
                    pose=[*center_base.tolist(), 1.0, 0.0, 0.0, 0.0],
                    radius=radius,
                )
            )
        elif row["shape"] == "cuboid":
            world_rotation = np.asarray(row["rotation_world"], dtype=np.float64).reshape(3, 3)
            base_rotation = base_world[:3, :3] @ world_rotation
            quaternion = Rotation.from_matrix(base_rotation).as_quat()
            half = np.asarray(row["half_extents_world_axes_m"], dtype=np.float64)
            if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
                raise ValueError("invalid_MuJoCo_derived_obstacle_cuboid")
            cuboids.append(
                Cuboid(
                    name=str(row["name"]),
                    pose=[*center_base.tolist(), float(quaternion[3]), *quaternion[:3].tolist()],
                    dims=(2.0 * half).tolist(),
                )
            )
        else:
            raise ValueError(f"unsupported_MuJoCo_obstacle_shape:{row['shape']}")
    if not spheres and not cuboids and not meshes:
        raise ValueError("MuJoCo_derived_collision_world_is_empty")

    cfg = MotionPlannerCfg.create(
        robot="franka.yml",
        collision_cache={"sphere": len(spheres), "cuboid": len(cuboids), "mesh": len(meshes)},
        self_collision_check=True,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        use_cuda_graph=False,
        random_seed=int(source.get("random_seed", 20261026)),
        max_batch_size=1,
        max_goalset=len(source["goal_poses_world"]),
        interpolation_dt=float(source.get("interpolation_dt_s", 0.05)),
        interpolation_buffer_size=int(source.get("interpolation_buffer_size", 1800)),
    )
    planner = MotionPlanner(cfg)
    planner.update_world(SceneCfg(sphere=spheres or None, cuboid=cuboids or None, mesh=meshes or None))

    if source.get("attached_payload") is not None:
        payload = source["attached_payload"]
        obj_pose_base = base_world @ _pose_matrix(
            payload["root_position_world_m"], payload["root_quaternion_world_wxyz"]
        )
        payload_obstacles = [
            Sphere(
                name=f"attached_payload_geom_{index}",
                position=component["center_object_root_m"],
                radius=float(component["radius_m"]),
            )
            for index, component in enumerate(payload["collision_components"])
        ]
        if not payload_obstacles:
            raise ValueError("attached_payload_has_no_verified_collision_components")
        attached_spheres = planner.attachment_manager.fit_spheres(
            payload_obstacles,
            num_spheres=4,
            surface_radius=0.003,
            sphere_fit_type=SphereFitType.MORPHIT,
        )
        obj_pose = Pose(
            position=torch.as_tensor(obj_pose_base[:3, 3][None, :], device=device, dtype=dtype),
            quaternion=torch.as_tensor(
                Rotation.from_matrix(obj_pose_base[:3, :3]).as_quat()[[3, 0, 1, 2]][None, :],
                device=device,
                dtype=dtype,
            ),
        )
        planner.attachment_manager.update(
            attached_spheres,
            current_state,
            link_name="attached_object",
            world_objects_pose_offset=obj_pose,
        )

    goal_positions, goal_quaternions = [], []
    for goal in source["goal_poses_world"]:
        pose_world = _pose_matrix(
            goal["eef_position_world_m"], goal["eef_quaternion_world_wxyz"]
        )
        local = base_world @ pose_world
        position, quaternion = _matrix_pose(local)
        goal_positions.append(position)
        goal_quaternions.append(quaternion)

    ik_cfg = InverseKinematicsCfg.create(
        robot="franka.yml",
        num_seeds=32,
        max_batch_size=len(goal_positions),
        self_collision_check=True,
    )
    ik_solver = InverseKinematics(ik_cfg)
    ik_pose = Pose(
        position=torch.as_tensor(goal_positions, device=device, dtype=dtype),
        quaternion=torch.as_tensor(goal_quaternions, device=device, dtype=dtype),
        normalize_rotation=True,
    )
    ik_result = ik_solver.solve_pose(
        GoalToolPose.from_poses({ik_solver.tool_frames[0]: ik_pose}, num_goalset=1)
    )
    ik_success = ik_result.success.detach().cpu().numpy().reshape(-1).tolist()
    ik_position_error = ik_result.position_error.detach().cpu().numpy().reshape(-1).tolist()
    ik_solutions = ik_result.js_solution.position.detach().float().cpu().numpy().reshape(
        len(goal_positions), -1, ik_result.js_solution.position.shape[-1]
    )[:, 0, :7].tolist()

    goal = GoalToolPose(
        tool_frames=["panda_hand"],
        position=torch.as_tensor(goal_positions, device=device, dtype=dtype)
        .reshape(1, 1, 1, len(goal_positions), 3),
        quaternion=torch.as_tensor(goal_quaternions, device=device, dtype=dtype)
        .reshape(1, 1, 1, len(goal_quaternions), 4),
    )
    result = planner.plan_pose(goal, current_state, max_attempts=3)
    success = bool(result is not None and result.success is not None and result.success.any().item())
    selected = None
    q_path = None
    eef_path = None
    if success:
        selection = getattr(result, "goalset_index", None)
        if selection is not None and selection.numel():
            selected = int(selection.detach().reshape(-1)[0].item())
        trajectory = getattr(result, "interpolated_trajectory", None)
        if trajectory is None:
            trajectory = getattr(result, "js_solution", None)
        if trajectory is None:
            raise RuntimeError("CuRobo reported_success_without_joint_trajectory")
        q_tensor_path = trajectory.position.detach().float().cpu().numpy().reshape(-1, trajectory.position.shape[-1])
        q_path = q_tensor_path[:, :7].tolist()
        eef_path = []
        for q in q_tensor_path:
            joint = JointState.from_position(
                torch.as_tensor(q[None, :], device=device, dtype=dtype),
                joint_names=planner.joint_names,
            )
            eef = planner.compute_kinematics(joint).tool_poses.get_link_pose("panda_hand")
            local = _pose_matrix(
                eef.position[0].detach().cpu().numpy(),
                eef.quaternion[0].detach().cpu().numpy(),
            )
            world = world_base @ local
            position, quaternion = _matrix_pose(world)
            eef_path.append(
                {"position_world_m": position, "quaternion_world_wxyz": quaternion}
            )

    out = {
        "schema": "gs_opd_curobo_human_l0t2_joint_plan.v1",
        "status": "planned" if success else "no_route",
        "task": "obstacle_avoidance_human/L0T2",
        "curobo_distribution": "nvidia-curobo",
        "curobo_version": "0.0.post1.dev1",
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "world_base_transform": world_base.tolist(),
        "world_sphere_count": len(spheres),
        "world_sphere_names": [item.name for item in spheres],
        "world_cuboid_count": len(cuboids),
        "world_cuboid_names": [item.name for item in cuboids],
        "world_mesh_count": len(meshes),
        "world_mesh_names": [item.name for item in meshes],
        "self_collision_check": True,
        "attached_payload_in_robot_collision_model": source.get("attached_payload") is not None,
        "attached_payload_spheres": (
            attached_spheres.detach().cpu().numpy().tolist()
            if source.get("attached_payload") is not None else None
        ),
        "attached_payload_geometry_source": (
            "CuRobo MORPHIT four-sphere fit from every verified MuJoCo collision-Geom enclosure"
            if source.get("attached_payload") is not None else None
        ),
        "goal_candidates": source["goal_poses_world"],
        "parallel_self_collision_ik": {
            "candidate_count": len(goal_positions),
            "success_count": int(sum(bool(value) for value in ik_success)),
            "success_by_candidate": ik_success,
            "position_error_m_by_candidate": ik_position_error,
            "joint_solution_rad_by_candidate": ik_solutions,
            "environment_obstacle_collision_included": False,
        },
        "selected_goal_index": selected,
        "success": success,
        "planner_status": str(getattr(result, "status", "planner_returned_none")),
        "joint_path_rad": q_path,
        "eef_path_world": eef_path,
        "interpolation_dt_s": float(source.get("interpolation_dt_s", 0.05)),
        "elapsed_seconds": time.monotonic() - started,
    }
    planner.destroy()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = _run(json.loads(args.input.read_text(encoding="utf-8")))
    except Exception as exc:
        result = {
            "schema": "gs_opd_curobo_human_l0t2_joint_plan.v1",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result.get("status"),
        "goal_index": result.get("selected_goal_index"),
        "world_sphere_count": result.get("world_sphere_count"),
        "path_points": len(result.get("joint_path_rad") or []),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "error": result.get("error"),
        "output": str(args.output),
    }), flush=True)
    return 0 if result.get("status") == "planned" else 1


if __name__ == "__main__":
    raise SystemExit(main())
