# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0.

"""Plan Human L0T2 transport goals in a MuJoCo-derived CuRobo sphere world."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, Sphere
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.util.config_io import load_yaml
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.content import get_robot_configs_path
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


def _tensor_summary(value):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return value if isinstance(value, (str, int, float, bool)) else str(type(value))
    tensor = value.detach().cpu()
    summary = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if tensor.numel() <= 768:
        summary["values"] = tensor.tolist()
    elif tensor.dtype == torch.bool:
        summary["true_count"] = int(tensor.sum().item())
    elif tensor.is_floating_point():
        finite = tensor[torch.isfinite(tensor)]
        summary["finite_count"] = int(finite.numel())
        if finite.numel():
            summary["min"] = float(finite.min().item())
            summary["max"] = float(finite.max().item())
    return summary


def _tensor_values(value):
    """Return a compact numeric tensor payload for the chosen trajectory."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().float().cpu().numpy().tolist()


def _raw_planner_result(result):
    if result is None:
        return {"returned_none": True, "result_type": None, "attributes": {}}
    names = (
        "success", "status", "solve_time", "total_time", "position_error",
        "orientation_error", "goalset_index", "interpolated_last_tstep",
        "minimum_trajectory_dt", "maximum_trajectory_dt",
    )
    return {
        "returned_none": False,
        "result_type": f"{type(result).__module__}.{type(result).__qualname__}",
        "available_attribute_names": sorted(set(vars(result)) | {name for name in names if hasattr(result, name)}),
        "attributes": {
            name: _tensor_summary(getattr(result, name, None))
            for name in names
            if hasattr(result, name)
        },
    }


def _single_obstacle_scene(obstacle):
    if isinstance(obstacle, Sphere):
        return SceneCfg(sphere=[obstacle])
    if isinstance(obstacle, Cuboid):
        return SceneCfg(cuboid=[obstacle])
    if isinstance(obstacle, Mesh):
        return SceneCfg(mesh=[obstacle])
    raise TypeError(f"unsupported_CuRobo_obstacle_type:{type(obstacle).__name__}")


def _payload_sphere_coverage(source_components, fitted_spheres):
    fitted = np.asarray(fitted_spheres, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    component_rows = []
    total_samples = covered_samples = 0
    for component in source_components:
        center = np.asarray(component["center_object_root_m"], dtype=np.float64)
        radius = float(component["radius_m"])
        points = [center]
        for index in range(512):
            z = 1.0 - 2.0 * (index + 0.5) / 512.0
            radial = np.sqrt(max(0.0, 1.0 - z * z))
            angle = golden * index
            points.append(center + radius * np.asarray([
                radial * np.cos(angle), radial * np.sin(angle), z
            ]))
        points = np.asarray(points)
        distances = np.linalg.norm(
            points[:, None, :] - fitted[None, :, :3], axis=2
        ) - fitted[None, :, 3]
        covered = np.any(distances <= 1e-7, axis=1)
        total_samples += len(points)
        covered_samples += int(np.count_nonzero(covered))
        center_distances = np.linalg.norm(center[None, :] - fitted[:, :3], axis=1)
        contains_whole_source_sphere = bool(
            np.any(center_distances + radius <= fitted[:, 3] + 1e-7)
        )
        component_rows.append({
            "source_geom_id": component.get("source_geom_id"),
            "center_object_root_m": center.tolist(),
            "source_enclosure_radius_m": radius,
            "fully_enclosed_by_one_attached_sphere": contains_whole_source_sphere,
            "deterministic_surface_sample_coverage": float(np.mean(covered)),
            "coverage_is_sampling_only_not_a_containment_proof": True,
        })
    return {
        "source_collision_geom_count": len(source_components),
        "fitted_attached_sphere_count": int(len(fitted)),
        "source_component_frame": "moka_pot_1 object root",
        "fitted_sphere_frame": "moka_pot_1 object root before AttachmentManager object-to-panda_hand transform",
        "components": component_rows,
        "aggregate_surface_sample_coverage": float(covered_samples / total_samples) if total_samples else None,
        "fully_enclosed_component_count": sum(
            row["fully_enclosed_by_one_attached_sphere"] for row in component_rows
        ),
        "coverage_conclusion": (
            "sampled_union_covers_all_component_boundaries"
            if covered_samples == total_samples else
            "fitted_spheres_undercover_sampled_source_geometry"
        ),
    }


def _payload_attachment_frame_audit(planner, current_state, attached_spheres, payload, world_base):
    state = planner.compute_kinematics(current_state)
    sphere_tensor = state.robot_spheres[0, 0].detach().cpu().numpy()
    kconfig = planner.kinematics.config.kinematics_config
    sphere_link_map = kconfig.link_sphere_idx_map.detach().cpu().numpy().reshape(-1)
    link_by_index = {int(index): str(name) for name, index in kconfig.link_name_to_idx_map.items()}
    attached_indices = [
        i for i, link_index in enumerate(sphere_link_map)
        if link_by_index.get(int(link_index)) == "attached_object"
        and float(sphere_tensor[i, 3]) > 0.0
    ]
    fitted = attached_spheres.detach().cpu().numpy()
    object_world = _pose_matrix(
        payload["root_position_world_m"], payload["root_quaternion_world_wxyz"]
    )
    actual_world = []
    expected_world = []
    for sphere_index, fitted_sphere in zip(attached_indices, fitted):
        actual = world_base[:3, :3] @ sphere_tensor[sphere_index, :3] + world_base[:3, 3]
        expected = object_world[:3, :3] @ fitted_sphere[:3] + object_world[:3, 3]
        actual_world.append({
            "robot_collision_sphere_index": int(sphere_index),
            "center_world_m": actual.tolist(),
            "radius_m": float(sphere_tensor[sphere_index, 3]),
        })
        expected_world.append(expected)

    hand_link_indices = {
        link_by_index.get(int(index))
        for index in range(len(sphere_link_map))
        if link_by_index.get(int(sphere_link_map[index])) in {
            "panda_hand", "panda_leftfinger", "panda_rightfinger"
        }
    }
    hand_sphere_indices = [
        i for i, link_index in enumerate(sphere_link_map)
        if link_by_index.get(int(link_index)) in {
            "panda_hand", "panda_leftfinger", "panda_rightfinger"
        }
        and float(sphere_tensor[i, 3]) > 0.0
    ]
    overlaps = []
    for attached_index in attached_indices:
        for hand_index in hand_sphere_indices:
            gap = float(
                np.linalg.norm(sphere_tensor[attached_index, :3] - sphere_tensor[hand_index, :3])
                - sphere_tensor[attached_index, 3] - sphere_tensor[hand_index, 3]
            )
            if gap < 0.0:
                overlaps.append({
                    "attached_object_sphere_index": int(attached_index),
                    "robot_sphere_index": int(hand_index),
                    "robot_link": link_by_index.get(int(sphere_link_map[hand_index])),
                    "signed_gap_m": gap,
                    "self_collision_pair_is_ignored_by_franka_config": True,
                })
    actual_arr = np.asarray([row["center_world_m"] for row in actual_world], dtype=np.float64)
    expected_arr = np.asarray(expected_world, dtype=np.float64)
    return {
        "status": "complete" if len(attached_indices) == len(fitted) else "incomplete",
        "fitted_sphere_centers_preserved_after_AttachmentManager_transform": len(attached_indices) == len(fitted),
        "maximum_world_center_transform_error_m": float(np.max(np.linalg.norm(actual_arr - expected_arr, axis=1))) if len(expected_arr) else None,
        "actual_attached_spheres_world": actual_world,
        "expected_attached_sphere_centers_world": expected_arr.tolist(),
        "gripper_overlap_pairs": overlaps,
        "gripper_overlap_handling": "CuRobo franka.yml self_collision_ignore excludes panda_hand/panda_leftfinger/panda_rightfinger against attached_object",
        "other_scene_obstacle_interactions": "reported in configuration_collision_audit_complete_curobo_world by attached_object sphere link identity",
    }


def _audit_curobo_configuration_collisions(
    planner, full_world, obstacles, q_labels, q_values
):
    """Return per-state and per-obstacle collision status for CuRobo spheres."""
    if len(q_labels) != len(q_values) or not q_values:
        raise ValueError("collision_audit_requires_matching_nonempty_joint_state_labels")
    kinematic_states = []
    for q in q_values:
        q_tensor = torch.as_tensor(np.asarray(q)[None, :], device=planner.device_cfg.device, dtype=planner.device_cfg.dtype)
        state = JointState.from_position(q_tensor, joint_names=planner.joint_names)
        kinematic_states.append(planner.compute_kinematics(state).robot_spheres)
    sphere_states = torch.cat(kinematic_states, dim=0)
    kconfig = planner.kinematics.config.kinematics_config
    sphere_link_map = kconfig.link_sphere_idx_map.detach().cpu().numpy().reshape(-1)
    link_by_index = {int(index): str(name) for name, index in kconfig.link_name_to_idx_map.items()}
    if len(sphere_link_map) != int(sphere_states.shape[2]):
        raise ValueError("CuRobo_link_to_collision_sphere_map_length_mismatch")
    sphere_link_names = [link_by_index.get(int(index), f"unknown_link_{int(index)}") for index in sphere_link_map]
    device = planner.device_cfg.device
    dtype = planner.device_cfg.dtype
    weight = torch.ones((1,), device=device, dtype=dtype)
    activation = torch.zeros((1,), device=device, dtype=dtype)
    collision_buffer = CollisionBuffer.from_shape(sphere_states.shape, planner.device_cfg)

    def query_current_scene():
        collision_buffer.zero_()
        return planner.scene_collision_checker.get_sphere_distance_raw(
            sphere_states, collision_buffer, weight, activation
        ).detach().cpu().numpy()[:, 0, :]

    planner.scene_collision_checker.load_collision_model(full_world)
    full_costs = query_current_scene()
    colliding_state_indices = [
        index for index, row in enumerate(full_costs)
        if bool(np.any(row > 1e-7))
    ]
    rows = []
    for index, label in enumerate(q_labels):
        colliding_spheres = [
            {
                "sphere_index": int(sphere_index),
                "robot_link": sphere_link_names[sphere_index],
                "curobo_collision_cost_m": float(cost),
                "sphere_center_base_m": sphere_states[index, 0, sphere_index, :3].detach().cpu().tolist(),
                "sphere_radius_m": float(sphere_states[index, 0, sphere_index, 3].detach().cpu().item()),
            }
            for sphere_index, cost in enumerate(full_costs[index])
            if cost > 1e-7
        ]
        rows.append({
            "state_id": label,
            "status": "colliding" if colliding_spheres else "clear",
            "full_world_collision_cost_sum_m": float(np.sum(full_costs[index])),
            "colliding_spheres": colliding_spheres,
            "collisions": [],
        })

    if colliding_state_indices:
        for obstacle in obstacles:
            planner.scene_collision_checker.load_collision_model(_single_obstacle_scene(obstacle))
            obstacle_costs = query_current_scene()
            for state_index in colliding_state_indices:
                for sphere_index, cost in enumerate(obstacle_costs[state_index]):
                    if cost <= 1e-7:
                        continue
                    rows[state_index]["collisions"].append({
                        "robot_link": sphere_link_names[sphere_index],
                        "robot_collision_sphere_index": int(sphere_index),
                        "scene_geom": str(obstacle.name),
                        "penetration_proxy_m": float(cost),
                        "scene_geom_source": "exact_obstacle_primitive_or_compiled_mesh_enclosure_passed_to_CuRobo",
                    })
    planner.scene_collision_checker.load_collision_model(full_world)
    for row in rows:
        row["collisions"].sort(
            key=lambda pair: (-pair["penetration_proxy_m"], pair["scene_geom"], pair["robot_link"])
        )
    return {
        "status": "complete",
        "query": "CuRobo robot collision spheres against complete active SceneCfg; zero activation distance",
        "penetration_convention": "positive collision cost in meters equals sphere-vs-obstacle penetration for zero activation distance; the robot geometry remains CuRobo's configured sphere model",
        "obstacle_count": len(obstacles),
        "robot_collision_sphere_count": int(sphere_states.shape[2]),
        "attached_object_sphere_count": int(sum(name == "attached_object" for name in sphere_link_names)),
        "states": rows,
    }


def _run(source: dict, diagnostic_only: bool = False) -> dict:
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
    required_frame_fields = (
        "robot_base_pose_world_m",
        "robot_base_quaternion_world_wxyz",
        "live_planner_tool_pose_world_m",
        "live_planner_tool_quaternion_world_wxyz",
        "eef_to_planner_tool_transform",
    )
    missing_frame_fields = [name for name in required_frame_fields if name not in source]
    if missing_frame_fields:
        raise ValueError(
            "CuRobo_frame_context_missing_verified_MuJoCo_base_and_tool_transform:"
            + ",".join(missing_frame_fields)
        )
    world_base = _pose_matrix(
        source["robot_base_pose_world_m"],
        source["robot_base_quaternion_world_wxyz"],
    )
    live_planner_tool_world = _pose_matrix(
        source["live_planner_tool_pose_world_m"],
        source["live_planner_tool_quaternion_world_wxyz"],
    )
    eef_to_planner_tool = np.asarray(
        source["eef_to_planner_tool_transform"], dtype=np.float64
    )
    if (
        eef_to_planner_tool.shape != (4, 4)
        or not np.isfinite(eef_to_planner_tool).all()
        or not np.allclose(eef_to_planner_tool[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
    ):
        raise ValueError("MuJoCo_EEF_to_CuRobo_tool_transform_is_invalid")
    curobo_planner_tool_world = world_base @ curobo_base_eef
    planner_tool_position_error = float(
        np.linalg.norm(curobo_planner_tool_world[:3, 3] - live_planner_tool_world[:3, 3])
    )
    planner_tool_rotation_error = float(
        Rotation.from_matrix(
            curobo_planner_tool_world[:3, :3].T @ live_planner_tool_world[:3, :3]
        ).magnitude()
    )
    if planner_tool_position_error > 0.002 or planner_tool_rotation_error > np.deg2rad(0.10):
        raise ValueError(
            "CuRobo_panda_hand_FK_does_not_match_MuJoCo_robot0_right_hand:"
            f"position_error_m={planner_tool_position_error:.9g},"
            f"rotation_error_rad={planner_tool_rotation_error:.9g}"
        )
    legacy_world_base = live_eef_world @ np.linalg.inv(curobo_base_eef)
    legacy_base_position_error = float(
        np.linalg.norm(legacy_world_base[:3, 3] - world_base[:3, 3])
    )
    legacy_base_rotation_error = float(
        Rotation.from_matrix(world_base[:3, :3].T @ legacy_world_base[:3, :3]).magnitude()
    )
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

    robot_config = load_yaml(
        str(Path(get_robot_configs_path()) / "franka.yml")
    )
    payload_components = (
        source.get("attached_payload", {}).get("collision_components", [])
        if source.get("attached_payload") is not None else []
    )
    # Keep one CuRobo attachment sphere for every verified MuJoCo collision
    # Geom enclosure. MORPHIT's four-sphere fit both under-covered source
    # Geoms and produced bottle collisions that exact MuJoCo geomDistance
    # disproved for the audited goal configurations.
    if payload_components:
        robot_config["robot_cfg"]["kinematics"]["extra_collision_spheres"][
            "attached_object"
        ] = len(payload_components)
    cfg = MotionPlannerCfg.create(
        robot=robot_config,
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
    payload_geometry_coverage = None
    payload_attachment_frame_audit = None

    if source.get("attached_payload") is not None:
        payload = source["attached_payload"]
        obj_pose_base = base_world @ _pose_matrix(
            payload["root_position_world_m"], payload["root_quaternion_world_wxyz"]
        )
        if not payload["collision_components"]:
            raise ValueError("attached_payload_has_no_verified_collision_components")
        attached_spheres = torch.as_tensor(
            [
                [
                    *component["center_object_root_m"],
                    float(component["radius_m"]),
                ]
                for component in payload["collision_components"]
            ],
            device=device,
            dtype=dtype,
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
        payload_geometry_coverage = _payload_sphere_coverage(
            payload["collision_components"],
            attached_spheres.detach().cpu().numpy(),
        )
        payload_attachment_frame_audit = _payload_attachment_frame_audit(
            planner, current_state, attached_spheres, payload, world_base
        )

    goal_positions, goal_quaternions = [], []
    for goal in source["goal_poses_world"]:
        pose_world_eef = _pose_matrix(
            goal["eef_position_world_m"], goal["eef_quaternion_world_wxyz"]
        )
        pose_world = pose_world_eef @ eef_to_planner_tool
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

    collision_labels = ["start_t215"]
    collision_qs = [current_q]
    candidate_ik_status = []
    for index, goal_row in enumerate(source["goal_poses_world"]):
        candidate_id = goal_row.get("candidate_id", index)
        if bool(ik_success[index]):
            collision_labels.append(f"goal_{index}:{candidate_id}")
            collision_qs.append(np.asarray(ik_solutions[index], dtype=np.float64))
            candidate_ik_status.append({
                "candidate_index": index,
                "candidate_id": candidate_id,
                "status": "self_collision_ik_solution_available",
            })
        else:
            candidate_ik_status.append({
                "candidate_index": index,
                "candidate_id": candidate_id,
                "status": "no_self_collision_ik_solution",
                "collision_status": "unknown_no_endpoint_joint_configuration",
            })
    full_world = SceneCfg(sphere=spheres or None, cuboid=cuboids or None, mesh=meshes or None)
    configuration_collision_audit = _audit_curobo_configuration_collisions(
        planner,
        full_world,
        [*spheres, *cuboids, *meshes],
        collision_labels,
        collision_qs,
    )
    endpoint_collision_by_candidate = {
        row["state_id"].split(":", 1)[0].removeprefix("goal_"): row
        for row in configuration_collision_audit["states"][1:]
    }
    for candidate in candidate_ik_status:
        if candidate["status"] != "self_collision_ik_solution_available":
            continue
        state = endpoint_collision_by_candidate.get(str(candidate["candidate_index"]))
        if state is None:
            candidate["collision_status"] = "not_returned_by_collision_audit"
        else:
            candidate["collision_status"] = state["status"]
            candidate["collisions"] = state["collisions"]
            candidate["full_world_collision_cost_sum_m"] = state["full_world_collision_cost_sum_m"]
    start_clear = configuration_collision_audit["states"][0]["status"] == "clear"
    clear_endpoint_count = sum(
        item.get("collision_status") == "clear" for item in candidate_ik_status
    )
    planner_gate = {
        "status": "pass" if start_clear and clear_endpoint_count else "blocked",
        "start_clear_in_complete_world": start_clear,
        "clear_endpoint_count": int(clear_endpoint_count),
        "required": "start plus at least one IK endpoint clear against every CuRobo world obstacle with attached Pot1",
    }

    goal = GoalToolPose(
        tool_frames=["panda_hand"],
        position=torch.as_tensor(goal_positions, device=device, dtype=dtype)
        .reshape(1, 1, 1, len(goal_positions), 3),
        quaternion=torch.as_tensor(goal_quaternions, device=device, dtype=dtype)
        .reshape(1, 1, 1, len(goal_quaternions), 4),
    )
    raw_full_scene_ik_attempts = []
    original_solve_pose = planner.ik_solver.solve_pose

    def _traced_solve_pose(*args, **kwargs):
        raw = original_solve_pose(*args, **kwargs)
        raw_full_scene_ik_attempts.append({
            "result_type": f"{type(raw).__module__}.{type(raw).__qualname__}",
            "success": _tensor_summary(getattr(raw, "success", None)),
            "position_error": _tensor_summary(getattr(raw, "position_error", None)),
            "position_error_norm": _tensor_summary(getattr(raw, "position_error_norm", None)),
            "goalset_index": _tensor_summary(getattr(raw, "goalset_index", None)),
            "solution": _tensor_summary(getattr(raw, "solution", None)),
            "js_solution_position": _tensor_summary(
                getattr(getattr(raw, "js_solution", None), "position", None)
            ),
            "solve_time": getattr(raw, "solve_time", None),
            "total_time": getattr(raw, "total_time", None),
        })
        return raw

    planner.ik_solver.solve_pose = _traced_solve_pose
    if diagnostic_only:
        result = None
    elif planner_gate["status"] == "pass":
        result = planner.plan_pose(goal, current_state, max_attempts=3)
    else:
        result = None
    raw_motion_result = _raw_planner_result(result)
    success = bool(result is not None and result.success is not None and result.success.any().item())
    selected = None
    q_path = None
    planner_joint_path = None
    eef_path = None
    trajectory_profile = None
    optimized_trajectory_profile = None
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
        if q_tensor_path.ndim != 2 or q_tensor_path.shape[1] < 7:
            raise RuntimeError(
                "CuRobo reported_success_with_invalid_joint_path_shape:"
                f"{tuple(q_tensor_path.shape)}"
            )
        last_tstep_tensor = getattr(result, "interpolated_last_tstep", None)
        if last_tstep_tensor is None or not last_tstep_tensor.numel():
            raise RuntimeError("CuRobo_success_missing_interpolated_last_tstep")
        # CuRobo's get_batch_interpolated_trajectory returns this as a point
        # count: the interpolation kernel fills [0:last_tstep) and pads from
        # last_tstep onward with the final configuration.
        valid_point_count = int(last_tstep_tensor.detach().reshape(-1)[0].item())
        if valid_point_count < 2 or valid_point_count > len(q_tensor_path):
            raise RuntimeError(
                "CuRobo_interpolated_valid_point_count_out_of_range:"
                f"{valid_point_count}/{len(q_tensor_path)}"
            )
        trajectory_dt_tensor = getattr(trajectory, "dt", None)
        trajectory_dt_values = (
            trajectory_dt_tensor.detach().float().cpu().reshape(-1).tolist()
            if isinstance(trajectory_dt_tensor, torch.Tensor) else []
        )
        effective_dt = (
            float(trajectory_dt_values[0])
            if trajectory_dt_values and np.isfinite(trajectory_dt_values[0])
            else float(source.get("interpolation_dt_s", 0.05))
        )
        if effective_dt <= 0.0 or not np.isfinite(effective_dt):
            raise RuntimeError(f"CuRobo_interpolated_dt_invalid:{effective_dt}")

        def _trajectory_component(name):
            component = getattr(trajectory, name, None)
            if not isinstance(component, torch.Tensor):
                return None
            array = component.detach().float().cpu().numpy().reshape(
                -1, component.shape[-1]
            )
            if array.shape != q_tensor_path.shape:
                raise RuntimeError(
                    f"CuRobo_interpolated_{name}_shape_mismatch:"
                    f"{array.shape}/{q_tensor_path.shape}"
                )
            return array

        valid_velocity = _trajectory_component("velocity")
        valid_acceleration = _trajectory_component("acceleration")
        valid_jerk = _trajectory_component("jerk")
        transition = planner.trajopt_solver.transition_model
        velocity_limits = transition.max_velocity.detach().float().cpu().reshape(-1).tolist()
        acceleration_limits = transition.max_acceleration.detach().float().cpu().reshape(-1).tolist()
        valid_position = q_tensor_path[:valid_point_count]
        trajectory_profile = {
            "source": "CuRobo TrajOptSolverResult.interpolated_trajectory",
            "interpolated_last_tstep_semantics": "valid_point_count; zero-based last valid index is count-1",
            "buffer_point_count": int(len(q_tensor_path)),
            "effective_valid_point_count": valid_point_count,
            "effective_last_valid_point_index": valid_point_count - 1,
            "padding_point_count": int(len(q_tensor_path) - valid_point_count),
            "interpolation_dt_s": effective_dt,
            "configured_interpolation_dt_s": float(source.get("interpolation_dt_s", 0.05)),
            "valid_motion_duration_s": float((valid_point_count - 1) * effective_dt),
            "time_s": (np.arange(valid_point_count, dtype=np.float64) * effective_dt).tolist(),
            "position_rad": valid_position.tolist(),
            "velocity_rad_s": valid_velocity[:valid_point_count].tolist() if valid_velocity is not None else None,
            "acceleration_rad_s2": valid_acceleration[:valid_point_count].tolist() if valid_acceleration is not None else None,
            "jerk_rad_s3": valid_jerk[:valid_point_count].tolist() if valid_jerk is not None else None,
            "joint_velocity_limits_rad_s": velocity_limits,
            "joint_acceleration_limits_rad_s2": acceleration_limits,
            "max_abs_interpolated_velocity_rad_s": (
                np.max(np.abs(valid_velocity[:valid_point_count]), axis=0).tolist()
                if valid_velocity is not None else None
            ),
            "max_abs_interpolated_acceleration_rad_s2": (
                np.max(np.abs(valid_acceleration[:valid_point_count]), axis=0).tolist()
                if valid_acceleration is not None else None
            ),
            "derivative_arrays_are_from_planner_result": bool(
                valid_velocity is not None and valid_acceleration is not None
            ),
        }

        optimized = getattr(result, "js_solution", None)
        if optimized is not None:
            optimized_trajectory_profile = {
                "source": "CuRobo TrajOptSolverResult.js_solution",
                "position_rad": _tensor_values(getattr(optimized, "position", None)),
                "velocity_rad_s": _tensor_values(getattr(optimized, "velocity", None)),
                "acceleration_rad_s2": _tensor_values(getattr(optimized, "acceleration", None)),
                "jerk_rad_s3": _tensor_values(getattr(optimized, "jerk", None)),
                "dt_s": _tensor_values(getattr(optimized, "dt", None)),
            }
        planner_joint_path = q_tensor_path.tolist()
        q_path = q_tensor_path[:, :7].tolist()
        eef_path = []
        for q in q_tensor_path:
            joint = JointState.from_position(
                torch.as_tensor(q[None, :7], device=device, dtype=dtype),
                joint_names=kin.joint_names,
            )
            eef = kin.compute_kinematics(joint).tool_poses.get_link_pose("panda_hand")
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
        "base_pose_source": "live_read_only_MuJoCo_robot_model_root_body",
        "legacy_eef_to_panda_hand_base_error": {
            "position_m": legacy_base_position_error,
            "rotation_rad": legacy_base_rotation_error,
            "conclusion": "old single-pose calibration used MuJoCo gripper0_eef site as CuRobo panda_hand",
        },
        "planner_tool_frame_validation": {
            "curobo_tool_frame": "panda_hand",
            "mujoco_corresponding_body": "robot0_right_hand",
            "position_error_m": planner_tool_position_error,
            "rotation_error_rad": planner_tool_rotation_error,
            "eef_to_planner_tool_transform": eef_to_planner_tool.tolist(),
        },
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
            "one attached sphere per verified MuJoCo collision-Geom enclosure; centers and radii preserved exactly"
            if source.get("attached_payload") is not None else None
        ),
        "attached_payload_source_geometry_coverage": payload_geometry_coverage,
        "attached_payload_frame_and_gripper_audit": payload_attachment_frame_audit,
        "goal_candidates": source["goal_poses_world"],
        "parallel_self_collision_ik": {
            "candidate_count": len(goal_positions),
            "success_count": int(sum(bool(value) for value in ik_success)),
            "success_by_candidate": ik_success,
            "position_error_m_by_candidate": ik_position_error,
            "joint_solution_rad_by_candidate": ik_solutions,
            "environment_obstacle_collision_included": False,
        },
        "endpoint_ik_collision_statuses_complete_curobo_world": candidate_ik_status,
        "configuration_collision_audit_complete_curobo_world": configuration_collision_audit,
        "planner_execution_gate": planner_gate,
        "raw_motion_planner_result": raw_motion_result,
        "trajectory_profile": trajectory_profile,
        "optimized_trajectory_profile": optimized_trajectory_profile,
        "raw_full_scene_ik_attempts": raw_full_scene_ik_attempts,
        "planner_joint_path_rad_full": planner_joint_path,
        "planner_trajectory_joint_count": (
            int(len(planner_joint_path[0])) if planner_joint_path else None
        ),
        "execution_joint_path_rad_arm7": q_path,
        "eef_path_derived_from_arm7_rad": eef_path,
        "selected_goal_index": selected,
        "success": success,
        "diagnostic_only": bool(diagnostic_only),
        "configured_attached_object_sphere_count": len(payload_components),
        "planner_status": (
            "diagnostic_only_planner_not_called" if diagnostic_only else
            "planned" if success else
            "blocked_before_planner_start_or_goal_collision" if planner_gate["status"] != "pass" else
            "planner_returned_none_with_raw_full_scene_ik_state" if result is None else
            "planner_result_unsuccessful_with_raw_result_state"
        ),
        "planner_none_explanation": (
            "Planner call deliberately omitted after recording read-only CuRobo FK/collision diagnostics"
            if diagnostic_only else
            "MotionPlanner.plan_pose was not called because the verified start or all IK endpoints collide in the complete CuRobo world"
            if planner_gate["status"] != "pass" else
            "See raw_full_scene_ik_attempts: MotionPlanner._plan_pose_goalset returns None when its full-world IK returns zero successful seeds"
            if result is None else None
        ),
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
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()
    try:
        result = _run(
            json.loads(args.input.read_text(encoding="utf-8")),
            diagnostic_only=args.diagnostic_only,
        )
    except Exception as exc:
        result = {
            "schema": "gs_opd_curobo_human_l0t2_joint_plan.v1",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "exception_traceback": traceback.format_exc(),
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
