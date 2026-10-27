# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0.

"""Try at most two seeded OA/L0T0 Keypose bundles through the existing CuRobo."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from curobo._src.geom.types import Cuboid, Mesh, SceneCfg
from curobo._src.util.config_io import load_yaml
from curobo.content import get_robot_configs_path
from curobo.inverse_kinematics import InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose

from research.gs_opd.libero_safety_repro.keyposes import (
    DEFAULT_VARIANT_BOUNDS,
    KEYPOSE_STAGES,
    Keypose,
    generate_variants,
    inverse_transform,
    keypose_to_base_tool,
    sample_variant_bundle,
    transform_to_pose,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = Path("/workspace/zy_workspace/LIBERO-Safety")
LIBERO_PYTHON = LIBERO_ROOT / ".venv/bin/python"
BRIDGE_SCRIPT = REPO_ROOT / "research/gs_opd/smoke/safety_bridge.py"
AUTHKEY = b"gs-opd-local"
SCENE_SOURCE_DEFAULT = Path(
    "/opt/data/private/gs_opd/rollouts/gs_opd_libero_safety_repro_oa_l0t0_stage_b_20261027_attempt04/scene_source.json.gz"
)
GRASP_EXPERIENCE_REPORT = Path(
    "/opt/data/private/gs_opd/rollouts/task_success_first_cross_scene_recovery_L0T0_20261026_run50/task_success_first_executor_obstacle_avoidance_L0T0_20261026_140642_118627218/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T0_20261026_140743_703417002/run_report.json"
)


def _matrix(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("expected_finite_4x4_transform")
    return result


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _matrix_from_position_rotation(position: Any, rotation: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def _quat_wxyz(rotation: np.ndarray) -> list[float]:
    q = Rotation.from_matrix(rotation).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _pose_object(transform_base_tool: np.ndarray, device: torch.device, dtype) -> Pose:
    return Pose(
        position=torch.as_tensor(transform_base_tool[:3, 3][None, :], device=device, dtype=dtype),
        quaternion=torch.as_tensor(
            np.asarray(_quat_wxyz(transform_base_tool[:3, :3]))[None, :],
            device=device,
            dtype=dtype,
        ),
    )


def _goal_pose(transform_base_tool: np.ndarray, device: torch.device, dtype) -> GoalToolPose:
    return GoalToolPose.from_poses(
        {"panda_hand": _pose_object(transform_base_tool, device, dtype)},
        num_goalset=1,
    )


def _tensor(value: Any) -> np.ndarray | None:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return None


def _metric(value: Any) -> Any:
    array = _tensor(value)
    if array is None:
        if isinstance(value, (int, float, bool, str)) or value is None:
            return value
        return None
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return array.reshape(-1).tolist()


def _make_world(source: dict, world_base: np.ndarray, *, exclude_target: bool, target_world_pose: np.ndarray | None = None):
    world_base_inverse = inverse_transform(world_base)
    target_body = str(source["target_context"]["target_binding"]["root_body_name"])
    cuboids: list[Cuboid] = []
    meshes: list[Mesh] = []
    for row in source["collision_scene_export"]["scene_collision_geometries"]:
        if row.get("body_name") == target_body:
            if exclude_target:
                continue
            if target_world_pose is not None:
                continue
        name = str(row.get("name", f"geom_{row.get('geom_id', len(cuboids))}"))
        if row["shape"] == "mesh":
            vertices_world = np.asarray(row["vertices_world_m"], dtype=np.float64)
            vertices_base = vertices_world @ world_base_inverse[:3, :3].T + world_base_inverse[:3, 3]
            faces = np.asarray(row["faces"], dtype=np.int64)
            meshes.append(Mesh(
                name=name,
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vertices=vertices_base.tolist(),
                faces=faces.tolist(),
            ))
        elif row["shape"] == "cuboid":
            center_world = np.asarray(row["center_world_m"], dtype=np.float64)
            center_base = (world_base_inverse @ np.r_[center_world, 1.0])[:3]
            rotation_base = world_base_inverse[:3, :3] @ np.asarray(row["rotation_world"], dtype=np.float64)
            half = np.asarray(row["half_extents_world_axes_m"], dtype=np.float64)
            if half.shape != (3,) or np.any(half <= 0.0):
                raise ValueError(f"invalid_cuboid_half_extents:{name}")
            cuboids.append(Cuboid(
                name=name,
                pose=[*center_base.tolist(), *_quat_wxyz(rotation_base)],
                dims=(2.0 * half).tolist(),
            ))
        else:
            raise ValueError(f"unsupported_scene_shape:{row.get('shape')}")

    if not exclude_target and target_world_pose is not None:
        target = source["target_context"]["target_collision_geometries"]
        for row in target:
            center_object = np.asarray(row["center_root_m"], dtype=np.float64)
            rotation_object = np.asarray(row["rotation_root"], dtype=np.float64)
            center_world = target_world_pose[:3, :3] @ center_object + target_world_pose[:3, 3]
            rotation_world = target_world_pose[:3, :3] @ rotation_object
            center_base = (world_base_inverse @ np.r_[center_world, 1.0])[:3]
            rotation_base = world_base_inverse[:3, :3] @ rotation_world
            half = np.asarray(row["size_m"], dtype=np.float64)
            cuboids.append(Cuboid(
                name=f"released_target_geom_{int(row['geom_id'])}",
                pose=[*center_base.tolist(), *_quat_wxyz(rotation_base)],
                dims=(2.0 * half).tolist(),
            ))
    return SceneCfg(cuboid=cuboids or None, mesh=meshes or None), len(cuboids), len(meshes)


def _attached_spheres(source: dict) -> tuple[np.ndarray, dict]:
    rows = source["target_context"]["target_collision_geometries"]
    spheres = []
    proof = []
    # Split each exact MuJoCo cuboid in half along each of its local axes. The
    # eight circumspheres analytically cover the full source box, with tighter
    # coverage than one sphere around the unsplit mesh-decomposition box.
    for row in rows:
        half = np.asarray(row["size_m"], dtype=np.float64)
        center = np.asarray(row["center_root_m"], dtype=np.float64)
        rotation = np.asarray(row["rotation_root"], dtype=np.float64)
        sub_half = half / 2.0
        radius = float(np.linalg.norm(sub_half))
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                for sz in (-0.5, 0.5):
                    local_offset = np.asarray([sx, sy, sz]) * half
                    spheres.append([*(center + rotation @ local_offset).tolist(), radius])
        proof.append({
            "source_geom_id": int(row["geom_id"]),
            "partition_count": 8,
            "source_half_extents_object_m": half.tolist(),
            "partition_half_extents_m": sub_half.tolist(),
            "circumsphere_radius_m": radius,
            "analytic_full_box_coverage": bool(radius + 1e-12 >= np.linalg.norm(sub_half)),
        })
    return np.asarray(spheres, dtype=np.float32), {
        "source_collision_geom_count": len(rows),
        "attached_sphere_count": len(spheres),
        "source_box_partition_count": 8,
        "every_source_box_analytically_covered": all(row["analytic_full_box_coverage"] for row in proof),
        "components": proof,
    }


def _robot_yaml(gripper_mode: str, gripper_qpos_m=None) -> dict:
    path = Path(get_robot_configs_path()) / "franka.yml"
    cfg = load_yaml(str(path))
    cfg = copy.deepcopy(cfg)
    kinematics = cfg["robot_cfg"]["kinematics"]
    kinematics["grasp_contact_link_names"] = ["panda_leftfinger", "panda_rightfinger"]
    lock_joints = kinematics["lock_joints"]
    if gripper_mode == "open":
        lock_joints["panda_finger_joint1"] = 0.04
        lock_joints["panda_finger_joint2"] = 0.04
    elif gripper_mode == "closed":
        lock_joints["panda_finger_joint1"] = 0.0
        lock_joints["panda_finger_joint2"] = 0.0
    else:
        raise ValueError(f"unsupported_gripper_mode:{gripper_mode}")
    if gripper_qpos_m is not None:
        jaw = np.asarray(gripper_qpos_m, dtype=np.float64)
        if jaw.shape != (2,) or not np.isfinite(jaw).all() or not (
            0 <= jaw[0] <= .04 and -.04 <= jaw[1] <= 0
        ):
            raise ValueError("expected_Panda_MuJoCo_prismatic_joint_values_in_meters")
        # The URDF right joint axis is -Y with positive range; MuJoCo uses +Y
        # with negative range. Joint values therefore have opposite signs.
        lock_joints["panda_finger_joint1"] = float(jaw[0])
        lock_joints["panda_finger_joint2"] = float(-jaw[1])
    for link_name, ignored in kinematics["self_collision_ignore"].items():
        kinematics["self_collision_ignore"][link_name] = [
            item for item in ignored if item != "attached_object"
        ]
    return cfg


def _make_planner(source: dict, world_base: np.ndarray, gripper_mode: str, seed: int, *, gripper_qpos_m=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    cfg = _robot_yaml(gripper_mode, gripper_qpos_m)
    cfg["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = 320
    full_world, cuboid_count, mesh_count = _make_world(source, world_base, exclude_target=False)
    planner_cfg = MotionPlannerCfg.create(
        robot=cfg,
        collision_cache={"sphere": 0, "cuboid": cuboid_count, "mesh": mesh_count},
        self_collision_check=True,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        use_cuda_graph=False,
        random_seed=int(seed),
        max_batch_size=1,
        max_goalset=1,
        interpolation_dt=0.05,
        interpolation_buffer_size=1800,
    )
    planner = MotionPlanner(planner_cfg)
    planner.update_world(full_world)
    kin = planner.kinematics
    current_q = np.asarray(source["target_context"]["arm_qpos"], dtype=np.float64)
    current_state = JointState.from_position(
        torch.as_tensor(current_q[None, :], device=device, dtype=dtype),
        joint_names=kin.joint_names,
    )
    return planner, device, dtype, current_state, current_q, {
        "device": str(device),
        "gripper_model": gripper_mode,
        "corrected_lock_joint_values": copy.deepcopy(cfg["robot_cfg"]["kinematics"]["lock_joints"]),
        "collision_exceptions": {
            "grasp_contact_links": cfg["robot_cfg"]["kinematics"]["grasp_contact_link_names"],
            "attached_object_removed_from_self_collision_ignore": True,
            "hand_is_not_a_grasp_contact_exception": True,
        },
        "scene_cuboid_count": cuboid_count,
        "scene_mesh_count": mesh_count,
        "interpolation_buffer_size": 1800,
        "interpolation_dt_s": 0.05,
        "seed": int(seed),
    }


def _pose_from_kinematics(planner, q: np.ndarray, device, dtype) -> np.ndarray:
    q_tensor = torch.as_tensor(q[None, :], device=device, dtype=dtype)
    state = JointState.from_position(q_tensor, joint_names=planner.kinematics.joint_names)
    pose = planner.compute_kinematics(state).tool_poses.get_link_pose("panda_hand")
    position = pose.position[0].detach().float().cpu().numpy()
    quaternion = pose.quaternion[0].detach().float().cpu().numpy()
    return _matrix_from_position_rotation(
        position,
        Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix(),
    )


def _plan_stage(planner, device, dtype, current_state, transform_base_tool, *, grasp_goal: bool):
    goal = _goal_pose(transform_base_tool, device, dtype)
    ik = planner.ik_solver.solve_pose(goal, current_state=current_state, return_seeds=8)
    ik_success = _tensor(ik.success)
    ik_row = {
        "success": ik_success.reshape(-1).tolist() if ik_success is not None else None,
        "position_error": _metric(getattr(ik, "position_error", None)),
        "orientation_error": _metric(getattr(ik, "orientation_error", None)),
        "position_error_norm": _metric(getattr(ik, "position_error_norm", None)),
        "solver_status": str(getattr(ik, "status", "unavailable")),
        "solution_joint_states_rad": None,
    }
    js_solution = getattr(ik, "js_solution", None)
    if js_solution is not None:
        ik_positions = _tensor(getattr(js_solution, "position", None))
        if ik_positions is not None:
            ik_row["solution_joint_states_rad"] = ik_positions.reshape(-1, ik_positions.shape[-1])[:, :7].tolist()

    disabled = ["panda_leftfinger", "panda_rightfinger"] if grasp_goal else []
    if disabled:
        planner.disable_link_collision(disabled)
    try:
        result = planner.plan_pose(goal, current_state, max_attempts=2)
    finally:
        if disabled:
            planner.enable_link_collision(disabled)
    success_tensor = getattr(result, "success", None) if result is not None else None
    success = bool(isinstance(success_tensor, torch.Tensor) and success_tensor.any().item())
    plan_row = {
        "returned_none": result is None,
        "success": success,
        "status": str(getattr(result, "status", "no_result")) if result is not None else "no_result",
        "solve_time_s": _metric(getattr(result, "solve_time", None)) if result is not None else None,
        "total_time_s": _metric(getattr(result, "total_time", None)) if result is not None else None,
        "position_error": _metric(getattr(result, "position_error", None)) if result is not None else None,
        "orientation_error": _metric(getattr(result, "orientation_error", None)) if result is not None else None,
        "goalset_index": _metric(getattr(result, "goalset_index", None)) if result is not None else None,
        "valid_trajectory_point_count": None,
        "buffered_trajectory_point_count": None,
        "valid_joint_trajectory_rad": None,
        "valid_joint_velocity_rad_s": None,
        "valid_joint_acceleration_rad_s2": None,
    }
    path = None
    endpoint_matrix = None
    if success:
        trajectory = getattr(result, "interpolated_trajectory", None) or getattr(result, "js_solution", None)
        if trajectory is None:
            plan_row["rejection_reason"] = "CuRobo_success_without_trajectory"
            plan_row["success"] = False
        else:
            positions = _tensor(trajectory.position)
            last = _tensor(getattr(result, "interpolated_last_tstep", None))
            if positions is None or last is None:
                plan_row["rejection_reason"] = "CuRobo_trajectory_or_valid_point_count_missing"
                plan_row["success"] = False
            else:
                full = positions.reshape(-1, positions.shape[-1])
                valid_count = int(last.reshape(-1)[0])
                if valid_count < 2 or valid_count > len(full):
                    plan_row["rejection_reason"] = f"invalid_interpolated_last_tstep:{valid_count}/{len(full)}"
                    plan_row["success"] = False
                else:
                    path = full[:valid_count, :7]
                    if path.shape[1] != 7 or not np.isfinite(path).all():
                        plan_row["rejection_reason"] = "CuRobo_valid_trajectory_is_not_finite_N_by_7"
                        plan_row["success"] = False
                        path = None
                    else:
                        plan_row["valid_trajectory_point_count"] = valid_count
                        plan_row["buffered_trajectory_point_count"] = int(len(full))
                        plan_row["valid_joint_trajectory_rad"] = path.tolist()
                        plan_row["valid_joint_velocity_rad_s"] = (
                            _tensor(getattr(trajectory, "velocity", None)).reshape(-1, trajectory.velocity.shape[-1])[:valid_count, :7].tolist()
                            if isinstance(getattr(trajectory, "velocity", None), torch.Tensor) else None
                        )
                        plan_row["valid_joint_acceleration_rad_s2"] = (
                            _tensor(getattr(trajectory, "acceleration", None)).reshape(-1, trajectory.acceleration.shape[-1])[:valid_count, :7].tolist()
                            if isinstance(getattr(trajectory, "acceleration", None), torch.Tensor) else None
                        )
                        endpoint_matrix = _pose_from_kinematics(planner, path[-1], device, dtype)
    return ik_row, plan_row, path, endpoint_matrix


def _pose_error(target: np.ndarray, actual: np.ndarray) -> dict:
    return {
        "position_error_m": float(np.linalg.norm(target[:3, 3] - actual[:3, 3])),
        "orientation_error_rad": float(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude()),
    }


def _rpc(connection, request: dict) -> dict:
    connection.send(request)
    result = connection.recv()
    if not isinstance(result, dict):
        raise RuntimeError(f"Bridge_returned_non_mapping:{request.get('op')}")
    return result


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_bridge(source: dict, output_dir: Path):
    config = source["config"]
    port = _port()
    log_path = output_dir / "safety_bridge.log"
    env = os.environ.copy()
    env.update({
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LIBERO_CONFIG_PATH": "/workspace/zy_workspace/.libero_safety_config",
        "PYTHONPATH": os.pathsep.join((str(LIBERO_ROOT), str(LIBERO_ROOT / "third_party/robosuite-1.4"), str(REPO_ROOT))),
    })
    command = [
        str(LIBERO_PYTHON), str(BRIDGE_SCRIPT),
        "--benchmark", str(config["benchmark"]), "--level", str(config["level"]),
        "--task-id", str(config["task_id"]), "--init-state", str(config["init_state"]),
        "--seed", str(config["seed"]), "--camera-resolution", str(config["camera_resolution"]),
        "--port", str(port), "--demonstration-horizon", str(config["demonstration_horizon"]),
    ]
    log = log_path.open("wb", buffering=0)
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    connection = None
    deadline = time.monotonic() + 120.0
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Safety_Bridge_exited_before_ready:{process.returncode}")
            try:
                connection = Client(("127.0.0.1", port), authkey=AUTHKEY)
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.25)
        if connection is None:
            raise TimeoutError("Safety_Bridge_start_timeout_120s")
        observation = _rpc(connection, {"op": "observe", "include_snapshot_state": True})
        metadata = observation.get("scene_metadata", {})
        for field in ("benchmark", "level", "task_id", "init_state_index", "environment_seed", "model_geometry_sha256"):
            expected = source["scene_metadata"].get(field)
            if metadata.get(field) != expected:
                raise ValueError(f"live_Bridge_scene_mismatch:{field}:{metadata.get(field)}!={expected}")
        return process, log, connection, observation
    except Exception:
        if connection is not None:
            connection.close()
        process.terminate()
        log.close()
        raise


def _object_keypose_world_target(
    keypose: Keypose,
    owner_world: np.ndarray,
    world_base: np.ndarray,
):
    target = keypose_to_base_tool(
        keypose,
        transform_world_base=world_base,
        transforms_world_owner={keypose.owner_frame_id: owner_world},
    )
    return target.transform_base_tool


def _target_geometry_extents(source: dict, world_region: np.ndarray, world_object: np.ndarray):
    region_inverse = inverse_transform(world_region)
    plate_body = source["semantic_geometry_bindings"]["bindings"][
        source["config"]["goal_region_id"]
    ]["root_body_name"]
    plate_top = -np.inf
    for row in source["collision_scene_export"]["scene_collision_geometries"]:
        if row.get("body_name") != plate_body or row.get("shape") != "cuboid":
            continue
        center_region = (region_inverse @ np.r_[row["center_world_m"], 1.0])[:3]
        rotation_region_box = region_inverse[:3, :3] @ np.asarray(row["rotation_world"], dtype=np.float64)
        half = np.asarray(row["half_extents_world_axes_m"], dtype=np.float64)
        extent_z = float(np.abs(rotation_region_box[2, :]) @ half)
        plate_top = max(plate_top, float(center_region[2] + extent_z))
    if not np.isfinite(plate_top):
        raise ValueError("plate_root_has_no_supported_compiled_collision_cuboids")

    bowl_bottom = np.inf
    for row in source["target_context"]["target_collision_geometries"]:
        center_object = np.asarray(row["center_root_m"], dtype=np.float64)
        rotation_object_geom = np.asarray(row["rotation_root"], dtype=np.float64)
        half = np.asarray(row["size_m"], dtype=np.float64)
        extent_z = float(np.abs(rotation_object_geom[2, :]) @ half)
        bowl_bottom = min(bowl_bottom, float(center_object[2] - extent_z))
    if not np.isfinite(bowl_bottom):
        raise ValueError("target_object_has_no_supported_compiled_collision_cuboids")
    rotation_region_object = world_region[:3, :3].T @ world_object[:3, :3]
    # Engineering choice: keep a 15 mm object-to-plate clearance during
    # collision-free CuRobo transport; physical release/settling must establish
    # actual support and native On(..., plate) success in Stage C.
    placement_clearance_m = 0.015
    region_object = np.eye(4, dtype=np.float64)
    region_object[:3, :3] = rotation_region_object
    region_object[:3, 3] = [0.0, 0.0, plate_top - bowl_bottom + placement_clearance_m]
    return region_object, {
        "plate_top_region_z_m": plate_top,
        "bowl_bottom_object_z_m": bowl_bottom,
        "object_to_plate_clearance_m": placement_clearance_m,
        "source": "exact_compiled_cuboid_bounds_and_region_root; 15 mm is a reproduction engineering choice",
    }


def _base_keyposes(source: dict, world_base: np.ndarray, world_object: np.ndarray, world_region: np.ndarray, eef_to_hand: np.ndarray, grasp_obj_eef: np.ndarray):
    grasp_obj_hand = grasp_obj_eef @ eef_to_hand
    world_goal_object, extents = _target_geometry_extents(source, world_region, world_object)
    world_pre_object = world_goal_object.copy()
    world_pre_object[:3, 3] += world_region[:3, 2] * 0.12
    world_pre_tool = world_pre_object @ grasp_obj_hand
    world_place_tool = world_goal_object @ grasp_obj_hand
    world_retreat_tool = world_place_tool.copy()
    world_retreat_tool[:3, 3] += world_region[:3, 2] * 0.08
    region_inverse = inverse_transform(world_region)
    lift_delta = np.eye(4, dtype=np.float64)
    lift_delta[:3, 3] = [0.0, 0.0, 0.04]
    pregrasp_delta = np.eye(4, dtype=np.float64)
    pregrasp_delta[:3, 3] = [0.0, 0.0, -0.06]
    base = {
        "pre_grasp": Keypose("pre_grasp", "target_object", source["config"]["target_object_id"], "panda_hand", grasp_obj_hand @ pregrasp_delta, "open"),
        "grasp": Keypose("grasp", "target_object", source["config"]["target_object_id"], "panda_hand", grasp_obj_hand, "close"),
        "lift": Keypose("lift", "target_object", source["config"]["target_object_id"], "panda_hand", lift_delta @ grasp_obj_hand, "hold"),
        "pre_place": Keypose("pre_place", "goal_region", source["config"]["goal_region_id"], "panda_hand", region_inverse @ world_pre_tool, "hold"),
        "place": Keypose("place", "goal_region", source["config"]["goal_region_id"], "panda_hand", region_inverse @ world_place_tool, "hold"),
        "release": Keypose("release", "goal_region", source["config"]["goal_region_id"], "panda_hand", region_inverse @ world_place_tool, "open"),
        "retreat": Keypose("retreat", "goal_region", source["config"]["goal_region_id"], "panda_hand", region_inverse @ world_retreat_tool, "open"),
    }
    return base, grasp_obj_hand, world_goal_object, extents


def _open_bridge_audit(connection, path: np.ndarray, open_gripper: list[float], object_id: str, *, scope="none", eef_to_object=None, world_object=None):
    request = {
        "op": "scratch_robot_path_contact_audit",
        "arm_joint_states_rad": path.tolist(),
        "gripper_joint_states_rad": np.tile(np.asarray(open_gripper, dtype=np.float64), (len(path), 1)).tolist(),
        "object_id": object_id,
        "fingertip_contact_scope": scope,
    }
    if eef_to_object is not None:
        request["T_eef_object_m"] = eef_to_object.tolist()
    if world_object is not None:
        request["T_world_object_m"] = world_object.tolist()
    return _rpc(connection, request)


def _run_bundle(
    bundle_seed: int,
    source: dict,
    planner_open,
    planner_closed,
    bridge,
    world_base: np.ndarray,
    world_object: np.ndarray,
    world_region: np.ndarray,
    eef_to_hand: np.ndarray,
    grasp_obj_eef: np.ndarray,
    attached_sphere_np: np.ndarray,
    attachment_coverage: dict,
    planner_devices: dict,
):
    object_id = source["config"]["target_object_id"]
    open_gripper = source["target_context"]["active_open_gripper_qpos"]
    closed_gripper = [0.0, 0.0]
    variants = {
        stage: generate_variants(base, seed=20261027, bounds=DEFAULT_VARIANT_BOUNDS[stage])
        for stage, base in planner_devices["base_keyposes"].items()
    }
    selected = sample_variant_bundle(variants, seed=bundle_seed)
    selected_grasp_obj_hand = selected["grasp"].keypose.transform_owner_tool
    planner_devices["selected_grasp_obj_hand"] = selected_grasp_obj_hand
    selected_rows = {}
    targets = {}
    for stage in KEYPOSE_STAGES:
        selected_variant = selected[stage]
        keypose = selected_variant.keypose
        owner_world = world_object if keypose.owner_role == "target_object" else world_region
        target_base = _object_keypose_world_target(keypose, owner_world, world_base)
        targets[stage] = target_base
        selected_rows[stage] = {
            "variant_id": selected_variant.variant_id,
            "owner_role": keypose.owner_role,
            "owner_frame_id": keypose.owner_frame_id,
            "tool_frame_id": keypose.tool_frame_id,
            "gripper_intent": keypose.gripper_intent,
            "T_owner_tool": keypose.transform_owner_tool.tolist(),
            "translation_delta_owner_m": list(selected_variant.translation_delta_owner_m),
            "rotation_delta_owner_xyz_rad": list(selected_variant.rotation_delta_owner_xyz_rad),
            "T_base_panda_hand": target_base.tolist(),
        }
    all_variants = {
        stage: [
            {
                "variant_id": item.variant_id,
                "translation_delta_owner_m": list(item.translation_delta_owner_m),
                "rotation_delta_owner_xyz_rad": list(item.rotation_delta_owner_xyz_rad),
                "T_owner_tool": item.keypose.transform_owner_tool.tolist(),
            }
            for item in variants[stage]
        ]
        for stage in KEYPOSE_STAGES
    }
    result = {
        "bundle_seed": int(bundle_seed),
        "status": "running",
        "variant_library": all_variants,
        "selected_keyposes": selected_rows,
        "stages": [],
        "attachment_geometry_coverage": attachment_coverage,
        "grasp_contact_policy": "only exact MuJoCo left/right fingertip pad geom IDs versus target bowl geoms; CuRobo Hand and attached-object exemptions removed",
    }
    state = planner_devices["initial_state_open"]
    current_q = np.asarray(planner_devices["initial_q"], dtype=np.float64)
    grasp_relation_eef_object = None
    final_release_object_pose = None
    carried = False

    for stage_index, stage in enumerate(KEYPOSE_STAGES):
        if stage == "retreat":
            # Release detaches the object at the sampled release pose. Reinsert
            # its exact original collision boxes at that pose before retreat.
            final_release_object_pose = world_base @ targets["release"] @ inverse_transform(
                selected_grasp_obj_hand
            )
            world, cube_count, mesh_count = _make_world(
                source, world_base, exclude_target=False,
                target_world_pose=final_release_object_pose,
            )
            planner_open.update_world(world)
            planner_devices["retreat_world_shape_counts"] = {"cuboid": cube_count, "mesh": mesh_count}
            active_planner = planner_open
            active_device = planner_devices["device_open"]
            active_dtype = planner_devices["dtype_open"]
            active_state = JointState.from_position(
                torch.as_tensor(current_q[None, :], device=active_device, dtype=active_dtype),
                joint_names=active_planner.kinematics.joint_names,
            )
        elif carried or stage == "lift":
            active_planner = planner_closed
            active_device = planner_devices["device_closed"]
            active_dtype = planner_devices["dtype_closed"]
            active_state = JointState.from_position(
                torch.as_tensor(current_q[None, :], device=active_device, dtype=active_dtype),
                joint_names=active_planner.kinematics.joint_names,
            )
        else:
            active_planner = planner_open
            active_device = planner_devices["device_open"]
            active_dtype = planner_devices["dtype_open"]
            active_state = JointState.from_position(
                torch.as_tensor(current_q[None, :], device=active_device, dtype=active_dtype),
                joint_names=active_planner.kinematics.joint_names,
            )

        if stage == "lift":
            # The grasp pose supplies the exact EEF-to-object attachment offset.
            world_object_base = inverse_transform(world_base) @ world_object
            object_pose = _pose_object(world_object_base, active_device, active_dtype)
            attached = torch.as_tensor(attached_sphere_np, device=active_device, dtype=active_dtype)
            active_planner.attachment_manager.update(
                attached,
                active_state,
                link_name="attached_object",
                world_objects_pose_offset=object_pose,
            )
            world, cube_count, mesh_count = _make_world(source, world_base, exclude_target=True)
            active_planner.update_world(world)
            result["attachment_transition"] = {
                "at_stage": stage,
                "target_removed_from_static_world": True,
                "target_geom_count": len(source["target_context"]["target_collision_geometries"]),
                "attached_sphere_count": len(attached_sphere_np),
                "static_scene_cuboids": cube_count,
                "static_scene_meshes": mesh_count,
                "T_eef_object_m": grasp_relation_eef_object.tolist(),
            }
            carried = True

        grasp_goal = stage == "grasp"
        ik_row, plan_row, path, endpoint = _plan_stage(
            active_planner,
            active_device,
            active_dtype,
            active_state,
            targets[stage],
            grasp_goal=grasp_goal,
        )
        stage_row = {
            "stage": stage,
            "selected_variant_id": selected[stage].variant_id,
            "target_T_base_panda_hand": targets[stage].tolist(),
            "ik": ik_row,
            "curobo": plan_row,
            "collision_check": None,
            "rejection_reason": None,
        }
        if path is None or not plan_row["success"]:
            stage_row["rejection_reason"] = plan_row.get("rejection_reason") or "CuRobo_did_not_return_a_successful_path"
            result["stages"].append(stage_row)
            result["status"] = "blocked"
            result["rejection_reason"] = f"{stage}: {stage_row['rejection_reason']}"
            break

        endpoint_error = _pose_error(targets[stage], endpoint)
        stage_row["endpoint_fk_check"] = endpoint_error
        if endpoint_error["position_error_m"] > 0.002 or endpoint_error["orientation_error_rad"] > np.deg2rad(0.10):
            stage_row["rejection_reason"] = "CuRobo_endpoint_FK_exceeds_2mm_or_0.10deg"
            result["stages"].append(stage_row)
            result["status"] = "blocked"
            result["rejection_reason"] = f"{stage}: {stage_row['rejection_reason']}"
            break

        if stage == "grasp":
            planned_world_hand = world_base @ endpoint
            planned_world_eef = planned_world_hand @ inverse_transform(eef_to_hand)
            grasp_relation_eef_object = inverse_transform(planned_world_eef) @ world_object
            actual_obj_hand = inverse_transform(world_object) @ planned_world_hand
            grasp_pose_difference = _pose_error(
                selected["grasp"].keypose.transform_owner_tool,
                actual_obj_hand,
            )
            stage_row["object_to_tool_grasp_relation_check"] = grasp_pose_difference
            if grasp_pose_difference["position_error_m"] > 0.002 or grasp_pose_difference["orientation_error_rad"] > np.deg2rad(0.10):
                stage_row["rejection_reason"] = "planned_grasp_relation_differs_from_selected_object_keypose"
                result["stages"].append(stage_row)
                result["status"] = "blocked"
                result["rejection_reason"] = f"grasp: {stage_row['rejection_reason']}"
                break

        scope = "none"
        audit_eef_object = None
        fixed_world_object = None
        audit_gripper = open_gripper
        if stage == "grasp":
            scope = "final_state"
        elif carried and stage != "retreat":
            scope = "all_states"
            audit_eef_object = grasp_relation_eef_object
            audit_gripper = closed_gripper
        elif stage == "retreat":
            fixed_world_object = final_release_object_pose
        audit = _open_bridge_audit(
            bridge,
            path,
            audit_gripper,
            object_id,
            scope=scope,
            eef_to_object=audit_eef_object,
            world_object=fixed_world_object,
        )
        stage_row["collision_check"] = {
            "method": "CuRobo full-scene collision planner plus exact MuJoCo scratch contact audit over every valid interpolated point",
            "curobo_success": plan_row["success"],
            "mujoco_status": audit.get("status"),
            "physical_state_unchanged": audit.get("physical_state_unchanged"),
            "state_count": audit.get("state_count"),
            "illegal_contact_count": audit.get("illegal_contact_count"),
            "contact_policy": audit.get("allowed_contact_policy"),
            "verified_fingertip_pad_geom_ids": audit.get("verified_fingertip_pad_geom_ids"),
            "target_geom_ids": audit.get("target_geom_ids"),
            "contacts_by_state": [row.get("contacts", []) for row in audit.get("states", [])],
        }
        if audit.get("status") != "complete" or audit.get("physical_state_unchanged") is not True:
            stage_row["rejection_reason"] = f"MuJoCo_scratch_contact_audit_incomplete:{audit.get('error')}"
            result["stages"].append(stage_row)
            result["status"] = "blocked"
            result["rejection_reason"] = f"{stage}: {stage_row['rejection_reason']}"
            break
        if audit.get("illegal_contact_count") != 0:
            stage_row["rejection_reason"] = f"MuJoCo_illegal_robot_contact_count:{audit.get('illegal_contact_count')}"
            result["stages"].append(stage_row)
            result["status"] = "blocked"
            result["rejection_reason"] = f"{stage}: {stage_row['rejection_reason']}"
            break

        if stage == "grasp":
            # Check the real closed-jaw geometry at the exact planned pose.
            closed_audit = _open_bridge_audit(
                bridge,
                path[-1:,:],
                closed_gripper,
                object_id,
                scope="final_state",
                eef_to_object=grasp_relation_eef_object,
            )
            stage_row["closed_gripper_contact_check"] = {
                "status": closed_audit.get("status"),
                "physical_state_unchanged": closed_audit.get("physical_state_unchanged"),
                "contact_count": closed_audit.get("states", [{}])[0].get("contact_count"),
                "illegal_contact_count": closed_audit.get("illegal_contact_count"),
                "contacts": closed_audit.get("states", [{}])[0].get("contacts", []),
                "contact_policy": closed_audit.get("allowed_contact_policy"),
            }
            if (
                closed_audit.get("status") != "complete"
                or closed_audit.get("physical_state_unchanged") is not True
                or closed_audit.get("illegal_contact_count") != 0
            ):
                stage_row["rejection_reason"] = "closed_gripper_exact_contact_check_failed"
                result["stages"].append(stage_row)
                result["status"] = "blocked"
                result["rejection_reason"] = f"grasp: {stage_row['rejection_reason']}"
                break
            allowed_pad_contacts = [
                contact for contact in closed_audit.get("states", [{}])[0].get("contacts", [])
                if contact.get("allowed") is True
            ]
            contacted_pad_ids = {
                geom_id
                for contact in allowed_pad_contacts
                for geom_id in contact.get("geom_ids", [])
                if geom_id in closed_audit.get("verified_fingertip_pad_geom_ids", [])
            }
            verified_pad_ids = set(closed_audit.get("verified_fingertip_pad_geom_ids", []))
            if not verified_pad_ids or contacted_pad_ids != verified_pad_ids:
                stage_row["rejection_reason"] = "closed_gripper_lacks_bilateral_verified_fingertip_target_contact"
                result["stages"].append(stage_row)
                result["status"] = "blocked"
                result["rejection_reason"] = f"grasp: {stage_row['rejection_reason']}"
                break

        result["stages"].append(stage_row)
        current_q = path[-1].copy()
        state = JointState.from_position(
            torch.as_tensor(current_q[None, :], device=active_device, dtype=active_dtype),
            joint_names=active_planner.kinematics.joint_names,
        )
        if stage == "release":
            result["release_object_pose_world_m"] = (
                world_base @ targets["release"] @ inverse_transform(selected_grasp_obj_hand)
            ).tolist()

    if result["status"] == "running":
        result["status"] = "complete_route"
        result["all_seven_stages_planned_and_audited"] = len(result["stages"]) == len(KEYPOSE_STAGES)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-source", type=Path, default=SCENE_SOURCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "stage_b_route_report.json"
    report = {
        "schema": "gs_opd_libero_safety_stage_b_route_report.v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "scope": "one OA/L0T0 task, at most two deterministic Keypose bundles",
        "physical_actions_sent": 0,
        "training": False,
        "source": str(args.scene_source.resolve()),
        "source_sha256": None,
        "route_bundles": [],
    }
    process = log_handle = connection = None
    try:
        source_bytes = args.scene_source.read_bytes()
        import hashlib
        report["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        source = json.loads(gzip.decompress(source_bytes))
        scene = source["scene_metadata"]
        if (scene.get("benchmark"), scene.get("level"), scene.get("task_id")) != ("obstacle_avoidance", 0, 0):
            raise ValueError("route_script_only_accepts_obstacle_avoidance_L0T0")
        if scene.get("horizon_mode") != "standard_benchmark" or scene.get("bridge_horizon") != 1000:
            raise ValueError("Stage_B_requires_the_standard_1000_step_initial_scene")
        if source.get("read_only_state_equal_after_queries") is not True:
            raise ValueError("source_scene_was_not_captured_read_only")
        if source["collision_scene_export"].get("unsupported_geometries"):
            raise ValueError("full_collision_scene_contains_unsupported_geometries")

        world_base = _matrix(source["planner_frames"]["world_base_transform_m"])
        world_object = _matrix_from_position_rotation(
            source["target_context"]["target_root_pose_world"]["position_m"],
            source["target_context"]["target_root_pose_world"]["rotation"],
        )
        world_region = _matrix(source["planner_frames"]["world_goal_region_transform_m"])
        eef_to_hand = _matrix(source["planner_frames"]["eef_to_panda_hand_transform"])
        grasp_report = json.loads(GRASP_EXPERIENCE_REPORT.read_text(encoding="utf-8"))
        if grasp_report.get("status") != "task_success_candidate_created" or grasp_report.get("scope", {}).get("task_id") != 0:
            raise ValueError("prior_grasp_keypose_evidence_is_not_an_OA_L0T0_task_candidate")
        source_candidate = next(
            row for row in grasp_report["candidates"]
            if row.get("candidate_id") == "geom112_jaw0_approach2_+1_flip+1"
        )
        grasp_obj_eef = _matrix(source_candidate["candidate_geometry"]["object_to_eef"])
        report["keypose_provenance"] = {
            "grasp_T_object_eef_source": str(GRASP_EXPERIENCE_REPORT),
            "grasp_T_object_eef": grasp_obj_eef.tolist(),
            "source_scope": grasp_report.get("scope"),
            "source_candidate_id": source_candidate.get("candidate_id"),
            "source_actions_replayed": False,
            "source_episode_not_a_complete_task_success": True,
            "other_keyposes": "geometry-derived or engineering-defined in this reproduction; no source actions copied",
        }
        base_keyposes, base_obj_hand, world_goal_object, object_extents = _base_keyposes(
            source, world_base, world_object, world_region, eef_to_hand, grasp_obj_eef
        )
        attached_sphere_np, attachment_coverage = _attached_spheres(source)
        if not attachment_coverage["every_source_box_analytically_covered"]:
            raise ValueError("attached_payload_lacks_analytic_collision_geometry_coverage")
        report["placement_geometry"] = object_extents
        report["attached_payload_geometry_coverage"] = attachment_coverage
        report["keypose_engineering_choices"] = {
            "pre_grasp_offset_object_z_m": -0.06,
            "lift_offset_object_z_m": 0.04,
            "pre_place_clearance_above_plate_m": 0.12,
            "retreat_clearance_above_release_m": 0.08,
            "variant_bounds": {
                stage: {
                    "translation_half_range_m": list(DEFAULT_VARIANT_BOUNDS[stage].translation_half_range_m),
                    "rotation_half_range_rad": list(DEFAULT_VARIANT_BOUNDS[stage].rotation_half_range_rad),
                }
                for stage in KEYPOSE_STAGES
            },
        }

        process, log_handle, connection, live_observation = _start_bridge(source, output_dir)
        live_context = _rpc(connection, {
            "op": "task_success_first_context",
            "expected_instruction": source["config"]["instruction"],
            "target_predicate": source["config"]["target_predicate"],
        })
        if live_context.get("status") != "complete" or live_context.get("read_only") is not True:
            raise ValueError(f"live_target_context_is_unavailable:{live_context.get('error')}")
        if live_context.get("model_geometry_sha256") != scene.get("model_geometry_sha256"):
            raise ValueError("live_model_geometry_does_not_match_scene_probe")
        report["live_bridge"] = {
            "scene_metadata": live_observation.get("scene_metadata"),
            "action_count_before": live_observation.get("scene_metadata", {}).get("action_count", 0),
            "native_initial_task_success": live_observation.get("task_success"),
            "target_context_complete": True,
            "model_geometry_sha256": live_context.get("model_geometry_sha256"),
        }

        kin_cfg = KinematicsCfg.from_robot_yaml_file("franka.yml")
        kin = Kinematics(kin_cfg)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        initial_q = np.asarray(source["target_context"]["arm_qpos"], dtype=np.float64)
        initial_state = JointState.from_position(
            torch.as_tensor(initial_q[None, :], device=device, dtype=dtype),
            joint_names=kin.joint_names,
        )
        fk = kin.compute_kinematics(initial_state).tool_poses.get_link_pose("panda_hand")
        curobo_hand_base = _matrix_from_position_rotation(
            fk.position[0].detach().float().cpu().numpy(),
            Rotation.from_quat(fk.quaternion[0].detach().float().cpu().numpy()[[1, 2, 3, 0]]).as_matrix(),
        )
        live_hand_base = inverse_transform(world_base) @ _matrix(source["planner_frames"]["world_panda_hand_transform_m"])
        calibration_error = _pose_error(live_hand_base, curobo_hand_base)
        report["base_tool_registration_check"] = {
            **calibration_error,
            "source": "MuJoCo robot0_base + robot0_right_hand and gripper0_eef_site calibration from scratch FK",
            "planner_tool": "panda_hand",
            "passes": calibration_error["position_error_m"] <= 0.002 and calibration_error["orientation_error_rad"] <= np.deg2rad(0.10),
        }
        if not report["base_tool_registration_check"]["passes"]:
            raise ValueError("CuRobo_panda_hand_FK_does_not_match_verified_MuJoCo_base_tool")

        seeds = [20261027, 20261028]
        for bundle_index, bundle_seed in enumerate(seeds):
            if bundle_index > 0 and report["route_bundles"] and report["route_bundles"][-1].get("status") == "complete_route":
                break
            planner_open, device_open, dtype_open, state_open, current_q, open_info = _make_planner(
                source, world_base, "open", seed=bundle_seed,
            )
            planner_closed, device_closed, dtype_closed, _, _, closed_info = _make_planner(
                source, world_base, "closed", seed=bundle_seed + 1000,
            )
            devices = {
                "base_keyposes": base_keyposes,
                "initial_state_open": state_open,
                "initial_q": current_q,
                "device_open": device_open,
                "dtype_open": dtype_open,
                "device_closed": device_closed,
                "dtype_closed": dtype_closed,
                "selected_grasp_obj_hand": base_obj_hand,
            }
            bundle = _run_bundle(
                bundle_seed,
                source,
                planner_open,
                planner_closed,
                connection,
                world_base,
                world_object,
                world_region,
                eef_to_hand,
                grasp_obj_eef,
                attached_sphere_np,
                attachment_coverage,
                devices,
            )
            bundle["planner_configuration"] = {
                "open_gripper": open_info,
                "closed_gripper": closed_info,
                "payload_sphere_count": len(attached_sphere_np),
            }
            report["route_bundles"].append(bundle)
            report["route_bundles_attempted"] = len(report["route_bundles"])
            report["status"] = bundle["status"]
            report_path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")

        report["complete_route_found"] = any(row.get("status") == "complete_route" for row in report["route_bundles"])
        report["c_stage_authorized_by_route_gate"] = report["complete_route_found"]
        report["status"] = "pass_route_available" if report["complete_route_found"] else "blocked_no_complete_route"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        if connection is not None:
            try:
                final_observation = _rpc(connection, {"op": "observe", "include_snapshot_state": True})
                report["bridge_final_state"] = {
                    "action_count": final_observation.get("scene_metadata", {}).get("action_count", 0),
                    "timestep": final_observation.get("timestep"),
                    "task_success": final_observation.get("task_success"),
                    "no_actions_sent": final_observation.get("scene_metadata", {}).get("action_count", 0) == 0,
                }
            except Exception as final_exc:
                report["bridge_final_state"] = {"status": "unavailable", "error": f"{type(final_exc).__name__}: {final_exc}"}
            connection.close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if log_handle is not None:
            log_handle.close()
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["route_report_path"] = str(report_path)
        report_path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report.get("status"),
        "route_bundles_attempted": len(report.get("route_bundles", [])),
        "complete_route_found": report.get("complete_route_found", False),
        "output": str(report_path),
        "error": report.get("error"),
    }, indent=2, allow_nan=False))
    return 0 if report.get("status") == "pass_route_available" else 1


if __name__ == "__main__":
    raise SystemExit(main())
