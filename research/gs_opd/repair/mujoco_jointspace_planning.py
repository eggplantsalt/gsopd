# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""MuJoCo adapter for the reusable joint-space RRT-Connect planner."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from research.gs_opd.repair.jointspace_rrt import (
    ConfigurationCheck,
    JointSpacePlannerConfig,
    JointSpaceRRTConnect,
    KinematicSample,
)
from research.gs_opd.repair.trajectory_tracking import (
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_vector,
)


def _unit(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name}_must_be_a_finite_vector_of_length_{size}")
    if size == 4:
        norm = float(np.linalg.norm(array))
        if norm < 1e-12:
            raise ValueError(f"{name}_has_zero_rotation_norm")
        array = array / norm
    return array


def _minimum_distance(mujoco, model, data, robot_geom_ids, object_geom_ids, excluded_pairs=None):
    excluded_pairs = excluded_pairs or set()
    pairs = []
    for robot_id in robot_geom_ids:
        for object_id in object_geom_ids:
            if tuple(sorted((int(robot_id), int(object_id)))) in excluded_pairs:
                continue
            lower_bound = float(
                np.linalg.norm(data.geom_xpos[robot_id] - data.geom_xpos[object_id])
                - model.geom_rbound[robot_id]
                - model.geom_rbound[object_id]
            )
            pairs.append((lower_bound, int(robot_id), int(object_id)))
    if not pairs:
        return 10.0
    pairs.sort(key=lambda item: item[0])
    best = None
    attempted = 0
    for lower_bound, robot_id, object_id in pairs:
        if best is not None and lower_bound >= best:
            break
        fromto = np.zeros(6, dtype=np.float64)
        value = float(mujoco.mj_geomDistance(model, data, robot_id, object_id, 10.0, fromto))
        attempted += 1
        if not np.isfinite(value):
            return None
        if value < 10.0 and (best is None or value < best):
            best = value
    if best is None:
        return 10.0
    if any(bound < best for bound, _, _ in pairs[attempted:]):
        return None
    return best


def _exact_pair_distance(mujoco, model, data, geom_a, geom_b):
    segment = np.zeros(6, dtype=np.float64)
    value = float(mujoco.mj_geomDistance(model, data, int(geom_a), int(geom_b), 10.0, segment))
    return value if np.isfinite(value) and np.isfinite(segment).all() else None


def carried_object_world_transform(eef_world_transform, eef_to_object_transform):
    """Compose the measured grasp transform at a candidate EEF pose."""
    eef = np.asarray(eef_world_transform, dtype=np.float64)
    relative = np.asarray(eef_to_object_transform, dtype=np.float64)
    if (
        eef.shape != (4, 4)
        or relative.shape != (4, 4)
        or not np.isfinite(eef).all()
        or not np.isfinite(relative).all()
        or not np.allclose(eef[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        or not np.allclose(relative[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        or not np.allclose(eef[:3, :3].T @ eef[:3, :3], np.eye(3), atol=1e-5)
        or not np.allclose(relative[:3, :3].T @ relative[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(eef[:3, :3]), 1.0, atol=1e-5)
        or not np.isclose(np.linalg.det(relative[:3, :3]), 1.0, atol=1e-5)
    ):
        raise ValueError("carried_object_transforms_must_be_finite_rigid_4x4_matrices")
    return eef @ relative


def apply_carried_object_pose_to_scratch(mujoco, data, eef_site_id, carried):
    """Move a free-root held object with the candidate EEF pose in scratch data."""
    relative = np.asarray(carried["eef_to_object_transform"], dtype=np.float64)
    world_eef = np.eye(4, dtype=np.float64)
    world_eef[:3, :3] = np.asarray(data.site_xmat[eef_site_id]).reshape(3, 3)
    world_eef[:3, 3] = data.site_xpos[eef_site_id]
    world_object = carried_object_world_transform(world_eef, relative)
    address = int(carried["free_joint_qpos_adr"])
    data.qpos[address:address + 3] = world_object[:3, 3]
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, world_object[:3, :3].reshape(9))
    data.qpos[address + 3:address + 7] = quaternion / np.linalg.norm(quaternion)
    mujoco.mj_forward(data.model, data)
    return world_object


def plan_current_scene(
    environment,
    goal_pose_world: dict[str, Any],
    normalized_safety_policy: dict[str, Any],
    *,
    start_state_sha256: str,
    expected_instruction: str,
    actual_instruction: str,
    ik_seed_qpos: list[list[float]] | None = None,
    planner_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a world EEF goal, then plan from the active MuJoCo 7D qpos.

    Planning and every geometry query run on independent MjData. This function
    never steps the live environment or edits its qpos.
    """
    import hashlib
    import mujoco
    from scipy.optimize import least_squares

    if expected_instruction != actual_instruction:
        raise ValueError("task_instruction_does_not_match_the_active_BDDL_task")
    if not start_state_sha256 or len(start_state_sha256) != 64:
        raise ValueError("start_state_sha256_is_missing_or_malformed")
    if not isinstance(normalized_safety_policy, dict):
        raise ValueError("verified_safety_geometry_policy_is_required")
    sim = environment.sim
    robot = environment.robots[0]
    model = sim.model._model
    live_data = sim.data._data
    arm_joint_ids = np.asarray(robot._ref_joint_indexes, dtype=np.int64)
    if arm_joint_ids.shape != (7,):
        raise ValueError("active_robot_does_not_expose_seven_arm_joints")
    qpos_addresses = np.asarray(model.jnt_qposadr[arm_joint_ids], dtype=np.int64)
    dof_addresses = np.asarray(model.jnt_dofadr[arm_joint_ids], dtype=np.int64)
    eef_site_id = int(robot.eef_site_id)
    q_start = np.asarray(live_data.qpos[qpos_addresses], dtype=np.float64).copy()
    integration_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration_state = np.zeros(mujoco.mj_stateSize(model, integration_spec), dtype=np.float64)
    mujoco.mj_getState(model, live_data, integration_state, integration_spec)
    if hashlib.sha256(integration_state.tobytes()).hexdigest() != start_state_sha256:
        raise ValueError("live_qpos_state_changed_before_rrt_planning")

    goal = {
        "position_world_m": _unit(goal_pose_world.get("position_world_m"), 3, "goal_position"),
        "quaternion_wxyz": _unit(goal_pose_world.get("quaternion_wxyz"), 4, "goal_quaternion"),
    }
    scratch = mujoco.MjData(model)
    mujoco.mj_setState(model, scratch, integration_state, integration_spec)
    mujoco.mj_forward(model, scratch)
    robot_geoms = np.asarray(sorted(normalized_safety_policy["robot_geom_ids"]), dtype=np.int64)
    robot_bodies = set(normalized_safety_policy["robot_body_ids"])
    forbidden = normalized_safety_policy["forbidden_objects"]
    checked_objects = normalized_safety_policy.get("checked_objects", forbidden)
    allowed_pair_ids = set(normalized_safety_policy["allowed_contact_pair_ids"])
    robot_body_ids = set(normalized_safety_policy["robot_body_ids"])
    forbidden_body_ids = set(normalized_safety_policy["forbidden_scene_body_ids"])
    allowed_target_body_ids = set(normalized_safety_policy["allowed_target_body_ids"])
    other_scene_body_ids = set(normalized_safety_policy["other_scene_body_ids"])
    forbidden_object_pairs = normalized_safety_policy["forbidden_object_pairs"]
    lower = np.asarray(model.jnt_range[arm_joint_ids, 0], dtype=np.float64).copy()
    upper = np.asarray(model.jnt_range[arm_joint_ids, 1], dtype=np.float64).copy()
    limited = np.asarray(model.jnt_limited[arm_joint_ids], dtype=bool)
    if not limited.all() or np.any(lower >= upper):
        raise ValueError("active_arm_joint_limits_are_incomplete")
    if np.any(q_start < lower - 1e-8) or np.any(q_start > upper + 1e-8):
        raise ValueError("current_real_qpos_is_outside_controller_joint_limits")

    def set_scratch_qpos(qpos):
        scratch.qpos[qpos_addresses] = np.asarray(qpos, dtype=np.float64)
        mujoco.mj_forward(model, scratch)
        carried = normalized_safety_policy.get("carried_object")
        if carried is not None:
            apply_carried_object_pose_to_scratch(mujoco, scratch, eef_site_id, carried)

    def pose_at(qpos):
        set_scratch_qpos(qpos)
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, np.asarray(scratch.site_xmat[eef_site_id]).reshape(9))
        return scratch.site_xpos[eef_site_id].copy(), quaternion / np.linalg.norm(quaternion)

    goal_position = goal["position_world_m"]
    goal_quaternion = goal["quaternion_wxyz"]
    seeds = [q_start]
    if ik_seed_qpos is not None:
        for index, seed in enumerate(ik_seed_qpos):
            vector = _unit(seed, 7, f"ik_seed_{index}")
            if np.all(vector >= lower) and np.all(vector <= upper):
                seeds.append(vector)
    ik_candidates = []
    for seed_index, seed in enumerate(seeds):
        def residual(candidate):
            position, quaternion = pose_at(candidate)
            rotation = quaternion_to_rotation_vector(
                quaternion_multiply(goal_quaternion, quaternion_conjugate(quaternion))
            )
            return np.r_[goal_position - position, 0.15 * rotation, 1e-4 * (candidate - seed)]

        solution = least_squares(
            residual,
            np.clip(seed, lower + 1e-7, upper - 1e-7),
            bounds=(lower + 1e-7, upper - 1e-7),
            x_scale="jac",
            max_nfev=1000,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        q_goal_candidate = np.asarray(solution.x, dtype=np.float64)
        position, quaternion = pose_at(q_goal_candidate)
        position_error = float(np.linalg.norm(position - goal_position))
        rotation_error = float(
            np.linalg.norm(
                quaternion_to_rotation_vector(
                    quaternion_multiply(goal_quaternion, quaternion_conjugate(quaternion))
                )
            )
        )
        ik_candidates.append({
            "seed_index": seed_index,
            "qpos": q_goal_candidate,
            "position_error_m": position_error,
            "orientation_error_rad": rotation_error,
            "joint_distance_rad": float(np.linalg.norm(q_goal_candidate - q_start)),
            "solver_status": int(solution.status),
            "solver_evaluations": int(solution.nfev),
        })
    ik_valid = [
        item for item in ik_candidates
        if item["position_error_m"] <= 0.003 and item["orientation_error_rad"] <= 0.03
    ]
    if not ik_valid:
        raise ValueError("task_goal_has_no_bounded_current_scene_inverse_kinematics_solution")
    ik_valid.sort(key=lambda item: item["joint_distance_rad"])
    q_goal = ik_valid[0]["qpos"]

    def kinematics(qpos):
        position, quaternion = pose_at(qpos)
        centers = np.asarray(scratch.geom_xpos[robot_geoms], dtype=np.float64).copy()
        return KinematicSample(position, quaternion, centers)

    def check_configuration(qpos):
        set_scratch_qpos(qpos)
        clearances = {}
        forbidden_contacts = []
        for item in checked_objects:
            group_allowed_pairs = {
                pair for pair in allowed_pair_ids
                if next(geom_id for geom_id in pair if geom_id not in set(robot_geoms))
                in item["geom_ids"]
            } if item["policy_class"] == "allowed_target_objects" else set()
            distance = _minimum_distance(
                mujoco, model, scratch, robot_geoms, item["geom_ids"], group_allowed_pairs
            )
            if distance is None or not np.isfinite(distance):
                return ConfigurationCheck("incomplete", None, {}, diagnostics={"object_id": item["object_id"]})
            clearances[item["object_id"]] = distance
            if distance < item["minimum_clearance_m"]:
                forbidden_contacts.append({
                    "object_id": item["object_id"],
                    "distance_m": float(distance),
                    "minimum_required_m": item["minimum_clearance_m"],
                })
            for pair in group_allowed_pairs:
                pair_distance = _exact_pair_distance(mujoco, model, scratch, *pair)
                minimum_allowed = normalized_safety_policy[
                    "allowed_pair_minimum_clearance_m"
                ].get(pair)
                if pair_distance is None or minimum_allowed is None:
                    return ConfigurationCheck(
                        "incomplete", None, {}, diagnostics={"allowed_pair": list(pair)}
                    )
                if pair_distance < minimum_allowed:
                    forbidden_contacts.append({
                        "allowed_pair_penetration_exceeded": True,
                        "geom_ids": list(pair),
                        "distance_m": pair_distance,
                        "minimum_allowed_distance_m": minimum_allowed,
                    })
        carried = normalized_safety_policy.get("carried_object")
        carried_forbidden_clearances = []
        if carried is not None:
            carried_geoms = np.asarray(sorted(carried["geom_ids"]), dtype=np.int64)
            for item in checked_objects:
                if item["object_id"] == carried["object_id"]:
                    continue
                allowed_pairs = {
                    pair for pair in carried["allowed_contact_pair_ids"]
                    if next(geom_id for geom_id in pair if geom_id in carried["geom_ids"])
                    in carried["geom_ids"]
                    and next(geom_id for geom_id in pair if geom_id not in carried["geom_ids"])
                    in item["geom_ids"]
                }
                distance = _minimum_distance(
                    mujoco, model, scratch, carried_geoms, item["geom_ids"], allowed_pairs
                )
                if distance is None or not np.isfinite(distance):
                    return ConfigurationCheck(
                        "incomplete", None, {}, diagnostics={"carried_object_id": carried["object_id"], "scene_object_id": item["object_id"]}
                    )
                key = f"carried:{carried['object_id']}->{item['object_id']}"
                clearances[key] = distance
                required = (
                    item["minimum_clearance_m"]
                    if item["policy_class"] == "forbidden_objects"
                    else 0.0
                )
                if item["policy_class"] == "forbidden_objects":
                    carried_forbidden_clearances.append(distance)
                if distance < required:
                    forbidden_contacts.append({
                        "carried_object_clearance_requirement_violated": True,
                        "carried_object_id": carried["object_id"],
                        "object_id": item["object_id"],
                        "distance_m": distance,
                        "minimum_required_m": required,
                    })
                for pair in allowed_pairs:
                    pair_distance = _exact_pair_distance(mujoco, model, scratch, *pair)
                    minimum_allowed = carried["allowed_pair_minimum_clearance_m"].get(pair)
                    if pair_distance is None or minimum_allowed is None:
                        return ConfigurationCheck(
                            "incomplete", None, {}, diagnostics={"carried_allowed_pair": list(pair)}
                        )
                    if pair_distance < minimum_allowed:
                        forbidden_contacts.append({
                            "carried_object_allowed_pair_penetration_exceeded": True,
                            "geom_ids": list(pair),
                            "distance_m": pair_distance,
                            "minimum_allowed_distance_m": minimum_allowed,
                        })
        minimum_forbidden = min(
            [clearances[item["object_id"]] for item in forbidden]
            + carried_forbidden_clearances,
            default=10.0,
        )
        self_contacts = []
        for contact_index in range(int(scratch.ncon)):
            contact = scratch.contact[contact_index]
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            body_a, body_b = int(model.geom_bodyid[geom_a]), int(model.geom_bodyid[geom_b])
            if body_a in robot_bodies and body_b in robot_bodies:
                self_contacts.append({"geom_ids": [geom_a, geom_b], "distance_m": float(contact.dist)})
            elif body_a in robot_bodies or body_b in robot_bodies:
                scene_body = body_b if body_a in robot_body_ids else body_a
                if scene_body in forbidden_body_ids:
                    forbidden_contacts.append({"geom_ids": [geom_a, geom_b], "body_ids": [body_a, body_b], "distance_m": float(contact.dist)})
                elif scene_body in allowed_target_body_ids and tuple(sorted((geom_a, geom_b))) not in allowed_pair_ids:
                    forbidden_contacts.append({"geom_ids": [geom_a, geom_b], "body_ids": [body_a, body_b], "distance_m": float(contact.dist)})
                elif scene_body in other_scene_body_ids:
                    forbidden_contacts.append({
                        "unallowed_scene_contact": True,
                        "geom_ids": [geom_a, geom_b],
                        "body_ids": [body_a, body_b],
                        "distance_m": float(contact.dist),
                    })
                elif scene_body not in allowed_target_body_ids | forbidden_body_ids:
                    forbidden_contacts.append({"unresolved_scene_body_id": scene_body, "geom_ids": [geom_a, geom_b]})
            else:
                bodies = {body_a, body_b}
                if any(
                    bodies & pair["body_ids_a"] and bodies & pair["body_ids_b"]
                    for pair in forbidden_object_pairs
                ):
                    forbidden_contacts.append({
                        "forbidden_object_pair": [body_a, body_b],
                        "geom_ids": [geom_a, geom_b],
                        "distance_m": float(contact.dist),
                    })
        return ConfigurationCheck(
            "complete",
            minimum_forbidden,
            clearances,
            forbidden_contacts,
            self_contacts,
            {"qpos_rad": np.asarray(qpos).tolist()},
        )

    planner_values = dict(planner_config or {})
    configured_margin = float(
        planner_values.pop("minimum_clearance_m", normalized_safety_policy["minimum_clearance_m"])
    )
    if configured_margin < float(normalized_safety_policy["minimum_clearance_m"]):
        raise ValueError("planner_clearance_cannot_weaken_the_task_hard_margin")
    planner_cfg = JointSpacePlannerConfig(
        **planner_values,
        minimum_clearance_m=configured_margin,
    )
    planner = JointSpaceRRTConnect(lower, upper, check_configuration, kinematics, config=planner_cfg)
    result = planner.plan(q_start, q_goal)
    live_state_after = np.zeros_like(integration_state)
    mujoco.mj_getState(model, live_data, live_state_after, integration_spec)
    unchanged = np.array_equal(integration_state, live_state_after)
    if not unchanged:
        raise RuntimeError("live_environment_mutated_during_scratch_planning")
    if result.status != "offline_feasible":
        return {
            "status": result.status,
            "failure_reason": result.failure_reason,
            "planning_seconds": result.planning_seconds,
            "config_checks": result.config_checks,
            "iterations": result.iterations,
            "ik_candidates": [
                {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in item.items()}
                for item in ik_candidates
            ],
            "route_candidates": result.candidates,
            "live_state_unchanged": unchanged,
        }
    eef_path = []
    for qpos in result.qpos_path:
        position, quaternion = pose_at(qpos)
        eef_path.append(np.r_[position, quaternion].tolist())
    return {
        "status": result.status,
        "planner_source": result.planner_source,
        "planning_seconds": result.planning_seconds,
        "config_checks": result.config_checks,
        "iterations": result.iterations,
        "selected_route_index": result.selected_route_index,
        "minimum_forbidden_clearance_m": result.minimum_forbidden_clearance_m,
        "qpos_path": result.qpos_path.tolist(),
        "eef_reference_poses_world": eef_path,
        "ik_goal_qpos": q_goal.tolist(),
        "ik_candidates": [
            {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in item.items()}
            for item in ik_candidates
        ],
        "route_candidates": result.candidates,
        "live_state_unchanged": unchanged,
    }
