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


def validate_finger_only_contact_geoms(
    left_finger_geom_ids: list[int],
    right_finger_geom_ids: list[int],
    allowed_gripper_geom_ids: list[int],
) -> set[int]:
    """Require the contact exception to equal the two physical finger sides."""
    left = {int(value) for value in left_finger_geom_ids}
    right = {int(value) for value in right_finger_geom_ids}
    allowed = {int(value) for value in allowed_gripper_geom_ids}
    if not left or not right or left & right:
        raise ValueError("both_disjoint_verified_gripper_finger_sides_are_required")
    if allowed != left | right:
        raise ValueError("grasp_contact_exception_must_contain_only_the_verified_finger_geoms")
    return allowed


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
    ik_selection_seed_qpos: list[float] | None = None,
    planner_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a world EEF goal, then plan from the active MuJoCo 7D qpos.

    Planning and every geometry query run on independent MjData. This function
    never steps the live environment or edits its qpos.
    """
    import hashlib
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

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
    selection_seed = q_start
    if ik_selection_seed_qpos is not None:
        selection_seed = _unit(ik_selection_seed_qpos, 7, "ik_selection_seed_qpos")
        if np.any(selection_seed < lower) or np.any(selection_seed > upper):
            raise ValueError("ik_selection_seed_qpos_is_outside_joint_limits")
    for item in ik_valid:
        item["selection_seed_distance_rad"] = float(
            np.linalg.norm(item["qpos"] - selection_seed)
        )
    ik_valid.sort(key=lambda item: (item["selection_seed_distance_rad"], item["joint_distance_rad"]))
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


def screen_grasp_candidates(
    environment,
    candidates: list[dict[str, Any]],
    normalized_safety_policy: dict[str, Any],
    *,
    target_object_id: str,
    target_root_body_id: int,
    left_finger_geom_ids: list[int],
    right_finger_geom_ids: list[int],
    gripper_geom_ids: list[int],
    closed_gripper_qpos: list[float],
    predicted_stable_object_pose_world: list[list[float]],
    start_state_sha256: str,
    expected_instruction: str,
    actual_instruction: str,
    clearance_m: float = 0.010,
) -> dict[str, Any]:
    """Offline scratch-only grasp, lift, and exit feasibility screening.

    Candidate results are geometric/IK evidence only. This function never
    modifies or steps the active episode and does not claim a physical grasp.
    """
    import hashlib
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    if expected_instruction != actual_instruction:
        raise ValueError("task_instruction_does_not_match_the_active_BDDL_task")
    if not 0.010 <= float(clearance_m) <= float(normalized_safety_policy["minimum_clearance_m"]):
        raise ValueError("candidate_clearance_must_preserve_the_10mm_hard_margin")
    if not isinstance(candidates, list) or len(candidates) > 24:
        raise ValueError("candidate_screen_requires_at_most_24_explicit_candidates")
    sim = environment.sim
    robot = environment.robots[0]
    model, live = sim.model._model, sim.data._data
    arm_ids = np.asarray(robot._ref_joint_indexes, dtype=np.int64)
    qadr = np.asarray(model.jnt_qposadr[arm_ids], dtype=np.int64)
    gripper_qadr = np.asarray(robot._ref_gripper_joint_pos_indexes, dtype=np.int64)
    closed_gripper = np.asarray(closed_gripper_qpos, dtype=np.float64)
    if closed_gripper.shape != gripper_qadr.shape or not np.isfinite(closed_gripper).all():
        raise ValueError("historical_stable_gripper_qpos_does_not_match_the_active_gripper")
    gripper_joint_ids = np.asarray(
        [sim.model.joint_name2id(name) for name in robot.gripper_joints], dtype=np.int64
    )
    gripper_limits = np.asarray(model.jnt_range[gripper_joint_ids], dtype=np.float64)
    if np.any(closed_gripper < gripper_limits[:, 0] - 1e-8) or np.any(closed_gripper > gripper_limits[:, 1] + 1e-8):
        raise ValueError("historical_closed_gripper_qpos_is_outside_gripper_joint_limits")
    integration_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.zeros(mujoco.mj_stateSize(model, integration_spec), dtype=np.float64)
    mujoco.mj_getState(model, live, integration, integration_spec)
    state_hash = hashlib.sha256(integration.tobytes()).hexdigest()
    if state_hash != start_state_sha256:
        raise ValueError("live_state_changed_before_candidate_screen")
    scratch = mujoco.MjData(model)
    mujoco.mj_setState(model, scratch, integration, integration_spec)
    mujoco.mj_forward(model, scratch)
    eef_site_id = int(robot.eef_site_id)
    root_body_id = int(target_root_body_id)
    root_joints = [
        int(model.body_jntadr[root_body_id]) + offset
        for offset in range(int(model.body_jntnum[root_body_id]))
        if int(model.jnt_type[int(model.body_jntadr[root_body_id]) + offset])
        == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(root_joints) != 1:
        raise ValueError("candidate_target_must_have_one_verified_free_root_joint")
    target_qadr = int(model.jnt_qposadr[root_joints[0]])
    lower = np.asarray(model.jnt_range[arm_ids, 0], dtype=np.float64)
    upper = np.asarray(model.jnt_range[arm_ids, 1], dtype=np.float64)
    if np.any(live.qpos[qadr] < lower - 1e-8) or np.any(live.qpos[qadr] > upper + 1e-8):
        raise ValueError("active_arm_qpos_is_outside_joint_limits")

    robot_geoms = np.asarray(sorted(normalized_safety_policy["robot_geom_ids"]), dtype=np.int64)
    robot_bodies = set(normalized_safety_policy["robot_body_ids"])
    forbidden_body_ids = set(normalized_safety_policy["forbidden_scene_body_ids"])
    allowed_target_body_ids = set(normalized_safety_policy["allowed_target_body_ids"])
    other_scene_body_ids = set(normalized_safety_policy["other_scene_body_ids"])
    forbidden_groups = normalized_safety_policy["forbidden_objects"]
    carried_support_geoms = set(
        int(item) for item in normalized_safety_policy.get("allowed_carried_support_contact_geom_ids", set())
    )
    all_groups = normalized_safety_policy.get("checked_objects", forbidden_groups)
    target_group = next(
        (item for item in all_groups if item["object_id"] == target_object_id), None
    )
    if target_group is None or root_body_id not in target_group["body_ids"]:
        raise ValueError("candidate_target_root_body_does_not_match_verified_scene_binding")
    target_geoms = set(target_group["geom_ids"])
    finger_sides = {
        "left": set(int(item) for item in left_finger_geom_ids),
        "right": set(int(item) for item in right_finger_geom_ids),
    }
    finger_ids = validate_finger_only_contact_geoms(
        left_finger_geom_ids,
        right_finger_geom_ids,
        gripper_geom_ids,
    )
    gripper_geoms = set(finger_ids)
    object_geoms = np.asarray(sorted(target_geoms), dtype=np.int64)
    object_body_transform = np.eye(4, dtype=np.float64)
    object_body_transform[:3, :3] = np.asarray(live.xmat[root_body_id]).reshape(3, 3)
    object_body_transform[:3, 3] = np.asarray(live.xpos[root_body_id])
    predicted_object = np.asarray(predicted_stable_object_pose_world, dtype=np.float64)
    if predicted_object.shape != (4, 4) or not np.isfinite(predicted_object).all():
        raise ValueError("predicted_stable_target_pose_is_not_a_finite_rigid_transform")
    if not np.allclose(predicted_object[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("predicted_stable_target_pose_has_an_invalid_homogeneous_row")
    lower_all = lower + 1e-7
    upper_all = upper - 1e-7
    open_gripper = np.asarray(live.qpos[gripper_qadr], dtype=np.float64).copy()

    def set_arm(q):
        scratch.qpos[qadr] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(model, scratch)

    def set_gripper(qpos):
        scratch.qpos[gripper_qadr] = np.asarray(qpos, dtype=np.float64)
        mujoco.mj_forward(model, scratch)

    def eef_pose():
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, np.asarray(scratch.site_xmat[eef_site_id]).reshape(9))
        return np.asarray(scratch.site_xpos[eef_site_id]).copy(), quat / np.linalg.norm(quat)

    def solve_ik(position, quaternion, seeds):
        best = None
        for seed in seeds:
            seed = np.asarray(seed, dtype=np.float64)
            if seed.shape != (7,) or not np.isfinite(seed).all():
                continue
            seed = np.clip(seed, lower_all, upper_all)

            def residual(q):
                set_arm(q)
                p, quat = eef_pose()
                r = quaternion_to_rotation_vector(
                    quaternion_multiply(quaternion, quaternion_conjugate(quat))
                )
                return np.r_[position - p, 0.15 * r, 1e-4 * (q - seed)]

            solution = least_squares(
                residual, seed, bounds=(lower_all, upper_all), x_scale="jac",
                max_nfev=300, ftol=1e-9, xtol=1e-9, gtol=1e-9,
            )
            q = np.asarray(solution.x)
            p, quat = eef_pose()
            pe = float(np.linalg.norm(p - position))
            re = float(np.linalg.norm(quaternion_to_rotation_vector(
                quaternion_multiply(quaternion, quaternion_conjugate(quat))
            )))
            score = pe + 0.15 * re
            if pe <= 0.003 and re <= 0.035 and (best is None or score < best[0]):
                best = (score, q.copy(), pe, re)
        return best

    def apply_carried_pose(eef_to_object):
        eef_rotation = np.asarray(scratch.site_xmat[eef_site_id]).reshape(3, 3)
        world_eef = np.eye(4, dtype=np.float64)
        world_eef[:3, :3] = eef_rotation
        world_eef[:3, 3] = scratch.site_xpos[eef_site_id]
        world_object = world_eef @ eef_to_object
        scratch.qpos[target_qadr:target_qadr + 3] = world_object[:3, 3]
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, world_object[:3, :3].reshape(9))
        scratch.qpos[target_qadr + 3:target_qadr + 7] = quat / np.linalg.norm(quat)
        mujoco.mj_forward(model, scratch)

    def set_target_pose(world_object):
        scratch.qpos[target_qadr:target_qadr + 3] = world_object[:3, 3]
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, world_object[:3, :3].reshape(9))
        scratch.qpos[target_qadr + 3:target_qadr + 7] = quat / np.linalg.norm(quat)
        mujoco.mj_forward(model, scratch)

    def evaluate(q, *, held=False, eef_to_object=None, allow_target_gripper=False):
        set_arm(q)
        if held:
            apply_carried_pose(eef_to_object)
        distances = {}
        violations = []
        for group in all_groups:
            group_id = group["object_id"]
            allowed = set()
            if (allow_target_gripper or held) and group_id == target_object_id:
                allowed = {
                    tuple(sorted((int(gripper_geom), int(target))))
                    for gripper_geom in gripper_geoms for target in group["geom_ids"]
                }
            distance = _minimum_distance(
                mujoco, model, scratch, robot_geoms,
                np.asarray(sorted(group["geom_ids"]), dtype=np.int64), allowed,
            )
            if distance is None:
                return {"status": "incomplete", "distances_m": distances, "violations": ["geom_distance_unavailable"]}
            distances[group_id] = distance
            required = float(group["minimum_clearance_m"])
            if group_id in {item["object_id"] for item in forbidden_groups}:
                required = max(required, clearance_m)
            if distance < required - 1e-9:
                violations.append({"object_id": group_id, "distance_m": distance, "required_m": required})
        for index in range(int(scratch.ncon)):
            contact = scratch.contact[index]
            ga, gb = int(contact.geom1), int(contact.geom2)
            ba, bb = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
            if ba in robot_bodies and bb in robot_bodies:
                violations.append({"self_contact_geoms": [ga, gb], "distance_m": float(contact.dist)})
                continue
            if ba in robot_bodies or bb in robot_bodies:
                scene_geom = gb if ba in robot_bodies else ga
                scene_body = bb if ba in robot_bodies else ba
                allowed = (allow_target_gripper or held) and scene_body in target_group["body_ids"] and (ga in gripper_geoms or gb in gripper_geoms)
                if allowed and float(contact.dist) >= -0.002:
                    continue
                if scene_body in forbidden_body_ids or scene_body in allowed_target_body_ids or scene_body in other_scene_body_ids:
                    violations.append({"robot_scene_contact_geoms": [ga, gb], "body_ids": [ba, bb], "distance_m": float(contact.dist)})
            elif held and (ba in target_group["body_ids"] or bb in target_group["body_ids"]):
                other_body = bb if ba in target_group["body_ids"] else ba
                # The two fingers may hold the target, and a supporting surface
                # may still touch it during the first lift millimeters. BDDL
                # forbidden object-pair contacts remain disallowed.
                target_pair_forbidden = any(
                    ({ba, bb} & pair["body_ids_a"] and {ba, bb} & pair["body_ids_b"])
                    for pair in normalized_safety_policy["forbidden_object_pairs"]
                )
                if (ga in gripper_geoms or gb in gripper_geoms) and float(contact.dist) >= -0.002:
                    continue
                if other_body in forbidden_body_ids or target_pair_forbidden:
                    violations.append({"held_object_forbidden_contact_geoms": [ga, gb], "body_ids": [ba, bb], "distance_m": float(contact.dist)})
        if held:
            carried_geoms = np.asarray(sorted(target_geoms), dtype=np.int64)
            for group in all_groups:
                if group["object_id"] == target_object_id:
                    continue
                distance = _minimum_distance(
                    mujoco, model, scratch, carried_geoms,
                    np.asarray(sorted(group["geom_ids"]), dtype=np.int64),
                    {
                        tuple(sorted((int(carried_geom), int(scene_geom))))
                        for carried_geom in target_geoms
                        for scene_geom in set(group["geom_ids"]) & carried_support_geoms
                    },
                )
                if distance is None:
                    return {"status": "incomplete", "distances_m": distances, "violations": ["carried_geom_distance_unavailable"]}
                key = f"carried:{target_object_id}->{group['object_id']}"
                distances[key] = distance
                required = clearance_m if group in forbidden_groups else 0.0
                if distance < required - 1e-9:
                    violations.append({"object_id": key, "distance_m": distance, "required_m": required})
                support_pairs = {
                    tuple(sorted((int(carried_geom), int(scene_geom))))
                    for carried_geom in target_geoms
                    for scene_geom in set(group["geom_ids"]) & carried_support_geoms
                }
                for pair in support_pairs:
                    pair_distance = _exact_pair_distance(mujoco, model, scratch, *pair)
                    if pair_distance is None or pair_distance < -0.002:
                        violations.append({"carried_support_contact_exceeded_tolerance": True, "geom_ids": list(pair), "distance_m": pair_distance, "minimum_allowed_m": -0.002})
        minimum = min(distances.values(), default=10.0)
        return {"status": "unsafe" if violations else "pass", "distances_m": distances, "minimum_distance_m": minimum, "violations": violations}

    results = []
    for candidate in candidates:
        entry = {"candidate_id": str(candidate.get("candidate_id")), "status": "rejected", "screening_type": "offline_scratch_only"}
        try:
            mujoco.mj_setState(model, scratch, integration, integration_spec)
            mujoco.mj_forward(model, scratch)
            transform = np.asarray(candidate["object_to_eef"], dtype=np.float64)
            direction = np.asarray(candidate["approach_direction_object"], dtype=np.float64)
            if transform.shape != (4, 4) or direction.shape != (3,) or not np.isfinite(transform).all() or not np.isfinite(direction).all():
                raise ValueError("candidate_transform_or_approach_is_invalid")
            if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
                raise ValueError("candidate_transform_is_not_rigid_homogeneous")
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                raise ValueError("candidate_approach_direction_is_degenerate")
            direction /= norm
            world_eef = object_body_transform @ transform
            position = world_eef[:3, 3]
            quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quaternion, world_eef[:3, :3].reshape(9))
            quaternion /= np.linalg.norm(quaternion)
            approach_world = object_body_transform[:3, :3] @ direction
            distance = float(candidate.get("approach_distance_m", 0.0606))
            pregrasp_position = position - approach_world * distance
            seeds = candidate.get("ik_seed_qpos", [])
            pre_ik = solve_ik(pregrasp_position, quaternion, seeds or [live.qpos[qadr].copy()])
            if pre_ik is None:
                entry["reason"] = "pregrasp_ik_not_found_within_joint_limits"
                results.append(entry)
                continue
            pre_check = evaluate(pre_ik[1])
            if pre_check["status"] != "pass":
                entry.update({"reason": "pregrasp_full_geom_or_self_collision", "pregrasp_check": pre_check})
                results.append(entry)
                continue
            set_gripper(closed_gripper)
            grasp_ik = solve_ik(position, quaternion, [pre_ik[1], *(seeds or [])])
            if grasp_ik is None:
                entry["reason"] = "closed_gripper_grasp_ik_not_found_within_joint_limits"
                results.append(entry)
                continue
            # Check the entire final approach using the live open-gripper
            # posture. Only the two verified finger sides may enter the
            # target's contact envelope; Palm/hand remains in the full 10 mm
            # geometry check performed by evaluate().
            set_gripper(open_gripper)
            approach_checks = []
            approach_failure = None
            for alpha in np.linspace(0.1, 1.0, 10):
                q = pre_ik[1] + float(alpha) * (grasp_ik[1] - pre_ik[1])
                check = evaluate(q, allow_target_gripper=True)
                approach_checks.append(check)
                if check["status"] != "pass":
                    approach_failure = "approach_full_geom_or_contact_semantics_failed"
                    break
            if approach_failure:
                entry.update({
                    "reason": approach_failure,
                    "approach_path_checks": approach_checks,
                })
                results.append(entry)
                continue
            # The archived object moved during the successful bilateral close.
            # Reuse that object-seat pose only as a scratch hypothesis while all
            # obstacles remain at their verified t10 positions.
            set_target_pose(predicted_object)
            target_layout_clearances = {}
            target_layout_safe = True
            rotation_start = Rotation.from_matrix(object_body_transform[:3, :3])
            rotation_end = Rotation.from_matrix(predicted_object[:3, :3])
            rotation_delta = (rotation_end * rotation_start.inv()).as_rotvec()
            for alpha in np.linspace(0.1, 1.0, 10):
                interpolation = np.eye(4, dtype=np.float64)
                interpolation[:3, 3] = (
                    (1.0 - alpha) * object_body_transform[:3, 3]
                    + alpha * predicted_object[:3, 3]
                )
                interpolation[:3, :3] = (
                    Rotation.from_rotvec(float(alpha) * rotation_delta) * rotation_start
                ).as_matrix()
                set_target_pose(interpolation)
                for group in forbidden_groups:
                    distance_to_forbidden = _minimum_distance(
                        mujoco, model, scratch, object_geoms,
                        np.asarray(sorted(group["geom_ids"]), dtype=np.int64), set(),
                    )
                    prior = target_layout_clearances.get(group["object_id"])
                    if distance_to_forbidden is not None:
                        target_layout_clearances[group["object_id"]] = (
                            distance_to_forbidden if prior is None else min(prior, distance_to_forbidden)
                        )
                    if distance_to_forbidden is None or distance_to_forbidden < clearance_m:
                        target_layout_safe = False
            set_target_pose(predicted_object)
            if not target_layout_safe:
                entry.update({"reason": "predicted_stable_target_pose_violates_forbidden_object_clearance", "target_layout_clearances_m": target_layout_clearances})
                results.append(entry)
                continue
            grasp_check = evaluate(grasp_ik[1], allow_target_gripper=True)
            side_distances = {}
            for side, geom_ids in finger_sides.items():
                pairs = []
                for finger in geom_ids:
                    for target in target_geoms:
                        value = _exact_pair_distance(mujoco, model, scratch, finger, target)
                        if value is not None:
                            pairs.append(value)
                side_distances[side] = min(pairs, default=None)
            if grasp_check["status"] != "pass" or any(
                value is None or value > 0.002 or value < -0.002
                for value in side_distances.values()
            ):
                entry.update({"reason": "grasp_contact_semantics_or_geometry_failed", "grasp_check": grasp_check, "finger_target_distance_m": side_distances})
                results.append(entry)
                continue

            # Candidate carried transform follows the candidate grasp, then the
            # exact Cartesian lift/exit geometry used by the existing controller.
            set_arm(grasp_ik[1])
            eef_r = np.asarray(scratch.site_xmat[eef_site_id]).reshape(3, 3)
            world_eef_grasp = np.eye(4); world_eef_grasp[:3, :3] = eef_r; world_eef_grasp[:3, 3] = scratch.site_xpos[eef_site_id]
            eef_to_object = np.linalg.inv(world_eef_grasp) @ predicted_object
            lift_positions = [position + np.asarray([0.0, 0.0, 0.030]), position + np.asarray([0.0, 0.0, 0.080])]
            held_q = [grasp_ik[1]]
            held_checks = []
            previous = grasp_ik[1]
            failed = None
            for lift_index, lift_position in enumerate(lift_positions):
                lift_ik = solve_ik(lift_position, quaternion, [previous, *(seeds or [])])
                if lift_ik is None:
                    failed = "lift_or_exit_ik_not_found_within_joint_limits"
                    break
                for alpha in np.linspace(0.1, 1.0, 10):
                    q = previous + float(alpha) * (lift_ik[1] - previous)
                    check = evaluate(q, held=True, eef_to_object=eef_to_object)
                    held_checks.append(check)
                    if check["status"] != "pass":
                        failed = "held_object_lift_or_exit_geometry_failed"
                        break
                if failed:
                    break
                previous = lift_ik[1]
                held_q.append(previous)
            if failed:
                entry.update({"reason": failed, "held_path_checks": held_checks})
                results.append(entry)
                continue
            entry.update({
                "status": "offline_feasible_candidate",
                "reason": None,
                "pregrasp_qpos": pre_ik[1].tolist(),
                "grasp_qpos": grasp_ik[1].tolist(),
                "lift30_qpos": held_q[1].tolist(),
                "exit80_qpos": held_q[2].tolist(),
                "ik_position_error_m": {"pregrasp": pre_ik[2], "grasp": grasp_ik[2]},
                "finger_target_distance_m": side_distances,
                "pregrasp_check": pre_check,
                "grasp_check": grasp_check,
                "approach_path_checks": approach_checks,
                "predicted_stable_target_pose_clearance_m": target_layout_clearances,
                "predicted_stable_object_pose_source": "historical_t141_close_result_as_unverified_scratch_prior",
                "held_path_checks": held_checks,
                "minimum_observed_clearance_m": min((item["minimum_distance_m"] for item in [pre_check, grasp_check, *held_checks] if item.get("minimum_distance_m") is not None), default=None),
                "minimum_forbidden_clearance_m": min((
                    distance
                    for check in [pre_check, grasp_check, *held_checks]
                    for name, distance in check.get("distances_m", {}).items()
                    if name in {group["object_id"] for group in forbidden_groups}
                    or name.split("->")[-1] in {group["object_id"] for group in forbidden_groups}
                ), default=None),
                "preferred_ik_seed_qpos": pre_ik[1].tolist(),
            })
        except Exception as exc:
            entry.update({"reason": f"{type(exc).__name__}:{exc}"})
        results.append(entry)
    live_after = np.zeros_like(integration)
    mujoco.mj_getState(model, live, live_after, integration_spec)
    unchanged = np.array_equal(integration, live_after)
    feasible = [item for item in results if item["status"] == "offline_feasible_candidate"]
    feasible.sort(key=lambda item: -float(item.get("minimum_forbidden_clearance_m") or -1.0))
    return {
        "operation": "screen_grasp_candidates",
        "status": "complete" if unchanged else "unsafe",
        "screening_type": "offline_scratch_only_not_physical_execution",
        "candidate_count": len(candidates),
        "feasible_count": len(feasible),
        "candidates": results,
        "ranked_candidate_ids": [item["candidate_id"] for item in feasible],
        "live_state_unchanged": unchanged,
        "start_state_sha256": state_hash,
    }
