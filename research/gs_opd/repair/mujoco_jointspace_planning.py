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


def _minimum_distance(
    mujoco, model, data, robot_geom_ids, object_geom_ids, excluded_pairs=None,
    *, return_detail=False,
):
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
        return {"distance_m": 10.0, "pair": None} if return_detail else 10.0
    pairs.sort(key=lambda item: item[0])
    best = None
    best_pair = None
    attempted = 0
    for lower_bound, robot_id, object_id in pairs:
        if best is not None and lower_bound >= best:
            break
        fromto = np.zeros(6, dtype=np.float64)
        value = float(mujoco.mj_geomDistance(model, data, robot_id, object_id, 10.0, fromto))
        attempted += 1
        if not np.isfinite(value):
            return {"distance_m": None, "pair": None} if return_detail else None
        if value < 10.0 and (best is None or value < best):
            best = value
            if return_detail:
                robot_body_id = int(model.geom_bodyid[robot_id])
                object_body_id = int(model.geom_bodyid[object_id])
                best_pair = {
                    "robot_geom_id": robot_id,
                    "robot_geom_name": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, robot_id
                    ),
                    "robot_body_id": robot_body_id,
                    "robot_body_name": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, robot_body_id
                    ),
                    "object_geom_id": object_id,
                    "object_geom_name": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, object_id
                    ),
                    "object_body_id": object_body_id,
                    "object_body_name": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, object_body_id
                    ),
                    "closest_points_world": fromto.tolist(),
                }
    if best is None:
        return {"distance_m": 10.0, "pair": None} if return_detail else 10.0
    if any(bound < best for bound, _, _ in pairs[attempted:]):
        return {"distance_m": None, "pair": None} if return_detail else None
    return {"distance_m": best, "pair": best_pair} if return_detail else best


def _verified_static_pairs_for_scene_group(robot_geom_ids, scene_geom_ids, verified_pairs):
    """Keep only audited robot-to-scene installation pairs for this exact group."""
    robot = {int(value) for value in robot_geom_ids}
    scene = {int(value) for value in scene_geom_ids}
    selected = set()
    for pair in verified_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            continue
        a, b = (int(value) for value in pair)
        if (a in robot and b in scene) or (b in robot and a in scene):
            selected.add(tuple(sorted((a, b))))
    return selected


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


def _candidate_gripper_close_samples(open_qpos, closed_qpos, joint_limits, *, sample_count=9):
    """Return a bounded, in-range gripper close sweep for scratch evaluation."""
    opened = np.asarray(open_qpos, dtype=np.float64)
    closed = np.asarray(closed_qpos, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    if (
        opened.ndim != 1
        or closed.shape != opened.shape
        or limits.shape != (opened.size, 2)
        or not np.isfinite(opened).all()
        or not np.isfinite(closed).all()
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] > limits[:, 1])
    ):
        raise ValueError("open_closed_gripper_postures_or_joint_limits_are_malformed")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 2 or sample_count > 33:
        raise ValueError("gripper_close_sample_count_must_be_between_2_and_33")
    tolerance = 1e-8
    if (
        np.any(opened < limits[:, 0] - tolerance)
        or np.any(opened > limits[:, 1] + tolerance)
        or np.any(closed < limits[:, 0] - tolerance)
        or np.any(closed > limits[:, 1] + tolerance)
    ):
        raise ValueError("open_or_closed_gripper_posture_is_outside_joint_limits")
    samples = []
    for alpha in np.linspace(0.0, 1.0, sample_count):
        qpos = (1.0 - float(alpha)) * opened + float(alpha) * closed
        if np.any(qpos < limits[:, 0] - tolerance) or np.any(qpos > limits[:, 1] + tolerance):
            raise ValueError("interpolated_gripper_posture_is_outside_joint_limits")
        samples.append({"alpha": float(alpha), "qpos": qpos.tolist()})
    return samples


def _candidate_closed_gripper_qpos(
    candidate, fallback_qpos, joint_limits, *, pad_gap_joint_offset_m=0.0
):
    """Resolve the scratch close posture for one candidate's estimated jaw width.

    Geometry candidates describe the desired pad separation in meters. The
    Panda gripper uses symmetric finger-joint positions. The measured active
    model offset between joint separation and physical pad gap is added before
    mapping the requested physical width. Candidates without width metadata
    retain the verified historical posture as a compatibility fallback.
    """
    fallback = np.asarray(fallback_qpos, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    if (
        fallback.ndim != 1
        or not np.isfinite(fallback).all()
        or limits.shape != (fallback.size, 2)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] > limits[:, 1])
    ):
        raise ValueError("fallback_gripper_posture_or_joint_limits_are_malformed")
    width_value = candidate.get("estimated_clamp_width_m")
    if width_value is None:
        closed = fallback.copy()
        source = "historical_stable_gripper_qpos_fallback_missing_candidate_width"
    else:
        if isinstance(width_value, bool):
            raise ValueError("candidate_estimated_clamp_width_m_must_be_finite_positive")
        width = float(width_value)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("candidate_estimated_clamp_width_m_must_be_finite_positive")
        maximum = candidate.get("maximum_opening_m")
        if maximum is not None:
            if isinstance(maximum, bool):
                raise ValueError("candidate_maximum_opening_m_must_be_finite_positive")
            maximum = float(maximum)
            if not np.isfinite(maximum) or maximum <= 0.0:
                raise ValueError("candidate_maximum_opening_m_must_be_finite_positive")
            if width > maximum + 1e-9:
                raise ValueError("candidate_estimated_clamp_width_exceeds_maximum_opening")
        if fallback.size != 2:
            raise ValueError("candidate_width_mapping_requires_two_Panda_gripper_joints")
        gap_offset = float(pad_gap_joint_offset_m)
        if not np.isfinite(gap_offset):
            raise ValueError("active_pad_gap_joint_offset_m_must_be_finite")
        joint_separation = width + gap_offset
        if joint_separation <= 0.0:
            raise ValueError("calibrated_candidate_gripper_joint_separation_must_be_positive")
        closed = np.asarray(
            [joint_separation / 2.0, -joint_separation / 2.0], dtype=np.float64
        )
        source = "candidate_pad_width_with_active_MuJoCo_pad_gap_calibration"
    if np.any(closed < limits[:, 0] - 1e-8) or np.any(closed > limits[:, 1] + 1e-8):
        raise ValueError("candidate_closed_gripper_qpos_is_outside_gripper_joint_limits")
    return closed, source


def _bilateral_finger_contact_result(side_distances_m, *, tolerance_m=0.002):
    """Require both verified finger sides to reach the target within tolerance."""
    if not isinstance(side_distances_m, dict) or set(side_distances_m) != {"left", "right"}:
        return {"pass": False, "reason": "verified_left_and_right_finger_distances_are_required"}
    failures = {}
    for side, value in side_distances_m.items():
        if value is None or not np.isfinite(float(value)):
            failures[side] = "distance_unavailable"
        elif float(value) > tolerance_m:
            failures[side] = "finger_side_does_not_reach_target"
        elif float(value) < -tolerance_m:
            failures[side] = "finger_side_penetration_exceeds_tolerance"
    return {
        "pass": not failures,
        "reason": None if not failures else "bilateral_finger_contact_geometry_failed",
        "failed_sides": failures,
        "tolerance_m": float(tolerance_m),
    }


def _candidate_eef_to_live_object_transform(
    world_eef_at_grasp: np.ndarray, world_object_at_grasp: np.ndarray
) -> np.ndarray:
    """Return the held transform from co-located candidate grasp poses.

    A historical post-grasp object pose is not aligned with the current
    candidate EEF pose. Use the live object pose at the candidate close instant;
    the historical pose remains a separate scene-layout prior.
    """
    eef = np.asarray(world_eef_at_grasp, dtype=np.float64)
    target = np.asarray(world_object_at_grasp, dtype=np.float64)
    for name, transform in (("eef", eef), ("object", target)):
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        ):
            raise ValueError(f"candidate_{name}_grasp_pose_must_be_a_finite_rigid_transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-5
        ):
            raise ValueError(f"candidate_{name}_grasp_pose_rotation_is_not_in_SO3")
    return np.linalg.inv(eef) @ target


def _candidate_robot_contact_violation(
    geom_a,
    geom_b,
    body_a,
    body_b,
    distance_m,
    *,
    robot_body_ids,
    target_body_ids,
    gripper_geom_ids,
    forbidden_scene_body_ids,
    allowed_target_body_ids,
    other_scene_body_ids,
    allowed_static_installation_pair_ids,
    allow_target_gripper,
    allowed_target_contact_geom_ids=None,
):
    """Classify one MuJoCo contact using the screen's unchanged contact policy."""
    robot_bodies = set(robot_body_ids)
    body_a, body_b = int(body_a), int(body_b)
    geom_a, geom_b = int(geom_a), int(geom_b)
    if body_a in robot_bodies and body_b in robot_bodies:
        return {"self_contact_geoms": [geom_a, geom_b], "distance_m": float(distance_m)}
    if (body_a in robot_bodies) == (body_b in robot_bodies):
        return None
    pair = tuple(sorted((geom_a, geom_b)))
    if pair in {tuple(sorted(map(int, item))) for item in allowed_static_installation_pair_ids}:
        return None
    scene_body = body_b if body_a in robot_bodies else body_a
    robot_geom = geom_a if body_a in robot_bodies else geom_b
    target_contact = scene_body in set(target_body_ids)
    target_contact_geom_ids = (
        set(gripper_geom_ids)
        if allowed_target_contact_geom_ids is None
        else set(int(value) for value in allowed_target_contact_geom_ids)
    )
    if (
        allow_target_gripper
        and target_contact
        and robot_geom in target_contact_geom_ids
        and float(distance_m) >= -0.002
    ):
        return None
    if scene_body in set(forbidden_scene_body_ids) | set(allowed_target_body_ids) | set(other_scene_body_ids):
        return {
            "robot_scene_contact_geoms": [geom_a, geom_b],
            "body_ids": [body_a, body_b],
            "distance_m": float(distance_m),
        }
    return None


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
    goal_configuration_preflight_only: bool = False,
    hypothetical_eef_to_object_transform: list[list[float]] | np.ndarray | None = None,
    scene_joint_qpos_overrides: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve a world EEF goal, then plan from the active MuJoCo 7D qpos.

    Planning and every geometry query run on independent MjData. This function
    never steps the live environment or edits its qpos.
    """
    import hashlib
    import mujoco
    import time
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    planning_started = time.monotonic()
    if expected_instruction != actual_instruction:
        raise ValueError("task_instruction_does_not_match_the_active_BDDL_task")
    if not start_state_sha256 or len(start_state_sha256) != 64:
        raise ValueError("start_state_sha256_is_missing_or_malformed")
    if not isinstance(normalized_safety_policy, dict):
        raise ValueError("verified_safety_geometry_policy_is_required")
    planner_values = dict(planner_config or {})
    start_joint_limit_tolerance_rad = float(
        planner_values.get("start_joint_limit_tolerance_rad", 0.0)
    )
    if (
        not np.isfinite(start_joint_limit_tolerance_rad)
        or start_joint_limit_tolerance_rad < 0.0
    ):
        raise ValueError("start_joint_limit_tolerance_must_be_finite_and_non_negative")
    if not isinstance(goal_configuration_preflight_only, bool):
        raise ValueError("goal_configuration_preflight_only_must_be_boolean")
    if scene_joint_qpos_overrides is not None and not goal_configuration_preflight_only:
        raise ValueError("scene_joint_qpos_overrides_are_diagnostic_goal_preflight_only")
    if goal_configuration_preflight_only:
        if hypothetical_eef_to_object_transform is None:
            raise ValueError("hypothetical_attachment_transform_is_required_for_goal_preflight")
        carried = normalized_safety_policy.get("carried_object")
        if not isinstance(carried, dict):
            raise ValueError("verified_carried_object_binding_is_required_for_goal_preflight")
        hypothetical_transform = np.asarray(
            hypothetical_eef_to_object_transform, dtype=np.float64,
        )
        if (
            hypothetical_transform.shape != (4, 4)
            or not np.isfinite(hypothetical_transform).all()
            or not np.allclose(hypothetical_transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
            or not np.allclose(
                hypothetical_transform[:3, :3].T @ hypothetical_transform[:3, :3],
                np.eye(3), atol=1e-6,
            )
            or not np.isclose(np.linalg.det(hypothetical_transform[:3, :3]), 1.0, atol=1e-6)
        ):
            raise ValueError("hypothetical_eef_to_object_transform_must_be_finite_SE3")
        normalized_safety_policy = dict(normalized_safety_policy)
        normalized_safety_policy["carried_object"] = dict(carried)
        normalized_safety_policy["carried_object"]["eef_to_object_transform"] = (
            hypothetical_transform.copy()
        )
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
    validated_joint_overrides = []
    for index, row in enumerate(scene_joint_qpos_overrides or []):
        if not isinstance(row, dict):
            raise ValueError(f"scene_joint_qpos_overrides_{index}_must_be_an_object")
        address = row.get("qpos_address")
        position_value = row.get("position")
        if (
            isinstance(address, bool) or not isinstance(address, int)
            or not 0 <= address < int(model.nq)
            or isinstance(position_value, bool)
        ):
            raise ValueError(f"scene_joint_qpos_overrides_{index}_address_or_position_is_invalid")
        position = float(position_value)
        if not np.isfinite(position):
            raise ValueError(f"scene_joint_qpos_overrides_{index}_position_must_be_finite")
        validated_joint_overrides.append({
            "qpos_address": int(address),
            "position": position,
            "joint_name": str(row.get("joint_name", "")),
        })
    if len({row["qpos_address"] for row in validated_joint_overrides}) != len(validated_joint_overrides):
        raise ValueError("scene_joint_qpos_overrides_duplicate_qpos_addresses")
    for row in validated_joint_overrides:
        scratch.qpos[row["qpos_address"]] = row["position"]
    mujoco.mj_forward(model, scratch)
    robot_geoms = np.asarray(sorted(normalized_safety_policy["robot_geom_ids"]), dtype=np.int64)
    robot_bodies = set(normalized_safety_policy["robot_body_ids"])
    forbidden = normalized_safety_policy["forbidden_objects"]
    checked_objects = normalized_safety_policy.get("checked_objects", forbidden)
    allowed_pair_ids = set(normalized_safety_policy["allowed_contact_pair_ids"])
    static_installation_pair_ids = {
        tuple(sorted(map(int, pair)))
        for pair in normalized_safety_policy.get(
            "allowed_static_installation_contact_pair_ids", set()
        )
    }
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
    if (
        np.any(q_start < lower - start_joint_limit_tolerance_rad)
        or np.any(q_start > upper + start_joint_limit_tolerance_rad)
    ):
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
        clearance_pair_details = {}
        forbidden_contacts = []
        for item in checked_objects:
            group_allowed_pairs = {
                pair for pair in allowed_pair_ids
                if next(geom_id for geom_id in pair if geom_id not in set(robot_geoms))
                in item["geom_ids"]
            } if item["policy_class"] == "allowed_target_objects" else set()
            group_static_installation_pairs = _verified_static_pairs_for_scene_group(
                robot_geoms, item["geom_ids"], static_installation_pair_ids,
            )
            distance_detail = _minimum_distance(
                mujoco, model, scratch, robot_geoms, item["geom_ids"],
                group_allowed_pairs | group_static_installation_pairs,
                return_detail=True,
            )
            distance = distance_detail["distance_m"]
            if distance is None or not np.isfinite(distance):
                return ConfigurationCheck("incomplete", None, {}, diagnostics={"object_id": item["object_id"]})
            clearances[item["object_id"]] = distance
            clearance_pair_details[item["object_id"]] = distance_detail.get("pair")
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
                distance_detail = _minimum_distance(
                    mujoco, model, scratch, carried_geoms, item["geom_ids"], allowed_pairs,
                    return_detail=True,
                )
                distance = distance_detail["distance_m"]
                if distance is None or not np.isfinite(distance):
                    return ConfigurationCheck(
                        "incomplete", None, {}, diagnostics={"carried_object_id": carried["object_id"], "scene_object_id": item["object_id"]}
                    )
                key = f"carried:{carried['object_id']}->{item['object_id']}"
                clearances[key] = distance
                clearance_pair_details[key] = distance_detail.get("pair")
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
            if tuple(sorted((geom_a, geom_b))) in static_installation_pair_ids:
                continue
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
            {
                "qpos_rad": np.asarray(qpos).tolist(),
                "clearance_pairs_by_object": clearance_pair_details,
            },
        )

    configured_margin = float(
        planner_values.pop("minimum_clearance_m", normalized_safety_policy["minimum_clearance_m"])
    )
    if not np.isfinite(configured_margin) or configured_margin < 0.0:
        raise ValueError("planner_clearance_must_be_finite_and_non_negative")
    planner_cfg = JointSpacePlannerConfig(
        **planner_values,
        minimum_clearance_m=configured_margin,
    )
    if goal_configuration_preflight_only and not planner_cfg.require_safe_configurations:
        raise ValueError("goal_configuration_preflight_requires_full_collision_checks")
    goal_ik_configuration_checks = 0
    if planner_cfg.require_safe_configurations:
        safe_ik_candidates = []
        for item in ik_valid:
            goal_check = check_configuration(item["qpos"])
            goal_ik_configuration_checks += 1
            safe = bool(
                goal_check.status == "complete"
                and goal_check.minimum_forbidden_clearance_m is not None
                and np.isfinite(float(goal_check.minimum_forbidden_clearance_m))
                and float(goal_check.minimum_forbidden_clearance_m)
                >= planner_cfg.minimum_clearance_m
                and not goal_check.forbidden_contacts
                and not goal_check.self_contacts
            )
            item["goal_configuration_safe"] = safe
            item["goal_configuration_minimum_forbidden_clearance_m"] = (
                goal_check.minimum_forbidden_clearance_m
            )
            item["goal_configuration_clearances_by_object_m"] = (
                goal_check.clearances_by_object_m
            )
            item["goal_configuration_clearance_pairs_by_object"] = (
                goal_check.diagnostics.get("clearance_pairs_by_object", {})
            )
            item["goal_configuration_forbidden_contacts"] = goal_check.forbidden_contacts
            item["goal_configuration_self_contacts"] = goal_check.self_contacts
            if safe:
                safe_ik_candidates.append(item)
        if not safe_ik_candidates:
            live_state_after = np.zeros_like(integration_state)
            mujoco.mj_getState(model, live_data, live_state_after, integration_spec)
            unchanged = np.array_equal(integration_state, live_state_after)
            if not unchanged:
                raise RuntimeError("live_environment_mutated_during_scratch_planning")
            candidate_diagnostics = [
                {
                    "seed_index": item["seed_index"],
                    "selection_seed_distance_rad": item["selection_seed_distance_rad"],
                    "minimum_forbidden_clearance_m": item.get(
                        "goal_configuration_minimum_forbidden_clearance_m"
                    ),
                    "clearances_by_object_m": item.get(
                        "goal_configuration_clearances_by_object_m", {}
                    ),
                    "clearance_pairs_by_object": item.get(
                        "goal_configuration_clearance_pairs_by_object", {}
                    ),
                    "forbidden_contact_count": len(
                        item.get("goal_configuration_forbidden_contacts", [])
                    ),
                    "self_contact_count": len(
                        item.get("goal_configuration_self_contacts", [])
                    ),
                }
                for item in ik_valid
            ]
            return {
                "status": "planning_failed",
                "failure_reason": (
                    "ValueError: goal_configuration_is_unsafe:no_collision_free_IK_candidate:"
                    f"{candidate_diagnostics}"
                ),
                "planner_source": "jointspace_rrt_connect.v1",
                "planning_seconds": time.monotonic() - planning_started,
                "config_checks": 0,
                "goal_ik_configuration_checks": goal_ik_configuration_checks,
                "iterations": 0,
                "selected_route_index": None,
                "minimum_forbidden_clearance_m": None,
                "ik_goal_qpos": None,
                "selected_ik_seed_index": None,
                "ik_candidates": [
                    {key: value.tolist() if isinstance(value, np.ndarray) else value
                     for key, value in item.items()}
                    for item in ik_candidates
                ],
                "route_candidates": [],
                "live_state_unchanged": unchanged,
                "goal_configuration_preflight_only": goal_configuration_preflight_only,
                "hypothetical_attachment_transform_used": goal_configuration_preflight_only,
                "scene_joint_qpos_overrides": validated_joint_overrides,
            }
        q_goal = safe_ik_candidates[0]["qpos"]
        selected_ik_seed_index = int(safe_ik_candidates[0]["seed_index"])
    else:
        selected_ik_seed_index = int(ik_valid[0]["seed_index"])
    if goal_configuration_preflight_only:
        live_state_after = np.zeros_like(integration_state)
        mujoco.mj_getState(model, live_data, live_state_after, integration_spec)
        unchanged = np.array_equal(integration_state, live_state_after)
        if not unchanged:
            raise RuntimeError("live_environment_mutated_during_goal_configuration_preflight")
        selected = next(
            item for item in ik_valid
            if int(item["seed_index"]) == selected_ik_seed_index
        )
        return {
            "status": "goal_configuration_preflight_passed",
            "planner_source": "jointspace_rrt_connect.v1",
            "planning_seconds": time.monotonic() - planning_started,
            "config_checks": 0,
            "goal_ik_configuration_checks": goal_ik_configuration_checks,
            "iterations": 0,
            "selected_ik_seed_index": selected_ik_seed_index,
            "minimum_forbidden_clearance_m": selected.get(
                "goal_configuration_minimum_forbidden_clearance_m"
            ),
            "goal_configuration_clearances_by_object_m": selected.get(
                "goal_configuration_clearances_by_object_m", {}
            ),
            "goal_configuration_clearance_pairs_by_object": selected.get(
                "goal_configuration_clearance_pairs_by_object", {}
            ),
            "ik_goal_qpos": q_goal.tolist(),
            "ik_candidates": [
                {key: value.tolist() if isinstance(value, np.ndarray) else value
                 for key, value in item.items()}
                for item in ik_candidates
            ],
            "route_candidates": [],
            "live_state_unchanged": unchanged,
            "goal_configuration_preflight_only": True,
            "hypothetical_attachment_transform_used": True,
            "scene_joint_qpos_overrides": validated_joint_overrides,
            "safety_certified": False,
        }
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
            "planning_seconds": time.monotonic() - planning_started,
            "config_checks": result.config_checks,
            "goal_ik_configuration_checks": goal_ik_configuration_checks,
            "iterations": result.iterations,
            "selected_ik_seed_index": selected_ik_seed_index,
            "ik_candidates": [
                {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in item.items()}
                for item in ik_candidates
            ],
            "route_candidates": result.candidates,
            "live_state_unchanged": unchanged,
        }
    def eef_path_for(qpos_path):
        path = []
        for qpos in qpos_path:
            position, quaternion = pose_at(qpos)
            path.append(np.r_[position, quaternion].tolist())
        return path

    eef_path = eef_path_for(result.qpos_path)
    route_candidates = []
    for route in result.candidates:
        route_candidates.append({
            "route_index": int(route["route_index"]),
            "iterations": int(route["iterations"]),
            "metrics": route["metrics"],
            "eef_reference_poses_world": eef_path_for(route["path"]),
        })
    return {
        "status": result.status,
        "planner_source": result.planner_source,
        "planning_seconds": time.monotonic() - planning_started,
        "config_checks": result.config_checks,
        "goal_ik_configuration_checks": goal_ik_configuration_checks,
        "iterations": result.iterations,
        "selected_route_index": result.selected_route_index,
        "selected_ik_seed_index": selected_ik_seed_index,
        "minimum_forbidden_clearance_m": result.minimum_forbidden_clearance_m,
        "qpos_path": result.qpos_path.tolist(),
        "eef_reference_poses_world": eef_path,
        "ik_goal_qpos": q_goal.tolist(),
        "ik_candidates": [
            {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in item.items()}
            for item in ik_candidates
        ],
        "route_candidates": route_candidates,
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
    strict_target_contact_policy: bool = False,
    task_approach_only: bool = False,
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
    if not isinstance(strict_target_contact_policy, bool):
        raise ValueError("strict_target_contact_policy_must_be_boolean")
    if not isinstance(task_approach_only, bool):
        raise ValueError("task_approach_only_must_be_boolean")
    if strict_target_contact_policy and task_approach_only:
        raise ValueError("strict_safety_and_task_approach_screen_modes_are_mutually_exclusive")
    if strict_target_contact_policy and float(clearance_m) < 0.010:
        raise ValueError("strict_target_contact_policy_requires_at_least_10mm_nonpad_clearance")
    if (
        not np.isfinite(float(clearance_m))
        or float(clearance_m) < 0.0
        or float(clearance_m) > float(normalized_safety_policy["minimum_clearance_m"])
    ):
        raise ValueError("candidate_clearance_must_be_non_negative_and_within_the_declared_policy")
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
    static_installation_pairs = {
        tuple(sorted(int(value) for value in pair))
        for pair in normalized_safety_policy.get("allowed_static_installation_contact_pair_ids", set())
    }
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
    if (
        not np.isfinite(open_gripper).all()
        or np.any(open_gripper < gripper_limits[:, 0] - 1e-8)
        or np.any(open_gripper > gripper_limits[:, 1] + 1e-8)
    ):
        raise ValueError("active_open_gripper_qpos_is_outside_gripper_joint_limits")

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

    pad_geom_ids = {}
    for side, expected_suffix in (("left", "finger1_pad_collision"), ("right", "finger2_pad_collision")):
        matches = [
            geom_id for geom_id in finger_sides[side]
            if str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or "").endswith(expected_suffix)
        ]
        if len(matches) != 1:
            raise ValueError(f"verified_{side}_fingertip_pad_geom_is_not_unique")
        pad_geom_ids[side] = int(matches[0])

    pad_geom_id_set = set(pad_geom_ids.values())
    target_contact_geom_ids = (
        pad_geom_id_set
        if strict_target_contact_policy or task_approach_only
        else gripper_geoms
    )

    def fingertip_distances():
        distances = {}
        for side in ("left", "right"):
            values = [
                _exact_pair_distance(mujoco, model, scratch, pad_geom_ids[side], target_geom)
                for target_geom in sorted(target_geoms)
            ]
            valid = [float(value) for value in values if value is not None]
            distances[side] = min(valid) if valid else None
        return distances

    def fingertip_target_distance_details():
        details = {}
        for side in ("left", "right"):
            rows = [
                (float(distance), int(target_geom))
                for target_geom in sorted(target_geoms)
                if (distance := _exact_pair_distance(
                    mujoco, model, scratch, pad_geom_ids[side], target_geom,
                )) is not None
            ]
            if rows:
                distance, target_geom = min(rows, key=lambda row: row[0])
                details[side] = {
                    "distance_m": distance,
                    "pad_geom_id": pad_geom_ids[side],
                    "target_geom_id": target_geom,
                }
            else:
                details[side] = None
        return details

    def pad_opening_width():
        value = _exact_pair_distance(
            mujoco,
            model,
            scratch,
            pad_geom_ids["left"],
            pad_geom_ids["right"],
        )
        return value

    active_open_pad_gap_m = pad_opening_width()
    if active_open_pad_gap_m is None:
        raise ValueError("active_open_pad_gap_unavailable_for_candidate_width_calibration")
    active_open_joint_separation_m = float(open_gripper[0] - open_gripper[1])
    pad_gap_joint_offset_m = active_open_joint_separation_m - float(active_open_pad_gap_m)

    def evaluate(q, *, held=False, eef_to_object=None, allow_target_gripper=False):
        set_arm(q)
        if held:
            apply_carried_pose(eef_to_object)
        distances = {}
        minimum_pairs = {}
        violations = []
        for group in all_groups:
            group_id = group["object_id"]
            excluded_pairs = set()
            if (allow_target_gripper or held) and group_id == target_object_id:
                excluded_pairs.update({
                    tuple(sorted((int(gripper_geom), int(target))))
                    for gripper_geom in target_contact_geom_ids for target in group["geom_ids"]
                })
            excluded_pairs.update(
                pair for pair in static_installation_pairs
                if any(geom_id not in robot_geoms for geom_id in pair)
                and any(geom_id in set(group["geom_ids"]) for geom_id in pair)
            )
            clearance_robot_geoms = robot_geoms
            if strict_target_contact_policy and group_id == target_object_id:
                clearance_robot_geoms = np.asarray(
                    sorted(set(int(value) for value in robot_geoms) - pad_geom_id_set),
                    dtype=np.int64,
                )
            # Task-approach screens need the exact pair for every reported
            # scene group. In particular, the aggregated unmapped environment
            # group can otherwise hide a fixed installation contact behind a
            # generic collision label. This adds diagnostics only; it does not
            # change the collision threshold or candidate acceptance result.
            record_pair = task_approach_only or (
                strict_target_contact_policy and group_id == target_object_id
            )
            distance_result = _minimum_distance(
                mujoco, model, scratch, clearance_robot_geoms,
                np.asarray(sorted(group["geom_ids"]), dtype=np.int64), excluded_pairs,
                return_detail=record_pair,
            )
            if record_pair:
                distance = distance_result["distance_m"]
                minimum_pairs[group_id] = distance_result["pair"]
            else:
                distance = distance_result
            if distance is None:
                return {
                    "status": "incomplete", "distances_m": distances,
                    "minimum_pair_by_object": minimum_pairs,
                    "violations": ["geom_distance_unavailable"],
                }
            distances[group_id] = distance
            required = 0.0 if task_approach_only else float(group["minimum_clearance_m"])
            # Task-feasibility approach screening rejects actual contact or
            # penetration, but it must not inherit the Strict-10mm safety
            # margin as a candidate-search gate. Keep that margin in the
            # non-task-feasibility safety screening modes.
            if (
                group_id in {item["object_id"] for item in forbidden_groups}
                and not task_approach_only
            ):
                required = max(required, clearance_m)
            if distance < required - 1e-9:
                violations.append({"object_id": group_id, "distance_m": distance, "required_m": required})
        for index in range(int(scratch.ncon)):
            contact = scratch.contact[index]
            ga, gb = int(contact.geom1), int(contact.geom2)
            ba, bb = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
            contact_violation = _candidate_robot_contact_violation(
                ga,
                gb,
                ba,
                bb,
                float(contact.dist),
                robot_body_ids=robot_bodies,
                target_body_ids=target_group["body_ids"],
                gripper_geom_ids=gripper_geoms,
                forbidden_scene_body_ids=forbidden_body_ids,
                allowed_target_body_ids=allowed_target_body_ids,
                other_scene_body_ids=other_scene_body_ids,
                allowed_static_installation_pair_ids=static_installation_pairs,
                allow_target_gripper=allow_target_gripper or held,
                allowed_target_contact_geom_ids=target_contact_geom_ids,
            )
            if contact_violation is not None:
                contact_violation["geom_names"] = [
                    str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
                    for geom_id in contact_violation.get("robot_scene_contact_geoms", contact_violation.get("self_contact_geoms", []))
                ]
                contact_violation["body_names"] = [
                    str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
                    for body_id in contact_violation.get("body_ids", [])
                ]
                violations.append(contact_violation)
            if ba not in robot_bodies and bb not in robot_bodies and held and (ba in target_group["body_ids"] or bb in target_group["body_ids"]):
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
                    return {
                        "status": "incomplete", "distances_m": distances,
                        "minimum_pair_by_object": minimum_pairs,
                        "violations": ["carried_geom_distance_unavailable"],
                    }
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
        return {
            "status": "unsafe" if violations else "pass",
            "distances_m": distances,
            "minimum_distance_m": minimum,
            "minimum_pair_by_object": minimum_pairs,
            "violations": violations,
        }

    results = []
    forbidden_object_ids = {item["object_id"] for item in forbidden_groups}
    mujoco.mj_setState(model, scratch, integration, integration_spec)
    mujoco.mj_forward(model, scratch)
    set_gripper(open_gripper)
    t0_baseline_check = evaluate(live.qpos[qadr].copy())
    t0_forbidden_clearance = {
        object_id: t0_baseline_check.get("distances_m", {}).get(object_id)
        for object_id in forbidden_object_ids
    }

    def finish_entry():
        if stage_failures and not entry.get("failure_reasons"):
            entry["failure_reasons"] = list(stage_failures)
        if stage_failures and not entry.get("reason"):
            entry["reason"] = stage_failures[0]
        checks = []
        for key in ("pregrasp_check", "grasp_check"):
            if isinstance(entry.get(key), dict):
                checks.append(entry[key])
        for key in ("approach_path_checks", "gripper_close_path_checks", "held_path_checks"):
            for item in entry.get(key, []):
                if not isinstance(item, dict):
                    continue
                checks.append(item.get("geometry_check", item))
        minimum_by_object = {}
        for check in checks:
            for object_id, value in check.get("distances_m", {}).items():
                if object_id in forbidden_object_ids or object_id.split("->")[-1] in forbidden_object_ids:
                    if value is not None:
                        minimum_by_object[object_id] = min(
                            float(value), minimum_by_object.get(object_id, float("inf"))
                        )
        for object_id, value in entry.get("target_layout_clearances_m", {}).items():
            if value is not None:
                key = f"book->{object_id}"
                minimum_by_object[key] = min(float(value), minimum_by_object.get(key, float("inf")))
        entry["minimum_forbidden_clearance_by_object_m"] = minimum_by_object
        entry["t0_robot_clearance_by_forbidden_object_m"] = t0_forbidden_clearance
        entry["minimum_forbidden_clearance_m"] = min(minimum_by_object.values(), default=None)
        reasons = list(dict.fromkeys(entry.get("failure_reasons", [])))
        reason = str(entry.get("reason") or "")
        if reason and reason not in reasons:
            reasons.insert(0, reason)
        classes = set()
        for failure_reason in reasons:
            if "ik_not_found" in failure_reason:
                classes.add("ik_unreachable")
            elif (
                "bilateral_finger" in failure_reason
                or "finger_contact_geometry" in failure_reason
                or "gripper_opening_range" in failure_reason
            ):
                classes.add("grasp_geometry_infeasible")
            elif any(token in failure_reason for token in ("collision", "geometry_failed", "clearance", "unsafe", "contact_semantics")):
                classes.add("collision_constraint_failure")
            else:
                classes.add("candidate_or_screening_error")
        if entry.get("status") == "offline_feasible_candidate":
            entry["failure_class"] = None
            entry["failure_classes"] = []
        else:
            entry["failure_classes"] = sorted(classes)
            entry["failure_class"] = next(iter(reasons), "candidate_or_screening_error")
            if entry["failure_class"] != "candidate_or_screening_error":
                matching = next((
                    failure_class for failure_class in (
                        "ik_unreachable", "grasp_geometry_infeasible", "collision_constraint_failure"
                    )
                    if failure_class in classes
                ), None)
                entry["failure_class"] = matching or "candidate_or_screening_error"
        results.append(entry)

    for candidate in candidates:
        stage_failures = []
        entry = {
            "candidate_id": str(candidate.get("candidate_id")),
            "estimated_clamp_width_m": candidate.get("estimated_clamp_width_m"),
            "candidate_maximum_opening_m": candidate.get("maximum_opening_m"),
            "status": "rejected",
            "screening_type": (
                "offline_scratch_task_approach_feasibility"
                if task_approach_only else "offline_scratch_only"
            ),
            "feasibility_scope": (
                "open_gripper_pregrasp_to_grasp_cartesian_path_only"
                if task_approach_only else None
            ),
            "closed_grasp_target_pose_world_source": "active_scene_target_pose_at_candidate_close",
            "historical_object_pose_prior_used_for": "target_layout_clearance_only",
            "finger_target_distance_m": {"left": None, "right": None},
            "finger_distance_status": "not_evaluated_before_closed_grasp_ik",
            "joint_reachability": {},
        }
        try:
            mujoco.mj_setState(model, scratch, integration, integration_spec)
            mujoco.mj_forward(model, scratch)
            if task_approach_only:
                candidate_closed_gripper = open_gripper.copy()
                close_qpos_source = "active_open_pose_task_approach_feasibility_only"
            else:
                candidate_closed_gripper, close_qpos_source = _candidate_closed_gripper_qpos(
                    candidate,
                    closed_gripper,
                    gripper_limits,
                    pad_gap_joint_offset_m=pad_gap_joint_offset_m,
                )
            close_samples = _candidate_gripper_close_samples(
                open_gripper,
                candidate_closed_gripper,
                gripper_limits,
                sample_count=9,
            )
            entry["gripper_close_qpos"] = {
                "source": close_qpos_source,
                "joint_qpos": candidate_closed_gripper.tolist(),
                "estimated_pad_separation_m": candidate.get("estimated_clamp_width_m"),
                "active_pad_gap_joint_offset_m": (
                    pad_gap_joint_offset_m
                    if candidate.get("estimated_clamp_width_m") is not None
                    else None
                ),
            }
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
                entry["joint_reachability"] = {"pregrasp": {"reachable": False}}
                stage_failures.append("pregrasp_ik_not_found_within_joint_limits")
            else:
                entry["joint_reachability"] = {
                    "pregrasp": {
                        "reachable": True,
                        "qpos": pre_ik[1].tolist(),
                        "position_error_m": pre_ik[2],
                        "rotation_error_rad": pre_ik[3],
                    }
                }
            pre_check = evaluate(pre_ik[1]) if pre_ik is not None else None
            if pre_check is not None:
                entry["pregrasp_check"] = pre_check
                if pre_check["status"] != "pass":
                    stage_failures.append("pregrasp_full_geom_or_self_collision")
            set_gripper(candidate_closed_gripper)
            grasp_seeds = ([pre_ik[1]] if pre_ik is not None else []) + (seeds or []) + [live.qpos[qadr].copy()]
            grasp_ik = solve_ik(position, quaternion, grasp_seeds)
            grasp_ik_key = "grasp_open_gripper" if task_approach_only else "grasp_closed_gripper"
            if grasp_ik is None:
                entry["joint_reachability"][grasp_ik_key] = {"reachable": False}
                stage_failures.append(
                    "open_gripper_grasp_ik_not_found_within_joint_limits"
                    if task_approach_only
                    else "closed_gripper_grasp_ik_not_found_within_joint_limits"
                )
                finish_entry()
                continue
            entry["joint_reachability"][grasp_ik_key] = {
                "reachable": True,
                "qpos": grasp_ik[1].tolist(),
                "position_error_m": grasp_ik[2],
                "rotation_error_rad": grasp_ik[3],
            }
            entry["finger_distance_status"] = "awaiting_target_close_sweep"
            # Check the entire final approach using the live open-gripper
            # posture. Only the two verified finger sides may enter the
            # target's contact envelope; Palm/hand remains constrained by the
            # per-object clearance tier checked by evaluate().
            set_gripper(open_gripper)
            approach_checks = []
            approach_failure = None
            if pre_ik is not None:
                previous_approach_q = pre_ik[1].copy()
                for alpha in np.linspace(0.1, 1.0, 10):
                    alpha = float(alpha)
                    if task_approach_only:
                        sample_position = (1.0 - alpha) * pregrasp_position + alpha * position
                        sample_ik = solve_ik(
                            sample_position,
                            quaternion,
                            [previous_approach_q, *(seeds or []), pre_ik[1]],
                        )
                        if sample_ik is None:
                            check = {
                                "status": "unreachable",
                                "alpha": alpha,
                                "distances_m": {},
                                "violations": [{
                                    "surface_approach_cartesian_ik_not_found_within_joint_limits": True,
                                }],
                            }
                        else:
                            previous_approach_q = sample_ik[1].copy()
                            check = evaluate(sample_ik[1], allow_target_gripper=True)
                            pad_distances = fingertip_target_distance_details()
                            check["alpha"] = alpha
                            check["pad_target_distances"] = pad_distances
                            if any(
                                item is not None and float(item["distance_m"]) < -1e-6
                                for item in pad_distances.values()
                            ):
                                check["violations"].append({
                                    "premature_fingertip_target_penetration": True,
                                    "maximum_allowed_penetration_m": 0.0,
                                    "pad_target_distances": pad_distances,
                                })
                                check["status"] = "unsafe"
                    else:
                        q = pre_ik[1] + alpha * (grasp_ik[1] - pre_ik[1])
                        check = evaluate(q, allow_target_gripper=True)
                    approach_checks.append(check)
                    if check["status"] != "pass" and approach_failure is None:
                        if task_approach_only:
                            approach_failure = (
                                "surface_approach_cartesian_ik_not_found_within_joint_limits"
                                if check.get("status") == "unreachable"
                                else "task_surface_approach_physical_collision_or_premature_pad_contact"
                            )
                        else:
                            approach_failure = "approach_full_geom_or_contact_semantics_failed"
                entry["approach_path_checks"] = approach_checks
                if approach_failure:
                    stage_failures.append(approach_failure)
            if task_approach_only:
                entry["feasibility_scope"] = "open_gripper_pregrasp_to_grasp_cartesian_path_only"
                entry["status"] = (
                    "offline_feasible_candidate" if not stage_failures else "rejected"
                )
                entry["safety_certified"] = False
                finish_entry()
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
            entry["target_layout_clearances_m"] = target_layout_clearances
            if not target_layout_safe:
                stage_failures.append("predicted_stable_target_pose_violates_forbidden_object_clearance")
            # The post-close pose above is a layout prior only. Restore the
            # target to the active scene pose before checking pad contact at
            # this candidate's t0 grasp pose; otherwise a historical lifted
            # object is tested against a gripper that remains at t0.
            set_target_pose(object_body_transform)
            # The pregrasp-to-target path above is intentionally checked with
            # the live OPEN posture. At the target, hold the arm at its closed-
            # gripper IK and sweep only scratch gripper qpos through the full
            # in-range close path, checking every robot/scene geometry pair.
            set_arm(grasp_ik[1])
            set_gripper(open_gripper)
            close_path_checks = []
            close_failure = None
            for sample in close_samples:
                set_gripper(sample["qpos"])
                check = evaluate(grasp_ik[1], allow_target_gripper=True)
                side_distances = fingertip_distances()
                jaw_width = pad_opening_width()
                close_path_checks.append({
                    "alpha": sample["alpha"],
                    "gripper_qpos": sample["qpos"],
                    "joint_limits_valid": True,
                    "fingertip_to_book_distance_m": side_distances,
                    "pad_to_pad_opening_m": jaw_width,
                    "geometry_check": check,
                })
                excessive_finger_penetration = any(
                    value is not None and value < -0.002
                    for value in side_distances.values()
                )
                if check["status"] != "pass" or excessive_finger_penetration:
                    if close_failure is None:
                        close_failure = "gripper_close_path_collision_or_contact_semantics_failed"
            final_close = close_path_checks[-1] if close_path_checks else {}
            side_distances = final_close.get("fingertip_to_book_distance_m", {})
            grasp_check = final_close.get("geometry_check", {"status": "incomplete"})
            bilateral_result = _bilateral_finger_contact_result(side_distances)
            opening_values = [item.get("pad_to_pad_opening_m") for item in close_path_checks]
            entry["gripper_close_path_checks"] = close_path_checks
            entry["finger_target_distance_m"] = side_distances
            entry["finger_distance_status"] = "measured_in_closed_gripper_scratch_state"
            entry["bilateral_fingertip_contact"] = bilateral_result
            entry["gripper_opening_range"] = {
                "open_joint_qpos": open_gripper.tolist(),
                "closed_joint_qpos": candidate_closed_gripper.tolist(),
                "open_pad_gap_m": opening_values[0] if opening_values else None,
                "closed_pad_gap_m": opening_values[-1] if opening_values else None,
                "pad_gap_change_m": (
                    opening_values[0] - opening_values[-1]
                    if len(opening_values) >= 2
                    and opening_values[0] is not None
                    and opening_values[-1] is not None
                    else None
                ),
                "sample_count": len(close_path_checks),
                "all_joint_limits_valid": all(item["joint_limits_valid"] for item in close_path_checks),
            }
            entry["grasp_check"] = grasp_check
            if (
                len(opening_values) != len(close_samples)
                or any(value is None for value in opening_values)
                or any(
                    float(opening_values[index + 1]) > float(opening_values[index]) + 1e-5
                    for index in range(max(0, len(opening_values) - 1))
                )
            ):
                stage_failures.append("gripper_opening_range_or_close_sweep_invalid")
            if close_failure:
                stage_failures.append(close_failure)
            if not bilateral_result["pass"]:
                stage_failures.append("bilateral_fingertip_contact_geometry_failed")
            if grasp_check["status"] != "pass":
                stage_failures.append("closed_grasp_full_geometry_or_contact_semantics_failed")
            if stage_failures:
                entry["failure_reasons"] = stage_failures
                entry["reason"] = stage_failures[0]
                finish_entry()
                continue

            # Candidate carried transform follows the candidate grasp, then the
            # exact Cartesian lift/exit geometry used by the existing controller.
            set_arm(grasp_ik[1])
            eef_r = np.asarray(scratch.site_xmat[eef_site_id]).reshape(3, 3)
            world_eef_grasp = np.eye(4); world_eef_grasp[:3, :3] = eef_r; world_eef_grasp[:3, 3] = scratch.site_xpos[eef_site_id]
            eef_to_object = _candidate_eef_to_live_object_transform(
                world_eef_grasp, object_body_transform
            )
            lift_positions = [position + np.asarray([0.0, 0.0, 0.030]), position + np.asarray([0.0, 0.0, 0.080])]
            held_q = [grasp_ik[1]]
            held_checks = []
            previous = grasp_ik[1]
            failed = None
            for lift_index, lift_position in enumerate(lift_positions):
                lift_ik = solve_ik(lift_position, quaternion, [previous, *(seeds or [])])
                if lift_ik is None:
                    failed = "lift_or_exit_ik_not_found_within_joint_limits"
                    entry["joint_reachability"]["lift30" if lift_index == 0 else "exit80"] = {"reachable": False}
                    break
                entry["joint_reachability"]["lift30" if lift_index == 0 else "exit80"] = {
                    "reachable": True,
                    "qpos": lift_ik[1].tolist(),
                    "position_error_m": lift_ik[2],
                    "rotation_error_rad": lift_ik[3],
                }
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
                finish_entry()
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
                "bilateral_fingertip_contact": bilateral_result,
                "gripper_close_path_checks": close_path_checks,
                "closed_grasp_target_pose_world_source": "active_scene_target_pose_at_candidate_close",
                "candidate_eef_to_object_transform_source": "active_scene_target_pose_at_candidate_close",
                "gripper_opening_range": entry["gripper_opening_range"],
                "pregrasp_check": pre_check,
                "grasp_check": grasp_check,
                "approach_path_checks": approach_checks,
                "predicted_stable_target_pose_clearance_m": target_layout_clearances,
                "predicted_stable_object_pose_source": "historical_t141_close_result_used_for_target_layout_prior_only",
                "held_path_checks": held_checks,
                "minimum_observed_clearance_m": min((item["minimum_distance_m"] for item in [pre_check, grasp_check, *held_checks] if item.get("minimum_distance_m") is not None), default=None),
                "preferred_ik_seed_qpos": pre_ik[1].tolist(),
            })
        except Exception as exc:
            entry.update({"reason": f"{type(exc).__name__}:{exc}"})
        finish_entry()
    live_after = np.zeros_like(integration)
    mujoco.mj_getState(model, live, live_after, integration_spec)
    unchanged = np.array_equal(integration, live_after)
    feasible = [item for item in results if item["status"] == "offline_feasible_candidate"]
    feasible.sort(key=lambda item: -float(
        item.get("minimum_forbidden_clearance_m")
        if item.get("minimum_forbidden_clearance_m") is not None else -1.0
    ))
    failure_counts = {}
    failure_membership_counts = {}
    for item in results:
        failure_class = item.get("failure_class") or "pass"
        failure_counts[failure_class] = failure_counts.get(failure_class, 0) + 1
        for member_class in item.get("failure_classes", []):
            failure_membership_counts[member_class] = failure_membership_counts.get(member_class, 0) + 1
    return {
        "operation": "screen_grasp_candidates",
        "status": "complete" if unchanged else "unsafe",
        "screening_type": (
            "offline_scratch_task_approach_feasibility_not_safety_certification"
            if task_approach_only else "offline_scratch_only_not_physical_execution"
        ),
        "task_approach_only": task_approach_only,
        "target_contact_policy": {
            "mode": (
                "pad_only_zero_penetration_task_feasibility"
                if task_approach_only
                else "pad_only_with_nonpad_clearance"
                if strict_target_contact_policy
                else "legacy_all_finger_geometries"
            ),
            "strict_pad_only": bool(strict_target_contact_policy),
            "task_approach_pad_only": task_approach_only,
            "allowed_target_contact_geom_ids": sorted(
                int(value) for value in target_contact_geom_ids
            ),
            "minimum_nonpad_robot_to_target_clearance_m": (
                float(clearance_m) if strict_target_contact_policy else None
            ),
            "maximum_authorized_pad_penetration_m": (
                -0.002 if strict_target_contact_policy else None
            ),
        },
        "object_pose_alignment": {
            "close_contact_object_pose_source": "active_scene_target_pose_at_candidate_close",
            "historical_post_close_pose_use": "target_layout_clearance_prior_only",
            "held_object_transform_source": "candidate_grasp_eef_relative_to_active_scene_target_pose",
        },
        "candidate_count": len(candidates),
        "feasible_count": len(feasible),
        "failure_class_counts": failure_counts,
        "failure_class_membership_counts": failure_membership_counts,
        "gripper_close_sweep_sample_count": 9,
        "gripper_open_joint_qpos": open_gripper.tolist(),
        "gripper_open_pad_gap_m": float(active_open_pad_gap_m),
        "gripper_open_joint_separation_m": active_open_joint_separation_m,
        "pad_gap_joint_offset_m": pad_gap_joint_offset_m,
        "historical_fallback_closed_joint_qpos": closed_gripper.tolist(),
        "candidate_closed_joint_qpos_source": "candidate_physical_pad_width_calibrated_from_active_open_pad_gap_when_present_otherwise_historical_fallback",
        "candidates": results,
        "ranked_candidate_ids": [item["candidate_id"] for item in feasible],
        "live_state_unchanged": unchanged,
        "start_state_sha256": state_hash,
    }
