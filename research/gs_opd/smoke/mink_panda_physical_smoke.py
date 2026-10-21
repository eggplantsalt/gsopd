# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Generate joint paths with Mink and execute them in isolated real scene physics.

No Bridge or Student is started. qpos is set only to restore the archived initial
state or for explicitly kinematic planning scratch data, never to execute a path.
"""

# Repository imports follow the explicit standalone-script sys.path bootstrap.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import mink
import mujoco
import numpy as np
from mink.limits.limit import Constraint, Limit

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gs_opd.smoke.contact_safety_compiler_smoke import compile_forbid_contact
from research.gs_opd.smoke.mink_panda_model import (
    ARM_VELOCITY_LIMITS_RAD_S,
    DEFAULT_ARCHIVE,
    DEFAULT_LIBERO_CONFIG,
    DEFAULT_LIBERO_PYTHON,
    DEFAULT_LIBERO_ROOT,
    FRAME_NAME,
    _compile_real_model,
    _make_mink_limits,
    _quat_angle,
    _validate_model_and_initial_state,
    quaternion_multiply_wxyz,
    quaternion_rotate_wxyz,
    select_archived_routes,
)

DT = 0.05
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


class JointReferenceSlewLimit(Limit):
    """Conservative reference acceleration budget, not a relaxed robot limit."""

    def __init__(self, model, dofs, previous_velocity, acceleration):
        self.projection = np.eye(model.nv)[dofs]
        self.previous = np.asarray(previous_velocity, dtype=float).copy()
        self.acceleration = acceleration

    def compute_qp_inequalities(self, configuration, dt):
        del configuration
        lower = self.previous - self.acceleration * dt
        upper = self.previous + self.acceleration * dt
        return Constraint(
            G=np.vstack([self.projection, -self.projection]),
            h=np.r_[upper * dt, -lower * dt],
        )


def state_vector(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE))
    mujoco.mj_getState(model, data, state, STATE)
    return state


def restore(model, state):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, STATE)
    mujoco.mj_forward(model, data)
    # forward changes derived/warm-start fields: restore exact integration data.
    mujoco.mj_setState(model, data, state, STATE)
    return data


def pose(data, site):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, data.site_xmat[site])
    return data.site_xpos[site].copy(), q


def inverse_quat(q):
    return np.asarray(q) * np.array([1.0, -1.0, -1.0, -1.0])


def resample_guide(points, speed):
    points = np.asarray(points, dtype=float)
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    count = max(1, math.ceil(float(distances[-1]) / (speed * DT)))
    targets = np.linspace(0.0, distances[-1], count + 1)
    return np.column_stack(
        [np.interp(targets, distances, points[:, k]) for k in range(3)]
    )


def assess_bypass_intent(points, start, obstacle, side_direction):
    """Check crossing the obstacle plane on the requested side; not collision proof."""
    points = np.asarray(points, dtype=float)
    start = np.asarray(start, dtype=float)
    obstacle = np.asarray(obstacle, dtype=float)
    side = np.asarray(side_direction, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 2
        or any(v.shape != (3,) for v in (start, obstacle, side))
        or not all(np.isfinite(v).all() for v in (points, start, obstacle, side))
        or np.linalg.norm(obstacle - start) < 1e-8
        or np.linalg.norm(side) < 1e-8
    ):
        return {"status": "unavailable", "reason": "degenerate_bypass_reference"}
    forward = (obstacle - start) / np.linalg.norm(obstacle - start)
    side = side / np.linalg.norm(side)
    progress = (points - obstacle) @ forward
    crossings = np.where((progress[:-1] <= 0.0) & (progress[1:] >= 0.0))[0]
    if not len(crossings) or progress[-1] <= 0.0:
        return {
            "status": "fail",
            "reason": "did_not_reach_opposite_obstacle_side",
            "endpoint_forward_m": float(progress[-1]),
        }
    i = int(crossings[0])
    weight = (
        float(-progress[i] / (progress[i + 1] - progress[i]))
        if progress[i + 1] != progress[i]
        else 0.0
    )
    crossing = points[i] * (1.0 - weight) + points[i + 1] * weight
    projection = float((crossing - obstacle) @ side)
    return {
        "status": "pass" if projection > 1e-6 else "fail",
        "reason": None
        if projection > 1e-6
        else "wrong_or_degenerate_bypass_family_side",
        "crossing_world_m": crossing.tolist(),
        "family_side_projection_m": projection,
        "endpoint_forward_m": float(progress[-1]),
        "collision_safety_proof": False,
    }


def classify_contact(g1, g2, ctx):
    """Conservative experiment policy in addition to the unchanged Compiler."""
    pair = {int(g1), int(g2)}
    a, b = int(ctx["model"].geom_bodyid[g1]), int(ctx["model"].geom_bodyid[g2])
    robot = ctx["robot_bodies"]
    if pair & ctx["carried_geoms"] and pair & ctx["gripper_geoms"]:
        return "allowed_teacher_grasp"
    if a in robot and b in robot:
        return "robot_self_collision"
    if a in robot or b in robot:
        return "robot_scene_collision"
    if pair & ctx["carried_geoms"] and pair & ctx["obstacle_geoms"]:
        return "teacher_forbidden_payload_obstacle"
    if pair & ctx["carried_geoms"]:
        return "unplanned_payload_support_or_other_contact"
    return "other_scene_support"


def scan_contacts(model, data, ctx, control_step=0, substep=0):
    records = []
    sides = {"left": False, "right": False}
    for contact_index in range(int(data.ncon)):
        c = data.contact[contact_index]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 < 0 or g2 < 0:
            raise ValueError("unsupported_non_geom_contact")
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_index, force)
        if not np.isfinite(force).all():
            raise ValueError("nonfinite_contact_force")
        classification = classify_contact(g1, g2, ctx)
        pair = {g1, g2}
        if pair & ctx["carried_geoms"]:
            for side in sides:
                sides[side] |= bool(pair & ctx["fingers"][side])
        records.append(
            {
                "geom_pair": [
                    {
                        "id": g,
                        "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
                    }
                    for g in (g1, g2)
                ],
                "body_pair": [
                    {
                        "id": int(model.geom_bodyid[g]),
                        "name": mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])
                        ),
                    }
                    for g in (g1, g2)
                ],
                "classification": (
                    "robot_robot"
                    if int(model.geom_bodyid[g1]) in ctx["robot_bodies"]
                    and int(model.geom_bodyid[g2]) in ctx["robot_bodies"]
                    else "robot_scene"
                    if pair & ctx["robot_geoms"]
                    else "scene_scene"
                ),
                "experiment_classification": classification,
                "robot_involved": bool(pair & ctx["robot_geoms"]),
                "first_control_step": control_step,
                "first_physics_substep": substep,
                "first_candidate_physics_substep": control_step * ctx["substeps"]
                + substep,
                "first_contact_position_world": c.pos.tolist(),
                "first_contact_normal_geom1_to_geom2_world": c.frame[:3].tolist(),
                "contact_position_status": "available",
                "contact_normal_status": "available",
                "contact_normal_geom_ids": [g1, g2],
                "first_contact_sample_recorded": True,
                "max_force_norm": float(np.linalg.norm(force[:3])),
                "force_sample_count": 1,
                "force_unavailable_count": 0,
                "occurrence_count": 1,
                "contact_point_count": 1,
            }
        )
    return records, sides


def dangerous(records):
    return next(
        (
            r
            for r in records
            if r["experiment_classification"]
            not in ("allowed_teacher_grasp", "other_scene_support")
        ),
        None,
    )


def make_context(model, archive, metadata):
    names = archive["active_osc_configuration"]["robots"][0]["joint_names"]
    joint_ids = [model.joint(n).id for n in names]
    qidx = np.array([int(model.jnt_qposadr[j]) for j in joint_ids])
    didx = np.array([int(model.jnt_dofadr[j]) for j in joint_ids])
    actuator_ids = []
    for joint in joint_ids:
        matches = [
            a for a in range(model.nu) if int(model.actuator_trnid[a, 0]) == joint
        ]
        if len(matches) != 1:
            raise ValueError("arm_actuator_mapping_ambiguous")
        a = matches[0]
        if (
            int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT)
            or int(model.actuator_biastype[a]) != 0
            or not np.allclose(model.actuator_gainprm[a, :3], [1.0, 0.0, 0.0])
            or not np.allclose(model.actuator_gear[a], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ):
            raise ValueError("arm_actuator_not_verified_unit_gear_torque_motor")
        actuator_ids.append(a)
    inventory = archive["geometry_inventory"]
    for item in inventory["geometries"]:
        g = item["geom_id"]
        if (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) != item["geom_name"]
            or int(model.geom_bodyid[g]) != item["body_id"]
        ):
            raise ValueError("current_geometry_inventory_identity_mismatch")
    robot_bodies = set(archive["gripper_geometry_inventory"]["robot_body_ids"])
    robot_geoms = {
        g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in robot_bodies
    }
    carried = archive["scene"]["carried"]
    carried_id = carried["body_id"]
    j = int(model.body_jntadr[carried_id])
    if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("payload_is_not_a_verified_single_free_body")
    fingers = {}
    for side, key in [("left", "left_finger"), ("right", "right_finger")]:
        fingers[side] = set()
        for item in metadata["finger_geom_groups"][key]:
            if model.geom(item["name"]).id != item["id"]:
                raise ValueError("live_finger_geom_identity_mismatch")
            fingers[side].add(item["id"])
    teacher = archive["teacher_source"]["safety_ir_v2"]
    if not any(
        r.get("decision") == "allow"
        and r.get("target_id") == carried["object_id"]
        and r.get("subject_parts") == ["gripper"]
        for r in teacher["rules"]
    ):
        raise ValueError("no_real_teacher_allow_grasp_rule")
    expected_forbid = {o["object_id"] for o in archive["scene"]["obstacles"]}
    if not expected_forbid or not all(
        any(
            r["subject_id"] == carried["object_id"]
            and r["target_id"] == o
            and r["decision"] == "forbid"
            for r in teacher.get("object_contact_rules", [])
        )
        for o in expected_forbid
    ):
        raise ValueError("bound_obstacle_forbid_rules_incomplete")
    substeps = round(DT / model.opt.timestep)
    if abs(substeps * model.opt.timestep - DT) > 1e-9:
        raise ValueError("control_physics_timestep_not_integral")
    ctx = {
        "model": model,
        "names": names,
        "joint_ids": joint_ids,
        "qidx": qidx,
        "didx": didx,
        "actuators": np.array(actuator_ids),
        "robot_bodies": robot_bodies,
        "robot_geoms": robot_geoms,
        "carried": carried,
        "carried_id": carried_id,
        "carried_qadr": int(model.jnt_qposadr[j]),
        "carried_geoms": set(carried["geom_ids"]),
        "obstacle_geoms": {
            g for o in archive["scene"]["obstacles"] for g in o["geom_ids"]
        },
        "gripper_geoms": {
            g["geom_id"]
            for gr in archive["gripper_geometry_inventory"]["grippers"]
            for g in gr["geometries"]
        },
        "fingers": fingers,
        "substeps": substeps,
        "site": model.site(FRAME_NAME).id,
        "joint_ranges": model.jnt_range[joint_ids].copy(),
        "vmax": np.array(ARM_VELOCITY_LIMITS_RAD_S),
    }
    return ctx


def collision_pair_pool(model, ctx):
    active = [
        g
        for g in range(model.ngeom)
        if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    robot = sorted(ctx["robot_geoms"] & set(active))
    scene = sorted(set(active) - ctx["robot_geoms"] - ctx["carried_geoms"])
    limit = mink.CollisionAvoidanceLimit(model, [(robot, robot), (robot, scene)])
    excludes = set(map(int, model.exclude_signature))
    pairs = set()
    for a, b in limit.geom_id_pairs:
        bodya, bodyb = sorted((int(model.geom_bodyid[a]), int(model.geom_bodyid[b])))
        if ((bodya << 16) + bodyb) not in excludes:
            pairs.add((a, b))
    return np.asarray(sorted(pairs), dtype=int).reshape(-1, 2)


def active_collision_limit(model, data, pairs):
    a, b = pairs[:, 0], pairs[:, 1]
    bound = (
        np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=1)
        - model.geom_rbound[a]
        - model.geom_rbound[b]
    )
    plane = (model.geom_type[a] == mujoco.mjtGeom.mjGEOM_PLANE) | (
        model.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE
    )
    near = pairs[(bound < 0.04) | plane]
    return mink.CollisionAvoidanceLimit(
        model,
        [([int(a)], [int(b)]) for a, b in near],
        minimum_distance_from_collisions=0.002,
        collision_detection_distance=0.03,
    )


def attached_scratch(model, config, ctx, offset, relquat, sphere_local):
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = config.q
    eefpos, eefquat = pose(config.data, ctx["site"])
    bodyquat = quaternion_multiply_wxyz(eefquat, inverse_quat(relquat))
    center = eefpos - quaternion_rotate_wxyz(bodyquat, offset)
    root = center - quaternion_rotate_wxyz(bodyquat, sphere_local)
    adr = ctx["carried_qadr"]
    scratch.qpos[adr : adr + 7] = np.r_[root, bodyquat]
    mujoco.mj_forward(model, scratch)
    return scratch, center


def plan_route(model, initial, route, ctx, args):
    guides = resample_guide(route["reference_path"], args.speed)
    result = {
        "status": "planning",
        "candidate_id": route["candidate_id"],
        "direction": route["direction"],
        "source_guide_world_m": guides.tolist(),
        "guide_world_m": [guides[0].tolist()],
        "requested_guide_steps": len(guides) - 1,
        "joint_trajectory_rad": [],
        "eef_trajectory_world_m": [],
        "eef_quaternion_world_wxyz": [],
        "sphere_trajectory_world_m": [],
        "failure_reason": None,
        "first_failure": None,
        "guide_reference_speed_m_s": args.speed,
        "time_step_s": DT,
        "collision_constraints": "current model robot-scene/self; attached payload exact-Geom postcheck",
    }
    data = restore(model, initial)
    primary = None
    for obstacle in ctx["obstacle_definitions"]:
        if obstacle["object_id"] == ctx["primary_obstacle_id"]:
            primary = obstacle
            break
    if primary is None:
        raise ValueError("primary_bound_obstacle_missing")
    result["bypass_reference"] = {
        "start_world_m": route["reference_path"][0],
        "obstacle_center_world_m": data.xpos[primary["body_id"]].tolist(),
        "side_direction_world": route["geometric_candidate"]["direction_world"],
        "obstacle_object_id": primary["object_id"],
        "goal_world_m": route["reference_path"][-1],
    }
    config = mink.Configuration(model, q=data.qpos.copy())
    q0 = data.qpos.copy()
    initial_grasp = route["grasp_evidence"]
    offset = np.asarray(
        initial_grasp["initial_eef_sphere_center_relative_position_body_local_m"]
    )
    relquat = np.asarray(
        initial_grasp["initial_eef_body_relative_quaternion_local_wxyz"]
    )
    sphere_local = np.asarray(route["sphere_center_body_local_offset_m"])
    bodyquat = data.xquat[ctx["carried_id"]].copy()
    frame = mink.FrameTask(
        FRAME_NAME,
        "site",
        position_cost=1.0,
        orientation_cost=args.orientation_cost,
        gain=1.0,
        lm_damping=1e-5,
    )
    posture = mink.PostureTask(model, cost=0.001, gain=0.2)
    posture.set_target(q0)
    limits = _make_mink_limits(mink, model, ctx["didx"].tolist(), ctx["names"])
    limits[1] = mink.VelocityLimit(
        model,
        dict(zip(ctx["names"], np.minimum(ctx["vmax"], args.joint_reference_speed))),
    )
    slew = JointReferenceSlewLimit(
        model, ctx["didx"], data.qvel[ctx["didx"]], args.joint_reference_acceleration
    )
    pool = collision_pair_pool(model, ctx)
    result["potential_robot_collision_pair_count"] = len(pool)
    result["joint_trajectory_rad"].append(config.q[ctx["qidx"]].tolist())
    initial_eef_pos, initial_eef_quat = pose(config.data, ctx["site"])
    result["eef_trajectory_world_m"].append(initial_eef_pos.tolist())
    result["eef_quaternion_world_wxyz"].append(initial_eef_quat.tolist())
    result["sphere_trajectory_world_m"].append(guides[0].tolist())
    if len(guides) > args.max_steps + 1:
        result.update(status="rejected", failure_reason="planning_step_budget_exceeded")
        return result
    max_pos_error = 0.0
    max_ori_error = 0.0
    index = 1
    for planned_step in range(args.max_steps):
        guide = guides[index]
        bad = None
        pos_error = None
        ori_error = None
        target = guide + quaternion_rotate_wxyz(bodyquat, offset)
        targetquat = quaternion_multiply_wxyz(bodyquat, relquat)
        frame.set_target(
            mink.SE3.from_rotation_and_translation(mink.SO3(targetquat), target)
        )
        try:
            coll = active_collision_limit(model, config.data, pool)
            v = mink.solve_ik(
                config,
                [frame, posture],
                DT,
                "quadprog",
                damping=1e-7,
                safety_break=True,
                limits=[*limits, coll, slew],
            )
            if not np.isfinite(v).all() or np.any(
                abs(v[ctx["didx"]]) > ctx["vmax"] + 1e-7
            ):
                raise ValueError("planned_joint_velocity_limit")
            config.integrate_inplace(v, DT)
            slew.previous = v[ctx["didx"]].copy()
            scratch, center = attached_scratch(
                model, config, ctx, offset, relquat, sphere_local
            )
            records, _ = scan_contacts(model, scratch, ctx)
            bad = dangerous(records)
            if bad:
                raise ValueError("planned_" + bad["experiment_classification"])
            pos_error = float(np.linalg.norm(center - guide))
            _, eq = pose(config.data, ctx["site"])
            ori_error = _quat_angle(eq, targetquat)
            if pos_error > args.guide_tolerance or ori_error > 0.20:
                raise ValueError("guided_IK_tracking_error")
            max_pos_error = max(max_pos_error, pos_error)
            max_ori_error = max(max_ori_error, ori_error)
            result["joint_trajectory_rad"].append(config.q[ctx["qidx"]].tolist())
            eef_position, eef_quaternion = pose(config.data, ctx["site"])
            result["eef_trajectory_world_m"].append(eef_position.tolist())
            result["eef_quaternion_world_wxyz"].append(eef_quaternion.tolist())
            result["sphere_trajectory_world_m"].append(center.tolist())
            result["guide_world_m"].append(guide.tolist())
            if pos_error <= 0.008 or (
                index == len(guides) - 1 and pos_error <= args.guide_tolerance
            ):
                index += 1
            if index == len(guides):
                result["status"] = "complete"
                break
        except Exception as error:
            result.update(
                status="rejected",
                failure_reason=str(error),
                first_failure={
                    "guide_index": index,
                    "guide_world_m": guide.tolist(),
                    "position_error_m": pos_error,
                    "orientation_error_rad": ori_error,
                    "contact": bad,
                },
            )
            break
    else:
        result.update(
            status="rejected",
            failure_reason="bounded_adaptive_time_parameterization_exhausted",
            first_failure={"guide_index": index},
        )
    result["planned_control_steps"] = len(result["joint_trajectory_rad"]) - 1
    result["reference_joint_velocity_budget_rad_s"] = args.joint_reference_speed
    result["reference_joint_acceleration_budget_rad_s2"] = (
        args.joint_reference_acceleration
    )
    result["maximum_guide_error_m"] = max_pos_error
    result["maximum_eef_orientation_error_rad"] = max_ori_error
    q = np.asarray(result["joint_trajectory_rad"])
    margins = np.minimum(q - ctx["joint_ranges"][:, 0], ctx["joint_ranges"][:, 1] - q)
    result["minimum_joint_margins_rad"] = dict(
        zip(ctx["names"], margins.min(axis=0).tolist())
    )
    result["maximum_joint_velocity_rad_s"] = dict(
        zip(
            ctx["names"],
            (
                np.abs(np.diff(q, axis=0)).max(axis=0) / DT
                if len(q) > 1
                else np.zeros(7)
            ).tolist(),
        )
    )
    ref = result["bypass_reference"]
    result["bypass_intent"] = assess_bypass_intent(
        result["sphere_trajectory_world_m"],
        ref["start_world_m"],
        ref["obstacle_center_world_m"],
        ref["side_direction_world"],
    )
    if result["status"] == "complete" and result["bypass_intent"]["status"] != "pass":
        result.update(
            status="rejected", failure_reason="planned_bypass_side_or_exit_failed"
        )
    return result


def torque_command(model, data, ctx, qtarget, vtarget, kp=50.0):
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    error = qtarget - data.qpos[ctx["qidx"]]
    acceleration = kp * error + 2.0 * math.sqrt(kp) * (vtarget - data.qvel[ctx["didx"]])
    torque = (
        mass[np.ix_(ctx["didx"], ctx["didx"])] @ acceleration
        + data.qfrc_bias[ctx["didx"]]
    )
    bounds = model.actuator_ctrlrange[ctx["actuators"]]
    if (
        not np.isfinite(torque).all()
        or np.any(torque < bounds[:, 0])
        or np.any(torque > bounds[:, 1])
    ):
        raise ValueError(
            "computed_torque_outside_actual_motor_limits:"
            + json.dumps(
                {"requested": torque.tolist(), "actual_bounds": bounds.tolist()}
            )
        )
    return torque


def aggregate_contacts(events, records):
    seen = set()
    for record in records:
        key = tuple(g["id"] for g in record["geom_pair"])
        if key not in events:
            events[key] = dict(record)
            events[key]["occurrence_count"] = 0
            events[key]["contact_point_count"] = 0
            events[key]["force_sample_count"] = 0
        event = events[key]
        if key not in seen:
            event["occurrence_count"] += 1
            seen.add(key)
        event["contact_point_count"] += 1
        event["force_sample_count"] += 1
        event["max_force_norm"] = max(event["max_force_norm"], record["max_force_norm"])


def execute_route(model, initial, route, plan, ctx, archive, args):
    result = {
        "candidate_id": route["candidate_id"],
        "status": "rejected",
        "requested_steps": len(plan["joint_trajectory_rad"]) - 1,
        "executed_steps": 0,
        "executed_physics_substeps": 0,
        "steps": [],
        "failure_reason": None,
        "execution_complete": False,
        "physical_safety_verified": False,
        "executable": False,
        "controller": "isolated mass-matrix joint PD + desired velocity + qfrc_bias; actual torque motors",
        "no_physical_execution_via_qpos_setting": True,
    }
    data = restore(model, initial)
    mujoco.mj_forward(model, data)
    initial_contacts, sides = scan_contacts(model, data, ctx)
    result["initial_finger_contacts"] = sides
    result["initial_gripper_ctrl"] = data.ctrl[
        [a for a in range(model.nu) if a not in ctx["actuators"]]
    ].tolist()
    if not all(sides.values()) or dangerous(initial_contacts):
        result["failure_reason"] = "current_initial_grasp_or_contact_check_failed"
        return result
    qs = np.asarray(plan["joint_trajectory_rad"])
    sphere_local = np.asarray(route["sphere_center_body_local_offset_m"])
    offset = np.asarray(
        route["grasp_evidence"][
            "initial_eef_sphere_center_relative_position_body_local_m"
        ]
    )
    relquat = np.asarray(
        route["grasp_evidence"]["initial_eef_body_relative_quaternion_local_wxyz"]
    )
    events = {}
    controls = []
    counts = Counter()
    failure = None
    physics_started = False
    try:
        for i in range(len(qs) - 1):
            samples = Counter()
            captured = 0
            velocity = (qs[i + 1] - qs[i]) / DT
            peak_ratio = 0.0
            last_records = []
            sides = {"left": False, "right": False}
            for sub in range(ctx["substeps"]):
                alpha = (sub + 1) / ctx["substeps"]
                target = qs[i] * (1 - alpha) + qs[i + 1] * alpha
                data.ctrl[ctx["actuators"]] = torque_command(
                    model, data, ctx, target, velocity
                )
                mujoco.mj_step(model, data)
                physics_started = True
                # Read contacts/forces from every actual physics step before synchronization.
                records, sides = scan_contacts(model, data, ctx, i, sub)
                aggregate_contacts(events, records)
                samples["contact" if records else "no_contact"] += 1
                captured += 1
                result["executed_physics_substeps"] += 1
                last_records = records
                bad = dangerous(records)
                q = data.qpos[ctx["qidx"]]
                v = data.qvel[ctx["didx"]]
                ratio = float(np.max(abs(v) / ctx["vmax"]))
                peak_ratio = max(peak_ratio, ratio)
                if bad:
                    failure = {
                        "reason": bad["experiment_classification"],
                        "control_step": i,
                        "physics_substep": sub,
                        "contact": bad,
                    }
                    break
                if (
                    not np.isfinite(data.qpos).all()
                    or not np.isfinite(data.qvel).all()
                    or np.any(q < ctx["joint_ranges"][:, 0])
                    or np.any(q > ctx["joint_ranges"][:, 1])
                ):
                    failure = {
                        "reason": "actual_joint_position_limit",
                        "control_step": i,
                        "physics_substep": sub,
                    }
                    break
                if ratio > 1.0 + 1e-6:
                    failure = {
                        "reason": "actual_joint_velocity_limit",
                        "control_step": i,
                        "physics_substep": sub,
                        "ratio": ratio,
                    }
                    break
            controls.append(
                {
                    "control_step_index": i,
                    "status": "complete"
                    if captured == ctx["substeps"]
                    else "incomplete",
                    "expected_physics_substeps": ctx["substeps"],
                    "captured_physics_substeps": captured,
                    "unobserved_expected_substeps": ctx["substeps"] - captured,
                    "contact_samples": {
                        k: samples[k] for k in ("contact", "no_contact", "unavailable")
                    },
                }
            )
            counts.update(samples)
            mujoco.mj_forward(model, data)
            eef, eq = pose(data, ctx["site"])
            bq = data.xquat[ctx["carried_id"]].copy()
            endpoint_contacts, sides = scan_contacts(
                model, data, ctx, i, ctx["substeps"] - 1
            )
            endpoint_bad = dangerous(endpoint_contacts)
            if endpoint_bad and failure is None:
                failure = {
                    "reason": endpoint_bad["experiment_classification"],
                    "control_step": i,
                    "contact": endpoint_bad,
                }
            center = data.xpos[ctx["carried_id"]] + quaternion_rotate_wxyz(
                bq, sphere_local
            )
            relpos = quaternion_rotate_wxyz(inverse_quat(bq), eef - center)
            relative_quat = quaternion_multiply_wxyz(inverse_quat(bq), eq)
            relative_error = float(np.linalg.norm(relpos - offset))
            relative_rotation_error = _quat_angle(relative_quat, relquat)
            row = {
                "step_index": i,
                "qpos_rad": data.qpos[ctx["qidx"]].tolist(),
                "qvel_rad_s": data.qvel[ctx["didx"]].tolist(),
                "target_qpos_rad": qs[i + 1].tolist(),
                "eef_world_m": eef.tolist(),
                "sphere_center_world_m": center.tolist(),
                "sphere_guide_error_m": float(
                    np.linalg.norm(center - np.asarray(plan["guide_world_m"][i + 1]))
                ),
                "eef_joint_plan_error_m": float(
                    np.linalg.norm(
                        eef - np.asarray(plan["eef_trajectory_world_m"][i + 1])
                    )
                ),
                "relative_position_error_m": relative_error,
                "relative_rotation_error_rad": relative_rotation_error,
                "eef_quaternion_world_wxyz": eq.tolist(),
                "eef_absolute_orientation_error_rad": _quat_angle(
                    eq, route["start_state"]["robot_states"][0]["eef_quat"]
                ),
                "object_root_quaternion_world_wxyz": bq.tolist(),
                "object_root_position_world_m": data.xpos[ctx["carried_id"]].tolist(),
                "finger_contacts": sides,
                "peak_joint_velocity_ratio": peak_ratio,
                "contacts_at_end": last_records,
            }
            result["steps"].append(row)
            if captured == ctx["substeps"]:
                result["executed_steps"] += 1
            if failure:
                break
            if not all(sides.values()):
                failure = {"reason": "opposing_finger_contacts_lost", "control_step": i}
                break
            if relative_error > 0.012 or relative_rotation_error > 0.20:
                failure = {
                    "reason": "payload_grasp_relative_pose_changed",
                    "control_step": i,
                }
                break
            if (
                row["eef_joint_plan_error_m"] > 0.025
                or row["sphere_guide_error_m"] > args.guide_tolerance
                or row["eef_absolute_orientation_error_rad"] > 0.20
            ):
                failure = {"reason": "physical_path_tracking_error", "control_step": i}
                break
            if (i + 1) % 100 == 0:
                print(
                    f"{route['candidate_id']}: physical control step {i + 1}/{len(qs) - 1}",
                    flush=True,
                )
        result["execution_complete"] = (
            failure is None and result["executed_steps"] == result["requested_steps"]
        )
        result["failure_reason"] = failure["reason"] if failure else None
        result["first_failure"] = failure
    except Exception as error:
        result["failure_reason"] = f"{type(error).__name__}:{error}"
    finally:
        final = state_vector(model, data)
        result["final_integration_state"] = final.tolist()
        restored = restore(model, initial)
        restore_error = float(np.max(np.abs(state_vector(model, restored) - initial)))
        result["isolated_snapshot_restore_max_abs_error"] = restore_error
        result["isolated_snapshot_restored"] = restore_error <= 1e-6
    monitor = {
        "status": "complete"
        if len(controls) == len(result["steps"])
        and all(c["status"] == "complete" for c in controls)
        and physics_started
        else "incomplete",
        "sampling_point": "immediately_after_each_successful_MjSim.step",
        "actual_sampling_api": "mujoco.mj_step on isolated MjData",
        "robot_classification_status": "available",
        "control_steps": controls,
        "expected_physics_substeps_total": sum(
            c["expected_physics_substeps"] for c in controls
        ),
        "captured_physics_substeps": result["executed_physics_substeps"],
        "unobserved_expected_substeps": sum(
            c["unobserved_expected_substeps"] for c in controls
        ),
        "contact_substeps": counts["contact"],
        "no_contact_substeps": counts["no_contact"],
        "unavailable_contact_substeps": 0,
        "events": list(events.values()),
    }
    result["contact_monitoring"] = monitor
    compiler_candidate = {
        **result,
        "status": "ok" if monitor["status"] == "complete" else "error",
    }
    branch = {
        "main_state_restored": result["isolated_snapshot_restored"],
        "candidates": [compiler_candidate],
    }
    # Compile all explicit object-object rules and the bound payload's gripper
    # rule. Other robot/scene contacts are already rejected conservatively.
    semantics = archive["semantic_binding_source"]["semantic_bindings"]
    compiler = compile_forbid_contact(
        archive["teacher_source"]["safety_ir_v2"],
        ctx["carried"]["object_id"],
        semantics[ctx["carried"]["object_id"]],
        branch,
        archive["geometry_inventory"],
        archive["gripper_geometry_inventory"],
        object_semantic_results=semantics,
    )
    verdicts = [c.get("verdict", "unknown") for c in compiler.get("candidates", [])]
    compiler["verdict"] = (
        "violation"
        if "violation" in verdicts
        else "safe"
        if verdicts and all(v == "safe" for v in verdicts)
        else "unknown"
    )
    result["contact_compiler"] = compiler
    result["contact_classification_counts"] = dict(
        Counter(e["experiment_classification"] for e in events.values())
    )
    result["snapshot_restore_scope"] = (
        "isolated simulator only; normal Bridge never connected"
    )
    result["task_reward_and_done"] = (
        "unavailable_in_raw_isolated_MuJoCo; no final task-success claim"
    )
    ref = plan["bypass_reference"]
    result["bypass_intent"] = assess_bypass_intent(
        [ref["start_world_m"]] + [s["sphere_center_world_m"] for s in result["steps"]],
        ref["start_world_m"],
        ref["obstacle_center_world_m"],
        ref["side_direction_world"],
    )
    if result["steps"]:
        result["endpoint_error_m"] = float(
            np.linalg.norm(
                np.asarray(result["steps"][-1]["sphere_center_world_m"])
                - ref["goal_world_m"]
            )
        )
        result["maximum_physical_metrics"] = {
            key: max(s[key] for s in result["steps"])
            for key in (
                "sphere_guide_error_m",
                "eef_joint_plan_error_m",
                "relative_position_error_m",
                "relative_rotation_error_rad",
                "peak_joint_velocity_ratio",
            )
        }
        actual_q = np.asarray([s["qpos_rad"] for s in result["steps"]])
        result["minimum_actual_joint_margins_rad"] = dict(
            zip(
                ctx["names"],
                np.minimum(
                    actual_q - ctx["joint_ranges"][:, 0],
                    ctx["joint_ranges"][:, 1] - actual_q,
                )
                .min(axis=0)
                .tolist(),
            )
        )
    if (
        result["execution_complete"]
        and result["isolated_snapshot_restored"]
        and compiler["verdict"] == "safe"
        and result["bypass_intent"]["status"] == "pass"
    ):
        result["status"] = "physical_segment_passed"
        result["physical_safety_verified"] = True
    result["actual_execution_via_mj_step"] = physics_started
    return result


def save_svg(path, plan, execution):
    groups = [
        ("guide", np.asarray(plan["guide_world_m"]), "#999"),
        ("Mink", np.asarray(plan["sphere_trajectory_world_m"]), "#1684c4"),
    ]
    actual = np.asarray(
        [s["sphere_center_world_m"] for s in execution.get("steps", [])]
    )
    if actual.size:
        groups.append(("physics", actual, "#d44"))
    chunks = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="440" viewBox="0 0 1200 440">',
        '<rect width="1200" height="440" fill="white"/>',
        '<text x="20" y="24">Gray: original guide; blue: Mink; red: actual mj_step execution (meters)</text>',
    ]
    for view, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
        allp = np.vstack([g[1] for g in groups if g[1].size])
        lo = allp[:, [a, b]].min(axis=0) - 0.03
        hi = allp[:, [a, b]].max(axis=0) + 0.03
        scale = 330.0 / max(float(np.max(hi - lo)), 0.1)
        for label, points, color in groups:
            xy = (points[:, [a, b]] - lo) * scale
            text = " ".join(f"{view * 400 + 25 + x:.2f},{400 - y:.2f}" for x, y in xy)
            chunks.append(
                f'<polyline points="{text}" fill="none" stroke="{color}" stroke-width="2"><title>{label}</title></polyline>'
            )
        chunks.append(
            f'<text x="{view * 400 + 25}" y="425">{"XYZ"[a]} / {"XYZ"[b]}</text>'
        )
    chunks.append("</svg>")
    path.write_text("\n".join(chunks))


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive = json.loads(Path(args.archive).read_text())
    routes = select_archived_routes(archive)
    with tempfile.TemporaryDirectory(prefix="gsopd_mink_physics_") as tmp:
        model_path = Path(tmp) / "scene.mjb"
        meta_path = Path(tmp) / "scene.json"
        metadata = _compile_real_model(
            DEFAULT_LIBERO_PYTHON,
            model_path,
            meta_path,
            "obstacle_avoidance",
            0,
            0,
            DEFAULT_LIBERO_ROOT,
            DEFAULT_LIBERO_CONFIG,
        )
        model = mujoco.MjModel.from_binary_path(str(model_path))
        compatibility = _validate_model_and_initial_state(
            mujoco, model, archive, metadata
        )
        blockers = [
            e
            for e in compatibility["errors"]
            if e != "compiled_model_signature_mismatch"
        ]
        if blockers:
            raise ValueError(
                "current_state_or_identity_incompatible:" + ",".join(blockers)
            )
        compatibility["current_model_execution_preflight_status"] = "pass"
        compatibility["strict_archive_identity_status"] = compatibility["status"]
        compatibility["model_signature_difference_policy"] = (
            "diagnostic only; current model joint, EEF, and Body/Geom checks passed"
        )
        ctx = make_context(model, archive, metadata)
        ctx["obstacle_definitions"] = archive["scene"]["obstacles"]
        ctx["primary_obstacle_id"] = archive["trajectory_bundle"]["primary_obstacle_id"]
        initial = np.asarray(
            archive["planning_geometry"]["mujoco_integration_state"], dtype=float
        )
        initial_data = restore(model, initial)
        mujoco.mj_forward(model, initial_data)
        initial_contacts, sides = scan_contacts(model, initial_data, ctx)
        report = {
            "operation": "mink_panda_isolated_physical_execution",
            "compatibility": compatibility,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "finger_geom_groups": metadata["finger_geom_groups"],
            "initial_finger_contacts": sides,
            "initial_contacts": initial_contacts,
            "model_signature_difference_is_diagnostic": True,
            "controller_source": "robosuite v1.4 joint_pos.py mass-matrix PD with explicit desired velocity feedforward",
            "normal_bridge_modified_or_connected": False,
            "routes": [],
            "training_label_created": False,
            "task_success_claim": False,
            "config": vars(args),
        }
        if not all(sides.values()) or dangerous(initial_contacts):
            report["status"] = "blocked_initial_grasp_or_collision"
        else:
            for route in routes:
                print(
                    f"Planning {route['candidate_id']} {route['direction']}", flush=True
                )
                plan = plan_route(model, initial, route, ctx, args)
                (out / f"{route['candidate_id']}_joint_plan.json").write_text(
                    json.dumps(plan, indent=2, allow_nan=False)
                )
                if plan["status"] == "complete":
                    print(
                        f"Executing isolated physics: {len(plan['joint_trajectory_rad']) - 1} control steps",
                        flush=True,
                    )
                    execution = execute_route(
                        model, initial, route, plan, ctx, archive, args
                    )
                else:
                    execution = {
                        "status": "not_executed",
                        "executed_steps": 0,
                        "failure_reason": "planning:" + str(plan["failure_reason"]),
                    }
                save_svg(out / f"{route['candidate_id']}.svg", plan, execution)
                entry = {
                    "candidate_id": route["candidate_id"],
                    "direction": route["direction"],
                    "plan_status": plan["status"],
                    "plan_file": f"{route['candidate_id']}_joint_plan.json",
                    "visualization": f"{route['candidate_id']}.svg",
                    "execution": execution,
                }
                (out / f"{route['candidate_id']}_physical.json").write_text(
                    json.dumps(entry, indent=2, allow_nan=False)
                )
                report["routes"].append(entry)
                print(
                    json.dumps(
                        {
                            "route": route["candidate_id"],
                            "plan": plan["status"],
                            "physical": execution["status"],
                            "executed_steps": execution["executed_steps"],
                            "reason": execution.get("failure_reason"),
                        }
                    ),
                    flush=True,
                )
                (out / "report.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False)
                )
            report["status"] = (
                "physical_segment_passed"
                if any(
                    r["execution"]["status"] == "physical_segment_passed"
                    for r in report["routes"]
                )
                else "no_complete_physical_candidate"
            )
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        return report


def run_live_plan(args):
    """Plan a current Bridge grasp route; never executes simulator actions."""
    from types import SimpleNamespace

    from research.gs_opd.smoke.mink_panda_model import compute_model_signature

    source = json.loads(args.live_plan_input.read_text(encoding="utf-8"))
    archive = source["planning_archive"]
    model = mujoco.MjModel.from_binary_path(str(args.model_binary))
    state = np.asarray(source["mujoco_integration_state"], dtype=np.float64)
    expected_state_size = int(mujoco.mj_stateSize(model, STATE))
    if state.shape != (expected_state_size,) or not np.isfinite(state).all():
        raise ValueError("live_Snapshot_integration_state_is_incompatible_with_Mink_model")
    initial = state.copy()
    data = restore(model, initial)
    ctx = make_context(model, archive, json.loads(args.model_metadata.read_text()))
    ctx["obstacle_definitions"] = archive["scene"]["obstacles"]
    ctx["primary_obstacle_id"] = archive["trajectory_bundle"]["primary_obstacle_id"]
    records, finger_sides = scan_contacts(model, data, ctx)
    plan_args = SimpleNamespace(
        speed=args.speed,
        max_steps=args.max_steps,
        guide_tolerance=args.guide_tolerance,
        orientation_cost=args.orientation_cost,
        joint_reference_speed=args.joint_reference_speed,
        joint_reference_acceleration=args.joint_reference_acceleration,
    )
    route = source["route"]
    report = {
        "operation": "live_snapshot_mink_left_route_plan",
        "status": "blocked",
        "snapshot_state_id": source.get("snapshot_state_id"),
        "model_geometry_signature": compute_model_signature(model),
        "model_fingerprint_comparison": source.get("model_fingerprint_comparison"),
        "mujoco_integration_state_size": expected_state_size,
        "initial_grasp_contacts": finger_sides,
        "initial_contacts": records,
        "mink_planning_performed": False,
        "physics_actions_executed": 0,
        "joint_plan_executed": False,
        "candidate_id": route.get("candidate_id"),
        "direction": route.get("direction"),
    }
    if not all(finger_sides.values()):
        report["failure_reason"] = "Mink_model_snapshot_lacks_opposing_finger_contacts"
    elif dangerous(records):
        report["failure_reason"] = (
            "Mink_model_snapshot_contains_forbidden_contact:"
            + dangerous(records)["experiment_classification"]
        )
    else:
        plan = plan_route(model, initial, route, ctx, plan_args)
        report["plan"] = plan
        report["mink_planning_performed"] = True
        report["status"] = plan.get("status")
        report["failure_reason"] = plan.get("failure_reason")
    args.live_plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.live_plan_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live-plan-input", type=Path)
    p.add_argument("--live-plan-output", type=Path)
    p.add_argument("--model-binary", type=Path)
    p.add_argument("--model-metadata", type=Path)
    p.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    p.add_argument(
        "--out-dir", default="/opt/data/private/gs_opd/rollouts/mink_panda_physical_v1"
    )
    p.add_argument("--speed", type=float, default=0.02)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--guide-tolerance", type=float, default=0.025)
    p.add_argument("--orientation-cost", type=float, default=0.1)
    p.add_argument("--joint-reference-speed", type=float, default=0.6)
    p.add_argument("--joint-reference-acceleration", type=float, default=2.0)
    args = p.parse_args()
    if (
        not 0.0 < args.speed <= 0.04
        or not 1 <= args.max_steps <= 2400
        or not 0.0 < args.guide_tolerance <= 0.05
        or not 0.0 < args.orientation_cost <= 1.0
        or not 0.0 < args.joint_reference_speed <= 1.0
        or not 0.0 < args.joint_reference_acceleration <= 5.0
    ):
        p.error("invalid bounded planning configuration")
    if args.live_plan_input is not None:
        if not all((args.live_plan_output, args.model_binary, args.model_metadata)):
            p.error("live planning requires --live-plan-output, --model-binary and --model-metadata")
        report = run_live_plan(args)
    else:
        report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(Path(args.out_dir) / "report.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
