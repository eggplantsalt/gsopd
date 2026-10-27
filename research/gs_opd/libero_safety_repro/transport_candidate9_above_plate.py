# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Resume OA/L0T0 candidate9 at its exact t=190 snapshot and transport above plate_1.

This is a bounded Task-Success feasibility experiment.  It never releases the
bowl or attempts placement.  Existing strict safety geometry is monitored and
reported separately; it is not relabeled as task success.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import pickle
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from research.gs_opd.libero_safety_repro import plan_oa_l0t0 as curobo_api
from research.gs_opd.libero_safety_repro.scene_probe import _read_only_rpc


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_SCENE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "gs_opd_libero_safety_repro_oa_l0t0_stage_b_20261027_attempt04/scene_source.json.gz"
)
DEFAULT_SOURCE_SNAPSHOT = Path(
    "/opt/data/private/gs_opd/rollouts/oa_l0t0_c09_feasible_lift_5sy0p49a/"
    "terminal_runtime_snapshot.pkl"
)
DEFAULT_OUTPUT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "oa_l0t0_candidate9_transport_above_plate_20261027"
)
EXPECTED_MODEL_SHA = "40fd3ac3f0efd60a4a3de827b45e7228f4e67c6238c86d7957586dc8edd408d2"
EXPECTED_BDDL_SHA = "b631d2f9be0884f5fb908abc0cf271e34a9c368085252dc48bbd5207d5b7fe12"
TARGET_OBJECT_ID = "akita_black_bowl_1"
TARGET_REGION_ID = "plate_1"
INSTRUCTION = "pick up the akita black bowl on the stove and place it on the plate"
EXPECTED_START_TIMESTEP = 190
MAX_PHYSICAL_ACTIONS = 180
MAX_ACTIONS_PER_WAYPOINT = 3
MAX_ENDPOINT_FEEDBACK_ACTIONS = 20
WAYPOINT_TOLERANCE_RAD = 0.03
HARD_TRACKING_STOP_RAD = 0.08
ENDPOINT_TOLERANCE_RAD = 0.005
ENDPOINT_VELOCITY_TOLERANCE_RAD_S = 0.03
MUJOCO_SUBSTEPS_PER_ACTION = 25
PLATE_ABOVE_CLEARANCE_ENGINEERING_CHOICE_M = 0.060


class _PythonObjectState:
    """Unpickle marker type used by old runtime-snapshot archives."""


class _Bridge:
    def __init__(self, connection):
        self.connection = connection

    def request(self, request: dict[str, Any]) -> dict[str, Any]:
        return curobo_api._rpc(self.connection, request)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    safe = curobo_api._json_safe(data)
    path.write_text(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _pose(position: Any, rotation: Any) -> np.ndarray:
    p = np.asarray(position, dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)
    if p.shape != (3,) or r.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("live_pose_must_be_finite_3d_position_and_3x3_rotation")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1.0, atol=1e-6):
        raise ValueError("live_pose_rotation_is_not_SO3")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = p
    return out


def _pose_from_wxyz(position: Any, quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError("live_pose_quaternion_must_be_finite_nonzero_wxyz")
    return _pose(position, Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix())


def _object_pose(context: dict[str, Any]) -> np.ndarray:
    row = context["target_root_pose_world"]
    return _pose(row["position_m"], row["rotation"])


def _frame_pose(row: dict[str, Any]) -> np.ndarray:
    return _pose(row["position_m"], row["rotation"])


def _box_corners(center: np.ndarray, rotation: np.ndarray, half: np.ndarray) -> np.ndarray:
    if center.shape != (3,) or rotation.shape != (3, 3) or half.shape != (3,) or np.any(half <= 0):
        raise ValueError("compiled_collision_cuboid_has_invalid_dimensions")
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    return center[None, :] + (signs * half[None, :]) @ rotation.T


def _plate_above_object_pose(
    context: dict[str, Any],
    semantic: dict[str, Any],
    collision_scene: dict[str, Any],
    world_base: np.ndarray,
    requested_clearance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not np.isfinite(requested_clearance_m) or requested_clearance_m <= 0.0:
        raise ValueError("above_plate_clearance_must_be_positive_and_finite")
    binding = semantic["bindings"].get(TARGET_REGION_ID)
    if not isinstance(binding, dict):
        raise ValueError("active_semantic_binding_for_plate_1_is_missing")
    body_id = int(binding["root_body_id"])
    body_name = str(binding["root_body_name"])
    rows = [
        row for row in collision_scene.get("scene_collision_geometries", [])
        if int(row.get("body_id", -1)) == body_id and row.get("body_name") == body_name
    ]
    if not rows:
        raise ValueError("bound_plate_body_has_no_compiled_collision_geometry")
    vertices = []
    used = []
    for row in rows:
        if row.get("shape") != "cuboid":
            raise ValueError(f"plate_collision_shape_not_supported_for_exact_extent:{row.get('shape')}")
        center = np.asarray(row["center_world_m"], dtype=np.float64)
        rotation = np.asarray(row["rotation_world"], dtype=np.float64)
        half = np.asarray(row["half_extents_world_axes_m"], dtype=np.float64)
        corners = _box_corners(center, rotation, half)
        vertices.extend(corners)
        used.append({
            "geom_id": int(row["geom_id"]),
            "geom_name": str(row["geom_name"]),
            "center_world_m": center.tolist(),
            "half_extents_local_axes_m": half.tolist(),
            "rotation_world": rotation.tolist(),
        })
    plate_vertices = np.asarray(vertices, dtype=np.float64)
    lower = np.min(plate_vertices, axis=0)
    upper = np.max(plate_vertices, axis=0)
    plate_xy_aabb_center = (lower[:2] + upper[:2]) * 0.5
    plate_top_world_z = float(np.max(plate_vertices[:, 2]))

    object_pose = _object_pose(context)
    object_rotation = object_pose[:3, :3]
    object_bottom_delta_world_z = np.inf
    bowl_world_offsets = []
    bowl_rows = context.get("target_collision_geometries", [])
    if not bowl_rows:
        raise ValueError("live_bowl_collision_geometry_is_empty")
    for row in bowl_rows:
        center_root = np.asarray(row["center_root_m"], dtype=np.float64)
        rotation_root = np.asarray(row["rotation_root"], dtype=np.float64)
        half = np.asarray(row["size_m"], dtype=np.float64)
        corners_root = _box_corners(center_root, rotation_root, half)
        delta_world_z = corners_root @ object_rotation[2, :]
        object_bottom_delta_world_z = min(object_bottom_delta_world_z, float(np.min(delta_world_z)))
        bowl_world_offsets.extend(corners_root @ object_rotation.T)
    if not np.isfinite(object_bottom_delta_world_z):
        raise ValueError("live_bowl_collision_geometry_has_no_finite_bottom")

    bowl_offsets = np.asarray(bowl_world_offsets, dtype=np.float64)
    bowl_xy_min = np.min(bowl_offsets[:, :2], axis=0)
    bowl_xy_max = np.max(bowl_offsets[:, :2], axis=0)
    base_to_plate_xy = np.asarray(world_base[:2, 3], dtype=np.float64) - plate_xy_aabb_center
    direction_norm = float(np.linalg.norm(base_to_plate_xy))
    if direction_norm <= 1e-9:
        raise ValueError("robot_base_and_plate_collision_aabb_have_ambiguous_xy_direction")
    direction_toward_robot = base_to_plate_xy / direction_norm
    maximum_fit_shift_m = np.inf
    for axis in range(2):
        component = float(direction_toward_robot[axis])
        if component > 1e-9:
            maximum_fit_shift_m = min(
                maximum_fit_shift_m,
                float((upper[axis] - plate_xy_aabb_center[axis] - bowl_xy_max[axis]) / component),
            )
        elif component < -1e-9:
            maximum_fit_shift_m = min(
                maximum_fit_shift_m,
                float((plate_xy_aabb_center[axis] + bowl_xy_min[axis] - lower[axis]) / -component),
            )
    if not np.isfinite(maximum_fit_shift_m) or maximum_fit_shift_m < 0.0:
        raise ValueError("bowl_collision_projection_does_not_fit_inside_plate_collision_aabb")
    # Engineering reach adjustment: shift toward the live robot base only
    # within the exact plate collision AABB, leaving 20% of the available
    # projected footprint slack. This selects an above-plate transit waypoint,
    # not a contact or placement pose.
    selected_fit_shift_m = 0.80 * maximum_fit_shift_m
    selected_xy = plate_xy_aabb_center + direction_toward_robot * selected_fit_shift_m
    goal = object_pose.copy()
    goal[:3, 3] = [
        float(selected_xy[0]),
        float(selected_xy[1]),
        float(plate_top_world_z + requested_clearance_m - object_bottom_delta_world_z),
    ]
    details = {
        "target_semantic_object_id": TARGET_REGION_ID,
        "target_bound_root_body_id": body_id,
        "target_bound_root_body_name": body_name,
        "plate_collision_geom_count": len(used),
        "plate_collision_geometries": used,
        "plate_collision_aabb_world_m": {"min": lower.tolist(), "max": upper.tolist()},
        "plate_xy_target_from_collision_aabb_center_world_m": plate_xy_aabb_center.tolist(),
        "live_robot_base_xy_world_m": world_base[:2, 3].tolist(),
        "reach_adjustment_direction_toward_live_robot_base_xy": direction_toward_robot.tolist(),
        "maximum_bowl_contained_shift_in_plate_collision_aabb_m": maximum_fit_shift_m,
        "selected_reach_adjustment_shift_m": selected_fit_shift_m,
        "selected_object_xy_world_m": selected_xy.tolist(),
        "plate_top_world_z_from_max_compiled_collision_vertex_m": plate_top_world_z,
        "bowl_bottom_offset_world_z_m_at_current_orientation": object_bottom_delta_world_z,
        "requested_bowl_bottom_clearance_above_plate_collision_m": requested_clearance_m,
        "clearance_is_reproduction_engineering_choice_not_a_paper_parameter": True,
        "target_pose_world_object_m": goal.tolist(),
        "method": "active BDDL semantic body binding plus exact current MuJoCo compiled collision cuboids; no BDDL Site lookup",
        "placement_or_release_requested": False,
    }
    return goal, details


def _collision_policy(context: dict[str, Any], semantic: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    target_binding = semantic["bindings"].get(TARGET_OBJECT_ID)
    if not isinstance(target_binding, dict):
        raise ValueError("active_semantic_binding_for_held_bowl_is_missing")
    target_body_ids = sorted(int(x) for x in target_binding.get("collision_body_ids", []))
    target_geom_ids = sorted(int(x) for x in target_binding.get("geom_ids", []))
    target_body_names = [str(x) for x in target_binding.get("collision_body_names", [])]
    if not target_body_ids or len(target_body_ids) != len(target_body_names) or not target_geom_ids:
        raise ValueError("active_held_bowl_collision_binding_is_incomplete")

    all_geoms = [
        row for row in inventory.get("geometries", [])
        if row.get("collision_enabled") is True and row.get("category") == "scene"
    ]
    if inventory.get("status") != "complete" or not all_geoms:
        raise ValueError("active_full_scene_collision_inventory_is_incomplete")
    bodies_by_id = {
        int(row["body_id"]): str(row["body_name"])
        for row in inventory.get("bodies", [])
        if isinstance(row.get("body_id"), int) and isinstance(row.get("body_name"), str)
    }
    collision_by_body: dict[int, set[int]] = {}
    for row in all_geoms:
        collision_by_body.setdefault(int(row["body_id"]), set()).add(int(row["geom_id"]))
    covered_bodies: set[int] = set()
    forbidden_objects: list[dict[str, Any]] = []
    other_scene_objects: list[dict[str, Any]] = []
    target_group = {
        "object_id": TARGET_OBJECT_ID,
        "body_ids": target_body_ids,
        "body_names": target_body_names,
        "geom_ids": target_geom_ids,
        "minimum_clearance_m": 0.002,
    }
    covered_bodies.update(target_body_ids)
    for object_id, binding in semantic.get("bindings", {}).items():
        if object_id == TARGET_OBJECT_ID:
            continue
        body_ids = sorted(int(x) for x in binding.get("collision_body_ids", []))
        body_names = [str(x) for x in binding.get("collision_body_names", [])]
        geom_ids = sorted(int(x) for x in binding.get("geom_ids", []))
        if not body_ids or not geom_ids:
            continue
        if len(body_ids) != len(body_names) or set(body_ids) & covered_bodies:
            raise ValueError(f"active_scene_semantic_body_binding_is_ambiguous:{object_id}")
        forbidden_objects.append({
            "object_id": str(object_id), "body_ids": body_ids,
            "body_names": body_names, "geom_ids": geom_ids,
            "minimum_clearance_m": 0.010,
        })
        covered_bodies.update(body_ids)
    residual_bodies = sorted(set(collision_by_body) - covered_bodies)
    if residual_bodies:
        residual_geoms = sorted(
            geom_id for body_id in residual_bodies for geom_id in collision_by_body[body_id]
        )
        residual_group = {
            "object_id": "unmapped_compiled_scene_geometry",
            "body_ids": residual_bodies,
            "body_names": [bodies_by_id[body_id] for body_id in residual_bodies],
            "geom_ids": residual_geoms,
            # The established policy treats residual world geometry separately
            # from semantic forbidden objects. Preserve that existing 0 m
            # residual rule rather than silently changing the certification.
            "minimum_clearance_m": 0.0,
        }
        other_scene_objects.append(residual_group)
    classified = set(target_body_ids)
    classified.update(body_id for row in forbidden_objects for body_id in row["body_ids"])
    classified.update(body_id for row in other_scene_objects for body_id in row["body_ids"])
    if set(collision_by_body) != classified:
        raise ValueError("strict_safety_policy_does_not_cover_all_collision_scene_bodies")

    pads = context.get("fingertip_pad_geometries", {})
    pad_ids: set[int] = set()
    for side in ("left", "right"):
        rows = pads.get(side)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"verified_{side}_fingertip_pad_geometry_is_missing")
        for row in rows:
            pad_ids.add(int(row["geom_id"]))
    if len(pad_ids) != 2:
        raise ValueError(f"expected_two_exact_fingertip_pad_geoms_got_{len(pad_ids)}")

    allowed_pad_pairs = [
        {"robot_geom_id": pad_id, "scene_geom_id": bowl_geom, "minimum_clearance_m": -0.002}
        for pad_id in sorted(pad_ids) for bowl_geom in target_geom_ids
    ]
    forbidden_pairs = [
        {
            "object_id_a": TARGET_OBJECT_ID,
            "object_id_b": row["object_id"],
            "body_ids_a": target_body_ids,
            "body_ids_b": row["body_ids"],
            "geom_ids_a": target_geom_ids,
            "geom_ids_b": row["geom_ids"],
        }
        for row in forbidden_objects + other_scene_objects
    ]
    target_geom_names = {
        int(geom_id): str(geom_name)
        for geom_id, geom_name in zip(target_binding["geom_ids"], target_binding["geom_names"])
    }
    if set(target_geom_names) != set(target_geom_ids):
        raise ValueError("held_bowl_geom_names_do_not_match_live_collision_geom_ids")
    grasp_binding = {
        "object_id": TARGET_OBJECT_ID,
        "body_id": int(target_binding["root_body_id"]),
        "body_name": str(target_binding["root_body_name"]),
        "geom_ids": sorted(int(x) for x in target_binding["geom_ids"]),
        "geom_names": [
            target_geom_names[x] for x in sorted(int(x) for x in target_binding["geom_ids"])
        ],
    }
    return {
        "source": "active MuJoCo semantic bindings; existing 10 mm forbidden-object and 2 mm held-target margins; only exact pad/bowl pairs may use the existing -2 mm contact allowance",
        "minimum_clearance_m": 0.010,
        "forbidden_objects": forbidden_objects,
        "allowed_target_objects": [target_group],
        "other_scene_objects": other_scene_objects,
        "allowed_contact_pairs": allowed_pad_pairs,
        "forbidden_object_pairs": forbidden_pairs,
        "allowed_carried_support_contact_geom_ids": [],
        "carried_object": {
            "object_id": TARGET_OBJECT_ID,
            "grasp_binding": grasp_binding,
            "allowed_contact_pairs": [],
            "allowed_support_contact_pairs": [],
        },
        "_proof": {
            "target_geom_ids": target_geom_ids,
            "target_geom_names": target_geom_names,
            "fingertip_pad_geom_ids": sorted(pad_ids),
            "scene_collision_body_count": len(collision_by_body),
            "classified_scene_collision_body_count": len(classified),
            "all_non_target_scene_groups_require_10mm": all(row["minimum_clearance_m"] == 0.010 for row in forbidden_objects),
            "residual_scene_rule_matches_existing_policy": all(row["minimum_clearance_m"] == 0.0 for row in other_scene_objects),
            "held_target_nonpad_minimum_clearance_m": 0.002,
            "allowed_pad_pair_minimum_clearance_m": -0.002,
        },
    }


def _safety_policy_for_bridge(policy: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in policy.items() if not key.startswith("_")}


def _current_arm_state(snapshot_state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows = snapshot_state.get("robot_states", [])
    if len(rows) != 1:
        raise ValueError("snapshot_must_contain_exactly_one_robot_state")
    robot = rows[0]
    arm = np.asarray(robot.get("joint_qpos"), dtype=np.float64)
    gripper = np.asarray(robot.get("gripper_qpos"), dtype=np.float64)
    if arm.ndim != 1 or arm.size < 7 or not np.isfinite(arm[:7]).all():
        raise ValueError("snapshot_arm_state_must_contain_finite_seven_joint_positions")
    if gripper.shape != (2,) or not np.isfinite(gripper).all():
        raise ValueError("snapshot_gripper_state_must_contain_two_finite_joint_positions")
    return arm[:7].copy(), gripper.copy()


def _tracking(snapshot_state: dict[str, Any], target_q: np.ndarray) -> dict[str, Any]:
    q, _ = _current_arm_state(snapshot_state)
    robot = snapshot_state["robot_states"][0]
    qvel = np.asarray(robot.get("joint_qvel"), dtype=np.float64)
    if qvel.ndim != 1 or qvel.size < 7 or not np.isfinite(qvel[:7]).all():
        raise ValueError("snapshot_arm_velocity_must_contain_finite_seven_joint_velocities")
    err = np.asarray(target_q, dtype=np.float64) - q
    return {
        "actual_joint_qpos_rad": q.tolist(),
        "actual_joint_qvel_rad_s": qvel[:7].tolist(),
        "target_joint_qpos_rad": np.asarray(target_q, dtype=np.float64).tolist(),
        "joint_error_rad": err.tolist(),
        "joint_error_linf_rad": float(np.max(np.abs(err))),
        "joint_error_l2_rad": float(np.linalg.norm(err)),
        "maximum_absolute_joint_velocity_rad_s": float(np.max(np.abs(qvel[:7]))),
    }


def _monitor_evidence(reply: dict[str, Any]) -> dict[str, Any]:
    monitor = reply.get("contact_monitoring", {})
    geometry = monitor.get("geometry_monitoring", {})
    substeps = reply.get("substep_contacts", [])
    geom_substeps = geometry.get("substeps", [])
    control_steps = monitor.get("control_steps", [])
    raw_complete = (
        monitor.get("expected_physics_substeps_per_control_step") == MUJOCO_SUBSTEPS_PER_ACTION
        and monitor.get("captured_physics_substeps") == MUJOCO_SUBSTEPS_PER_ACTION
        and monitor.get("unavailable_contact_substeps") == 0
        and len(substeps) == MUJOCO_SUBSTEPS_PER_ACTION
        and len(control_steps) == 1
        and control_steps[0].get("captured_physics_substeps") == MUJOCO_SUBSTEPS_PER_ACTION
        and control_steps[0].get("unobserved_expected_substeps") == 0
        and sum(control_steps[0].get("contact_samples", {}).values()) == MUJOCO_SUBSTEPS_PER_ACTION
    )
    geometry_observed = (
        len(geom_substeps) == MUJOCO_SUBSTEPS_PER_ACTION
        and all(row.get("status") in {"complete", "unsafe"} for row in geom_substeps)
    )
    hard_violations = []
    margin_violations = []
    for step_index, row in enumerate(geom_substeps):
        if row.get("self_contacts"):
            hard_violations.extend({"substep": step_index, "kind": "robot_self_contact", "evidence": x} for x in row["self_contacts"])
        if row.get("unallowed_scene_contacts"):
            hard_violations.extend({"substep": step_index, "kind": "unallowed_robot_scene_contact", "evidence": x} for x in row["unallowed_scene_contacts"])
        if row.get("unresolved_contacts"):
            hard_violations.extend({"substep": step_index, "kind": "unresolved_contact", "evidence": x} for x in row["unresolved_contacts"])
        if row.get("forbidden_object_contacts"):
            hard_violations.extend({"substep": step_index, "kind": "forbidden_object_contact", "evidence": x} for x in row["forbidden_object_contacts"])
        if row.get("joint_limit_violations"):
            hard_violations.extend({"substep": step_index, "kind": "operational_joint_limit_violation", "evidence": x} for x in row["joint_limit_violations"])
        for item in row.get("forbidden_contacts", []):
            if (
                item.get("geometry_clearance_requirement_violated")
                or item.get("carried_object_clearance_requirement_violated")
                or item.get("carried_object_allowed_pair_penetration_exceeded")
            ):
                margin_violations.append({"substep": step_index, **item})
            else:
                hard_violations.append({"substep": step_index, "kind": "forbidden_robot_scene_contact", "evidence": item})
    return {
        "raw_contact_monitoring_complete": bool(raw_complete),
        "bridge_combined_contact_and_safety_status": monitor.get("status"),
        "geometry_monitoring_observed_for_all_25_substeps": bool(geometry_observed),
        "strict_geometry_monitoring_status": geometry.get("status"),
        "minimum_clearance_m": geometry.get("minimum_clearance_m"),
        "hard_contact_or_limit_violation_count": len(hard_violations),
        "hard_violations": hard_violations,
        "strict_clearance_violation_count": len(margin_violations),
        "strict_clearance_violations": margin_violations,
        "safety_certification_pass_for_action": bool(raw_complete and geometry_observed and geometry.get("status") == "complete" and not hard_violations and not margin_violations),
    }


def _query_context(bridge: _Bridge) -> dict[str, Any]:
    return bridge.request({
        "op": "task_success_first_context",
        "expected_instruction": INSTRUCTION,
        "target_predicate": ["on", TARGET_OBJECT_ID, TARGET_REGION_ID],
    })


def _grasp(bridge: _Bridge, binding: dict[str, Any]) -> dict[str, Any]:
    return bridge.request({
        "op": "grasp_state",
        "carried_binding": binding,
        "expected_instruction": INSTRUCTION,
    })


def _terminal_metrics(context: dict[str, Any], target_object_pose_world: np.ndarray, plate_details: dict[str, Any]) -> dict[str, Any]:
    actual = _object_pose(context)
    expected = np.asarray(target_object_pose_world, dtype=np.float64)
    position_error = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
    orientation_error = float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())
    current_bottom = float(plate_details["plate_top_world_z_from_max_compiled_collision_vertex_m"])
    object_rotation = actual[:3, :3]
    min_offset = np.inf
    for row in context["target_collision_geometries"]:
        corners = _box_corners(
            np.asarray(row["center_root_m"], dtype=np.float64),
            np.asarray(row["rotation_root"], dtype=np.float64),
            np.asarray(row["size_m"], dtype=np.float64),
        )
        min_offset = min(min_offset, float(np.min(corners @ object_rotation[2, :])))
    bottom_world_z = float(actual[2, 3] + min_offset)
    clearance = bottom_world_z - current_bottom
    minimum_above_clearance = max(
        0.020,
        float(plate_details["requested_bowl_bottom_clearance_above_plate_collision_m"]) - 0.020,
    )
    return {
        "actual_object_root_pose_world_m": actual.tolist(),
        "requested_object_root_pose_world_m": expected.tolist(),
        "root_position_error_m": position_error,
        "root_orientation_error_rad": orientation_error,
        "object_collision_bottom_world_z_m": bottom_world_z,
        "plate_collision_top_world_z_m": current_bottom,
        "actual_bottom_clearance_above_plate_collision_m": clearance,
        "arrival_tolerance_position_m": 0.020,
        "arrival_tolerance_orientation_rad": 0.12,
        "minimum_above_plate_clearance_m": minimum_above_clearance,
        "arrived_above_plate": bool(
            position_error <= 0.020
            and orientation_error <= 0.12
            and clearance >= minimum_above_clearance
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    scene_source_path = args.scene_source.resolve()
    snapshot_path = args.snapshot.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    source_payload = snapshot_path.read_bytes()
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    envelope = pickle.loads(source_payload)
    source_meta = envelope.get("scene_metadata", {})
    source_timestep = int(envelope.get("snapshot", {}).get("task_bookkeeping", {}).get("timestep", -1))
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != EXPECTED_MODEL_SHA
        or source_meta.get("bddl_sha256") != EXPECTED_BDDL_SHA
        or source_meta.get("benchmark") != "obstacle_avoidance"
        or int(source_meta.get("level", -1)) != 0
        or int(source_meta.get("task_id", -1)) != 0
        or int(source_meta.get("environment_seed", -1)) != 0
        or int(source_meta.get("init_state_index", -1)) != 0
        or source_timestep != EXPECTED_START_TIMESTEP
    ):
        raise ValueError("candidate9_source_runtime_snapshot_identity_or_timestep_mismatch")
    expected_integration = np.asarray(envelope["snapshot"]["mujoco_state"], dtype=np.float64)
    if expected_integration.shape != (452,) or not np.isfinite(expected_integration).all():
        raise ValueError("candidate9_source_integration_state_is_not_expected_finite_452_vector")

    report: dict[str, Any] = {
        "schema": "gs_opd_oa_l0t0_candidate9_transport_above_plate.v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "experiment_scope": "one exact resumed candidate9 OA/L0T0 carry to a pose above the compiled plate; no place, release, or retreat",
        "task_feasibility_and_safety_certification_separated": True,
        "source_snapshot": {
            "path": str(snapshot_path), "sha256": _sha(source_payload),
            "timestep": source_timestep, "integration_state_shape": list(expected_integration.shape),
            "integration_state_sha256": hashlib.sha256(expected_integration.tobytes()).hexdigest(),
            "model_geometry_sha256": envelope.get("model_geometry_sha256"),
            "bddl_sha256": source_meta.get("bddl_sha256"),
        },
        "task": {
            "benchmark": "obstacle_avoidance", "level": 0, "task_id": 0,
            "instruction": INSTRUCTION,
            "target_object_id": TARGET_OBJECT_ID,
            "target_region_id": TARGET_REGION_ID,
            "native_predicate": ["on", TARGET_OBJECT_ID, TARGET_REGION_ID],
        },
        "limits": {
            "standard_task_horizon": 1000,
            "maximum_physical_actions": MAX_PHYSICAL_ACTIONS,
            "starting_timestep": source_timestep,
            "maximum_resulting_timestep": source_timestep + MAX_PHYSICAL_ACTIONS,
            "mujoco_substeps_per_physical_action": MUJOCO_SUBSTEPS_PER_ACTION,
            "no_gripper_open_action": True,
        },
        "planning": {},
        "physical_actions": 0,
        "recorded_physics_substeps": 0,
        "all_physical_actions_have_25_contact_substeps": False,
        "task_success": False,
        "bowl_remains_grasped": False,
        "arrived_above_plate": False,
        "contact_substep_sampling_complete": False,
        "geometry_monitoring_observed_for_all_substeps": False,
        "safety_monitoring_complete": False,
        "safety_monitoring_status": "not_started",
        "safety_certified": False,
        "new_trainable_episodes": 0,
        "pi05_native_action_compatible": False,
        "control_action_space": "7D_joint_target_qpos_rad",
        "release_or_place_attempted": False,
    }
    report_path = output_dir / "transport_report.json"
    _write_json(report_path, report)
    (output_dir / "source_t190_runtime_snapshot.pkl").write_bytes(source_payload)
    source_scene = json.loads(gzip.decompress(scene_source_path.read_bytes()))
    if source_scene.get("scene_metadata", {}).get("model_geometry_sha256") != EXPECTED_MODEL_SHA:
        raise ValueError("reference_scene_source_model_signature_mismatch")
    report["reference_scene_source"] = str(scene_source_path)

    process = log = connection = recorder = None
    action_rows = None
    actual_monitor_rows: list[dict[str, Any]] = []
    bridge = None
    final_context = None
    final_grasp = None
    strict_policy: dict[str, Any] | None = None
    goal_object_world: np.ndarray | None = None
    plate_details: dict[str, Any] | None = None
    try:
        process, log, connection, initial = curobo_api._start_bridge(source_scene, output_dir)
        bridge = _Bridge(connection)
        if int(initial.get("timestep", -1)) != 0:
            raise ValueError("fresh_existing_Bridge_did_not_start_at_t0_before_exact_restore")
        restore = bridge.request({"op": "restore_runtime_snapshot", "payload": source_payload})
        restored_state = restore.get("snapshot_state", {}).get("mujoco_integration_state")
        restored_state = np.asarray(restored_state, dtype=np.float64)
        if (
            restore.get("status") != "restored"
            or restore.get("model_geometry_sha256") != EXPECTED_MODEL_SHA
            or restored_state.shape != expected_integration.shape
            or not np.array_equal(restored_state, expected_integration)
        ):
            raise ValueError(f"candidate9_t190_exact_restore_failed:{restore.get('error') or restore.get('status')}")
        report["restore"] = {
            "status": restore.get("status"),
            "model_geometry_sha256": restore.get("model_geometry_sha256"),
            "timestep": int(restore.get("snapshot_state", {}).get("timestep", EXPECTED_START_TIMESTEP)),
            "integration_state_exact_match": True,
            "integration_state_sha256": hashlib.sha256(restored_state.tobytes()).hexdigest(),
        }
        report["status"] = "restored_exact_t190"

        live_obs = bridge.request({"op": "observe", "include_snapshot_state": True})
        if int(live_obs.get("timestep", -1)) != EXPECTED_START_TIMESTEP:
            raise ValueError(f"restored_live_timestep_not_190:{live_obs.get('timestep')}")
        native_before = bridge.request({"op": "native_task_status"})
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        context = _query_context(bridge)
        collision_scene = bridge.request({"op": "collision_scene_export"})
        geometry_inventory = bridge.request({"op": "geometry_inventory"})
        gripper_inventory = bridge.request({"op": "gripper_geometry_inventory"})
        if semantic.get("status") != "complete" or semantic.get("model_geometry_sha256") != EXPECTED_MODEL_SHA:
            raise ValueError("active_semantic_geometry_binding_is_not_complete_for_candidate9")
        if context.get("status") != "complete" or context.get("read_only") is not True:
            raise ValueError(f"active_target_context_unavailable:{context.get('error')}")
        if collision_scene.get("status") != "complete" or collision_scene.get("physical_state_unchanged") is not True:
            raise ValueError("current_full_collision_scene_export_is_incomplete_or_mutating")
        if collision_scene.get("unsupported_geometries"):
            raise ValueError("active_scene_collision_export_has_unsupported_geometry")
        if geometry_inventory.get("status") != "complete" or gripper_inventory.get("status") != "complete":
            raise ValueError("current_geometry_or_gripper_inventory_is_incomplete")
        target_bind = semantic["bindings"].get(TARGET_OBJECT_ID)
        if not isinstance(target_bind, dict):
            raise ValueError("active_target_object_binding_missing")
        grasp_binding = {
            "object_id": TARGET_OBJECT_ID,
            "body_id": int(target_bind["root_body_id"]),
            "body_name": str(target_bind["root_body_name"]),
            "geom_ids": sorted(int(x) for x in target_bind["geom_ids"]),
            "geom_names": [str(x) for x in target_bind["geom_names"]],
        }
        live_grasp = _grasp(bridge, grasp_binding)
        if live_grasp.get("status") != "pass" or live_grasp.get("grasp_confirmed") is not True:
            raise ValueError(f"candidate9_t190_live_grasp_not_confirmed:{live_grasp.get('reason') or live_grasp.get('error')}")

        arm_q, gripper_q = _current_arm_state(live_obs["snapshot_state"])
        if not np.allclose(arm_q, np.asarray(context["arm_qpos"], dtype=np.float64), atol=1e-10, rtol=0):
            raise ValueError("active_context_arm_state_disagrees_with_exact_restored_snapshot")
        fk = bridge.request({"op": "scratch_robot_fk_audit", "arm_joint_positions_rad": [arm_q.tolist()]})
        if fk.get("status") != "complete" or fk.get("physical_state_unchanged") is not True:
            raise ValueError("current_robot_base_and_tool_fk_audit_is_incomplete")
        fk_pose = fk["poses"][0]
        world_base = _pose_from_wxyz(
            fk_pose["bodies"]["robot0_base"]["position_world_m"],
            fk_pose["bodies"]["robot0_base"]["quaternion_world_wxyz"],
        )
        world_hand = _pose_from_wxyz(
            fk_pose["bodies"]["robot0_right_hand"]["position_world_m"],
            fk_pose["bodies"]["robot0_right_hand"]["quaternion_world_wxyz"],
        )
        live_hand_base = np.linalg.inv(world_base) @ world_hand
        planning_source = copy.deepcopy(source_scene)
        planning_source["target_context"] = copy.deepcopy(context)
        planning_source["target_context"]["target_binding"] = target_bind
        planning_source["target_context"]["arm_qpos"] = arm_q.tolist()
        planning_source["collision_scene_export"] = collision_scene
        planner, device, dtype, start_state, current_q, planner_info = curobo_api._make_planner(
            planning_source, world_base, "closed", seed=20261028, gripper_qpos_m=gripper_q
        )
        if not np.allclose(current_q, arm_q, atol=1e-10, rtol=0):
            raise ValueError("CuRobo_start_state_does_not_equal_restored_live_arm_qpos")
        curobo_fk = curobo_api._pose_from_kinematics(planner, arm_q, device, dtype)
        registration_error = curobo_api._pose_error(live_hand_base, curobo_fk)
        report["current_state"] = {
            "native_task_status": native_before,
            "live_grasp_state": live_grasp,
            "arm_qpos_rad": arm_q.tolist(),
            "actual_gripper_qpos_m": gripper_q.tolist(),
            "target_binding": grasp_binding,
            "live_fk_model_sha256": fk.get("model_geometry_sha256"),
            "world_base_transform_m": world_base.tolist(),
            "world_panda_hand_transform_m": world_hand.tolist(),
            "curobo_hand_registration_error": registration_error,
        }
        if registration_error["position_error_m"] > 0.002 or registration_error["orientation_error_rad"] > np.deg2rad(0.10):
            raise ValueError("live_CuRobo_tool_registration_differs_from_active_MuJoCo_FK")

        goal_object_world, plate_details = _plate_above_object_pose(
            context, semantic, collision_scene, world_base, float(args.plate_clearance_m)
        )
        object_world = _object_pose(context)
        world_eef = _frame_pose(context["eef_pose_world"])
        eef_object = np.linalg.inv(world_eef) @ object_world
        world_hand_to_object = np.linalg.inv(world_hand) @ object_world
        world_hand_goal = goal_object_world @ np.linalg.inv(world_hand_to_object)
        base_hand_goal = np.linalg.inv(world_base) @ world_hand_goal
        report["goal_geometry"] = plate_details
        if args.prior_route_report is not None:
            prior = json.loads(args.prior_route_report.read_text(encoding="utf-8"))
            report["prior_planning_attempt"] = {
                "report": str(args.prior_route_report.resolve()),
                "status": prior.get("status"),
                "physical_actions": prior.get("physical_actions"),
                "goal_clearance_m": prior.get("goal_geometry", {}).get("requested_bowl_bottom_clearance_above_plate_collision_m"),
                "ik_success": prior.get("planning", {}).get("ik", {}).get("success"),
                "curobo_status": prior.get("planning", {}).get("curobo", {}).get("status"),
                "curobo_returned_none": prior.get("planning", {}).get("curobo", {}).get("returned_none"),
                "reason_for_bounded_adjustment": "prior above-plate target had no successful IK seed and CuRobo returned no route; only its vertical clearance is reduced, with the same live plate collision bounds and held-object relation",
            }
        report["goal_geometry"]["T_world_panda_hand_goal_m"] = world_hand_goal.tolist()
        report["goal_geometry"]["T_base_panda_hand_goal_m"] = base_hand_goal.tolist()
        report["goal_geometry"]["T_eef_object_at_t190_m"] = eef_object.tolist()
        report["goal_geometry"]["T_panda_hand_object_at_t190_m"] = world_hand_to_object.tolist()

        strict_policy = _collision_policy(context, semantic, geometry_inventory)
        policy_for_bridge = _safety_policy_for_bridge(strict_policy)
        report["safety_policy"] = strict_policy["_proof"] | {
            "source": strict_policy["source"],
            "unchanged_existing_clearance_rules": True,
            "certification_is_separate_from_task_feasibility": True,
        }
        current_safety = bridge.request({
            "op": "safety_geometry_check",
            "safety_geometry_policy": policy_for_bridge,
        })
        report["current_state"]["strict_safety_geometry_check"] = current_safety

        attached_spheres, coverage = curobo_api._attached_spheres(planning_source)
        if not coverage.get("every_source_box_analytically_covered"):
            raise ValueError("attached_payload_collision_coverage_proof_failed")
        object_base = np.linalg.inv(world_base) @ object_world
        attached_tensor = curobo_api.torch.as_tensor(attached_spheres, device=device, dtype=dtype)
        object_pose = curobo_api._pose_object(object_base, device, dtype)
        planner.attachment_manager.update(
            attached_tensor,
            start_state,
            link_name="attached_object",
            world_objects_pose_offset=object_pose,
        )
        world_without_held_bowl, cuboid_count, mesh_count = curobo_api._make_world(
            planning_source, world_base, exclude_target=True
        )
        planner.update_world(world_without_held_bowl)
        ik, curobo_plan, path, endpoint = curobo_api._plan_stage(
            planner, device, dtype, start_state, base_hand_goal,
            grasp_goal=bool(args.ignore_coarse_curobo_finger_links),
        )
        plan_summary = {
            "seed": 20261028,
            "start_joint_state_rad": current_q.tolist(),
            "goal_T_base_panda_hand_m": base_hand_goal.tolist(),
            "ik": ik,
            "curobo": curobo_plan,
            "path_point_count_valid": len(path) if path is not None else 0,
            "path_point_count_buffered": curobo_plan.get("buffered_trajectory_point_count"),
            "world_cuboid_count_after_held_bowl_removal": cuboid_count,
            "world_mesh_count_after_held_bowl_removal": mesh_count,
            "attachment_geometry_coverage": coverage,
            "attached_sphere_count": int(len(attached_spheres)),
            "attached_object_pose_base_m": object_base.tolist(),
            "target_collision_geom_ids_excluded_from_static_world": grasp_binding["geom_ids"],
            "full_current_scene_geometry_used": True,
            "effective_valid_trajectory_only_is_executed": True,
            "coarse_curobo_finger_link_collision_filter_used": bool(args.ignore_coarse_curobo_finger_links),
            "finger_link_filter_is_not_a_safety_or_contact_permission": True,
            "exact_active_mujoco_path_audit_required_after_planning": True,
            "curobo_configuration": planner_info,
        }
        if endpoint is not None:
            plan_summary["endpoint_fk_error"] = curobo_api._pose_error(base_hand_goal, endpoint)
        report["planning"] = plan_summary
        _write_json(output_dir / "curobo_route.json", plan_summary)
        report["status"] = "planned_transport_route" if path is not None and curobo_plan.get("success") else "blocked_no_curobo_route"
        _write_json(report_path, report)
        if path is None or not curobo_plan.get("success"):
            raise RuntimeError(f"CuRobo_transport_route_not_available:{curobo_plan.get('rejection_reason') or curobo_plan.get('status')}")
        if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 or not np.isfinite(path).all():
            raise ValueError("CuRobo_valid_route_is_not_finite_N_by_7_with_two_or_more_points")
        if float(np.max(np.abs(path[0] - arm_q))) > 5e-4:
            raise ValueError("CuRobo_route_start_does_not_match_exact_live_t190_arm_state")
        if len(path) - 1 + MAX_ENDPOINT_FEEDBACK_ACTIONS > MAX_PHYSICAL_ACTIONS:
            raise ValueError("valid_CuRobo_route_exceeds_bounded_180_action_physical_budget")
        endpoint_error = curobo_api._pose_error(base_hand_goal, endpoint)
        if endpoint_error["position_error_m"] > 0.002 or endpoint_error["orientation_error_rad"] > np.deg2rad(0.10):
            raise ValueError("CuRobo_route_endpoint_FK_exceeds_2mm_or_0.10deg")

        contact_audit = bridge.request({
            "op": "scratch_robot_path_contact_audit",
            "arm_joint_states_rad": path.tolist(),
            "gripper_joint_states_rad": np.repeat(gripper_q[None, :], len(path), axis=0).tolist(),
            "object_id": TARGET_OBJECT_ID,
            "fingertip_contact_scope": "all_states",
            "T_eef_object_m": eef_object.tolist(),
        })
        plan_summary["active_mujoco_path_contact_audit"] = contact_audit
        if (
            contact_audit.get("status") != "complete"
            or contact_audit.get("physical_state_unchanged") is not True
            or int(contact_audit.get("illegal_contact_count", -1)) != 0
            or int(contact_audit.get("state_count", -1)) != len(path)
        ):
            report["planning"] = plan_summary
            report["status"] = "blocked_active_mujoco_path_contact_audit"
            raise RuntimeError("active_MuJoCo_scratch_path_contact_audit_rejected_or_incomplete")
        plan_summary["scratch_path_contact_audit_passed"] = True
        _write_json(output_dir / "curobo_route.json", plan_summary)
        report["planning"] = plan_summary
        report["status"] = "ready_for_bounded_physical_transport"
        _write_json(report_path, report)

        binding = strict_policy["carried_object"]["grasp_binding"]
        start_grasp = _grasp(bridge, binding)
        if start_grasp.get("status") != "pass" or start_grasp.get("grasp_confirmed") is not True:
            raise RuntimeError("held_bowl_grasp_lost_before_first_physical_transport_action")

        from research.gs_opd.repair.run_human_l0t2_curobo_transport import EpisodeRecorder

        recorder = EpisodeRecorder(
            output_dir,
            bridge,
            record_schema="gs_opd_oa_l0t0_candidate9_transport_step.v1",
        )
        recorder.snapshot("resumed_t190")
        action_rows_path = output_dir / "action_safety_summary.jsonl"
        action_rows = action_rows_path.open("w", encoding="utf-8", buffering=1)
        report["physical_execution"] = {
            "status": "running",
            "control_mode": "joint_space_mass_matrix_PD_torque",
            "control_action_space": "7D_joint_target_qpos_rad",
            "pi05_native_action_compatible": False,
            "gripper_action": 0.0,
            "gripper_open_command_issued": False,
            "waypoint_feedback_tolerance_rad": WAYPOINT_TOLERANCE_RAD,
            "endpoint_feedback_tolerance_rad": ENDPOINT_TOLERANCE_RAD,
            "endpoint_velocity_tolerance_rad_s": ENDPOINT_VELOCITY_TOLERANCE_RAD_S,
            "episode_recorder": "existing dual-camera physical EpisodeRecorder with OA/L0T0 record schema",
            "scratch_preflight_policy": "strict geometry values are recorded independently; only incomplete monitor or predicted actual unallowed contact blocks the next physical action",
        }
        _write_json(report_path, report)

        def execute_one(q_target: np.ndarray, phase: str, logical_waypoint: int) -> tuple[dict[str, Any], dict[str, Any]]:
            if recorder.action_count >= MAX_PHYSICAL_ACTIONS:
                raise RuntimeError("bounded_physical_action_budget_exhausted")
            preview = bridge.request({
                "op": "preview_joint_trajectory_step",
                "action_index": recorder.action_count,
                "joint_target_qpos_rad": np.asarray(q_target, dtype=np.float64).tolist(),
                "gripper_action": 0.0,
                "joint_position_kp": 50.0,
                "safety_geometry_policy": policy_for_bridge,
            })
            preview_evidence = _monitor_evidence(preview)

            def save_blocked_preview(reason: str) -> None:
                preview_path = output_dir / f"blocked_preflight_waypoint_{logical_waypoint:03d}.json"
                _write_json(preview_path, {
                    "logical_waypoint_index": logical_waypoint,
                    "action_index": recorder.action_count,
                    "joint_target_qpos_rad": np.asarray(q_target, dtype=np.float64).tolist(),
                    "stop_reason": reason,
                    "grasp_before_preview": _grasp(bridge, binding),
                    "preview": preview,
                    "evidence": preview_evidence,
                })
                report.setdefault("physical_execution", {})["blocked_preflight_preview"] = str(preview_path)
                _write_json(report_path, report)

            if preview.get("status") != "complete" or preview.get("main_restore_verified") is not True:
                reason = f"pre_action_scratch_preview_incomplete:{logical_waypoint}:{preview.get('error')}"
                save_blocked_preview(reason)
                raise RuntimeError(reason)
            if not preview_evidence["raw_contact_monitoring_complete"] or not preview_evidence["geometry_monitoring_observed_for_all_25_substeps"]:
                reason = f"pre_action_scratch_monitor_incomplete:{logical_waypoint}"
                save_blocked_preview(reason)
                raise RuntimeError(reason)
            if preview_evidence["hard_contact_or_limit_violation_count"]:
                reason = f"pre_action_scratch_predicted_unallowed_contact_or_limit:{logical_waypoint}"
                save_blocked_preview(reason)
                raise RuntimeError(reason)
            physical = bridge.request({
                "op": "joint_trajectory_step",
                "action_index": recorder.action_count,
                "joint_target_qpos_rad": np.asarray(q_target, dtype=np.float64).tolist(),
                "gripper_action": 0.0,
                "joint_position_kp": 50.0,
                "safety_geometry_policy": policy_for_bridge,
            })
            if physical.get("status") == "failed":
                raise RuntimeError(f"MuJoCo_joint_trajectory_step_failed:{physical.get('error')}")
            recorder.record(physical, np.asarray(q_target, dtype=np.float64), phase, 0.0)
            evidence = _monitor_evidence(physical)
            actual_monitor_rows.append(evidence)
            report["physical_actions"] = recorder.action_count
            report["recorded_physics_substeps"] = recorder.recorded_physics_substeps
            report["all_physical_actions_have_25_contact_substeps"] = bool(
                len(actual_monitor_rows) == recorder.action_count
                and all(
                    row["raw_contact_monitoring_complete"]
                    and row["geometry_monitoring_observed_for_all_25_substeps"]
                    for row in actual_monitor_rows
                )
                and recorder.recorded_physics_substeps == recorder.action_count * MUJOCO_SUBSTEPS_PER_ACTION
            )
            if not evidence["raw_contact_monitoring_complete"] or not evidence["geometry_monitoring_observed_for_all_25_substeps"]:
                raise RuntimeError(f"physical_25_substep_monitoring_incomplete_at_action:{recorder.action_count - 1}")
            if evidence["hard_contact_or_limit_violation_count"]:
                raise RuntimeError(f"physical_unallowed_contact_or_joint_limit_at_action:{recorder.action_count - 1}")
            current_context = _query_context(bridge)
            current_grasp = _grasp(bridge, binding)
            if current_grasp.get("status") != "pass" or current_grasp.get("grasp_confirmed") is not True:
                raise RuntimeError(f"held_bowl_grasp_lost_at_action:{recorder.action_count - 1}")
            current_object_pose = _object_pose(current_context)
            current_eef = _frame_pose(current_context["eef_pose_world"])
            relation = np.linalg.inv(current_eef) @ current_object_pose
            relation_position_error = float(np.linalg.norm(relation[:3, 3] - eef_object[:3, 3]))
            relation_orientation_error = float(Rotation.from_matrix(eef_object[:3, :3].T @ relation[:3, :3]).magnitude())
            if relation_position_error > 0.010 or relation_orientation_error > np.deg2rad(10.0):
                raise RuntimeError(f"held_bowl_slipped_relative_to_eef_at_action:{recorder.action_count - 1}")
            track = _tracking(physical["snapshot_state"], np.asarray(q_target, dtype=np.float64))
            action_rows.write(json.dumps({
                "action_index": recorder.action_count - 1,
                "scene_timestep": physical.get("timestep"),
                "phase": phase,
                "logical_waypoint_index": logical_waypoint,
                "preview_monitoring": preview_evidence,
                "physical_monitoring": evidence,
                "grasp_state": current_grasp,
                "eef_object_relative_translation_error_m": relation_position_error,
                "eef_object_relative_orientation_error_rad": relation_orientation_error,
                "tracking": track,
                "native_task_success": physical.get("task_success"),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
            if recorder.action_count % 10 == 0:
                recorder.snapshot(f"step_{recorder.action_count:04d}")
                report["latest_feedback"] = track
                _write_json(report_path, report)
                print(f"physical transport actions={recorder.action_count}/{MAX_PHYSICAL_ACTIONS} t={physical.get('timestep')} q_error={track['joint_error_linf_rad']:.4f}m grasp=confirmed")
            return physical, track

        for waypoint_index, q_target in enumerate(path[1:], start=1):
            attempts = 0
            latest_track = None
            while attempts < MAX_ACTIONS_PER_WAYPOINT:
                physical, latest_track = execute_one(q_target, "transport_above_plate", waypoint_index)
                attempts += 1
                if latest_track["joint_error_linf_rad"] <= WAYPOINT_TOLERANCE_RAD:
                    break
                if latest_track["joint_error_linf_rad"] > HARD_TRACKING_STOP_RAD:
                    raise RuntimeError(f"closed_loop_waypoint_tracking_error_exceeded:{waypoint_index}:{latest_track['joint_error_linf_rad']:.6f}")
                if attempts >= MAX_ACTIONS_PER_WAYPOINT:
                    raise RuntimeError(f"closed_loop_waypoint_feedback_stalled:{waypoint_index}:{latest_track['joint_error_linf_rad']:.6f}")
            if recorder.action_count >= MAX_PHYSICAL_ACTIONS:
                raise RuntimeError("bounded_physical_action_budget_exhausted_during_route")

        endpoint_events = []
        endpoint_done = False
        for _ in range(MAX_ENDPOINT_FEEDBACK_ACTIONS + 1):
            state = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
            feedback = _tracking(state, path[-1])
            endpoint_events.append(feedback)
            if (
                feedback["joint_error_linf_rad"] <= ENDPOINT_TOLERANCE_RAD
                and feedback["maximum_absolute_joint_velocity_rad_s"] <= ENDPOINT_VELOCITY_TOLERANCE_RAD_S
            ):
                endpoint_done = True
                break
            if len(endpoint_events) > 2:
                recent = endpoint_events[-3:]
                improvement = recent[0]["joint_error_linf_rad"] - recent[-1]["joint_error_linf_rad"]
                if improvement < 0.001 and feedback["joint_error_linf_rad"] > WAYPOINT_TOLERANCE_RAD:
                    raise RuntimeError("closed_loop_endpoint_feedback_stalled")
            if len(endpoint_events) > MAX_ENDPOINT_FEEDBACK_ACTIONS:
                break
            _, feedback = execute_one(path[-1], "transport_above_plate_endpoint_feedback", len(path) - 1)
        if not endpoint_done:
            state = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
            feedback = _tracking(state, path[-1])
            endpoint_done = (
                feedback["joint_error_linf_rad"] <= ENDPOINT_TOLERANCE_RAD
                and feedback["maximum_absolute_joint_velocity_rad_s"] <= ENDPOINT_VELOCITY_TOLERANCE_RAD_S
            )
        report["physical_execution"]["endpoint_feedback"] = {
            "converged": endpoint_done,
            "max_actions": MAX_ENDPOINT_FEEDBACK_ACTIONS,
            "samples": endpoint_events,
            "final": feedback,
        }
        if not endpoint_done:
            raise RuntimeError("closed_loop_transport_endpoint_not_reached_or_stable")

        final_context = _query_context(bridge)
        final_grasp = _grasp(bridge, binding)
        final_native = bridge.request({"op": "native_task_status"})
        report["physical_execution"]["final_grasp_state"] = final_grasp
        report["physical_execution"]["final_native_task_status"] = final_native
        arrival = _terminal_metrics(final_context, goal_object_world, plate_details)
        report["physical_execution"]["terminal_transport_metrics"] = arrival
        report["arrived_above_plate"] = arrival["arrived_above_plate"]
        report["bowl_remains_grasped"] = bool(final_grasp.get("status") == "pass" and final_grasp.get("grasp_confirmed") is True)
        report["task_success"] = bool(final_native.get("task_success") is True)
        report["physical_execution"]["status"] = "transport_arrived_above_plate" if arrival["arrived_above_plate"] else "route_executed_but_target_not_reached"
        report["status"] = report["physical_execution"]["status"]
        if not arrival["arrived_above_plate"]:
            raise RuntimeError("physical_object_pose_did_not_reach_above_plate_target_tolerance")
        if not report["bowl_remains_grasped"]:
            raise RuntimeError("held_bowl_grasp_not_confirmed_at_terminal_state")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["status"] = report.get("status") if report.get("status", "running") not in {"running", "restored_exact_t190", "planned_transport_route", "ready_for_bounded_physical_transport"} else "stopped_before_transport_or_during_execution"
        report.setdefault("physical_execution", {})["status"] = report["status"]
        report["physical_execution"]["stop_reason"] = report["error"]
        if recorder is not None and bridge is not None:
            try:
                final_context = _query_context(bridge)
                binding = strict_policy["carried_object"]["grasp_binding"] if strict_policy else None
                final_grasp = _grasp(bridge, binding) if binding else None
                report["physical_execution_terminal_state"] = {
                    "context": final_context,
                    "grasp_state": final_grasp,
                    "native_task_status": bridge.request({"op": "native_task_status"}),
                }
                report["bowl_remains_grasped"] = bool(final_grasp and final_grasp.get("status") == "pass" and final_grasp.get("grasp_confirmed") is True)
            except Exception as state_exc:
                report["terminal_state_query_error"] = f"{type(state_exc).__name__}: {state_exc}"
    finally:
        if bridge is not None:
            try:
                terminal = bridge.request({"op": "export_runtime_snapshot"})
                if terminal.get("status") == "complete" and isinstance(terminal.get("payload"), bytes):
                    terminal_path = output_dir / "terminal_runtime_snapshot.pkl"
                    terminal_path.write_bytes(terminal["payload"])
                    terminal_envelope = pickle.loads(terminal["payload"])
                    terminal_timestep = int(
                        terminal_envelope.get("snapshot", {}).get("task_bookkeeping", {}).get(
                            "timestep", terminal.get("snapshot_state", {}).get("timestep", -1)
                        )
                    )
                    report["terminal_snapshot"] = {
                        "path": str(terminal_path),
                        "sha256": _sha(terminal["payload"]),
                        "timestep": terminal_timestep,
                        "recoverable": True,
                    }
                else:
                    report["terminal_snapshot"] = {"recoverable": False, "error": terminal.get("error") or terminal.get("status")}
            except Exception as exc:
                report["terminal_snapshot"] = {"recoverable": False, "error": f"{type(exc).__name__}: {exc}"}
        if recorder is not None:
            report["physical_actions"] = recorder.action_count
            report["recorded_physics_substeps"] = recorder.recorded_physics_substeps
            report["monitoring_incomplete_actions"] = recorder.monitoring_incomplete_actions
            report["all_physical_actions_have_25_contact_substeps"] = bool(
                recorder.action_count > 0
                and len(actual_monitor_rows) == recorder.action_count
                and all(
                    row["raw_contact_monitoring_complete"]
                    and row["geometry_monitoring_observed_for_all_25_substeps"]
                    for row in actual_monitor_rows
                )
                and recorder.recorded_physics_substeps == recorder.action_count * MUJOCO_SUBSTEPS_PER_ACTION
            )
            report["artifacts"] = {
                "physical_records": str(output_dir / "physical_step_records.jsonl.gz"),
                "action_safety_summary": str(output_dir / "action_safety_summary.jsonl"),
                "dual_camera_video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                "dual_camera_frames": str(output_dir / "dual_camera_frames"),
            }
            blocked_preview = report.get("physical_execution", {}).get("blocked_preflight_preview")
            if blocked_preview:
                report["artifacts"]["blocked_preflight_preview"] = blocked_preview
            report.setdefault("physical_execution", {})["strict_safety_action_count"] = len(actual_monitor_rows)
            report["physical_execution"]["final_safety_all_actions_passed"] = bool(
                actual_monitor_rows
                and all(row["safety_certification_pass_for_action"] for row in actual_monitor_rows)
            )
            report["physical_execution"]["strict_safety_segment_status"] = (
                "pass" if report["physical_execution"]["final_safety_all_actions_passed"] else "not_certified"
            )
            try:
                recorder.close()
            except Exception as exc:
                report["video_or_recorder_close_error"] = f"{type(exc).__name__}: {exc}"
        report["contact_substep_sampling_complete"] = bool(
            report.get("physical_actions", 0) > 0
            and actual_monitor_rows
            and all(row["raw_contact_monitoring_complete"] for row in actual_monitor_rows)
        )
        report["geometry_monitoring_observed_for_all_substeps"] = bool(
            report.get("physical_actions", 0) > 0
            and actual_monitor_rows
            and all(row["geometry_monitoring_observed_for_all_25_substeps"] for row in actual_monitor_rows)
        )
        report["safety_monitoring_complete"] = bool(
            report.get("contact_substep_sampling_complete") is True
            and report.get("geometry_monitoring_observed_for_all_substeps") is True
            and all(
                row["bridge_combined_contact_and_safety_status"] == "complete"
                and row["strict_geometry_monitoring_status"] == "complete"
                for row in actual_monitor_rows
            )
        )
        report["safety_monitoring_status"] = (
            "complete" if report["safety_monitoring_complete"]
            else "incomplete_or_unsafe"
        )
        # A strict safety certificate requires every actual geometry monitor to
        # pass its unchanged 10 mm / 2 mm constraints; feasibility alone never
        # promotes a route into an authenticated safe episode.
        report["safety_segment_monitor_passed"] = bool(
            report.get("safety_monitoring_complete") is True
            and report.get("physical_execution", {}).get("final_safety_all_actions_passed") is True
        )
        # This segment intentionally stops above the plate; it is not a
        # completed, release-verified safe-success Episode.
        report["safety_certified"] = False
        if not report.get("task_success"):
            report["new_trainable_episodes"] = 0
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            _write_json(report_path, report)
            summary = [
                "# OA/L0T0 candidate9 transport above plate",
                "",
                f"- Status: `{report.get('status')}`",
                f"- Restored exact source timestep: `{report.get('restore', {}).get('timestep', EXPECTED_START_TIMESTEP)}`",
                f"- Physical actions: `{report.get('physical_actions', 0)}`",
                f"- Recorded MuJoCo substeps: `{report.get('recorded_physics_substeps', 0)}`",
                f"- Arrived above plate: `{report.get('arrived_above_plate')}`",
                f"- Bowl grasp confirmed at end: `{report.get('bowl_remains_grasped')}`",
                f"- Native task success: `{report.get('task_success')}` (place/release intentionally not attempted)",
                f"- Safety monitoring complete: `{report.get('safety_monitoring_complete')}`",
                f"- Safety certified: `{report.get('safety_certified')}`",
                f"- Error / stop reason: `{report.get('error')}`",
                f"- Terminal Snapshot: `{report.get('terminal_snapshot', {}).get('path')}`",
                f"- Dual-camera video: `{report.get('artifacts', {}).get('dual_camera_video')}`",
                "",
            ]
            (output_dir / "transport_report.md").write_text("\n".join(summary), encoding="utf-8")
        except Exception:
            pass
        if action_rows is not None:
            try:
                action_rows.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    pass
        if log is not None:
            try:
                log.close()
            except Exception:
                pass
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-source", type=Path, default=DEFAULT_SOURCE_SCENE)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plate-clearance-m", type=float, default=PLATE_ABOVE_CLEARANCE_ENGINEERING_CHOICE_M)
    parser.add_argument("--prior-route-report", type=Path, default=None)
    parser.add_argument(
        "--ignore-coarse-curobo-finger-links",
        action="store_true",
        help="temporarily omit whole finger links inside CuRobo only; route is rejected unless full active MuJoCo exact-Geom path audit and live safety monitor pass their own criteria",
    )
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "status": result.get("status"),
        "physical_actions": result.get("physical_actions"),
        "recorded_physics_substeps": result.get("recorded_physics_substeps"),
        "arrived_above_plate": result.get("arrived_above_plate"),
        "bowl_remains_grasped": result.get("bowl_remains_grasped"),
        "task_success": result.get("task_success"),
        "safety_monitoring_complete": result.get("safety_monitoring_complete"),
        "safety_certified": result.get("safety_certified"),
        "error": result.get("error"),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
