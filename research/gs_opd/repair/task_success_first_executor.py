# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Student-first, bounded task-success execution for active LIBERO BDDL goals.

The executor owns orchestration only. It reuses the original Pi0.5 inference
adapter, Safety Bridge, Runtime Snapshots, geometry driven grasp search,
MuJoCo full-geometry 7D RRT, native OSC_POSE tracking, and TaskPhaseController.
Safety Reconstruction is not called and every physical action remains a real
MuJoCo step. Search artifacts are candidates, never safe training data.
"""

from __future__ import annotations

import argparse
import cv2
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.repair.task_phase_controller import PhaseEvidence, TaskPhaseController
from research.gs_opd.repair.task_success_first_search import (
    MAX_DEMONSTRATION_HORIZON,
    inspect_native_goal_coverage,
    pending_object_goals,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_ROOT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "obstacle_avoidance_L0T1_raw_student_seed0_init0_20261025_234627"
)
DEFAULT_OUTPUT_ROOT = Path("/opt/data/private/gs_opd/rollouts")
DEFAULT_CHECKPOINT = Path(
    "/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft"
)
LIBERO_PYTHON = Path("/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python")
BRIDGE_AUTHKEY = b"gs-opd-local"
MAX_NATIVE_PLACEMENT_ACTIONS = 96
MAX_NATIVE_RELEASE_ACTIONS = 32
MAX_NATIVE_WITHDRAWAL_ACTIONS = 64
MAX_NATIVE_SETTLE_ACTIONS = 144
_STAGE_A_NEAR_GOAL_RELEASE_PROBE_ENABLED = False
_STAGE_A_NEAR_GOAL_RELEASE_PROBE_MAX_ERROR_M = 0.012
FUTURE_OBJECT_GOAL_ACTION_RESERVE = 1200
MAX_JOINT_LIMIT_RECOVERY_ACTIONS = 8
NATIVE_PLACEMENT_RRT_BUDGET_S = 12.0
NATIVE_PLACEMENT_RRT_MAX_CONFIG_CHECKS = 8000
NATIVE_PLACEMENT_RRT_MAX_ITERATIONS = 2000
NATIVE_REGION_PROGRESS_WAYPOINT_ACTION_BUDGET = 128
ARTICULATED_FIXTURE_PRECONTACT_CLEARANCE_M = 0.010
ARTICULATED_FIXTURE_PRECONTACT_APPROACH_DISTANCE_M = 0.180
ARTICULATED_FIXTURE_SLIDE_PRECONTACT_OUTWARD_M = 0.140
ARTICULATED_FIXTURE_SLIDE_PRECONTACT_HEIGHT_M = 0.220
ARTICULATED_FIXTURE_SLIDE_UNDER_TOP_HEIGHT_OFFSET_M = 0.094
ARTICULATED_FIXTURE_SLIDE_HANDLE_APPROACH_OUTWARD_M = 0.050
ARTICULATED_FIXTURE_SLIDE_HANDLE_EEF_HEIGHT_OFFSET_M = 0.0
ARTICULATED_FIXTURE_SLIDE_CONTACT_OUTWARD_OFFSET_M = 0.010
ARTICULATED_FIXTURE_SLIDE_TOOL_DOWN_ANGLE_RAD = float(np.deg2rad(25.0))
ARTICULATED_FIXTURE_PRECONTACT_ACTION_BUDGET = 400
ARTICULATED_FIXTURE_TOTAL_ACTION_BUDGET = 1000
ARTICULATED_FIXTURE_CONTACT_ACTION_BUDGET = ARTICULATED_FIXTURE_TOTAL_ACTION_BUDGET
ARTICULATED_FIXTURE_CONTACT_TRACKING_REGION_M = 0.065
ARTICULATED_FIXTURE_CONTACT_POSITION_TOLERANCE_M = 0.006
ARTICULATED_FIXTURE_ROUTE_POSITION_TOLERANCE_M = 0.004
ARTICULATED_FIXTURE_HINGE_FOLLOW_LEAD_RAD = 0.03
ARTICULATED_FIXTURE_HINGE_FOLLOW_MAX_LEAD_RAD = 0.04
ARTICULATED_FIXTURE_HINGE_FOLLOW_RAMP_STEP_RAD = 0.005
ARTICULATED_FIXTURE_SLIDE_FOLLOW_LEAD_M = 0.006
ARTICULATED_FIXTURE_SLIDE_FOLLOW_RAMP_STEP_M = 0.0005
ARTICULATED_FIXTURE_FOLLOW_POSITION_TOLERANCE_M = 0.00005
ARTICULATED_FIXTURE_FOLLOW_ACTION_CHUNK = 32
ARTICULATED_FIXTURE_SWITCH_TURN_GRIPPER_ACTION = 1.0


def _articulated_fixture_rrt_policy(
    policy: dict[str, Any], fixture_id: str,
) -> dict[str, Any]:
    """Keep a small route-following cushion from the articulated target fixture."""
    result = copy.deepcopy(policy)
    matches = []
    for group_name in ("forbidden_objects", "allowed_target_objects", "other_scene_objects"):
        groups = result.get(group_name, [])
        if not isinstance(groups, list):
            raise ValueError("articulated_fixture_route_policy_geometry_groups_must_be_lists")
        for group in groups:
            if isinstance(group, dict) and group.get("object_id") == fixture_id:
                group["minimum_clearance_m"] = ARTICULATED_FIXTURE_PRECONTACT_CLEARANCE_M
                matches.append(group_name)
    if len(matches) != 1:
        raise ValueError(
            "articulated_fixture_route_policy_must_contain_one_verified_fixture_binding:"
            f"{fixture_id}:{matches}"
        )
    result["source"] = (
        f"{policy.get('source', 'active_verified_geometry')}:"
        "articulated_fixture_OSC_tracking_clearance"
    )
    return result


def _articulated_fixture_press_quaternion(
    current_quaternion_wxyz: np.ndarray,
    outward_direction_world: np.ndarray,
) -> np.ndarray:
    """Aim the Panda finger axis inward and keep the finger pair tangent to a hinge path."""
    from scipy.spatial.transform import Rotation

    current = np.array(current_quaternion_wxyz, dtype=np.float64, copy=True)
    outward = np.array(outward_direction_world, dtype=np.float64, copy=True)
    if (
        current.shape != (4,) or outward.shape != (3,)
        or not np.isfinite(current).all() or not np.isfinite(outward).all()
        or float(np.linalg.norm(current)) <= 1e-12
        or float(np.linalg.norm(outward)) <= 1e-12
    ):
        raise ValueError("articulated_fixture_press_orientation_requires_finite_pose_and_direction")
    outward /= np.linalg.norm(outward)
    tool_z_world = -outward
    finger_axis_world = np.cross([0.0, 0.0, 1.0], outward)
    if float(np.linalg.norm(finger_axis_world)) <= 1e-8:
        current_rotation = Rotation.from_quat(current[[1, 2, 3, 0]] / np.linalg.norm(current))
        current_finger_axis = current_rotation.as_matrix()[:, 0]
        finger_axis_world = current_finger_axis - tool_z_world * float(
            np.dot(current_finger_axis, tool_z_world)
        )
    if float(np.linalg.norm(finger_axis_world)) <= 1e-8:
        reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        finger_axis_world = reference - tool_z_world * float(np.dot(reference, tool_z_world))
    finger_axis_world /= np.linalg.norm(finger_axis_world)
    tool_y_world = np.cross(tool_z_world, finger_axis_world)
    tool_y_world /= np.linalg.norm(tool_y_world)
    rotation = Rotation.from_matrix(
        np.column_stack([finger_axis_world, tool_y_world, tool_z_world])
    )
    quaternion_xyzw = rotation.as_quat()
    result = quaternion_xyzw[[3, 0, 1, 2]]
    result /= np.linalg.norm(result)
    current_unit = current / np.linalg.norm(current)
    if float(np.dot(result, current_unit)) < 0.0:
        result = -result
    return result


def _articulated_fixture_slide_grasp_quaternion(
    outward_direction_world: np.ndarray,
) -> np.ndarray:
    """Face down toward a slide handle while closing across its lateral bar."""
    from scipy.spatial.transform import Rotation

    outward = np.asarray(outward_direction_world, dtype=np.float64)
    if outward.shape != (3,) or not np.isfinite(outward).all():
        raise ValueError("slide_handle_outward_direction_must_be_finite_3d")
    norm = float(np.linalg.norm(outward))
    if norm < 1e-8:
        raise ValueError("slide_handle_outward_direction_must_be_nonzero")
    outward /= norm
    tool_axis = (
        -outward * float(np.cos(ARTICULATED_FIXTURE_SLIDE_TOOL_DOWN_ANGLE_RAD))
        + np.asarray([0.0, 0.0, -float(np.sin(ARTICULATED_FIXTURE_SLIDE_TOOL_DOWN_ANGLE_RAD))])
    )
    tool_axis /= np.linalg.norm(tool_axis)
    closing_axis = np.cross(np.asarray([0.0, 0.0, 1.0]), outward)
    closing_norm = float(np.linalg.norm(closing_axis))
    if closing_norm < 1e-8:
        raise ValueError("slide_handle_pull_axis_must_not_be_vertical")
    closing_axis /= closing_norm
    second_axis = np.cross(tool_axis, closing_axis)
    second_axis /= np.linalg.norm(second_axis)
    matrix = np.column_stack((closing_axis, second_axis, tool_axis))
    quaternion_xyzw = Rotation.from_matrix(matrix).as_quat()
    result = quaternion_xyzw[[3, 0, 1, 2]]
    return result / np.linalg.norm(result)


def _articulated_fixture_slide_approach_waypoints(
    contact_point_world_m: np.ndarray,
    outward_direction_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower outside the cabinet top, enter below it, then descend at the handle."""
    contact_point = np.asarray(contact_point_world_m, dtype=np.float64)
    outward = np.asarray(outward_direction_world, dtype=np.float64)
    if (
        contact_point.shape != (3,) or outward.shape != (3,)
        or not np.isfinite(contact_point).all() or not np.isfinite(outward).all()
    ):
        raise ValueError("slide_handle_approach_requires_finite_3d_points")
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm < 1e-8:
        raise ValueError("slide_handle_approach_outward_direction_must_be_nonzero")
    outward = outward / outward_norm
    under_top_height = float(
        contact_point[2] + ARTICULATED_FIXTURE_SLIDE_UNDER_TOP_HEIGHT_OFFSET_M
    )
    outside = contact_point + outward * ARTICULATED_FIXTURE_SLIDE_PRECONTACT_OUTWARD_M
    outside[2] = under_top_height
    front = contact_point + outward * ARTICULATED_FIXTURE_SLIDE_HANDLE_APPROACH_OUTWARD_M
    front[2] = under_top_height
    handle_height = front.copy()
    handle_height[2] = contact_point[2] + ARTICULATED_FIXTURE_SLIDE_HANDLE_EEF_HEIGHT_OFFSET_M
    return outside, front, handle_height


def _articulated_fixture_terminal_outcome(
    failure_reason: str | None, final_native_task_status: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Classify environment termination caused by a native success as a stop, not failure."""
    if (
        final_native_task_status.get("task_success") is True
        and failure_reason == "RuntimeError: environment_done_during_grasp_only_branch"
    ):
        return None, "native_task_success_terminated_environment"
    return failure_reason, None


def _articulated_fixture_follow_pose(
    anchor_eef_position_world: np.ndarray,
    anchor_eef_quaternion_wxyz: np.ndarray,
    articulation_context: dict[str, Any],
    *,
    joint_position_at_anchor: float | None = None,
    follow_lead: float | None = None,
    allow_terminal_stop_follow: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Track a fixed contact anchor with a bounded lead toward the fixture goal."""
    from scipy.spatial.transform import Rotation

    position = np.asarray(anchor_eef_position_world, dtype=np.float64)
    quaternion = np.asarray(anchor_eef_quaternion_wxyz, dtype=np.float64)
    joint = articulation_context.get("joint", {})
    axis = np.asarray(articulation_context.get("joint_axis_world"), dtype=np.float64)
    if (
        position.shape != (3,) or quaternion.shape != (4,) or axis.shape != (3,)
        or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
        or not np.isfinite(axis).all() or float(np.linalg.norm(axis)) <= 1e-12
    ):
        raise ValueError("articulated_fixture_follow_pose_requires_finite_live_joint_geometry")
    current = float(joint.get("position"))
    target = float(joint.get("target_position"))
    anchor = current if joint_position_at_anchor is None else float(joint_position_at_anchor)
    if not np.isfinite([current, target, anchor]).all():
        raise ValueError("articulated_fixture_follow_joint_positions_must_be_finite")
    axis /= np.linalg.norm(axis)
    eef_rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    joint_type = str(joint.get("joint_type", "")).lower()
    if joint_type == "hinge":
        origin = np.asarray(articulation_context.get("joint_origin_world_m"), dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("articulated_fixture_hinge_follow_requires_finite_world_origin")
        lead = ARTICULATED_FIXTURE_HINGE_FOLLOW_LEAD_RAD if follow_lead is None else float(follow_lead)
        if not np.isfinite(lead) or lead < 0.0:
            raise ValueError("articulated_fixture_hinge_follow_lead_must_be_finite_and_nonnegative")
        follow_target = target
        joint_range = joint.get("range")
        if allow_terminal_stop_follow and joint_range is not None:
            bounds = np.asarray(joint_range, dtype=np.float64)
            if bounds.shape == (2,) and np.isfinite(bounds).all():
                lower = float(bounds[0])
                if 0.0 <= target - lower <= 0.005:
                    follow_target = lower
        remaining = follow_target - current
        joint_position_ahead = current + np.sign(remaining) * min(abs(remaining), lead)
        fixture_delta = Rotation.from_rotvec(axis * (joint_position_ahead - anchor))
        target_position = origin + fixture_delta.apply(position - origin)
        target_rotation = fixture_delta * eef_rotation
    elif joint_type == "slide":
        lead = ARTICULATED_FIXTURE_SLIDE_FOLLOW_LEAD_M if follow_lead is None else float(follow_lead)
        if not np.isfinite(lead) or lead < 0.0:
            raise ValueError("articulated_fixture_slide_follow_lead_must_be_finite_and_nonnegative")
        remaining = target - current
        joint_position_ahead = current + np.sign(remaining) * min(abs(remaining), lead)
        target_position = position + axis * (joint_position_ahead - anchor)
        target_rotation = eef_rotation
    else:
        raise ValueError(f"articulated_fixture_follow_unsupported_joint_type:{joint_type}")
    target_quaternion_xyzw = target_rotation.as_quat()
    target_quaternion = target_quaternion_xyzw[[3, 0, 1, 2]]
    if float(np.dot(target_quaternion, quaternion)) < 0.0:
        target_quaternion = -target_quaternion
    return target_position, target_quaternion


def _articulated_fixture_robot_contact_pairs(
    record: dict[str, Any], fixture_id: str, articulated_geom_names: list[str] | None = None,
) -> list[dict[str, str]]:
    """Return observed robot contacts with the active fixture's moving collision geometry."""
    contact_report = record.get("contacts", {})
    contacts = contact_report.get("contacts", []) if isinstance(contact_report, dict) else []
    articulated_names = set(articulated_geom_names or [])
    if not articulated_names:
        return []
    result = []
    for contact in contacts:
        if not isinstance(contact, dict):
            continue
        geom1 = contact.get("geom1", {})
        geom2 = contact.get("geom2", {})
        body1 = contact.get("body1", {})
        body2 = contact.get("body2", {})
        name1 = str(geom1.get("name", ""))
        name2 = str(geom2.get("name", ""))
        body_name1 = str(body1.get("name", ""))
        body_name2 = str(body2.get("name", ""))
        robot1 = "robot0" in name1 or "gripper0" in name1 or "robot0" in body_name1
        robot2 = "robot0" in name2 or "gripper0" in name2 or "robot0" in body_name2
        fixture1 = fixture_id in name1 or fixture_id in body_name1
        fixture2 = fixture_id in name2 or fixture_id in body_name2
        if robot1 and fixture2:
            if not articulated_names or name2 in articulated_names:
                result.append({"robot_geom": name1, "fixture_geom": name2})
        elif robot2 and fixture1:
            if not articulated_names or name1 in articulated_names:
                result.append({"robot_geom": name2, "fixture_geom": name1})
    return result


def _thin_articulated_fixture_rrt_route(
    route: Any, *, stride: int = 2,
) -> tuple[np.ndarray, list[int]]:
    """Retain regularly spaced RRT EEF poses and always preserve its endpoint."""
    poses = np.asarray(route, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) < 2 or not np.isfinite(poses).all():
        raise ValueError("articulated_fixture_RRT_route_must_be_a_finite_N_by_7_pose_array")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("articulated_fixture_RRT_route_stride_must_be_a_positive_integer")
    indices = list(range(0, len(poses), stride))
    if indices[-1] != len(poses) - 1:
        indices.append(len(poses) - 1)
    return poses[indices], indices


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_payload_sha256(value: Any) -> str:
    """Hash a JSON-compatible planner identity without platform-specific spacing."""
    def normalize(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if isinstance(item, float) and not np.isfinite(item):
            raise ValueError("planner_identity_contains_non_finite_float")
        return item

    payload = json.dumps(
        normalize(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return _sha(payload)


def _keypose_rrt_planner_config(
    *, seed: int, search_time_budget_s: float,
    maximum_iterations: int, maximum_config_checks: int,
) -> dict[str, Any]:
    return {
        "step_rad": 0.22,
        "connect_max_steps": 16,
        "goal_bias": 0.18,
        "max_iterations": int(maximum_iterations),
        "max_config_checks": int(maximum_config_checks),
        "max_candidate_routes": 1,
        "search_time_budget_s": float(search_time_budget_s),
        "shortcut_attempts": 12,
        "minimum_clearance_m": 0.0,
        "start_joint_limit_tolerance_rad": 0.001,
        "require_safe_configurations": False,
        "seed": int(seed),
    }


def _task_success_first_grasp_search_command(
    args: argparse.Namespace,
    *,
    snapshot_root: Path,
    expert_root: Path,
    predicate: list[str],
    resume_from_runtime_snapshot: Path | None = None,
) -> list[str]:
    """Build the bounded grasp-search command with its complete candidate window."""
    command = [
        str(LIBERO_PYTHON), "-m", "research.gs_opd.repair.task_success_first_grasp_search",
        "--snapshot-root", str(snapshot_root), "--output-root", str(expert_root),
        "--benchmark", args.benchmark, "--level", str(args.level),
        "--task-id", str(args.task_id), "--init-state", str(args.init_state),
        "--seed", str(args.seed), "--camera-resolution", str(args.camera_resolution),
        "--port", str(args.expert_port),
        "--maximum-candidates", str(args.maximum_candidates),
        "--candidate-start-index", str(args.candidate_start_index),
        "--maximum-rrt-preflights", str(args.maximum_rrt_preflights),
        "--local-replan-rrt-time-budget-s", str(args.local_replan_rrt_time_budget_s),
        "--maximum-physical-branches", str(args.maximum_physical_branches),
        "--branch-control-action-budget", str(args.branch_control_action_budget),
        "--stage-a-unilateral-close-probe-position-m",
        str(getattr(args, "stage_a_unilateral_close_probe_position_m", 0.010)),
        "--stage-a-pregrasp-staged-clearance-height-m",
        str(getattr(args, "stage_a_pregrasp_staged_clearance_height_m", 0.0)),
        "--stage-a-pregrasp-lateral-detour-m",
        str(getattr(args, "stage_a_pregrasp_lateral_detour_m", 0.0)),
        "--rrt-time-budget-s", str(args.rrt_time_budget_s),
        "--approach-route-strategy", str(getattr(args, "approach_route_strategy", "auto")),
        "--geometry-candidate-pool-offset", str(getattr(args, "geometry_candidate_pool_offset", 0)),
        "--demonstration-horizon", str(args.demonstration_horizon),
        "--target-predicate", *predicate,
    ]
    if bool(getattr(args, "stage_a_pregrasp_lift_before_lateral_detour", False)):
        command.append("--stage-a-pregrasp-lift-before-lateral-detour")
    if resume_from_runtime_snapshot is not None:
        command.extend([
            "--resume-from-runtime-snapshot", str(resume_from_runtime_snapshot),
        ])
    candidate_ids = getattr(args, "candidate_id", None) or []
    if isinstance(candidate_ids, str):
        candidate_ids = [candidate_ids]
    for candidate_id in candidate_ids:
        command.extend(["--candidate-id", str(candidate_id)])
    for angle in getattr(args, "candidate_tilt_degrees", []) or []:
        command.extend(["--candidate-tilt-degrees", str(angle)])
    for angle in getattr(args, "candidate_roll_degrees", []) or []:
        command.extend(["--candidate-roll-degrees", str(angle)])
    if getattr(args, "candidate_coupled_perturbations", False):
        command.append("--candidate-coupled-perturbations")
    for offset_mm in getattr(args, "candidate_offset_object_z_mm", []) or []:
        command.extend(["--candidate-offset-object-z-mm", str(offset_mm)])
    for offset_mm in getattr(args, "candidate_offset_object_y_mm", []) or []:
        command.extend(["--candidate-offset-object-y-mm", str(offset_mm)])
    for offset_mm in getattr(args, "candidate_offset_object_x_mm", []) or []:
        command.extend(["--candidate-offset-object-x-mm", str(offset_mm)])
    if getattr(args, "contact_only_substep_audit", False):
        command.append("--contact-only-substep-audit")
    if args.resume_from_keypose_preflight_report is not None:
        command.extend([
            "--resume-from-keypose-preflight-report",
            str(args.resume_from_keypose_preflight_report.resolve()),
        ])
    if getattr(args, "cold_start", False):
        command.append("--cold-start")
    if getattr(args, "cold_start_continuation", False):
        command.append("--cold-start-continuation")
    if args.experience_library is not None:
        command.extend(["--experience-library", str(args.experience_library)])
    previous_search_report = getattr(args, "previous_search_report", None)
    if previous_search_report is not None:
        previous_reports = (
            previous_search_report
            if isinstance(previous_search_report, (list, tuple))
            else [previous_search_report]
        )
        for previous_report in previous_reports:
            command.extend(["--previous-search-report", str(Path(previous_report).resolve())])
    return command


def _task_success_collision_route_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Keep verified geometry bindings, but remove safety margins for route search."""
    result = copy.deepcopy(policy)
    result["minimum_clearance_m"] = 0.0
    if "measure_forbidden_object_pair_clearances" in result:
        result["measure_forbidden_object_pair_clearances"] = False
    for group_name in ("forbidden_objects", "allowed_target_objects", "other_scene_objects"):
        groups = result.get(group_name, [])
        if not isinstance(groups, list):
            raise ValueError("collision_route_policy_geometry_groups_must_be_lists")
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("collision_route_policy_geometry_group_must_be_an_object")
            group["minimum_clearance_m"] = 0.0
    result["source"] = (
        f"{policy.get('source', 'active_verified_geometry')}:"
        "task_success_collision_avoidance_zero_extra_margin"
    )
    return result


def _canonical_eef_pose_identity(pose: Any) -> dict[str, list[float]]:
    """Canonicalize an EEF pose, ignoring only sub-picometer serialization noise."""
    if isinstance(pose, dict):
        values = np.r_[
            np.asarray(pose.get("position_world_m"), dtype=np.float64),
            np.asarray(pose.get("quaternion_wxyz"), dtype=np.float64),
        ]
    else:
        values = np.asarray(pose, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("planner_goal_pose_must_be_finite_seven_vector")
    quaternion = values[3:].copy()
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("planner_goal_quaternion_is_degenerate")
    quaternion /= norm
    if quaternion[0] < 0:
        quaternion *= -1
    # A pose generated from the same rigid transform can differ in the last
    # few bits after matrix/quaternion round trips. This canonicalization is
    # much tighter than the controller's execution tolerances.
    return {
        "position_world_m": np.round(values[:3], 12).tolist(),
        "quaternion_wxyz": np.round(quaternion, 12).tolist(),
    }


def _make_precomputed_rrt_route_cache(
    *, result: dict[str, Any], model_sha256: str, bddl_sha256: str,
    start_state_sha256: str, goal_pose_world: Any,
    safety_geometry_policy: dict[str, Any], planner_config: dict[str, Any],
) -> dict[str, Any] | None:
    if result.get("status") != "offline_feasible":
        return None
    qpos_path = np.asarray(result.get("qpos_path"), dtype=np.float64)
    eef_path = np.asarray(result.get("eef_reference_poses_world"), dtype=np.float64)
    if (
        qpos_path.ndim != 2 or qpos_path.shape[1:] != (7,)
        or eef_path.ndim != 2 or eef_path.shape[1:] != (7,)
        or len(qpos_path) < 2 or len(qpos_path) != len(eef_path)
        or not np.isfinite(qpos_path).all() or not np.isfinite(eef_path).all()
    ):
        raise ValueError("feasible_RRT_result_has_invalid_or_unaligned_joint_and_EEF_paths")
    identity = {
        "model_geometry_sha256": str(model_sha256),
        "bddl_sha256": str(bddl_sha256),
        "start_state_sha256": str(start_state_sha256),
        "goal_pose_world": _canonical_eef_pose_identity(goal_pose_world),
        "safety_geometry_policy_sha256": _canonical_payload_sha256(safety_geometry_policy),
        "planner_config_sha256": _canonical_payload_sha256(planner_config),
    }
    paths = {
        "qpos_path_rad": qpos_path.tolist(),
        "eef_reference_poses_world": eef_path.tolist(),
    }
    return {
        "schema_version": "gs_opd_precomputed_rrt_route.v1",
        "identity": identity,
        "identity_sha256": _canonical_payload_sha256(identity),
        "paths": paths,
        "paths_sha256": _canonical_payload_sha256(paths),
        "planner_status": result.get("status"),
        "planning_seconds": result.get("planning_seconds"),
        "configuration_checks": result.get("config_checks"),
        "iterations": result.get("iterations"),
        "safety_certified": False,
    }


def _validate_precomputed_rrt_route(
    cache: Any, *, model_sha256: str, bddl_sha256: str,
    start_state_sha256: str, goal_pose_world: Any,
    safety_geometry_policy: dict[str, Any], planner_config: dict[str, Any],
) -> dict[str, Any]:
    """Accept a route only when all planning inputs and both paths still match."""
    if not isinstance(cache, dict) or cache.get("schema_version") != "gs_opd_precomputed_rrt_route.v1":
        return {"reusable": False, "reason": "route_cache_missing_or_unsupported"}
    identity = cache.get("identity")
    if not isinstance(identity, dict):
        return {"reusable": False, "reason": "route_cache_identity_missing"}
    if cache.get("identity_sha256") != _canonical_payload_sha256(identity):
        return {"reusable": False, "reason": "route_cache_identity_integrity_mismatch"}
    expected = {
        "model_geometry_sha256": str(model_sha256),
        "bddl_sha256": str(bddl_sha256),
        "start_state_sha256": str(start_state_sha256),
        "goal_pose_world": _canonical_eef_pose_identity(goal_pose_world),
        "safety_geometry_policy_sha256": _canonical_payload_sha256(safety_geometry_policy),
        "planner_config_sha256": _canonical_payload_sha256(planner_config),
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            return {"reusable": False, "reason": f"route_cache_{key}_mismatch"}
    paths = cache.get("paths")
    if not isinstance(paths, dict) or cache.get("paths_sha256") != _canonical_payload_sha256(paths):
        return {"reusable": False, "reason": "route_cache_path_integrity_mismatch"}
    qpos_path = np.asarray(paths.get("qpos_path_rad"), dtype=np.float64)
    eef_path = np.asarray(paths.get("eef_reference_poses_world"), dtype=np.float64)
    if (
        qpos_path.ndim != 2 or qpos_path.shape[1:] != (7,)
        or eef_path.ndim != 2 or eef_path.shape[1:] != (7,)
        or len(qpos_path) < 2 or len(qpos_path) != len(eef_path)
        or not np.isfinite(qpos_path).all() or not np.isfinite(eef_path).all()
    ):
        return {"reusable": False, "reason": "route_cache_joint_and_EEF_paths_invalid"}
    return {
        "reusable": True,
        "reason": "all_planning_inputs_and_route_paths_match",
        "qpos_path": qpos_path,
        "eef_reference_poses_world": eef_path,
        "route_identity_sha256": cache["identity_sha256"],
        "route_paths_sha256": cache["paths_sha256"],
    }


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _integration_state(snapshot_state: dict[str, Any]) -> np.ndarray:
    value = np.asarray(snapshot_state.get("mujoco_integration_state", []), dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError("runtime_snapshot_integration_state_is_invalid")
    return value


def _state_digest(value: Any) -> str:
    return _sha(np.ascontiguousarray(value, dtype=np.float64).tobytes())


def _keypose_rrt_call_count(candidate_rows: list[dict[str, Any]]) -> int:
    """Count planner calls with recorded runtimes, excluding budget placeholders."""
    count = 0

    def was_called(result: Any) -> bool:
        return isinstance(result, dict) and result.get("planning_seconds") is not None

    for row in candidate_rows:
        preplace = row.get("preplace_RRT")
        if isinstance(preplace, dict):
            initial = preplace.get("initial_RRT_attempt")
            extended = preplace.get("bounded_extended_RRT_attempt")
            if isinstance(initial, dict) or isinstance(extended, dict):
                count += int(was_called(initial)) + int(was_called(extended))
            else:
                count += int(was_called(preplace))
        terminal = row.get("terminal_pose_IK_RRT_diagnostic")
        count += int(was_called(terminal))
    return count


def _task_success_waypoint_fallback_reason(preplace_result: Any) -> str | None:
    """Identify a valid-IK waypoint whose RRT stopped only at its time bound."""
    if not isinstance(preplace_result, dict):
        return None
    if (
        preplace_result.get("status") == "no_safe_path"
        and "planning_time_budget_exhausted" in str(preplace_result.get("failure_reason", ""))
        and int(preplace_result.get("ik_candidate_count", 0)) > 0
    ):
        return "RRT_time_budget_exhausted_with_valid_IK;_try_bounded_real_OSC_waypoint"
    return None


def _preflight_row_is_task_success_admissible(row: dict[str, Any]) -> bool:
    preplace = row.get("preplace_RRT", {})
    terminal = row.get("terminal_pose_IK_RRT_diagnostic", {})
    if preplace.get("status") == "offline_feasible":
        return bool(
            row.get("admitted_to_physical_search") is True
            or terminal.get("status") == "budget_exhausted"
            or (
                int(terminal.get("ik_candidate_count", 0)) > 0
                and terminal.get("status") in {"offline_feasible", "no_safe_path"}
            )
        )
    return _task_success_waypoint_fallback_reason(preplace) is not None


def _verify_planner_live_state(
    planner_result: dict[str, Any],
    observed_state_sha256: str,
    expected_start_state_sha256: str,
) -> dict[str, Any]:
    """Use the post-planner live integration state as mutation authority."""
    unchanged = bool(
        isinstance(observed_state_sha256, str)
        and observed_state_sha256 == expected_start_state_sha256
    )
    return {
        "status": "exact_match" if unchanged else "state_changed",
        "live_state_unchanged": unchanged,
        "planner_reported_live_state_unchanged": planner_result.get("live_state_unchanged"),
        "observed_state_sha256": observed_state_sha256,
        "expected_start_state_sha256": expected_start_state_sha256,
        "authority": "post_planner_exact_MuJoCo_integration_state_hash",
    }


def _transport_exception_outcome(exc: Exception, final_native: dict[str, Any]) -> dict[str, Any]:
    """Keep a post-success Bridge terminal exception from downgrading BDDL success."""
    reason = f"{type(exc).__name__}: {exc}"
    if isinstance(final_native, dict) and final_native.get("task_success") is True:
        return {
            "status": "complete_task_success",
            "failure_reason": None,
            "post_success_terminal_exception": reason,
        }
    return {
        "status": "native_goal_transport_failed",
        "failure_reason": reason,
        "post_success_terminal_exception": None,
    }


def _native_episode_terminated(native: dict[str, Any] | None) -> bool:
    """Return whether the environment has reached its terminal timestep."""
    return isinstance(native, dict) and native.get("done") is True


def _object_state(snapshot: dict[str, Any], body_id: int) -> dict[str, Any]:
    matches = [
        row for row in snapshot.get("object_states", {}).values()
        if int(row.get("body_id", -1)) == int(body_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"active_target_body_state_not_unique:{body_id}:{len(matches)}")
    return matches[0]


def _finger_contacts_clear(finger_contacts: dict[str, Any]) -> bool:
    """Require explicit empty, model-verified target contact on both pads.

    Safety Bridge reports ``status=fail`` when a bilateral grasp is absent,
    including the valid released state with zero matched contacts.  The
    previous ``status=pass`` check therefore made the release transition
    unreachable after a physically successful release.
    """
    if not isinstance(finger_contacts, dict):
        return False
    matched = finger_contacts.get("matched_contacts")
    return bool(
        finger_contacts.get("status") in {"pass", "fail"}
        and finger_contacts.get("left") is False
        and finger_contacts.get("right") is False
        and finger_contacts.get("contact_count") == 0
        and isinstance(matched, dict)
        and matched.get("left") == []
        and matched.get("right") == []
    )


def _object_contact_escape_direction(
    contacts: dict[str, Any], target_binding: dict[str, Any],
    target_root_position_world_m: Any,
) -> dict[str, Any] | None:
    """Infer an object-outward direction from live target/scene contacts."""
    if not isinstance(contacts, dict) or contacts.get("status") not in {"contact", "no_contact"}:
        return None
    target_names = {str(name) for name in target_binding.get("geom_names", [])}
    root = np.asarray(target_root_position_world_m, dtype=np.float64)
    if not target_names or root.shape != (3,) or not np.isfinite(root).all():
        return None
    weighted: list[tuple[float, np.ndarray, np.ndarray, str]] = []
    for row in contacts.get("contacts", []):
        if not isinstance(row, dict):
            continue
        geom1 = row.get("geom1", {})
        geom2 = row.get("geom2", {})
        name1, name2 = str(geom1.get("name", "")), str(geom2.get("name", ""))
        target_is_1, target_is_2 = name1 in target_names, name2 in target_names
        if target_is_1 == target_is_2:
            continue
        blocker_name = name2 if target_is_1 else name1
        if blocker_name.startswith(("robot0_", "gripper0_")):
            continue
        normal = np.asarray(row.get("normal_geom1_to_geom2"), dtype=np.float64)
        position = np.asarray(row.get("position"), dtype=np.float64)
        if (
            normal.shape != (3,) or position.shape != (3,)
            or not np.isfinite(normal).all() or not np.isfinite(position).all()
        ):
            continue
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        # MuJoCo's contact normal points geom1 -> geom2. Move the target
        # along the opposite direction when it is geom1, and along the
        # contact normal when it is geom2.
        outward = (-normal if target_is_1 else normal) / norm
        force = np.asarray(row.get("force_and_torque", [])[:3], dtype=np.float64)
        force_norm = float(np.linalg.norm(force)) if force.shape == (3,) and np.isfinite(force).all() else 0.0
        weighted.append((max(1.0, force_norm), outward, position, blocker_name))
    if not weighted:
        return None
    resultant = sum((weight * direction for weight, direction, _, _ in weighted), np.zeros(3))
    norm = float(np.linalg.norm(resultant))
    if norm <= 1e-9:
        _, direction, _, _ = max(weighted, key=lambda row: row[0])
    else:
        direction = resultant / norm
    support_radius = max(
        0.0,
        max(float(np.dot(root - position, direction)) for _, _, position, _ in weighted),
    )
    return {
        "direction_world": direction.tolist(),
        "blocker_geom_names": sorted({row[3] for row in weighted}),
        "contact_count": len(weighted),
        "contact_positions_world_m": [row[2].tolist() for row in weighted],
        "contact_escape_source": "live_MuJoCo_target_object_to_scene_contact_normals",
        "support_radius_estimate_m": support_radius,
    }


def _robot_scene_contact_escape_direction(
    contacts: dict[str, Any],
    snapshot_state: dict[str, Any],
    *,
    target_body_id: int,
    eef_position_world_m: Any,
) -> dict[str, Any] | None:
    """Find a horizontal escape direction for live robot-to-scene contacts.

    Target-object contact escape alone misses a frequent transport failure:
    the carried object is clear, but an upstream Panda link or the palm is
    already pressing a nearby scene object. Use actual MuJoCo contact points
    and the contacted body's live root pose to move the held object away before
    requesting a vertical lift. This is a Stage A motion hypothesis; contacts
    and native BDDL still decide the physical result.
    """
    if (
        not isinstance(contacts, dict)
        or contacts.get("status") not in {"contact", "no_contact"}
        or not isinstance(snapshot_state, dict)
    ):
        return None
    objects = snapshot_state.get("object_states")
    if not isinstance(objects, dict):
        return None
    objects_by_body = {
        int(row.get("body_id", -1)): row
        for row in objects.values()
        if isinstance(row, dict)
    }
    eef = np.asarray(eef_position_world_m, dtype=np.float64)
    if eef.shape != (3,) or not np.isfinite(eef).all():
        return None
    # Aggregate each contacted scene body independently. Selecting the
    # strongest resultant avoids cancellation between unrelated obstacles.
    grouped: dict[int, list[tuple[float, np.ndarray, np.ndarray, str, str]]] = {}
    for row in contacts.get("contacts", []):
        if not isinstance(row, dict):
            continue
        body1 = row.get("body1", {})
        body2 = row.get("body2", {})
        name1, name2 = str(body1.get("name", "")), str(body2.get("name", ""))
        first_robot = name1.startswith(("robot0_", "gripper0_"))
        second_robot = name2.startswith(("robot0_", "gripper0_"))
        if first_robot == second_robot:
            continue
        robot_body = body1 if first_robot else body2
        scene_body = body2 if first_robot else body1
        scene_name = str(scene_body.get("name", ""))
        scene_id = int(scene_body.get("id", -1))
        if (
            scene_id == int(target_body_id)
            or scene_name in {"floor", "table", "table_top"}
        ):
            continue
        scene_state = objects_by_body.get(scene_id)
        position = np.asarray(row.get("position"), dtype=np.float64)
        if (
            not isinstance(scene_state, dict)
            or position.shape != (3,)
            or not np.isfinite(position).all()
        ):
            continue
        root = np.asarray(scene_state.get("root_pos"), dtype=np.float64)
        if root.shape != (3,) or not np.isfinite(root).all():
            continue
        outward = position - root
        outward[2] = 0.0
        norm = float(np.linalg.norm(outward))
        if norm <= 1e-6:
            outward = eef - root
            outward[2] = 0.0
            norm = float(np.linalg.norm(outward))
        if norm <= 1e-6:
            continue
        outward /= norm
        force = np.asarray(row.get("force_and_torque", [])[:3], dtype=np.float64)
        force_norm = (
            float(np.linalg.norm(force))
            if force.shape == (3,) and np.isfinite(force).all() else 0.0
        )
        grouped.setdefault(scene_id, []).append((
            max(0.1, force_norm), outward, position, scene_name,
            str(robot_body.get("name", "")),
        ))
    if not grouped:
        return None
    candidates = []
    for scene_id, rows in grouped.items():
        resultant = sum((weight * direction for weight, direction, *_ in rows), np.zeros(3))
        norm = float(np.linalg.norm(resultant))
        if norm <= 1e-6:
            weight, direction, *_ = max(rows, key=lambda item: item[0])
            del weight
        else:
            direction = resultant / norm
        total_weight = sum(row[0] for row in rows)
        root = np.asarray(objects_by_body[scene_id]["root_pos"], dtype=np.float64)
        radial_extent = max(
            float(np.linalg.norm((position - root)[:2]))
            for _, _, position, _, _ in rows
        )
        candidates.append((total_weight, direction, rows, radial_extent))
    total_weight, direction, rows, radial_extent = max(
        candidates, key=lambda item: item[0],
    )
    return {
        "direction_world": direction.tolist(),
        "scene_body_id": int(next(
            body_id for body_id, grouped_rows in grouped.items()
            if grouped_rows is rows
        )),
        "scene_body_name": rows[0][3],
        "robot_body_names": sorted({row[4] for row in rows}),
        "contact_count": len(rows),
        "contact_positions_world_m": [row[2].tolist() for row in rows],
        "weighted_contact_force_norm": float(total_weight),
        "contact_radial_extent_m": float(radial_extent),
        "escape_direction_source": "live_MuJoCo_robot_scene_contact_point_radial",
        "safety_certified": False,
    }


def _task_values(native: dict[str, Any]) -> tuple[dict[tuple[str, ...], bool], bool]:
    values: dict[tuple[str, ...], bool] = {}
    for row in native.get("native_goal_values", []):
        predicate, value = row.get("predicate"), row.get("value")
        if isinstance(predicate, list) and isinstance(value, bool):
            values[tuple(str(item) for item in predicate)] = value
    if native.get("status") != "complete" or native.get("task_success") is None or not values:
        raise ValueError("native_BDDL_goal_status_is_incomplete")
    return values, bool(native["task_success"])


def _joint_limit_diagnostics(
    joint_qpos: Any,
    joint_names: list[str],
    joint_ranges: Any,
    *,
    tolerance_rad: float = 1e-8,
) -> dict[str, Any]:
    qpos = np.asarray(joint_qpos, dtype=np.float64)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if (
        qpos.ndim != 1 or ranges.shape != (len(qpos), 2)
        or len(joint_names) != len(qpos) or not np.isfinite(qpos).all()
        or not np.isfinite(ranges).all() or not np.isfinite(tolerance_rad)
        or tolerance_rad < 0.0
    ):
        raise ValueError("joint_limit_diagnostics_require_aligned_finite_state_and_ranges")
    violations = []
    for name, value, (lower, upper) in zip(joint_names, qpos, ranges):
        lower_excess = max(0.0, float(lower - value))
        upper_excess = max(0.0, float(value - upper))
        excess = max(lower_excess, upper_excess)
        violations.append({
            "joint_name": str(name),
            "position_rad": float(value),
            "lower_rad": float(lower),
            "upper_rad": float(upper),
            "lower_excess_rad": lower_excess,
            "upper_excess_rad": upper_excess,
            "excess_rad": excess,
            "within_limits": excess <= tolerance_rad,
        })
    maximum_excess = max((row["excess_rad"] for row in violations), default=0.0)
    return {
        "within_limits": maximum_excess <= tolerance_rad,
        "maximum_excess_rad": maximum_excess,
        "violating_joint_count": sum(not row["within_limits"] for row in violations),
        "joints": violations,
    }


def _live_joint_limit_specification(repair_context: dict[str, Any]) -> tuple[list[str], Any]:
    """Read finite Panda joint identity/ranges from the active Bridge controller context."""
    robots = repair_context.get("robots") if isinstance(repair_context, dict) else None
    if (
        not isinstance(repair_context, dict)
        or repair_context.get("status") != "available"
        or not isinstance(robots, list) or len(robots) != 1
        or not isinstance(robots[0], dict)
        or robots[0].get("status") != "available"
    ):
        raise RuntimeError("joint_limit_recovery_requires_live_controller_joint_ranges")
    names = robots[0].get("joint_names")
    ranges = robots[0].get("joint_ranges")
    if not isinstance(names, list) or not names or ranges is None:
        raise RuntimeError("joint_limit_recovery_requires_live_controller_joint_ranges")
    return [str(name) for name in names], ranges


def _transport_grasp_loss_reason(record: dict[str, Any]) -> str | None:
    phase = str(record.get("phase", "")) if isinstance(record, dict) else ""
    if not (phase.startswith("transport_") or phase == "precise_placement"):
        return None
    grasp = record.get("grasp_state")
    if isinstance(grasp, dict) and grasp.get("grasp_confirmed") is False:
        return "bilateral_grasp_lost_during_native_goal_transport"
    return None


def _native_transport_action_budget(
    route: Any,
    native_task_status: dict[str, Any],
    *,
    predicate: list[str],
    demonstration_horizon: int,
    observed_speed_m_s: float = 0.012,
    native_region_route_includes_final_placement: bool = False,
) -> dict[str, Any]:
    """Allocate transport actions from route length and the remaining demo horizon."""
    from research.gs_opd.repair.task_success_first_search import route_control_action_budget

    timestep = native_task_status.get("timestep")
    if (
        isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0
        or isinstance(demonstration_horizon, bool)
        or not isinstance(demonstration_horizon, int)
        or demonstration_horizon <= timestep
    ):
        raise ValueError("native_transport_action_budget_requires_remaining_demo_horizon")
    pending = pending_object_goals(native_task_status)
    future_goals = [row for row in pending if list(row) != list(predicate)]
    completion_reserve = (
        MAX_NATIVE_PLACEMENT_ACTIONS
        + MAX_NATIVE_RELEASE_ACTIONS
        + MAX_NATIVE_WITHDRAWAL_ACTIONS
        + MAX_NATIVE_SETTLE_ACTIONS
    )
    if native_region_route_includes_final_placement:
        # This Stage A route tracks the current object through its native region
        # placement waypoint. Do not also reserve the full placement/release/
        # withdrawal/settle bundle for every intermediate goal in a multi-goal
        # task: that double-count can consume the entire remaining horizon before
        # the current goal takes a physical step. The future-goal reserve below
        # is capped only after preserving the estimated route budget.
        completion_reserve = 0
    estimated_actions = route_control_action_budget(
        route,
        observed_speed_m_s=observed_speed_m_s,
        minimum_actions=74,
        maximum_actions=10_000_000,
    )
    remaining_actions = demonstration_horizon - timestep
    requested_future_reserve = len(future_goals) * FUTURE_OBJECT_GOAL_ACTION_RESERVE
    # A future-object reserve must not consume the current goal's route budget.
    # In short standard episodes, cap it to slack remaining after the estimated
    # current route and required placement/release/withdrawal/settle actions.
    future_reserve = min(
        requested_future_reserve,
        max(0, remaining_actions - completion_reserve - estimated_actions),
    )
    available = max(0, remaining_actions - completion_reserve - future_reserve)
    if available == 0:
        raise RuntimeError("native_transport_action_budget_reserved_for_completion_and_pending_goals")
    return {
        "actions": min(estimated_actions, available),
        "route_length_m": float(np.linalg.norm(
            np.diff(np.asarray(route, dtype=np.float64)[:, :3], axis=0), axis=1,
        ).sum()),
        "estimated_actions_for_route": estimated_actions,
        "available_after_reserves": available,
        "current_goal_completion_reserve": completion_reserve,
        "future_pending_object_goal_count": len(future_goals),
        "future_goal_reserve_each": FUTURE_OBJECT_GOAL_ACTION_RESERVE,
        "future_goal_reserve_requested": requested_future_reserve,
        "future_goal_reserve_applied": future_reserve,
        "future_goal_reserve_limited_to_preserve_current_route": future_reserve < requested_future_reserve,
        "remaining_actions_at_budget_start": remaining_actions,
        "timestep_at_budget_start": timestep,
        "demonstration_horizon": demonstration_horizon,
        "observed_speed_assumption_m_s": observed_speed_m_s,
        "termination_policy": "route_or_remaining_horizon_budget; no fixed per-transport action cap",
        "native_region_route_includes_final_placement": native_region_route_includes_final_placement,
    }


def _recover_joint_limits_with_osc(
    bridge,
    *,
    target_binding: dict[str, Any],
    instruction: str,
    safety_policy: dict[str, Any],
    controller_fields: dict[str, Any],
    output_dir: Path,
    fallback_snapshot: dict[str, Any] | None,
    defer_micro_violation_to_transport: bool = False,
) -> dict[str, Any]:
    """Try a short physical OSC retreat from a near-limit state, then restore a verified hold."""
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _record_step,
        _snapshot_state_vector,
        _write_recording,
    )
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor
    from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action
    from scipy.spatial.transform import Rotation

    output_dir.mkdir(parents=True, exist_ok=True)
    observed = bridge.request({"op": "observe", "include_snapshot_state": True})
    robot = observed["snapshot_state"]["robot_states"][0]
    repair = bridge.request({"op": "repair_context"})
    joint_names, joint_ranges = _live_joint_limit_specification(repair)
    initial_limits = _joint_limit_diagnostics(
        robot.get("joint_qpos"), joint_names, joint_ranges,
    )
    if initial_limits["within_limits"]:
        return {
            "status": "not_needed",
            "initial_joint_limits": initial_limits,
            "actions": 0,
            "safety_certified": False,
        }
    if (
        defer_micro_violation_to_transport
        # Native live-IK rejects any out-of-range qpos before it can select a
        # route. Even a 0.097 mrad joint7 excess left grasped-object posture
        # search with zero verified waypoints and caused OSC stagnation. Defer
        # only sub-0.05 mrad integration noise; physically recover larger
        # excess before route generation. The 0.05 mrad boundary was already
        # physically verified on a 0.141 mrad joint7 excess.
        and initial_limits["maximum_excess_rad"] <= 0.00005
    ):
        return {
            "status": "micro_violation_deferred_to_transport_route",
            "initial_joint_limits": initial_limits,
            "actions": 0,
            "physical_actions": 0,
            "deferred_limit_rad": 0.00005,
            "bilateral_grasp_must_remain_confirmed_during_physical_transport": True,
            "safety_certified": False,
        }

    start_snapshot = bridge.request({"op": "export_runtime_snapshot"})
    if start_snapshot.get("status") != "complete" or not isinstance(start_snapshot.get("payload"), bytes):
        raise RuntimeError("out_of_limit_recovery_start_snapshot_export_failed")
    start_path = output_dir / "limit_recovery_start_runtime_snapshot.pkl"
    start_path.write_bytes(start_snapshot["payload"])
    monitor = GraspStabilityMonitor(required_samples=3)
    records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    durable_records = output_dir / "durable_physical_step_records.jsonl"
    durable_frames = output_dir / "durable_dual_camera_frames"
    durable_frames.mkdir(parents=True, exist_ok=True)
    max_excess = float(initial_limits["maximum_excess_rad"])
    reason = None
    try:
        grasp = bridge.request({
            "op": "grasp_state",
            "carried_binding": target_binding,
            "expected_instruction": instruction,
        })
        if (
            grasp.get("grasp_confirmed") is not True
            or grasp.get("finger_contacts", {}).get("left") is not True
            or grasp.get("finger_contacts", {}).get("right") is not True
        ):
            reason = "out_of_limit_snapshot_does_not_have_verified_bilateral_grasp"
        elif max_excess > 0.02:
            reason = "joint_limit_excess_exceeds_bounded_OSC_recovery_range"
        else:
            for action_index in range(MAX_JOINT_LIMIT_RECOVERY_ACTIONS):
                current = bridge.request({"op": "observe", "include_snapshot_state": True})
                current_robot = current["snapshot_state"]["robot_states"][0]
                current_pos = np.asarray(current_robot["eef_pos"], dtype=np.float64)
                current_quat = np.asarray(current_robot["eef_quat"], dtype=np.float64)
                target_pos = current_pos + np.asarray([0.0, 0.0, 0.002], dtype=np.float64)
                before = _joint_limit_diagnostics(
                    current_robot.get("joint_qpos"), joint_names, joint_ranges,
                )
                limiting_joint = max(
                    before["joints"], key=lambda row: float(row["excess_rad"]),
                )
                target_quat = current_quat.copy()
                recovery_rotation = None
                if str(limiting_joint["joint_name"]).lower().endswith("joint7"):
                    # Panda joint7 is the wrist roll about the EEF local z axis.
                    # A vertical OSC retreat barely changes this joint, as the
                    # recorded route-stagnation Snapshot demonstrated; unwind
                    # it directly with a small real world-frame pose rotation.
                    current_rotation = Rotation.from_quat(
                        current_quat[[1, 2, 3, 0]],
                    )
                    eef_roll_axis_world = current_rotation.as_matrix()[:, 2]
                    direction = (
                        1.0 if limiting_joint["lower_excess_rad"] > 0.0 else -1.0
                    )
                    rotation_vector = eef_roll_axis_world * direction * 0.01
                    target_rotation = (
                        Rotation.from_rotvec(rotation_vector) * current_rotation
                    )
                    xyzw = target_rotation.as_quat()
                    target_quat = xyzw[[3, 0, 1, 2]]
                    recovery_rotation = {
                        "joint_name": limiting_joint["joint_name"],
                        "axis_world": eef_roll_axis_world.tolist(),
                        "direction": direction,
                        "command_rotation_rad": 0.01,
                    }
                action = world_pose_tracking_action(
                    current_pos, current_quat, target_pos, target_quat,
                    controller_fields, gripper_action=0.0,
                ).astype(np.float32)
                _record_step(
                    bridge, action, "joint_limit_OSC_recovery", "native_goal_recovery",
                    action_index, target_binding, instruction, safety_policy, monitor,
                    records, frames, raise_on_done=False,
                    planning_tracking={
                        "tracking_mode": "bounded_native_OSC_pose_recovery_from_joint_limit_violation",
                        "target_eef_pose_world": np.r_[target_pos, target_quat].tolist(),
                        "joint_limits_before": before,
                        "limiting_joint_wrist_unwind": recovery_rotation,
                        "safety_certified": False,
                    },
                )
                record = records[-1]
                after_robot = record.get("snapshot_state", {}).get("robot_states", [{}])[0]
                after = _joint_limit_diagnostics(
                    after_robot.get("joint_qpos"), joint_names, joint_ranges,
                )
                if record.get("done"):
                    reason = "environment_done_during_joint_limit_OSC_recovery"
                    break
                if record.get("grasp_state", {}).get("grasp_confirmed") is not True:
                    reason = "bilateral_grasp_lost_during_joint_limit_OSC_recovery"
                    break
                with durable_records.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(
                        record, ensure_ascii=False, separators=(",", ":"),
                        default=lambda item: item.tolist() if isinstance(item, np.ndarray) else str(item),
                    ))
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                frame = frames[-1]
                combined = np.concatenate((frame["agentview_rgb"], frame["wrist_rgb"]), axis=1)
                if not cv2.imwrite(
                    str(durable_frames / f"frame_{action_index:04d}.jpg"),
                    cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
                ):
                    raise RuntimeError("joint_limit_recovery_dual_camera_frame_write_failed")
                if after["within_limits"]:
                    break
                if after["maximum_excess_rad"] >= max_excess - 1e-6:
                    reason = "OSC_recovery_did_not_reduce_joint_limit_excess"
                    break
                max_excess = float(after["maximum_excess_rad"])
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"

    if records:
        _write_recording(output_dir, records, frames)
    current = bridge.request({"op": "observe", "include_snapshot_state": True})
    current_robot = current["snapshot_state"]["robot_states"][0]
    current_limits = _joint_limit_diagnostics(
        current_robot.get("joint_qpos"), joint_names, joint_ranges,
    )
    current_grasp = bridge.request({
        "op": "grasp_state",
        "carried_binding": target_binding,
        "expected_instruction": instruction,
    })
    if current_limits["within_limits"] and current_grasp.get("grasp_confirmed") is True:
        recovered_snapshot = bridge.request({"op": "export_runtime_snapshot"})
        recovered_path = output_dir / "OSC_recovered_runtime_snapshot.pkl"
        if recovered_snapshot.get("status") == "complete" and isinstance(recovered_snapshot.get("payload"), bytes):
            recovered_path.write_bytes(recovered_snapshot["payload"])
        return {
            "status": "recovered_by_real_OSC",
            "initial_joint_limits": initial_limits,
            "final_joint_limits": current_limits,
            "physical_actions": len(records),
            "bilateral_grasp_retained": True,
            "start_snapshot": str(start_path),
            "recovered_snapshot": str(recovered_path) if recovered_path.is_file() else None,
            "reason": reason,
            "safety_certified": False,
            "artifacts": {
                "records": str(output_dir / "physical_step_records.jsonl.gz"),
                "actions_states": str(output_dir / "task_success_first_actions_states.npz"),
                "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                "durable_records": str(durable_records),
                "durable_frames": str(durable_frames),
            },
        }

    if fallback_snapshot is not None:
        restored = bridge.request({
            "op": "restore_runtime_snapshot", "payload": fallback_snapshot["payload"],
        })
        actual = _snapshot_state_vector(restored.get("snapshot_state", {}))
        expected = _snapshot_state_vector(fallback_snapshot["snapshot_state"])
        if (
            restored.get("status") != "restored"
            or restored.get("model_geometry_sha256") != fallback_snapshot.get("model_geometry_sha256")
            or actual.shape != expected.shape or not np.array_equal(actual, expected)
        ):
            raise RuntimeError("verified_joint_limit_fallback_snapshot_exact_restore_failed")
        native = bridge.request({"op": "native_task_status"})
        values, success = _task_values(native)
        if success:
            raise RuntimeError("joint_limit_fallback_snapshot_unexpectedly_has_task_success")
        fallback_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
        fallback_robot = fallback_observation["snapshot_state"]["robot_states"][0]
        fallback_limits = _joint_limit_diagnostics(
            fallback_robot.get("joint_qpos"), joint_names, joint_ranges,
        )
        fallback_grasp = bridge.request({
            "op": "grasp_state",
            "carried_binding": target_binding,
            "expected_instruction": instruction,
        })
        if not fallback_limits["within_limits"] or fallback_grasp.get("grasp_confirmed") is not True:
            raise RuntimeError("fallback_runtime_snapshot_is_not_a_valid_held_object_start")
        return {
            "status": "restored_verified_legal_runtime_snapshot",
            "initial_joint_limits": initial_limits,
            "failed_OSC_joint_limits": current_limits,
            "fallback_joint_limits": fallback_limits,
            "physical_recovery_actions": len(records),
            "bilateral_grasp_retained": True,
            "recovery_failure_reason": reason,
            "fallback_snapshot": fallback_snapshot["path"],
            "fallback_snapshot_sha256": fallback_snapshot["payload_sha256"],
            "fallback_integration_state_sha256": _state_digest(actual),
            "fallback_native_goal_values": native.get("native_goal_values"),
            "safety_certified": False,
            "artifacts": {
                "records": str(output_dir / "physical_step_records.jsonl.gz"),
                "actions_states": str(output_dir / "task_success_first_actions_states.npz"),
                "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                "durable_records": str(durable_records),
                "durable_frames": str(durable_frames),
            },
        }
    return {
        "status": "joint_limit_recovery_failed_without_legal_fallback",
        "initial_joint_limits": initial_limits,
        "final_joint_limits": current_limits,
        "physical_actions": len(records),
        "bilateral_grasp_retained": current_grasp.get("grasp_confirmed") is True,
        "reason": reason,
        "safety_certified": False,
    }


def _select_active_target_predicate(
    native_status: dict[str, Any], requested: list[str] | None
) -> tuple[list[str] | None, str, list[list[str]]]:
    """Select an explicit goal or the first pending native object goal."""
    coverage = inspect_native_goal_coverage(native_status)
    pending = pending_object_goals(native_status)
    if requested is not None:
        selected = [str(token) for token in requested]
        match = next((row for row in coverage["goals"] if row["predicate"] == selected), None)
        if match is None or not match["operation_supported"]:
            raise ValueError("requested_native_goal_missing_or_operator_not_supported")
        return selected, "explicit_cli", pending
    if pending:
        return pending[0], "first_pending_native_object_goal_bddl_order", pending
    fixture = next((row["predicate"] for row in coverage["goals"]
                    if not row["value"] and row["operator"] in {"open", "close", "turnon", "turnoff"}), None)
    if fixture is not None:
        return fixture, "first_pending_supported_fixture_goal_bddl_order", pending
    if coverage["pending_unsupported_goals"]:
        raise ValueError("pending_native_goal_operator_not_supported:" + json.dumps(
            [row["predicate"] for row in coverage["pending_unsupported_goals"]],
            ensure_ascii=False,
        ))
    return None, "no_pending_object_goal", pending


def _verified_grasp_resume_descriptor(
    report: dict[str, Any],
    *,
    benchmark: str,
    level: int,
    task_id: int,
    seed: int,
    init_state: int,
    model_sha: str,
    bddl_sha: str,
    target_id: str,
    predicate: list[str],
) -> dict[str, Any]:
    scope = report.get("scope", {})
    success = report.get("success") or {}
    snapshot = success.get("snapshot") or {}
    expected_scope = {
        "benchmark": benchmark,
        "level": level,
        "task_id": task_id,
        "environment_seed": seed,
        "init_state": init_state,
        "model_geometry_sha256": model_sha,
        "bddl_sha256": bddl_sha,
    }
    if report.get("status") != "task_success_candidate_created":
        raise ValueError("resume_report_is_not_a_successful_grasp_skill_report")
    if success.get("task_success_candidate") is not True or success.get("safety_certified") is not False:
        raise ValueError("resume_report_candidate_provenance_is_invalid")
    if success.get("target_object_id") != target_id:
        raise ValueError("resume_report_target_object_differs_from_active_BDDL")
    if report.get("active_target", {}).get("native_goal_predicate") != predicate:
        raise ValueError("resume_report_native_predicate_differs_from_active_BDDL")
    for key, value in expected_scope.items():
        if scope.get(key) != value:
            raise ValueError(f"resume_report_{key}_mismatch")
    if report.get("route_constraints", {}).get("official_safety_demonstrations_used") is not False:
        raise ValueError("resume_report_official_demo_provenance_is_not_allowed")
    snapshot_path = Path(str(snapshot.get("path") or ""))
    snapshot_state = snapshot.get("snapshot_state")
    if not snapshot_path.is_file() or not isinstance(snapshot_state, dict):
        raise ValueError("resume_report_success_snapshot_is_missing_or_incomplete")
    return {"path": snapshot_path, "snapshot_state": snapshot_state, "success": success}


def _verified_placement_resume_descriptor(
    report: dict[str, Any],
    *,
    benchmark: str,
    level: int,
    task_id: int,
    seed: int,
    init_state: int,
    model_sha: str,
    bddl_sha: str,
    target_id: str,
    predicate: list[str],
) -> dict[str, Any]:
    """Validate a physically placed, still-held object prefix for release recovery."""
    scope = report.get("scope", {})
    task = report.get("task", {})
    segment = next((
        row for row in reversed(report.get("segments", []))
        if row.get("failure_phase") == "precise_placement"
    ), None)
    status = report.get("status")
    if segment is None and status == "native_goal_transport_failed":
        # A transport timeout can happen after real contact has already made
        # the native On/In predicate true. Keep that exact physical branch as a
        # release-resume source when the report identifies the matching branch
        # snapshot and the final native query confirms the predicate.
        for parent in reversed(report.get("segments", [])):
            if parent.get("type") != "expert_native_goal_stage_control":
                continue
            attempts = parent.get("placement_search", {}).get("candidate_attempts", [])
            for attempt in reversed(attempts):
                candidate = attempt.get("candidate", {})
                if (
                    candidate.get("predicate") == predicate
                    and attempt.get("target_predicate_value") is True
                    and attempt.get("terminal_snapshot")
                ):
                    segment = {
                        "failure_phase": "transport_after_native_goal_contact",
                        "native_goal_predicate": predicate,
                        "terminal_snapshot": attempt["terminal_snapshot"],
                        "candidate_id": candidate.get("candidate_id"),
                        "candidate_status": attempt.get("status"),
                    }
                    break
            if segment is not None:
                break
    expected_scope = {
        "benchmark": benchmark,
        "level": level,
        "task_id": task_id,
        "seed": seed,
        "init_state": init_state,
        "model_geometry_sha256": model_sha,
    }
    if status not in {"native_goal_precise_placement_failed", "native_goal_transport_failed"} or segment is None:
        raise ValueError("placement_resume_report_is_not_an_archived_precise_placement_prefix")
    for key, value in expected_scope.items():
        if scope.get(key) != value:
            raise ValueError(f"placement_resume_{key}_mismatch")
    if task.get("BDDL_sha256") != bddl_sha:
        raise ValueError("placement_resume_BDDL_fingerprint_mismatch")
    if task.get("target_object_from_active_BDDL") != target_id:
        raise ValueError("placement_resume_target_object_differs_from_active_BDDL")
    if task.get("native_goal_predicate") != predicate or segment.get("native_goal_predicate") != predicate:
        raise ValueError("placement_resume_native_predicate_differs_from_active_BDDL")
    policy = report.get("policy", {})
    if (
        policy.get("official_safety_demonstration_data_used") is not False
        or policy.get("object_or_robot_state_written_directly") is not False
        or policy.get("all_actions_are_real_MuJoCo_steps") is not True
        or report.get("safety_certified") is True
    ):
        raise ValueError("placement_resume_provenance_is_not_allowed")
    native = segment.get("final_native_task_status") or report.get("final_native_task_status")
    if not isinstance(native, dict) or native.get("task_success") is True:
        raise ValueError("placement_resume_requires_an_incomplete_native_task")
    values = {
        tuple(row.get("predicate", [])): row.get("value")
        for row in native.get("native_goal_values", [])
        if isinstance(row, dict)
    }
    if values.get(tuple(predicate)) is not True:
        raise ValueError("placement_resume_requires_current_native_goal_to_be_true")
    snapshot_path = Path(str(segment.get("terminal_snapshot") or ""))
    if not snapshot_path.is_file():
        raise ValueError("placement_resume_terminal_snapshot_is_missing")
    from research.gs_opd.repair.task_success_first_grasp_search import _PythonObjectState

    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    payload = snapshot_path.read_bytes()
    envelope = pickle.loads(payload)
    scene = envelope.get("scene_metadata", {})
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != model_sha
        or scene.get("benchmark") != benchmark
        or scene.get("level") != level
        or scene.get("task_id") != task_id
        or scene.get("environment_seed") != seed
        or scene.get("init_state_index") != init_state
        or scene.get("bddl_sha256") != bddl_sha
    ):
        raise ValueError("placement_resume_snapshot_scene_or_model_identity_mismatch")
    mujoco_state = envelope.get("snapshot", {}).get("mujoco_state")
    if mujoco_state is None:
        raise ValueError("placement_resume_snapshot_has_no_MuJoCo_integration_state")
    return {
        "path": snapshot_path,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot_state": {"mujoco_integration_state": np.asarray(mujoco_state, dtype=np.float64)},
        "native_task_status": native,
        "segment": segment,
    }


def _verified_runtime_snapshot_descriptor(
    snapshot_path: Path,
    *,
    benchmark: str,
    level: int,
    task_id: int,
    seed: int,
    init_state: int,
    model_sha: str,
    bddl_sha: str,
    instruction: str,
) -> dict[str, Any]:
    """Validate a direct continuation Snapshot against its compiled scene identity."""
    from research.gs_opd.repair.task_success_first_grasp_search import _PythonObjectState

    snapshot_path = Path(snapshot_path)
    if not snapshot_path.is_file():
        raise ValueError("runtime_snapshot_resume_source_is_missing")
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    payload = snapshot_path.read_bytes()
    envelope = pickle.loads(payload)
    scene = envelope.get("scene_metadata", {})
    expected = {
        "benchmark": benchmark,
        "level": level,
        "task_id": task_id,
        "environment_seed": seed,
        "init_state_index": init_state,
        "bddl_sha256": bddl_sha,
    }
    if envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
        raise ValueError("runtime_snapshot_resume_schema_mismatch")
    if envelope.get("model_geometry_sha256") != model_sha:
        raise ValueError("runtime_snapshot_resume_model_fingerprint_mismatch")
    for key, value in expected.items():
        if scene.get(key) != value:
            raise ValueError(f"runtime_snapshot_resume_{key}_mismatch")
    if scene.get("instruction") != instruction:
        raise ValueError("runtime_snapshot_resume_instruction_mismatch")
    mujoco_state = envelope.get("snapshot", {}).get("mujoco_state")
    state = np.asarray(mujoco_state, dtype=np.float64)
    if state.ndim != 1 or not len(state) or not np.isfinite(state).all():
        raise ValueError("runtime_snapshot_resume_integration_state_is_invalid")
    return {
        "path": snapshot_path,
        "payload": payload,
        "payload_sha256": _sha(payload),
        "snapshot_state": {"mujoco_integration_state": state},
        "scene_metadata": scene,
        "model_geometry_sha256": model_sha,
        "bddl_sha256": bddl_sha,
    }


def _register_verified_operation_skill_from_report(
    library_path: Path | None,
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any] | None:
    """Add an executed grasp/placement cycle after native placement and release."""
    if library_path is None or not library_path.is_file():
        return None
    task = report.get("task", {})
    target_id = str(task.get("target_object_from_active_BDDL") or "")
    predicate = task.get("native_goal_predicate")
    final_native = report.get("final_native_task_status")
    if not target_id or not isinstance(predicate, list) or not isinstance(final_native, dict):
        return None
    native_value = next((
        row.get("value") for row in final_native.get("native_goal_values", [])
        if row.get("predicate") == predicate
    ), None)
    if native_value is not True:
        return None
    placement = next((
        segment for segment in reversed(report.get("segments", []))
        if segment.get("type") in {
            "expert_native_goal_stage_control",
            "resumed_expert_native_goal_stage_control",
            "resumed_expert_release_withdrawal",
        }
    ), None)
    if not isinstance(placement, dict) or int(placement.get("recorded_actions", 0) or 0) <= 0:
        return None
    released_and_stable = bool(
        placement.get("partial_goal_settled") is True
        or (final_native.get("task_success") is True
            and placement.get("task_phase_controller_phase") in {"settle", "verify_task"})
    )
    if not released_and_stable:
        return None

    from research.gs_opd.repair.task_skill_experience import (
        _object_family,
        append_verified_search_operation_skill,
        load_experience_library,
        matching_operation_skill,
    )

    library = load_experience_library(library_path)
    source_operation = matching_operation_skill(library, target_id)
    phases = list((source_operation or {}).get("phase_sequence", []))
    if not phases:
        phases = [
            "approach", "stable_grasp", "lift", "transport", "precise_placement",
            "release", "withdrawal", "settle",
        ]

    documents: list[tuple[Path, dict[str, Any]]] = [(report_path, report)]
    seen = {str(report_path.resolve())}
    cursor = 0
    while cursor < len(documents):
        current_path, current = documents[cursor]
        cursor += 1
        pending_paths: list[Path] = []
        for key in ("resumed_from_grasp_skill", "resumed_from_native_goal_contact_snapshot"):
            source = current.get(key)
            if isinstance(source, dict) and source.get("source_report"):
                pending_paths.append(Path(source["source_report"]))
        for segment in current.get("segments", []):
            if segment.get("report"):
                pending_paths.append(Path(segment["report"]))
        while pending_paths:
            candidate = pending_paths.pop()
            resolved = str(candidate.resolve())
            if resolved in seen or not candidate.is_file():
                continue
            seen.add(resolved)
            try:
                child = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            documents.append((candidate, child))
    stable_grasp = False
    physical_actions = 0
    source_reports = []
    for source_path, document in documents:
        source_reports.append(str(source_path))
        if document.get("policy", {}).get("official_safety_demonstration_data_used") is True:
            raise ValueError("official_safety_demonstration_provenance_found_in_operation_skill_sources")
        if document.get("route_constraints", {}).get("official_safety_demonstrations_used") is True:
            raise ValueError("official_safety_demonstration_provenance_found_in_operation_skill_sources")
        for container in [document.get("success", {})] + [
            segment.get("success", {}) for segment in document.get("segments", [])
        ]:
            stable = container.get("stable_bilateral_grasp", {}) if isinstance(container, dict) else {}
            stable_grasp |= bool(stable.get("stable") is True or stable.get("stable_grasp_observed") is True)
        for segment in document.get("segments", []):
            physical_actions += int(segment.get("actions_executed", 0) or 0)
            physical_actions += int(segment.get("recorded_actions", 0) or 0)
            if not segment.get("report"):
                for branch in segment.get("physical_branches", []):
                    physical_actions += int(
                        branch.get("attempted_control_actions_archived", branch.get("branch_control_actions", 0))
                        or 0
                    )
        for branch in document.get("physical_branches", []):
            physical_actions += int(
                branch.get("attempted_control_actions_archived", branch.get("branch_control_actions", 0))
                or 0
            )
            grasp = branch.get("grasp_before_lift", {})
            stable_grasp |= bool(
                branch.get("stable_bilateral_grasp_observed") is True
                or grasp.get("stable") is True or grasp.get("stable_grasp_observed") is True
            )
    if not stable_grasp or physical_actions <= 0:
        return None

    scope = report.get("scope", {})
    scene = f"{scope.get('benchmark')}/L{scope.get('level')}T{scope.get('task_id')}"
    signature = hashlib.sha256(json.dumps(
        [scene, int(scope.get("seed", 0)), int(scope.get("init_state", 0)), target_id, predicate],
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()[:10]
    template = {
        "skill_id": f"object_transfer_cycle:{_object_family(target_id)}:{signature}",
        "object_family": _object_family(target_id),
        "operation": "pick_transport_place_release",
        "phase_sequence": phases,
        "source_task": task.get("instruction"),
        "source_scene": scene,
        "source_object_id": target_id,
        "native_goal_predicate": predicate,
        "source_native_task_success": final_native.get("task_success") is True,
        "skill_success": True,
        "native_object_goal_satisfied": True,
        "released_and_stable": True,
        "stable_bilateral_grasp_verified": True,
        "physical_actions": physical_actions,
        "source_execution_reports": list(dict.fromkeys(source_reports)),
        "historical_actions_replayed": False,
        "source_actions_used_as_direct_cross_scene_labels": False,
        "official_safety_demonstrations_used": False,
        "safety_certified": False,
    }
    added = append_verified_search_operation_skill(library_path, template)
    return {
        "status": "skill_added" if added else "duplicate_skill_already_present",
        "skill_id": template["skill_id"],
        "physical_actions_in_source_prefix": physical_actions,
        "source_execution_reports": template["source_execution_reports"],
        "cross_scene_physical_validation_count": 1 if added else None,
        "source_native_task_success": template["source_native_task_success"],
        "safety_certified": False,
        "official_safety_demonstrations_used": False,
    }


def _controller_config(target_id: str, predicate: list[str], grasp_candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "objects": [{
            "object_id": target_id,
            "predicate": predicate,
            "goal_source": "active_native_BDDL_goal_and_native_goal_region_query",
        }],
        "grasp_pose_candidates": [{"candidate_id": str(grasp_candidate.get("candidate_id", "live_verified_grasp"))}],
        "phase_control_step_limits": {
            "pregrasp": 64, "approach": 128, "close_gripper": 32,
            "stable_grasp": 32, "lift": 32, "transport": 160,
            "precise_placement": 96, "release": 32, "withdrawal": 64,
            "settle": 32,
        },
        "stable_grasp_control_steps": 3,
        "lift_validation_height_m": 0.03,
        "settle_position_span_m": 0.002,
        "settle_linear_speed_m_s": 0.08,
        "settle_angular_speed_rad_s": 0.8,
    }


def _phase_evidence(
    bridge,
    controller: TaskPhaseController,
    target_id: str,
    target_body_id: int,
    predicate: list[str],
    *,
    target_binding: dict[str, Any] | None = None,
    eef_goal_reached: bool = False,
) -> tuple[PhaseEvidence, dict[str, Any]]:
    observed = bridge.request({"op": "observe", "include_snapshot_state": True})
    snapshot = observed["snapshot_state"]
    native = bridge.request({"op": "native_task_status"})
    values, task_success = _task_values(native)
    obj = _object_state(snapshot, target_body_id)
    robot_rows = snapshot.get("robot_states", [])
    if len(robot_rows) != 1:
        raise ValueError("task_success_first_executor_requires_one_Panda_state")
    robot = robot_rows[0]
    binding = dict(target_binding) if target_binding is not None else {
        "object_id": target_id,
        "body_id": target_body_id,
        "body_name": obj.get("body_name"),
        "geom_ids": obj.get("geom_ids", []),
        "geom_names": obj.get("geom_names", []),
    }
    if (
        binding.get("object_id") != target_id
        or int(binding.get("body_id", -1)) != int(target_body_id)
        or not isinstance(binding.get("geom_ids"), list)
        or not isinstance(binding.get("geom_names"), list)
        or not binding["geom_ids"]
        or len(binding["geom_ids"]) != len(binding["geom_names"])
    ):
        raise ValueError("phase_evidence_requires_complete_model_verified_target_binding")
    grasp = bridge.request({
        "op": "grasp_state", "carried_binding": binding,
        "expected_instruction": observed.get("instruction"),
    })
    grasped = grasp.get("grasp_confirmed") is True
    finger_contacts = grasp.get("finger_contacts", {})
    finger_contacts_clear = _finger_contacts_clear(finger_contacts)
    all_satisfied = bool(values) and all(values.values())
    state = PhaseEvidence(
        timestep=int(observed.get("timestep", -1)),
        eef_goal_reached=eef_goal_reached,
        held_object=target_id if grasped else None,
        grasp_confirmed=grasped,
        object_position_world_m=tuple(float(x) for x in obj["root_pos"]),
        object_quaternion_wxyz=tuple(float(x) for x in obj["root_quat"]),
        object_qvel=tuple(float(x) for x in obj.get("joint_qvel", [0.0] * 6)),
        eef_position_world_m=tuple(float(x) for x in robot["eef_pos"]),
        task_goal_satisfied=bool(values.get(tuple(predicate), False)),
        all_task_goals_satisfied=all_satisfied,
        satisfied_object_ids=tuple(
            target_id for key, value in values.items() if value and len(key) == 3 and key[1] == target_id
        ),
        fixture_predicate_satisfied=None,
        trusted_fixture_activation_available=False,
        task_success=task_success,
        finger_contacts_clear=finger_contacts_clear,
        eef_position_error_m=0.0 if eef_goal_reached else None,
        eef_orientation_error_rad=0.0 if eef_goal_reached else None,
    )
    return state, {"observation": observed, "native_task_status": native, "grasp_state": grasp}


def _release_retreat_target(
    eef_position_world_m: Any,
    object_position_world_m: Any,
    *,
    retreat_distance_m: float = 0.003,
) -> np.ndarray | None:
    """Return a short horizontal escape waypoint away from a released object."""
    eef = np.asarray(eef_position_world_m, dtype=np.float64)
    obj = np.asarray(object_position_world_m, dtype=np.float64)
    if (
        eef.shape != (3,) or obj.shape != (3,)
        or not np.isfinite(eef).all() or not np.isfinite(obj).all()
        or not np.isfinite(retreat_distance_m) or retreat_distance_m <= 0.0
    ):
        return None
    away = eef - obj
    away[2] = 0.0
    norm = float(np.linalg.norm(away))
    if norm <= 1e-6:
        return None
    return eef + away * (float(retreat_distance_m) / norm)


def _placement_footprint_radius(
    context: dict[str, Any], region_rotation: np.ndarray,
    root_rotation_override: np.ndarray | None = None,
) -> float:
    """Conservative horizontal radius of a live object's compiled collision geometry."""
    root_rotation = np.asarray(
        root_rotation_override if root_rotation_override is not None
        else context.get("target_root_pose_world", {}).get("rotation"), dtype=np.float64,
    )
    if root_rotation.shape != (3, 3) or not np.isfinite(root_rotation).all():
        raise ValueError("placement_geometry_context_has_no_root_rotation")
    region_rotation = np.asarray(region_rotation, dtype=np.float64)
    if region_rotation.shape != (3, 3) or not np.isfinite(region_rotation).all():
        raise ValueError("placement_region_rotation_is_invalid")
    region_from_root = region_rotation.T @ root_rotation
    radius = 0.0
    for geom in context.get("target_collision_geometries", []):
        center = np.asarray(geom.get("center_root_m"), dtype=np.float64)
        geom_rotation = np.asarray(geom.get("rotation_root"), dtype=np.float64)
        size = np.asarray(geom.get("size_m"), dtype=np.float64)
        if center.shape != (3,) or geom_rotation.shape != (3, 3) or size.shape != (3,):
            continue
        if not np.isfinite(center).all() or not np.isfinite(geom_rotation).all() or not np.isfinite(size).all():
            continue
        local_center = region_from_root @ center
        local_axes = region_from_root @ geom_rotation
        xy_extent = np.abs(local_axes[:2, :]) @ np.maximum(size, 0.0)
        radius = max(radius, float(np.linalg.norm(local_center[:2]) + np.linalg.norm(xy_extent)))
    if radius <= 1e-6:
        raise ValueError("placement_collision_geometry_has_no_usable_footprint")
    return radius


def _placement_lowest_geometry_offset(
    context: dict[str, Any], region_rotation: np.ndarray,
    root_rotation_override: np.ndarray | None = None,
) -> float:
    """Return the lowest compiled collision point relative to an object's root pose."""
    root_rotation = np.asarray(
        root_rotation_override if root_rotation_override is not None
        else context.get("target_root_pose_world", {}).get("rotation"), dtype=np.float64,
    )
    region_rotation = np.asarray(region_rotation, dtype=np.float64)
    if root_rotation.shape != (3, 3) or region_rotation.shape != (3, 3):
        raise ValueError("placement_geometry_context_has_no_root_rotation")
    region_from_root = region_rotation.T @ root_rotation
    lowest = np.inf
    for geom in context.get("target_collision_geometries", []):
        center = np.asarray(geom.get("center_root_m"), dtype=np.float64)
        geom_rotation = np.asarray(geom.get("rotation_root"), dtype=np.float64)
        size = np.asarray(geom.get("size_m"), dtype=np.float64)
        if center.shape != (3,) or geom_rotation.shape != (3, 3) or size.shape != (3,):
            continue
        if not np.isfinite(center).all() or not np.isfinite(geom_rotation).all() or not np.isfinite(size).all():
            continue
        center_height = float((region_from_root @ center)[2])
        vertical_extent = float(np.abs((region_from_root @ geom_rotation)[2, :]) @ np.maximum(size, 0.0))
        lowest = min(lowest, center_height - vertical_extent)
    if not np.isfinite(lowest):
        raise ValueError("placement_collision_geometry_has_no_support_extent")
    return float(lowest)


def _placement_collision_bounds_in_region(
    context: dict[str, Any],
    region_rotation: np.ndarray,
    root_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the compiled collision envelope in native-region coordinates."""
    region_rotation = np.asarray(region_rotation, dtype=np.float64)
    root_rotation = np.asarray(root_rotation, dtype=np.float64)
    if (
        region_rotation.shape != (3, 3) or root_rotation.shape != (3, 3)
        or not np.isfinite(region_rotation).all() or not np.isfinite(root_rotation).all()
    ):
        raise ValueError("placement_collision_bounds_require_finite_region_and_root_rotations")
    region_from_root = region_rotation.T @ root_rotation
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    for geom in context.get("target_collision_geometries", []):
        center = np.asarray(geom.get("center_root_m"), dtype=np.float64)
        geom_rotation = np.asarray(geom.get("rotation_root"), dtype=np.float64)
        size = np.asarray(geom.get("size_m"), dtype=np.float64)
        if (
            center.shape != (3,) or geom_rotation.shape != (3, 3) or size.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(geom_rotation).all()
            or not np.isfinite(size).all() or np.any(size < 0.0)
        ):
            continue
        center_region = region_from_root @ center
        axes_region = region_from_root @ geom_rotation
        extent = np.abs(axes_region) @ size
        lower = np.minimum(lower, center_region - extent)
        upper = np.maximum(upper, center_region + extent)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("placement_collision_bounds_have_no_valid_compiled_geometry")
    return lower, upper


def _native_in_region_orientation_candidates(
    *,
    target_context: dict[str, Any],
    region_rotation: np.ndarray,
    target_rotation: np.ndarray,
    region_size: np.ndarray,
    maximum_candidates: int,
    preferred_yaw_delta_degrees: float | None = None,
) -> list[dict[str, Any]]:
    """Search container-plane yaw poses using the target's compiled geometry."""
    from scipy.spatial.transform import Rotation

    axis = np.asarray(region_rotation, dtype=np.float64)[:, 2]
    axis_norm = float(np.linalg.norm(axis))
    target_rotation = np.asarray(target_rotation, dtype=np.float64)
    region_size = np.asarray(region_size, dtype=np.float64)
    if (
        axis.shape != (3,) or target_rotation.shape != (3, 3)
        or region_size.shape != (3,) or axis_norm <= 1e-8
        or not np.isfinite(axis).all() or not np.isfinite(target_rotation).all()
        or not np.isfinite(region_size).all() or np.any(region_size[:2] <= 0.0)
    ):
        raise ValueError("native_In_orientation_search_requires_valid_region_and_target_geometry")
    axis /= axis_norm
    rows = []
    for angle_degrees in range(-180, 180, 5):
        angle_radians = np.deg2rad(float(angle_degrees))
        rotation = Rotation.from_rotvec(axis * angle_radians).as_matrix() @ target_rotation
        lower, upper = _placement_collision_bounds_in_region(
            target_context, region_rotation, rotation,
        )
        half_extent = 0.5 * (upper[:2] - lower[:2])
        center_offset = 0.5 * (upper[:2] + lower[:2])
        overflow = np.maximum(half_extent - region_size[:2], 0.0) / region_size[:2]
        root_center_overflow = np.maximum(np.abs(center_offset) - region_size[:2], 0.0) / region_size[:2]
        utilization = half_extent / region_size[:2]
        fit_cost = float(
            np.dot(utilization, utilization)
            + 4.0 * np.dot(overflow, overflow)
            + np.dot(root_center_overflow, root_center_overflow)
        )
        rows.append({
            "rotation_world": rotation,
            "yaw_delta_degrees": float(angle_degrees),
            "collision_bounds_region_m": [lower.tolist(), upper.tolist()],
            "footprint_half_extents_region_m": half_extent.tolist(),
            "geometry_center_offset_region_m": center_offset.tolist(),
            "fit_cost": fit_cost,
        })
    rows.sort(key=lambda row: (row["fit_cost"], abs(row["yaw_delta_degrees"]), row["yaw_delta_degrees"]))
    # Native In predicates do not constrain object yaw. Keep the live physical
    # orientation available as the first real MuJoCo branch even when a
    # different yaw has a slightly better static envelope-fit score; otherwise
    # the planner can demand a large, unnecessary wrist/object rotation before
    # entry. Geometry still ranks the alternate branches, and MuJoCo plus BDDL
    # remain the only acceptance test.
    selected = []
    if preferred_yaw_delta_degrees is not None:
        if not np.isfinite(preferred_yaw_delta_degrees) or not -180.0 <= preferred_yaw_delta_degrees <= 180.0:
            raise ValueError("native_In_preferred_yaw_delta_must_be_finite_degrees_in_minus180_to180")
        selected.append(min(
            rows,
            key=lambda row: abs(
                (float(row["yaw_delta_degrees"]) - preferred_yaw_delta_degrees + 180.0)
                % 360.0 - 180.0
            ),
        ))
    for preferred_delta in (
        0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0,
        75.0, -75.0, 90.0, -90.0, 120.0, -120.0, 180.0,
    ):
        if len(selected) >= maximum_candidates:
            break
        preferred = next(
            (row for row in rows
             if abs(float(row["yaw_delta_degrees"]) - preferred_delta) < 1e-9),
            None,
        )
        if preferred is not None and not any(preferred is prior for prior in selected):
            selected.append(preferred)
            if len(selected) >= maximum_candidates:
                break
    # Preserve the collision-envelope best-fit orientation when the bounded
    # wrist-search portfolio still has room for it.
    best_fit = rows[0]
    best_fit_angle = float(best_fit["yaw_delta_degrees"])
    if len(selected) < maximum_candidates and not any(
        abs((best_fit_angle - float(prior["yaw_delta_degrees"]) + 180.0) % 360.0 - 180.0) < 1e-6
        for prior in selected
    ):
        selected.append(best_fit)
    for row in rows:
        angle = float(row["yaw_delta_degrees"])
        if any(
            abs((angle - float(prior["yaw_delta_degrees"]) + 180.0) % 360.0 - 180.0) < 1e-6
            for prior in selected
        ):
            continue
        if any(
            abs((angle - float(prior["yaw_delta_degrees"]) + 180.0) % 360.0 - 180.0) < 10.0
            for prior in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= maximum_candidates:
            break
    if not selected:
        raise ValueError("native_In_orientation_search_produced_no_yaw_candidates")
    return selected


def _placement_shape_signature(context: dict[str, Any]) -> tuple[tuple[float, ...], ...]:
    """Orientation-independent signature for recognizing matching compiled objects."""
    rows = []
    for geom in context.get("target_collision_geometries", []):
        center = np.asarray(geom.get("center_root_m"), dtype=np.float64)
        size = np.asarray(geom.get("size_m"), dtype=np.float64)
        if center.shape == (3,) and size.shape == (3,) and np.isfinite(center).all() and np.isfinite(size).all():
            rows.append(tuple(round(float(x), 4) for x in np.r_[center, size]))
    return tuple(sorted(rows))


def _placement_orientation_compatible(
    target_context: dict[str, Any], occupied_context: dict[str, Any],
) -> bool:
    """Match placement orientation from similar compiled collision envelopes.

    Exact geom-by-geom signatures are too strict for two instances of the same
    asset when their generated geom names or decomposition differ. Compare the
    union collision-envelope dimensions instead; this is a placement heuristic,
    never a safety or task-success check.
    """
    def envelope(context: dict[str, Any]) -> np.ndarray | None:
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for geom in context.get("target_collision_geometries", []):
            center = np.asarray(geom.get("center_root_m"), dtype=np.float64)
            rotation = np.asarray(geom.get("rotation_root"), dtype=np.float64)
            size = np.asarray(geom.get("size_m"), dtype=np.float64)
            if (
                center.shape != (3,) or rotation.shape != (3, 3) or size.shape != (3,)
                or not np.isfinite(center).all() or not np.isfinite(rotation).all()
                or not np.isfinite(size).all()
            ):
                continue
            extent = np.abs(rotation) @ np.maximum(size, 0.0)
            lows.append(center - extent)
            highs.append(center + extent)
        if not lows:
            return None
        return np.max(np.stack(highs), axis=0) - np.min(np.stack(lows), axis=0)

    target_envelope = envelope(target_context)
    occupied_envelope = envelope(occupied_context)
    if target_envelope is None or occupied_envelope is None:
        return False
    scale = np.maximum(np.maximum(target_envelope, occupied_envelope), 1e-3)
    return bool(np.max(np.abs(target_envelope - occupied_envelope) / scale) <= 0.35)


def _native_region_entry_offset_m(predicate: list[str]) -> float:
    """Keep an object-relative lateral staging point for native On and In goals."""
    if not isinstance(predicate, list) or len(predicate) != 3 or predicate[0].lower() not in {"on", "in"}:
        raise ValueError("placement_entry_offset_requires_native_On_or_In_predicate")
    return 0.06


def _native_region_local_z_for_world_height(
    region: dict[str, Any], local_xy: np.ndarray, world_z: float,
) -> float:
    """Solve the region-local z coordinate that places a point at world height z."""
    region_position = np.asarray(region.get("site_position_world_m"), dtype=np.float64)
    region_rotation = np.asarray(region.get("site_rotation_world"), dtype=np.float64)
    local_xy = np.asarray(local_xy, dtype=np.float64)
    if (
        region_position.shape != (3,) or region_rotation.shape != (3, 3)
        or local_xy.shape != (2,) or not np.isfinite(region_position).all()
        or not np.isfinite(region_rotation).all() or not np.isfinite(local_xy).all()
        or not np.isfinite(world_z)
    ):
        raise ValueError("native_region_world_height_requires_valid_region_and_local_xy")
    vertical_axis_projection = float(region_rotation[2, 2])
    if abs(vertical_axis_projection) < 1e-5:
        raise ValueError("native_region_progress_requires_region_z_axis_with_world_vertical_component")
    return float(
        (float(world_z) - region_position[2]
         - region_rotation[2, 0] * float(local_xy[0])
         - region_rotation[2, 1] * float(local_xy[1]))
        / vertical_axis_projection
    )


def _placement_ik_seed_portfolio(robot_state: dict[str, Any]) -> list[list[float]]:
    """Create deterministic redundant-arm IK seeds around the live arm posture."""
    qpos = np.asarray(robot_state.get("joint_qpos"), dtype=np.float64)
    if qpos.ndim != 1 or len(qpos) < 7 or not np.isfinite(qpos[:7]).all():
        raise ValueError("placement_IK_seed_portfolio_requires_live_seven_joint_arm_state")
    arm = qpos[:7].copy()
    offsets = (
        (2, 0.45), (2, -0.45),
        (3, 0.45), (3, -0.45),
        (4, 0.45), (4, -0.45),
    )
    seeds = []
    for joint_index, delta in offsets:
        seed = arm.copy()
        seed[joint_index] += delta
        seeds.append(seed.tolist())
    return seeds


def _native_placement_planner_target(
    predicate: list[str], goal_eef: np.ndarray, high_eef: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Plan to an elevated, region-relative entry; OSC performs final placement."""
    if not isinstance(predicate, list) or len(predicate) != 3 or predicate[0].lower() not in {"on", "in"}:
        raise ValueError("placement_RRT_target_requires_native_On_or_In_predicate")
    role = "open_support_high_entry_pose" if predicate[0].lower() == "on" else "container_entry_high_pose"
    return np.asarray(high_eef, dtype=np.float64), role


def _native_goal_insertion_rrt_path_arrays(
    planner_result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a live-state full-geometry RRT path for the final native In insertion."""
    if (
        not isinstance(planner_result, dict)
        or planner_result.get("status") != "offline_feasible"
        or planner_result.get("live_state_unchanged") is not True
    ):
        raise RuntimeError(
            "native_goal_insertion_RRT_not_feasible_before_OSC:"
            f"{planner_result.get('failure_reason') or planner_result.get('status') if isinstance(planner_result, dict) else 'malformed_planner_result'}"
        )
    eef_path = np.asarray(planner_result.get("eef_reference_poses_world"), dtype=np.float64)
    qpos_path = np.asarray(planner_result.get("qpos_path"), dtype=np.float64)
    if (
        eef_path.ndim != 2 or eef_path.shape[1:] != (7,)
        or qpos_path.ndim != 2 or qpos_path.shape[1:] != (7,)
        or len(eef_path) != len(qpos_path) or len(eef_path) < 1
        or not np.isfinite(eef_path).all() or not np.isfinite(qpos_path).all()
    ):
        raise RuntimeError("native_goal_insertion_RRT_joint_and_EEF_path_alignment_invalid")
    return eef_path, qpos_path


def _native_region_aperture_tilted_goal_pose(
    goal_object_world: np.ndarray,
    *,
    region: dict[str, Any],
    target_context: dict[str, Any],
    object_to_eef: np.ndarray,
    tilt_degrees: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a carried object and its attached hand through a low aperture."""
    from scipy.spatial.transform import Rotation

    goal = np.asarray(goal_object_world, dtype=np.float64).copy()
    region_position = np.asarray(region.get("site_position_world_m"), dtype=np.float64)
    region_rotation = np.asarray(region.get("site_rotation_world"), dtype=np.float64)
    region_size = np.asarray(region.get("site_size_m"), dtype=np.float64)
    object_to_eef = np.asarray(object_to_eef, dtype=np.float64)
    if (
        goal.shape != (4, 4) or region_position.shape != (3,)
        or region_rotation.shape != (3, 3) or region_size.shape != (3,)
        or object_to_eef.shape != (4, 4)
        or not all(np.isfinite(row).all() for row in (
            goal, region_position, region_rotation, region_size, object_to_eef,
        ))
        or not np.isfinite(tilt_degrees) or not 0.0 <= float(tilt_degrees) <= 90.0
    ):
        raise ValueError("native_region_aperture_tilt_requires_valid_goal_geometry_and_angle")
    current_rotation = goal[:3, :3].copy()
    eef_offset_world = current_rotation @ object_to_eef[:3, 3]
    tilt_axis_world = np.cross(np.asarray([0.0, 0.0, 1.0]), eef_offset_world)
    tilt_axis_world[2] = 0.0
    axis_norm = float(np.linalg.norm(tilt_axis_world))
    if axis_norm <= 1e-8:
        tilt_axis_world = region_rotation[:, 0].copy()
        tilt_axis_world[2] = 0.0
        axis_norm = float(np.linalg.norm(tilt_axis_world))
    if axis_norm <= 1e-8:
        raise ValueError("native_region_aperture_tilt_axis_is_undefined")
    tilt_axis_world /= axis_norm
    world_tilt = Rotation.from_rotvec(
        tilt_axis_world * np.deg2rad(float(tilt_degrees))
    ).as_matrix()
    goal[:3, :3] = world_tilt @ current_rotation
    lower, upper = _placement_collision_bounds_in_region(
        target_context, region_rotation, goal[:3, :3],
    )
    original_lower, original_upper = _placement_collision_bounds_in_region(
        target_context, region_rotation, current_rotation,
    )
    original_local = region_rotation.T @ (np.asarray(goal_object_world)[:3, 3] - region_position)
    centered_xy = -0.5 * (lower[:2] + upper[:2])
    region_z_world_up = float(region_rotation[2, 2])
    if abs(region_z_world_up) > 0.5:
        support_local_z = -float(np.sign(region_z_world_up)) * float(region_size[2])
        root_local_z = support_local_z - float(lower[2])
    else:
        support_local_z = float(original_local[2] + original_lower[2])
        root_local_z = support_local_z - float(lower[2])
    goal[:3, 3] = region_position + region_rotation @ np.asarray([
        centered_xy[0], centered_xy[1], root_local_z,
    ])
    eef_offset_after = goal[:3, :3] @ object_to_eef[:3, 3]
    return goal, {
        "status": "geometry_retargeted_aperture_posture",
        "tilt_degrees": float(tilt_degrees),
        "tilt_axis_world": tilt_axis_world.tolist(),
        "native_support_local_z_m": support_local_z,
        "goal_region_local_m": (
            region_rotation.T @ (goal[:3, 3] - region_position)
        ).tolist(),
        "target_collision_bounds_before_m": [original_lower.tolist(), original_upper.tolist()],
        "target_collision_bounds_after_m": [lower.tolist(), upper.tolist()],
        "target_collision_vertical_span_before_m": float(original_upper[2] - original_lower[2]),
        "target_collision_vertical_span_after_m": float(upper[2] - lower[2]),
        "eef_offset_from_object_root_before_world_m": eef_offset_world.tolist(),
        "eef_offset_from_object_root_after_world_m": eef_offset_after.tolist(),
        "robot_or_object_state_written_directly": False,
        "safety_certified": False,
    }


def _native_region_obstacle_aware_xy_path(
    start_world_xy: np.ndarray,
    goal_world_xy: np.ndarray,
    *,
    obstacle_clearance_geometry: dict[str, Any] | None,
    goal_region_object_id: str | None,
    resolution_m: float = 0.012,
) -> tuple[list[np.ndarray] | None, dict[str, Any]]:
    """Find a short 2D object-center path around measured scene envelopes."""
    import heapq

    start = np.asarray(start_world_xy, dtype=np.float64)
    goal = np.asarray(goal_world_xy, dtype=np.float64)
    if (
        start.shape != (2,) or goal.shape != (2,)
        or not np.isfinite(start).all() or not np.isfinite(goal).all()
        or obstacle_clearance_geometry is None
        or not np.isfinite(resolution_m) or resolution_m <= 0.0
    ):
        return None, {"status": "obstacle_aware_xy_path_unavailable", "reason": "missing_or_invalid_inputs"}
    carried_radius = float(
        obstacle_clearance_geometry.get("carried_object_footprint_radius_m", 0.0)
    )
    circles: list[dict[str, Any]] = []
    for row in obstacle_clearance_geometry.get("intersecting_objects", []):
        object_id = str(row.get("object_id", ""))
        if goal_region_object_id and object_id == goal_region_object_id:
            continue
        center = np.asarray(row.get("center_world_m", []), dtype=np.float64)
        radius = row.get("horizontal_envelope_radius_m")
        if (
            center.shape != (3,) or not np.isfinite(center).all()
            or not isinstance(radius, (int, float)) or not np.isfinite(float(radius))
            or float(radius) <= 0.0
        ):
            continue
        circles.append({
            "object_id": object_id,
            "center_world_xy_m": center[:2],
            "inflated_radius_m": float(radius) + carried_radius + 0.008,
            "source_radius_m": float(radius),
        })
    if not circles:
        return None, {"status": "no_intersecting_obstacles", "obstacle_count": 0}

    all_points = [start, goal]
    for circle in circles:
        center = circle["center_world_xy_m"]
        radius = float(circle["inflated_radius_m"])
        all_points.extend((center + [-radius, -radius], center + [radius, radius]))
    stacked = np.asarray(all_points, dtype=np.float64)
    lower = stacked.min(axis=0) - 0.06
    upper = stacked.max(axis=0) + 0.06
    shape = np.maximum(3, np.ceil((upper - lower) / resolution_m).astype(int) + 1)
    if int(shape[0] * shape[1]) > 40000:
        return None, {
            "status": "obstacle_aware_xy_path_grid_too_large",
            "grid_shape": shape.tolist(), "obstacle_count": len(circles),
        }

    def to_index(point: np.ndarray) -> tuple[int, int]:
        index = np.rint((point - lower) / resolution_m).astype(int)
        return int(index[0]), int(index[1])

    def to_point(index: tuple[int, int]) -> np.ndarray:
        return lower + resolution_m * np.asarray(index, dtype=np.float64)

    def blocked(index: tuple[int, int]) -> bool:
        point = to_point(index)
        return any(
            float(np.linalg.norm(point - circle["center_world_xy_m"]))
            < float(circle["inflated_radius_m"])
            for circle in circles
        )

    start_index, goal_index = to_index(start), to_index(goal)
    # Exact live/goal poses remain valid graph endpoints even if conservative
    # envelope inflation overlaps their grid cell.
    blocked_indices = {start_index, goal_index}
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 2**0.5), (-1, 1, 2**0.5),
        (1, -1, 2**0.5), (1, 1, 2**0.5),
    ]
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start_index)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_index: None}
    cost_so_far = {start_index: 0.0}
    expansions = 0
    width, height = int(shape[0]), int(shape[1])
    while frontier and expansions < 40000:
        _, current = heapq.heappop(frontier)
        expansions += 1
        if current == goal_index:
            break
        for dx, dy, cost in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if nxt not in blocked_indices and blocked(nxt):
                continue
            if dx and dy:
                side_a, side_b = (current[0] + dx, current[1]), (current[0], current[1] + dy)
                if (
                    side_a not in blocked_indices and blocked(side_a)
                    or side_b not in blocked_indices and blocked(side_b)
                ):
                    continue
            next_cost = cost_so_far[current] + cost * resolution_m
            if next_cost >= cost_so_far.get(nxt, float("inf")):
                continue
            cost_so_far[nxt] = next_cost
            came_from[nxt] = current
            heuristic = float(np.linalg.norm(to_point(nxt) - goal))
            heapq.heappush(frontier, (next_cost + heuristic, nxt))
    if goal_index not in came_from:
        return None, {
            "status": "obstacle_aware_xy_path_not_found",
            "obstacle_count": len(circles), "grid_shape": shape.tolist(),
            "expanded_nodes": expansions,
        }
    indices = []
    cursor: tuple[int, int] | None = goal_index
    while cursor is not None:
        indices.append(cursor)
        cursor = came_from[cursor]
    raw = [start.copy()] + [to_point(index) for index in reversed(indices[1:-1])] + [goal.copy()]

    def segment_is_clear(a: np.ndarray, b: np.ndarray) -> bool:
        delta = b - a
        denom = float(np.dot(delta, delta))
        for circle in circles:
            center = circle["center_world_xy_m"]
            fraction = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(center - a, delta) / denom, 0.0, 1.0))
            closest = a + fraction * delta
            if float(np.linalg.norm(closest - center)) < float(circle["inflated_radius_m"]) + 0.002:
                return False
        return True

    compressed = [raw[0]]
    current_index = 0
    while current_index < len(raw) - 1:
        next_index = len(raw) - 1
        while next_index > current_index + 1 and not segment_is_clear(
            raw[current_index], raw[next_index],
        ):
            next_index -= 1
        compressed.append(raw[next_index])
        current_index = next_index
    return compressed, {
        "status": "obstacle_aware_xy_path_found",
        "obstacle_count": len(circles),
        "obstacles": [
            {
                "object_id": circle["object_id"],
                "center_world_xy_m": circle["center_world_xy_m"].tolist(),
                "source_radius_m": circle["source_radius_m"],
                "inflated_radius_m": circle["inflated_radius_m"],
            }
            for circle in circles
        ],
        "resolution_m": float(resolution_m),
        "expanded_nodes": expansions,
        "path_world_xy_m": [point.tolist() for point in compressed],
        "uses_physical_MuJoCo_for_execution": True,
        "object_or_robot_state_written_directly": False,
        "safety_certified": False,
    }


def _native_region_progress_route(
    start_object_world: np.ndarray,
    goal_object_world: np.ndarray,
    *,
    region: dict[str, Any],
    route_variant: str,
    keypose_side_region: np.ndarray | None = None,
    obstacle_clearance_height_world_m: float | None = None,
    obstacle_clearance_geometry: dict[str, Any] | None = None,
    goal_region_object_id: str | None = None,
    transit_height_override_world_m: float | None = None,
    transit_clearance_mode: str = "compiled_obstacle",
    enable_entry_alignment: bool = False,
    entry_direction_world: np.ndarray | None = None,
    entry_clearance_radius_m: float = 0.04,
    entry_mode: str = "over_rim",
    horizontal_entry_height_offset_m: float | None = None,
    horizontal_entry_staging_offset_limit_m: float | None = None,
    horizontal_entry_minimum_offset_m: float | None = None,
    side_detour_offset_m: float | None = None,
    preclearance_side_escape: bool = True,
    aperture_start_contact_free: bool = False,
) -> dict[str, Any]:
    """Build a task-equivalent object-space route from live and native goal poses."""
    from scipy.spatial.transform import Rotation

    start = np.asarray(start_object_world, dtype=np.float64)
    goal = np.asarray(goal_object_world, dtype=np.float64)
    region_position = np.asarray(region.get("site_position_world_m"), dtype=np.float64)
    region_rotation = np.asarray(region.get("site_rotation_world"), dtype=np.float64)
    region_size = np.asarray(region.get("site_size_m"), dtype=np.float64)
    if (
        start.shape != (4, 4) or goal.shape != (4, 4)
        or region_position.shape != (3,) or region_rotation.shape != (3, 3)
        or region_size.shape != (3,)
        or not all(np.isfinite(row).all() for row in (start, goal, region_position, region_rotation, region_size))
        or transit_clearance_mode not in {
            "compiled_obstacle", "reachability_biased", "surface_slide",
            "low_clearance",
        }
        or entry_mode not in {"over_rim", "horizontal_aperture"}
        or (
            horizontal_entry_height_offset_m is not None
            and (
                not np.isfinite(horizontal_entry_height_offset_m)
                or not 0.0 <= float(horizontal_entry_height_offset_m) <= 0.15
            )
        )
        or (
            horizontal_entry_staging_offset_limit_m is not None
            and (
                not np.isfinite(horizontal_entry_staging_offset_limit_m)
                or not 0.01 <= float(horizontal_entry_staging_offset_limit_m) <= 2.0
            )
        )
        or (
            horizontal_entry_minimum_offset_m is not None
            and (
                not np.isfinite(horizontal_entry_minimum_offset_m)
                or not 0.0 <= float(horizontal_entry_minimum_offset_m) <= 2.0
            )
        )
        or (
            side_detour_offset_m is not None
            and (
                not np.isfinite(side_detour_offset_m)
                or not 0.045 <= float(side_detour_offset_m) <= 0.35
            )
        )
        or route_variant not in {
            "direct_region_progress",
            "keypose_side_region_progress",
            "keypose_side_region_progress_reverse",
            "obstacle_aware_xy_region_progress",
        }
    ):
        raise ValueError("native_region_progress_route_requires_valid_poses_region_and_route_variant")
    if transit_height_override_world_m is not None and (
        not np.isfinite(transit_height_override_world_m)
        or not 0.02 <= float(transit_height_override_world_m) <= 2.50
    ):
        raise ValueError("native_region_transit_height_override_must_be_in_0.02_to_2.50")
    rotation = region_rotation
    local_start = rotation.T @ (start[:3, 3] - region_position)
    local_goal = rotation.T @ (goal[:3, 3] - region_position)
    entry_local_xy = None
    entry_direction_local = None
    entry_offset_m = None
    if entry_direction_world is not None:
        entry_direction = np.asarray(entry_direction_world, dtype=np.float64)
        if (
            entry_direction.shape != (3,) or not np.isfinite(entry_direction).all()
            or not np.isfinite(entry_clearance_radius_m) or entry_clearance_radius_m < 0.0
        ):
            raise ValueError("native_region_entry_direction_requires_finite_vector_and_radius")
        entry_direction_local = rotation.T @ entry_direction
        entry_direction_local[2] = 0.0
        entry_norm = float(np.linalg.norm(entry_direction_local[:2]))
        if entry_norm <= 1e-8:
            raise ValueError("native_region_entry_direction_must_lie_in_native_region_plane")
        entry_direction_local[:2] /= entry_norm
        boundary_distances = [
            float(
                region_size[index]
                - np.sign(entry_direction_local[index]) * local_goal[index]
            ) / abs(float(entry_direction_local[index]))
            for index in range(2)
            if abs(float(entry_direction_local[index])) > 1e-8
        ]
        boundary_distance = max(0.0, min(boundary_distances))
        entry_offset_m = boundary_distance + float(entry_clearance_radius_m) + 0.02
        # Placement candidates encode the direction from the live object
        # toward the selected native goal. The exterior staging point lies
        # behind that inward vector, on the side the robot approaches from.
        entry_local_xy = local_goal[:2] - entry_direction_local[:2] * entry_offset_m
    horizontal_delta = local_goal[:2] - local_start[:2]
    horizontal_distance = float(np.linalg.norm(horizontal_delta))
    clearance_delta = float(np.clip(
        max(0.35 * float(np.max(region_size[:2])), 0.20 * horizontal_distance),
        0.045, 0.16,
    ))
    minimum_transit_height = max(
        float(start[2, 3]), float(goal[2, 3]) + clearance_delta,
    )
    if transit_clearance_mode == "surface_slide":
        # Stage A may use real support-surface contact to move an object into
        # a native region.  Keep fragile low-profile grasps at the lower of
        # their live and goal heights instead of lifting into a transit band.
        # MuJoCo contacts and the native predicate remain the feasibility and
        # success tests; this route carries no safety claim.
        transit_height = min(float(start[2, 3]), float(goal[2, 3]))
    elif transit_clearance_mode == "low_clearance":
        # Clear fragile held objects just enough to avoid scraping uneven
        # support geometry during long table-to-container transports. Base the
        # small lift on the higher live/goal root height; using the lower height
        # can leave an object resting on the table for the entire transport.
        transit_height = max(float(start[2, 3]), float(goal[2, 3])) + 0.015
    elif transit_clearance_mode == "reachability_biased":
        reachability_height = max(
            float(goal[2, 3]) + 0.10,
            min(float(start[2, 3]), float(goal[2, 3]) + 0.16),
        )
        # This is Stage A task-success routing: compiled obstacle envelopes
        # provide diagnostics, while MuJoCo contacts decide what the object
        # can physically pass.  Keeping their conservative clearance as a
        # hard lower bound can place a held object beyond the Panda's vertical
        # workspace at a far-side waypoint, so prefer the reachability band.
        transit_height = reachability_height
    else:
        transit_height = minimum_transit_height
    if obstacle_clearance_height_world_m is not None and transit_clearance_mode == "compiled_obstacle":
        obstacle_height = float(obstacle_clearance_height_world_m)
        if not np.isfinite(obstacle_height):
            raise ValueError("obstacle_clearance_height_must_be_finite")
        transit_height = max(transit_height, obstacle_height)
    if transit_height_override_world_m is not None:
        # A bounded Stage A physical probe may use a measured lower transit
        # band when the compiled envelope is unreachable for the live grasp.
        # MuJoCo contacts and the native predicate still decide feasibility.
        transit_height = float(transit_height_override_world_m)
    # A native region frame can be rotated with a tipped container. Treating
    # its local z coordinate as world height becomes singular when that axis is
    # nearly horizontal: a 0.19 m world lift can turn into multi-metre local
    # coordinates and send OSC diagonally across the workspace. In that case
    # build the route in world coordinates: lift vertically first, carry to an
    # outside staging point, then enter along the measured inward direction.
    region_z_vertical_projection = abs(float(region_rotation[2, 2]))
    obstacle_aware_xy_path_world_m: list[np.ndarray] | None = None
    obstacle_aware_xy_path_diagnostic: dict[str, Any] = {
        "status": "not_requested",
    }
    if (
        route_variant == "obstacle_aware_xy_region_progress"
        and entry_mode == "over_rim"
        and region_z_vertical_projection >= 0.50
    ):
        obstacle_aware_xy_path_world_m, obstacle_aware_xy_path_diagnostic = (
            _native_region_obstacle_aware_xy_path(
                start[:2, 3], goal[:2, 3],
                obstacle_clearance_geometry=obstacle_clearance_geometry,
                goal_region_object_id=goal_region_object_id,
            )
        )
        if obstacle_aware_xy_path_world_m is not None:
            # The planner already routes to the actual goal center around the
            # measured blockers. Keeping the independently derived straight
            # over-rim staging point would reintroduce the collision corridor
            # immediately before the final placement waypoint.
            entry_local_xy = None
            entry_offset_m = None
    if region_z_vertical_projection < 0.50:
        inward = (
            np.asarray(entry_direction_world, dtype=np.float64)
            if entry_direction_world is not None else goal[:3, 3] - start[:3, 3]
        ).copy()
        inward[2] = 0.0
        inward_norm = float(np.linalg.norm(inward))
        if inward_norm <= 1e-8:
            inward = goal[:3, 3] - start[:3, 3]
            inward[2] = 0.0
            inward_norm = float(np.linalg.norm(inward))
        if inward_norm <= 1e-8:
            raise ValueError("tilted_native_region_route_requires_a_horizontal_entry_direction")
        inward /= inward_norm
        side_entry_offset = float(np.clip(
            max(float(entry_clearance_radius_m) + 0.09, 0.10), 0.10, 0.20,
        ))
        outside_position = goal[:3, 3] - inward * side_entry_offset
        transit_z = float(transit_height)
        entry_z = max(
            float(goal[2, 3]),
            min(transit_z - 0.01, float(goal[2, 3]) + min(
                0.03, max(0.01, 0.5 * float(entry_clearance_radius_m)),
            )),
        )

        def world_pose(position: np.ndarray, pose_rotation: np.ndarray) -> np.ndarray:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = pose_rotation
            pose[:3, 3] = position
            return pose

        points: list[tuple[np.ndarray, str]] = []
        start_high_position = start[:3, 3].copy()
        start_high_position[2] = transit_z
        if abs(float(start[2, 3]) - transit_z) > 0.012:
            points.append((world_pose(start_high_position, start[:3, :3]), "adaptive_clearance"))
        outside_high_position = outside_position.copy()
        outside_high_position[2] = transit_z
        rotation_start = Rotation.from_matrix(start[:3, :3])
        rotation_delta = (Rotation.from_matrix(goal[:3, :3]) * rotation_start.inv()).as_rotvec()
        high_distance = float(np.linalg.norm(outside_high_position - start_high_position))
        high_steps = max(1, int(np.ceil(high_distance / 0.045)))
        for index in range(1, high_steps + 1):
            fraction = index / high_steps
            position = start_high_position + fraction * (
                outside_high_position - start_high_position
            )
            pose_rotation = (
                Rotation.from_rotvec(rotation_delta * fraction) * rotation_start
            ).as_matrix()
            points.append((world_pose(position, pose_rotation), "region_progress"))
        outside_entry_position = outside_position.copy()
        outside_entry_position[2] = entry_z
        points.append((
            world_pose(outside_entry_position, goal[:3, :3]),
            "native_region_entry_approach",
        ))
        inside_entry_position = goal[:3, 3].copy()
        inside_entry_position[2] = entry_z
        points.append((
            world_pose(inside_entry_position, goal[:3, :3]),
            "native_region_entry_alignment",
        ))
        points.append((goal.copy(), "native_region_placement"))
        deduplicated: list[dict[str, Any]] = []
        previous = start
        for pose, role in points:
            if (
                float(np.linalg.norm(pose[:3, 3] - previous[:3, 3])) < 0.006
                and role not in {
                    "native_region_entry_approach",
                    "native_region_entry_alignment",
                    "native_region_placement",
                }
            ):
                continue
            deduplicated.append({"object_pose_world": pose.tolist(), "role": role})
            previous = pose
        entry_object_world = goal.copy()
        entry_object_world[:3, 3] = inside_entry_position
        return {
            "route_variant": route_variant,
            "waypoints": deduplicated,
            "start_region_local_m": local_start.tolist(),
            "goal_region_local_m": local_goal.tolist(),
            "horizontal_goal_distance_m": float(np.linalg.norm(
                (goal[:3, 3] - start[:3, 3])[:2]
            )),
            "adaptive_transit_height_world_m": transit_z,
            "transit_height_override_world_m": transit_height_override_world_m,
            "obstacle_aware_xy_path": obstacle_aware_xy_path_diagnostic,
            "adaptive_clearance_delta_m": clearance_delta,
            "transit_clearance_mode": transit_clearance_mode,
            "entry_mode": "world_vertical_lift_then_side_aperture_for_tilted_region",
            "horizontal_entry_height_offset_m": float(entry_z - goal[2, 3]),
            "obstacle_clearance_height_world_m": obstacle_clearance_height_world_m,
            "entry_direction_world": inward.tolist(),
            "entry_offset_m": side_entry_offset,
            "entry_object_world_pose": entry_object_world.tolist(),
            "keypose_side_reference_region": (
                np.asarray(keypose_side_region, dtype=np.float64).tolist()
                if keypose_side_region is not None else None
            ),
            "no_fixed_preplace_pose": True,
            "waypoint_frame": "world_vertical_clearance_then_measured_side_aperture_entry",
            "region_local_z_world_vertical_projection": region_z_vertical_projection,
            "tilted_region_world_route_fallback": True,
            "object_or_robot_state_written_directly": False,
        }
    side = None
    applied_side_detour_offset_m = None
    if route_variant in {
        "keypose_side_region_progress", "keypose_side_region_progress_reverse",
    } and horizontal_distance > 0.04:
        if keypose_side_region is not None:
            side = np.asarray(keypose_side_region, dtype=np.float64).copy()
            if side.shape != (2,) or not np.isfinite(side).all():
                raise ValueError("keypose_side_reference_must_be_a_finite_region_plane_vector")
            side_norm = float(np.linalg.norm(side))
        else:
            side_norm = 0.0
        if side_norm <= 1e-8:
            side = np.asarray([-horizontal_delta[1], horizontal_delta[0]], dtype=np.float64)
            side_norm = float(np.linalg.norm(side))
        if side_norm <= 1e-8:
            side = np.asarray([0.0, 1.0], dtype=np.float64)
        else:
            side /= side_norm
        if route_variant == "keypose_side_region_progress_reverse":
            side *= -1.0
        offset = float(np.clip(0.25 * horizontal_distance, 0.045, 0.12))
        if side_detour_offset_m is not None:
            # Obstacle-envelope radii are useful as diagnostics, but Stage A
            # permits physical contact.  Large envelope-derived detours can
            # exceed OSC reachability and starve the actual aperture route.
            # Keep a bounded 120 mm physical probe; MuJoCo contacts report
            # whether the smaller route clears or pushes the obstacle.
            offset = max(offset, min(float(side_detour_offset_m), 0.12))
        applied_side_detour_offset_m = offset
        midpoint_local = np.asarray([
            0.5 * (local_start[0] + local_goal[0]) + side[0] * offset,
            0.5 * (local_start[1] + local_goal[1]) + side[1] * offset,
            0.0,
        ])
    else:
        midpoint_local = None

    rotation_start = Rotation.from_matrix(start[:3, :3])
    rotation_delta = (Rotation.from_matrix(goal[:3, :3]) * rotation_start.inv()).as_rotvec()
    rotate_before_aperture_transit = bool(
        entry_mode == "horizontal_aperture"
        and float(np.linalg.norm(rotation_delta)) > 0.04
    )
    exterior_projection_m = None
    start_is_exterior_to_aperture = False
    if entry_mode == "horizontal_aperture" and entry_direction_local is not None:
        outward_local = -entry_direction_local[:2]
        exterior_projection_m = float(np.dot(
            local_start[:2] - local_goal[:2], outward_local,
        ))
        # A held object already beyond its own footprint plus a small margin
        # can descend before the remaining horizontal approach. Use the live
        # native-region boundary and verified absence of object/robot fixture
        # contacts: the carried-object footprint is not a reason to traverse a
        # second, unnecessarily distant high waypoint after the live object
        # center has already cleared the aperture. Stage A permits contact;
        # every descent and insertion remains a real MuJoCo action and BDDL
        # remains the only success oracle.
        aperture_boundary_clearance_m = boundary_distance + 0.02
        required_exterior_projection_m = max(
            float(aperture_boundary_clearance_m),
            float(horizontal_entry_minimum_offset_m)
            if horizontal_entry_minimum_offset_m is not None else 0.0,
        )
        start_is_exterior_to_aperture = bool(
            aperture_start_contact_free
            and exterior_projection_m >= required_exterior_projection_m
        )
        if (
            start_is_exterior_to_aperture
            and entry_local_xy is not None
            and entry_offset_m is not None
            and exterior_projection_m is not None
        ):
            # The live object is already beyond the aperture boundary with a
            # 20 mm margin. Descend at that measured x/y instead of spending
            # more high-approach actions moving to the conservative footprint
            # staging point. Any resulting fixture response is monitored by
            # MuJoCo and may be retried from the exact parent Snapshot.
            entry_offset_m = max(
                float(aperture_boundary_clearance_m),
                float(exterior_projection_m),
            )
            entry_local_xy = (
                local_goal[:2] - entry_direction_local[:2] * entry_offset_m
            )
        if (
            entry_mode == "horizontal_aperture"
            and entry_local_xy is not None
            and entry_offset_m is not None
            and horizontal_entry_staging_offset_limit_m is not None
        ):
            # A far exterior start can lie above a shelf or placement box.
            # Live obstacle envelopes may move the low staging point inward,
            # past blockers, while the object is still at transit height.
            entry_offset_m = min(
                float(entry_offset_m),
                float(horizontal_entry_staging_offset_limit_m),
            )
            entry_local_xy = (
                local_goal[:2] - entry_direction_local[:2] * entry_offset_m
            )
        if (
            entry_mode == "horizontal_aperture"
            and entry_local_xy is not None
            and entry_offset_m is not None
            and horizontal_entry_minimum_offset_m is not None
            and not start_is_exterior_to_aperture
        ):
            # The native-region boundary alone can place a low horizontal
            # waypoint inside the owning cabinet's compiled collision
            # envelope.  Keep descending/insertion staged outside the actual
            # fixture envelope until the object and grasping hand have cleared
            # it by the geometry-derived minimum projection.
            entry_offset_m = max(
                float(entry_offset_m),
                float(horizontal_entry_minimum_offset_m),
            )
            entry_local_xy = (
                local_goal[:2] - entry_direction_local[:2] * entry_offset_m
            )
    points: list[tuple[np.ndarray, str]] = []
    reachability_transition_local = None
    staged_reachability_transition = bool(
        entry_mode == "horizontal_aperture"
        and transit_clearance_mode == "reachability_biased"
        and entry_local_xy is not None
        and not start_is_exterior_to_aperture
        and float(start[2, 3]) - transit_height > 0.025
    )
    staged_side_escape_before_descent = bool(
        staged_reachability_transition
        and midpoint_local is not None
        and route_variant in {
            "keypose_side_region_progress",
            "keypose_side_region_progress_reverse",
        }
    )
    if staged_reachability_transition:
        outward = np.asarray(entry_local_xy, dtype=np.float64) - local_start[:2]
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm > 1e-8:
            transition_distance = min(
                0.14, max(0.06, 1.25 * float(entry_clearance_radius_m)),
                outward_norm,
            )
            reachability_transition_local = local_start.copy()
            reachability_transition_local[:2] += outward / outward_norm * transition_distance
            high_pose = start.copy()
            high_pose[:3, 3] = region_position + rotation @ np.asarray([
                reachability_transition_local[0],
                reachability_transition_local[1],
                _native_region_local_z_for_world_height(
                    region, reachability_transition_local[:2], float(start[2, 3]),
                ),
            ])
            # First move outward in the live wrist posture. Rotating while
            # holding position near a fixture can sweep Panda link 5 through
            # the fixture even when the mug itself is still clear. Once the
            # measured exterior waypoint is reached, change posture there.
            side_high_pose = None
            if staged_side_escape_before_descent:
                # A high held object can saturate a wrist joint when it lowers
                # at the exterior point before the lateral detour. MuJoCo
                # traces showed the side waypoint was being reached only after
                # this failed descent. Move to the keypose side while still at
                # the live height, then lower at that side before crossing the
                # aperture. Every pose remains an OSC waypoint and physics is
                # still the only success oracle.
                side_high_pose = start.copy()
                side_high_pose[:3, 3] = region_position + rotation @ np.asarray([
                    midpoint_local[0], midpoint_local[1],
                    _native_region_local_z_for_world_height(
                        region, midpoint_local[:2], float(start[2, 3]),
                    ),
                ])
                reachability_transition_local = midpoint_local.copy()
            if rotate_before_aperture_transit:
                points.append((high_pose.copy(), "reachability_posture_escape"))
                if side_high_pose is not None:
                    points.append((side_high_pose.copy(), "keypose_side_high_clearance"))
                posture = (
                    side_high_pose.copy() if side_high_pose is not None
                    else high_pose.copy()
                )
                posture[:3, :3] = goal[:3, :3]
                points.append((posture, "native_region_aperture_posture"))
            else:
                points.append((high_pose, "reachability_high_clearance_step"))
                if side_high_pose is not None:
                    points.append((side_high_pose.copy(), "keypose_side_high_clearance"))
            low_pose = (
                side_high_pose.copy() if side_high_pose is not None
                else high_pose.copy()
            )
            if rotate_before_aperture_transit:
                low_pose[:3, :3] = goal[:3, :3]
            low_pose[:3, 3] = region_position + rotation @ np.asarray([
                reachability_transition_local[0],
                reachability_transition_local[1],
                _native_region_local_z_for_world_height(
                    region, reachability_transition_local[:2], transit_height,
                ),
            ])
            points.append((low_pose, "reachability_descent"))
    needs_clearance = transit_height - float(start[2, 3]) > 0.012
    needs_descent = (
        not staged_reachability_transition
        and float(start[2, 3]) - transit_height > 0.012
    )
    horizontal_aperture_stage_before_descent = bool(
        entry_mode == "horizontal_aperture"
        and entry_local_xy is not None
        and needs_descent
        and not start_is_exterior_to_aperture
    )
    clearance_start_local = local_start.copy()
    preclearance_escape_pose: np.ndarray | None = None
    if (
        needs_clearance
        and side is not None
        and midpoint_local is not None
        and preclearance_side_escape
        and route_variant in {
            "keypose_side_region_progress",
            "keypose_side_region_progress_reverse",
        }
    ):
        # A side route can be selected precisely because the direct vertical
        # lift intersects a nearby fixture.  Shifting laterally only after
        # lifting still sends the held object and the robot palm through that
        # fixture; descending before the shift can drag a supported object.
        # First move the confirmed grasp sideways at its current physical
        # height, then lift at the displaced x/y and follow the high detour.
        clearance_start_local[:2] += side * offset
        clearance_escape = start.copy()
        clearance_escape[:3, 3] = region_position + rotation @ np.asarray([
            clearance_start_local[0], clearance_start_local[1],
            _native_region_local_z_for_world_height(
                region, clearance_start_local[:2], float(start[2, 3]),
            ),
        ])
        points.append((clearance_escape, "keypose_side_clearance_escape"))
        preclearance_escape_pose = clearance_escape
    if rotate_before_aperture_transit and not staged_reachability_transition:
        # Move outward in the live wrist posture before rotating beside the
        # fixture.  Use the measured side-escape pose when the route has one;
        # an in-place posture change at the original pose can sweep robot links
        # through the microwave frame before the mug starts moving away.
        posture = (
            preclearance_escape_pose.copy()
            if preclearance_escape_pose is not None else start.copy()
        )
        posture[:3, :3] = goal[:3, :3]
        points.append((posture, "native_region_aperture_posture"))
    if needs_clearance and not staged_reachability_transition:
        lifted = start.copy()
        lifted[:3, 3] = region_position + rotation @ np.asarray([
            clearance_start_local[0], clearance_start_local[1],
            _native_region_local_z_for_world_height(
                region, clearance_start_local[:2], transit_height,
            ),
        ])
        if rotate_before_aperture_transit:
            lifted[:3, :3] = goal[:3, :3]
        points.append((lifted, "adaptive_clearance"))
    elif needs_descent and not horizontal_aperture_stage_before_descent:
        lowered = start.copy()
        lowered[:3, 3] = region_position + rotation @ np.asarray([
            local_start[0], local_start[1],
            _native_region_local_z_for_world_height(
                region, local_start[:2], transit_height,
            ),
        ])
        if rotate_before_aperture_transit:
            lowered[:3, :3] = goal[:3, :3]
        points.append((lowered, "reachability_descent"))
    if (
        midpoint_local is not None
        and preclearance_escape_pose is None
        and transit_clearance_mode in {"low_clearance", "surface_slide"}
        and route_variant in {
            "keypose_side_region_progress",
            "keypose_side_region_progress_reverse",
        }
    ):
        # A single diagonal start-to-midpoint segment can still sweep a carried
        # object through the very blocker the side route is meant to avoid.
        # First translate laterally at the live longitudinal coordinate and
        # height, then advance along the detoured corridor. Keep this a physical
        # OSC waypoint; Stage A records any contact and native BDDL remains the
        # success oracle.
        side_escape_local = local_start.copy()
        side_escape_local[:2] += side * offset
        side_escape_local[2] = _native_region_local_z_for_world_height(
            region, side_escape_local[:2], (
                float(transit_height) if needs_clearance else float(start[2, 3])
            ),
        )
        side_escape = np.eye(4, dtype=np.float64)
        side_escape[:3, :3] = start[:3, :3]
        side_escape[:3, 3] = region_position + rotation @ side_escape_local
        points.append((side_escape, "keypose_side_lateral_escape"))
    if midpoint_local is not None:
        midpoint_local[2] = _native_region_local_z_for_world_height(
            region, midpoint_local[:2], transit_height,
        )
        midpoint = np.eye(4, dtype=np.float64)
        midpoint[:3, :3] = (
            goal[:3, :3] if rotate_before_aperture_transit else start[:3, :3]
        )
        midpoint[:3, 3] = region_position + rotation @ midpoint_local
        points.append((midpoint, "keypose_side_detour"))

    transit_start = (
        midpoint_local
        if midpoint_local is not None
        else reachability_transition_local.copy()
        if reachability_transition_local is not None
        else local_start.copy()
    )
    transit_goal = local_goal.copy()
    if entry_local_xy is not None:
        transit_goal[:2] = entry_local_xy
    # A horizontal aperture must be approached from outside the fixture. When
    # the live object starts above the container, first move laterally to that
    # exterior staging point at the live height, then descend there. Lowering
    # at the live x/y can drive a mug into the microwave top/front before the
    # aperture route has reached its outside waypoint.
    transit_path_height = (
        float(start[2, 3])
        if horizontal_aperture_stage_before_descent else transit_height
    )
    transit_start[2] = _native_region_local_z_for_world_height(
        region, transit_start[:2], transit_path_height,
    )
    transit_goal[2] = _native_region_local_z_for_world_height(
        region, transit_goal[:2], transit_path_height,
    )
    if obstacle_aware_xy_path_world_m is not None:
        path_lengths = [0.0]
        for a, b in zip(
            obstacle_aware_xy_path_world_m[:-1], obstacle_aware_xy_path_world_m[1:],
        ):
            path_lengths.append(path_lengths[-1] + float(np.linalg.norm(b - a)))
        total_path_length = max(path_lengths[-1], 1e-8)
        for index, xy in enumerate(obstacle_aware_xy_path_world_m[1:-1], start=1):
            fraction = float(path_lengths[index] / total_path_length)
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = [float(xy[0]), float(xy[1]), float(transit_path_height)]
            pose[:3, :3] = (
                goal[:3, :3].copy()
                if rotate_before_aperture_transit
                else (Rotation.from_rotvec(rotation_delta * fraction) * rotation_start).as_matrix()
            )
            points.append((pose, "obstacle_aware_region_progress"))
    else:
        transit_distance = float(np.linalg.norm(transit_goal[:2] - transit_start[:2]))
        segment_count = max(1, int(np.ceil(transit_distance / 0.045)))
        for index in range(1, segment_count + 1):
            fraction = index / segment_count
            local = transit_start + fraction * (transit_goal - transit_start)
            local[2] = _native_region_local_z_for_world_height(
                region, local[:2], transit_path_height,
            )
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = region_position + rotation @ local
            pose[:3, :3] = (
                goal[:3, :3].copy()
                if rotate_before_aperture_transit
                else start[:3, :3].copy()
                if entry_mode == "horizontal_aperture"
                else (Rotation.from_rotvec(rotation_delta * fraction) * rotation_start).as_matrix()
            )
            points.append((
                pose,
                "native_region_entry_approach"
                if entry_local_xy is not None and index == segment_count
                else "region_progress",
            ))
    if (
        entry_local_xy is not None
        and entry_mode == "horizontal_aperture"
        and float(np.linalg.norm(rotation_delta)) > 0.04
        and not rotate_before_aperture_transit
    ):
        # Reach the aperture outside waypoint before changing the object's
        # posture; this avoids coupling a long translation with wrist rotation.
        posture = np.eye(4, dtype=np.float64)
        posture[:3, :3] = goal[:3, :3]
        posture[:3, 3] = region_position + rotation @ np.asarray([
            entry_local_xy[0], entry_local_xy[1],
            _native_region_local_z_for_world_height(
                region, entry_local_xy, transit_height,
            ),
        ])
        points.append((posture, "native_region_aperture_posture"))
    entry_object_world = None
    if entry_local_xy is not None:
        entry_object_world = goal.copy()
        entry_inside_local = local_goal.copy()
        if entry_mode == "horizontal_aperture":
            # Some native In regions are reached through a front or side
            # aperture (for example, an open microwave door), not through an
            # open top. Lower the carried object outside the container and
            # cross the aperture horizontally before the final descent.
            default_entry_offset = min(
                0.05, max(0.02, 0.25 * (transit_height - float(goal[2, 3]))),
            )
            entry_height_offset = (
                default_entry_offset
                if horizontal_entry_height_offset_m is None
                else float(horizontal_entry_height_offset_m)
            )
            entry_height_world = max(
                float(goal[2, 3]),
                min(
                    transit_path_height - 0.01,
                    float(goal[2, 3]) + entry_height_offset,
                ),
            )
            outside_local = np.asarray([
                entry_local_xy[0], entry_local_xy[1],
                _native_region_local_z_for_world_height(
                    region, entry_local_xy, entry_height_world,
                ),
            ], dtype=np.float64)
            outside_low = goal.copy()
            outside_low[:3, 3] = region_position + rotation @ outside_local
            points.append((outside_low, "native_region_entry_descent"))
            entry_inside_local[2] = _native_region_local_z_for_world_height(
                region, local_goal[:2], entry_height_world,
            )
            entry_object_world[:3, 3] = region_position + rotation @ entry_inside_local
            points.append((entry_object_world.copy(), "native_region_entry_alignment"))
        else:
            # Open-top containers should cross their rim above its clearance
            # height, then descend while already inside the region.
            entry_inside_local[2] = _native_region_local_z_for_world_height(
                region, local_goal[:2], transit_height,
            )
            entry_object_world[:3, 3] = region_position + rotation @ entry_inside_local
            points.append((entry_object_world.copy(), "native_region_entry_alignment"))
    elif enable_entry_alignment:
        entry = goal.copy()
        entry_local = local_goal.copy()
        entry_local[2] = _native_region_local_z_for_world_height(
            region, entry_local[:2], transit_height,
        )
        entry[:3, 3] = region_position + region_rotation @ entry_local
        points.append((entry, "native_region_entry_alignment"))
    else:
        # Finish horizontal alignment while the object is still at transit
        # height. Receding 45 mm waypoints can each end with a small residual;
        # coupling their accumulated xy error to a long final descent caused
        # OSC to lower an object beside the support and then stall.
        alignment = goal.copy()
        alignment_local = local_goal.copy()
        alignment_local[2] = _native_region_local_z_for_world_height(
            region, alignment_local[:2], transit_height,
        )
        alignment[:3, 3] = region_position + region_rotation @ alignment_local
        points.append((alignment, "native_region_goal_alignment"))
    if transit_height - float(goal[2, 3]) > 0.12:
        # Split a long vertical placement into bounded physical segments so
        # each segment fits the waypoint controller's action cap.
        intermediate_descent = goal.copy()
        intermediate_descent[:3, 3] = region_position + region_rotation @ np.asarray([
            local_goal[0], local_goal[1],
            _native_region_local_z_for_world_height(
                region, local_goal[:2], 0.5 * (transit_height + float(goal[2, 3])),
            ),
        ])
        points.append((intermediate_descent, "native_region_preplacement_descent"))
    final = goal.copy()
    points.append((final, "native_region_placement"))
    deduplicated: list[dict[str, Any]] = []
    previous = start
    for pose, role in points:
        if (
            float(np.linalg.norm(pose[:3, 3] - previous[:3, 3])) < 0.006
            and role not in {
                "native_region_placement", "native_region_entry_alignment",
                "native_region_goal_alignment", "native_region_preplacement_descent",
                "native_region_entry_approach", "native_region_entry_descent",
                "native_region_aperture_posture",
                "native_region_entry_posture",
            }
        ):
            continue
        deduplicated.append({"object_pose_world": pose.tolist(), "role": role})
        previous = pose
    return {
        "route_variant": route_variant,
        "waypoints": deduplicated,
        "start_region_local_m": local_start.tolist(),
        "goal_region_local_m": local_goal.tolist(),
        "goal_object_world": goal.tolist(),
        "horizontal_goal_distance_m": horizontal_distance,
        "adaptive_transit_height_world_m": transit_height,
        "transit_height_override_world_m": transit_height_override_world_m,
        "obstacle_aware_xy_path": obstacle_aware_xy_path_diagnostic,
        "adaptive_clearance_delta_m": clearance_delta,
        "transit_clearance_mode": transit_clearance_mode,
        "entry_mode": entry_mode,
        "requested_side_detour_offset_m": side_detour_offset_m,
        "applied_side_detour_offset_m": applied_side_detour_offset_m,
        "staged_reachability_transition": (
            {
                "enabled": True,
                "local_transition_point_m": reachability_transition_local.tolist(),
                "world_height_before_m": float(start[2, 3]),
                "world_height_after_m": float(transit_height),
                "keypose_side_escape_before_descent": staged_side_escape_before_descent,
                "compiled_obstacle_clearance_is_diagnostic_only": True,
            }
            if reachability_transition_local is not None else {"enabled": False}
        ),
        "horizontal_aperture_staged_outward_before_descent": {
            "enabled": horizontal_aperture_stage_before_descent,
            "source_object_world_height_m": float(start[2, 3]),
            "horizontal_staging_height_world_m": (
                float(transit_path_height)
                if horizontal_aperture_stage_before_descent else None
            ),
            "descent_world_height_m": float(transit_height),
            "outside_entry_height_world_m": (
                float(entry_height_world)
                if entry_mode == "horizontal_aperture" and entry_local_xy is not None
                else None
            ),
            "native_BDDL_remains_success_oracle": True,
        },
        "horizontal_aperture_exterior_staging": {
            "enabled": entry_mode == "horizontal_aperture",
            "start_is_exterior_to_aperture": start_is_exterior_to_aperture,
            "outward_projection_from_goal_m": exterior_projection_m,
            "required_projection_m": (
                float(required_exterior_projection_m)
                if entry_mode == "horizontal_aperture" else None
            ),
            "projection_reference": (
                "compiled_fixture_and_carried_robot_envelope"
                if horizontal_entry_minimum_offset_m is not None
                and horizontal_entry_minimum_offset_m > float(boundary_distance) + 0.02
                else "native_region_boundary_plus_20mm"
            ),
            "uses_measured_current_xy_after_clearance": start_is_exterior_to_aperture,
            "outside_starts_descend_before_horizontal_transit": start_is_exterior_to_aperture,
            "physical_contact_free_status_is_verified_by_live_MuJoCo_contacts": bool(
                aperture_start_contact_free
            ),
        },
        "horizontal_entry_height_offset_m": (
            horizontal_entry_height_offset_m
            if entry_mode == "horizontal_aperture" else None
        ),
        "horizontal_entry_staging_offset_limit_m": horizontal_entry_staging_offset_limit_m,
        "side_detour_offset_m": side_detour_offset_m,
        "obstacle_clearance_height_world_m": obstacle_clearance_height_world_m,
        "entry_direction_region_local": (
            entry_direction_local.tolist() if entry_direction_local is not None else None
        ),
        "entry_offset_m": entry_offset_m,
        "entry_object_world_pose": entry_object_world.tolist() if entry_object_world is not None else None,
        "keypose_side_reference_region": (
            np.asarray(keypose_side_region, dtype=np.float64).tolist()
            if keypose_side_region is not None else None
        ),
        "no_fixed_preplace_pose": True,
        "waypoint_frame": "target_object_world_pose_retargeted_from_live_object_to_EEF_transform",
    }


def _native_region_entry_transit_poses(
    start_object_world: np.ndarray,
    goal_object_world: np.ndarray,
    *,
    region: dict[str, Any],
    entry_direction_world: np.ndarray,
    entry_clearance_radius_m: float,
    obstacle_clearance_height_world_m: float,
) -> dict[str, Any]:
    """Return the geometry-derived outside and inside high poses for an In goal."""
    route = _native_region_progress_route(
        start_object_world,
        goal_object_world,
        region=region,
        route_variant="direct_region_progress",
        obstacle_clearance_height_world_m=obstacle_clearance_height_world_m,
        entry_direction_world=entry_direction_world,
        entry_clearance_radius_m=entry_clearance_radius_m,
    )
    outside_rows = [
        row for row in route["waypoints"]
        if row.get("role") == "native_region_entry_approach"
    ]
    inside_rows = [
        row for row in route["waypoints"]
        if row.get("role") == "native_region_entry_alignment"
    ]
    if not outside_rows or not inside_rows:
        raise RuntimeError("native_In_entry_route_missing_outside_or_inside_high_pose")
    return {
        "route": route,
        "outside_entry_object_world": np.asarray(
            outside_rows[-1]["object_pose_world"], dtype=np.float64,
        ),
        "inside_alignment_object_world": np.asarray(
            inside_rows[-1]["object_pose_world"], dtype=np.float64,
        ),
    }


def _native_region_progress_waypoint_target(
    region: dict[str, Any],
    current_object_world: np.ndarray,
    waypoint_object_world: np.ndarray,
    transit_height_world_m: float,
    *,
    maximum_segment_m: float = 0.045,
    waypoint_tolerance_m: float = 0.015,
) -> tuple[np.ndarray | None, float]:
    """Track the active route waypoint in region coordinates, preserving side detours."""
    current = np.asarray(current_object_world, dtype=np.float64)
    waypoint = np.asarray(waypoint_object_world, dtype=np.float64)
    region_position = np.asarray(region.get("site_position_world_m"), dtype=np.float64)
    region_rotation = np.asarray(region.get("site_rotation_world"), dtype=np.float64)
    if (
        current.shape != (4, 4) or waypoint.shape != (4, 4)
        or region_position.shape != (3,) or region_rotation.shape != (3, 3)
        or not all(np.isfinite(value).all() for value in (current, waypoint, region_position, region_rotation))
        or not np.isfinite([transit_height_world_m, maximum_segment_m, waypoint_tolerance_m]).all()
        or maximum_segment_m <= 0.0 or waypoint_tolerance_m < 0.0
    ):
        raise ValueError("native_region_waypoint_tracking_requires_valid_poses_and_limits")
    local_current = region_rotation.T @ (current[:3, 3] - region_position)
    local_waypoint = region_rotation.T @ (waypoint[:3, 3] - region_position)
    delta = local_waypoint[:2] - local_current[:2]
    distance = float(np.linalg.norm(delta))
    if distance <= waypoint_tolerance_m:
        return None, distance
    next_local = local_current.copy()
    next_local[:2] += delta * (min(maximum_segment_m, distance) / distance)
    next_local[2] = _native_region_local_z_for_world_height(
        region, next_local[:2], float(transit_height_world_m),
    )
    target = waypoint.copy()
    target[:3, 3] = region_position + region_rotation @ next_local
    return target, distance


def _native_object_relative_eef_target_world(
    desired_object_world: np.ndarray,
    live_object_world: np.ndarray,
    live_eef_world: np.ndarray,
) -> np.ndarray:
    """Retarget EEF pose from the live held-object-to-EEF transform."""
    desired = np.asarray(desired_object_world, dtype=np.float64)
    live_object = np.asarray(live_object_world, dtype=np.float64)
    live_eef = np.asarray(live_eef_world, dtype=np.float64)
    if (
        desired.shape != (4, 4) or live_object.shape != (4, 4) or live_eef.shape != (4, 4)
        or not all(np.isfinite(value).all() for value in (desired, live_object, live_eef))
        or abs(float(np.linalg.det(live_object[:3, :3]))) < 1e-8
    ):
        raise ValueError("native_object_relative_eef_target_requires_valid_finite_poses")
    return desired @ np.linalg.inv(live_object) @ live_eef


def _native_region_progress_tracking_tolerances(
    role: str,
    distance_to_goal_m: float | None,
    *,
    native_goal_operator: str | None = None,
    clearance_lift_delta_m: float | None = None,
) -> tuple[float, float]:
    """Set route-following tolerances; native BDDL remains the task-success gate."""
    if role == "native_region_aperture_posture":
        # This waypoint only changes the held object's aperture orientation.
        # Its position is deliberately retargeted from the live object pose,
        # then the remaining route is rebuilt from the physically reached pose.
        # Match the route builder's 0.04 rad rotation-change threshold so a
        # live-pose replan cannot regenerate the posture it just completed.
        return 0.012, 0.035
    if role == "adaptive_clearance":
        # Clearance is a prerequisite for later horizontal transit for both
        # On and In goals. Base tolerance on the actual vertical step: require
        # at least half of a small support-clearance lift, while allowing a
        # 20 mm residual on a tall lift. Physical trials reached a stable
        # 17 mm EEF residual with the carried object's bottom still 36 mm
        # above the highest corridor obstacle; the old 15 mm cap turned that
        # adequate Stage A transit state into a false terminal failure.
        delta = (
            abs(float(clearance_lift_delta_m))
            if clearance_lift_delta_m is not None
            and np.isfinite(float(clearance_lift_delta_m))
            else 0.020
        )
        tolerance = 0.020 if delta >= 0.040 else max(0.006, 0.5 * delta)
        return tolerance, 0.65
    if role == "keypose_side_clearance_escape":
        # This waypoint must physically clear the near fixture before the
        # subsequent vertical lift.  A loose transit tolerance could lift
        # while the object is still inside the obstacle's horizontal envelope.
        return 0.012, 0.12
    if role == "robot_environment_contact_escape":
        # The contact-driven retreat is measured from a live robot/scene
        # collision. Track it closely so the next lift starts beyond the
        # observed contact rather than repeating it from a partially moved pose.
        return 0.012, 0.12
    if role == "robot_environment_contact_reorientation":
        # A cylindrical held object can be yawed in place to unthread a Panda
        # link trapped between it and a neighboring movable fixture. Preserve
        # its live root position and require actual orientation progress.
        return 0.012, 0.08
    if role == "native_region_goal_alignment":
        # This is the final xy correction above the support, so prevent its
        # residual from being carried into the downward placement segment.
        return 0.005, 0.65 if native_goal_operator in {"on", "in"} else 0.12
    if role == "native_region_preplacement_descent":
        return 0.012, 0.12
    if role == "native_region_placement" and native_goal_operator == "on":
        # On has no yaw constraint, but the object root must physically settle
        # onto its support closely enough for the native predicate to fire.
        return 0.005, 0.65
    if native_goal_operator == "in" and role in {
        "adaptive_clearance_with_keypose_side_detour", "region_progress",
    }:
        # In container transit, the held object can lag a receding waypoint in
        # yaw while making steady physical progress toward the aperture.  A
        # cheese-to-basket branch plateaued at 18 mm after moving the object
        # toward a 45 mm transit waypoint. Accept that physically measured
        # residual and retarget the next segment from the live grasp transform;
        # this remains below the route segment length and does not relax final
        # entry, placement, or native BDDL success checks.
        return 0.020, 0.65
    if role == "region_progress":
        # On has no orientation predicate. The former 40 mm position radius
        # let each ~45 mm carry segment advance only a few millimetres, leaving
        # the object far from its final placement waypoint after the route.
        # A 15 mm local tolerance lets a live-replanned segment advance when
        # OSC stalls within a few millimetres of its waypoint; the next segment
        # retargets from the actual object pose and native BDDL remains the
        # sole placement-success gate.
        return 0.015, 0.65 if native_goal_operator == "on" else 0.12
    if role in {
        "adaptive_clearance",
        "adaptive_clearance_with_keypose_side_detour",
        "keypose_side_detour",
        "reachability_descent",
        "region_progress",
        "native_region_entry_approach",
    }:
        # Side-detour, height-adjustment, and entry-approach points guide
        # transit rather than define precise placement poses.
        return 0.040, 0.12
    return 0.015, 0.12


def _native_region_corridor_obstacle_clearance(
    *,
    start_object_world: np.ndarray,
    goal_object_world: np.ndarray,
    region: dict[str, Any],
    target_context: dict[str, Any],
    object_to_eef: np.ndarray,
    scene_envelopes: dict[str, Any],
    excluded_object_ids: set[str],
    expected_model_sha256: str,
    clearance_margin_m: float = 0.05,
) -> dict[str, Any]:
    """Derive an overflight height from compiled objects crossing the carried path."""
    start = np.asarray(start_object_world, dtype=np.float64)
    goal = np.asarray(goal_object_world, dtype=np.float64)
    region_position = np.asarray(region.get("site_position_world_m"), dtype=np.float64)
    region_rotation = np.asarray(region.get("site_rotation_world"), dtype=np.float64)
    object_to_eef = np.asarray(object_to_eef, dtype=np.float64)
    if (
        scene_envelopes.get("status") != "complete"
        or scene_envelopes.get("model_geometry_sha256") != expected_model_sha256
        or start.shape != (4, 4) or goal.shape != (4, 4)
        or region_position.shape != (3,) or region_rotation.shape != (3, 3)
        or object_to_eef.shape != (4, 4)
        or not np.isfinite(start).all() or not np.isfinite(goal).all()
        or not np.isfinite(region_position).all() or not np.isfinite(region_rotation).all()
        or not np.isfinite(object_to_eef).all()
        or not np.isfinite(clearance_margin_m) or clearance_margin_m < 0.0
    ):
        raise ValueError("native_region_obstacle_clearance_requires_complete_live_geometry")
    target_radius = _placement_footprint_radius(
        target_context, region_rotation, start[:3, :3],
    )
    eef_offset = float(np.linalg.norm(object_to_eef[:2, 3]))
    corridor_radius = target_radius + eef_offset + 0.035
    start_xy, goal_xy = start[:2, 3], goal[:2, 3]
    line = goal_xy - start_xy
    line_sq = float(np.dot(line, line))
    intersecting = []
    for row in scene_envelopes.get("objects", []):
        object_id = str(row.get("object_id", ""))
        if object_id in excluded_object_ids:
            continue
        root_position = np.asarray(row.get("root_position_world_m"), dtype=np.float64)
        root_rotation = np.asarray(row.get("root_rotation_world"), dtype=np.float64)
        lower = np.asarray(row.get("envelope_lower_root_m"), dtype=np.float64)
        upper = np.asarray(row.get("envelope_upper_root_m"), dtype=np.float64)
        if (
            root_position.shape != (3,) or root_rotation.shape != (3, 3)
            or lower.shape != (3,) or upper.shape != (3,)
            or not all(np.isfinite(value).all() for value in (root_position, root_rotation, lower, upper))
            or np.any(upper < lower)
        ):
            continue
        center_local = (lower + upper) * 0.5
        half_local = (upper - lower) * 0.5
        center_world = root_position + root_rotation @ center_local
        half_world = np.abs(root_rotation) @ half_local
        if line_sq <= 1e-12:
            nearest = start_xy
        else:
            fraction = float(np.clip(np.dot(center_world[:2] - start_xy, line) / line_sq, 0.0, 1.0))
            nearest = start_xy + fraction * line
        distance_xy = float(np.linalg.norm(center_world[:2] - nearest))
        obstacle_radius = float(np.linalg.norm(half_world[:2]))
        if distance_xy > corridor_radius + obstacle_radius:
            continue
        intersecting.append({
            "object_id": object_id,
            "mapping_name": row.get("mapping_name"),
            "center_world_m": center_world.tolist(),
            "horizontal_distance_to_route_m": distance_xy,
            "horizontal_envelope_radius_m": obstacle_radius,
            "top_world_m": float(center_world[2] + half_world[2]),
            "collision_geom_count": row.get("collision_geom_count"),
        })
    maximum_obstacle_top = max(
        (row["top_world_m"] for row in intersecting),
        default=float("-inf"),
    )
    # Obstacle top heights and transit heights are expressed in world z.
    # Native container frames may point their z axis downward, so their local
    # support offset cannot be combined with a world-space obstacle height.
    lowest_target_offset = _placement_lowest_geometry_offset(
        target_context, np.eye(3, dtype=np.float64), start[:3, :3],
    )
    required_root_height_world = (
        maximum_obstacle_top + float(clearance_margin_m) - lowest_target_offset
        if np.isfinite(maximum_obstacle_top) else None
    )
    return {
        "status": "complete",
        "geometry_source": "active_compiled_MuJoCo_object_and_fixture_collision_envelopes",
        "corridor_centerline_world_m": [start_xy.tolist(), goal_xy.tolist()],
        "carried_object_footprint_radius_m": target_radius,
        "object_to_eef_horizontal_offset_m": eef_offset,
        "corridor_radius_m": corridor_radius,
        "clearance_margin_m": float(clearance_margin_m),
        "intersecting_objects": intersecting,
        "maximum_intersecting_obstacle_top_world_m": (
            maximum_obstacle_top if np.isfinite(maximum_obstacle_top) else None
        ),
        "required_carried_object_root_height_world_m": required_root_height_world,
        "required_transit_height_world_m": required_root_height_world,
        "safety_certified": False,
    }


def _generate_native_region_placement_candidates(
    *,
    predicate: list[str],
    region: dict[str, Any],
    target_context: dict[str, Any],
    occupied_contexts: list[dict[str, Any]],
    maximum_candidates: int = 4,
    preferred_yaw_delta_degrees: float | None = None,
) -> list[dict[str, Any]]:
    """Rank object-relative placements inside a native BDDL support/container site.

    Geometry is used to rank physical layouts, never to certify safety or reject a
    candidate solely for a conservative clearance estimate. Every selected target
    is still reached and settled by real OSC actions in MuJoCo.
    """
    from scipy.spatial.transform import Rotation

    if not isinstance(predicate, list) or len(predicate) != 3 or predicate[0].lower() not in {"on", "in"}:
        raise ValueError("placement_search_requires_an_active_native_On_or_In_predicate")
    if not 1 <= maximum_candidates <= 16:
        raise ValueError("placement_search_candidate_budget_must_be_in_1_to_16")
    region_position = np.asarray(region.get("site_position_world_m"), dtype=np.float64)
    region_rotation = np.asarray(region.get("site_rotation_world"), dtype=np.float64)
    region_size = np.asarray(region.get("site_size_m"), dtype=np.float64)
    target_root = np.asarray(target_context.get("target_root_pose_world", {}).get("position_m"), dtype=np.float64)
    target_rotation = np.asarray(target_context.get("target_root_pose_world", {}).get("rotation"), dtype=np.float64)
    if (
        region_position.shape != (3,) or region_rotation.shape != (3, 3)
        or region_size.shape != (3,) or target_root.shape != (3,) or target_rotation.shape != (3, 3)
        or not all(np.isfinite(value).all() for value in (region_position, region_rotation, region_size, target_root, target_rotation))
        or np.any(region_size[:2] <= 0.0)
    ):
        raise ValueError("placement_search_native_region_or_target_geometry_is_malformed")
    target_rotation = target_rotation.copy()
    target_shape = _placement_shape_signature(target_context)
    matching_occupants = [
        context for context in occupied_contexts
        if (
            target_shape and _placement_shape_signature(context) == target_shape
        ) or _placement_orientation_compatible(target_context, context)
    ]
    # Native On constrains support membership and height, not yaw. Preserve the
    # held object's live orientation so a completed sibling does not impose an
    # unnecessary final pose that may be outside the current arm's IK workspace.
    verb = predicate[0].lower()
    support_z = region_position[2] + region_size[2] if verb == "on" else region_position[2] - region_size[2]
    support_local_z = float((region_rotation.T @ (np.asarray([region_position[0], region_position[1], support_z]) - region_position))[2])
    if verb == "in":
        # Containment checks the object's body position against the native site
        # box. For inverted site Z frames, the physical floor is at +local-Z;
        # keep the root inside that boundary while placing collision geometry
        # against the gravity-facing side when possible.
        region_z_world_up = float(region_rotation[2, 2])
        if abs(region_z_world_up) > 0.5:
            support_local_z = -float(np.sign(region_z_world_up)) * float(region_size[2])
        else:
            support_local_z = 0.0
    orientation_rows = (
        _native_in_region_orientation_candidates(
            target_context=target_context,
            region_rotation=region_rotation,
            target_rotation=target_rotation,
            region_size=region_size,
            maximum_candidates=(
                1 if preferred_yaw_delta_degrees is not None
                else min(maximum_candidates, 16)
            ),
            preferred_yaw_delta_degrees=preferred_yaw_delta_degrees,
        )
        if verb == "in" else ([{
            "rotation_world": (
                Rotation.from_rotvec(
                    np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
                    * np.deg2rad(float(preferred_yaw_delta_degrees))
                ).as_matrix() @ target_rotation
            ),
            "yaw_delta_degrees": float(preferred_yaw_delta_degrees),
            "collision_bounds_region_m": [None, None],
            "footprint_half_extents_region_m": None,
            "geometry_center_offset_region_m": [0.0, 0.0],
            "fit_cost": 0.0,
        }] if preferred_yaw_delta_degrees is not None else [{
            "rotation_world": target_rotation,
            "yaw_delta_degrees": 0.0,
            "collision_bounds_region_m": [None, None],
            "footprint_half_extents_region_m": None,
            "geometry_center_offset_region_m": [0.0, 0.0],
            "fit_cost": 0.0,
        }])
    )
    matching_heights: list[float] = []
    if verb == "on":
        # Reuse a physically settled same-region height when available (for
        # example, the first member of a multi-object On goal).
        target_radius = _placement_footprint_radius(target_context, region_rotation, target_rotation)
        for context in occupied_contexts:
            pose = context.get("target_root_pose_world", {})
            position = np.asarray(pose.get("position_m"), dtype=np.float64)
            if position.shape == (3,) and np.isfinite(position).all():
                occupied_radius = _placement_footprint_radius(context, region_rotation)
                if abs(occupied_radius - target_radius) <= max(0.015, 0.30 * target_radius):
                    matching_heights.append(float((region_rotation.T @ (position - region_position))[2]))
    matching_height = float(np.median(matching_heights)) if matching_heights and verb == "on" else None

    target_half = np.minimum(region_size[:2] * 0.78, region_size[:2] - 1e-4)
    sample_x = np.linspace(-target_half[0], target_half[0], 5)
    sample_y = np.linspace(-target_half[1], target_half[1], 5)
    offsets = {(0.0, 0.0)}
    offsets.update((float(x), float(y)) for x in sample_x for y in sample_y)
    occupied: list[tuple[np.ndarray, float, str]] = []
    for context in occupied_contexts:
        pose = context.get("target_root_pose_world", {})
        position = np.asarray(pose.get("position_m"), dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            continue
        local = region_rotation.T @ (position - region_position)
        radius = _placement_footprint_radius(context, region_rotation)
        occupied.append((local[:2], radius, str(context.get("target_object_id", "occupied_object"))))

    current_local = region_rotation.T @ (target_root - region_position)
    rows = []
    for orientation in orientation_rows:
        candidate_rotation = np.asarray(orientation["rotation_world"], dtype=np.float64)
        target_radius = _placement_footprint_radius(
            target_context, region_rotation, candidate_rotation,
        )
        target_bottom = _placement_lowest_geometry_offset(
            target_context, region_rotation, candidate_rotation,
        )
        bounds = orientation["collision_bounds_region_m"]
        if verb == "in":
            collision_lower = np.asarray(bounds[0], dtype=np.float64)
            collision_upper = np.asarray(bounds[1], dtype=np.float64)
            centered_root_xy = -0.5 * (collision_lower[:2] + collision_upper[:2])
            orientation_fit_cost = float(orientation["fit_cost"])
            layout_offsets = set(offsets)
            layout_offsets.add((float(centered_root_xy[0]), float(centered_root_xy[1])))
            if abs(region_z_world_up) > 0.5:
                support_geometry_offset = (
                    float(collision_upper[2]) if support_local_z > 0.0
                    else float(collision_lower[2])
                )
                support_root_local_z = support_local_z - support_geometry_offset
                region_interior_margin = min(0.005, 0.15 * float(region_size[2]))
                root_local_z = float(np.clip(
                    support_root_local_z,
                    -float(region_size[2]) + region_interior_margin,
                    float(region_size[2]) - region_interior_margin,
                ))
            else:
                root_local_z = 0.0
        else:
            collision_lower = collision_upper = np.zeros(3, dtype=np.float64)
            centered_root_xy = np.zeros(2, dtype=np.float64)
            orientation_fit_cost = 0.0
            layout_offsets = offsets
            root_local_z = support_local_z - target_bottom
            if matching_height is not None:
                root_local_z = matching_height
        for offset in sorted(layout_offsets, key=lambda value: (value[0], value[1])):
            local_xy = np.asarray(offset, dtype=np.float64)
            position = region_position + region_rotation @ np.asarray([local_xy[0], local_xy[1], root_local_z])
            clearances = [
                float(np.linalg.norm(local_xy - other_xy) - target_radius - other_radius)
                for other_xy, other_radius, _ in occupied
            ]
            limiting_clearance = min(clearances, default=float("inf"))
            # Prefer room from completed objects, a collision-envelope fit for
            # In goals, and then the shortest physical transfer from the live pose.
            travel = float(np.linalg.norm(position - target_root))
            support_centering_penalty = (
                2.0 * float(np.linalg.norm(local_xy))
                if (
                    verb == "on"
                    or region.get("support_geometry_kind") == "compiled_object_collision_envelope"
                )
                else 0.0
            )
            clearance_score = limiting_clearance if np.isfinite(limiting_clearance) else 0.0
            envelope_center_error = float(np.linalg.norm(local_xy - centered_root_xy)) if verb == "in" else 0.0
            fit_penalty = 10.0 * orientation_fit_cost
            if verb == "in":
                translated_lower = collision_lower[:2] + local_xy
                translated_upper = collision_upper[:2] + local_xy
                outside = (
                    np.maximum(-region_size[:2] - translated_lower, 0.0)
                    + np.maximum(translated_upper - region_size[:2], 0.0)
                ) / region_size[:2]
                fit_penalty += 10.0 * float(np.dot(outside, outside))
            score = clearance_score - 0.15 * travel - support_centering_penalty
            if verb == "in":
                score -= fit_penalty + 2.0 * envelope_center_error
            if occupied:
                nearest = min(occupied, key=lambda row: float(np.linalg.norm(local_xy - row[0])))
                away_local = local_xy - nearest[0]
            else:
                away_local = local_xy - current_local[:2]
            norm = float(np.linalg.norm(away_local))
            if norm <= 1e-8:
                away_local = np.asarray([0.0, 1.0])
            else:
                away_local = away_local / norm
            entry = region_rotation @ np.asarray([away_local[0], away_local[1], 0.0])
            rows.append({
                "candidate_id": f"region_layout_{len(rows):02d}",
                "predicate": list(predicate),
                "position_world_m": position.tolist(),
                "rotation_world": candidate_rotation.tolist(),
                "region_local_xy_m": local_xy.tolist(),
                "entry_direction_world": entry.tolist(),
                "estimated_completed_object_clearance_m": limiting_clearance if np.isfinite(limiting_clearance) else None,
                "support_centering_penalty_m": support_centering_penalty,
                "travel_distance_m": travel,
                "ranking_score": score,
                "root_height_source": "settled_same_region_object" if matching_heights and verb == "on" else "compiled_collision_geometry_support_extent",
                "orientation_source": (
                    "region_collision_envelope_fit_yaw_search" if verb == "in"
                    else "current_object_pose_native_On_has_no_orientation_constraint"
                    if verb == "on" and matching_occupants
                    else "current_object_pose"
                ),
                "orientation_yaw_delta_degrees": orientation["yaw_delta_degrees"],
                "orientation_yaw_delta_degrees_from_live_object": orientation["yaw_delta_degrees"],
                "footprint_half_extents_region_m": orientation["footprint_half_extents_region_m"],
                "estimated_region_fit_cost": orientation_fit_cost,
                "geometry_candidate_only_not_safety_certified": True,
            })
    rows.sort(key=lambda row: (-float(row["ranking_score"]), float(row["travel_distance_m"]), row["candidate_id"]))
    if verb == "in":
        selected = []
        # Keep a high-ranked aperture pose for each yaw sector. Online IK then
        # orders this portfolio by measured live joint-limit margin before any
        # physical branch starts.
        for orientation in orientation_rows:
            yaw = float(orientation["yaw_delta_degrees"])
            matching = next((
                row for row in rows
                if abs(float(row.get("orientation_yaw_delta_degrees", float("inf"))) - yaw) < 1e-9
            ), None)
            if matching is not None:
                selected.append(matching)
                if len(selected) >= maximum_candidates:
                    break
        selected_ids = {id(row) for row in selected}
        selected.extend(
            row for row in rows
            if id(row) not in selected_ids and len(selected) < maximum_candidates
        )
    else:
        selected = rows[:maximum_candidates]
    # Placement layouts and their entrances form one bounded search portfolio.
    # Rotate the object/region-relative entrance around the nearest completed
    # object so a blocked route can try a perpendicular or opposite side.
    direction_angles = (0.0, 180.0, 90.0, -90.0)
    for index, row in enumerate(selected):
        base = region_rotation.T @ np.asarray(row["entry_direction_world"], dtype=np.float64)
        yaw_delta = float(row.get("orientation_yaw_delta_degrees", 0.0))
        entry_variant = (
            0.0 if verb == "in" and yaw_delta in (-15.0, 0.0, 15.0)
            else direction_angles[index % len(direction_angles)]
        )
        angle = np.deg2rad(entry_variant)
        c, s = float(np.cos(angle)), float(np.sin(angle))
        rotated = np.asarray([c * base[0] - s * base[1], s * base[0] + c * base[1], 0.0])
        row["entry_direction_world"] = (region_rotation @ rotated).tolist()
        row["entry_direction_variant_degrees"] = entry_variant
        row["route_strategy"] = "jointspace_rrt" if index % 2 == 0 else "elevated_osc_waypoints"
    return selected


def _rank_native_region_candidates_by_live_ik_margin(
    bridge,
    candidates: list[dict[str, Any]],
    *,
    target_binding: dict[str, Any],
    target_context: dict[str, Any],
    region: dict[str, Any],
    instruction: str,
    model_sha: str,
    safety_policy: dict[str, Any],
    route_variant: str,
    transit_clearance_mode: str,
    entry_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Soft-rank In poses using joint-limit margin across representative route points.

    A placement yaw can be reachable at the first waypoint and still drive the
    wrist into a hard stop later in the route. Probe the beginning, middle, and
    end of each generated route with the same live-state IK seeds. These
    read-only probes only change candidate order: incomplete IK coverage never
    removes a Stage A physical candidate, and MuJoCo plus native BDDL remain the
    success oracle.
    """
    from scipy.spatial.transform import Rotation

    if not candidates:
        return [], []
    observed = bridge.request({"op": "observe", "include_snapshot_state": True})
    snapshot_state = observed.get("snapshot_state", {})
    robot = snapshot_state.get("robot_states", [{}])[0]
    arm_qpos = np.asarray(robot.get("joint_qpos", []), dtype=np.float64)
    object_state = _object_state(snapshot_state, int(target_binding["body_id"]))
    object_world = np.eye(4, dtype=np.float64)
    object_world[:3, :3] = Rotation.from_quat(
        np.asarray(object_state["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
    ).as_matrix()
    object_world[:3, 3] = np.asarray(object_state["root_pos"], dtype=np.float64)
    eef_world = np.eye(4, dtype=np.float64)
    eef_world[:3, :3] = Rotation.from_quat(
        np.asarray(robot["eef_quat"], dtype=np.float64)[[1, 2, 3, 0]]
    ).as_matrix()
    eef_world[:3, 3] = np.asarray(robot["eef_pos"], dtype=np.float64)
    if (
        arm_qpos.shape != (7,) or not np.isfinite(arm_qpos).all()
        or not np.isfinite(object_world).all() or not np.isfinite(eef_world).all()
    ):
        return candidates, [{
            "status": "live_arm_or_grasp_pose_unavailable_candidates_retained",
            "physical_actions": 0,
            "safety_certified": False,
        }]
    start_state_sha = _state_digest(_integration_state(snapshot_state))
    inventory = bridge.request({"op": "robot_joint_limit_inventory"})
    if (
        inventory.get("status") != "complete"
        or inventory.get("physical_state_unchanged") is not True
    ):
        return candidates, [{
            "status": "live_joint_limit_inventory_unavailable_candidates_retained",
            "inventory_status": inventory.get("status"),
            "physical_state_unchanged": inventory.get("physical_state_unchanged"),
            "physical_actions": 0,
            "safety_certified": False,
        }]
    inventory_by_name = {
        str(row.get("joint_name")): row
        for row in inventory.get("joints", [])
        if isinstance(row, dict)
    }
    joint_names = [f"robot0_joint{index}" for index in range(1, 8)]
    limits = []
    for name in joint_names:
        row = inventory_by_name.get(name, {})
        lower, upper = row.get("lower"), row.get("upper")
        if (
            not isinstance(lower, (int, float)) or not isinstance(upper, (int, float))
            or not np.isfinite([lower, upper]).all() or float(upper) <= float(lower)
            or row.get("unit") != "rad"
        ):
            return candidates, [{
                "status": "live_joint_limit_inventory_incomplete_candidates_retained",
                "inventory_status": inventory.get("status"),
                "missing_joint": name,
                "physical_actions": 0,
                "safety_certified": False,
            }]
        limits.append((float(lower), float(upper)))
    seed_rows = [arm_qpos.copy()]
    for joint_index in (5, 6):
        lower, upper = limits[joint_index]
        for delta in (-0.70, 0.70):
            seed = arm_qpos.copy()
            seed[joint_index] = float(np.clip(
                seed[joint_index] + delta, lower + 1e-4, upper - 1e-4,
            ))
            if not any(np.array_equal(seed, prior) for prior in seed_rows):
                seed_rows.append(seed)

    diagnostics = []
    scored = []
    for ordinal, candidate in enumerate(candidates):
        desired_rotation = np.asarray(candidate.get("rotation_world"), dtype=np.float64)
        if desired_rotation.shape != (3, 3) or not np.isfinite(desired_rotation).all():
            diagnostics.append({
                "candidate_id": candidate.get("candidate_id"),
                "status": "candidate_rotation_invalid_physical_candidate_retained",
                "physical_actions": 0,
            })
            scored.append(((1, ordinal, 0.0, float("inf")), candidate))
            continue
        goal_object = np.eye(4, dtype=np.float64)
        goal_object[:3, :3] = desired_rotation
        goal_position = np.asarray(candidate.get("position_world_m"), dtype=np.float64)
        if goal_position.shape != (3,) or not np.isfinite(goal_position).all():
            diagnostics.append({
                "candidate_id": candidate.get("candidate_id"),
                "status": "candidate_position_invalid_physical_candidate_retained",
                "physical_actions": 0,
            })
            scored.append(((1, ordinal, 0.0, float("inf")), candidate))
            continue
        goal_object[:3, 3] = goal_position
        route_build_error = None
        route_waypoints: list[dict[str, Any]] = []
        try:
            region_rotation = np.asarray(region.get("site_rotation_world"), dtype=np.float64)
            route_entry_direction = np.asarray(
                candidate.get("entry_direction_world"), dtype=np.float64,
            )
            if route_entry_direction.shape != (3,) or not np.isfinite(route_entry_direction).all():
                route_entry_direction = None
            route = _native_region_progress_route(
                object_world,
                goal_object,
                region=region,
                route_variant=route_variant,
                transit_clearance_mode=transit_clearance_mode,
                entry_mode=entry_mode,
                entry_direction_world=route_entry_direction,
                entry_clearance_radius_m=_placement_footprint_radius(
                    target_context, region_rotation, desired_rotation,
                ),
                horizontal_entry_height_offset_m=(
                    0.0 if entry_mode == "horizontal_aperture" else None
                ),
            )
            route_waypoints = [
                row for row in route.get("waypoints", [])
                if isinstance(row, dict)
                and isinstance(row.get("object_pose_world"), list)
                and np.asarray(row["object_pose_world"], dtype=np.float64).shape == (4, 4)
                and np.isfinite(np.asarray(row["object_pose_world"], dtype=np.float64)).all()
                and np.linalg.norm(
                    np.asarray(row["object_pose_world"], dtype=np.float64)[:3, 3]
                    - object_world[:3, 3]
                ) > 0.015
            ]
        except (TypeError, ValueError, KeyError) as exc:
            route_build_error = f"{type(exc).__name__}: {exc}"
        sampled_waypoints: list[dict[str, Any]] = []
        if route_waypoints:
            for index in sorted(set((0, len(route_waypoints) // 2, len(route_waypoints) - 1))):
                sampled_waypoints.append(route_waypoints[index])
        if not sampled_waypoints:
            sampled_waypoints = [{
                "role": "native_goal_fallback",
                "object_pose_world": goal_object.tolist(),
            }]
        waypoint_results: list[dict[str, Any]] = []
        for waypoint in sampled_waypoints:
            desired_object = np.asarray(waypoint["object_pose_world"], dtype=np.float64)
            eef_target = _native_object_relative_eef_target_world(
                desired_object, object_world, eef_world,
            )
            reply = bridge.request({
                "op": "task_success_first_ik_preflight",
                "expected_instruction": instruction,
                "model_geometry_sha256": model_sha,
                "start_state_sha256": start_state_sha,
                "goal_pose_world": {
                    "position_world_m": eef_target[:3, 3].tolist(),
                    "quaternion_wxyz": Rotation.from_matrix(
                        eef_target[:3, :3],
                    ).as_quat()[[3, 0, 1, 2]].tolist(),
                },
                "ik_seed_qpos": [row.tolist() for row in seed_rows],
                "ik_selection_seed_qpos": arm_qpos.tolist(),
                "ik_solver_max_nfev": 80,
                "safety_geometry_policy": safety_policy,
            })
            if reply.get("live_state_unchanged") is not True:
                diagnostics.append({
                    "candidate_id": candidate.get("candidate_id"),
                    "yaw_delta_degrees": candidate.get("orientation_yaw_delta_degrees"),
                    "status": "live_state_changed_during_IK_ranking_candidates_retained",
                    "ik_status": reply.get("status"),
                    "probe_waypoint_role": waypoint.get("role"),
                    "physical_actions": 0,
                })
                return candidates, diagnostics
            valid_ik = []
            if reply.get("status") == "goal_ik_preflight_passed":
                for ik_row in reply.get("ik_candidates", []):
                    qpos = np.asarray(ik_row.get("qpos", []), dtype=np.float64)
                    if (
                        qpos.shape != (7,) or not np.isfinite(qpos).all()
                        or float(ik_row.get("position_error_m", float("inf"))) > 0.003
                        or float(ik_row.get("orientation_error_rad", float("inf"))) > 0.03
                    ):
                        continue
                    normalized_margins = [
                        min(qpos[index] - low, high - qpos[index]) / (high - low)
                        for index, (low, high) in enumerate(limits)
                    ]
                    minimum_margin = float(min(normalized_margins))
                    travel = float(np.linalg.norm(qpos - arm_qpos))
                    valid_ik.append({
                        "minimum_normalized_joint_limit_margin": minimum_margin,
                        "joint6_margin_rad": float(min(qpos[5] - limits[5][0], limits[5][1] - qpos[5])),
                        "joint7_margin_rad": float(min(qpos[6] - limits[6][0], limits[6][1] - qpos[6])),
                        "joint_travel_l2_rad": travel,
                        "score": minimum_margin - 0.005 * travel,
                        "seed_index": ik_row.get("seed_index"),
                    })
            if valid_ik:
                best = max(valid_ik, key=lambda row: (row["score"], -row["joint_travel_l2_rad"]))
                waypoint_results.append({
                    "role": waypoint.get("role"),
                    "status": "IK_passed",
                    **best,
                    "evaluated_IK_solutions": len(valid_ik),
                })
            else:
                waypoint_results.append({
                    "role": waypoint.get("role"),
                    "status": "IK_not_verified_candidate_retained",
                    "ik_status": reply.get("status"),
                    "failure_reason": reply.get("failure_reason"),
                    "live_state_unchanged": reply.get("live_state_unchanged"),
                })
        verified = [row for row in waypoint_results if row.get("status") == "IK_passed"]
        full_route_verified = len(verified) == len(waypoint_results)
        if verified:
            worst_margin = min(
                float(row["minimum_normalized_joint_limit_margin"]) for row in verified
            )
            worst_joint7_margin = min(float(row["joint7_margin_rad"]) for row in verified)
            total_travel = sum(float(row["joint_travel_l2_rad"]) for row in verified)
            rank_score = worst_margin - 0.001 * total_travel
            diagnostic = {
                "candidate_id": candidate.get("candidate_id"),
                "yaw_delta_degrees": candidate.get("orientation_yaw_delta_degrees"),
                "status": (
                    "representative_route_waypoint_IK_passed_soft_joint_margin_ranked"
                    if full_route_verified else
                    "partial_route_waypoint_IK_coverage_candidate_retained"
                ),
                "rank_basis": "representative_start_middle_goal_route_waypoints",
                "sampled_waypoint_count": len(waypoint_results),
                "verified_waypoint_count": len(verified),
                "waypoint_results": waypoint_results,
                "minimum_normalized_joint_limit_margin_across_route": worst_margin,
                "minimum_joint7_margin_rad_across_route": worst_joint7_margin,
                "aggregate_joint_travel_l2_rad": total_travel,
                "score": rank_score,
                "physical_actions": 0,
                "candidate_retained_for_physical_search": True,
            }
            if route_build_error is not None:
                diagnostic["route_build_error_fallback_to_native_goal"] = route_build_error
            candidate["live_endpoint_ik_joint_margin"] = diagnostic
            diagnostics.append(diagnostic)
            scored.append((
                (0 if full_route_verified else 1, -rank_score, total_travel, ordinal),
                candidate,
            ))
        else:
            diagnostic = {
                "candidate_id": candidate.get("candidate_id"),
                "yaw_delta_degrees": candidate.get("orientation_yaw_delta_degrees"),
                "status": "route_waypoint_IK_not_verified_physical_candidate_retained",
                "rank_basis": "representative_start_middle_goal_route_waypoints",
                "sampled_waypoint_count": len(waypoint_results),
                "verified_waypoint_count": 0,
                "waypoint_results": waypoint_results,
                "route_build_error_fallback_to_native_goal": route_build_error,
                "physical_actions": 0,
                "candidate_retained_for_physical_search": True,
            }
            candidate["live_endpoint_ik_joint_margin"] = diagnostic
            diagnostics.append(diagnostic)
            scored.append(((2, ordinal, 0.0, float("inf")), candidate))
    scored.sort(key=lambda row: row[0])
    return [row[1] for row in scored], diagnostics


def _student_segment(
    bridge,
    student,
    *,
    max_actions: int,
    target_binding: dict[str, Any],
    instruction: str,
    policy: dict[str, Any],
    output_dir: Path,
    segment: str,
) -> dict[str, Any]:
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _record_step,
        _write_recording,
    )
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor

    records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    monitor = GraspStabilityMonitor(required_samples=3)
    data = bridge.request({"op": "observe", "include_snapshot_state": True})
    initial_eef = np.asarray(data["snapshot_state"]["robot_states"][0]["eef_pos"], dtype=np.float64)
    initial_object = np.asarray(
        _object_state(data["snapshot_state"], int(target_binding["body_id"]))["root_pos"],
        dtype=np.float64,
    )
    native = bridge.request({"op": "native_task_status"})
    _, success = _task_values(native)
    prediction: np.ndarray | None = None
    cursor = 0
    environment_done = False
    failure_trigger = None
    target_distance_history: list[float] = []
    object_displacement_history: list[float] = []
    for action_index in range(max_actions):
        if success:
            break
        if prediction is None or cursor >= len(prediction):
            prediction = student.predict(data)
            cursor = 0
        action = prediction[cursor]
        cursor += 1
        try:
            _record_step(
                bridge, action, f"student_{segment}", f"student_{segment}", action_index,
                target_binding, instruction, policy, monitor, records, frames,
            )
        except RuntimeError as exc:
            # _record_step archives the executed transition before reporting
            # done. A non-success terminal state is a Student skill failure;
            # the caller may restore its branch Snapshot and hand off.
            if "environment_done_during_grasp_only_branch" not in str(exc):
                raise
            environment_done = True
        data = bridge.request({"op": "observe", "include_snapshot_state": True})
        snapshot_state = data["snapshot_state"]
        target_position = np.asarray(
            _object_state(snapshot_state, int(target_binding["body_id"]))["root_pos"],
            dtype=np.float64,
        )
        eef_position = np.asarray(snapshot_state["robot_states"][0]["eef_pos"], dtype=np.float64)
        target_distance_history.append(float(np.linalg.norm(eef_position - target_position)))
        object_displacement_history.append(float(np.linalg.norm(target_position - initial_object)))
        native = bridge.request({"op": "native_task_status"})
        _, success = _task_values(native)
        if success or environment_done:
            break
        if len(target_distance_history) >= 12 and not monitor.stable:
            start_distance = target_distance_history[-12]
            best_distance = min(target_distance_history[-12:])
            object_progress = max(object_displacement_history[-12:]) - min(object_displacement_history[-12:])
            if start_distance - best_distance < 0.006 and object_progress < 0.002:
                failure_trigger = "student_goal_approach_and_object_progress_stagnated_for_12_actions"
                break
    _write_recording(output_dir, records, frames)
    final_snapshot = bridge.request({"op": "export_runtime_snapshot"})
    if final_snapshot.get("status") != "complete" or not isinstance(final_snapshot.get("payload"), bytes):
        raise RuntimeError(f"student_{segment}_terminal_snapshot_export_failed")
    path = output_dir / f"student_{segment}_terminal_runtime_snapshot.pkl"
    path.write_bytes(final_snapshot["payload"])
    final = final_snapshot["snapshot_state"]
    final_eef = np.asarray(final["robot_states"][0]["eef_pos"], dtype=np.float64)
    final_obj = np.asarray(
        _object_state(final, int(target_binding["body_id"]))["root_pos"], dtype=np.float64
    )
    return {
        "segment": segment,
        "policy_source": "original_pi05_rlinf_libero_sft_base_only_no_LoRA",
        "actions_executed": len(records),
        "task_success": bool(success),
        "native_task_status": native,
        "environment_done_without_native_success": bool(environment_done and not success),
        "failure_trigger": failure_trigger,
        "stagnation_window_actions": 12,
        "target_displacement_m": float(np.linalg.norm(final_obj - initial_object)),
        "eef_displacement_m": float(np.linalg.norm(final_eef - initial_eef)),
        "grasp_stability": monitor.diagnostics(),
        "terminal_snapshot": str(path),
        "terminal_snapshot_sha256": _sha(path.read_bytes()),
        "artifacts": {
            "records": str(output_dir / "physical_step_records.jsonl.gz"),
            "npz": str(output_dir / "task_success_first_actions_states.npz"),
            "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
        },
    }


def _stage_a_unheld_in_freefall_probe(
    bridge,
    *,
    target_id: str,
    target_binding: dict[str, Any],
    predicate: list[str],
    instruction: str,
    model_sha: str,
    safety_policy: dict[str, Any],
    controller_fields: dict[str, Any],
    output_dir: Path,
    maximum_actions: int,
    allow_near_aperture_release: bool = False,
) -> dict[str, Any]:
    """Run a short physical release probe near an active native In region.

    This is a bounded Stage A release probe near an open container. The default
    mode requires a live footprint projection directly above the opening.
    Explicit near-aperture mode accepts an object whose compiled footprint is
    within 15 cm of the native region and opens either a verified or partial
    grasp for a real MuJoCo settle. The native BDDL predicate decides whether
    passive settling completed the task.
    """
    from scipy.spatial.transform import Rotation
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _record_step,
        _write_recording,
    )
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor
    from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action

    output_dir.mkdir(parents=True, exist_ok=True)
    initial_native = bridge.request({"op": "native_task_status"})
    initial_values, initial_success = _task_values(initial_native)
    if (
        predicate[0].lower() != "in"
        or initial_success
        or initial_values.get(tuple(predicate)) is not False
    ):
        return {
            "status": "not_eligible_native_goal_or_relation_changed",
            "initial_native_task_status": initial_native,
            "physical_actions": 0,
            "direct_state_write": False,
            "safety_certified": False,
        }
    before = bridge.request({"op": "observe", "include_snapshot_state": True})
    before_state = before.get("snapshot_state", {})
    before_digest = _state_digest(_integration_state(before_state))
    grasp = bridge.request({
        "op": "grasp_state",
        "carried_binding": target_binding,
        "expected_instruction": instruction,
    })
    fingers = grasp.get("finger_contacts", {})
    bilateral_grasp_remains = bool(
        grasp.get("grasp_confirmed") is True
        or (fingers.get("left") is True and fingers.get("right") is True)
    )
    if bilateral_grasp_remains and not allow_near_aperture_release:
        return {
            "status": "not_eligible_bilateral_grasp_remains",
            "initial_grasp_state": grasp,
            "physical_actions": 0,
            "direct_state_write": False,
            "safety_certified": False,
        }
    goal_query = bridge.request({"op": "native_goal_query", "predicate": predicate})
    target_context = bridge.request({
        "op": "task_success_first_context",
        "expected_instruction": instruction,
        "target_predicate": predicate,
    })
    after = bridge.request({"op": "observe", "include_snapshot_state": True})
    after_state = after.get("snapshot_state", {})
    if (
        _state_digest(_integration_state(after_state)) != before_digest
        or goal_query.get("status") != "complete"
        or goal_query.get("predicate_is_native_goal") is not True
        or target_context.get("status") != "complete"
        or target_context.get("model_geometry_sha256") != model_sha
    ):
        return {
            "status": "not_eligible_read_only_region_preflight_unavailable_or_state_changed",
            "native_goal_query_status": goal_query.get("status"),
            "target_context_status": target_context.get("status"),
            "physical_state_unchanged": _state_digest(_integration_state(after_state)) == before_digest,
            "physical_actions": 0,
            "direct_state_write": False,
            "safety_certified": False,
        }
    region = goal_query.get("region", {})
    robot = after_state.get("robot_states", [{}])[0]
    try:
        region_position = np.asarray(region.get("site_position_world_m"), dtype=np.float64)
        region_rotation = np.asarray(region.get("site_rotation_world"), dtype=np.float64)
        region_half_extents = np.asarray(region.get("site_size_m"), dtype=np.float64)
        root_quaternion = np.asarray(
            _object_state(after_state, int(target_binding["body_id"]))["root_quat"],
            dtype=np.float64,
        )
        object_position = np.asarray(
            _object_state(after_state, int(target_binding["body_id"]))["root_pos"],
            dtype=np.float64,
        )
        object_rotation = Rotation.from_quat(root_quaternion[[1, 2, 3, 0]]).as_matrix()
        object_local = region_rotation.T @ (object_position - region_position)
        collision_lower, collision_upper = _placement_collision_bounds_in_region(
            target_context, region_rotation, object_rotation,
        )
        footprint_lower = object_local + collision_lower
        footprint_upper = object_local + collision_upper
        bottom_clearance = float(footprint_lower[2] - region_half_extents[2])
        horizontal_inside = bool(
            np.all(footprint_lower[:2] >= -region_half_extents[:2] - 0.005)
            and np.all(footprint_upper[:2] <= region_half_extents[:2] + 0.005)
        )
        xy_gap = np.maximum(
            np.abs(object_local[:2]) - region_half_extents[:2]
            - 0.5 * (footprint_upper[:2] - footprint_lower[:2]),
            0.0,
        )
        distance_to_native_region_xy_m = float(np.linalg.norm(xy_gap))
        near_aperture = bool(
            distance_to_native_region_xy_m <= 0.15
            and abs(float(object_local[2])) <= float(region_half_extents[2]) + 0.25
        )
        vertically_above_opening = bool(
            region_rotation.shape == (3, 3)
            and abs(float(region_rotation[2, 2])) >= 0.50
            and 0.0 < bottom_clearance <= 0.35
        )
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return {
            "status": "not_eligible_live_object_or_region_geometry_unavailable",
            "preflight_error": f"{type(exc).__name__}: {exc}",
            "physical_actions": 0,
            "direct_state_write": False,
            "safety_certified": False,
        }
    geometry_gate = {
        "object_position_world_m": object_position.tolist(),
        "object_position_region_local_m": object_local.tolist(),
        "object_collision_bounds_region_m": [
            footprint_lower.tolist(), footprint_upper.tolist(),
        ],
        "region_half_extents_m": region_half_extents.tolist(),
        "object_footprint_inside_opening": horizontal_inside,
        "distance_to_native_region_xy_m": distance_to_native_region_xy_m,
        "near_aperture_release_requested": bool(allow_near_aperture_release),
        "object_within_near_aperture_release_distance": near_aperture,
        "object_bottom_clearance_above_region_top_m": bottom_clearance,
        "region_vertical_axis_projection": float(abs(region_rotation[2, 2])),
        "single_sided_pad_contact_at_start": bool(
            fingers.get("left") is True or fingers.get("right") is True
        ),
        "bilateral_grasp_at_start": bilateral_grasp_remains,
        "explicit_near_aperture_mode_allows_physical_gripper_open": bool(
            allow_near_aperture_release
        ),
        "eligibility": (
            "eligible_above_opening" if horizontal_inside and vertically_above_opening
            else "eligible_near_aperture" if allow_near_aperture_release and near_aperture
            else "outside_opening_or_not_overhead"
        ),
        "read_only_physical_state_sha256": before_digest,
    }
    default_release_eligible = horizontal_inside and vertically_above_opening
    near_aperture_release_eligible = (
        allow_near_aperture_release and near_aperture
    )
    if not default_release_eligible and not near_aperture_release_eligible:
        return {
            "status": "not_eligible_object_not_fitting_above_native_opening",
            "geometry_gate": geometry_gate,
            "physical_actions": 0,
            "direct_state_write": False,
            "native_BDDL_success_required": True,
            "safety_certified": False,
        }

    monitor = GraspStabilityMonitor(required_samples=3)
    records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    native = initial_native
    action_index = 0
    for action_index in range(maximum_actions):
        observed = bridge.request({"op": "observe", "include_snapshot_state": True})
        live = observed.get("snapshot_state", {})
        live_robot = live.get("robot_states", [{}])[0]
        eef_position = np.asarray(live_robot.get("eef_pos"), dtype=np.float64)
        eef_quaternion = np.asarray(live_robot.get("eef_quat"), dtype=np.float64)
        action = world_pose_tracking_action(
            eef_position, eef_quaternion, eef_position, eef_quaternion,
            controller_fields, gripper_action=-1.0,
        )
        _record_step(
            bridge,
            action,
            "stage_a_unheld_in_region_freefall_settle",
            f"{target_id}_unheld_native_In_freefall_settle",
            action_index,
            target_binding,
            instruction,
            safety_policy,
            monitor,
            records,
            frames,
            raise_on_done=False,
            planning_tracking={
                "tracking_mode": "zero_arm_motion_physical_open_gripper_settle_probe",
                "native_goal_predicate": predicate,
                "object_fall_is_passive_MuJoCo_physics": True,
                "direct_state_write": False,
                "safety_certified": False,
            },
        )
        native = bridge.request({"op": "native_task_status"})
        _, success = _task_values(native)
        if success:
            action_index += 1
            break
        if records[-1].get("done"):
            action_index += 1
            break
    recording = _write_recording(output_dir, records, frames)
    terminal = bridge.request({"op": "export_runtime_snapshot"})
    snapshot_path = output_dir / "unheld_in_region_settle_terminal_runtime_snapshot.pkl"
    snapshot_sha = None
    state_sha = None
    if terminal.get("status") == "complete" and isinstance(terminal.get("payload"), bytes):
        snapshot_path.write_bytes(terminal["payload"])
        snapshot_sha = _sha(terminal["payload"])
        state_sha = _state_digest(_integration_state(terminal.get("snapshot_state", {})))
    final_native = bridge.request({"op": "native_task_status"})
    _, success = _task_values(final_native)
    return {
        "status": "complete_task_success" if success else "freefall_settle_window_expired",
        "geometry_gate": geometry_gate,
        "initial_native_task_status": initial_native,
        "initial_grasp_state": grasp,
        "physical_actions": len(records),
        "first_action_index": records[0].get("timestep") if records else None,
        "final_native_task_status": final_native,
        "native_task_success": success,
        "release_probe_mode": (
            "near_aperture_physical_release"
            if near_aperture_release_eligible else "overhead_freefall_settle"
        ),
        "recording": recording,
        "terminal_snapshot": str(snapshot_path) if snapshot_sha else None,
        "terminal_snapshot_sha256": snapshot_sha,
        "terminal_integration_state_sha256": state_sha,
        "all_actions_real_MuJoCo_steps": True,
        "all_contact_substeps_recorded": all(
            row.get("substep_contact_monitoring_complete") is True for row in records
        ),
        "direct_state_write": False,
        "native_BDDL_success_required": True,
        "safety_certified": False,
    }


def _run_articulated_fixture_goal(
    bridge,
    *,
    predicate: list[str],
    target_binding: dict[str, Any],
    instruction: str,
    policy: dict[str, Any],
    articulation_context: dict[str, Any],
    controller_fields: dict[str, Any],
    output_dir: Path,
    model_sha: str,
    rrt_time_budget_s: float = NATIVE_PLACEMENT_RRT_BUDGET_S,
) -> dict[str, Any]:
    """Physically push a native articulated fixture goal through OSC_POSE."""
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _advance_pose,
        _snapshot_state_vector,
        _state_sha,
        _write_recording,
    )
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor

    if articulation_context.get("status") != "complete" or not articulation_context.get("read_only"):
        raise RuntimeError(f"articulated_fixture_context_unavailable:{articulation_context.get('error')}")
    path = np.asarray(articulation_context.get("contact_point_path_world_m"), dtype=np.float64)
    outward = np.asarray(articulation_context.get("outward_direction_world"), dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or outward.shape != (3,):
        raise ValueError("articulation_context_does_not_contain_a_finite_physical_contact_path")
    records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    monitor = GraspStabilityMonitor(required_samples=3)
    observed = bridge.request({"op": "observe", "include_snapshot_state": True})
    start_state_sha = _state_sha(_snapshot_state_vector(observed.get("snapshot_state", {})))
    current = observed["snapshot_state"]["robot_states"][0]
    joint = articulation_context.get("joint", {})
    joint_type = str(joint.get("joint_type", "")).lower()
    quaternion = (
        _articulated_fixture_slide_grasp_quaternion(outward)
        if joint_type == "slide"
        else _articulated_fixture_press_quaternion(
            np.asarray(current["eef_quat"], dtype=np.float64), outward,
        )
    )
    _advance_pose.controller_fields = controller_fields
    counter = [0]
    contact_state: dict[str, Any] = {
        "active": False,
        "last_action_has_fixture_contact": False,
        "first_action_index": None,
        "contact_pairs": [],
        "latest_context": articulation_context,
        "anchor_eef_position_world_m": None,
        "anchor_eef_quaternion_wxyz": None,
        "anchor_joint_position": None,
        "anchor_joint_axis_world": None,
        "anchor_joint_origin_world_m": None,
        "follow_chunks": [],
        "switch_bilateral_contact_observed": False,
        "switch_gripper_close_commanded": False,
        "fixture_gripper_close_commanded": False,
    }
    contact_phase_start_action: int | None = None
    precontact = path[0] + outward * (
        ARTICULATED_FIXTURE_SLIDE_PRECONTACT_OUTWARD_M if joint_type == "slide"
        else ARTICULATED_FIXTURE_PRECONTACT_APPROACH_DISTANCE_M
    )
    if joint_type == "slide":
        precontact = precontact + np.asarray(
            [0.0, 0.0, ARTICULATED_FIXTURE_SLIDE_PRECONTACT_HEIGHT_M], dtype=np.float64,
        )
        slide_clearance_waypoints = list(
            _articulated_fixture_slide_approach_waypoints(path[0], outward)
        )
        slide_clearance_phase_names = [
            "slide_clearance_descent_outside_cabinet",
            "slide_under_top_inward_approach",
            "slide_handle_height_alignment",
        ]
        contact_offset = outward * ARTICULATED_FIXTURE_SLIDE_CONTACT_OUTWARD_OFFSET_M
        contact_height_offset = np.asarray(
            [0.0, 0.0, ARTICULATED_FIXTURE_SLIDE_HANDLE_EEF_HEIGHT_OFFSET_M],
            dtype=np.float64,
        )
    else:
        slide_clearance_waypoints = []
        slide_clearance_phase_names = []
        contact_offset = outward * 0.035
        contact_height_offset = np.zeros(3, dtype=np.float64)
    contact_waypoints = [
        point + contact_offset + contact_height_offset for point in path
    ]
    failure_reason = None
    failed_waypoint_index = None
    route_tracking: dict[str, Any] = {
        "status": "not_started",
        "live_state_unchanged": None,
        "route_source": None,
        "fixture_target_clearance_m": ARTICULATED_FIXTURE_PRECONTACT_CLEARANCE_M,
        "approach_waypoints_world_m": [
            precontact.tolist(),
            *[point.tolist() for point in slide_clearance_waypoints],
        ],
        "plan_goal_role": (
            "full_geometry_high_front_on_precontact_then_enter_below_cabinet_top_for_slide_handle_contact"
            if joint_type == "slide"
            else "full_geometry_low_precontact_before_fixture_contact"
        ),
        "press_orientation_quaternion_wxyz": quaternion.tolist(),
        "press_orientation_tool_axis_world": None,
        "press_orientation_finger_closing_axis_world": None,
        "raw_pose_count": 0,
        "tracked_pose_count": 0,
        "eef_reference_poses_world": [],
    }

    def record_articulation_feedback(record, _frame):
        nonlocal contact_phase_start_action
        feedback = bridge.request({
            "op": "task_success_first_articulation_context",
            "expected_instruction": instruction,
            "predicate": predicate,
        })
        if isinstance(feedback, dict) and feedback.get("status") == "complete":
            contact_state["latest_context"] = feedback
        contact_pairs = _articulated_fixture_robot_contact_pairs(
            record, str(articulation_context.get("fixture_id_from_active_BDDL", "")),
            list(feedback.get("articulated_moving_collision_geom_names", []))
            if isinstance(feedback, dict) else None,
        )
        current_joint = feedback.get("joint", {}) if isinstance(feedback, dict) else {}
        if contact_pairs:
            if not contact_state["active"]:
                contact_state["first_action_index"] = counter[0]
                contact_phase_start_action = counter[0]
                robot = record.get("snapshot_state", {}).get("robot_states", [{}])[0]
                contact_state["anchor_eef_position_world_m"] = robot.get("eef_pos")
                contact_state["anchor_eef_quaternion_wxyz"] = robot.get("eef_quat")
                contact_state["anchor_joint_position"] = current_joint.get("position")
                contact_state["anchor_joint_axis_world"] = feedback.get("joint_axis_world")
                contact_state["anchor_joint_origin_world_m"] = feedback.get("joint_origin_world_m")
            contact_state["active"] = True
            contact_state["contact_pairs"] = contact_pairs
        contact_state["last_action_has_fixture_contact"] = bool(contact_pairs)
        contacting_fingers = {
            pair.get("robot_geom", "") for pair in contact_pairs
            if "finger1" in pair.get("robot_geom", "")
            or "finger2" in pair.get("robot_geom", "")
        }
        finger_sides = {
            "finger1" if "finger1" in name else "finger2"
            for name in contacting_fingers
        }
        if {"finger1", "finger2"}.issubset(finger_sides):
            contact_state["switch_bilateral_contact_observed"] = True
        record["articulation_tracking"] = {
            "joint_name": current_joint.get("joint_name", joint.get("joint_name")),
            "joint_position_after": current_joint.get("position"),
            "joint_target_position": current_joint.get("target_position", joint.get("target_position")),
            "native_goal_value_after": feedback.get("native_goal_value") if isinstance(feedback, dict) else None,
            "joint_axis_world": feedback.get("joint_axis_world") if isinstance(feedback, dict) else None,
            "joint_origin_world_m": feedback.get("joint_origin_world_m") if isinstance(feedback, dict) else None,
            "robot_fixture_contact": bool(contact_pairs),
            "robot_fixture_contact_pairs": contact_pairs,
            "contact_anchor_eef_position_world_m": contact_state.get("anchor_eef_position_world_m"),
            "contact_anchor_joint_position": contact_state.get("anchor_joint_position"),
            "fixture_follow_lead": record.get("planning_tracking", {}).get("fixture_follow_lead"),
            "eef_target_position_world_m": record.get("planning_tracking", {}).get(
                "active_subgoal_eef_pose_world", [None, None, None],
            )[:3],
        }

    # Plan the complete route to the slide-specific elevated top-down approach
    # or the established low precontact pose used for hinged fixtures.
    goal_pose = {
        "position_world_m": precontact.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
    }
    planner_config = _keypose_rrt_planner_config(
        seed=74139,
        search_time_budget_s=float(rrt_time_budget_s),
        maximum_iterations=1200,
        maximum_config_checks=12000,
    )
    # This route is executed directly around movable scene geometry. Require
    # every sampled configuration to pass the active per-object policy before
    # allowing OSC to follow it. This is a route feasibility gate, not a safety
    # certification of the resulting physical episode.
    planner_config["require_safe_configurations"] = True
    planner_config["minimum_clearance_m"] = 0.0
    route_policy = _articulated_fixture_rrt_policy(
        policy, str(articulation_context.get("fixture_id_from_active_BDDL")),
    )
    route_forbidden_thresholds = [
        float(item.get("minimum_clearance_m", 0.0))
        for item in route_policy.get("forbidden_objects", [])
        if isinstance(item, dict)
    ]
    reported_clearance_threshold_m = min(route_forbidden_thresholds, default=0.0)
    from scipy.spatial.transform import Rotation

    press_rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    route_tracking["press_orientation_tool_axis_world"] = (
        press_rotation.as_matrix()[:, 2].tolist()
    )
    route_tracking["press_orientation_finger_closing_axis_world"] = (
        press_rotation.as_matrix()[:, 0].tolist()
    )
    try:
        planned = bridge.request({
            "op": "task_success_first_rrt",
            "expected_instruction": instruction,
            "model_geometry_sha256": model_sha,
            "start_state_sha256": start_state_sha,
            "goal_pose_world": goal_pose,
            "safety_geometry_policy": route_policy,
            "planner_config": planner_config,
            "ik_seed_qpos": _placement_ik_seed_portfolio(current),
        })
        after_plan = bridge.request({"op": "observe", "include_snapshot_state": True})
        after_plan_sha = _state_sha(_snapshot_state_vector(after_plan.get("snapshot_state", {})))
        unchanged = after_plan_sha == start_state_sha
        route_tracking.update({
            "status": planned.get("status"),
            "failure_reason": planned.get("failure_reason"),
            "live_state_unchanged": unchanged,
            "reported_live_state_unchanged": planned.get("live_state_unchanged"),
            "planner_source": planned.get("planner_source"),
            "planning_seconds": planned.get("planning_seconds"),
            "config_checks": planned.get("config_checks"),
            "goal_ik_configuration_checks": planned.get("goal_ik_configuration_checks"),
            "selected_ik_seed_index": planned.get("selected_ik_seed_index"),
            "ik_candidates": planned.get("ik_candidates", []),
            "route_candidates": planned.get("route_candidates", []),
            "minimum_forbidden_clearance_m": planned.get(
                "minimum_forbidden_clearance_m"
            ),
            "minimum_forbidden_clearance_required_m": reported_clearance_threshold_m,
            "require_safe_configurations": planned.get("require_safe_configurations"),
            "planner_minimum_clearance_m": planned.get("planner_minimum_clearance_m"),
            "start_state_sha256": start_state_sha,
            "goal_pose_world": goal_pose,
        })
        if not unchanged:
            raise RuntimeError("articulated_fixture_RRT_changed_live_integration_state")
        if planned.get("status") != "offline_feasible" or planned.get("live_state_unchanged") is not True:
            raise RuntimeError(
                "articulated_fixture_precontact_RRT_not_feasible:"
                f"{planned.get('status')}:{planned.get('failure_reason')}"
            )
        if planned.get("require_safe_configurations") is not True:
            raise RuntimeError("articulated_fixture_precontact_RRT_safe_configuration_gate_not_applied")
        minimum_forbidden_clearance = planned.get("minimum_forbidden_clearance_m")
        if (
            isinstance(minimum_forbidden_clearance, bool)
            or not isinstance(minimum_forbidden_clearance, (int, float))
            or not np.isfinite(float(minimum_forbidden_clearance))
            or float(minimum_forbidden_clearance) < reported_clearance_threshold_m
        ):
            raise RuntimeError(
                "articulated_fixture_precontact_RRT_forbidden_clearance_below_policy:"
                f"{minimum_forbidden_clearance}:{reported_clearance_threshold_m}"
            )
        planned_route = planned.get("eef_reference_poses_world")
        raw_route = np.asarray(planned_route, dtype=np.float64)
        if raw_route.ndim != 2 or raw_route.shape[1:] != (7,) or len(raw_route) < 2 or not np.isfinite(raw_route).all():
            raise RuntimeError("articulated_fixture_precontact_RRT_returned_invalid_EEF_path")
        raw_qpos_path = np.asarray(planned.get("qpos_path"), dtype=np.float64)
        if (
            raw_qpos_path.ndim != 2
            or raw_qpos_path.shape[1:] != (7,)
            or len(raw_qpos_path) != len(raw_route)
            or not np.isfinite(raw_qpos_path).all()
        ):
            raise RuntimeError("articulated_fixture_precontact_RRT_returned_invalid_qpos_path")
        route, route_raw_indices = _thin_articulated_fixture_rrt_route(raw_route, stride=2)
        route_tracking.update({
            "raw_pose_count": int(len(raw_route)),
            "tracked_pose_count": int(len(route)),
            "route_sampling_stride": 2,
            "tracked_pose_raw_indices": route_raw_indices,
            "qpos_path_rad": raw_qpos_path.tolist(),
            "eef_reference_poses_world": route.tolist(),
            "route_source": "MuJoCo_7D_RRT_full_geometry_EEF_path_tracked_by_native_OSC",
        })
        precontact_waypoints = route[1:]
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        failed_waypoint_index = -1
        precontact_waypoints = []

    approach_route_waypoint_count = len(precontact_waypoints) + len(slide_clearance_waypoints)
    waypoints = [*precontact_waypoints, *slide_clearance_waypoints, *contact_waypoints]
    for waypoint_index, waypoint in enumerate(waypoints):
        if failure_reason is not None:
            break
        if contact_state["active"]:
            route_tracking["contact_transition"] = {
                "first_contact_action_index": contact_state["first_action_index"],
                "route_waypoint_index": waypoint_index,
                "contact_pairs": contact_state["contact_pairs"],
                "joint_position_at_transition": contact_state["latest_context"].get("joint", {}).get("position"),
            }
            break
        is_rrt_waypoint = waypoint_index < len(precontact_waypoints)
        is_slide_clearance_waypoint = (
            len(precontact_waypoints) <= waypoint_index < approach_route_waypoint_count
        )
        is_approach_waypoint = waypoint_index < approach_route_waypoint_count
        if not is_approach_waypoint and contact_phase_start_action is None:
            contact_phase_start_action = counter[0]
        if is_rrt_waypoint:
            waypoint_phase = "RRT_low_precontact"
            eef_outward_offset_m = None
            contact_point_index = 0
        elif is_slide_clearance_waypoint:
            clearance_waypoint_index = waypoint_index - len(precontact_waypoints)
            waypoint_phase = slide_clearance_phase_names[clearance_waypoint_index]
            eef_outward_offset_m = (
                ARTICULATED_FIXTURE_SLIDE_PRECONTACT_OUTWARD_M
                if clearance_waypoint_index == 0
                else ARTICULATED_FIXTURE_SLIDE_HANDLE_APPROACH_OUTWARD_M
            )
            contact_point_index = 0
        else:
            waypoint_phase = "fixture_contact"
            eef_outward_offset_m = (
                ARTICULATED_FIXTURE_SLIDE_CONTACT_OUTWARD_OFFSET_M
                if joint_type == "slide" else 0.035
            )
            contact_point_index = waypoint_index - approach_route_waypoint_count
        target_position = np.asarray(waypoint[:3] if is_rrt_waypoint else waypoint, dtype=np.float64)
        target_quaternion = (
            np.asarray(waypoint[3:], dtype=np.float64) if is_rrt_waypoint else quaternion
        )
        distance_to_articulation_path = float(np.min(np.linalg.norm(path - target_position, axis=1)))
        position_tolerance_m = (
            ARTICULATED_FIXTURE_CONTACT_POSITION_TOLERANCE_M
            if is_rrt_waypoint
            and distance_to_articulation_path <= ARTICULATED_FIXTURE_CONTACT_TRACKING_REGION_M
            else ARTICULATED_FIXTURE_ROUTE_POSITION_TOLERANCE_M
        )
        tracking = {
            "tracking_mode": (
                "collision_aware_RRT_high_front_on_slide_precontact_then_under_top_handle_entry_OSC"
                if joint_type == "slide"
                else "collision_aware_RRT_low_precontact_then_outward_fixture_approach_OSC"
            ),
            "native_goal_predicate": list(predicate),
            "fixture_id": articulation_context.get("fixture_id_from_active_BDDL"),
            "joint_name": joint.get("joint_name"),
            "joint_position_at_plan_start": joint.get("position"),
            "joint_target_position": joint.get("target_position"),
            "waypoint_index": waypoint_index,
            "waypoint_count": len(waypoints),
            "waypoint_phase": waypoint_phase,
            "contact_point_world_m": path[min(max(contact_point_index, 0), len(path) - 1)].tolist(),
            "outward_direction_world": outward.tolist(),
            "eef_outward_offset_m": eef_outward_offset_m,
            "eef_position_tolerance_m": position_tolerance_m,
        }

        def route_target_provider(_snapshot_state):
            if not contact_state["active"]:
                return target_position, target_quaternion
            feedback = contact_state["latest_context"]
            anchored_feedback = {
                **feedback,
                "joint_axis_world": contact_state["anchor_joint_axis_world"],
                "joint_origin_world_m": contact_state["anchor_joint_origin_world_m"],
            }
            target = _articulated_fixture_follow_pose(
                np.asarray(contact_state["anchor_eef_position_world_m"], dtype=np.float64),
                np.asarray(contact_state["anchor_eef_quaternion_wxyz"], dtype=np.float64),
                anchored_feedback,
                joint_position_at_anchor=float(contact_state["anchor_joint_position"]),
                allow_terminal_stop_follow=(str(predicate[0]).lower() == "turnoff"),
            )
            tracking["tracking_mode"] = "live_articulated_fixture_follow_after_contact"
            tracking["fixture_joint_position_at_feedback"] = (
                feedback.get("joint", {}).get("position")
            )
            tracking["fixture_contact_anchor_joint_position"] = contact_state["anchor_joint_position"]
            return target

        try:
            remaining_actions = (
                ARTICULATED_FIXTURE_PRECONTACT_ACTION_BUDGET - counter[0] if is_approach_waypoint
                else ARTICULATED_FIXTURE_CONTACT_ACTION_BUDGET
                - (counter[0] - int(contact_phase_start_action or 0))
            )
            if not is_approach_waypoint:
                remaining_actions = min(
                    remaining_actions, ARTICULATED_FIXTURE_TOTAL_ACTION_BUDGET - counter[0],
                )
            if remaining_actions <= 0:
                raise TimeoutError("articulated_fixture_control_action_budget_exhausted")
            reached_waypoint = _advance_pose(
                bridge, target_position, target_quaternion, f"articulated_fixture_{predicate[0].lower()}",
                f"native_{predicate[0].lower()}_{predicate[1]}", counter,
                remaining_actions, target_binding, instruction, policy,
                monitor, records, frames,
                stop_condition=lambda: bool(contact_state["active"]),
                planning_tracking=tracking,
                step_callback=record_articulation_feedback,
                position_tolerance_m=position_tolerance_m,
                target_pose_provider=route_target_provider,
                absolute_action_limit=ARTICULATED_FIXTURE_TOTAL_ACTION_BUDGET,
            )
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            failed_waypoint_index = waypoint_index
            break
        if contact_state["active"]:
            route_tracking["contact_transition"] = {
                "first_contact_action_index": contact_state["first_action_index"],
                "route_waypoint_index": waypoint_index,
                "contact_pairs": contact_state["contact_pairs"],
                "joint_position_at_transition": contact_state["latest_context"].get("joint", {}).get("position"),
                "static_waypoint_reached": bool(reached_waypoint),
            }
            break
        native = bridge.request({"op": "native_task_status"})
        if any(row.get("predicate") == predicate and row.get("value") is True
               for row in native.get("native_goal_values", [])):
            break
        if counter[0] >= ARTICULATED_FIXTURE_TOTAL_ACTION_BUDGET:
            failure_reason = "TimeoutError: articulated_fixture_control_action_budget_exhausted"
            failed_waypoint_index = waypoint_index
            break
    if (
        failure_reason is None
        and contact_state["active"]
        and contact_state["latest_context"].get("native_goal_value") is not True
    ):
        contact_actions_used = counter[0] - int(contact_phase_start_action or counter[0])
        remaining_contact_actions = min(
            ARTICULATED_FIXTURE_CONTACT_ACTION_BUDGET - contact_actions_used,
            ARTICULATED_FIXTURE_TOTAL_ACTION_BUDGET - counter[0],
        )
        if remaining_contact_actions <= 0:
            failure_reason = "TimeoutError: articulated_fixture_control_action_budget_exhausted"
        else:
            last_robot = records[-1]["snapshot_state"]["robot_states"][0]
            joint_type = str(joint.get("joint_type", "")).lower()
            follow_lead = {
                "value": (
                    ARTICULATED_FIXTURE_HINGE_FOLLOW_LEAD_RAD
                    if joint_type == "hinge" else ARTICULATED_FIXTURE_SLIDE_FOLLOW_LEAD_M
                ),
                "maximum": (
                    ARTICULATED_FIXTURE_HINGE_FOLLOW_MAX_LEAD_RAD
                    if joint_type == "hinge" else 0.009
                ),
            }
            no_joint_progress_chunks = 0
            dynamic_phase = f"articulated_fixture_{predicate[0].lower()}_live_contact_follow"
            dynamic_tracking = {
                "tracking_mode": "live_hinge_or_slide_follow_from_measured_robot_fixture_contact",
                "native_goal_predicate": list(predicate),
                "fixture_id": articulation_context.get("fixture_id_from_active_BDDL"),
                "joint_name": joint.get("joint_name"),
                "joint_target_position": joint.get("target_position"),
                "first_contact_action_index": contact_state["first_action_index"],
                "contact_pairs_at_transition": contact_state["contact_pairs"],
                "contact_action_budget": remaining_contact_actions,
                "contact_anchor_joint_position": contact_state["anchor_joint_position"],
                "contact_anchor_eef_position_world_m": contact_state[
                    "anchor_eef_position_world_m"
                ],
                "gripper_strategy": (
                    "close_after_measured_fixture_contact_to_create_pinch"
                    if (
                        str(predicate[0]).lower() in {"turnon", "turnoff"}
                        or joint_type == "slide"
                    )
                    else "neutral_hold"
                ),
                "gripper_action_command": (
                    ARTICULATED_FIXTURE_SWITCH_TURN_GRIPPER_ACTION
                    if (
                        str(predicate[0]).lower() in {"turnon", "turnoff"}
                        or joint_type == "slide"
                    )
                    else 0.0
                ),
            }
            follow_gripper_action = float(dynamic_tracking["gripper_action_command"])
            contact_state["fixture_gripper_close_commanded"] = follow_gripper_action > 0.5
            contact_state["switch_gripper_close_commanded"] = (
                str(predicate[0]).lower() in {"turnon", "turnoff"}
                and follow_gripper_action > 0.5
            )

            def live_articulation_target_provider(snapshot_state):
                feedback = contact_state["latest_context"]
                current_joint = feedback.get("joint", {})
                anchored_feedback = {
                    **feedback,
                    "joint_axis_world": contact_state["anchor_joint_axis_world"],
                    "joint_origin_world_m": contact_state["anchor_joint_origin_world_m"],
                }
                dynamic_tracking.update({
                    "fixture_joint_position_at_feedback": current_joint.get("position"),
                    "fixture_joint_target_position": current_joint.get("target_position"),
                    "fixture_contact_anchor_joint_position": contact_state["anchor_joint_position"],
                    "fixture_contact_anchor_eef_position_world_m": contact_state[
                        "anchor_eef_position_world_m"
                    ],
                    "fixture_follow_lead": follow_lead["value"],
                    "joint_axis_world": contact_state["anchor_joint_axis_world"],
                    "joint_origin_world_m": contact_state["anchor_joint_origin_world_m"],
                })
                return _articulated_fixture_follow_pose(
                    np.asarray(contact_state["anchor_eef_position_world_m"], dtype=np.float64),
                    np.asarray(contact_state["anchor_eef_quaternion_wxyz"], dtype=np.float64),
                    anchored_feedback,
                    joint_position_at_anchor=float(contact_state["anchor_joint_position"]),
                    follow_lead=follow_lead["value"],
                    allow_terminal_stop_follow=(str(predicate[0]).lower() == "turnoff"),
                )

            while contact_state["latest_context"].get("native_goal_value") is not True:
                contact_actions_used = counter[0] - int(contact_phase_start_action or counter[0])
                remaining_contact_actions = min(
                    ARTICULATED_FIXTURE_CONTACT_ACTION_BUDGET - contact_actions_used,
                    ARTICULATED_FIXTURE_TOTAL_ACTION_BUDGET - counter[0],
                )
                if remaining_contact_actions <= 0:
                    failure_reason = (
                        "TimeoutError: articulated_fixture_live_contact_follow_action_budget_exhausted"
                    )
                    break
                chunk_actions = min(
                    ARTICULATED_FIXTURE_FOLLOW_ACTION_CHUNK, remaining_contact_actions,
                )
                actions_before_chunk = counter[0]
                joint_before_chunk = float(
                    contact_state["latest_context"].get("joint", {}).get("position")
                )
                dynamic_tracking["fixture_follow_lead"] = follow_lead["value"]
                dynamic_tracking["follow_chunk_index"] = len(contact_state["follow_chunks"])
                chunk_timed_out = False
                try:
                    _advance_pose(
                        bridge,
                        np.asarray(last_robot["eef_pos"], dtype=np.float64),
                        np.asarray(last_robot["eef_quat"], dtype=np.float64),
                        dynamic_phase,
                        f"native_{predicate[0].lower()}_{predicate[1]}",
                        counter,
                        chunk_actions,
                        target_binding,
                        instruction,
                        policy,
                        monitor,
                        records,
                        frames,
                        stop_condition=lambda: (
                            contact_state["latest_context"].get("native_goal_value") is True
                        ),
                        planning_tracking=dynamic_tracking,
                        step_callback=record_articulation_feedback,
                        position_tolerance_m=ARTICULATED_FIXTURE_FOLLOW_POSITION_TOLERANCE_M,
                        target_pose_provider=live_articulation_target_provider,
                        gripper_action=follow_gripper_action,
                        absolute_action_limit=ARTICULATED_FIXTURE_TOTAL_ACTION_BUDGET,
                    )
                except TimeoutError as exc:
                    expected_timeout = f"{dynamic_phase}_did_not_reach_pose_within_{chunk_actions}_actions"
                    if str(exc) != expected_timeout:
                        failure_reason = f"{type(exc).__name__}: {exc}"
                        break
                    chunk_timed_out = True
                except Exception as exc:
                    failure_reason = f"{type(exc).__name__}: {exc}"
                    break
                actions_in_chunk = counter[0] - actions_before_chunk
                joint_after_chunk = float(
                    contact_state["latest_context"].get("joint", {}).get("position")
                )
                joint_progress = abs(joint_after_chunk - joint_before_chunk)
                contact_state["follow_chunks"].append({
                    "chunk_index": len(contact_state["follow_chunks"]),
                    "action_count": actions_in_chunk,
                    "joint_position_before": joint_before_chunk,
                    "joint_position_after": joint_after_chunk,
                    "joint_progress_abs": joint_progress,
                    "follow_lead": follow_lead["value"],
                    "last_action_has_fixture_contact": contact_state[
                        "last_action_has_fixture_contact"
                    ],
                    "chunk_timed_out_at_lead_pose": chunk_timed_out,
                })
                if contact_state["latest_context"].get("native_goal_value") is True:
                    break
                if actions_in_chunk == 0:
                    no_joint_progress_chunks += 1
                elif joint_progress < 0.003:
                    no_joint_progress_chunks += 1
                else:
                    no_joint_progress_chunks = 0
                step = 0.015 if joint_type == "hinge" else 0.0015
                ramp_step = (
                    ARTICULATED_FIXTURE_HINGE_FOLLOW_RAMP_STEP_RAD
                    if joint_type == "hinge"
                    else ARTICULATED_FIXTURE_SLIDE_FOLLOW_RAMP_STEP_M
                )
                if contact_state["last_action_has_fixture_contact"]:
                    if no_joint_progress_chunks and follow_lead["value"] < follow_lead["maximum"]:
                        follow_lead["value"] = min(
                            follow_lead["maximum"], follow_lead["value"] + step,
                        )
                    elif no_joint_progress_chunks >= 3:
                        failure_reason = (
                            "RuntimeError: articulated_fixture_live_follow_joint_progress_stalled"
                        )
                        break
                    elif joint_progress >= 0.003 and follow_lead["value"] < follow_lead["maximum"]:
                        follow_lead["value"] = min(
                            follow_lead["maximum"], follow_lead["value"] + ramp_step,
                        )
                elif follow_lead["value"] > 0.0:
                    follow_lead["value"] = max(0.0, follow_lead["value"] - step)
                elif no_joint_progress_chunks >= 3:
                    failure_reason = (
                        "RuntimeError: articulated_fixture_live_follow_joint_progress_stalled"
                    )
                    break
    final_native = bridge.request({"op": "native_task_status"})
    articulation_after = bridge.request({
        "op": "task_success_first_articulation_context",
        "expected_instruction": instruction,
        "predicate": predicate,
    })
    _write_recording(output_dir, records, frames)
    final_snapshot = bridge.request({"op": "export_runtime_snapshot"})
    snapshot_path = output_dir / "articulated_fixture_terminal_runtime_snapshot.pkl"
    if final_snapshot.get("status") == "complete" and isinstance(final_snapshot.get("payload"), bytes):
        snapshot_path.write_bytes(final_snapshot["payload"])
    predicate_value = next(
        (row.get("value") for row in final_native.get("native_goal_values", [])
         if row.get("predicate") == predicate), None
    )
    failure_reason, termination_reason = _articulated_fixture_terminal_outcome(
        failure_reason, final_native,
    )
    return {
        "status": "complete_task_success" if final_native.get("task_success") is True
        else "partial_predicate_completed" if predicate_value is True
        else "articulated_fixture_goal_incomplete",
        "failure_reason": failure_reason,
        "termination_reason": termination_reason,
        "failed_waypoint_index": failed_waypoint_index,
        "precontact_route": route_tracking,
        "fixture_contact_tracking": {
            "active": bool(contact_state["active"]),
            "last_action_has_fixture_contact": bool(
                contact_state["last_action_has_fixture_contact"]
            ),
            "first_contact_action_index": contact_state["first_action_index"],
            "contact_pairs": contact_state["contact_pairs"],
            "switch_gripper_close_commanded": contact_state[
                "switch_gripper_close_commanded"
            ],
            "fixture_gripper_close_commanded": contact_state[
                "fixture_gripper_close_commanded"
            ],
            "switch_bilateral_contact_observed": contact_state[
                "switch_bilateral_contact_observed"
            ],
            "joint_position_after": contact_state["latest_context"].get("joint", {}).get("position"),
            "joint_target_position": contact_state["latest_context"].get("joint", {}).get("target_position"),
            "joint_axis_world": contact_state["latest_context"].get("joint_axis_world"),
            "joint_origin_world_m": contact_state["latest_context"].get("joint_origin_world_m"),
            "contact_anchor_joint_axis_world": contact_state["anchor_joint_axis_world"],
            "contact_anchor_joint_origin_world_m": contact_state["anchor_joint_origin_world_m"],
            "contact_anchor_joint_position": contact_state["anchor_joint_position"],
            "contact_anchor_eef_position_world_m": contact_state["anchor_eef_position_world_m"],
            "follow_chunks": contact_state["follow_chunks"],
        },
        "native_goal_predicate": predicate,
        "native_goal_value": predicate_value,
        "native_task_success": final_native.get("task_success"),
        "final_native_task_status": final_native,
        "articulation_before": articulation_context,
        "articulation_after": articulation_after,
        "recorded_actions": len(records),
        "recorded_physics_substeps": int(sum(
            item.get("contact_monitoring", {}).get("captured_physics_substeps", 0)
            for item in records
        )),
        "terminal_snapshot": str(snapshot_path) if snapshot_path.is_file() else None,
        "safety_certified": False,
        "artifacts": {
            "records": str(output_dir / "physical_step_records.jsonl.gz"),
            "npz": str(output_dir / "task_success_first_actions_states.npz"),
            "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
        },
    }


def _native_place_single(
    bridge,
    *,
    target_id: str,
    target_binding: dict[str, Any],
    target_context: dict[str, Any] | None = None,
    predicate: list[str],
    instruction: str,
    model_sha: str,
    safety_policy: dict[str, Any],
    controller_fields: dict[str, Any],
    controller: TaskPhaseController,
    output_dir: Path,
    placement_candidate: dict[str, Any] | None = None,
    keypose_experience: dict[str, Any] | None = None,
    expected_bddl_sha256: str | None = None,
    demonstration_horizon: int = 1000,
    rrt_time_budget_s: float = NATIVE_PLACEMENT_RRT_BUDGET_S,
    native_region_transit_clearance_mode: str = "compiled_obstacle",
    native_region_transit_height_override_m: float | None = None,
    native_region_progress_max_step_m: float = 0.008,
    native_region_entry_mode: str = "over_rim",
    native_region_aperture_tilt_degrees: float = 0.0,
    native_region_entry_height_offset_m: float | None = None,
) -> dict[str, Any]:
    if (
        isinstance(native_region_progress_max_step_m, bool)
        or not isinstance(native_region_progress_max_step_m, (int, float))
        or not np.isfinite(float(native_region_progress_max_step_m))
        or not 0.001 <= float(native_region_progress_max_step_m) <= 0.02
    ):
        raise ValueError("native_region_progress_max_step_must_be_in_0.001_to_0.02")
    if native_region_transit_height_override_m is not None and (
        isinstance(native_region_transit_height_override_m, bool)
        or not isinstance(native_region_transit_height_override_m, (int, float))
        or not np.isfinite(float(native_region_transit_height_override_m))
        or not 0.02 <= float(native_region_transit_height_override_m) <= 2.50
    ):
        raise ValueError("native_region_transit_height_override_must_be_in_0.02_to_2.50")
    if native_region_entry_mode not in {"over_rim", "horizontal_aperture"}:
        raise ValueError("native_region_entry_mode_must_be_over_rim_or_horizontal_aperture")
    if (
        native_region_entry_height_offset_m is not None
        and (
            isinstance(native_region_entry_height_offset_m, bool)
            or not isinstance(native_region_entry_height_offset_m, (int, float))
            or not np.isfinite(float(native_region_entry_height_offset_m))
            or not 0.0 <= float(native_region_entry_height_offset_m) <= 0.15
            or native_region_entry_mode != "horizontal_aperture"
        )
    ):
        raise ValueError(
            "native_region_entry_height_offset_requires_horizontal_aperture_and_value_in_0_to_0.15"
        )
    if (
        isinstance(native_region_aperture_tilt_degrees, bool)
        or not isinstance(native_region_aperture_tilt_degrees, (int, float))
        or not np.isfinite(float(native_region_aperture_tilt_degrees))
        or not 0.0 <= float(native_region_aperture_tilt_degrees) <= 90.0
        or (
            native_region_entry_mode != "horizontal_aperture"
            and float(native_region_aperture_tilt_degrees) != 0.0
        )
    ):
        raise ValueError("native_region_aperture_tilt_requires_horizontal_aperture_and_angle_in_0_to_90")
    output_dir.mkdir(parents=True, exist_ok=True)
    from scipy.spatial.transform import Rotation
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _advance_pose,
        _compress_eef_route,
        _record_step,
        _snapshot_state_vector,
        _state_sha,
        _wxyz,
        _write_recording,
    )
    from research.gs_opd.repair.task_success_first_search import route_control_action_budget
    from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor
    from research.gs_opd.repair.keypose_driven_planning import (
        retarget_keypose_sequence,
        select_keypose_suffix,
        update_pose_record,
    )

    native_goal = bridge.request({"op": "native_goal_query", "predicate": predicate})
    if native_goal.get("status") != "complete" or native_goal.get("predicate_is_native_goal") is not True:
        raise RuntimeError(f"active_BDDL_goal_region_unavailable:{native_goal.get('error')}")
    region = native_goal["region"]
    observed = bridge.request({"op": "observe", "include_snapshot_state": True})
    state = observed["snapshot_state"]
    obj = _object_state(state, int(target_binding["body_id"]))
    robot = state["robot_states"][0]
    object_world = np.eye(4)
    object_world[:3, :3] = Rotation.from_quat(
        np.asarray(obj["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
    ).as_matrix()
    object_world[:3, 3] = obj["root_pos"]
    eef_world = np.eye(4)
    eef_world[:3, :3] = Rotation.from_quat(
        np.asarray(robot["eef_quat"], dtype=np.float64)[[1, 2, 3, 0]]
    ).as_matrix()
    eef_world[:3, 3] = robot["eef_pos"]
    object_to_eef = np.linalg.inv(object_world) @ eef_world
    route_strategy = (
        str(placement_candidate.get("route_strategy", "jointspace_rrt"))
        if placement_candidate is not None else "jointspace_rrt"
    )
    goal_object = np.eye(4)
    if placement_candidate is None:
        goal_object[:3, :3] = object_world[:3, :3]
        goal_object[:3, 3] = object_world[:3, 3]
    else:
        goal_object[:3, :3] = np.asarray(placement_candidate["rotation_world"], dtype=np.float64)
        goal_object[:3, 3] = np.asarray(placement_candidate["position_world_m"], dtype=np.float64)
    goal_eef = goal_object @ object_to_eef
    near_goal_release_probe = None
    release_probe_alignment = None
    if controller.near_goal_release_probe_enabled and placement_candidate is not None:
        initial_native_status = bridge.request({"op": "native_task_status"})
        initial_native_values, _ = _task_values(initial_native_status)
        initial_grasp = bridge.request({
            "op": "grasp_state",
            "carried_binding": target_binding,
            "expected_instruction": instruction,
        })
        initial_finger_contacts = initial_grasp.get("finger_contacts", {})
        probe_max_error = 0.012
        configured_probe_max_error = float(_STAGE_A_NEAR_GOAL_RELEASE_PROBE_MAX_ERROR_M)
        if configured_probe_max_error > probe_max_error and predicate[0].lower() == "in":
            try:
                region_position = np.asarray(region["site_position_world_m"], dtype=np.float64)
                region_rotation = np.asarray(region["site_rotation_world"], dtype=np.float64)
                region_size = np.asarray(region["site_size_m"], dtype=np.float64)
                current_local = region_rotation.T @ (object_world[:3, 3] - region_position)
                target_local = region_rotation.T @ (goal_object[:3, 3] - region_position)
                collision_lower, collision_upper = _placement_collision_bounds_in_region(
                    target_context, region_rotation, object_world[:3, :3],
                )
                current_geometry_lower = current_local[:2] + collision_lower[:2]
                current_geometry_upper = current_local[:2] + collision_upper[:2]
                horizontal_goal_error = float(np.linalg.norm(
                    current_local[:2] - target_local[:2]
                ))
                vertical_drop = float(current_local[2] - target_local[2])
                root_center_within_region = bool(
                    region_size.shape == (3,)
                    and np.all(np.abs(current_local[:2]) <= region_size[:2] + 0.005)
                )
                footprint_within_aperture = bool(
                    region_size.shape == (3,)
                    and np.all(current_geometry_lower >= -region_size[:2] - 0.015)
                    and np.all(current_geometry_upper <= region_size[:2] + 0.015)
                )
                release_probe_alignment = {
                    "status": "eligible" if (
                        0.0 < vertical_drop <= configured_probe_max_error
                        and root_center_within_region
                    ) else "not_eligible",
                    "basis": "live object root center projected into native In site; Stage A physical drop permits collisions",
                    "native_region_site_position_world_m": region_position.tolist(),
                    "native_region_site_rotation_world": region_rotation.tolist(),
                    "native_region_half_extents_m": region_size.tolist(),
                    "live_object_root_position_world_m": object_world[:3, 3].tolist(),
                    "candidate_object_root_position_world_m": goal_object[:3, 3].tolist(),
                    "live_object_root_position_region_local_m": current_local.tolist(),
                    "candidate_object_root_position_region_local_m": target_local.tolist(),
                    "horizontal_goal_error_m": horizontal_goal_error,
                    "horizontal_goal_error_is_diagnostic_only": True,
                    "vertical_drop_to_candidate_m": vertical_drop,
                    "object_collision_bounds_region_m": [
                        collision_lower.tolist(), collision_upper.tolist(),
                    ],
                    "current_object_collision_bounds_region_m": [
                        current_geometry_lower.tolist(), current_geometry_upper.tolist(),
                    ],
                    "region_half_extents_m": region_size.tolist(),
                    "root_center_within_native_region_xy": root_center_within_region,
                    "footprint_within_aperture_with_15mm_tolerance": footprint_within_aperture,
                    "footprint_gate_is_safety_only_and_not_used_in_stage_a": True,
                    "direct_state_write": False,
                    "native_BDDL_success_still_required": True,
                }
                if release_probe_alignment["status"] == "eligible":
                    probe_max_error = configured_probe_max_error
            except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
                release_probe_alignment = {
                    "status": "not_eligible",
                    "reason": f"{type(exc).__name__}:{exc}",
                }
        near_goal_release_probe = controller.begin_near_goal_release_probe(
            object_id=target_id,
            native_goal_satisfied=initial_native_values.get(tuple(predicate)) is True,
            held_object=(
                target_id if (
                    initial_grasp.get("grasp_confirmed") is True
                    and initial_finger_contacts.get("left") is True
                    and initial_finger_contacts.get("right") is True
                ) else None
            ),
            bilateral_grasp_confirmed=(
                initial_grasp.get("grasp_confirmed") is True
                and initial_finger_contacts.get("left") is True
                and initial_finger_contacts.get("right") is True
            ),
            object_goal_position_error_m=float(
                np.linalg.norm(object_world[:3, 3] - goal_object[:3, 3])
            ),
            maximum_position_error_m=probe_max_error,
        )
        if near_goal_release_probe is not None:
            if release_probe_alignment is not None:
                near_goal_release_probe["aperture_alignment"] = release_probe_alignment
                near_goal_release_probe["probe_mode"] = (
                    "gravity_drop_inside_native_In_aperture"
                    if release_probe_alignment.get("status") == "eligible"
                    else "euclidean_near_goal"
                )
            _json(output_dir / "stage_a_near_goal_release_probe.json", near_goal_release_probe)
    if (
        controller.near_goal_release_probe_enabled
        and predicate[0].lower() == "in"
        and float(_STAGE_A_NEAR_GOAL_RELEASE_PROBE_MAX_ERROR_M) > 0.012
        and near_goal_release_probe is None
    ):
        probe_status = bridge.request({"op": "native_task_status"})
        preflight = {
            "status": "not_eligible_no_physical_action",
            "reason": (
                release_probe_alignment.get("status")
                if release_probe_alignment is not None
                else "live_aperture_projection_unavailable"
            ),
            "aperture_alignment": release_probe_alignment,
            "direct_state_write": False,
            "native_task_status": probe_status,
            "safety_certified": False,
        }
        _json(output_dir / "stage_a_near_goal_release_probe.json", preflight)
        return {
            "status": "stage_a_near_goal_release_probe_not_eligible",
            "failure_reason": preflight["reason"],
            "stage_a_near_goal_release_probe": preflight,
            "native_goal_predicate": predicate,
            "native_task_success": probe_status.get("task_success"),
            "final_native_task_status": probe_status,
            "recorded_actions": 0,
            "safety_certified": False,
        }
    resume_at_release = controller.phase == "release"
    keypose_sequence = None
    if keypose_experience is not None and not resume_at_release:
        source_task = keypose_experience.get("source", {})
        if (
            keypose_experience.get("schema_version") != "gs_opd_keypose_experience.v1"
            or source_task.get("native_task_success") is not True
            or source_task.get("official_safety_demonstrations_used") is not False
            or keypose_experience.get("object_family") != target_id.rsplit("_", 1)[0]
            or str(keypose_experience.get("target_relation", "")).lower() != predicate[0].lower()
            or keypose_experience.get("target_region_name") != predicate[-1]
        ):
            raise ValueError("keypose_experience_does_not_match_verified_source_and_active_native_goal")
        region_world = np.eye(4, dtype=np.float64)
        region_world[:3, :3] = np.asarray(region["site_rotation_world"], dtype=np.float64)
        region_world[:3, 3] = np.asarray(region["site_position_world_m"], dtype=np.float64)
        if route_strategy != "native_region_progress_osc":
            relocated = retarget_keypose_sequence(
                keypose_experience,
                held_object_world=object_world,
                current_eef_world=eef_world,
                destination_region_world=region_world,
                destination_object_world=goal_object,
            )
            keypose_sequence = [row for row in relocated if row["name"] in {
                "lift", "high_west_clearance", "high_south_corridor",
                "yaw_alignment", "preplace", "place",
            }]
            route_start_after = keypose_experience.get("_route_start_after")
            keypose_sequence = select_keypose_suffix(keypose_sequence, route_start_after)
    keypose_side_reference_region = None
    if keypose_experience is not None and route_strategy == "native_region_progress_osc":
        rows = {
            row.get("name"): row for row in keypose_experience.get("keypose_sequence", [])
        }
        lift = rows.get("lift", {}).get("region_to_object", {}).get("matrix")
        preplace = rows.get("preplace", {}).get("region_to_object", {}).get("matrix")
        if lift is not None and preplace is not None:
            lift_position = np.asarray(lift, dtype=np.float64)[:3, 3]
            preplace_position = np.asarray(preplace, dtype=np.float64)[:3, 3]
            approach = preplace_position[:2] - lift_position[:2]
            norm = float(np.linalg.norm(approach))
            if norm > 1e-8:
                approach /= norm
                keypose_side_reference_region = np.asarray([-approach[1], approach[0]])
    entry_direction = np.asarray(
        placement_candidate.get("entry_direction_world", [0.0, 0.0, 0.0])
        if placement_candidate is not None else [0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    if entry_direction.shape != (3,) or not np.isfinite(entry_direction).all():
        raise ValueError("native_goal_placement_entry_direction_is_invalid")
    entry_norm = float(np.linalg.norm(entry_direction))
    if entry_norm > 1e-8:
        entry_direction = entry_direction / entry_norm
    approach_height_m = float(
        placement_candidate.get("approach_height_m", 0.14)
        if placement_candidate is not None else 0.14
    )
    if not np.isfinite(approach_height_m) or not 0.08 <= approach_height_m <= 0.50:
        raise ValueError("native_goal_approach_height_must_be_in_0.08_to_0.50")
    if keypose_sequence is not None:
        for row in keypose_sequence:
            if row["name"] == "preplace":
                preplace_position = np.asarray(
                    row["eef_pose_world"]["position_m"], dtype=np.float64,
                )
                preplace_position += entry_direction * _native_region_entry_offset_m(predicate)
                preplace_position[2] = float(goal_eef[2, 3] + approach_height_m)
                update_pose_record(
                    row["eef_pose_world"], position_m=preplace_position,
                )
    container_entry_transit = None
    if (
        route_strategy == "jointspace_rrt"
        and predicate[0].lower() == "in"
        and entry_norm > 1e-8
    ):
        if not isinstance(target_context, dict) or target_context.get("status") != "complete":
            raise RuntimeError("native_In_RRT_requires_live_compiled_target_geometry")
        scene_envelopes = bridge.request({
            "op": "task_scene_object_collision_envelopes",
        })
        obstacle_clearance = _native_region_corridor_obstacle_clearance(
            start_object_world=object_world,
            goal_object_world=goal_object,
            region=region,
            target_context=target_context,
            object_to_eef=object_to_eef,
            scene_envelopes=scene_envelopes,
            excluded_object_ids={target_id, predicate[-1]},
            expected_model_sha256=model_sha,
            clearance_margin_m=0.02,
        )
        container_entry_transit = _native_region_entry_transit_poses(
            object_world,
            goal_object,
            region=region,
            entry_direction_world=entry_direction,
            entry_clearance_radius_m=_placement_footprint_radius(
                target_context,
                np.asarray(region["site_rotation_world"], dtype=np.float64),
                goal_object[:3, :3],
            ),
            obstacle_clearance_height_world_m=(
                obstacle_clearance.get("required_transit_height_world_m")
            ),
        )
        high_eef = container_entry_transit["outside_entry_object_world"] @ object_to_eef
        container_route = container_entry_transit["route"]
        _json(output_dir / "native_region_entry_transit_route.json", {
            "schema_version": "gs_opd_native_region_entry_transit_route.v1",
            "native_goal_predicate": predicate,
            "model_geometry_sha256": model_sha,
            "compiled_obstacle_clearance": obstacle_clearance,
            "route": container_route,
            "outside_entry_eef_pose_world": {
                "position_m": high_eef[:3, 3].tolist(),
                "rotation": high_eef[:3, :3].tolist(),
            },
            "inside_alignment_object_pose_world": container_entry_transit[
                "inside_alignment_object_world"
            ].tolist(),
            "safety_certified": False,
        })
    else:
        entry_eef = goal_eef.copy()
        entry_eef[:3, 3] += entry_direction * _native_region_entry_offset_m(predicate)
        high_eef = entry_eef.copy()
        high_eef[:3, 3] += np.asarray([0.0, 0.0, approach_height_m])
    high_quat = _wxyz(high_eef[:3, :3])
    state_sha = _state_sha(_snapshot_state_vector(state))
    rrt_goal_eef, rrt_goal_role = _native_placement_planner_target(
        predicate, goal_eef, high_eef,
    )
    start_snapshot = bridge.request({"op": "export_runtime_snapshot"})
    if start_snapshot.get("status") != "complete" or not isinstance(start_snapshot.get("payload"), bytes):
        raise RuntimeError("native_goal_transport_start_snapshot_export_failed")
    start_snapshot_path = output_dir / "native_goal_transport_start_runtime_snapshot.pkl"
    start_snapshot_path.write_bytes(start_snapshot["payload"])
    planning_policy = copy.deepcopy(safety_policy)
    if "measure_forbidden_object_pair_clearances" in planning_policy:
        planning_policy["measure_forbidden_object_pair_clearances"] = False
    if not resume_at_release:
        planning_policy["carried_object"] = {
            "object_id": target_id,
            "grasp_binding": target_binding,
            "allowed_contact_pairs": [],
            "allowed_support_contact_pairs": [],
        }
    if route_strategy == "jointspace_rrt":
        planning_policy = _task_success_collision_route_policy(planning_policy)
    if resume_at_release:
        planned = {
            "status": "not_required_native_goal_satisfied_resume_at_release",
            "live_state_unchanged": True,
            "safety_certified": False,
        }
    elif keypose_sequence is not None:
        planned = {
            "status": "actual_source_keypose_sequence_pending_per_waypoint_RRT",
            "live_state_unchanged": True,
            "safety_certified": False,
        }
    elif route_strategy == "elevated_osc_waypoints":
        planned = {
            "status": "bounded_alternative_route_selected",
            "failure_reason": "RRT route family skipped by placement portfolio",
            "live_state_unchanged": True,
            "safety_certified": False,
        }
    elif route_strategy == "native_region_progress_osc":
        planned = {
            "status": "native_region_progress_route_selected",
            "planner_skipped": "RRT/IK are optional diagnostics; physical OSC progress decides continuation",
            "live_state_unchanged": True,
            "safety_certified": False,
        }
    else:
        planned = bridge.request({
            "op": "rrt_connect" if route_strategy == "jointspace_rrt" else "task_success_first_rrt",
            "expected_instruction": instruction,
            "model_geometry_sha256": model_sha,
            "start_state_sha256": state_sha,
            "goal_pose_world": {
                "position_world_m": rrt_goal_eef[:3, 3].tolist(),
                "quaternion_wxyz": _wxyz(rrt_goal_eef[:3, :3]).tolist(),
            },
            "safety_geometry_policy": planning_policy,
            "planner_config": {
                "step_rad": 0.22, "connect_max_steps": 20, "goal_bias": 0.15,
                "max_iterations": NATIVE_PLACEMENT_RRT_MAX_ITERATIONS,
                "max_config_checks": NATIVE_PLACEMENT_RRT_MAX_CONFIG_CHECKS,
                "max_candidate_routes": 2,
                "search_time_budget_s": rrt_time_budget_s,
                "shortcut_attempts": 24, "minimum_clearance_m": 0.0,
                "start_joint_limit_tolerance_rad": 0.001,
                # Stage A optimizes native task completion. Keep full robot and
                # carried-object geometry in the planner for diagnostics, but
                # do not reject a route solely for safety contacts; the real
                # MuJoCo execution records every contact substep.
                "require_safe_configurations": False, "seed": 74103,
            },
            "ik_seed_qpos": _placement_ik_seed_portfolio(robot),
        })
        if route_strategy == "jointspace_rrt":
            planned["planning_mode"] = "stage_a_full_geometry_RRT_contact_diagnostics_without_collision_rejection"
            planned["safety_certified"] = False
    planned["goal_role"] = rrt_goal_role
    planned["route_strategy"] = route_strategy
    planner_state_verification = {"status": "not_required_resume_at_release", "live_state_unchanged": True}
    if not resume_at_release:
        post_planner_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
        post_planner_sha = _state_sha(
            _snapshot_state_vector(post_planner_observation.get("snapshot_state", {}))
        )
        planner_state_verification = _verify_planner_live_state(
            planned, post_planner_sha, state_sha,
        )
        planned["reported_live_state_unchanged"] = planned.get("live_state_unchanged")
        planned["live_state_verification"] = planner_state_verification
        planned["live_state_unchanged"] = planner_state_verification["live_state_unchanged"]
        if not planner_state_verification["live_state_unchanged"]:
            raise RuntimeError("native_goal_transport_RRT_changed_live_integration_state")
    planned_route_archive: dict[str, Any] | None = None
    if (
        route_strategy == "jointspace_rrt"
        and planned.get("status") == "offline_feasible"
        and planned.get("live_state_unchanged") is True
    ):
        route_record = {
            "schema_version": "gs_opd_collision_aware_RRT_OSC_route.v1",
            "task": (
                f"{observed.get('scene_metadata', {}).get('benchmark', 'unknown')}/"
                f"L{observed.get('scene_metadata', {}).get('level', 'unknown')}"
                f"T{observed.get('scene_metadata', {}).get('task_id', 'unknown')}"
            ),
            "candidate_id": placement_candidate.get("candidate_id") if placement_candidate else None,
            "native_goal_predicate": predicate,
            "model_geometry_sha256": model_sha,
            "bddl_sha256": expected_bddl_sha256,
            "start_state_sha256": state_sha,
            "goal_pose_world": {
                "position_world_m": rrt_goal_eef[:3, 3].tolist(),
                "quaternion_wxyz": _wxyz(rrt_goal_eef[:3, :3]).tolist(),
            },
            "planner_source": planned.get("planner_source"),
            "planner_mode": planned.get("planning_mode"),
            "collision_constraint": (
                "full robot and carried-object geometry checked for diagnostics; "
                "Stage A route search does not reject collision configurations; "
                "real MuJoCo contact substeps remain recorded"
            ),
            "qpos_path_rad": planned.get("qpos_path"),
            "eef_reference_poses_world": planned.get("eef_reference_poses_world"),
            "planning_seconds": planned.get("planning_seconds"),
            "configuration_checks": planned.get("config_checks"),
            "iterations": planned.get("iterations"),
            "safety_certified": False,
        }
        route_path = output_dir / "native_goal_transport_collision_aware_rrt_route.json"
        _json(route_path, route_record)
        route_bytes = route_path.read_bytes()
        planned_route_archive = {
            "path": str(route_path),
            "sha256": hashlib.sha256(route_bytes).hexdigest(),
            "qpos_path_rows": len(route_record["qpos_path_rad"]),
            "eef_path_rows": len(route_record["eef_reference_poses_world"]),
            "safety_certified": False,
        }
        planned["route_archive"] = planned_route_archive
    _advance_pose.controller_fields = controller_fields
    records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    keypose_rrt_runs: list[dict[str, Any]] = []
    monitor = GraspStabilityMonitor(required_samples=3)
    counter = [0]
    region_progress_route: dict[str, Any] | None = None
    region_progress_history: list[dict[str, Any]] = []
    region_progress_snapshots: list[dict[str, Any]] = []
    native_goal_insertion_rrt: dict[str, Any] | None = None
    durable_records_path = output_dir / "durable_physical_step_records.jsonl"
    durable_step_profile_path = output_dir / "durable_step_profile.jsonl"
    durable_frames_dir = output_dir / "durable_dual_camera_frames"
    durable_frames_dir.mkdir(parents=True, exist_ok=True)
    contact_escape: dict[str, Any] | None = None

    def persist_step(record: dict[str, Any], frame: dict[str, Any]) -> None:
        # Preserve each executed action before the phase returns. The final
        # bundle still writes compressed records, NPZ and MP4; these files make
        # the physical prefix recoverable if a long RRT/OSC stage is interrupted.
        encode_started = time.perf_counter()
        encoded_record = json.dumps(
            record, ensure_ascii=False, separators=(",", ":"),
            default=lambda item: item.tolist() if isinstance(item, np.ndarray) else str(item),
        ).encode("utf-8")
        encode_wall_s = time.perf_counter() - encode_started
        durable_write_started = time.perf_counter()
        with durable_records_path.open("ab") as handle:
            handle.write(encoded_record)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        durable_write_wall_s = time.perf_counter() - durable_write_started
        frame_write_started = time.perf_counter()
        combined = np.concatenate((frame["agentview_rgb"], frame["wrist_rgb"]), axis=1)
        frame_path = durable_frames_dir / f"frame_{int(frame['step_index']):06d}.jpg"
        if not cv2.imwrite(str(frame_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("durable_dual_camera_frame_write_failed")
        frame_write_wall_s = time.perf_counter() - frame_write_started
        profile = {
            "step_index": int(record.get("step_index", -1)),
            "timestep": record.get("timestep"),
            "step_monitored_rpc_wall_s": record.get("execution_timing_s", {}).get("step_monitored_rpc"),
            "grasp_state_rpc_wall_s": record.get("execution_timing_s", {}).get("grasp_state_rpc"),
            "record_json_encode_wall_s": encode_wall_s,
            "record_write_flush_fsync_wall_s": durable_write_wall_s,
            "record_json_bytes": len(encoded_record),
            "camera_frame_write_wall_s": frame_write_wall_s,
            "camera_frame_bytes": frame_path.stat().st_size,
        }
        with durable_step_profile_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(profile, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
        grasp_failure = _transport_grasp_loss_reason(record)
        if grasp_failure is not None:
            raise RuntimeError(grasp_failure)

    def persist_latest_step() -> None:
        if records and frames:
            persist_step(records[-1], frames[-1])

    def archive_failure(failure_phase: str, exc: Exception, route_source: str | None = None):
        """Persist every executed physical prefix before reporting a failed phase."""
        _write_recording(output_dir, records, frames)
        failure_snapshot_path = output_dir / f"native_goal_{failure_phase}_failed_runtime_snapshot.pkl"
        snapshot_error = None
        try:
            final_snapshot = bridge.request({"op": "export_runtime_snapshot"})
            if final_snapshot.get("status") == "complete" and isinstance(final_snapshot.get("payload"), bytes):
                failure_snapshot_path.write_bytes(final_snapshot["payload"])
        except Exception as snapshot_exc:
            snapshot_error = f"{type(snapshot_exc).__name__}: {snapshot_exc}"
        try:
            final_native = bridge.request({"op": "native_task_status"})
        except Exception as native_exc:
            final_native = {
                "status": "unavailable_bridge_terminated",
                "error": f"{type(native_exc).__name__}: {native_exc}",
                "last_archived_task_success": records[-1].get("task_success") if records else None,
            }
        terminal_outcome = _transport_exception_outcome(exc, final_native)
        post_success_exception = terminal_outcome.get("post_success_terminal_exception")
        return {
            "status": (
                terminal_outcome["status"]
                if post_success_exception is not None
                else f"native_goal_{failure_phase}_failed"
            ),
            "failure_reason": terminal_outcome.get("failure_reason"),
            "post_success_terminal_exception": post_success_exception,
            "stable_release_verified": False,
            "native_goal_predicate": predicate,
            "native_task_success": final_native.get("task_success"),
            "final_native_task_status": final_native,
            "stage_a_near_goal_release_probe": near_goal_release_probe,
            "transport_route_source": route_source,
            "target_object_contact_escape": contact_escape,
            "region_progress_route": region_progress_route,
            "region_progress_feedback": region_progress_history,
            "region_progress_snapshots": region_progress_snapshots,
            "placement_candidate": placement_candidate,
            "transport_route_RRT": {
                "status": "not_run" if planned.get("planner_skipped") else planned.get("status"),
                "route_selection_status": planned.get("status"),
                "failure_reason": planned.get("failure_reason"),
                "time_budget_s": rrt_time_budget_s,
                "planning_seconds": planned.get("planning_seconds"),
                "iterations": planned.get("iterations"),
                "configuration_checks": planned.get("config_checks"),
                "route_candidate_count": len(planned.get("route_candidates", [])),
                "planner_source": planned.get("planner_source"),
                "planning_mode": planned.get("planning_mode"),
                "carried_object_geometry_included": planning_policy.get("carried_object") is not None,
                "maximum_configuration_checks": NATIVE_PLACEMENT_RRT_MAX_CONFIG_CHECKS,
                "safety_certified": False,
                "live_state_verification": planner_state_verification,
            },
            "transport_route_archive": planned_route_archive,
            "native_goal_insertion_RRT": native_goal_insertion_rrt,
            "keypose_RRT_waypoints": keypose_rrt_runs,
            "keypose_transfer": {
                "enabled": keypose_experience is not None,
                "experience_id": keypose_experience.get("experience_id") if keypose_experience else None,
                "retargeted_keyposes": keypose_sequence,
                "per_waypoint_planning": keypose_rrt_runs,
                "source_actions_replayed": False,
                "safety_certified": False,
            },
            "recorded_actions": len(records),
            "recorded_physics_substeps": int(sum(
                item.get("contact_monitoring", {}).get("captured_physics_substeps", 0)
                for item in records
            )),
            "failure_phase": failure_phase,
            "start_snapshot": str(start_snapshot_path),
            "terminal_snapshot": str(failure_snapshot_path) if failure_snapshot_path.is_file() else None,
            "terminal_snapshot_export_error": snapshot_error,
            "bridge_process_returncode": (
                bridge.process.poll() if getattr(bridge, "process", None) is not None else None
            ),
            "terminal_snapshot_sha256": (
                hashlib.sha256(failure_snapshot_path.read_bytes()).hexdigest()
                if failure_snapshot_path.is_file() else None
            ),
            "artifacts": {
                "records": str(output_dir / "physical_step_records.jsonl.gz"),
                "npz": str(output_dir / "task_success_first_actions_states.npz"),
                "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                "durable_records": str(durable_records_path),
                "durable_frames": str(durable_frames_dir),
            },
            "safety_certified": False,
        }

    def native_goal_satisfied() -> bool:
        status = bridge.request({"op": "native_task_status"})
        if status.get("status") != "complete":
            raise RuntimeError("native_goal_status_unavailable_during_precise_placement")
        return any(
            row.get("predicate") == predicate and row.get("value") is True
            for row in status.get("native_goal_values", [])
        )
    if resume_at_release:
        route = [np.r_[robot["eef_pos"], _wxyz(eef_world[:3, :3])]]
        transport_route_source = (
            "stage_a_near_goal_physical_release_probe"
            if near_goal_release_probe is not None
            else "verified_native_goal_snapshot_resume_at_release"
        )
    elif keypose_sequence is not None:
        start_pose = np.r_[robot["eef_pos"], _wxyz(eef_world[:3, :3])]
        keypose_poses = [
            np.r_[
                np.asarray(row["eef_pose_world"]["position_m"], dtype=np.float64),
                np.asarray(row["eef_pose_world"]["quaternion_wxyz"], dtype=np.float64),
            ]
            for row in keypose_sequence
        ]
        route = [start_pose, *keypose_poses]
        transport_route_source = "GS_OPD_own_success_object_and_native_region_frame_keypose_sequence"
    elif planned.get("status") == "offline_feasible" and planned.get("live_state_unchanged") is True:
        route = np.asarray(planned.get("eef_reference_poses_world"), dtype=np.float64)
        if route.ndim != 2 or route.shape[1:] != (7,) or not np.isfinite(route).all():
            raise RuntimeError("collision_aware_RRT_returned_an_invalid_EEF_reference_path")
        transport_route_source = "MuJoCo_7D_RRT_full_geometry_EEF_path_tracked_by_native_OSC"
    elif route_strategy == "elevated_osc_waypoints":
        start_pose = np.r_[robot["eef_pos"], _wxyz(eef_world[:3, :3])]
        vertically_lifted = start_pose.copy()
        vertically_lifted[2] += 0.12
        route = _compress_eef_route(np.vstack((
            start_pose,
            vertically_lifted,
            np.r_[high_eef[:3, 3], high_quat],
        )))
        transport_route_source = "portfolio_alternative_vertical_lift_then_high_OSC_waypoints"
    elif route_strategy == "native_region_progress_osc":
        try:
            if not isinstance(target_context, dict) or target_context.get("status") != "complete":
                raise ValueError("native_region_compiled_target_context_unavailable")
            scene_envelopes = bridge.request({
                "op": "task_scene_object_collision_envelopes",
            })
            obstacle_clearance = _native_region_corridor_obstacle_clearance(
                start_object_world=object_world,
                goal_object_world=goal_object,
                region=region,
                target_context=target_context,
                object_to_eef=object_to_eef,
                scene_envelopes=scene_envelopes,
                excluded_object_ids={target_id, predicate[-1]},
                expected_model_sha256=model_sha,
            )
        except Exception as exc:
            return archive_failure(
                "route_geometry",
                RuntimeError(
                    "native_region_live_obstacle_clearance_unavailable:"
                    f"{type(exc).__name__}: {exc}"
                ),
                "live_compiled_object_corridor_geometry",
            )
        route_variant = str(
            placement_candidate.get("route_variant", "direct_region_progress")
            if placement_candidate is not None else "direct_region_progress"
        )
        live_aperture_contacts = None
        live_aperture_contact_escape = None
        if (
            native_region_entry_mode == "horizontal_aperture"
            and predicate[0].lower() == "in"
        ):
            live_aperture_contacts = bridge.request({"op": "inspect_contacts"})
            live_aperture_contact_escape = _object_contact_escape_direction(
                live_aperture_contacts, target_binding, obj["root_pos"],
            )
        route_entry_direction = (
            entry_direction.copy()
            if predicate[0].lower() == "in" and entry_norm > 1e-8
            else None
        )
        aperture_entry_direction_selection = {
            "source": "ranked_native_region_layout",
            "inward_direction_world": (
                route_entry_direction.tolist()
                if route_entry_direction is not None else None
            ),
            "fixture_object_id": None,
        }
        aperture_fixture_object_id = None
        if (
            native_region_entry_mode == "horizontal_aperture"
            and predicate[0].lower() == "in"
        ):
            target_region_fixture_id = None
            target_region_id = str(predicate[-1])
            for region_suffix in (
                "_heating_region", "_back_contain_region",
                "_front_contain_region", "_contain_region",
            ):
                if target_region_id.endswith(region_suffix):
                    target_region_fixture_id = target_region_id[:-len(region_suffix)]
                    break
            contact_blockers = set(
                live_aperture_contact_escape.get("blocker_geom_names", [])
                if live_aperture_contact_escape is not None else []
            )
            contact_is_with_target_aperture = bool(
                target_region_fixture_id
                and any(name.startswith(f"{target_region_fixture_id}_")
                        for name in contact_blockers)
            )
            if contact_is_with_target_aperture:
                # A live target-to-frame contact reveals the actual opening
                # normal. Reuse its opposite as the insertion axis so the
                # carried object's lateral offset is corrected while still
                # outside, rather than driving a diagonal path into the frame.
                measured_outward = np.asarray(
                    live_aperture_contact_escape["direction_world"],
                    dtype=np.float64,
                )
                measured_inward = -measured_outward
                measured_inward[2] = 0.0
                measured_inward_norm = float(np.linalg.norm(measured_inward))
                if measured_inward_norm > 1e-8:
                    route_entry_direction = measured_inward / measured_inward_norm
                    aperture_entry_direction_selection = {
                        "source": "live_MuJoCo_target_contact_normal_at_native_aperture",
                        "fixture_object_id": target_region_fixture_id,
                        "blocker_geom_names": sorted(contact_blockers),
                        "measured_outward_direction_world": measured_outward.tolist(),
                        "inward_direction_world": route_entry_direction.tolist(),
                        "lateral_alignment_before_insertion_required": True,
                        "physical_entry_direction_is_a_hypothesis_not_a_safety_certificate": True,
                    }
            # A target that is already near the native region can make the
            # ranked object-to-goal vector nearly tangent to the container.
            # For a front aperture, derive an inward normal from the nearest
            # intersecting fixture envelope toward the native region center;
            # this avoids approaching through the fixture roof/side wall.
            region_center = np.asarray(
                region.get("site_position_world_m"), dtype=np.float64,
            )
            fixture_rows = [
                row for row in obstacle_clearance.get("intersecting_objects", [])
                if isinstance(row, dict)
                and isinstance(row.get("center_world_m"), list)
                and len(row["center_world_m"]) == 3
            ]
            fixture = min(
                fixture_rows,
                key=lambda row: float(np.linalg.norm(
                    (region_center - np.asarray(row["center_world_m"], dtype=np.float64))[:2]
                )),
                default=None,
            )
            candidate_exterior_projection_m = None
            if route_entry_direction is not None:
                candidate_outward = -route_entry_direction.copy()
                candidate_outward[2] = 0.0
                candidate_outward_norm = float(np.linalg.norm(candidate_outward))
                if candidate_outward_norm > 1e-8:
                    candidate_outward /= candidate_outward_norm
                    candidate_exterior_projection_m = float(np.dot(
                        (object_world[:3, 3] - goal_object[:3, 3])[:2],
                        candidate_outward[:2],
                    ))
            # Keep the ranked layout's entry side when the live object is
            # already clearly outside it. A fixture-center heuristic can
            # otherwise send an accessible start around the far side of a
            # large cabinet before re-entering the same aperture.
            candidate_is_exterior = bool(
                candidate_exterior_projection_m is not None
                and candidate_exterior_projection_m >= 0.06
            )
            if (
                fixture is not None and region_center.shape == (3,)
                and np.isfinite(region_center).all() and not candidate_is_exterior
            ):
                outward = region_center - np.asarray(
                    fixture["center_world_m"], dtype=np.float64,
                )
                outward[2] = 0.0
                outward_norm = float(np.linalg.norm(outward))
                if outward_norm >= 0.04:
                    route_entry_direction = -outward / outward_norm
                    aperture_fixture_object_id = str(fixture.get("object_id"))
                    aperture_entry_direction_selection = {
                        "source": "nearest_intersecting_fixture_center_to_native_region_center",
                        "fixture_object_id": fixture.get("object_id"),
                        "fixture_center_world_m": fixture["center_world_m"],
                        "native_region_center_world_m": region_center.tolist(),
                        "outward_direction_world": (outward / outward_norm).tolist(),
                        "inward_direction_world": route_entry_direction.tolist(),
                        "ranked_layout_exterior_projection_m": candidate_exterior_projection_m,
                        "ranked_layout_preserved_because_live_start_is_exterior": candidate_is_exterior,
                        "physical_entry_direction_is_a_hypothesis_not_a_safety_certificate": True,
                    }
            aperture_entry_direction_selection.update({
                "ranked_layout_exterior_projection_m": candidate_exterior_projection_m,
                "ranked_layout_preserved_because_live_start_is_exterior": candidate_is_exterior,
            })
        # Resolve the fixture owning the native target region before either
        # contact-recovery pass uses it. Exterior-ranked layouts can leave
        # aperture_fixture_object_id unset, while live MuJoCo contacts still
        # need to be matched against the BDDL region's fixture bodies.
        robot_contact_fixture_id = aperture_fixture_object_id
        if robot_contact_fixture_id is None:
            target_region_id = str(predicate[-1])
            for region_suffix in (
                "_heating_region", "_back_contain_region",
                "_front_contain_region", "_contain_region",
            ):
                if target_region_id.endswith(region_suffix):
                    robot_contact_fixture_id = target_region_id[:-len(region_suffix)]
                    break
        horizontal_entry_staging_offset_limit = None
        obstacle_aware_side_detour_offset = None
        obstacle_aware_entry_staging: dict[str, Any] = {
            "status": "not_applicable",
            "blocking_obstacles": [],
            "native_BDDL_remains_success_oracle": True,
        }
        if (
            native_region_entry_mode == "horizontal_aperture"
            and predicate[0].lower() == "in"
            and route_entry_direction is not None
        ):
            inward_xy = np.asarray(route_entry_direction, dtype=np.float64)[:2]
            inward_norm = float(np.linalg.norm(inward_xy))
            footprint_radius = float(
                obstacle_clearance.get("carried_object_footprint_radius_m", 0.0)
            )
            staging_limits: list[float] = []
            side_clearance_offsets: list[float] = []
            blocking_rows: list[dict[str, Any]] = []
            if inward_norm > 1e-8 and footprint_radius > 0.0:
                inward_xy = inward_xy / inward_norm
                for obstacle in obstacle_clearance.get("intersecting_objects", []):
                    if not isinstance(obstacle, dict):
                        continue
                    obstacle_id = str(obstacle.get("object_id", ""))
                    # The target container is the aperture being entered, not
                    # a blocker for choosing the low exterior staging point.
                    if (
                        obstacle_id == target_id
                        or predicate[-1] == obstacle_id
                        or str(predicate[-1]).startswith(obstacle_id + "_")
                    ):
                        continue
                    center = obstacle.get("center_world_m")
                    radius = obstacle.get("horizontal_envelope_radius_m")
                    top = obstacle.get("top_world_m")
                    if (
                        not isinstance(center, list) or len(center) != 3
                        or isinstance(radius, bool) or not isinstance(radius, (int, float))
                        or not np.isfinite(np.asarray(center, dtype=np.float64)).all()
                        or not np.isfinite(float(radius)) or float(radius) <= 0.0
                        or isinstance(top, bool) or not isinstance(top, (int, float))
                        or not np.isfinite(float(top))
                    ):
                        continue
                    if float(top) <= float(goal_object[2, 3]) + 0.015:
                        continue
                    goal_to_obstacle = goal_object[:2, 3] - np.asarray(
                        center[:2], dtype=np.float64,
                    )
                    projection_m = float(np.dot(goal_to_obstacle, inward_xy))
                    if projection_m <= 0.0:
                        continue
                    lateral = goal_to_obstacle - projection_m * inward_xy
                    lateral_distance_m = float(np.linalg.norm(lateral))
                    required_offset = float(radius) + footprint_radius + 0.010
                    if lateral_distance_m >= required_offset:
                        continue
                    staging_limit = projection_m - required_offset
                    row = {
                        "object_id": obstacle_id,
                        "center_world_m": center,
                        "horizontal_envelope_radius_m": float(radius),
                        "carried_object_footprint_radius_m": footprint_radius,
                        "top_world_m": float(top),
                        "lateral_distance_to_entry_axis_m": lateral_distance_m,
                        "required_combined_clearance_radius_m": required_offset,
                        "low_staging_offset_limit_m": staging_limit,
                    }
                    blocking_rows.append(row)
                    side_clearance_offsets.append(required_offset)
                    if staging_limit >= 0.01:
                        staging_limits.append(staging_limit)
            if staging_limits:
                horizontal_entry_staging_offset_limit = min(staging_limits)
            if (
                route_variant in {
                    "keypose_side_region_progress",
                    "keypose_side_region_progress_reverse",
                }
                and side_clearance_offsets
            ):
                obstacle_aware_side_detour_offset = min(
                    0.35, max(side_clearance_offsets),
                )
            obstacle_aware_entry_staging = {
                "status": (
                    "staging_shifted_past_live_corridor_blockers"
                    if horizontal_entry_staging_offset_limit is not None
                    else "no_projected_blocker_requires_staging_shift"
                ),
                "blocking_obstacles": blocking_rows,
                "horizontal_entry_staging_offset_limit_m": horizontal_entry_staging_offset_limit,
                "side_detour_offset_m": obstacle_aware_side_detour_offset,
                "source": "live_compiled_obstacle_envelopes_and_carried_object_footprint",
                "safety_certified": False,
                "native_BDDL_remains_success_oracle": True,
            }
        route_region_rotation = np.asarray(region["site_rotation_world"], dtype=np.float64)
        route_goal_object = goal_object.copy()
        aperture_tilt_report = None
        route_obstacle_clearance_height = obstacle_clearance.get(
            "required_transit_height_world_m"
        )
        if native_region_entry_mode == "horizontal_aperture" and native_region_aperture_tilt_degrees:
            lowest_offsets = []
            route_goal_object, aperture_tilt_report = _native_region_aperture_tilted_goal_pose(
                goal_object,
                region=region,
                target_context=target_context,
                object_to_eef=object_to_eef,
                tilt_degrees=native_region_aperture_tilt_degrees,
            )
            obstacle_top = obstacle_clearance.get(
                "maximum_intersecting_obstacle_top_world_m"
            )
            if isinstance(obstacle_top, (int, float)) and np.isfinite(obstacle_top):
                goal_rotation = Rotation.from_matrix(route_goal_object[:3, :3])
                start_rotation = Rotation.from_matrix(object_world[:3, :3])
                orientation_delta = (goal_rotation * start_rotation.inv()).as_rotvec()
                lowest_offsets = []
                for fraction in np.linspace(0.0, 1.0, 19):
                    sample_rotation = (
                        Rotation.from_rotvec(orientation_delta * float(fraction))
                        * start_rotation
                    ).as_matrix()
                    lowest_offsets.append(_placement_lowest_geometry_offset(
                        target_context, np.eye(3, dtype=np.float64), sample_rotation,
                    ))
                route_obstacle_clearance_height = float(
                    obstacle_top
                    + float(obstacle_clearance.get("clearance_margin_m", 0.05))
                    - min(lowest_offsets)
                )
            aperture_tilt_report["minimum_sampled_transit_lowest_geometry_offset_m"] = (
                min(lowest_offsets) if lowest_offsets else None
            )
            aperture_tilt_report["adjusted_transit_clearance_height_world_m"] = (
                route_obstacle_clearance_height
            )
        horizontal_entry_height_offset = None
        contact_derived_entry_height_clearance_m = 0.0
        if native_region_entry_mode == "horizontal_aperture":
            # Place the object at its exact native support height before sliding
            # through a low front aperture. A positive clearance lift makes the
            # held object's top clear while often driving the larger Panda hand
            # into the cabinet roof; the BDDL goal pose is the lower physical
            # bound that already fits the compiled container geometry.
            horizontal_entry_height_offset = 0.0
            if (
                contact_is_with_target_aperture
                and live_aperture_contact_escape is not None
            ):
                measured_outward = np.asarray(
                    live_aperture_contact_escape.get("direction_world", []),
                    dtype=np.float64,
                )
                if measured_outward.shape == (3,) and np.isfinite(measured_outward).all():
                    # A contact normal with a clear upward escape component is
                    # evidence that the current insertion height catches the
                    # aperture's lower lip. Enter above that lip, then descend
                    # to the native support pose after the object is inside.
                    upward_escape_component = max(0.0, float(measured_outward[2]))
                    if upward_escape_component >= 0.25:
                        contact_derived_entry_height_clearance_m = float(np.clip(
                            0.075 * upward_escape_component, 0.025, 0.06,
                        ))
                        horizontal_entry_height_offset = (
                            contact_derived_entry_height_clearance_m
                        )
            if native_region_entry_height_offset_m is not None:
                horizontal_entry_height_offset = max(
                    float(horizontal_entry_height_offset or 0.0),
                    float(native_region_entry_height_offset_m),
                )
            aperture_entry_direction_selection[
                "contact_derived_entry_height_clearance_m"
            ] = contact_derived_entry_height_clearance_m
            aperture_entry_direction_selection[
                "requested_minimum_entry_height_offset_m"
            ] = native_region_entry_height_offset_m
        route_start_object = object_world
        contact_escape_waypoints: list[dict[str, Any]] = []
        robot_contact_escape_waypoints: list[dict[str, Any]] = []
        robot_contact_escape: dict[str, Any] | None = None
        aperture_hand_extension_m = 0.0
        requested_transit_clearance_mode = native_region_transit_clearance_mode
        if (
            predicate[0].lower() == "in"
            and native_region_transit_clearance_mode not in {"surface_slide", "low_clearance"}
        ):
            # A held object can resume from a legal Snapshot while already
            # touching an open container door or rim.  The native-region
            # route previously ignored those live contacts and immediately
            # replayed a geometric waypoint through the same fixture.  Move
            # the object out along MuJoCo's observed contact normal, add a
            # small vertical clearance, then build the route from that real
            # escaped pose so downstream waypoints do not steer back into
            # the contact that caused the branch.
            live_contacts = (
                live_aperture_contacts
                if live_aperture_contacts is not None
                else bridge.request({"op": "inspect_contacts"})
            )
            contact_escape = _object_contact_escape_direction(
                live_contacts, target_binding, obj["root_pos"],
            )
            if contact_escape is not None:
                measured_normal = np.asarray(
                    contact_escape["direction_world"], dtype=np.float64,
                )
                outward = measured_normal.copy()
                root_position = np.asarray(obj["root_pos"], dtype=np.float64)
                support_radius = float(contact_escape["support_radius_estimate_m"])
                escape_distance = min(0.16, max(0.06, support_radius + 0.04))
                escape_strategy = "measured_contact_normal"
                # If the carried object is resting on a horizontal fixture but
                # the robot itself is not touching that fixture, first lift it
                # clear under the already verified grasp.  Sliding a supported
                # mug sideways across a microwave roof can unload one finger
                # before the later clearance waypoint is ever reached.  Keep
                # the older outward-first sequence for actual robot/fixture
                # contact, where the palm may need to leave the aperture before
                # a vertical motion is physically possible.
                robot_fixture_contact_rows = []
                if robot_contact_fixture_id is not None:
                    for live_contact in live_contacts.get("contacts", []):
                        body1 = str(live_contact.get("body1", {}).get("name", ""))
                        body2 = str(live_contact.get("body2", {}).get("name", ""))
                        robot_is_first = body1.startswith(("robot0_", "gripper0_"))
                        robot_is_second = body2.startswith(("robot0_", "gripper0_"))
                        scene_body = body2 if robot_is_first else body1
                        if (
                            (robot_is_first or robot_is_second)
                            and scene_body.startswith(f"{robot_contact_fixture_id}_")
                        ):
                            robot_fixture_contact_rows.append({
                                "robot_body": body1 if robot_is_first else body2,
                                "fixture_body": scene_body,
                            })
                robot_scene_escape: dict[str, Any] | None = None
                combined_robot_scene_escape = False
                if robot_fixture_contact_rows:
                    contact_state = bridge.request({
                        "op": "observe", "include_snapshot_state": True,
                    }).get("snapshot_state", {})
                    robot_scene_escape = _robot_scene_contact_escape_direction(
                        live_contacts, contact_state,
                        target_body_id=int(target_binding["body_id"]),
                        eef_position_world_m=robot["eef_pos"],
                    )
                    if robot_scene_escape is not None:
                        robot_outward = np.asarray(
                            robot_scene_escape["direction_world"], dtype=np.float64,
                        )
                        robot_outward[2] = 0.0
                        robot_outward_norm = float(np.linalg.norm(robot_outward))
                        if robot_outward_norm > 1e-8:
                            robot_outward /= robot_outward_norm
                            combined_outward = outward + robot_outward
                            combined_norm = float(np.linalg.norm(combined_outward))
                            if combined_norm > 1e-8:
                                outward = combined_outward / combined_norm
                                robot_escape_distance = float(np.clip(
                                    float(robot_scene_escape["contact_radial_extent_m"])
                                    + 0.05, 0.09, 0.16,
                                ))
                                escape_distance = max(
                                    escape_distance, robot_escape_distance,
                                )
                                escape_strategy = (
                                    "combined_object_and_robot_fixture_contact_escape"
                                )
                                combined_robot_scene_escape = True
                lift_supported_object_first = bool(
                    native_region_entry_mode == "horizontal_aperture"
                    and predicate[0].lower() == "in"
                    and float(measured_normal[2]) >= 0.70
                    and not robot_fixture_contact_rows
                )
                support_lift_m = 0.0
                if lift_supported_object_first:
                    required_clearance_height = obstacle_clearance.get(
                        "required_transit_height_world_m"
                    )
                    if isinstance(required_clearance_height, (int, float)) and np.isfinite(
                        float(required_clearance_height)
                    ):
                        support_lift_m = max(
                            0.04,
                            float(required_clearance_height) - float(root_position[2]) + 0.01,
                        )
                    else:
                        support_lift_m = 0.05
                    escape_strategy = "vertical_support_lift_before_horizontal_aperture_escape"
                # A top-face contact can have a perfectly valid upward normal
                # while the held object is trapped under the same fixture roof.
                # For horizontal container entry, use the already selected
                # aperture outward direction when the live normal is transverse
                # to it. Slide the object clear of the fixture first; only then
                # request vertical clearance. The footprint-derived travel
                # distance avoids the near-zero contact-point radius estimate
                # that otherwise reduced this escape to only 60 mm.
                if (
                    native_region_entry_mode == "horizontal_aperture"
                    and predicate[0].lower() == "in"
                    and route_entry_direction is not None
                    and float(np.linalg.norm(route_entry_direction)) > 1e-8
                ):
                    aperture_outward = -np.asarray(
                        route_entry_direction, dtype=np.float64,
                    )
                    aperture_outward[2] = 0.0
                    aperture_norm = float(np.linalg.norm(aperture_outward))
                    if aperture_norm > 1e-8:
                        aperture_outward /= aperture_norm
                        normal_alignment = abs(float(np.dot(
                            measured_normal / max(float(np.linalg.norm(measured_normal)), 1e-12),
                            aperture_outward,
                        )))
                        if (
                            (normal_alignment < 0.5 and not combined_robot_scene_escape)
                            or lift_supported_object_first
                        ):
                            outward = aperture_outward
                            footprint_radius = _placement_footprint_radius(
                                target_context, route_region_rotation,
                                object_world[:3, :3],
                            )
                            escape_distance = min(
                                0.18, max(0.08, float(footprint_radius) + 0.04),
                            )
                            escape_strategy = "horizontal_aperture_outward_before_vertical_clearance"
                escaped_object = object_world.copy()
                escaped_object[:3, 3] = root_position + outward * escape_distance
                if lift_supported_object_first:
                    lifted_object = object_world.copy()
                    lifted_object[:3, 3] = root_position + np.asarray(
                        [0.0, 0.0, support_lift_m], dtype=np.float64,
                    )
                    raised_escape = escaped_object.copy()
                    raised_escape[:3, 3] += np.asarray(
                        [0.0, 0.0, support_lift_m], dtype=np.float64,
                    )
                    contact_escape_waypoints = [
                        {
                            "object_pose_world": lifted_object.tolist(),
                            "role": "target_object_support_lift",
                        },
                        {
                            "object_pose_world": raised_escape.tolist(),
                            "role": "target_object_contact_normal_escape",
                        },
                    ]
                else:
                    raised_escape = escaped_object.copy()
                    raised_escape[:3, 3] += np.asarray([0.0, 0.0, 0.04])
                    contact_escape_waypoints = [
                        {
                            "object_pose_world": escaped_object.tolist(),
                            "role": "target_object_contact_normal_escape",
                        },
                        {
                            "object_pose_world": raised_escape.tolist(),
                            "role": "target_object_contact_vertical_clearance",
                        },
                    ]
                route_start_object = raised_escape
                # Keep the contact-normal escape as a real physical waypoint,
                # but preserve the caller's downstream transit policy.  In
                # reachability-biased mode the old unconditional override to
                # compiled-obstacle height made high held-object waypoints
                # unreachable even after the observed contact had been
                # cleared; the physical branch and native BDDL remain the
                # acceptance test for the lower route.
                contact_escape.update({
                    "status": "contact_normal_waypoints_scheduled",
                    "escape_strategy": escape_strategy,
                    "measured_contact_normal_world": measured_normal.tolist(),
                    "selected_escape_direction_world": outward.tolist(),
                    "escape_displacement_m": escape_distance,
                    "vertical_clearance_m": (
                        support_lift_m if lift_supported_object_first else 0.04
                    ),
                    "supported_object_lifted_before_horizontal_motion": lift_supported_object_first,
                    "live_robot_fixture_contacts": robot_fixture_contact_rows,
                    "robot_scene_contact_escape": robot_scene_escape,
                    "object_and_robot_contacts_share_combined_escape": (
                        combined_robot_scene_escape
                    ),
                    "route_uses_real_OSC_MuJoCo_actions": True,
                    "object_or_robot_state_written_directly": False,
                })
            # A held mug may be clear of the microwave while the Panda palm
            # still lies under its top. If the exact live Snapshot has a
            # robot-to-fixture contact, measure how far the contact point sits
            # behind the wrist along the selected aperture exit direction and
            # physically slide the held object past that footprint before
            # descending. Object-only footprint clearance missed this 90 mm
            # palm extension in the prior L2T4 physical branch.
            if (
                contact_escape is None
                and native_region_entry_mode == "horizontal_aperture"
                and robot_contact_fixture_id is not None
                and route_entry_direction is not None
            ):
                aperture_outward = -np.asarray(
                    route_entry_direction, dtype=np.float64,
                )
                aperture_outward[2] = 0.0
                aperture_norm = float(np.linalg.norm(aperture_outward))
                if aperture_norm > 1e-8:
                    aperture_outward /= aperture_norm
                    robot_contact_rows = []
                    for contact in live_contacts.get("contacts", []):
                        body1 = str(contact.get("body1", {}).get("name", ""))
                        body2 = str(contact.get("body2", {}).get("name", ""))
                        robot_is_first = body1.startswith(("robot0_", "gripper0_"))
                        robot_is_second = body2.startswith(("robot0_", "gripper0_"))
                        scene_body = body2 if robot_is_first else body1
                        if (
                            (robot_is_first or robot_is_second)
                            and scene_body.startswith(f"{robot_contact_fixture_id}_")
                        ):
                            position = np.asarray(contact.get("position"), dtype=np.float64)
                            if position.shape == (3,) and np.isfinite(position).all():
                                inward_extension = max(0.0, -float(np.dot(
                                    position - np.asarray(robot["eef_pos"], dtype=np.float64),
                                    aperture_outward,
                                )))
                                robot_contact_rows.append({
                                    "robot_body": body1 if robot_is_first else body2,
                                    "fixture_body": scene_body,
                                    "contact_position_world_m": position.tolist(),
                                    "inward_projection_from_EEF_m": inward_extension,
                                })
                    if robot_contact_rows:
                        aperture_hand_extension_m = min(
                            0.18,
                            max(0.06, max(
                                row["inward_projection_from_EEF_m"]
                                for row in robot_contact_rows
                            ) + 0.025),
                        )
                        robot_escape_object = route_start_object.copy()
                        robot_escape_object[:3, 3] += (
                            aperture_outward * aperture_hand_extension_m
                        )
                        route_start_object = robot_escape_object
                        robot_contact_escape_waypoints = [{
                            "object_pose_world": robot_escape_object.tolist(),
                            "role": "robot_fixture_aperture_outward_escape",
                        }]
                        robot_contact_escape = {
                            "status": "measured_panda_palm_aperture_escape_scheduled",
                            "fixture_object_id": aperture_fixture_object_id,
                            "outward_direction_world": aperture_outward.tolist(),
                            "measured_hand_extension_m": aperture_hand_extension_m,
                            "margin_m": 0.025,
                            "live_robot_fixture_contacts": robot_contact_rows,
                            "route_uses_real_OSC_MuJoCo_actions": True,
                            "object_or_robot_state_written_directly": False,
                            "safety_certified": False,
                        }
                        # The next route begins at this pose only after the
                        # scheduled physical escape waypoint is actually tracked.
                        aperture_start_contact_free = True
                    else:
                        aperture_start_contact_free = contact_escape is None
            else:
                aperture_start_contact_free = contact_escape is None
        elif predicate[0].lower() == "in":
            # This Stage A branch deliberately keeps support contact and
            # skips safety-style object/robot escape lifts.  The next native
            # region waypoints move the object physically at support height.
            live_contacts = bridge.request({"op": "inspect_contacts"})
            target_geom_names = set(str(x) for x in target_binding.get("geom_names", []))
            other_contact_geometries = sorted({
                str(contact.get("geom2", {}).get("name", ""))
                if str(contact.get("geom1", {}).get("name", "")) in target_geom_names
                else str(contact.get("geom1", {}).get("name", ""))
                for contact in live_contacts.get("contacts", [])
                if (
                    str(contact.get("geom1", {}).get("name", "")) in target_geom_names
                    or str(contact.get("geom2", {}).get("name", "")) in target_geom_names
                )
            })
            contact_escape = {
                "status": (
                    "support_contacts_retained_for_stage_a_surface_slide"
                    if native_region_transit_clearance_mode == "surface_slide"
                    else "minimal_clearance_transit_from_live_support_state"
                ),
                "observed_other_contact_geometries": other_contact_geometries,
                "support_contact_escape_not_scheduled": True,
                "vertical_clearance_not_scheduled": (
                    native_region_transit_clearance_mode == "surface_slide"
                ),
                "object_or_robot_state_written_directly": False,
                "safety_certified": False,
            }
            aperture_start_contact_free = False
            # Stage A surface-slide keeps the carried object's real support
            # contacts, but it must still be able to recover when the Panda
            # palm itself starts intersecting the target container.  Without
            # this measured outward waypoint, OSC repeatedly commands the
            # first insertion/descent target from a physically blocked wrist
            # pose and trips the waypoint-stagnation guard without moving the
            # mug.  Move the held object outward by the observed palm extension
            # plus margin, then build the low-clearance route from that future
            # pose; the escape is accepted only if MuJoCo physically tracks it.
            robot_contact_fixture_id = aperture_fixture_object_id
            if robot_contact_fixture_id is None:
                # Exterior-ranked entry layouts deliberately leave the nearest
                # fixture unset.  The active BDDL region still identifies its
                # owning container, so use that exact object ID to match live
                # MuJoCo body names such as `microwave_1_main` or
                # `desk_caddy_1_main` and retain the same measured-contact rule.
                target_region_id = str(predicate[-1])
                for region_suffix in (
                    "_heating_region", "_back_contain_region",
                    "_front_contain_region", "_contain_region",
                ):
                    if target_region_id.endswith(region_suffix):
                        robot_contact_fixture_id = target_region_id[:-len(region_suffix)]
                        break
            if (
                native_region_entry_mode == "horizontal_aperture"
                and robot_contact_fixture_id is not None
                and route_entry_direction is not None
            ):
                aperture_outward = -np.asarray(
                    route_entry_direction, dtype=np.float64,
                )
                aperture_outward[2] = 0.0
                aperture_norm = float(np.linalg.norm(aperture_outward))
                if aperture_norm > 1e-8:
                    aperture_outward /= aperture_norm
                    robot_contact_rows = []
                    for contact in live_contacts.get("contacts", []):
                        body1 = str(contact.get("body1", {}).get("name", ""))
                        body2 = str(contact.get("body2", {}).get("name", ""))
                        robot_is_first = body1.startswith(("robot0_", "gripper0_"))
                        robot_is_second = body2.startswith(("robot0_", "gripper0_"))
                        scene_body = body2 if robot_is_first else body1
                        if (
                            (robot_is_first or robot_is_second)
                            and scene_body.startswith(f"{robot_contact_fixture_id}_")
                        ):
                            position = np.asarray(contact.get("position"), dtype=np.float64)
                            if position.shape == (3,) and np.isfinite(position).all():
                                inward_extension = max(0.0, -float(np.dot(
                                    position - np.asarray(robot["eef_pos"], dtype=np.float64),
                                    aperture_outward,
                                )))
                                robot_contact_rows.append({
                                    "robot_body": body1 if robot_is_first else body2,
                                    "fixture_body": scene_body,
                                    "contact_position_world_m": position.tolist(),
                                    "inward_projection_from_EEF_m": inward_extension,
                                })
                    if robot_contact_rows:
                        aperture_hand_extension_m = min(
                            0.18,
                            max(0.06, max(
                                row["inward_projection_from_EEF_m"]
                                for row in robot_contact_rows
                            ) + 0.025),
                        )
                        robot_escape_object = object_world.copy()
                        robot_escape_object[:3, 3] += (
                            aperture_outward * aperture_hand_extension_m
                        )
                        route_start_object = robot_escape_object
                        robot_contact_escape_waypoints = [{
                            "object_pose_world": robot_escape_object.tolist(),
                            "role": "robot_fixture_aperture_outward_escape",
                        }]
                        robot_contact_escape = {
                            "status": "measured_panda_palm_aperture_escape_scheduled",
                            "fixture_object_id": robot_contact_fixture_id,
                            "outward_direction_world": aperture_outward.tolist(),
                            "measured_hand_extension_m": aperture_hand_extension_m,
                            "margin_m": 0.025,
                            "live_robot_fixture_contacts": robot_contact_rows,
                            "route_uses_real_OSC_MuJoCo_actions": True,
                            "object_or_robot_state_written_directly": False,
                            "safety_certified": False,
                        }
                        aperture_start_contact_free = True
                    else:
                        object_environment_contacts = [
                            name for name in other_contact_geometries
                            if not name.startswith(("gripper0_", "robot0_"))
                        ]
                        aperture_start_contact_free = not bool(
                            object_environment_contacts
                        )
        else:
            aperture_start_contact_free = False
        if predicate[0].lower() == "on":
            live_contacts = bridge.request({"op": "inspect_contacts"})
        if (
            predicate[0].lower() in {"in", "on"}
            and not robot_contact_escape_waypoints
            and not contact_escape_waypoints
        ):
            # Container-specific aperture escape logic above is not enough
            # for an open-top/over-rim route: a wrist link may already be
            # touching an unrelated bottle or obstacle before the first
            # transport action. Turn that real contact into an initial held-
            # object retreat, preserving the actual grasp and world height.
            contact_state = bridge.request({
                "op": "observe", "include_snapshot_state": True,
            }).get("snapshot_state", {})
            robot_retreat = _robot_scene_contact_escape_direction(
                live_contacts, contact_state,
                target_body_id=int(target_binding["body_id"]),
                eef_position_world_m=robot["eef_pos"],
            )
            if robot_retreat is not None:
                retreat_distance = float(np.clip(
                    float(robot_retreat["contact_radial_extent_m"]) + 0.05,
                    0.09, 0.16,
                ))
                retreat_direction = np.asarray(
                    robot_retreat["direction_world"], dtype=np.float64,
                )
                reorientation_angle = float(np.linalg.norm(
                    (Rotation.from_matrix(route_goal_object[:3, :3])
                     * Rotation.from_matrix(object_world[:3, :3]).inv()).as_rotvec()
                ))
                if reorientation_angle > 0.35:
                    contact_reorientation = object_world.copy()
                    contact_reorientation[:3, :3] = route_goal_object[:3, :3]
                    robot_contact_escape_waypoints.append({
                        "object_pose_world": contact_reorientation.tolist(),
                        "role": "robot_environment_contact_reorientation",
                    })
                robot_escape_object = object_world.copy()
                robot_escape_object[:3, :3] = route_goal_object[:3, :3]
                robot_escape_object[:3, 3] += retreat_direction * retreat_distance
                route_start_object = robot_escape_object
                robot_contact_escape_waypoints.append({
                    "object_pose_world": robot_escape_object.tolist(),
                    "role": "robot_environment_contact_escape",
                })
                robot_contact_escape = {
                    **robot_retreat,
                    "status": "live_robot_scene_contact_retreat_scheduled",
                    "pre_escape_reorientation_angle_rad": reorientation_angle,
                    "pre_escape_reorientation_target_world": (
                        route_goal_object[:3, :3].tolist()
                        if reorientation_angle > 0.35 else None
                    ),
                    "escape_distance_m": retreat_distance,
                    "source_object_pose_world": object_world.tolist(),
                    "retreat_object_pose_world": robot_escape_object.tolist(),
                    "preserves_current_world_height": True,
                    "route_uses_real_OSC_MuJoCo_actions": True,
                    "object_or_robot_state_written_directly": False,
                }

        route_entry_clearance_radius_m = _placement_footprint_radius(
            target_context, route_region_rotation, goal_object[:3, :3],
        ) + aperture_hand_extension_m
        horizontal_entry_minimum_offset_m = None
        horizontal_aperture_fixture_staging: dict[str, Any] = {
            "status": "not_applicable",
            "native_BDDL_remains_success_oracle": True,
        }
        if (
            native_region_entry_mode == "horizontal_aperture"
            and predicate[0].lower() == "in"
            and route_entry_direction is not None
            and robot_contact_fixture_id is not None
        ):
            fixture_row = next((
                row for row in obstacle_clearance.get("intersecting_objects", [])
                if isinstance(row, dict)
                and str(row.get("object_id")) == robot_contact_fixture_id
            ), None)
            fixture_center = np.asarray(
                fixture_row.get("center_world_m"), dtype=np.float64,
            ) if isinstance(fixture_row, dict) else np.empty((0,), dtype=np.float64)
            fixture_radius = (
                float(fixture_row["horizontal_envelope_radius_m"])
                if isinstance(fixture_row, dict)
                and isinstance(fixture_row.get("horizontal_envelope_radius_m"), (int, float))
                else None
            )
            outward_world = -np.asarray(route_entry_direction, dtype=np.float64)
            outward_world[2] = 0.0
            outward_norm = float(np.linalg.norm(outward_world))
            if (
                fixture_center.shape == (3,) and np.isfinite(fixture_center).all()
                and fixture_radius is not None and np.isfinite(fixture_radius)
                and outward_norm > 1e-8
            ):
                outward_world /= outward_norm
                goal_to_fixture = goal_object[:2, 3] - fixture_center[:2]
                fixture_projection = float(np.dot(goal_to_fixture, outward_world[:2]))
                fixture_lateral_distance = float(np.linalg.norm(
                    goal_to_fixture - fixture_projection * outward_world[:2]
                ))
                object_footprint_radius = float(
                    obstacle_clearance.get("carried_object_footprint_radius_m", 0.0)
                )
                eef_horizontal_offset = float(
                    obstacle_clearance.get("object_to_eef_horizontal_offset_m", 0.0)
                )
                fixture_plus_carried_radius = (
                    fixture_radius + object_footprint_radius
                    + eef_horizontal_offset + 0.02
                )
                fixture_envelope_clearance_offset_m = 0.0
                if fixture_lateral_distance < fixture_plus_carried_radius:
                    fixture_envelope_clearance_offset_m = max(
                        0.0,
                        -fixture_projection + float(np.sqrt(max(
                            0.0,
                            fixture_plus_carried_radius ** 2
                            - fixture_lateral_distance ** 2,
                        ))),
                    )
                region_position_for_entry = np.asarray(
                    region.get("site_position_world_m"), dtype=np.float64,
                )
                region_rotation_for_entry = np.asarray(
                    region.get("site_rotation_world"), dtype=np.float64,
                )
                region_size_for_entry = np.asarray(
                    region.get("site_size_m"), dtype=np.float64,
                )
                goal_local_for_entry = region_rotation_for_entry.T @ (
                    goal_object[:3, 3] - region_position_for_entry
                )
                inward_local_for_entry = region_rotation_for_entry.T @ route_entry_direction
                inward_local_for_entry[2] = 0.0
                inward_local_for_entry[:2] /= max(
                    float(np.linalg.norm(inward_local_for_entry[:2])), 1e-12,
                )
                boundary_distances_for_entry = [
                    float(
                        region_size_for_entry[index]
                        - np.sign(inward_local_for_entry[index])
                        * goal_local_for_entry[index]
                    ) / abs(float(inward_local_for_entry[index]))
                    for index in range(2)
                    if abs(float(inward_local_for_entry[index])) > 1e-8
                ]
                native_boundary_distance_m = max(
                    0.0, min(boundary_distances_for_entry),
                ) if boundary_distances_for_entry else 0.0
                native_entry_offset_m = (
                    native_boundary_distance_m
                    + route_entry_clearance_radius_m + 0.02
                )
                # The owning fixture's compiled envelope encloses its entire
                # cabinet and therefore overlaps every valid aperture entry.
                # Applying that whole-body radius to an In opening sent the
                # mug 0.61 m outside the microwave, then required a long sweep
                # back through the same cabinet. Stage A permits fixture
                # contact; stage the carried object just beyond the native
                # aperture boundary and use measured live contacts to resolve
                # any real collision during entry.
                horizontal_entry_minimum_offset_m = native_entry_offset_m
                horizontal_aperture_fixture_staging = {
                    "status": "native_aperture_and_carried_footprint_staging",
                    "fixture_object_id": robot_contact_fixture_id,
                    "fixture_center_world_m": fixture_center.tolist(),
                    "fixture_horizontal_envelope_radius_m": fixture_radius,
                    "carried_object_footprint_radius_m": object_footprint_radius,
                    "object_to_eef_horizontal_offset_m": eef_horizontal_offset,
                    "additional_geometry_margin_m": 0.02,
                    "goal_to_fixture_outward_projection_m": fixture_projection,
                    "goal_to_fixture_lateral_distance_m": fixture_lateral_distance,
                    "required_fixture_clearance_radius_m": fixture_plus_carried_radius,
                    "whole_fixture_envelope_clearance_offset_m": fixture_envelope_clearance_offset_m,
                    "native_boundary_distance_m": native_boundary_distance_m,
                    "native_region_entry_clearance_radius_m": route_entry_clearance_radius_m,
                    "native_region_only_entry_offset_m": native_entry_offset_m,
                    "selected_minimum_entry_offset_m": horizontal_entry_minimum_offset_m,
                    "fixture_envelope_used_to_stage_around_entire_fixture": False,
                    "entry_offset_uses_native_aperture_and_carried_footprint": True,
                    "direct_state_write": False,
                    "native_BDDL_remains_success_oracle": True,
                    "safety_certified": False,
                }
        region_progress_route_build_options = {
            "route_variant": route_variant,
            "keypose_side_region": keypose_side_reference_region,
            "obstacle_clearance_height_world_m": route_obstacle_clearance_height,
            "obstacle_clearance_geometry": obstacle_clearance,
            "goal_region_object_id": (
                str(predicate[2]).rsplit("_contain_region", 1)[0]
                if len(predicate) > 2 and "_contain_region" in str(predicate[2])
                else None
            ),
            "transit_height_override_world_m": native_region_transit_height_override_m,
            "transit_clearance_mode": requested_transit_clearance_mode,
            "enable_entry_alignment": (
                predicate[0].lower() == "in" and route_entry_direction is None
            ),
            "entry_direction_world": route_entry_direction,
            "entry_clearance_radius_m": route_entry_clearance_radius_m,
            "entry_mode": native_region_entry_mode,
            "horizontal_entry_height_offset_m": horizontal_entry_height_offset,
            "horizontal_entry_staging_offset_limit_m": horizontal_entry_staging_offset_limit,
            "horizontal_entry_minimum_offset_m": horizontal_entry_minimum_offset_m,
            "side_detour_offset_m": obstacle_aware_side_detour_offset,
            "preclearance_side_escape": (
                not bool(robot_contact_escape_waypoints)
                and requested_transit_clearance_mode != "low_clearance"
            ),
            "aperture_start_contact_free": aperture_start_contact_free,
        }
        region_progress_route = _native_region_progress_route(
            route_start_object, route_goal_object, region=region,
            **region_progress_route_build_options,
        )
        if contact_escape_waypoints:
            region_progress_route["target_object_contact_escape"] = contact_escape
        if contact_escape_waypoints or robot_contact_escape_waypoints:
            region_progress_route["waypoints"] = (
                contact_escape_waypoints
                + robot_contact_escape_waypoints
                + region_progress_route["waypoints"]
            )
        if robot_contact_escape is not None:
            region_progress_route["target_robot_contact_escape"] = robot_contact_escape
        region_progress_route["requested_transit_clearance_mode"] = (
            native_region_transit_clearance_mode
        )
        if aperture_tilt_report is not None:
            aperture_tilt_report["original_goal_object_world_pose"] = goal_object.tolist()
            aperture_tilt_report["aperture_goal_object_world_pose"] = route_goal_object.tolist()
            region_progress_route["aperture_posture"] = aperture_tilt_report
        region_progress_route["aperture_entry_direction_selection"] = aperture_entry_direction_selection
        region_progress_route["obstacle_aware_entry_staging"] = obstacle_aware_entry_staging
        region_progress_route["compiled_obstacle_clearance"] = obstacle_clearance
        region_progress_route["horizontal_aperture_fixture_staging"] = (
            horizontal_aperture_fixture_staging
        )
        region_progress_route["contact_derived_entry_height_clearance_m"] = (
            contact_derived_entry_height_clearance_m
        )
        route_rows = [
            np.r_[robot["eef_pos"], _wxyz(eef_world[:3, :3])]
        ]
        for row in region_progress_route["waypoints"]:
            eef_pose = np.asarray(row["object_pose_world"], dtype=np.float64) @ object_to_eef
            route_rows.append(np.r_[eef_pose[:3, 3], _wxyz(eef_pose[:3, :3])])
        route = np.asarray(route_rows, dtype=np.float64)
        transport_route_source = "native_region_progress_object_relative_short_segment_OSC"
    else:
        if planned.get("live_state_unchanged") is not True:
            raise RuntimeError("native_goal_transport_RRT_modified_live_state")
        return archive_failure(
            "route_preflight",
            RuntimeError(
                "native_goal_transport_RRT_not_feasible_before_OSC:"
                f"{planned.get('failure_reason') or planned.get('status')}"
            ),
            "jointspace_RRT_preflight_rejected",
        )
    if (
        not resume_at_release and keypose_sequence is None
        and route_strategy != "native_region_progress_osc"
        and route_strategy != "jointspace_rrt"
    ):
        live_contacts = bridge.request({"op": "inspect_contacts"})
        contact_escape = _object_contact_escape_direction(
            live_contacts, target_binding, obj["root_pos"],
        )
        if contact_escape is not None:
            outward = np.asarray(contact_escape["direction_world"], dtype=np.float64)
            root_position = np.asarray(obj["root_pos"], dtype=np.float64)
            support_radius = float(contact_escape["support_radius_estimate_m"])
            escape_distance = min(0.25, max(0.06, support_radius + 0.04))
            escaped_object = object_world.copy()
            escaped_object[:3, 3] = root_position + outward * escape_distance
            escaped_eef = escaped_object @ object_to_eef
            start_pose = np.r_[robot["eef_pos"], _wxyz(eef_world[:3, :3])]
            escape_pose = np.r_[escaped_eef[:3, 3], _wxyz(escaped_eef[:3, :3])]
            raised_escape = escape_pose.copy()
            raised_escape[2] += max(0.10, support_radius + 0.05)
            raised_goal = np.r_[high_eef[:3, 3], high_quat]
            raised_goal[2] += max(0.08, support_radius + 0.04)
            route = _compress_eef_route([
                start_pose, escape_pose, raised_escape, raised_goal,
                np.r_[high_eef[:3, 3], high_quat],
            ])
            contact_escape.update({
                "status": "contact_normal_waypoint_scheduled",
                "escape_displacement_m": escape_distance,
                "object_support_radius_estimate_m": support_radius,
                "elevated_route_extra_height_m": max(0.10, support_radius + 0.05),
                "route_uses_real_OSC_MuJoCo_actions": True,
                "object_or_robot_state_written_directly": False,
            })
            transport_route_source = "target_object_contact_normal_escape_then_elevated_OSC_waypoints"
    # The RRT or elevated OSC route ends above the selected side entrance.
    # Move laterally to a high point above the native region before descending
    # to the physical object target pose. This keeps On routes from repeatedly
    # pushing a held object through the same blocked vertical approach.
    if (
        not resume_at_release and keypose_sequence is None
        and route_strategy != "native_region_progress_osc"
    ):
        if container_entry_transit is not None:
            target_high = (
                container_entry_transit["inside_alignment_object_world"] @ object_to_eef
            )
        else:
            target_high = goal_eef.copy()
            target_high[:3, 3] += np.asarray([0.0, 0.0, 0.14])
        route_with_final_approach = np.vstack((
            route,
            np.r_[target_high[:3, 3], _wxyz(target_high[:3, :3])],
            np.r_[goal_eef[:3, 3], _wxyz(goal_eef[:3, :3])],
        ))
        route = (
            route_with_final_approach
            if route_strategy == "jointspace_rrt"
            else _compress_eef_route(route_with_final_approach)
        )
    transport_budget_details = (
        {"actions": 0, "termination_policy": "resume_at_release"}
        if resume_at_release else _native_transport_action_budget(
            route,
            bridge.request({"op": "native_task_status"}),
            predicate=predicate,
            demonstration_horizon=demonstration_horizon,
            native_region_route_includes_final_placement=(
                route_strategy == "native_region_progress_osc"
            ),
        )
    )
    transport_action_budget = int(transport_budget_details["actions"])

    try:
        if route_strategy == "native_region_progress_osc" and not resume_at_release:
            if region_progress_route is None:
                raise RuntimeError("native_region_progress_route_was_not_generated")
            region_position = np.asarray(region["site_position_world_m"], dtype=np.float64)
            region_rotation = np.asarray(region["site_rotation_world"], dtype=np.float64)
            goal_local = np.asarray(region_progress_route["goal_region_local_m"], dtype=np.float64)
            runtime_contact_escape_count = 0
            flat_distance = lambda position: float(np.linalg.norm(
                (region_rotation.T @ (np.asarray(position, dtype=np.float64) - region_position))[:2]
                - goal_local[:2]
            ))
            for waypoint_index, waypoint_row in enumerate(region_progress_route["waypoints"]):
                live = bridge.request({"op": "observe", "include_snapshot_state": True})
                live_state = live["snapshot_state"]
                live_object = _object_state(live_state, int(target_binding["body_id"]))
                live_robot = live_state["robot_states"][0]
                live_object_world = np.eye(4, dtype=np.float64)
                live_object_world[:3, :3] = Rotation.from_quat(
                    np.asarray(live_object["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
                ).as_matrix()
                live_object_world[:3, 3] = live_object["root_pos"]
                live_eef_world = np.eye(4, dtype=np.float64)
                live_eef_world[:3, :3] = Rotation.from_quat(
                    np.asarray(live_robot["eef_quat"], dtype=np.float64)[[1, 2, 3, 0]]
                ).as_matrix()
                live_eef_world[:3, 3] = live_robot["eef_pos"]
                live_object_to_eef = np.linalg.inv(live_object_world) @ live_eef_world

                def target_eef_pose_provider(
                    desired_object_pose: np.ndarray,
                    *,
                    track_live_object_position: bool = False,
                    track_live_object_height: bool = False,
                    track_live_grasp_transform: bool = False,
                ):
                    desired = np.asarray(desired_object_pose, dtype=np.float64).copy()
                    # The default holds the measured grasp transform fixed for
                    # a waypoint. Object-centric feedback can opt into a
                    # receding transform estimate when the live object moves
                    # relative to the EEF despite bilateral finger contacts.
                    waypoint_object_to_eef = live_object_to_eef.copy()

                    def provide(snapshot_state: dict[str, Any]):
                        live_object = _object_state(
                            snapshot_state, int(target_binding["body_id"]),
                        )
                        object_pose = np.eye(4, dtype=np.float64)
                        object_pose[:3, :3] = Rotation.from_quat(
                            np.asarray(live_object["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
                        ).as_matrix()
                        object_pose[:3, 3] = live_object["root_pos"]
                        live_desired = desired.copy()
                        if track_live_object_position:
                            live_desired[:3, 3] = object_pose[:3, 3]
                        elif track_live_object_height:
                            # Horizontal aperture transit may physically lower
                            # the carried mug as the Panda moves toward its
                            # exterior workspace. Let this lateral waypoint
                            # follow that measured height; a later explicit
                            # descent/entry waypoint controls altitude.
                            live_desired[2, 3] = object_pose[2, 3]
                        object_to_eef = waypoint_object_to_eef
                        if track_live_grasp_transform:
                            live_robot = snapshot_state["robot_states"][0]
                            live_eef_pose = np.eye(4, dtype=np.float64)
                            live_eef_pose[:3, :3] = Rotation.from_quat(
                                np.asarray(live_robot["eef_quat"], dtype=np.float64)[[1, 2, 3, 0]]
                            ).as_matrix()
                            live_eef_pose[:3, 3] = live_robot["eef_pos"]
                            object_to_eef = np.linalg.inv(object_pose) @ live_eef_pose
                        target_pose = live_desired @ object_to_eef
                        return target_pose[:3, 3], _wxyz(target_pose[:3, :3])

                    return provide

                target_object_world = np.asarray(
                    waypoint_row["object_pose_world"], dtype=np.float64,
                ).copy()
                role = str(waypoint_row["role"])
                distance_before = flat_distance(live_object_world[:3, 3])
                contact_escape_completion_evidence: dict[str, Any] = {
                    "status": "not_applicable",
                }
                contact_escape_stop_condition = None
                if role == "target_object_contact_normal_escape":
                    escape_report = region_progress_route.get(
                        "target_object_contact_escape", {},
                    )
                    blocker_geom_names = {
                        str(name) for name in escape_report.get("blocker_geom_names", [])
                    }
                    target_geom_names = {
                        str(name) for name in target_binding.get("geom_names", [])
                    }
                    escape_direction = np.asarray(
                        escape_report.get(
                            "selected_escape_direction_world",
                            escape_report.get("direction_world", []),
                        ),
                        dtype=np.float64,
                    )
                    if (
                        blocker_geom_names
                        and target_geom_names
                        and escape_direction.shape == (3,)
                        and np.isfinite(escape_direction).all()
                        and float(np.linalg.norm(escape_direction)) > 1e-8
                    ):
                        escape_direction /= np.linalg.norm(escape_direction)
                        escape_start_position = live_object_world[:3, 3].copy()
                        planned_escape_distance = float(
                            escape_report.get("escape_displacement_m", 0.0)
                        )
                        minimum_outward_displacement_m = max(
                            0.03, min(0.65 * planned_escape_distance,
                                      planned_escape_distance - 0.01),
                        )

                        def contact_escape_is_physically_clear() -> bool:
                            if not records:
                                return False
                            last_record = records[-1]
                            snapshot_state = last_record.get("snapshot_state", {})
                            try:
                                current_object = _object_state(
                                    snapshot_state, int(target_binding["body_id"]),
                                )
                                current_position = np.asarray(
                                    current_object["root_pos"], dtype=np.float64,
                                )
                            except (KeyError, TypeError, ValueError):
                                return False
                            outward_displacement = float(np.dot(
                                current_position - escape_start_position,
                                escape_direction,
                            ))
                            contacts = last_record.get("contacts", {})
                            blocking_pair_is_live = False
                            if isinstance(contacts, dict):
                                for contact in contacts.get("contacts", []):
                                    geom1 = str(contact.get("geom1", {}).get("name", ""))
                                    geom2 = str(contact.get("geom2", {}).get("name", ""))
                                    if (
                                        geom1 in target_geom_names
                                        and geom2 in blocker_geom_names
                                    ) or (
                                        geom2 in target_geom_names
                                        and geom1 in blocker_geom_names
                                    ):
                                        blocking_pair_is_live = True
                                        break
                            enough_clearance = (
                                outward_displacement >= minimum_outward_displacement_m
                            )
                            contact_escape_completion_evidence.update({
                                "status": (
                                    "live_blocker_contact_cleared_after_outward_displacement"
                                    if enough_clearance and not blocking_pair_is_live
                                    else "blocker_contact_or_minimum_escape_displacement_remains"
                                ),
                                "blocking_pair_is_live": blocking_pair_is_live,
                                "outward_displacement_m": outward_displacement,
                                "minimum_outward_displacement_m": minimum_outward_displacement_m,
                                "blocker_geom_names": sorted(blocker_geom_names),
                                "measured_from_physical_step_record": True,
                                "native_BDDL_remains_success_oracle": True,
                            })
                            return enough_clearance and not blocking_pair_is_live

                        contact_escape_stop_condition = contact_escape_is_physically_clear
                if (
                    role in {"native_region_entry_approach", "native_region_entry_alignment"}
                    and predicate[0].lower() == "in"
                    and region_progress_route.get("entry_mode") != "horizontal_aperture"
                ):
                    # A resumed object can already be stably held directly over
                    # an open-top region.  Replaying the route's exterior staging
                    # waypoint in that state needlessly pulls the object back out
                    # of the aperture and can break an otherwise valid grasp.
                    try:
                        region_size = np.asarray(region["site_size_m"], dtype=np.float64)
                        object_local = region_rotation.T @ (
                            live_object_world[:3, 3] - region_position
                        )
                        footprint_lower, footprint_upper = _placement_collision_bounds_in_region(
                            target_context,
                            region_rotation,
                            live_object_world[:3, :3],
                        )
                        footprint_lower = np.asarray(footprint_lower, dtype=np.float64)
                        footprint_upper = np.asarray(footprint_upper, dtype=np.float64)
                        current_footprint_lower = object_local + footprint_lower
                        current_footprint_upper = object_local + footprint_upper
                        footprint_fits_aperture = bool(
                            region_size.shape == (3,)
                            and np.all(object_local[:2] >= -region_size[:2])
                            and np.all(object_local[:2] <= region_size[:2])
                            and np.all(current_footprint_lower[:2] >= -region_size[:2])
                            and np.all(current_footprint_upper[:2] <= region_size[:2])
                        )
                        clearance_above_rim_m = float(
                            current_footprint_lower[2] - region_size[2]
                        )
                        target_rotation_error_rad = float(np.linalg.norm(
                            (Rotation.from_matrix(goal_object[:3, :3]).inv()
                             * Rotation.from_matrix(live_object_world[:3, :3])).as_rotvec()
                        ))
                        already_overhead = clearance_above_rim_m >= 0.010
                        posture_already_matches = target_rotation_error_rad <= 0.08
                        skip_exterior_entry_waypoints = bool(
                            footprint_fits_aperture
                            and already_overhead
                            and posture_already_matches
                        )
                        region_progress_history.append({
                            "waypoint_index": waypoint_index,
                            "role": role,
                            "status": (
                                "overhead_aperture_projection_reused"
                                if skip_exterior_entry_waypoints
                                else "overhead_aperture_projection_not_reusable"
                            ),
                            "physical_action_count": 0,
                            "object_region_local_pose_m": object_local.tolist(),
                            "current_object_collision_bounds_region_m": [
                                current_footprint_lower.tolist(),
                                current_footprint_upper.tolist(),
                            ],
                            "region_half_extents_m": region_size.tolist(),
                            "object_bottom_clearance_above_region_top_m": clearance_above_rim_m,
                            "target_rotation_error_rad": target_rotation_error_rad,
                            "footprint_fits_aperture": footprint_fits_aperture,
                            "posture_already_matches": posture_already_matches,
                            "native_BDDL_success_still_required": True,
                        })
                        if skip_exterior_entry_waypoints:
                            continue
                    except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                        # Keep the original physical route unless the geometry
                        # evidence is complete and the live fit is exact.
                        pass
                if (
                    role == "native_region_entry_alignment"
                    and predicate[0].lower() == "in"
                    and region_progress_route.get("entry_mode") != "horizontal_aperture"
                ):
                    # Align the carried object's real XY pose at clearance height
                    # before the final descent into a tight container region.
                    object_distance = distance_before
                    # If the actual carried-object footprint already fits through
                    # the native region aperture, centering it more precisely is
                    # wasted OSC work.  Keep its live pose and continue along the
                    # insertion route; native BDDL remains the only success test.
                    # This matters when the controller can reach a valid in-region
                    # pose but stalls while chasing an unnecessary center point.
                    try:
                        region_size = np.asarray(region["site_size_m"], dtype=np.float64)
                        object_local_xy = region_rotation.T @ (
                            live_object_world[:3, 3] - region_position
                        )
                        footprint_lower, footprint_upper = _placement_collision_bounds_in_region(
                            target_context,
                            region_rotation,
                            live_object_world[:3, :3],
                        )
                        current_footprint_lower = object_local_xy[:2] + footprint_lower[:2]
                        current_footprint_upper = object_local_xy[:2] + footprint_upper[:2]
                        footprint_fits_aperture = bool(
                            region_size.shape == (3,)
                            and np.all(object_local_xy[:2] >= -region_size[:2])
                            and np.all(object_local_xy[:2] <= region_size[:2])
                            and np.all(current_footprint_lower >= -region_size[:2])
                            and np.all(current_footprint_upper <= region_size[:2])
                        )
                        if footprint_fits_aperture:
                            region_progress_history.append({
                                "waypoint_index": waypoint_index,
                                "role": role,
                                "status": "center_alignment_skipped_live_footprint_already_fits_aperture",
                                "goal_distance_m": object_distance,
                                "object_position_world_m": live_object_world[:3, 3].tolist(),
                                "object_region_local_xy_m": object_local_xy[:2].tolist(),
                                "object_collision_bounds_region_m": [
                                    footprint_lower.tolist(), footprint_upper.tolist(),
                                ],
                                "current_object_collision_bounds_region_m": [
                                    current_footprint_lower.tolist(),
                                    current_footprint_upper.tolist(),
                                ],
                                "region_half_extents_m": region_size.tolist(),
                                "physical_action_count": 0,
                                "native_BDDL_success_still_required": True,
                            })
                            continue
                    except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                        # Missing geometry evidence keeps the existing physical
                        # alignment behavior; it never weakens the placement gate.
                        pass
                    for alignment_index in range(3):
                        if object_distance <= 0.012:
                            break
                        entry_object_world = goal_object.copy()
                        entry_local = np.asarray([
                            goal_local[0], goal_local[1],
                            _native_region_local_z_for_world_height(
                                region, goal_local[:2],
                                float(region_progress_route["adaptive_transit_height_world_m"]),
                            ),
                        ], dtype=np.float64)
                        entry_object_world[:3, 3] = region_position + region_rotation @ entry_local
                        entry_eef_world = entry_object_world @ live_object_to_eef
                        action_budget_remaining = max(0, transport_action_budget - counter[0])
                        if action_budget_remaining <= 0:
                            raise RuntimeError("native_region_entry_alignment_transport_budget_exhausted")
                        _advance_pose(
                            bridge, entry_eef_world[:3, 3], _wxyz(entry_eef_world[:3, :3]),
                            f"transport_native_region_entry_alignment_{alignment_index}",
                            "native_region_progress", counter, action_budget_remaining,
                            target_binding, instruction, safety_policy, monitor, records, frames,
                            step_callback=persist_step,
                            stagnation_window_actions=24,
                            position_tolerance_m=0.004,
                            rotation_tolerance_rad=0.04,
                            target_pose_provider=target_eef_pose_provider(entry_object_world),
                            object_progress_target_world=entry_object_world,
                            planning_tracking={
                                "tracking_mode": "native_region_entry_alignment_OSC_feedback",
                                "route_variant": region_progress_route["route_variant"],
                                "waypoint_role": "native_region_entry_alignment",
                                "waypoint_index": alignment_index,
                                "native_goal_predicate": predicate,
                                "object_goal_world_pose": goal_object.tolist(),
                                "object_goal_distance_before_m": object_distance,
                                "object_goal_distance_metric": "actual_object_xy_distance_in_native_region_frame",
                                "physical_collision_response_enabled": True,
                                "safety_certified": False,
                            },
                            absolute_action_limit=None,
                        )
                        after = bridge.request({"op": "observe", "include_snapshot_state": True})
                        after_state = after["snapshot_state"]
                        after_object = _object_state(after_state, int(target_binding["body_id"]))
                        distance_after = flat_distance(after_object["root_pos"])
                        grasp = bridge.request({
                            "op": "grasp_state",
                            "carried_binding": target_binding,
                            "expected_instruction": instruction,
                        })
                        bilateral = (
                            grasp.get("grasp_confirmed") is True
                            and grasp.get("finger_contacts", {}).get("left") is True
                            and grasp.get("finger_contacts", {}).get("right") is True
                        )
                        region_progress_history.append({
                            "waypoint_index": f"entry_alignment_{alignment_index}",
                            "role": role,
                            "action_count_after": counter[0],
                            "object_position_world_before_m": live_object["root_pos"],
                            "object_position_world_after_m": after_object["root_pos"],
                            "goal_distance_before_m": object_distance,
                            "goal_distance_after_m": distance_after,
                            "goal_distance_progress_m": object_distance - distance_after,
                            "bilateral_grasp_confirmed": bilateral,
                            "native_goal_satisfied": native_goal_satisfied(),
                        })
                        if not bilateral:
                            raise RuntimeError("native_region_entry_alignment_lost_bilateral_grasp")
                        if distance_after > 0.012 and object_distance - distance_after < 0.003:
                            raise RuntimeError("native_region_entry_alignment_object_goal_stagnated")
                        live_state = after_state
                        live_object = after_object
                        live_robot = live_state["robot_states"][0]
                        live_object_world[:3, 3] = live_object["root_pos"]
                        live_object_world[:3, :3] = Rotation.from_quat(
                            np.asarray(live_object["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
                        ).as_matrix()
                        live_eef_world[:3, :3] = Rotation.from_quat(
                            np.asarray(live_robot["eef_quat"], dtype=np.float64)[[1, 2, 3, 0]]
                        ).as_matrix()
                        live_eef_world[:3, 3] = live_robot["eef_pos"]
                        live_object_to_eef = np.linalg.inv(live_object_world) @ live_eef_world
                        object_distance = distance_after
                    if object_distance > 0.012:
                        raise RuntimeError("native_region_entry_alignment_did_not_reach_12mm_pose")
                    continue
                distance_to_goal = None
                if role == "region_progress":
                    distance_to_goal = flat_distance(live_object_world[:3, 3])
                    waypoint_transit_height = (
                        region_progress_route.get(
                            "horizontal_aperture_staged_outward_before_descent", {}
                        ).get("horizontal_staging_height_world_m")
                        if region_progress_route.get("entry_mode") == "horizontal_aperture"
                        else None
                    )
                    if waypoint_transit_height is None:
                        waypoint_transit_height = region_progress_route[
                            "adaptive_transit_height_world_m"
                        ]
                    waypoint_target, waypoint_distance = _native_region_progress_waypoint_target(
                        region,
                        live_object_world,
                        target_object_world,
                        float(waypoint_transit_height),
                    )
                    if waypoint_target is None:
                        continue
                    target_object_world = waypoint_target
                else:
                    waypoint_distance = None
                position_tolerance_m, rotation_tolerance_rad = (
                    _native_region_progress_tracking_tolerances(
                        role,
                        distance_to_goal,
                        native_goal_operator=str(predicate[0]).lower() if predicate else None,
                        clearance_lift_delta_m=(
                            float(abs(target_object_world[2, 3] - live_object_world[2, 3]))
                            if role == "adaptive_clearance" else None
                        ),
                    )
                )
                waypoint_translation_step_limit_m = float(
                    native_region_progress_max_step_m
                )
                if role in {"region_progress", "native_region_placement"}:
                    # A larger bounded step speeds up vertical clearance
                    # recovery, but the same step can unload a fragile grasp
                    # when the carried object accelerates laterally. Keep
                    # horizontal carry and final insertion quasi-static while
                    # allowing the caller's larger step on clearance lifts.
                    waypoint_translation_step_limit_m = min(
                        waypoint_translation_step_limit_m, 0.008,
                    )
                if role in {
                    "native_region_aperture_posture",
                    "robot_environment_contact_reorientation",
                }:
                    # Wrist-posture changes already caused measurable held-mug
                    # slip in physical trials. Keep that quasi-static rotation
                    # at the proven small step while allowing larger bounded
                    # translation targets on the subsequent route segments.
                    waypoint_translation_step_limit_m = min(
                        waypoint_translation_step_limit_m, 0.008,
                    )
                target_eef_world = target_object_world @ live_object_to_eef
                action_budget_remaining = max(0, transport_action_budget - counter[0])
                if action_budget_remaining <= 0:
                    raise RuntimeError("native_region_progress_transport_horizon_budget_exhausted")
                # The short waypoint is bounded by the route-derived remaining
                # budget; OSC feedback stops at the waypoint or on real stagnation.
                segment_steps = min(
                    action_budget_remaining,
                    NATIVE_REGION_PROGRESS_WAYPOINT_ACTION_BUDGET,
                )
                progress_context = {
                    "tracking_mode": "native_region_progress_OSC_feedback",
                    "route_variant": region_progress_route["route_variant"],
                    "waypoint_role": role,
                    "waypoint_index": waypoint_index,
                    "native_goal_predicate": predicate,
                    "native_region_world_pose": {
                        "position_m": region_position.tolist(),
                        "rotation": region_rotation.tolist(),
                    },
                    "object_goal_world_pose": goal_object.tolist(),
                    "object_goal_distance_before_m": distance_before,
                    "object_goal_distance_metric": "actual_object_xy_distance_in_native_region_frame",
                    "active_route_waypoint_object_pose_world": target_object_world.tolist(),
                    "active_route_waypoint_distance_m": waypoint_distance,
                    "active_route_waypoint_transit_height_world_m": (
                        float(waypoint_transit_height)
                        if role == "region_progress" else None
                    ),
                    "horizontal_progress_ignores_live_height_error": bool(
                        role == "robot_fixture_aperture_outward_escape"
                        or (
                            role == "region_progress"
                            and region_progress_route.get(
                                "horizontal_aperture_staged_outward_before_descent", {}
                            ).get("enabled") is True
                        )
                    ),
                    # Keep the carried-object pose authoritative through all
                    # transit waypoints. A frozen object-to-EEF transform can
                    # become stale after a mug or utensil shifts a few mm in a
                    # bilateral grasp; the EEF then tracks a hypothetical pose
                    # while the physical object stops following the route.
                    "grasp_transform_tracking": "live_object_relative_receding",
                    "max_translation_step_m": waypoint_translation_step_limit_m,
                    "soft_keypose_reference_id": (
                        keypose_experience.get("experience_id") if keypose_experience else None
                    ),
                    "keypose_is_a_soft_route_reference": keypose_experience is not None,
                    "fixed_high_preplace_required": False,
                    "RRT_or_IK_timeout_is_terminal": False,
                    "physical_collision_response_enabled": True,
                    "safety_certified": False,
                }

                def native_or_contact_escape_complete() -> bool:
                    if native_goal_satisfied():
                        return True
                    return bool(
                        contact_escape_stop_condition is not None
                        and contact_escape_stop_condition()
                    )

                reached_waypoint = _advance_pose(
                    bridge, target_eef_world[:3, 3], _wxyz(target_eef_world[:3, :3]),
                    f"transport_region_progress_{role}", "native_region_progress",
                    counter, segment_steps, target_binding, instruction,
                    safety_policy, monitor, records, frames,
                    stop_condition=native_or_contact_escape_complete,
                    step_callback=persist_step,
                    stagnation_window_actions=24,
                    position_tolerance_m=position_tolerance_m,
                    rotation_tolerance_rad=rotation_tolerance_rad,
                    planning_tracking=progress_context,
                    max_translation_step_m=waypoint_translation_step_limit_m,
                    # A Stage A surface-slide route deliberately carries the
                    # Keep the Panda gripper actively closed during every
                    # carried-object waypoint. A zero command only holds the
                    # prior actuator target; physical contact branches have
                    # shown it can unload a finger before the route responds.
                    gripper_action=1.0,
                    target_pose_provider=target_eef_pose_provider(
                        target_object_world,
                        track_live_object_position=(role in {
                            "native_region_aperture_posture",
                            "robot_environment_contact_reorientation",
                        }),
                        track_live_object_height=(
                            role == "robot_fixture_aperture_outward_escape"
                            or (
                                role == "region_progress"
                                and region_progress_route.get(
                                    "horizontal_aperture_staged_outward_before_descent", {}
                                ).get("enabled") is True
                            )
                        ),
                        track_live_grasp_transform=True,
                    ),
                    object_progress_target_world=target_object_world,
                    object_progress_horizontal_only=(
                        role == "robot_fixture_aperture_outward_escape"
                        or (
                            role == "region_progress"
                            and region_progress_route.get(
                                "horizontal_aperture_staged_outward_before_descent", {}
                            ).get("enabled") is True
                        )
                    ),
                    object_progress_rotation_only=(role in {
                        "native_region_aperture_posture",
                        "robot_environment_contact_reorientation",
                    }),
                    absolute_action_limit=None,
                )
                after = bridge.request({"op": "observe", "include_snapshot_state": True})
                after_state = after["snapshot_state"]
                after_object = _object_state(after_state, int(target_binding["body_id"]))
                distance_after = flat_distance(after_object["root_pos"])
                grasp = bridge.request({
                    "op": "grasp_state",
                    "carried_binding": target_binding,
                    "expected_instruction": instruction,
                })
                row = {
                    "waypoint_index": waypoint_index,
                    "role": role,
                    "action_count_after": counter[0],
                    "object_position_world_before_m": live_object["root_pos"],
                    "object_position_world_after_m": after_object["root_pos"],
                    "goal_distance_before_m": distance_before,
                    "goal_distance_after_m": distance_after,
                    "goal_distance_progress_m": distance_before - distance_after,
                    "bilateral_grasp_confirmed": grasp.get("grasp_confirmed") is True
                    and grasp.get("finger_contacts", {}).get("left") is True
                    and grasp.get("finger_contacts", {}).get("right") is True,
                    "native_goal_satisfied": native_goal_satisfied(),
                    "controller_reached_waypoint_pose": bool(reached_waypoint),
                    "target_contact_escape_completion": (
                        contact_escape_completion_evidence
                        if role == "target_object_contact_normal_escape" else None
                    ),
                }
                region_progress_history.append(row)
                if not row["bilateral_grasp_confirmed"]:
                    raise RuntimeError("native_region_progress_transport_lost_bilateral_grasp")
                if (
                    role == "robot_environment_contact_reorientation"
                    and not row["native_goal_satisfied"] and records
                ):
                    # Recompute the retreat from post-rotation contacts. A
                    # yawed wrist can release an interlocked link or change
                    # which surface is pressed against the neighboring object;
                    # the original contact vector must not be replayed blindly.
                    last_record = records[-1]
                    retreat = _robot_scene_contact_escape_direction(
                        last_record.get("contacts", {}),
                        last_record.get("snapshot_state", {}),
                        target_body_id=int(target_binding["body_id"]),
                        eef_position_world_m=after_state["robot_states"][0]["eef_pos"],
                    )
                    live_object_world_after = np.eye(4, dtype=np.float64)
                    live_object_world_after[:3, :3] = Rotation.from_quat(
                        np.asarray(after_object["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
                    ).as_matrix()
                    live_object_world_after[:3, 3] = after_object["root_pos"]
                    waypoints = region_progress_route["waypoints"]
                    next_index = waypoint_index + 1
                    escaped_pose = live_object_world_after.copy()
                    if next_index < len(waypoints) and waypoints[next_index].get("role") == "robot_environment_contact_escape":
                        if retreat is None:
                            waypoints.pop(next_index)
                        else:
                            retreat_direction = np.asarray(
                                retreat["direction_world"], dtype=np.float64,
                            )
                            retreat_distance = float(np.clip(
                                float(retreat["contact_radial_extent_m"]) + 0.05,
                                0.09, 0.16,
                            ))
                            escaped_pose[:3, 3] += retreat_direction * retreat_distance
                            waypoints[next_index]["object_pose_world"] = escaped_pose.tolist()
                    elif retreat is not None:
                        retreat_direction = np.asarray(
                            retreat["direction_world"], dtype=np.float64,
                        )
                        retreat_distance = float(np.clip(
                            float(retreat["contact_radial_extent_m"]) + 0.05,
                            0.09, 0.16,
                        ))
                        escaped_pose[:3, 3] += retreat_direction * retreat_distance
                        waypoints.insert(next_index, {
                            "object_pose_world": escaped_pose.tolist(),
                            "role": "robot_environment_contact_escape",
                        })
                    for downstream in waypoints[next_index:]:
                        if downstream.get("role") != "adaptive_clearance":
                            continue
                        lift_pose = np.asarray(
                            downstream["object_pose_world"], dtype=np.float64,
                        )
                        escape_local = region_rotation.T @ (
                            escaped_pose[:3, 3] - region_position
                        )
                        lift_pose[:3, 3] = region_position + region_rotation @ np.asarray([
                            escape_local[0], escape_local[1],
                            _native_region_local_z_for_world_height(
                                region, escape_local[:2],
                                float(region_progress_route[
                                    "adaptive_transit_height_world_m"
                                ]),
                            ),
                        ])
                        downstream["object_pose_world"] = lift_pose.tolist()
                        break
                    route_replan = {
                        "status": (
                            "post_reorientation_live_contact_escape_updated"
                            if retreat is not None else
                            "post_reorientation_live_robot_contacts_cleared"
                        ),
                        "source_waypoint_index": waypoint_index,
                        "source_role": role,
                        "post_reorientation_object_pose_world": live_object_world_after.tolist(),
                        "contact_geometry": retreat,
                        "next_escape_object_pose_world": (
                            escaped_pose.tolist() if retreat is not None else None
                        ),
                        "uses_exact_post_action_MuJoCo_state": True,
                        "object_or_robot_state_written_directly": False,
                        "safety_certified": False,
                    }
                    region_progress_route.setdefault("runtime_route_replans", []).append(
                        route_replan
                    )
                    row["remaining_route_replan"] = route_replan
                if role in {
                    "keypose_side_clearance_escape",
                    "robot_environment_contact_escape",
                    "robot_fixture_aperture_outward_escape",
                } and not row["native_goal_satisfied"] and records:
                    last_record = records[-1]
                    retreat = _robot_scene_contact_escape_direction(
                        last_record.get("contacts", {}),
                        last_record.get("snapshot_state", {}),
                        target_body_id=int(target_binding["body_id"]),
                        eef_position_world_m=after_state["robot_states"][0]["eef_pos"],
                    )
                    if retreat is not None and runtime_contact_escape_count < 2:
                        # If a contact remains after the planned clearance
                        # waypoint, add one measured retreat before lifting.
                        # This replans only the downstream geometry from the
                        # actual held-object state; the expensive task prefix
                        # and already executed route are preserved verbatim.
                        retreat_direction = np.asarray(
                            retreat["direction_world"], dtype=np.float64,
                        )
                        retreat_distance = float(np.clip(
                            float(retreat["contact_radial_extent_m"]) + 0.035,
                            0.06, 0.12,
                        ))
                        live_object_world_after = np.eye(4, dtype=np.float64)
                        live_object_world_after[:3, :3] = Rotation.from_quat(
                            np.asarray(after_object["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
                        ).as_matrix()
                        live_object_world_after[:3, 3] = after_object["root_pos"]
                        escape_object_world = live_object_world_after.copy()
                        escape_object_world[:3, 3] += retreat_direction * retreat_distance
                        escape_waypoint = {
                            "object_pose_world": escape_object_world.tolist(),
                            "role": "robot_environment_contact_escape",
                        }
                        remaining_waypoints = region_progress_route["waypoints"]
                        remaining_waypoints.insert(waypoint_index + 1, escape_waypoint)
                        # Keep the following lift vertically above the new
                        # physically targeted XY. Retain the route's requested
                        # world clearance while respecting the native region
                        # frame when it is not perfectly level.
                        for downstream in remaining_waypoints[waypoint_index + 2:]:
                            if downstream.get("role") != "adaptive_clearance":
                                continue
                            lift_pose = np.asarray(
                                downstream["object_pose_world"], dtype=np.float64,
                            )
                            escape_local = region_rotation.T @ (
                                escape_object_world[:3, 3] - region_position
                            )
                            lift_pose[:3, 3] = region_position + region_rotation @ np.asarray([
                                escape_local[0], escape_local[1],
                                _native_region_local_z_for_world_height(
                                    region, escape_local[:2],
                                    float(region_progress_route[
                                        "adaptive_transit_height_world_m"
                                    ]),
                                ),
                            ])
                            downstream["object_pose_world"] = lift_pose.tolist()
                            break
                        runtime_contact_escape_count += 1
                        route_replan = {
                            "status": "live_robot_scene_contact_escape_inserted_before_clearance_lift",
                            "source_waypoint_index": waypoint_index,
                            "source_role": role,
                            "contact_geometry": retreat,
                            "escape_distance_m": retreat_distance,
                            "escape_object_pose_world": escape_object_world.tolist(),
                            "remaining_robot_contact_escape_replans": 2 - runtime_contact_escape_count,
                            "uses_exact_post_action_MuJoCo_snapshot": True,
                            "object_or_robot_state_written_directly": False,
                            "safety_certified": False,
                        }
                        region_progress_route.setdefault("runtime_route_replans", []).append(
                            route_replan
                        )
                        row["remaining_route_replan"] = route_replan
                if (
                    role == "native_region_aperture_posture"
                    and reached_waypoint
                    and not row["native_goal_satisfied"]
                ):
                    live_object_world_after = np.eye(4, dtype=np.float64)
                    live_object_world_after[:3, :3] = Rotation.from_quat(
                        np.asarray(after_object["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
                    ).as_matrix()
                    live_object_world_after[:3, 3] = after_object["root_pos"]
                    rebuilt_route = _native_region_progress_route(
                        live_object_world_after, route_goal_object,
                        region=region,
                        **region_progress_route_build_options,
                    )
                    remaining_waypoints = rebuilt_route["waypoints"]
                    region_progress_route["waypoints"][waypoint_index + 1:] = (
                        remaining_waypoints
                    )
                    route_replan = {
                        "status": "remaining_route_rebuilt_from_live_aperture_posture",
                        "source_waypoint_index": waypoint_index,
                        "source_role": role,
                        "object_pose_world_after_posture": live_object_world_after.tolist(),
                        "remaining_waypoint_count": len(remaining_waypoints),
                        "native_goal_pose_preserved": route_goal_object.tolist(),
                        "uses_live_MuJoCo_object_pose": True,
                        "object_or_robot_state_written_directly": False,
                    }
                    region_progress_route.setdefault("runtime_route_replans", []).append(
                        route_replan
                    )
                    row["remaining_route_replan"] = route_replan
                # Persist every reached physical waypoint as a legal branch
                # point.  Exterior staging and fixture detours can increase
                # distance to the native goal; that does not make their exact
                # MuJoCo states less useful for changing only the downstream
                # route.  Snapshot serialization is cheaper than repeating a
                # long grasp and transport prefix.
                if reached_waypoint or row["goal_distance_progress_m"] > 0.003:
                    snap = bridge.request({"op": "export_runtime_snapshot"})
                    if snap.get("status") == "complete" and isinstance(snap.get("payload"), bytes):
                        snapshot_path = output_dir / "region_progress_snapshots" / (
                            f"waypoint_{waypoint_index:03d}_{role}.pkl"
                        )
                        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                        snapshot_path.write_bytes(snap["payload"])
                        region_progress_snapshots.append({
                            "waypoint_index": waypoint_index,
                            "role": role,
                            "path": str(snapshot_path),
                            "sha256": hashlib.sha256(snap["payload"]).hexdigest(),
                            "native_goal_distance_m": distance_after,
                        })
                if row["native_goal_satisfied"]:
                    break
        if route_strategy == "native_region_progress_osc" and not resume_at_release:
            route = route[:1]
        for waypoint_index, waypoint in enumerate(route[1:]):
            if (
                native_goal_insertion_rrt is not None
                and container_entry_transit is not None
                and route_strategy == "jointspace_rrt"
                and waypoint_index == len(route) - 2
            ):
                # The exact native placement pose was reached through the
                # separately planned and tracked insertion path below.
                continue
            if keypose_sequence is not None:
                keypose = keypose_sequence[waypoint_index]
                live = bridge.request({"op": "observe", "include_snapshot_state": True})
                live_state_sha = _state_sha(_snapshot_state_vector(live.get("snapshot_state", {})))
                keypose_planner_budget_s = float(
                    placement_candidate.get(
                        "preplace_rrt_search_time_budget_s",
                        min(8.0, rrt_time_budget_s),
                    ) if placement_candidate is not None else min(8.0, rrt_time_budget_s)
                )
                keypose_planner_budget_s = min(keypose_planner_budget_s, rrt_time_budget_s)
                keypose_planner_iterations = int(
                    placement_candidate.get("preplace_rrt_max_iterations", 750)
                    if placement_candidate is not None else 750
                )
                keypose_planner_checks = int(
                    placement_candidate.get("preplace_rrt_max_config_checks", 3000)
                    if placement_candidate is not None else 3000
                )
                keypose_planner_seed = int(
                    placement_candidate.get("preplace_rrt_seed", 74103 + waypoint_index)
                    if placement_candidate is not None else 74103 + waypoint_index
                )
                planner_config = _keypose_rrt_planner_config(
                    seed=keypose_planner_seed,
                    search_time_budget_s=keypose_planner_budget_s,
                    maximum_iterations=keypose_planner_iterations,
                    maximum_config_checks=keypose_planner_checks,
                )
                goal_pose_request = {
                    "position_world_m": waypoint[:3].tolist(),
                    "quaternion_wxyz": waypoint[3:].tolist(),
                }
                route_cache = (
                    keypose_experience.get("_precomputed_keypose_routes", {}).get(keypose["name"])
                    if keypose_experience is not None else None
                )
                route_reuse = _validate_precomputed_rrt_route(
                    route_cache,
                    model_sha256=model_sha,
                    bddl_sha256=str(observed.get("scene_metadata", {}).get("bddl_sha256", "")),
                    start_state_sha256=live_state_sha,
                    goal_pose_world=goal_pose_request,
                    safety_geometry_policy=planning_policy,
                    planner_config=planner_config,
                )
                use_task_success_osc_fallback = bool(
                    keypose.get("name") == "preplace"
                    and placement_candidate is not None
                    and placement_candidate.get("task_success_waypoint_fallback") is True
                    and route_reuse.get("reusable") is not True
                )
                route_source = "live_RRT_replanned_segment"
                if use_task_success_osc_fallback:
                    keypose_plan = {
                        "status": "task_success_waypoint_fallback",
                        "failure_reason": placement_candidate.get(
                            "task_success_waypoint_fallback_reason"
                        ),
                        "live_state_unchanged": True,
                        "RRT_skipped": True,
                        "safety_certified": False,
                    }
                    after_plan = bridge.request({"op": "observe", "include_snapshot_state": True})
                    after_plan_sha = _state_sha(
                        _snapshot_state_vector(after_plan.get("snapshot_state", {}))
                    )
                    plan_state_check = _verify_planner_live_state(
                        keypose_plan, after_plan_sha, live_state_sha,
                    )
                    route_source = "task_success_closed_loop_OSC_waypoint_after_RRT_timeout"
                elif route_reuse.get("reusable") is True:
                    after_plan = bridge.request({"op": "observe", "include_snapshot_state": True})
                    after_plan_sha = _state_sha(
                        _snapshot_state_vector(after_plan.get("snapshot_state", {}))
                    )
                    unchanged = after_plan_sha == live_state_sha
                    if unchanged:
                        keypose_plan = {
                            "status": "offline_feasible",
                            "live_state_unchanged": True,
                            "qpos_path": route_reuse["qpos_path"].tolist(),
                            "eef_reference_poses_world": route_reuse["eef_reference_poses_world"].tolist(),
                            "planning_seconds": route_cache.get("planning_seconds"),
                            "config_checks": route_cache.get("configuration_checks"),
                            "iterations": route_cache.get("iterations"),
                            "route_cache_reused": True,
                        }
                        plan_state_check = _verify_planner_live_state(
                            keypose_plan, after_plan_sha, live_state_sha,
                        )
                        route_source = "exact_preflight_route_cache_reused"
                    else:
                        route_reuse = {
                            "reusable": False,
                            "reason": "route_cache_live_state_changed_after_cache_validation",
                        }
                if route_source not in {
                    "exact_preflight_route_cache_reused",
                    "task_success_closed_loop_OSC_waypoint_after_RRT_timeout",
                }:
                    keypose_plan = bridge.request({
                        "op": "task_success_first_rrt",
                        "expected_instruction": instruction,
                        "model_geometry_sha256": model_sha,
                        "start_state_sha256": live_state_sha,
                        "goal_pose_world": goal_pose_request,
                        "safety_geometry_policy": planning_policy,
                        "planner_config": planner_config,
                        "ik_seed_qpos": _placement_ik_seed_portfolio(live["snapshot_state"]["robot_states"][0]),
                    })
                    after_plan = bridge.request({"op": "observe", "include_snapshot_state": True})
                    after_plan_sha = _state_sha(
                        _snapshot_state_vector(after_plan.get("snapshot_state", {}))
                    )
                    plan_state_check = _verify_planner_live_state(
                        keypose_plan, after_plan_sha, live_state_sha,
                    )
                if not plan_state_check["live_state_unchanged"]:
                    raise RuntimeError(f"keypose_RRT_mutated_live_state:{keypose['name']}")
                route_reuse_report = {
                    key: value for key, value in route_reuse.items()
                    if key not in {"qpos_path", "eef_reference_poses_world"}
                }
                keypose_rrt_runs.append({
                    "keypose": keypose["name"],
                    "source_timestep_after": keypose["source_timestep_after"],
                    "status": keypose_plan.get("status"),
                    "failure_reason": keypose_plan.get("failure_reason"),
                    "planning_seconds": keypose_plan.get("planning_seconds"),
                    "configuration_checks": keypose_plan.get("config_checks"),
                    "iterations": keypose_plan.get("iterations"),
                    "search_time_budget_s": keypose_planner_budget_s,
                    "maximum_iterations": keypose_planner_iterations,
                    "maximum_config_checks": keypose_planner_checks,
                    "seed": keypose_planner_seed,
                    "route_source": route_source,
                    "preflight_ik_candidate_count": (
                        placement_candidate.get("preplace_rrt_ik_candidate_count")
                        if use_task_success_osc_fallback and placement_candidate is not None
                        else None
                    ),
                    "route_reuse_validation": route_reuse_report,
                    "route_cache_identity_sha256": (
                        route_cache.get("identity_sha256") if route_source == "exact_preflight_route_cache_reused" else None
                    ),
                    "minimum_forbidden_clearance_m": keypose_plan.get("minimum_forbidden_clearance_m"),
                    "record_only_diagnostics": True,
                    "carried_object_geometry_included": True,
                    "full_robot_link_geometry_included": True,
                    "live_state_verification": plan_state_check,
                })
                if keypose_plan.get("status") == "offline_feasible" and keypose_plan.get("live_state_unchanged") is True:
                    raw_eef_route = np.asarray(
                        keypose_plan.get("eef_reference_poses_world"), dtype=np.float64,
                    )
                    qpos_path = np.asarray(keypose_plan.get("qpos_path"), dtype=np.float64)
                    if (
                        raw_eef_route.ndim != 2 or raw_eef_route.shape[1:] != (7,)
                        or qpos_path.ndim != 2 or qpos_path.shape[1:] != (7,)
                        or len(raw_eef_route) != len(qpos_path)
                        or not np.isfinite(raw_eef_route).all()
                        or not np.isfinite(qpos_path).all()
                    ):
                        raise RuntimeError(
                            f"keypose_RRT_joint_and_EEF_path_alignment_invalid:{keypose['name']}"
                        )
                    source_path_indices = list(range(len(raw_eef_route)))
                    closed_loop_source_indices = source_path_indices[1:]
                    route_archive = {
                        "schema_version": "gs_opd_rrt_osc_route_archive.v1",
                        "keypose": keypose["name"],
                        "route_source": route_source,
                        "route_cache_identity_sha256": (
                            route_cache.get("identity_sha256")
                            if route_source == "exact_preflight_route_cache_reused" else None
                        ),
                        "model_geometry_sha256": model_sha,
                        "start_state_sha256": live_state_sha,
                        "goal_eef_pose_world": waypoint.tolist(),
                        "planner_status": keypose_plan.get("status"),
                        "planner_seed": keypose_planner_seed,
                        "planning_seconds": keypose_plan.get("planning_seconds"),
                        "qpos_path_rad": qpos_path.tolist(),
                        "raw_eef_reference_poses_world": raw_eef_route.tolist(),
                        "closed_loop_tracking_reference": [
                            {
                                "source_path_index": source_index,
                                "eef_pose_world": raw_eef_route[source_index].tolist(),
                                "rrt_reference_qpos_rad": qpos_path[source_index].tolist(),
                            }
                            for source_index in source_path_indices
                        ],
                        "compression": {
                            "applied": False,
                            "reason": "existing_live_feedback_tracker_consumes_full_RRT_reference_path",
                            "source_path_indices": source_path_indices,
                        },
                        "execution_policy": (
                            "existing_slow_OSC_POSE_feedback_to_each_dense_RRT_path_pose; "
                            "full_RRT_joint_and_EEF_paths_preserved; contacts_recorded_only"
                        ),
                        "safety_certified": False,
                    }
                    route_bytes = json.dumps(
                        route_archive, ensure_ascii=False, separators=(",", ":"),
                    ).encode("utf-8")
                    route_path = output_dir / (
                        f"rrt_osc_route_{len(keypose_rrt_runs) - 1:02d}_{keypose['name']}.json"
                    )
                    route_path.write_bytes(route_bytes)
                    keypose_rrt_runs[-1]["route_archive"] = {
                        "path": str(route_path),
                        "sha256": hashlib.sha256(route_bytes).hexdigest(),
                        "qpos_path_rows": int(len(qpos_path)),
                        "raw_eef_path_rows": int(len(raw_eef_route)),
                        "compressed_waypoint_count": 0,
                        "closed_loop_reference_count": int(len(raw_eef_route)),
                        "closed_loop_source_path_indices": closed_loop_source_indices,
                    }
                    keypose_rrt_runs[-1]["selected_qpos_path_saved"] = True
                    keypose_rrt_runs[-1]["raw_eef_path_saved"] = True
                    keypose_rrt_runs[-1]["execution_tracking_mode"] = (
                        "dense_RRT_EEF_path_existing_slow_closed_loop_feedback"
                    )
                elif keypose_plan.get("status") == "task_success_waypoint_fallback":
                    subroute = np.asarray([waypoint], dtype=np.float64)
                    keypose_rrt_runs[-1]["execution_fallback"] = {
                        "mode": "existing_slow_OSC_POSE_live_feedback_to_task_equivalent_preplace",
                        "RRT_failure_was_only_time_budget_with_valid_IK": True,
                        "physical_collision_response_enabled": True,
                        "contacts_recorded_only": True,
                        "safety_certified": False,
                    }
                else:
                    keypose_rrt_runs[-1]["execution_fallback"] = None
                    raise RuntimeError(
                        "keypose_RRT_not_feasible_before_OSC:"
                        f"{keypose['name']}:{keypose_plan.get('failure_reason') or keypose_plan.get('status')}"
                    )
            else:
                keypose = {"name": "native_goal_transport_waypoint"}
                subroute = np.asarray([waypoint], dtype=np.float64)
            if (
                keypose_sequence is not None
                and keypose_rrt_runs[-1].get("route_source")
                != "task_success_closed_loop_OSC_waypoint_after_RRT_timeout"
            ):
                route_info = keypose_rrt_runs[-1]["route_archive"]
                for source_path_index in closed_loop_source_indices:
                    reference_pose = raw_eef_route[source_path_index]
                    tracking_context = {
                        "tracking_mode": "dense_RRT_EEF_path_existing_slow_closed_loop_feedback",
                        "rrt_archive_path": route_info["path"],
                        "rrt_route_sha256": route_info["sha256"],
                        "keypose": keypose["name"],
                        "rrt_source_path_index": source_path_index,
                        "rrt_reference_eef_pose_world": reference_pose.tolist(),
                        "rrt_reference_qpos_rad": qpos_path[source_path_index].tolist(),
                    }
                    _advance_pose(
                        bridge, reference_pose[:3], reference_pose[3:],
                        f"transport_{keypose['name']}_dense_RRT_path",
                        "native_goal_transport",
                        counter,
                        max(0, transport_action_budget - counter[0]),
                        target_binding,
                        instruction,
                        safety_policy,
                        monitor,
                        records,
                        frames,
                        step_callback=persist_step,
                        stagnation_window_actions=24,
                        position_tolerance_m=0.010,
                        rotation_tolerance_rad=0.10,
                        planning_tracking=tracking_context,
                        absolute_action_limit=None,
                    )
                    keypose_rrt_runs[-1].setdefault(
                        "closed_loop_source_path_indices_executed", []
                    ).append(source_path_index)
                    if counter[0] > transport_action_budget:
                        raise RuntimeError("native_goal_transport_action_budget_exhausted")
            else:
                for subgoal in subroute:
                    fallback_tracking = None
                    if (
                        keypose_sequence is not None
                        and keypose_rrt_runs[-1].get("route_source")
                        == "task_success_closed_loop_OSC_waypoint_after_RRT_timeout"
                    ):
                        fallback_tracking = {
                            "tracking_mode": "existing_slow_OSC_POSE_feedback_task_success_waypoint",
                            "keypose": keypose["name"],
                            "preflight_failure_reason": keypose_rrt_runs[-1].get("failure_reason"),
                            "preflight_ik_candidate_count": keypose_rrt_runs[-1].get(
                                "preflight_ik_candidate_count"
                            ),
                            "target_pose_world": subgoal.tolist(),
                            "RRT_skipped_after_bounded_timeout": True,
                            "physical_collision_response_enabled": True,
                            "safety_certified": False,
                        }
                    rrt_tracking = None
                    if route_strategy == "jointspace_rrt":
                        reference_index = waypoint_index + 1
                        reference_path = np.asarray(
                            planned.get("eef_reference_poses_world", []), dtype=np.float64,
                        )
                        joint_path = np.asarray(planned.get("qpos_path", []), dtype=np.float64)
                        if reference_index < len(reference_path) and reference_index < len(joint_path):
                            rrt_tracking = {
                                "tracking_mode": "native_OSC_closed_loop_to_dense_collision_aware_RRT_EEF_path",
                                "route_archive": planned_route_archive,
                                "rrt_source_path_index": reference_index,
                                "rrt_reference_eef_pose_world": reference_path[reference_index].tolist(),
                                "rrt_reference_qpos_rad": joint_path[reference_index].tolist(),
                                "physical_collision_response_enabled": True,
                                "safety_certified": False,
                            }
                        else:
                            rrt_tracking = {
                                "tracking_mode": "native_OSC_final_task_region_approach_after_RRT_path",
                                "route_archive": planned_route_archive,
                                "target_pose_world": np.asarray(subgoal, dtype=np.float64).tolist(),
                                "physical_collision_response_enabled": True,
                                "safety_certified": False,
                            }
                    _advance_pose(
                        bridge, subgoal[:3], subgoal[3:],
                        f"transport_{keypose['name']}", "native_goal_transport", counter,
                        max(0, transport_action_budget - counter[0]), target_binding,
                        instruction, safety_policy, monitor, records, frames,
                        step_callback=persist_step,
                        stagnation_window_actions=24,
                        position_tolerance_m=0.010,
                        rotation_tolerance_rad=0.10,
                        planning_tracking=fallback_tracking or rrt_tracking,
                        absolute_action_limit=None,
                    )
                    if counter[0] > transport_action_budget:
                        raise RuntimeError("native_goal_transport_action_budget_exhausted")
            if (
                route_strategy == "jointspace_rrt"
                and container_entry_transit is not None
                and waypoint_index == len(route) - 3
            ):
                # The outer RRT ends above the selected container
                # entry. Once the inside-high alignment is reached,
                # plan the final insertion from the exact live carried
                # state instead of driving OSC through the fixture.
                live = bridge.request({"op": "observe", "include_snapshot_state": True})
                live_state = live["snapshot_state"]
                live_object = _object_state(
                    live_state, int(target_binding["body_id"]),
                )
                live_robot = live_state["robot_states"][0]
                live_object_world = np.eye(4, dtype=np.float64)
                live_object_world[:3, :3] = Rotation.from_quat(
                    np.asarray(live_object["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
                ).as_matrix()
                live_object_world[:3, 3] = live_object["root_pos"]
                live_eef_world = np.eye(4, dtype=np.float64)
                live_eef_world[:3, :3] = Rotation.from_quat(
                    np.asarray(live_robot["eef_quat"], dtype=np.float64)[[1, 2, 3, 0]]
                ).as_matrix()
                live_eef_world[:3, 3] = live_robot["eef_pos"]
                live_object_to_eef = np.linalg.inv(live_object_world) @ live_eef_world
                goal_eef = goal_object @ live_object_to_eef
                insertion_start_sha = _state_sha(_snapshot_state_vector(live_state))
                insertion_goal_pose = np.r_[
                    goal_eef[:3, 3], _wxyz(goal_eef[:3, :3]),
                ]
                insertion_plan = bridge.request({
                    "op": "rrt_connect",
                    "expected_instruction": instruction,
                    "model_geometry_sha256": model_sha,
                    "start_state_sha256": insertion_start_sha,
                    "goal_pose_world": {
                        "position_world_m": goal_eef[:3, 3].tolist(),
                        "quaternion_wxyz": _wxyz(goal_eef[:3, :3]).tolist(),
                    },
                    "safety_geometry_policy": planning_policy,
                    "planner_config": {
                        "step_rad": 0.22,
                        "connect_max_steps": 20,
                        "goal_bias": 0.15,
                        "max_iterations": NATIVE_PLACEMENT_RRT_MAX_ITERATIONS,
                        "max_config_checks": NATIVE_PLACEMENT_RRT_MAX_CONFIG_CHECKS,
                        "max_candidate_routes": 2,
                        "search_time_budget_s": rrt_time_budget_s,
                        "shortcut_attempts": 24,
                        "minimum_clearance_m": 0.0,
                        # The final native In predicate decides success in
                        # Stage A; permit contact-capable insertion paths and
                        # audit their real physical execution.
                        "require_safe_configurations": False,
                        "seed": 74104,
                    },
                    "ik_seed_qpos": _placement_ik_seed_portfolio(live_robot),
                })
                after_insertion_plan = bridge.request({
                    "op": "observe", "include_snapshot_state": True,
                })
                after_insertion_sha = _state_sha(_snapshot_state_vector(
                    after_insertion_plan.get("snapshot_state", {}),
                ))
                insertion_state_verification = _verify_planner_live_state(
                    insertion_plan, after_insertion_sha, insertion_start_sha,
                )
                insertion_plan["live_state_verification"] = insertion_state_verification
                native_goal_insertion_rrt = {
                    "status": insertion_plan.get("status"),
                    "failure_reason": insertion_plan.get("failure_reason"),
                    "goal_role": "exact_native_region_placement_from_live_object_to_EEF_transform",
                    "goal_eef_pose_world": insertion_goal_pose.tolist(),
                    "start_state_sha256": insertion_start_sha,
                    "planning_mode": "stage_a_full_geometry_RRT_contact_diagnostics_without_collision_rejection",
                    "planning_seconds": insertion_plan.get("planning_seconds"),
                    "iterations": insertion_plan.get("iterations"),
                    "configuration_checks": insertion_plan.get("config_checks"),
                    "time_budget_s": rrt_time_budget_s,
                    "maximum_configuration_checks": NATIVE_PLACEMENT_RRT_MAX_CONFIG_CHECKS,
                    "carried_object_geometry_included": True,
                    "full_robot_link_geometry_included": True,
                    "live_state_verification": insertion_state_verification,
                    "safety_certified": False,
                }
                if not insertion_state_verification["live_state_unchanged"]:
                    raise RuntimeError("native_goal_insertion_RRT_changed_live_integration_state")
                insertion_eef_path, insertion_qpos_path = (
                    _native_goal_insertion_rrt_path_arrays(insertion_plan)
                )
                insertion_route_record = {
                    "schema_version": "gs_opd_collision_aware_RRT_OSC_route.v1",
                    "route_role": "native_goal_final_insertion",
                    "native_goal_predicate": predicate,
                    "model_geometry_sha256": model_sha,
                    "bddl_sha256": expected_bddl_sha256,
                    "start_state_sha256": insertion_start_sha,
                    "goal_pose_world": {
                        "position_world_m": goal_eef[:3, 3].tolist(),
                        "quaternion_wxyz": _wxyz(goal_eef[:3, :3]).tolist(),
                    },
                    "planner_source": insertion_plan.get("planner_source"),
                    "planner_mode": "stage_a_full_geometry_RRT_contact_diagnostics_without_collision_rejection",
                    "collision_constraint": (
                        "full robot and carried-object geometry checked for diagnostics; "
                        "Stage A route search does not reject collision configurations; "
                        "real MuJoCo contact substeps remain recorded"
                    ),
                    "qpos_path_rad": insertion_qpos_path.tolist(),
                    "eef_reference_poses_world": insertion_eef_path.tolist(),
                    "planning_seconds": insertion_plan.get("planning_seconds"),
                    "configuration_checks": insertion_plan.get("config_checks"),
                    "iterations": insertion_plan.get("iterations"),
                    "safety_certified": False,
                }
                insertion_route_path = (
                    output_dir / "native_goal_insertion_collision_aware_rrt_route.json"
                )
                _json(insertion_route_path, insertion_route_record)
                insertion_route_bytes = insertion_route_path.read_bytes()
                native_goal_insertion_rrt["route_archive"] = {
                    "path": str(insertion_route_path),
                    "sha256": hashlib.sha256(insertion_route_bytes).hexdigest(),
                    "qpos_path_rows": int(len(insertion_qpos_path)),
                    "eef_path_rows": int(len(insertion_eef_path)),
                    "safety_certified": False,
                }
                for source_path_index in range(1, len(insertion_eef_path)):
                    reference_pose = insertion_eef_path[source_path_index]
                    tracking_context = {
                        "tracking_mode": "native_OSC_closed_loop_to_dense_collision_aware_insertion_RRT_EEF_path",
                        "route_archive": native_goal_insertion_rrt["route_archive"],
                        "rrt_source_path_index": source_path_index,
                        "rrt_reference_eef_pose_world": reference_pose.tolist(),
                        "rrt_reference_qpos_rad": insertion_qpos_path[source_path_index].tolist(),
                        "physical_collision_response_enabled": True,
                        "safety_certified": False,
                    }
                    _advance_pose(
                        bridge, reference_pose[:3], reference_pose[3:],
                        "transport_native_goal_insertion_dense_RRT_path",
                        "native_goal_transport", counter,
                        max(0, transport_action_budget - counter[0]),
                        target_binding, instruction, safety_policy, monitor,
                        records, frames,
                        stop_condition=native_goal_satisfied,
                        step_callback=persist_step,
                        stagnation_window_actions=24,
                        position_tolerance_m=0.010,
                        rotation_tolerance_rad=0.10,
                        planning_tracking=tracking_context,
                        absolute_action_limit=None,
                    )
                    native_goal_insertion_rrt.setdefault(
                        "closed_loop_source_path_indices_executed", []
                    ).append(source_path_index)
                    if counter[0] > transport_action_budget:
                        raise RuntimeError("native_goal_transport_action_budget_exhausted")
                    if native_goal_satisfied():
                        break
                native_goal_insertion_rrt["execution_status"] = "dense_RRT_path_tracked"
                native_goal_insertion_rrt["safety_certified"] = False

    except Exception as exc:
        _write_recording(output_dir, records, frames)
        failure_snapshot_path = output_dir / "native_goal_transport_failed_runtime_snapshot.pkl"
        snapshot_error = None
        try:
            final_snapshot = bridge.request({"op": "export_runtime_snapshot"})
            if final_snapshot.get("status") == "complete" and isinstance(final_snapshot.get("payload"), bytes):
                failure_snapshot_path.write_bytes(final_snapshot["payload"])
        except Exception as snapshot_exc:
            snapshot_error = f"{type(snapshot_exc).__name__}: {snapshot_exc}"
        try:
            final_native = bridge.request({"op": "native_task_status"})
        except Exception as native_exc:
            final_native = {
                "status": "unavailable_bridge_terminated",
                "error": f"{type(native_exc).__name__}: {native_exc}",
                "last_archived_task_success": records[-1].get("task_success") if records else None,
            }
        outcome = _transport_exception_outcome(exc, final_native)
        return {
            **outcome,
            "transport_failure_reason": f"{type(exc).__name__}: {exc}",
            "native_goal_predicate": predicate,
            "native_task_success": final_native.get("task_success"),
            "final_native_task_status": final_native,
            "transport_route_source": transport_route_source,
            "target_object_contact_escape": contact_escape,
            "region_progress_route": region_progress_route,
            "region_progress_feedback": region_progress_history,
            "region_progress_snapshots": region_progress_snapshots,
            "native_goal_insertion_RRT": native_goal_insertion_rrt,
            "transport_route_RRT": {
                "status": "not_run" if planned.get("planner_skipped") else planned.get("status"),
                "route_selection_status": planned.get("status"),
                "failure_reason": planned.get("failure_reason"),
                "time_budget_s": rrt_time_budget_s,
                "maximum_configuration_checks": NATIVE_PLACEMENT_RRT_MAX_CONFIG_CHECKS,
                "safety_certified": False,
                "live_state_verification": planner_state_verification,
            },
            "keypose_RRT_waypoints": keypose_rrt_runs,
            "keypose_transfer": {
                "enabled": keypose_experience is not None,
                "experience_id": keypose_experience.get("experience_id") if keypose_experience else None,
                "route_start_after": keypose_experience.get("_route_start_after") if keypose_experience else None,
                "retargeted_keyposes": keypose_sequence,
                "per_waypoint_planning": keypose_rrt_runs,
                "source_actions_replayed": False,
                "safety_certified": False,
            },
            "transport_actions": len(records),
            "transport_action_budget": transport_action_budget,
            "transport_action_budget_details": transport_budget_details,
            "physics_substeps": int(sum(
                item.get("contact_monitoring", {}).get("captured_physics_substeps", 0)
                for item in records
            )),
            "start_snapshot": str(start_snapshot_path),
            "terminal_snapshot": str(failure_snapshot_path) if failure_snapshot_path.is_file() else None,
            "terminal_snapshot_export_error": snapshot_error,
            "bridge_process_returncode": (
                bridge.process.poll() if getattr(bridge, "process", None) is not None else None
            ),
            "artifacts": {
                "records": str(output_dir / "physical_step_records.jsonl.gz"),
                "npz": str(output_dir / "task_success_first_actions_states.npz"),
                "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                "durable_records": str(durable_records_path),
                "durable_frames": str(durable_frames_dir),
            },
            "safety_certified": False,
        }
    evidence, stage = _phase_evidence(
        bridge, controller, target_id, int(target_binding["body_id"]), predicate,
        target_binding=target_binding,
        eef_goal_reached=True,
    )
    controller.update(evidence)
    phase_events = [{"timestep": evidence.timestep, "phase": controller.phase, "native": stage["native_task_status"]}]
    precise_pose_reached = True
    if not resume_at_release:
        try:
            precise_pose_reached = _advance_pose(
                bridge, goal_eef[:3, 3], _wxyz(goal_eef[:3, :3]), "precise_placement",
                "native_goal_place", counter, MAX_NATIVE_PLACEMENT_ACTIONS, target_binding, instruction,
                safety_policy, monitor, records, frames,
                stop_condition=native_goal_satisfied,
                step_callback=persist_step,
                stagnation_window_actions=24,
                absolute_action_limit=None,
            )
        except Exception as exc:
            return archive_failure("precise_placement", exc, transport_route_source)
        if counter[0] > transport_action_budget + MAX_NATIVE_PLACEMENT_ACTIONS:
            return archive_failure(
                "precise_placement", RuntimeError("native_goal_place_total_action_budget_exhausted"),
                transport_route_source,
            )
    evidence, stage = _phase_evidence(
        bridge, controller, target_id, int(target_binding["body_id"]), predicate,
        target_binding=target_binding,
        eef_goal_reached=precise_pose_reached,
    )
    controller.update(evidence)
    phase_events.append({"timestep": evidence.timestep, "phase": controller.phase, "native": stage["native_task_status"]})
    release_limit = (
        MAX_NATIVE_RELEASE_ACTIONS if resume_at_release
        else transport_action_budget + MAX_NATIVE_PLACEMENT_ACTIONS + MAX_NATIVE_RELEASE_ACTIONS
    )
    release_contact_escape_actions = 0
    while (
        counter[0] < release_limit
        and controller.phase == "release"
        and not _native_episode_terminated(stage.get("native_task_status"))
    ):
        release_action = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        grasp_state = stage.get("grasp_state", {})
        finger_contacts = grasp_state.get("finger_contacts", {})
        closure = grasp_state.get("gripper_closure", {})
        close_fraction = closure.get("close_command_fraction")
        if (
            closure.get("status") == "pass"
            and isinstance(close_fraction, (int, float))
            and float(close_fraction) <= 1e-6
            and (finger_contacts.get("left") is True or finger_contacts.get("right") is True)
        ):
            snapshot = stage.get("observation", {}).get("snapshot_state", {})
            robot = snapshot.get("robot_states", [{}])[0]
            obj = _object_state(snapshot, int(target_binding["body_id"]))
            current_eef = np.asarray(robot.get("eef_pos", []), dtype=np.float64)
            current_quat = np.asarray(robot.get("eef_quat", []), dtype=np.float64)
            escape_target = _release_retreat_target(current_eef, obj.get("root_pos", []))
            if escape_target is not None and current_quat.shape == (4,) and np.isfinite(current_quat).all():
                release_action = world_pose_tracking_action(
                    current_eef, current_quat, escape_target, current_quat,
                    controller_fields, gripper_action=-1.0,
                ).astype(np.float32)
                release_contact_escape_actions += 1
        _record_step(
            bridge, release_action,
            "release", "native_goal_place", counter[0], target_binding, instruction,
            safety_policy, monitor, records, frames, raise_on_done=False,
        )
        counter[0] += 1
        persist_latest_step()
        evidence, stage = _phase_evidence(
            bridge, controller, target_id, int(target_binding["body_id"]), predicate,
            target_binding=target_binding,
        )
        before_phase = controller.phase
        try:
            controller.update(evidence)
        except Exception as exc:
            return archive_failure("release", exc, transport_route_source)
        if controller.phase != before_phase:
            phase_events.append({"timestep": evidence.timestep, "phase": controller.phase, "native": stage["native_task_status"]})
        if _native_episode_terminated(stage.get("native_task_status")):
            break
    withdrawal_skipped_after_native_done = bool(
        controller.phase == "withdrawal"
        and _native_episode_terminated(stage.get("native_task_status"))
    )
    if controller.phase == "withdrawal" and not _native_episode_terminated(
        stage.get("native_task_status")
    ):
        current = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]
        withdraw = np.asarray(current["eef_pos"], dtype=np.float64) + np.asarray([0.0, 0.0, 0.10])
        try:
            _advance_pose(
                bridge, withdraw, np.asarray(current["eef_quat"], dtype=np.float64),
                "withdrawal", "native_goal_place", counter, MAX_NATIVE_WITHDRAWAL_ACTIONS, target_binding,
                instruction, safety_policy, monitor, records, frames,
                step_callback=persist_step,
                absolute_action_limit=None,
            )
        except Exception as exc:
            return archive_failure("withdrawal", exc, transport_route_source)
        evidence, stage = _phase_evidence(
            bridge, controller, target_id, int(target_binding["body_id"]), predicate,
            target_binding=target_binding,
            eef_goal_reached=True,
        )
        controller.update(evidence)
        phase_events.append({"timestep": evidence.timestep, "phase": controller.phase, "native": stage["native_task_status"]})
    partial_goal_settled = False
    settle_history: list[tuple[np.ndarray, np.ndarray]] = []
    stage_action_limit = (
        (MAX_NATIVE_RELEASE_ACTIONS if resume_at_release else
         transport_action_budget + MAX_NATIVE_PLACEMENT_ACTIONS + MAX_NATIVE_RELEASE_ACTIONS)
        + MAX_NATIVE_WITHDRAWAL_ACTIONS + MAX_NATIVE_SETTLE_ACTIONS
    )
    while (
        controller.phase == "settle"
        and counter[0] < stage_action_limit
        and not _native_episode_terminated(stage.get("native_task_status"))
    ):
        _record_step(
            bridge, np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
            "settle", "native_goal_place", counter[0], target_binding, instruction,
            safety_policy, monitor, records, frames, raise_on_done=False,
        )
        counter[0] += 1
        persist_latest_step()
        evidence, stage = _phase_evidence(
            bridge, controller, target_id, int(target_binding["body_id"]), predicate,
            target_binding=target_binding,
        )
        if evidence.task_goal_satisfied:
            settle_history.append((
                np.asarray(evidence.object_position_world_m, dtype=np.float64),
                np.asarray(evidence.object_qvel, dtype=np.float64),
            ))
            settle_history = settle_history[-5:]
        else:
            settle_history.clear()
        pending = [
            row.get("predicate") for row in stage["native_task_status"].get("native_goal_values", [])
            if row.get("value") is False and row.get("predicate") != predicate
        ]
        if pending and len(settle_history) == 5:
            positions = np.stack([item[0] for item in settle_history])
            velocities = settle_history[-1][1]
            stable_partial = bool(
                velocities.shape == (6,)
                and np.max(np.linalg.norm(positions - positions[-1], axis=1)) <= 0.004
                and np.linalg.norm(velocities[:3]) <= 0.10
                and np.linalg.norm(velocities[3:]) <= 1.0
            )
            if stable_partial:
                partial_goal_settled = True
                phase_events.append({
                    "timestep": evidence.timestep,
                    "phase": "partial_predicate_settled",
                    "native": stage["native_task_status"],
                    "pending_native_goals": pending,
                })
                break
        before_phase = controller.phase
        try:
            controller.update(evidence)
        except Exception as exc:
            return archive_failure("settle", exc, transport_route_source)
        if controller.phase != before_phase:
            phase_events.append({"timestep": evidence.timestep, "phase": controller.phase, "native": stage["native_task_status"]})
    final_native = bridge.request({"op": "native_task_status"})
    _write_recording(output_dir, records, frames)
    final_snapshot = bridge.request({"op": "export_runtime_snapshot"})
    path = output_dir / "native_goal_placement_terminal_runtime_snapshot.pkl"
    if final_snapshot.get("status") == "complete" and isinstance(final_snapshot.get("payload"), bytes):
        path.write_bytes(final_snapshot["payload"])
    return {
        "status": "complete_task_success" if final_native.get("task_success") is True
        else "partial_predicate_completed" if partial_goal_settled and any(
            row.get("predicate") == predicate and row.get("value") is True
            for row in final_native.get("native_goal_values", [])
        ) else "native_goal_placement_incomplete",
        "native_goal_predicate": predicate,
        "native_goal_value": next((row.get("value") for row in final_native.get("native_goal_values", []) if row.get("predicate") == predicate), None),
        "native_task_success": final_native.get("task_success"),
        "final_native_task_status": final_native,
        "stage_a_near_goal_release_probe": near_goal_release_probe,
        "task_phase_controller_phase": controller.phase,
        "partial_goal_settled": partial_goal_settled,
        "withdrawal_skipped_after_native_done": withdrawal_skipped_after_native_done,
        "next_pending_native_goals": [
            row.get("predicate") for row in final_native.get("native_goal_values", [])
            if row.get("value") is False
        ],
        "phase_events": phase_events,
        "release_contact_escape_actions": release_contact_escape_actions,
        "transport_route_source": transport_route_source,
        "target_object_contact_escape": contact_escape,
        "region_progress_route": region_progress_route,
        "region_progress_feedback": region_progress_history,
        "region_progress_snapshots": region_progress_snapshots,
        "native_goal_insertion_RRT": native_goal_insertion_rrt,
        "transport_route_RRT": {
            "status": "not_run" if planned.get("planner_skipped") else planned.get("status"),
            "route_selection_status": planned.get("status"),
            "failure_reason": planned.get("failure_reason"),
            "time_budget_s": rrt_time_budget_s,
            "planning_seconds": planned.get("planning_seconds"),
            "iterations": planned.get("iterations"),
            "configuration_checks": planned.get("config_checks"),
            "route_candidate_count": len(planned.get("route_candidates", [])),
            "planner_source": planned.get("planner_source"),
            "planning_mode": planned.get("planning_mode"),
            "carried_object_geometry_included": planning_policy.get("carried_object") is not None,
            "maximum_configuration_checks": NATIVE_PLACEMENT_RRT_MAX_CONFIG_CHECKS,
            "ik_seed_count": len(_placement_ik_seed_portfolio(robot)),
            "goal_role": rrt_goal_role,
            "planning_seconds": planned.get("planning_seconds"),
            "iterations": planned.get("iterations"),
            "configuration_checks": planned.get("config_checks"),
            "ik_candidate_count": len(planned.get("ik_candidates", [])),
            "safety_certified": False,
            "live_state_verification": planner_state_verification,
        },
        "transport_route_archive": planned_route_archive,
        "keypose_RRT_waypoints": keypose_rrt_runs,
        "transport_action_budget": transport_action_budget,
        "transport_action_budget_details": transport_budget_details,
        "stage_action_limit": stage_action_limit,
        "recorded_actions": len(records),
        "recorded_physics_substeps": int(sum(
            item.get("contact_monitoring", {}).get("captured_physics_substeps", 0) for item in records
        )),
        "transport_rrt": {
            "status": "not_run" if planned.get("planner_skipped") else planned.get("status"),
            "route_selection_status": planned.get("status"),
            "safety_certified": False,
            "diagnostic_minimum_clearance_m": planned.get("minimum_forbidden_clearance_m"),
        },
        "terminal_snapshot": str(path) if path.is_file() else None,
        "placement_candidate": placement_candidate,
        "keypose_transfer": {
            "enabled": keypose_experience is not None,
            "experience_id": keypose_experience.get("experience_id") if keypose_experience else None,
            "retargeted_keyposes": keypose_sequence,
            "per_waypoint_planning": keypose_rrt_runs,
            "source_actions_replayed": False,
            "safety_certified": False,
        },
        "artifacts": {
            "records": str(output_dir / "physical_step_records.jsonl.gz"),
            "npz": str(output_dir / "task_success_first_actions_states.npz"),
            "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
            "durable_records": str(durable_records_path),
            "durable_frames": str(durable_frames_dir),
        },
    }


def _screen_native_keypose_candidates(
    bridge,
    *,
    target_id: str,
    target_binding: dict[str, Any],
    predicate: list[str],
    instruction: str,
    model_sha: str,
    bddl_sha: str,
    safety_policy: dict[str, Any],
    start_snapshot_state: dict[str, Any],
    start_state_sha256: str,
    region: dict[str, Any],
    geometry_candidates: list[dict[str, Any]],
    experiences: list[dict[str, Any]],
    maximum_rrt_preflights: int,
    maximum_rrt_time_budget_s: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Compare own-success keyposes on one frozen held-object state.

    Candidate placement locations come from the active BDDL region and current
    occupancy. Every staging pose is screened with the full held-object and
    robot geometry RRT. The terminal contact pose must have a valid IK solution;
    its RRT diagnostics are retained because final support contact may be a
    task-success search contact and will be physically executed by bounded OSC.
    """
    from scipy.spatial.transform import Rotation
    from research.gs_opd.repair.keypose_driven_planning import (
        retarget_keypose_sequence,
        select_keypose_suffix,
        update_pose_record,
    )
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _snapshot_state_vector,
        _state_sha,
        _wxyz,
    )

    if not experiences:
        raise ValueError("keypose_candidate_screen_requires_verified_GS_OPD_experiences")
    robot = start_snapshot_state.get("robot_states", [{}])[0]
    obj = _object_state(start_snapshot_state, int(target_binding["body_id"]))
    object_world = np.eye(4, dtype=np.float64)
    object_world[:3, :3] = Rotation.from_quat(
        np.asarray(obj["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
    ).as_matrix()
    object_world[:3, 3] = np.asarray(obj["root_pos"], dtype=np.float64)
    eef_world = np.eye(4, dtype=np.float64)
    eef_world[:3, :3] = Rotation.from_quat(
        np.asarray(robot["eef_quat"], dtype=np.float64)[[1, 2, 3, 0]]
    ).as_matrix()
    eef_world[:3, 3] = np.asarray(robot["eef_pos"], dtype=np.float64)
    object_to_eef = np.linalg.inv(object_world) @ eef_world
    region_world = np.eye(4, dtype=np.float64)
    region_world[:3, :3] = np.asarray(region["site_rotation_world"], dtype=np.float64)
    region_world[:3, 3] = np.asarray(region["site_position_world_m"], dtype=np.float64)

    combinations: list[tuple[dict[str, Any], dict[str, Any], float, float]] = []
    if len(geometry_candidates) >= 4 and len(experiences) >= 2:
        combinations = [
            (geometry_candidates[0], experiences[1], 0.32, -90.0),
            (geometry_candidates[0], experiences[0], 0.38, 180.0),
            (geometry_candidates[3], experiences[1], 0.34, -90.0),
            (geometry_candidates[3], experiences[0], 0.40, 180.0),
        ]
    else:
        bases = geometry_candidates[:min(2, len(geometry_candidates))]
        angles = (-90.0, 180.0, -90.0, 180.0)
        heights = (0.32, 0.38, 0.34, 0.40)
        for base_index, base_candidate in enumerate(bases):
            for experience_index, experience in enumerate(experiences[:2]):
                if len(combinations) >= 4:
                    break
                combinations.append((
                    base_candidate, experience,
                    heights[len(combinations)], angles[len(combinations)],
                ))
        if len(experiences) == 1:
            for base_candidate in bases:
                if len(combinations) >= 4:
                    break
                combinations.append((
                    base_candidate, experiences[0],
                    heights[len(combinations)], angles[len(combinations)],
                ))

    planning_policy = dict(safety_policy)
    if "measure_forbidden_object_pair_clearances" in planning_policy:
        planning_policy["measure_forbidden_object_pair_clearances"] = False
    planning_policy["carried_object"] = {
        "object_id": target_id,
        "grasp_binding": target_binding,
        "allowed_contact_pairs": [],
        "allowed_support_contact_pairs": [],
    }
    preflight_count = 0
    preflight_rows: list[dict[str, Any]] = []
    physical_candidates: list[dict[str, Any]] = []
    experience_by_candidate: dict[str, dict[str, Any]] = {}

    def screen_pose(
        goal_pose: np.ndarray,
        seed: int,
        *,
        search_time_budget_s: float = 12.0,
        maximum_iterations: int = 750,
        maximum_config_checks: int = 3000,
    ) -> dict[str, Any]:
        nonlocal preflight_count
        if preflight_count >= maximum_rrt_preflights:
            return {"status": "budget_exhausted", "failure_class": "RRT_preflight_budget_exhausted"}
        live = bridge.request({"op": "observe", "include_snapshot_state": True})
        live_state_sha = _state_sha(_snapshot_state_vector(live.get("snapshot_state", {})))
        if live_state_sha != start_state_sha256:
            raise RuntimeError("keypose_candidate_screen_start_snapshot_drifted")
        planner_config = _keypose_rrt_planner_config(
            seed=seed,
            search_time_budget_s=min(search_time_budget_s, maximum_rrt_time_budget_s),
            maximum_iterations=maximum_iterations,
            maximum_config_checks=maximum_config_checks,
        )
        goal_pose_request = {
            "position_world_m": goal_pose[:3, 3].tolist(),
            "quaternion_wxyz": _wxyz(goal_pose[:3, :3]).tolist(),
        }
        result = bridge.request({
            "op": "task_success_first_rrt",
            "expected_instruction": instruction,
            "model_geometry_sha256": model_sha,
            "start_state_sha256": live_state_sha,
            "goal_pose_world": goal_pose_request,
            "safety_geometry_policy": planning_policy,
            "planner_config": planner_config,
            "ik_seed_qpos": _placement_ik_seed_portfolio(robot),
        })
        preflight_count += 1
        after = bridge.request({"op": "observe", "include_snapshot_state": True})
        after_sha = _state_sha(_snapshot_state_vector(after.get("snapshot_state", {})))
        unchanged = after_sha == live_state_sha
        if not unchanged:
            raise RuntimeError("keypose_candidate_RRT_screen_modified_live_snapshot")
        valid_ik_count = sum(
            1 for row in result.get("ik_candidates", [])
            if float(row.get("position_error_m", float("inf"))) <= 0.003
            and float(row.get("orientation_error_rad", float("inf"))) <= 0.03
        )
        route_cache = (
            _make_precomputed_rrt_route_cache(
                result=result,
                model_sha256=model_sha,
                bddl_sha256=bddl_sha,
                start_state_sha256=live_state_sha,
                goal_pose_world=goal_pose_request,
                safety_geometry_policy=planning_policy,
                planner_config=planner_config,
            )
            if unchanged else None
        )
        return {
            "status": result.get("status"),
            "failure_reason": result.get("failure_reason"),
            "failure_class": (
                "IK_unreachable"
                if "inverse_kinematics" in str(result.get("failure_reason", ""))
                else "RRT_or_full_geometry_route_unavailable"
                if result.get("status") != "offline_feasible"
                else None
            ),
            "planning_seconds": result.get("planning_seconds"),
            "configuration_checks": result.get("config_checks"),
            "iterations": result.get("iterations"),
            "ik_candidate_count": valid_ik_count,
            "minimum_forbidden_clearance_m": result.get("minimum_forbidden_clearance_m"),
            "live_state_unchanged": unchanged,
            "route_cache": route_cache,
            "full_robot_link_geometry_included": True,
            "held_object_geometry_included": True,
            "safety_certified": False,
        }

    for variant_index, (base_candidate, experience, approach_height_m, yaw_offset_degrees) in enumerate(combinations):
        candidate = dict(base_candidate)
        exp_id = str(experience.get("experience_id", "own_success"))
        candidate["candidate_id"] = (
            f"route_{variant_index:02d}_{base_candidate['candidate_id']}_"
            f"{exp_id.rsplit('_', 1)[-1]}_"
            f"yaw{int(yaw_offset_degrees):+04d}_h{int(round(approach_height_m * 1000)):03d}"
        )
        release = next(
            (row for row in experience.get("keypose_sequence", []) if row.get("name") == "release"),
            None,
        )
        if release is None:
            preflight_rows.append({
                "candidate_id": candidate["candidate_id"],
                "source_experience_id": exp_id,
                "status": "rejected",
                "failure_class": "source_release_keypose_missing",
            })
            continue
        release_local = np.asarray(release["region_to_object"]["matrix"], dtype=np.float64)
        yaw_variant = Rotation.from_euler("z", yaw_offset_degrees, degrees=True).as_matrix()
        candidate["rotation_world"] = (
            region_world[:3, :3] @ yaw_variant @ release_local[:3, :3]
        ).tolist()
        candidate["orientation_source"] = f"{exp_id}:region_relative_release_keypose"
        candidate["orientation_yaw_offset_degrees"] = yaw_offset_degrees
        candidate["approach_height_m"] = float(approach_height_m)
        candidate["route_strategy"] = "jointspace_rrt"
        candidate["source_keypose_experience_id"] = exp_id
        candidate["source_keypose_task"] = experience.get("source", {}).get("task")

        runtime_experience = dict(experience)
        runtime_experience["keypose_sequence"] = [
            row for row in experience.get("keypose_sequence", [])
            if row.get("name") in {"grasp", "lift", "preplace", "release"}
        ]
        runtime_experience["_route_start_after"] = "lift"
        required = {"grasp", "lift", "preplace", "release"}
        names = {row.get("name") for row in runtime_experience["keypose_sequence"]}
        if not required.issubset(names):
            preflight_rows.append({
                "candidate_id": candidate["candidate_id"],
                "source_experience_id": exp_id,
                "status": "rejected",
                "failure_class": "source_keypose_sequence_missing_required_phases",
            })
            continue

        goal_object = np.eye(4, dtype=np.float64)
        goal_object[:3, :3] = np.asarray(candidate["rotation_world"], dtype=np.float64)
        goal_object[:3, 3] = np.asarray(candidate["position_world_m"], dtype=np.float64)
        goal_eef = goal_object @ object_to_eef
        relocated = retarget_keypose_sequence(
            runtime_experience,
            held_object_world=object_world,
            current_eef_world=eef_world,
            destination_region_world=region_world,
            destination_object_world=goal_object,
        )
        sequence = [row for row in relocated if row.get("name") in {"lift", "preplace"}]
        sequence = select_keypose_suffix(sequence, "lift")
        if len(sequence) != 1 or sequence[0].get("name") != "preplace":
            raise RuntimeError("keypose_candidate_preflight_did_not_resolve_one_preplace_pose")
        preplace = sequence[0]
        preplace_position = np.asarray(preplace["eef_pose_world"]["position_m"], dtype=np.float64)
        entry = np.asarray(candidate.get("entry_direction_world", [0.0, 0.0, 0.0]), dtype=np.float64)
        entry_norm = float(np.linalg.norm(entry))
        if entry_norm > 1e-8:
            entry /= entry_norm
        preplace_position += entry * _native_region_entry_offset_m(predicate)
        preplace_position[2] = float(goal_eef[2, 3] + approach_height_m)
        update_pose_record(
            preplace["eef_pose_world"], position_m=preplace_position,
        )
        preplace_pose = np.eye(4, dtype=np.float64)
        preplace_pose[:3, :3] = Rotation.from_quat(
            np.asarray(preplace["eef_pose_world"]["quaternion_wxyz"], dtype=np.float64)[[1, 2, 3, 0]]
        ).as_matrix()
        preplace_pose[:3, 3] = preplace_position

        row: dict[str, Any] = {
            "candidate_id": candidate["candidate_id"],
            "source_experience_id": exp_id,
            "source_task": candidate["source_keypose_task"],
            "placement_candidate": candidate,
            "screened_keypose": "preplace",
            "approach_height_m": approach_height_m,
        }
        preplace_plan = screen_pose(preplace_pose, 74201 + variant_index * 2)
        if (
            preplace_plan.get("status") == "no_safe_path"
            and int(preplace_plan.get("ik_candidate_count", 0)) > 0
            and "planning_time_budget_exhausted" in str(preplace_plan.get("failure_reason", ""))
            and preflight_count < maximum_rrt_preflights
        ):
            # One longer, new-seed route attempt is allowed only when IK has
            # already established that the pose is reachable and the first
            # RRT stopped specifically at its time budget.
            extended = screen_pose(
                preplace_pose,
                84201 + variant_index,
                search_time_budget_s=20.0,
                maximum_iterations=2000,
                maximum_config_checks=8000,
            )
            first_attempt = preplace_plan
            preplace_plan = {
                **extended,
                "initial_RRT_attempt": first_attempt,
                "bounded_extended_RRT_attempt": extended,
            }
        row["preplace_RRT"] = preplace_plan
        if preplace_plan.get("bounded_extended_RRT_attempt", {}).get("status") == "offline_feasible":
            candidate.update({
                "preplace_rrt_search_time_budget_s": 20.0,
                "preplace_rrt_max_iterations": 2000,
                "preplace_rrt_max_config_checks": 8000,
                "preplace_rrt_seed": 84201 + variant_index,
            })
        else:
            candidate.update({
                "preplace_rrt_search_time_budget_s": 12.0,
                "preplace_rrt_max_iterations": 750,
                "preplace_rrt_max_config_checks": 3000,
                "preplace_rrt_seed": 74201 + variant_index * 2,
            })
        row["placement_candidate"] = candidate
        if preplace_plan.get("status") == "offline_feasible":
            route_cache = preplace_plan.get("route_cache")
            if route_cache is not None:
                runtime_experience["_precomputed_keypose_routes"] = {
                    "preplace": route_cache,
                }
            terminal_plan = screen_pose(goal_eef, 74202 + variant_index * 2)
            row["terminal_pose_IK_RRT_diagnostic"] = terminal_plan
            terminal_deferred = terminal_plan.get("status") == "budget_exhausted"
            terminal_ik_feasible = bool(
                terminal_plan.get("status") == "offline_feasible"
                or int(terminal_plan.get("ik_candidate_count", 0)) > 0
                or terminal_deferred
            )
            row["defer_terminal_ik_to_live_keypose_planning"] = terminal_deferred
        elif _task_success_waypoint_fallback_reason(preplace_plan) is not None:
            fallback_reason = _task_success_waypoint_fallback_reason(preplace_plan)
            candidate["task_success_waypoint_fallback"] = True
            candidate["task_success_waypoint_fallback_reason"] = fallback_reason
            candidate["preplace_rrt_ik_candidate_count"] = int(
                preplace_plan.get("ik_candidate_count", 0)
            )
            row["placement_candidate"] = candidate
            row["task_success_waypoint_fallback"] = {
                "enabled": True,
                "reason": fallback_reason,
                "physical_validation_required": True,
                "safety_certified": False,
            }
            row["terminal_pose_IK_RRT_diagnostic"] = {
                "status": "deferred_until_preplaced_by_real_OSC",
                "failure_class": "terminal_segment_is_planned_only_after_physical_progress",
                "ik_candidate_count": 0,
            }
            row["defer_terminal_ik_to_live_keypose_planning"] = True
            terminal_ik_feasible = True
        else:
            row["terminal_pose_IK_RRT_diagnostic"] = {
                "status": "not_run_after_unreachable_preplace_route",
                "failure_class": "candidate_stopped_before_unreachable_waypoint",
                "ik_candidate_count": 0,
            }
            terminal_ik_feasible = False
            row["defer_terminal_ik_to_live_keypose_planning"] = False
        row["admitted_to_physical_search"] = bool(
            (
                preplace_plan.get("status") == "offline_feasible" and terminal_ik_feasible
            ) or _task_success_waypoint_fallback_reason(preplace_plan) is not None
        )
        row["terminal_contact_policy"] = "bounded_OSC_descent_only_after_valid_terminal_IK"
        preflight_rows.append(row)
        if row["admitted_to_physical_search"]:
            physical_candidates.append(candidate)
            experience_by_candidate[candidate["candidate_id"]] = runtime_experience

    # Preserve a distinct second route for the optional branch, while ranking
    # the first by target-region occupancy score and the actual RRT planning time.
    physical_candidates.sort(key=lambda item: (
        -float(item.get("ranking_score", 0.0)),
        float(next((row["preplace_RRT"].get("planning_seconds", float("inf"))
                    for row in preflight_rows if row.get("candidate_id") == item["candidate_id"]), float("inf"))),
    ))
    if len(physical_candidates) > 1:
        alternate = next((item for item in physical_candidates[1:]
                          if item.get("region_local_xy_m") != physical_candidates[0].get("region_local_xy_m")
                          or item.get("entry_direction_world") != physical_candidates[0].get("entry_direction_world")), None)
        physical_candidates = [physical_candidates[0]] + ([alternate] if alternate else [])
    return physical_candidates, experience_by_candidate, preflight_rows


def _native_place(
    bridge,
    *,
    target_id: str,
    target_binding: dict[str, Any],
    predicate: list[str],
    instruction: str,
    model_sha: str,
    safety_policy: dict[str, Any],
    controller_fields: dict[str, Any],
    controller: TaskPhaseController,
    output_dir: Path,
    keypose_experiences: list[dict[str, Any]] | None = None,
    maximum_physical_branches: int = 2,
    maximum_rrt_preflights: int = 8,
    keypose_preflight_report: dict[str, Any] | None = None,
    expected_start_snapshot_sha256: str | None = None,
    expected_bddl_sha256: str | None = None,
    demonstration_horizon: int = 1000,
    rrt_time_budget_s: float = NATIVE_PLACEMENT_RRT_BUDGET_S,
    recovery_fallback_snapshot: dict[str, Any] | None = None,
    transport_strategy: str = "native_region_progress_osc",
    native_region_route_variant: str | None = None,
    native_region_candidate_offset: int = 0,
    native_region_transit_clearance_mode: str = "compiled_obstacle",
    native_region_transit_height_override_m: float | None = None,
    native_region_progress_max_step_m: float = 0.008,
    native_region_entry_mode: str = "over_rim",
    native_region_aperture_tilt_degrees: float = 0.0,
    native_region_entry_height_offset_m: float | None = None,
    native_region_yaw_delta_degrees: float | None = None,
) -> dict[str, Any]:
    """Search distinct native-region layouts from one exact held-object Snapshot."""
    if _STAGE_A_NEAR_GOAL_RELEASE_PROBE_ENABLED:
        controller.enable_near_goal_release_probe()
    output_dir.mkdir(parents=True, exist_ok=True)
    native_start = bridge.request({"op": "native_task_status"})
    values, task_success = _task_values(native_start)
    if task_success:
        return {
            "status": "complete_task_success",
            "final_native_task_status": native_start,
            "native_task_success": True,
            "recorded_actions": 0,
            "safety_certified": False,
        }
    limit_recovery = {"status": "not_needed", "physical_actions": 0, "safety_certified": False}
    if controller.phase != "release" and not values.get(tuple(predicate), False):
        limit_recovery = _recover_joint_limits_with_osc(
            bridge,
            target_binding=target_binding,
            instruction=instruction,
            safety_policy=safety_policy,
            controller_fields=controller_fields,
            output_dir=output_dir / "joint_limit_recovery",
            fallback_snapshot=recovery_fallback_snapshot,
            defer_micro_violation_to_transport=(
                transport_strategy in {"legacy_rrt", "native_region_progress_osc"}
            ),
        )
        if limit_recovery.get("status") == "joint_limit_recovery_failed_without_legal_fallback":
            final_status = bridge.request({"op": "native_task_status"})
            return {
                "status": "joint_limit_recovery_failed",
                "failure_reason": limit_recovery.get("reason"),
                "joint_limit_recovery": limit_recovery,
                "final_native_task_status": final_status,
                "native_task_success": final_status.get("task_success"),
                "recorded_actions": int(limit_recovery.get("physical_actions", 0)),
                "safety_certified": False,
            }
        native_start = bridge.request({"op": "native_task_status"})
        values, task_success = _task_values(native_start)
        if task_success:
            return {
                "status": "complete_task_success",
                "final_native_task_status": native_start,
                "native_task_success": True,
                "joint_limit_recovery": limit_recovery,
                "recorded_actions": int(limit_recovery.get("physical_actions", 0)),
                "safety_certified": False,
            }
    experience_portfolio = list(keypose_experiences or [])
    if controller.phase == "release" or values.get(tuple(predicate), False):
        result = _native_place_single(
            bridge, target_id=target_id, target_binding=target_binding,
            target_context=None,
            predicate=predicate, instruction=instruction, model_sha=model_sha,
            safety_policy=safety_policy, controller_fields=controller_fields,
            controller=controller, output_dir=output_dir,
            keypose_experience=experience_portfolio[0] if experience_portfolio else None,
            expected_bddl_sha256=expected_bddl_sha256,
            demonstration_horizon=demonstration_horizon,
            rrt_time_budget_s=rrt_time_budget_s,
            native_region_transit_clearance_mode=native_region_transit_clearance_mode,
            native_region_transit_height_override_m=native_region_transit_height_override_m,
            native_region_progress_max_step_m=native_region_progress_max_step_m,
            native_region_entry_mode=native_region_entry_mode,
            native_region_aperture_tilt_degrees=native_region_aperture_tilt_degrees,
            native_region_entry_height_offset_m=native_region_entry_height_offset_m,
        )
        result["joint_limit_recovery"] = limit_recovery
        recovery_actions = int(
            limit_recovery.get("physical_actions", limit_recovery.get("physical_recovery_actions", 0)) or 0
        )
        result["joint_limit_recovery_actions"] = recovery_actions
        result["recorded_actions"] = int(result.get("recorded_actions", 0) or 0) + recovery_actions
        return result

    base = bridge.request({"op": "export_runtime_snapshot"})
    if base.get("status") != "complete" or not isinstance(base.get("payload"), bytes):
        raise RuntimeError("native_region_candidate_branch_start_snapshot_export_failed")
    base_payload = base["payload"]
    base_state = _integration_state(base.get("snapshot_state", {}))
    target_query = bridge.request({"op": "native_goal_query", "predicate": predicate})
    target_context = bridge.request({
        "op": "task_success_first_context",
        "expected_instruction": instruction,
        "target_predicate": predicate,
    })
    if (
        target_query.get("status") != "complete"
        or target_query.get("predicate_is_native_goal") is not True
        or target_context.get("status") != "complete"
        or target_context.get("model_geometry_sha256") != model_sha
    ):
        raise RuntimeError(
            "native_region_candidate_geometry_binding_unavailable:"
            f"goal_query={target_query.get('status')}:{target_query.get('error')};"
            f"native_goal={target_query.get('predicate_is_native_goal')};"
            f"target_context={target_context.get('status')}:{target_context.get('error')};"
            f"model_match={target_context.get('model_geometry_sha256') == model_sha}"
        )
    occupied_contexts = []
    for row in native_start.get("native_goal_values", []):
        other = row.get("predicate")
        if (
            row.get("value") is True
            and isinstance(other, list) and len(other) == 3
            and other[0].lower() in {"on", "in"}
            and other[1] != target_id and other[2] == predicate[2]
        ):
            context = bridge.request({
                "op": "task_success_first_context",
                "expected_instruction": instruction,
                "target_predicate": other,
            })
            if context.get("status") == "complete":
                occupied_contexts.append(context)
    candidates = _generate_native_region_placement_candidates(
        predicate=predicate,
        region=target_query["region"],
        target_context=target_context,
        occupied_contexts=occupied_contexts,
        # An explicit yaw is a controlled physical hypothesis, not a request
        # to fan out across the full orientation/layout portfolio. Keep only
        # as many geometrically ranked layouts as can actually be executed;
        # otherwise the live-IK preflight pays for up to 16 near-duplicate
        # layouts even when this experiment permits a single physical branch.
        maximum_candidates=(
            max(1, min(16, maximum_physical_branches))
            if native_region_yaw_delta_degrees is not None else 16
        ),
        preferred_yaw_delta_degrees=native_region_yaw_delta_degrees,
    )
    candidate_ik_margin_ranking: list[dict[str, Any]] = []
    if predicate[0].lower() == "in" and len(candidates) > 1:
        candidates, candidate_ik_margin_ranking = (
            _rank_native_region_candidates_by_live_ik_margin(
                bridge, candidates,
                target_binding=target_binding,
                target_context=target_context,
                region=target_query["region"],
                instruction=instruction,
                model_sha=model_sha,
                safety_policy=safety_policy,
                route_variant=(
                    native_region_route_variant or "direct_region_progress"
                ),
                transit_clearance_mode=native_region_transit_clearance_mode,
                entry_mode=native_region_entry_mode,
            )
        )
    if (
        isinstance(native_region_candidate_offset, bool)
        or not isinstance(native_region_candidate_offset, int)
        or not 0 <= native_region_candidate_offset < len(candidates)
    ):
        raise ValueError("native_region_candidate_offset_must_select_a_generated_layout")
    candidates = candidates[native_region_candidate_offset:]
    keypose_preflight: list[dict[str, Any]] = []
    candidate_experiences: dict[str, dict[str, Any]] = {}
    preflight_reuse_summary: dict[str, Any] | None = None
    if keypose_preflight_report is not None:
        if not experience_portfolio:
            raise ValueError("precomputed_keypose_route_requires_verified_source_experiences")
        prior_scope = keypose_preflight_report.get("scope", {})
        prior_task = keypose_preflight_report.get("task", {})
        prior_resume = keypose_preflight_report.get("resumed_from_runtime_snapshot", {})
        if not prior_resume:
            handoff = keypose_preflight_report.get("student_handoff_snapshot_restore", {})
            handoff_path = Path(str(handoff.get("path") or ""))
            if handoff.get("status") == "verified_exact" and handoff_path.is_file():
                prior_resume = {
                    "path": str(handoff_path),
                    "payload_sha256": _sha(handoff_path.read_bytes()),
                    "integration_state_sha256": handoff.get("mujoco_integration_state_sha256"),
                }
        if (
            not isinstance(prior_scope, dict)
            or prior_task.get("instruction") != instruction
            or prior_task.get("native_goal_predicate") != predicate
            or prior_task.get("model_geometry_sha256") != model_sha
            or expected_bddl_sha256 is None
            or prior_task.get("BDDL_sha256") != expected_bddl_sha256
            or expected_start_snapshot_sha256 is None
            or prior_resume.get("payload_sha256") != expected_start_snapshot_sha256
        ):
            raise ValueError("precomputed_keypose_route_scene_or_snapshot_fingerprint_mismatch")
        prior_segments = keypose_preflight_report.get("segments", [])
        prior_search = next((
            row.get("placement_search") for row in reversed(prior_segments)
            if isinstance(row.get("placement_search"), dict)
        ), None)
        if not isinstance(prior_search, dict):
            raise ValueError("precomputed_keypose_route_report_has_no_candidate_preflight")
        prior_start_state_sha = prior_search.get("start_snapshot_state_sha256")
        if prior_start_state_sha != _state_digest(base_state):
            raise ValueError("precomputed_keypose_route_state_fingerprint_mismatch")
        prior_native = prior_task.get("initial_native_task_status", {}).get("native_goal_values", [])
        current_native = native_start.get("native_goal_values", [])
        prior_pairs = sorted((tuple(row.get("predicate", [])), row.get("value")) for row in prior_native)
        current_pairs = sorted((tuple(row.get("predicate", [])), row.get("value")) for row in current_native)
        if prior_pairs != current_pairs:
            raise ValueError("precomputed_keypose_route_native_predicates_differ_from_restored_snapshot")
        prior_preflight_rows = prior_search.get("keypose_candidate_preflight", [])
        if not prior_preflight_rows:
            prior_preflight_rows = next((
                row.get("keypose_route_preflight")
                for row in reversed(prior_segments)
                if isinstance(row.get("keypose_route_preflight"), list)
                and row.get("keypose_route_preflight")
            ), [])
        feasible_rows = [row for row in prior_preflight_rows
                         if _preflight_row_is_task_success_admissible(row)]
        if not feasible_rows:
            raise ValueError("precomputed_keypose_route_report_has_no_admitted_candidate")
        selected = feasible_rows[0]
        candidate = dict(selected["placement_candidate"])
        fallback_reason = _task_success_waypoint_fallback_reason(
            selected.get("preplace_RRT", {})
        )
        if fallback_reason is not None:
            candidate["task_success_waypoint_fallback"] = True
            candidate["task_success_waypoint_fallback_reason"] = fallback_reason
            candidate["preplace_rrt_ik_candidate_count"] = int(
                selected.get("preplace_RRT", {}).get("ik_candidate_count", 0)
            )
        experience_id = selected.get("source_experience_id")
        experience = next((row for row in experience_portfolio
                           if row.get("experience_id") == experience_id), None)
        if experience is None:
            raise ValueError("precomputed_keypose_route_source_experience_not_loaded")
        runtime_experience = dict(experience)
        runtime_experience["keypose_sequence"] = [
            row for row in experience.get("keypose_sequence", [])
            if row.get("name") in {"grasp", "lift", "preplace", "release"}
        ]
        runtime_experience["_route_start_after"] = "lift"
        cached_route = selected.get("preplace_RRT", {}).get("route_cache")
        if isinstance(cached_route, dict):
            runtime_experience["_precomputed_keypose_routes"] = {
                "preplace": cached_route,
            }
        candidate_id = str(selected.get("candidate_id", ""))
        try:
            variant_index = int(candidate_id.split("_")[1])
        except (IndexError, ValueError):
            raise ValueError("precomputed_keypose_route_candidate_id_is_malformed")
        extended = selected.get("preplace_RRT", {}).get("bounded_extended_RRT_attempt", {})
        if extended.get("status") == "offline_feasible":
            candidate.update({
                "preplace_rrt_search_time_budget_s": 20.0,
                "preplace_rrt_max_iterations": 2000,
                "preplace_rrt_max_config_checks": 8000,
                "preplace_rrt_seed": 84201 + variant_index,
            })
        else:
            candidate.update({
                "preplace_rrt_search_time_budget_s": 12.0,
                "preplace_rrt_max_iterations": 750,
                "preplace_rrt_max_config_checks": 3000,
                "preplace_rrt_seed": 74201 + variant_index * 2,
            })
        candidates = [candidate]
        keypose_preflight = list(prior_preflight_rows)
        candidate_experiences[candidate_id] = runtime_experience
        preflight_reuse_summary = {
            "status": "reused_after_exact_snapshot_hash_and_native_predicate_recheck",
            "candidate_id": candidate_id,
            "source_experience_id": experience_id,
            "terminal_ik_deferred_to_per_keypose_live_RRT": bool(
                selected.get("defer_terminal_ik_to_live_keypose_planning") is True
            ),
            "source_preflight_run_archive": keypose_preflight_report.get("archive"),
            "resume_snapshot_payload_sha256": expected_start_snapshot_sha256,
            "start_snapshot_state_sha256": _state_digest(base_state),
            "physical_waypoint_RRT_replanned_before_execution": cached_route is None,
            "precomputed_route_cache_available": cached_route is not None,
            "precomputed_route_cache_identity_sha256": (
                cached_route.get("identity_sha256") if cached_route else None
            ),
            "safety_certified": False,
        }
    elif transport_strategy == "native_region_progress_osc":
        matching_experiences = [
            experience for experience in experience_portfolio
            if (
                experience.get("object_family") == target_id.rsplit("_", 1)[0]
                and str(experience.get("target_relation", "")).lower() == predicate[0].lower()
                and experience.get("target_region_name") == predicate[-1]
            )
        ]
        for variant_index, candidate in enumerate(candidates):
            candidate["route_strategy"] = "native_region_progress_osc"
            route_variants = (
                "direct_region_progress",
                "keypose_side_region_progress",
                "keypose_side_region_progress_reverse",
            )
            candidate["route_variant"] = (
                native_region_route_variant
                or (
                    route_variants[min(variant_index, len(route_variants) - 1)]
                )
            )
            candidate["candidate_id"] = (
                f"{candidate['candidate_id']}_{candidate['route_variant']}"
            )
            if matching_experiences:
                experience = matching_experiences[min(variant_index, len(matching_experiences) - 1)]
                candidate["source_keypose_experience_id"] = experience.get("experience_id")
                candidate["source_keypose_task"] = experience.get("source", {}).get("task")
                candidate_experiences[candidate["candidate_id"]] = experience
        candidates = candidates[:max(1, min(3, int(maximum_physical_branches)))]
        keypose_preflight = [{
            "status": "soft_reference_only_no_RRT_gate",
            "source_experience_ids": [
                row.get("experience_id") for row in matching_experiences
            ],
            "physical_route_generation": "native_region_geometry_and_live_object_state",
            "RRT_or_IK_timeout_terminates_search": False,
            "safety_certified": False,
        }]
    elif transport_strategy == "legacy_rrt":
        # Keep the two best distinct target-region layouts, and route both with
        # the existing full-geometry MuJoCo 7D planner. When a completed object
        # occupies the same region, diversify the alternate toward shorter reach
        # while keeping its conservative footprint clearance close to the best.
        # Do not turn the second branch into an obstacle-blind OSC shortcut.
        for candidate in candidates:
            candidate["route_strategy"] = "jointspace_rrt"
        selection_note = "highest_ranked_distinct_region_layouts"
        alternate_clearance_slack_m = None
        if occupied_contexts and len(candidates) > 2:
            primary = candidates[0]
            primary_clearance = primary.get("estimated_completed_object_clearance_m")
            if isinstance(primary_clearance, (int, float)) and np.isfinite(primary_clearance):
                near_clearance = [
                    row for row in candidates[1:]
                    if isinstance(row.get("estimated_completed_object_clearance_m"), (int, float))
                    and np.isfinite(row["estimated_completed_object_clearance_m"])
                    and float(row["estimated_completed_object_clearance_m"])
                    >= float(primary_clearance) - 0.030
                ]
                if near_clearance:
                    alternate = min(
                        near_clearance,
                        key=lambda row: (
                            float(row.get("travel_distance_m", float("inf"))),
                            str(row.get("candidate_id", "")),
                        ),
                    )
                    candidates = [primary, alternate]
                    alternate_clearance_slack_m = float(
                        primary_clearance - alternate["estimated_completed_object_clearance_m"]
                    )
                    selection_note = (
                        "best_clearance_primary_plus_shortest_travel_alternate_within_30mm_clearance"
                    )
        candidates = candidates[:max(1, min(2, int(maximum_physical_branches)))]
        keypose_preflight = [{
            "status": "native_region_layout_candidates_selected_for_full_geometry_RRT",
            "candidate_ids": [row["candidate_id"] for row in candidates],
            "selection_rule": selection_note,
            "alternate_estimated_clearance_slack_m": alternate_clearance_slack_m,
            "RRT_preflight_count": 0,
            "RRT_planning_occurs_once_per_physical_candidate": True,
            "safety_certified": False,
        }]
    elif experience_portfolio:
        candidates, candidate_experiences, keypose_preflight = _screen_native_keypose_candidates(
            bridge,
            target_id=target_id,
            target_binding=target_binding,
            predicate=predicate,
            instruction=instruction,
            model_sha=model_sha,
            bddl_sha=str(expected_bddl_sha256 or ""),
            safety_policy=safety_policy,
            start_snapshot_state=base.get("snapshot_state", {}),
            start_state_sha256=_state_digest(base_state),
            region=target_query["region"],
            geometry_candidates=candidates,
            experiences=experience_portfolio,
            maximum_rrt_preflights=maximum_rrt_preflights,
            maximum_rrt_time_budget_s=rrt_time_budget_s,
        )
        route_cache_dir = output_dir / "preflight_route_cache"
        for row in keypose_preflight:
            route_cache = row.get("preplace_RRT", {}).get("route_cache")
            if not isinstance(route_cache, dict):
                continue
            route_cache_dir.mkdir(parents=True, exist_ok=True)
            route_file = route_cache_dir / f"{row['candidate_id']}.json"
            route_bytes = json.dumps(
                route_cache, ensure_ascii=False, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            route_file.write_bytes(route_bytes)
            route_cache["archive_ref"] = {
                "path": str(route_file),
                "sha256": _sha(route_bytes),
                "qpos_path_rows": len(route_cache["paths"]["qpos_path_rad"]),
                "eef_path_rows": len(route_cache["paths"]["eef_reference_poses_world"]),
            }
        if not candidates:
            screen_snapshot_path = output_dir / "keypose_candidate_screen_start_runtime_snapshot.pkl"
            screen_snapshot_path.write_bytes(base_payload)
            return {
                "status": "native_region_candidates_unreachable",
                "failure_reason": "no_candidate_has_a_full_geometry_RRT_reachable_preplace_and_a_valid_terminal_IK_solution",
                "native_goal_predicate": predicate,
                "native_goal_value": False,
                "native_task_success": False,
                "final_native_task_status": native_start,
                "recorded_actions": 0,
                "recorded_physics_substeps": 0,
                "keypose_route_preflight": keypose_preflight,
                "placement_search": {
                    "mode": "L0_L1_keypose_comparison_and_full_geometry_RRT_preflight",
                    "candidate_count": len(keypose_preflight),
                    "RRT_preflight_count": _keypose_rrt_call_count(keypose_preflight),
                    "physical_branch_count": 0,
                    "candidate_attempts": keypose_preflight,
                    "completed_object_occupancy_count": len(occupied_contexts),
                    "start_snapshot_state_sha256": _state_digest(base_state),
                    "start_snapshot_path": str(screen_snapshot_path),
                    "branch_actions_use_real_MuJoCo_OSC": True,
                    "direct_object_or_robot_state_write": False,
                    "safety_certified": False,
                },
            }
    attempts: list[dict[str, Any]] = []
    last_result: dict[str, Any] | None = None

    setattr(
        sys.modules["__main__"],
        "_PythonObjectState",
        __import__(
            "research.gs_opd.repair.task_success_first_grasp_search",
            fromlist=["_PythonObjectState"],
        )._PythonObjectState,
    )
    base_envelope = pickle.loads(base_payload)
    base_scene = base_envelope.get("scene_metadata", {})

    def restore_branch_start(snapshot_path: Path | None = None) -> dict[str, Any]:
        if snapshot_path is not None:
            # A failed physical transport can still make useful progress. Resume
            # the next ranked layout from that exact, same-task state instead of
            # paying for an identical approach prefix from the original grasp.
            descriptor = _verified_runtime_snapshot_descriptor(
                snapshot_path,
                benchmark=str(base_scene["benchmark"]),
                level=int(base_scene["level"]),
                task_id=int(base_scene["task_id"]),
                seed=int(base_scene["environment_seed"]),
                init_state=int(base_scene["init_state_index"]),
                model_sha=model_sha,
                bddl_sha=str(expected_bddl_sha256 or base_scene["bddl_sha256"]),
                instruction=instruction,
            )
            branch_payload = descriptor["payload"]
            branch_state = _integration_state(descriptor["snapshot_state"])
            branch_source = {
                "path": str(snapshot_path),
                "payload_sha256": descriptor["payload_sha256"],
                "integration_state_sha256": _state_digest(branch_state),
                "source": "exact_verified_failed_transport_snapshot",
            }
        else:
            branch_payload = base_payload
            branch_state = base_state
            branch_source = {
                "path": None,
                "payload_sha256": None,
                "integration_state_sha256": _state_digest(base_state),
                "source": "original_verified_held_object_snapshot",
            }
        try:
            restored = bridge.request({"op": "restore_runtime_snapshot", "payload": branch_payload})
        except (EOFError, BrokenPipeError, ConnectionError, OSError):
            # A terminated Bridge loses only its live session. Restart the
            # same compiled scene, then recover this branch's verified prefix.
            bridge.close()
            bridge.start()
            reset = bridge.request({"op": "reset"})
            if reset.get("scene_metadata", {}).get("model_geometry_sha256") != model_sha:
                raise RuntimeError("restarted_placement_branch_model_fingerprint_mismatch")
            restored = bridge.request({"op": "restore_runtime_snapshot", "payload": branch_payload})
        restored_state = _integration_state(restored.get("snapshot_state", {}))
        if (
            restored.get("status") != "restored"
            or restored.get("model_geometry_sha256") != model_sha
            or restored_state.shape != branch_state.shape
            or not np.array_equal(restored_state, branch_state)
        ):
            raise RuntimeError("placement_candidate_snapshot_restore_not_exact")
        restored_native = bridge.request({"op": "native_task_status"})
        restored_values, restored_success = _task_values(restored_native)
        if restored_success:
            raise RuntimeError("placement_candidate_resume_snapshot_already_has_native_task_success")
        if restored_values.get(tuple(predicate), False):
            raise RuntimeError("placement_candidate_resume_snapshot_already_satisfies_target_predicate")
        restored_grasp = bridge.request({
            "op": "grasp_state",
            "carried_binding": target_binding,
            "expected_instruction": instruction,
        })
        fingers = restored_grasp.get("finger_contacts", {})
        if not (
            restored_grasp.get("grasp_confirmed") is True
            and fingers.get("left") is True and fingers.get("right") is True
        ):
            raise RuntimeError("placement_candidate_restore_lost_bilateral_held_object")
        branch_source["restored_timestep"] = restored_native.get("timestep")
        branch_source["target_predicate_value"] = restored_values.get(tuple(predicate))
        return branch_source

    physical_branch_budget = max(1, min(3, int(maximum_physical_branches)))
    for ordinal, candidate in enumerate(candidates[:physical_branch_budget]):
        branch_start_source = {
            "path": None,
            "payload_sha256": None,
            "integration_state_sha256": _state_digest(base_state),
            "source": "original_verified_held_object_snapshot",
        }
        if ordinal:
            previous_attempt = attempts[-1] if attempts else {}
            previous_snapshot = previous_attempt.get("terminal_snapshot")
            continue_from_progress = bool(
                transport_strategy == "native_region_progress_osc"
                and previous_attempt.get("status") == "native_goal_transport_failed"
                and isinstance(previous_snapshot, str)
                and Path(previous_snapshot).is_file()
                and expected_bddl_sha256 is not None
            )
            try:
                if continue_from_progress:
                    try:
                        branch_start_source = restore_branch_start(Path(previous_snapshot))
                        # A failed OSC branch may have physically moved its
                        # destination fixture. Placement layouts generated at
                        # the original held-object snapshot are then stale,
                        # even though the next branch correctly resumes from
                        # the exact later Runtime Snapshot. Re-query the live
                        # native region and regenerate this branch's candidate
                        # from that restored state; do not pay for another
                        # grasp prefix or carry an obsolete world-space goal.
                        if transport_strategy == "native_region_progress_osc":
                            live_goal = bridge.request({
                                "op": "native_goal_query", "predicate": predicate,
                            })
                            live_context = bridge.request({
                                "op": "task_success_first_context",
                                "expected_instruction": instruction,
                                "target_predicate": predicate,
                            })
                            live_native = bridge.request({"op": "native_task_status"})
                            if (
                                live_goal.get("status") != "complete"
                                or live_goal.get("predicate_is_native_goal") is not True
                                or live_context.get("status") != "complete"
                                or live_context.get("model_geometry_sha256") != model_sha
                                or live_native.get("status") != "complete"
                            ):
                                raise RuntimeError(
                                    "live_native_region_retarget_query_failed_after_snapshot_restore"
                                )
                            refreshed_occupied: list[dict[str, Any]] = []
                            for live_row in live_native.get("native_goal_values", []):
                                other = live_row.get("predicate")
                                if (
                                    live_row.get("value") is True
                                    and isinstance(other, list) and len(other) == 3
                                    and other[0].lower() in {"on", "in"}
                                    and other[1] != target_id
                                    and other[2] == predicate[2]
                                ):
                                    live_occupant = bridge.request({
                                        "op": "task_success_first_context",
                                        "expected_instruction": instruction,
                                        "target_predicate": other,
                                    })
                                    if live_occupant.get("status") == "complete":
                                        refreshed_occupied.append(live_occupant)
                            fresh_candidates = _generate_native_region_placement_candidates(
                                predicate=predicate,
                                region=live_goal["region"],
                                target_context=live_context,
                                occupied_contexts=refreshed_occupied,
                                maximum_candidates=(
                                    max(1, min(16, int(maximum_physical_branches)))
                                    if native_region_yaw_delta_degrees is not None else 16
                                ),
                                preferred_yaw_delta_degrees=native_region_yaw_delta_degrees,
                            )
                            if not fresh_candidates:
                                raise RuntimeError(
                                    "live_native_region_retarget_generated_no_placement_candidate"
                                )
                            if native_region_candidate_offset >= len(fresh_candidates):
                                raise RuntimeError(
                                    "live_native_region_retarget_candidate_offset_out_of_range:"
                                    f"offset={native_region_candidate_offset};"
                                    f"generated={len(fresh_candidates)}"
                                )
                            stale_candidate = candidate
                            refreshed_candidate = dict(
                                fresh_candidates[native_region_candidate_offset]
                            )
                            route_variants = (
                                "direct_region_progress",
                                "keypose_side_region_progress",
                                "keypose_side_region_progress_reverse",
                            )
                            refreshed_variant = str(
                                native_region_route_variant
                                or stale_candidate.get("route_variant")
                                or route_variants[min(ordinal, len(route_variants) - 1)]
                            )
                            refreshed_candidate.update({
                                "route_strategy": "native_region_progress_osc",
                                "route_variant": refreshed_variant,
                                "candidate_id": (
                                    f"{refreshed_candidate['candidate_id']}_{refreshed_variant}"
                                ),
                                "live_snapshot_retarget": {
                                    "status": "regenerated_from_exact_failed_branch_snapshot",
                                    "source_snapshot": str(previous_snapshot),
                                    "stale_candidate_id": stale_candidate.get("candidate_id"),
                                    "stale_goal_position_world_m": stale_candidate.get(
                                        "position_world_m"
                                    ),
                                    "live_native_region_position_world_m": live_goal[
                                        "region"
                                    ].get("site_position_world_m"),
                                    "live_target_root_position_world_m": live_context.get(
                                        "target_root_pose_world", {}
                                    ).get("position_m"),
                                    "candidate_offset": native_region_candidate_offset,
                                    "physical_state_modified_during_retarget": False,
                                },
                            })
                            prior_experience = candidate_experiences.get(
                                str(stale_candidate.get("candidate_id", ""))
                            )
                            if prior_experience is not None:
                                refreshed_candidate["source_keypose_experience_id"] = (
                                    prior_experience.get("experience_id")
                                )
                                candidate_experiences[
                                    refreshed_candidate["candidate_id"]
                                ] = prior_experience
                            candidate = refreshed_candidate
                            target_context = live_context
                            occupied_contexts = refreshed_occupied
                            target_query = live_goal
                    except Exception as resume_exc:
                        if branch_start_source.get("source") == (
                            "exact_verified_failed_transport_snapshot"
                        ):
                            # Snapshot restoration succeeded; a live goal or
                            # geometry refresh failure must not silently fall
                            # back to the original pose and stale candidate.
                            raise
                        # Continue only when the failed branch retained the
                        # same verified bilateral grasp. Otherwise isolate the
                        # next layout from the original held-object snapshot.
                        branch_start_source = {
                            **restore_branch_start(),
                            "failed_progress_resume_reason": (
                                f"{type(resume_exc).__name__}: {resume_exc}"
                            ),
                        }
                else:
                    branch_start_source = restore_branch_start()
            except Exception as exc:
                attempts.append({
                    "candidate_id": candidate["candidate_id"],
                    "status": "candidate_branch_restore_failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                })
                break
            controller = TaskPhaseController(
                _controller_config(target_id, predicate, {"candidate_id": "verified_held_prefix"})
            )
            controller.adopt_held_object(target_id)
            if _STAGE_A_NEAR_GOAL_RELEASE_PROBE_ENABLED:
                controller.enable_near_goal_release_probe()
        branch_dir = output_dir / f"placement_branch_{ordinal:02d}_{candidate['candidate_id']}"
        try:
            result = _native_place_single(
                bridge, target_id=target_id, target_binding=target_binding,
                target_context=target_context,
                predicate=predicate, instruction=instruction, model_sha=model_sha,
                safety_policy=safety_policy, controller_fields=controller_fields,
                controller=controller, output_dir=branch_dir,
                placement_candidate=candidate,
                keypose_experience=candidate_experiences.get(candidate["candidate_id"]),
                expected_bddl_sha256=expected_bddl_sha256,
                demonstration_horizon=demonstration_horizon,
                rrt_time_budget_s=rrt_time_budget_s,
                native_region_transit_clearance_mode=native_region_transit_clearance_mode,
                native_region_transit_height_override_m=native_region_transit_height_override_m,
                native_region_progress_max_step_m=native_region_progress_max_step_m,
                native_region_entry_mode=native_region_entry_mode,
                native_region_aperture_tilt_degrees=native_region_aperture_tilt_degrees,
                native_region_entry_height_offset_m=native_region_entry_height_offset_m,
            )
        except Exception as exc:
            result = {
                "status": "placement_branch_error",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "native_goal_predicate": predicate,
                "native_task_success": None,
                "safety_certified": False,
            }
        last_result = result
        native_final = result.get("final_native_task_status", {})
        candidate_record = {
            "candidate": candidate,
            "source_keypose_experience_id": candidate.get("source_keypose_experience_id"),
            "status": result.get("status"),
            "failure_reason": result.get("failure_reason"),
            "failure_phase": result.get("failure_phase"),
            "transport_route_RRT": result.get("transport_route_RRT"),
            "transport_route_archive": result.get("transport_route_archive"),
            "native_goal_insertion_RRT": result.get("native_goal_insertion_RRT"),
            "keypose_RRT_waypoints": result.get("keypose_RRT_waypoints", []),
            "recorded_actions": result.get("recorded_actions", result.get("transport_actions", 0)),
            "native_task_success": native_final.get("task_success", result.get("native_task_success")),
            "target_predicate_value": next((
                row.get("value") for row in native_final.get("native_goal_values", [])
                if row.get("predicate") == predicate
            ), None),
            "terminal_snapshot": result.get("terminal_snapshot"),
            "branch_start_snapshot": branch_start_source,
            "branch_archive": str(branch_dir),
        }
        attempts.append(candidate_record)
        if candidate_record["native_task_success"] is True or candidate_record["target_predicate_value"] is True:
            break
    if last_result is None:
        last_result = {
            "status": "native_region_candidates_unavailable",
            "failure_reason": "no_native_region_placement_candidates_generated",
            "native_goal_predicate": predicate,
            "final_native_task_status": native_start,
            "native_task_success": False,
        }
    last_result["placement_search"] = {
        "mode": (
            "native_region_progress_OSC_with_soft_keypose_and_shared_physical_branch_budget"
            if transport_strategy == "native_region_progress_osc"
            else
            "L0_L1_keypose_guided_native_region_full_geometry_preflight"
            if experience_portfolio
            else "native_region_multi_candidate_snapshot_branches"
        ),
        "candidate_count": (
            len(candidates) if transport_strategy == "native_region_progress_osc"
            else len(keypose_preflight) if experience_portfolio else len(candidates)
        ),
        "native_region_candidate_offset": native_region_candidate_offset,
        "native_region_yaw_delta_degrees": native_region_yaw_delta_degrees,
        "endpoint_ik_joint_margin_ranking": candidate_ik_margin_ranking,
        "RRT_preflight_count": _keypose_rrt_call_count(keypose_preflight),
        "keypose_candidate_preflight": keypose_preflight,
        "preflight_reuse": preflight_reuse_summary,
        "verified_source_experiences": [
            {
                "experience_id": row.get("experience_id"),
                "task": row.get("source", {}).get("task"),
                "trace_sha256": row.get("source", {}).get("trace_sha256"),
                "acceptance_report_sha256": row.get("source", {}).get("acceptance_report_sha256"),
                "native_task_success": row.get("source", {}).get("native_task_success"),
                "official_safety_demonstrations_used": row.get("source", {}).get("official_safety_demonstrations_used"),
                "source_actions_replayed": False,
            }
            for row in experience_portfolio
        ],
        "physical_branch_count": sum(
            1 for row in attempts if row.get("branch_archive") is not None
        ),
        "candidate_attempts": attempts,
        "completed_object_occupancy_count": len(occupied_contexts),
        "start_snapshot_state_sha256": _state_digest(base_state),
        "branch_actions_use_real_MuJoCo_OSC": True,
        "direct_object_or_robot_state_write": False,
        "safety_certified": False,
    }
    last_result["joint_limit_recovery"] = limit_recovery
    recovery_actions = int(
        limit_recovery.get("physical_actions", limit_recovery.get("physical_recovery_actions", 0)) or 0
    )
    last_result["joint_limit_recovery_actions"] = recovery_actions
    last_result["recorded_actions"] = int(
        last_result.get("recorded_actions", last_result.get("transport_actions", 0)) or 0
    ) + recovery_actions
    return last_result


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _STAGE_A_NEAR_GOAL_RELEASE_PROBE_ENABLED
    global _STAGE_A_NEAR_GOAL_RELEASE_PROBE_MAX_ERROR_M
    from research.gs_opd.repair.automatic_recovery_runtime import Pi05StudentPolicy
    from research.gs_opd.repair.task_success_first_grasp_search import (
        BridgeSession, _PythonObjectState, _build_safety_policy, _build_bddl_grasp_screen_policy, _restore_initial,
        _snapshot_state_vector, _verify_cold_start_reset,
        configure_stage_a_contact_only_substep_audit,
    )

    configure_stage_a_contact_only_substep_audit(
        bool(getattr(args, "contact_only_substep_audit", False))
    )
    _STAGE_A_NEAR_GOAL_RELEASE_PROBE_ENABLED = bool(
        getattr(args, "stage_a_near_goal_release_probe", False)
    )
    _STAGE_A_NEAR_GOAL_RELEASE_PROBE_MAX_ERROR_M = float(
        getattr(args, "stage_a_near_goal_release_probe_max_error_m", 0.012)
    )
    cold_start_continuation = bool(getattr(args, "cold_start_continuation", False))
    cold_start = bool(getattr(args, "cold_start", False) or cold_start_continuation)
    continuation_root = getattr(args, "cold_start_continuation_root", None)
    if cold_start_continuation:
        sources = (args.resume_from_runtime_snapshot, args.resume_from_placement_report)
        if sum(source is not None for source in sources) != 1:
            raise ValueError("cold_start_continuation_requires_exactly_one_same_run_resume_source")
        if continuation_root is None:
            raise ValueError("cold_start_continuation_requires_its_V3_run_root")
        continuation_root = Path(continuation_root).resolve()
        source = Path(next(source for source in sources if source is not None)).resolve()
        if not source.is_relative_to(continuation_root):
            raise ValueError("cold_start_continuation_source_must_be_inside_its_V3_run")
    if cold_start and (
        args.experience_library is not None
        or args.keypose_experience
        or args.previous_search_report
        or args.resume_from_grasp_report is not None
        or (args.resume_from_placement_report is not None and not cold_start_continuation)
        or (args.resume_from_runtime_snapshot is not None and not cold_start_continuation)
        or args.resume_from_keypose_preflight_report is not None
        or args.recovery_fallback_runtime_snapshot is not None
        or args.candidate_start_index != 0
        # Object-local contact-patch offsets are generated from the active
        # scene geometry. They are cold-start-safe candidate construction,
        # unlike imported grasp poses, actions, or partial-task Snapshots.
    ):
        raise ValueError("cold_start_cannot_use_historical_experience_or_partial_task_inputs")
    for dependency_path in (
        Path("/workspace/zy_workspace/LIBERO-Safety"),
        Path("/workspace/zy_workspace/LIBERO-Safety/third_party/robosuite-1.4"),
        Path("/workspace/zy_workspace/RLinf"),
    ):
        if dependency_path.is_dir() and str(dependency_path) not in sys.path:
            sys.path.insert(0, str(dependency_path))
    snapshot_root = args.snapshot_root.resolve()
    snapshot_path = snapshot_root / "snapshots/initial_runtime_snapshot.pkl"
    manifest_path = snapshot_root / "snapshots/runtime_snapshot_manifest.json"
    if not snapshot_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("source_t0_Snapshot_or_manifest_missing")
    payload = snapshot_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["snapshots"]["initial"]
    if _sha(payload) != source["payload_sha256"]:
        raise ValueError("source_t0_Snapshot_hash_mismatch")
    model_sha = str(source["model_geometry_sha256"])
    # Archived Runtime Snapshots pickle a small LIBERO state adapter under
    # __main__; expose the existing compatibility wrapper before loading it.
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    envelope = pickle.loads(payload)
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != model_sha
        or envelope.get("scene_metadata", {}).get("instruction") != manifest.get("instruction")
    ):
        raise ValueError("source_t0_runtime_snapshot_envelope_identity_mismatch")
    initial_state = _snapshot_state_vector({
        "mujoco_integration_state": envelope["snapshot"]["mujoco_state"]
    })
    if _state_digest(initial_state) != source.get("mujoco_integration_state_sha256"):
        raise ValueError("source_t0_integration_state_hash_mismatch")
    output_dir = args.output_root.resolve() / (
        f"task_success_first_executor_{args.benchmark}_L{args.level}T{args.task_id}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    bridge_args = argparse.Namespace(
        benchmark=args.benchmark, level=args.level, task_id=args.task_id,
        init_state=args.init_state, seed=args.seed,
        camera_resolution=args.camera_resolution, port=args.port,
        demonstration_horizon=args.demonstration_horizon,
    )
    bridge = BridgeSession(("127.0.0.1", args.port), output_dir / "safety_bridge.log", bridge_args)
    student: Pi05StudentPolicy | None = None
    report: dict[str, Any] = {
        "schema_version": "gs_opd_task_success_first_execution.v1",
        "status": "in_progress",
        "cold_start_protocol": (
            {
                "enabled": True,
                "target_success_demonstration_used": False,
                "target_success_keyposes_used": False,
                "official_demonstration_used": False,
                "historical_actions_replayed": False,
                "initial_policy": "base_pi05",
                "experience_library_loaded": False,
                "success_reference_loaded": False,
                "previous_search_reports_loaded": False,
                "start_state": (
                    "same_run_identity_checked_partial_task_snapshot"
                    if cold_start_continuation else "identity_checked_legal_t0_snapshot"
                ),
                "continuation_snapshot": (
                    str(args.resume_from_runtime_snapshot.resolve())
                    if args.resume_from_runtime_snapshot is not None else None
                ),
                "continuation_snapshot_sha256": (
                    _sha(args.resume_from_runtime_snapshot.read_bytes())
                    if args.resume_from_runtime_snapshot is not None else None
                ),
                "continuation_placement_report": (
                    str(args.resume_from_placement_report.resolve())
                    if args.resume_from_placement_report is not None else None
                ),
                "continuation_placement_report_sha256": (
                    _sha(args.resume_from_placement_report.read_bytes())
                    if args.resume_from_placement_report is not None else None
                ),
            }
            if cold_start else {"enabled": False}
        ),
        "scope": {
            "benchmark": args.benchmark, "level": args.level, "task_id": args.task_id,
            "init_state": args.init_state, "seed": args.seed,
            "source_snapshot": str(snapshot_path), "source_snapshot_sha256": _sha(payload),
            "model_geometry_sha256": model_sha,
            "cross_scene_experience_library": str(args.experience_library) if args.experience_library else None,
        },
        "policy": {
            "student_first": True,
            "initial_policy": "base_pi05" if cold_start else "base_pi05_or_verified_prior_skill",
            "student_checkpoint": str(args.checkpoint.resolve()),
            "student_weights_only_no_LoRA": True,
            "official_safety_demonstration_data_used": False,
            "historical_student_actions_replayed": False,
            "strict_safety_reconstruction_modified": False,
            "search_traces_are_safe_training_data": False,
            "object_or_robot_state_written_directly": False,
            "all_actions_are_real_MuJoCo_steps": True,
            "collision_safety_policy": "record-only diagnostics during task-success search",
            "complete_success_predicate": "active native BDDL task_success only",
            "stage_a_near_goal_release_probe_enabled": args.stage_a_near_goal_release_probe,
            "stage_a_near_goal_release_probe_max_error_m": args.stage_a_near_goal_release_probe_max_error_m,
            "extended_In_release_probe_requires_live_aperture_projection": True,
            "stage_a_unheld_in_freefall_settle_probe_enabled": args.stage_a_unheld_in_freefall_settle_probe,
            "stage_a_near_aperture_release_probe_enabled": args.stage_a_near_aperture_release_probe,
        },
        "physics_audit_profile": {
            "contact_sampling": "all 25 MuJoCo physics substeps per control action",
            "all_geom_clearance_sweeps": (
                "disabled_for_stage_a_search"
                if args.contact_only_substep_audit else "enabled_per_physics_substep"
            ),
            "safety_certified": False,
        },
        "budgets": {
            "student_actions_before_expert_fallback": args.student_action_budget,
            "student_actions_after_verified_expert_stage": args.student_handoff_action_budget,
            "grasp_geometry_candidates": args.maximum_candidates,
            "RRT_preflights": args.maximum_rrt_preflights,
            "RRT_time_budget_s": args.rrt_time_budget_s,
            "physical_grasp_branches": args.maximum_physical_branches,
            "expected_MuJoCo_substeps_per_action": 25,
            "demonstration_horizon": args.demonstration_horizon,
            "standard_benchmark_horizon": 1000,
            "horizon_mode": "standard_benchmark" if args.demonstration_horizon == 1000 else "demonstration_only_extension",
            "transport_strategy": args.transport_strategy,
            "native_region_progress_max_step_m": args.native_region_progress_max_step_m,
            "native_region_transit_height_override_m": args.native_region_transit_height_override_m,
            "native_region_entry_mode": args.native_region_entry_mode,
            "native_region_aperture_tilt_degrees": args.native_region_aperture_tilt_degrees,
            "native_region_entry_height_offset_m": args.native_region_entry_height_offset_m,
        },
        "segments": [],
    }
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        if reset.get("scene_metadata", {}).get("model_geometry_sha256") != model_sha:
            raise RuntimeError("active_scene_model_fingerprint_differs_from_source_snapshot")
        if cold_start:
            if reset.get("scene_metadata", {}).get("bddl_sha256") != manifest.get("scene_metadata", {}).get("bddl_sha256"):
                raise RuntimeError("cold_start_fresh_reset_BDDL_differs_from_source_snapshot")
            report["cold_start_reset_verification"] = _verify_cold_start_reset(bridge, initial_state)
        initial = _restore_initial(bridge, payload, initial_state, model_sha)
        direct_resume = None
        if args.resume_from_runtime_snapshot is not None:
            direct_resume = _verified_runtime_snapshot_descriptor(
                args.resume_from_runtime_snapshot,
                benchmark=args.benchmark,
                level=args.level,
                task_id=args.task_id,
                seed=args.seed,
                init_state=args.init_state,
                model_sha=model_sha,
                bddl_sha=str(manifest.get("scene_metadata", {}).get("bddl_sha256")),
                instruction=str(manifest.get("instruction")),
            )
            restored = bridge.request({
                "op": "restore_runtime_snapshot",
                "payload": direct_resume["payload"],
            })
            actual = _integration_state(restored.get("snapshot_state", {}))
            expected = _integration_state(direct_resume["snapshot_state"])
            if (
                restored.get("status") != "restored"
                or restored.get("model_geometry_sha256") != model_sha
                or actual.shape != expected.shape
                or not np.array_equal(actual, expected)
            ):
                raise RuntimeError("direct_runtime_snapshot_exact_restore_verification_failed")
            report["resumed_from_runtime_snapshot"] = {
                "path": str(direct_resume["path"]),
                "payload_sha256": direct_resume["payload_sha256"],
                "status": "verified_exact",
                "integration_state_sha256": _state_digest(actual),
                "model_geometry_sha256": model_sha,
                "task_stage_snapshot_resume": True,
                "official_safety_demonstration_data_used": False,
                "safety_certified": False,
            }
        observed = bridge.request({"op": "observe", "include_snapshot_state": True})
        if observed.get("instruction") != manifest.get("instruction"):
            raise RuntimeError("source_Snapshot_instruction_mismatch")
        native = bridge.request({"op": "native_task_status"})
        values, success = _task_values(native)
        if success:
            raise RuntimeError("initial_t0_native_task_success_unexpectedly_true")
        report["native_goal_coverage"] = inspect_native_goal_coverage(native)
        # Match the batch scheduler so direct Snapshot resumes remain task
        # agnostic for multi-object scenes.
        active_target_predicate, target_selection_source, pending_goals = (
            _select_active_target_predicate(native, args.target_predicate)
        )
        if active_target_predicate is None:
            raise RuntimeError("no_executable_pending_native_goal_while_task_unsuccessful")
        report["target_goal_selection"] = {
            "predicate": active_target_predicate,
            "source": target_selection_source,
            "pending_native_object_goals": pending_goals,
        }
        geometry = bridge.request({"op": "geometry_inventory"})
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        gripper = bridge.request({"op": "gripper_geometry_inventory"})
        controller_inventory = bridge.request({"op": "repair_context"})
        context = bridge.request({
            "op": "task_success_first_context",
            "expected_instruction": observed["instruction"],
            "target_predicate": active_target_predicate,
        })
        if context.get("status") != "complete" or geometry.get("status") != "complete" or semantic.get("status") != "complete":
            raise RuntimeError("active_BDDL_target_or_geometry_binding_unavailable")
        target_id = str(context["target_object_id"])
        predicate = list(context["native_goal_predicate"])
        object_goal = len(predicate) == 3 and predicate[0].lower() in {"in", "on"}
        keypose_experiences: list[dict[str, Any]] = []
        keypose_transfer_rows: list[dict[str, Any]] = []
        for keypose_path in args.keypose_experience:
            keypose_experience = json.loads(keypose_path.read_text(encoding="utf-8"))
            source = keypose_experience.get("source", {})
            source_task = str(source.get("task", ""))
            if (
                keypose_experience.get("schema_version") != "gs_opd_keypose_experience.v1"
                or not any(task in source_task for task in (
                    "obstacle_avoidance/L0T2", "obstacle_avoidance/L1T2",
                ))
                or source.get("native_task_success") is not True
                or source.get("official_safety_demonstrations_used") is not False
                or keypose_experience.get("object_family") != target_id.rsplit("_", 1)[0]
                or keypose_experience.get("target_relation") != predicate[0].lower()
                or keypose_experience.get("target_region_name") != predicate[-1]
            ):
                raise ValueError("keypose_experience_source_or_active_native_goal_mismatch")
            if args.keypose_start_after is not None:
                keypose_experience["_route_start_after"] = args.keypose_start_after
            keypose_experiences.append(keypose_experience)
            keypose_transfer_rows.append({
                "path": str(keypose_path.resolve()),
                "experience_id": keypose_experience.get("experience_id"),
                "source_trace_sha256": source.get("trace_sha256"),
                "source_acceptance_report_sha256": source.get("acceptance_report_sha256"),
                "source_task": source_task,
                "source_native_task_success": source.get("native_task_success"),
                "source_safety_success": source.get("safety_success"),
                "target_match": predicate,
                "actual_source_actions_replayed": False,
                "safety_certified": False,
            })
        if keypose_transfer_rows:
            report["keypose_experience_transfers"] = keypose_transfer_rows
            report["keypose_experience_transfer"] = keypose_transfer_rows[0]
        keypose_preflight_report = None
        if args.resume_from_keypose_preflight_report is not None:
            keypose_preflight_report = json.loads(
                args.resume_from_keypose_preflight_report.read_text(encoding="utf-8")
            )
            report["keypose_preflight_reuse_source"] = str(
                args.resume_from_keypose_preflight_report.resolve()
            )
        experience_template = None
        if object_goal and args.experience_library and args.experience_library.is_file():
            from research.gs_opd.repair.task_skill_experience import (
                load_experience_library,
                matching_grasp_template,
                matching_operation_skill,
            )
            experience_library = load_experience_library(args.experience_library)
            experience_template = matching_grasp_template(experience_library, target_id)
            experience_operation_skill = matching_operation_skill(experience_library, target_id)
        else:
            experience_operation_skill = None
        experience_first = experience_template is not None
        report["experience_transfer_decision"] = {
            "library_path": str(args.experience_library) if args.experience_library else None,
            "matching_template_id": experience_template.get("template_id") if experience_template else None,
            "matching_object_family": experience_template.get("object_family") if experience_template else None,
            "matching_operation_skill_id": (
                experience_operation_skill.get("skill_id") if experience_operation_skill else None
            ),
            "historical_operation_phase_sequence": (
                experience_operation_skill.get("phase_sequence") if experience_operation_skill else []
            ),
            "historical_operation_structure_scheduled": experience_operation_skill is not None,
            "experience_attempt_before_Student": experience_first,
            "reuse_is_not_safety_certification": True,
            "official_safety_demonstration_data_used": False,
        }
        binding = semantic["bindings"][target_id]
        target_binding = {
            "object_id": target_id, "body_id": int(binding["root_body_id"]),
            "body_name": str(binding["root_body_name"]),
            "geom_ids": sorted(int(x) for x in binding["root_geom_ids"]),
            "geom_names": list(binding["root_geom_names"]),
        }
        recovery_fallback_snapshot = None
        if args.recovery_fallback_runtime_snapshot is not None:
            recovery_fallback_snapshot = _verified_runtime_snapshot_descriptor(
                args.recovery_fallback_runtime_snapshot,
                benchmark=args.benchmark,
                level=args.level,
                task_id=args.task_id,
                seed=args.seed,
                init_state=args.init_state,
                model_sha=model_sha,
                bddl_sha=str(observed.get("scene_metadata", {}).get("bddl_sha256")),
                instruction=str(observed.get("instruction")),
            )
            report["joint_limit_recovery_fallback"] = {
                "path": str(recovery_fallback_snapshot["path"]),
                "payload_sha256": recovery_fallback_snapshot["payload_sha256"],
                "model_geometry_sha256": model_sha,
                "bddl_sha256": observed.get("scene_metadata", {}).get("bddl_sha256"),
                "identity_validation": "passed",
                "bilateral_grasp_validation": "checked_after_exact_restore",
                "robot_or_object_state_written_directly": False,
                "safety_certified": False,
            }
        live_grasp_state = bridge.request({
            "op": "grasp_state", "carried_binding": target_binding,
            "expected_instruction": observed["instruction"],
        })
        live_target_grasp_verified = bool(
            object_goal
            and live_grasp_state.get("grasp_confirmed") is True
            and live_grasp_state.get("finger_contacts", {}).get("left") is True
            and live_grasp_state.get("finger_contacts", {}).get("right") is True
            and not values.get(tuple(predicate), False)
        )
        report["initial_live_target_grasp"] = {
            "verified_bilateral_grasp": live_target_grasp_verified,
            "grasp_state_status": live_grasp_state.get("status"),
            "target_finger_contacts": live_grasp_state.get("finger_contacts"),
            "resume_source": "model_verified_initial_Runtime_Snapshot" if live_target_grasp_verified else None,
        }
        if cold_start:
            bddl_text = (snapshot_root / "task.bddl").read_text(encoding="utf-8")
            if args.benchmark.startswith("libero_") and ":constraints" not in bddl_text.lower():
                policy = _build_safety_policy(context, semantic, geometry, gripper)
                policy_semantics = {"source": "generic_diagnostic_only", "BDDL_safety_constraints_declared": False}
                policy["source"] = "generic_scene_diagnostic_no_BDDL_safety_constraints_or_teacher_attestation"
                policy_source = "generic_scene_diagnostic"
            else:
                policy, policy_semantics = _build_bddl_grasp_screen_policy(context, semantic, geometry, bddl_text)
                policy["source"] = "active_BDDL_contact_constraints_no_teacher_attestation"
                policy_source = "active_BDDL"
            policy["measure_forbidden_object_pair_clearances"] = True
            policy_path = output_dir / "cold_start_safety_policy.json"
            _json(policy_path, policy)
            report["safety_policy_provenance"] = {
                "source": policy_source, "teacher_attested": False,
                "semantics": policy_semantics, "safety_certified": False,
                "policy_path": str(policy_path),
            }
        else:
            policy = _build_safety_policy(context, semantic, geometry, gripper)
        initial_export = bridge.request({"op": "export_runtime_snapshot"})
        if initial_export.get("status") != "complete" or not isinstance(initial_export.get("payload"), bytes):
            raise RuntimeError("task_success_first_initial_snapshot_export_failed")
        start_payload = initial_export["payload"]
        start_state = _snapshot_state_vector(initial_export["snapshot_state"])
        expected_start_state = (
            _integration_state(direct_resume["snapshot_state"])
            if direct_resume is not None else initial_state
        )
        if not np.array_equal(start_state, expected_start_state):
            raise RuntimeError("restored_start_state_does_not_match_verified_source_snapshot")
        controller = (
            TaskPhaseController(_controller_config(target_id, predicate, {"candidate_id": "not_yet_generated"}))
            if object_goal else None
        )
        report["task"] = {
            "instruction": observed["instruction"],
            "target_object_from_active_BDDL": target_id,
            "native_goal_predicate": predicate,
            "initial_native_task_status": native,
            "semantic_binding": binding,
            "model_geometry_sha256": model_sha,
            "BDDL_sha256": observed.get("scene_metadata", {}).get("bddl_sha256"),
            "task_phase_controller_initial_phase": controller.phase if controller is not None else None,
            "historical_operation_phase_sequence": (
                experience_operation_skill.get("phase_sequence") if experience_operation_skill else []
            ),
        }
        if (
            args.stage_a_unheld_in_freefall_settle_probe
            or args.stage_a_near_aperture_release_probe
        ):
            if not cold_start_continuation or direct_resume is None:
                raise ValueError(
                    "Stage_A_physical_release_probe_requires_an_identity_checked_Runtime_Snapshot_continuation"
                )
            settle_result = _stage_a_unheld_in_freefall_probe(
                bridge,
                target_id=target_id,
                target_binding=target_binding,
                predicate=predicate,
                instruction=observed["instruction"],
                model_sha=model_sha,
                safety_policy=policy,
                controller_fields=controller_inventory["robots"][0]["fields"],
                output_dir=output_dir / "stage_a_unheld_in_freefall_settle_probe",
                maximum_actions=(
                    args.stage_a_near_aperture_release_probe_actions
                    if args.stage_a_near_aperture_release_probe
                    else args.stage_a_unheld_in_freefall_settle_probe_actions
                ),
                allow_near_aperture_release=args.stage_a_near_aperture_release_probe,
            )
            report["segments"].append({
                "type": (
                    "stage_a_near_aperture_native_In_release_probe"
                    if args.stage_a_near_aperture_release_probe
                    else "stage_a_unheld_in_native_In_freefall_settle_probe"
                ),
                **settle_result,
            })
            report["final_native_task_status"] = settle_result.get(
                "final_native_task_status",
                bridge.request({"op": "native_task_status"}),
            )
            report["task_success"] = report["final_native_task_status"].get("task_success") is True
            report["status"] = (
                "complete_task_success" if report["task_success"]
                else settle_result.get("status", "freefall_settle_probe_incomplete")
            )
            if report["task_success"]:
                report["selected_success_episode_terminal"] = {
                    "status": "native_BDDL_success_after_physical_unheld_In_settle_probe",
                    "terminal_snapshot": settle_result.get("terminal_snapshot"),
                    "terminal_snapshot_sha256": settle_result.get("terminal_snapshot_sha256"),
                    "terminal_integration_state_sha256": settle_result.get(
                        "terminal_integration_state_sha256"
                    ),
                    "task_success": True,
                    "safety_certified": False,
                }
            report["archive"] = str(output_dir)
            _json(output_dir / "run_report.json", report)
            return report
        if args.resume_from_grasp_report is not None:
            source_report = json.loads(
                args.resume_from_grasp_report.read_text(encoding="utf-8")
            )
            resume = _verified_grasp_resume_descriptor(
                source_report,
                benchmark=args.benchmark,
                level=args.level,
                task_id=args.task_id,
                seed=args.seed,
                init_state=args.init_state,
                model_sha=model_sha,
                bddl_sha=str(observed.get("scene_metadata", {}).get("bddl_sha256")),
                target_id=target_id,
                predicate=predicate,
            )
            restored = bridge.request({
                "op": "restore_runtime_snapshot",
                "payload": resume["path"].read_bytes(),
            })
            actual_state = _integration_state(restored.get("snapshot_state", {}))
            expected_state = _integration_state(resume["snapshot_state"])
            if (
                restored.get("status") != "restored"
                or restored.get("model_geometry_sha256") != model_sha
                or actual_state.shape != expected_state.shape
                or not np.array_equal(actual_state, expected_state)
            ):
                raise RuntimeError("resume_grasp_snapshot_exact_restore_verification_failed")
            report["resumed_from_grasp_skill"] = {
                "source_report": str(args.resume_from_grasp_report),
                "snapshot": str(resume["path"]),
                "status": "verified_exact",
                "integration_state_sha256": _state_digest(actual_state),
                "model_geometry_sha256": model_sha,
                "source_task_success": False,
                "safety_certified": False,
            }
            resumed_native = bridge.request({"op": "native_task_status"})
            _, resumed_task_success = _task_values(resumed_native)
            if resumed_task_success:
                report["status"] = "complete_task_success"
                report["final_native_task_status"] = resumed_native
                report["archive"] = str(output_dir)
                _json(output_dir / "run_report.json", report)
                return report
            resumed_grasp = bridge.request({
                "op": "grasp_state",
                "carried_binding": target_binding,
                "expected_instruction": observed["instruction"],
            })
            if (
                resumed_grasp.get("grasp_confirmed") is not True
                or resumed_grasp.get("finger_contacts", {}).get("left") is not True
                or resumed_grasp.get("finger_contacts", {}).get("right") is not True
            ):
                raise RuntimeError("resume_grasp_snapshot_did_not_retain_stable_bilateral_contact")
            controller = TaskPhaseController(_controller_config(target_id, predicate, resume["success"]))
            controller.adopt_held_object(target_id)
            place_result = _native_place(
                bridge, target_id=target_id, target_binding=target_binding,
                predicate=predicate, instruction=observed["instruction"], model_sha=model_sha,
                safety_policy=policy, controller_fields=controller_inventory["robots"][0]["fields"],
                controller=controller, output_dir=output_dir / "expert_native_placement",
                keypose_experiences=keypose_experiences,
                maximum_physical_branches=args.maximum_physical_branches,
                maximum_rrt_preflights=args.maximum_rrt_preflights,
                keypose_preflight_report=keypose_preflight_report,
                demonstration_horizon=args.demonstration_horizon,
                rrt_time_budget_s=args.rrt_time_budget_s,
                recovery_fallback_snapshot=recovery_fallback_snapshot,
                transport_strategy=args.transport_strategy,
                native_region_route_variant=args.native_region_route_variant,
                native_region_candidate_offset=args.native_region_candidate_offset,
                native_region_transit_clearance_mode=args.native_region_transit_clearance_mode,
                native_region_transit_height_override_m=args.native_region_transit_height_override_m,
                native_region_progress_max_step_m=args.native_region_progress_max_step_m,
                native_region_entry_mode=args.native_region_entry_mode,
                native_region_aperture_tilt_degrees=args.native_region_aperture_tilt_degrees,
                native_region_entry_height_offset_m=args.native_region_entry_height_offset_m,
                native_region_yaw_delta_degrees=args.native_region_yaw_delta_degrees,
                expected_start_snapshot_sha256=(
                    direct_resume.get("payload_sha256") if direct_resume
                    else _sha(resume["path"].read_bytes())
                ),
                expected_bddl_sha256=str(observed.get("scene_metadata", {}).get("bddl_sha256")),
            )
            report["segments"].append({"type": "resumed_expert_native_goal_stage_control", **place_result})
            if not isinstance(place_result.get("final_native_task_status"), dict):
                place_result["final_native_task_status"] = bridge.request({"op": "native_task_status"})
            report["final_native_task_status"] = place_result["final_native_task_status"]
            report["task_success"] = report["final_native_task_status"].get("task_success") is True
            report["status"] = (
                "complete_task_success" if report["task_success"] else place_result["status"]
            )
            learned_operation = _register_verified_operation_skill_from_report(
                args.experience_library, report, output_dir / "run_report.json"
            )
            if learned_operation is not None:
                report["operation_experience_learning"] = learned_operation
            report["archive"] = str(output_dir)
            _json(output_dir / "run_report.json", report)
            return report
        if args.resume_from_placement_report is not None:
            source_report = json.loads(
                args.resume_from_placement_report.read_text(encoding="utf-8")
            )
            resume = _verified_placement_resume_descriptor(
                source_report,
                benchmark=args.benchmark,
                level=args.level,
                task_id=args.task_id,
                seed=args.seed,
                init_state=args.init_state,
                model_sha=model_sha,
                bddl_sha=str(observed.get("scene_metadata", {}).get("bddl_sha256")),
                target_id=target_id,
                predicate=predicate,
            )
            restored = bridge.request({
                "op": "restore_runtime_snapshot",
                "payload": resume["path"].read_bytes(),
            })
            actual_state = _integration_state(restored.get("snapshot_state", {}))
            expected_state = _integration_state(resume["snapshot_state"])
            if (
                restored.get("status") != "restored"
                or restored.get("model_geometry_sha256") != model_sha
                or actual_state.shape != expected_state.shape
                or not np.array_equal(actual_state, expected_state)
            ):
                raise RuntimeError("placement_failure_snapshot_exact_restore_verification_failed")
            resumed_native = bridge.request({"op": "native_task_status"})
            resumed_values, resumed_task_success = _task_values(resumed_native)
            if resumed_task_success or resumed_values.get(tuple(predicate)) is not True:
                raise RuntimeError("placement_failure_snapshot_native_goal_evidence_changed")
            resumed_grasp = bridge.request({
                "op": "grasp_state",
                "carried_binding": target_binding,
                "expected_instruction": observed["instruction"],
            })
            if (
                resumed_grasp.get("grasp_confirmed") is not True
                or resumed_grasp.get("finger_contacts", {}).get("left") is not True
                or resumed_grasp.get("finger_contacts", {}).get("right") is not True
            ):
                raise RuntimeError("placement_failure_snapshot_no_longer_has_bilateral_grasp")
            satisfied_objects = tuple(
                str(row["predicate"][1])
                for row in resumed_native.get("native_goal_values", [])
                if isinstance(row.get("predicate"), list)
                and len(row["predicate"]) == 3
                and str(row["predicate"][0]).lower() in {"in", "on"}
                and row.get("value") is True
            )
            controller = TaskPhaseController(
                _controller_config(target_id, predicate, {"candidate_id": "verified_native_goal_release_resume"})
            )
            controller.adopt_native_goal_contact_for_release(
                target_id,
                native_goal_satisfied=True,
                satisfied_object_ids=satisfied_objects,
                held_object=target_id,
                grasp_confirmed=True,
            )
            report["resumed_from_native_goal_contact_snapshot"] = {
                "source_report": str(args.resume_from_placement_report),
                "snapshot": str(resume["path"]),
                "snapshot_sha256": resume["payload_sha256"],
                "status": "verified_exact",
                "integration_state_sha256": _state_digest(actual_state),
                "model_geometry_sha256": model_sha,
                "native_goal_already_satisfied": True,
                "bilateral_grasp_retained": True,
                "next_phase": controller.phase,
                "safety_certified": False,
            }
            # This executor's stage boundary is the exact contact Snapshot,
            # after the legal t0 restore and identity-checked placement resume.
            report["task"]["initial_native_task_status"] = resumed_native
            place_result = _native_place(
                bridge, target_id=target_id, target_binding=target_binding,
                predicate=predicate, instruction=observed["instruction"], model_sha=model_sha,
                safety_policy=policy, controller_fields=controller_inventory["robots"][0]["fields"],
                controller=controller, output_dir=output_dir / "expert_native_release_resume",
                keypose_experiences=keypose_experiences,
                maximum_physical_branches=args.maximum_physical_branches,
                maximum_rrt_preflights=args.maximum_rrt_preflights,
                keypose_preflight_report=keypose_preflight_report,
                demonstration_horizon=args.demonstration_horizon,
                rrt_time_budget_s=args.rrt_time_budget_s,
                recovery_fallback_snapshot=recovery_fallback_snapshot,
                transport_strategy=args.transport_strategy,
                native_region_route_variant=args.native_region_route_variant,
                native_region_candidate_offset=args.native_region_candidate_offset,
                native_region_transit_clearance_mode=args.native_region_transit_clearance_mode,
                native_region_transit_height_override_m=args.native_region_transit_height_override_m,
                native_region_progress_max_step_m=args.native_region_progress_max_step_m,
                native_region_entry_mode=args.native_region_entry_mode,
                native_region_aperture_tilt_degrees=args.native_region_aperture_tilt_degrees,
                native_region_entry_height_offset_m=args.native_region_entry_height_offset_m,
                native_region_yaw_delta_degrees=args.native_region_yaw_delta_degrees,
                expected_start_snapshot_sha256=(direct_resume.get("payload_sha256") if direct_resume else None),
                expected_bddl_sha256=str(observed.get("scene_metadata", {}).get("bddl_sha256")),
            )
            report["segments"].append({"type": "resumed_expert_release_withdrawal", **place_result})
            if not isinstance(place_result.get("final_native_task_status"), dict):
                place_result["final_native_task_status"] = bridge.request({"op": "native_task_status"})
            report["final_native_task_status"] = place_result["final_native_task_status"]
            report["task_success"] = report["final_native_task_status"].get("task_success") is True
            report["status"] = (
                "complete_task_success" if report["task_success"] else place_result["status"]
            )
            learned_operation = _register_verified_operation_skill_from_report(
                args.experience_library, report, output_dir / "run_report.json"
            )
            if learned_operation is not None:
                report["operation_experience_learning"] = learned_operation
            report["archive"] = str(output_dir)
            _json(output_dir / "run_report.json", report)
            return report
        if live_target_grasp_verified:
            student_first = {
                "task_success": False,
                "actions_executed": 0,
                "target_displacement_m": None,
                "grasp_stability": {"stable": True, "source": "initial_Runtime_Snapshot"},
                "native_task_status": native,
                "skipped_reason": "verified_live_target_grasp_resumed_without_regrasp_or_Student_actions",
            }
            report["segments"].append({"type": "verified_live_grasp_snapshot_resume", **student_first})
        elif experience_first:
            student_first = {
                "task_success": False,
                "actions_executed": 0,
                "target_displacement_m": None,
                "grasp_stability": {},
                "native_task_status": native,
                "skipped_reason": "matching_GS_OPD_object_relative_success_skill_selected_before_Student",
                "experience_template_id": experience_template["template_id"],
            }
            report["segments"].append({"type": "student_first_skipped_transferable_experience", **student_first})
        elif args.skip_student_first:
            student_first = {
                "task_success": False,
                "actions_executed": 0,
                "target_displacement_m": None,
                "grasp_stability": {},
                "native_task_status": native,
                "skipped_reason": "Stage A bounded physical search selected; no Student prefix is needed",
            }
            report["segments"].append({"type": "student_first_skipped_for_search_efficiency", **student_first})
        else:
            student = Pi05StudentPolicy(args.checkpoint, device=args.device)
            student_first = _student_segment(
                bridge, student, max_actions=args.student_action_budget,
                target_binding=target_binding, instruction=observed["instruction"], policy=policy,
                output_dir=output_dir / "student_first", segment="first",
            )
            report["segments"].append({"type": "student_first", **student_first})
        if not object_goal:
            if student_first["task_success"]:
                report["status"] = "complete_task_success"
                report["final_native_task_status"] = student_first["native_task_status"]
            else:
                _restore_initial(bridge, start_payload, start_state, model_sha)
                articulation_context = bridge.request({
                    "op": "task_success_first_articulation_context",
                    "expected_instruction": observed["instruction"],
                    "predicate": predicate,
                })
                articulated = _run_articulated_fixture_goal(
                    bridge, predicate=predicate, target_binding=target_binding,
                    instruction=observed["instruction"], policy=policy,
                    articulation_context=articulation_context,
                    controller_fields=controller_inventory["robots"][0]["fields"],
                    output_dir=output_dir / "expert_articulated_fixture_goal",
                    model_sha=model_sha,
                    rrt_time_budget_s=args.rrt_time_budget_s,
                )
                report["segments"].append({"type": "expert_articulated_fixture_goal", **articulated})
                report["status"] = articulated["status"]
                report["final_native_task_status"] = articulated["final_native_task_status"]
                report["fixture_goal_failure_reason"] = (
                    None if articulated["native_goal_value"] is True
                    else articulated.get("failure_reason")
                    or "bounded_physical_OSC_fixture_push_did_not_satisfy_native_goal"
                )
            report["task_success"] = report["final_native_task_status"].get("task_success") is True
            report["archive"] = str(output_dir)
            _json(output_dir / "run_report.json", report)
            return report
        if live_target_grasp_verified and not student_first["task_success"]:
            controller = TaskPhaseController(
                _controller_config(target_id, predicate, {"candidate_id": "verified_live_grasp_snapshot"})
            )
            controller.adopt_held_object(target_id)
            if args.stage_a_near_goal_release_probe:
                controller.enable_near_goal_release_probe()
            place_result = _native_place(
                bridge, target_id=target_id, target_binding=target_binding,
                predicate=predicate, instruction=observed["instruction"], model_sha=model_sha,
                safety_policy=policy, controller_fields=controller_inventory["robots"][0]["fields"],
                controller=controller, output_dir=output_dir / "expert_native_goal_stage_control",
                keypose_experiences=keypose_experiences,
                maximum_physical_branches=args.maximum_physical_branches,
                maximum_rrt_preflights=args.maximum_rrt_preflights,
                keypose_preflight_report=keypose_preflight_report,
                demonstration_horizon=args.demonstration_horizon,
                rrt_time_budget_s=args.rrt_time_budget_s,
                recovery_fallback_snapshot=recovery_fallback_snapshot,
                transport_strategy=args.transport_strategy,
                native_region_route_variant=args.native_region_route_variant,
                native_region_candidate_offset=args.native_region_candidate_offset,
                native_region_transit_clearance_mode=args.native_region_transit_clearance_mode,
                native_region_transit_height_override_m=args.native_region_transit_height_override_m,
                native_region_progress_max_step_m=args.native_region_progress_max_step_m,
                native_region_entry_mode=args.native_region_entry_mode,
                native_region_aperture_tilt_degrees=args.native_region_aperture_tilt_degrees,
                native_region_entry_height_offset_m=args.native_region_entry_height_offset_m,
                native_region_yaw_delta_degrees=args.native_region_yaw_delta_degrees,
                expected_start_snapshot_sha256=(direct_resume.get("payload_sha256") if direct_resume else None),
                expected_bddl_sha256=str(observed.get("scene_metadata", {}).get("bddl_sha256")),
            )
            report["segments"].append({"type": "expert_native_goal_stage_control", **place_result})
            if not isinstance(place_result.get("final_native_task_status"), dict):
                place_result["final_native_task_status"] = bridge.request({"op": "native_task_status"})
            report["final_native_task_status"] = place_result["final_native_task_status"]
            report["task_success"] = place_result["final_native_task_status"].get("task_success") is True
            report["status"] = "complete_task_success" if report["task_success"] else place_result["status"]
            learned_operation = _register_verified_operation_skill_from_report(
                args.experience_library, report, output_dir / "run_report.json"
            )
            if learned_operation is not None:
                report["operation_experience_learning"] = learned_operation
            report["archive"] = str(output_dir)
            _json(output_dir / "run_report.json", report)
            return report
        if student_first["task_success"]:
            report["status"] = "complete_task_success"
            report["final_native_task_status"] = student_first["native_task_status"]
        else:
            _restore_initial(bridge, start_payload, start_state, model_sha)
            report["student_failure_trigger"] = {
                "triggered": True,
                "reason": (
                    "matching_successful_object_relative_skill_requires_physical_transfer_validation"
                    if experience_first else (
                        "Stage_A_Student_prefix_skipped_for_search_efficiency"
                        if args.skip_student_first
                        else "native_BDDL_task_success_not_reached_within_student_action_budget"
                    )
                ),
                "target_displacement_m": student_first["target_displacement_m"],
                "grasp_stable": student_first["grasp_stability"].get("stable"),
                "exact_t0_snapshot_restored": True,
            }
            controller.handoff_after_pregrasp()
            report["task_phase_controller_student_to_expert_handoff"] = controller.phase
            bridge.close()
            expert_root = output_dir / "expert_grasp_search"
            command = _task_success_first_grasp_search_command(
                args,
                snapshot_root=snapshot_root,
                expert_root=expert_root,
                predicate=predicate,
                resume_from_runtime_snapshot=(
                    Path(direct_resume["path"]) if direct_resume is not None else None
                ),
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
            with (output_dir / "expert_grasp_search.log").open("wb") as log:
                process = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            expert_runs = sorted(expert_root.glob("task_success_first_grasp_*"))
            if not expert_runs:
                raise RuntimeError(f"expert_grasp_search_produced_no_archive:exit={process.returncode}")
            expert_report_path = expert_runs[-1] / "run_report.json"
            expert_report = json.loads(expert_report_path.read_text(encoding="utf-8"))
            expert_failure_reasons = [
                str(value) for value in (
                    [expert_report.get("error")]
                    + [item.get("rrt_failure_reason") for item in expert_report.get("candidates", [])]
                    + [item.get("failure_reason") for item in expert_report.get("physical_branches", [])]
                ) if value
            ]
            report["segments"].append({
                "type": "bounded_expert_grasp_search",
                "status": expert_report.get("status"),
                "archive": str(expert_runs[-1]),
                "report": str(expert_report_path),
                "process_exit_code": process.returncode,
                "physical_branches": expert_report.get("physical_branches", []),
                "candidate_count": len(expert_report.get("candidates", [])),
                "failure_reasons": expert_failure_reasons,
                "success": expert_report.get("success"),
                "episode_native_success": expert_report.get("episode_native_success"),
                "snapshot_recovery_verification": expert_report.get(
                    "snapshot_recovery_verification"
                ),
                "safety_certified": False,
            })
            native_episode = expert_report.get("episode_native_success")
            native_recovery = expert_report.get("snapshot_recovery_verification", {})
            if (
                isinstance(native_episode, dict)
                and native_episode.get("native_task_status", {}).get("task_success") is True
                and native_recovery.get("status") == "pass"
                and native_recovery.get("native_task_status", {}).get("task_success") is True
            ):
                report["status"] = "complete_task_success"
                report["task_success"] = True
                report["final_native_task_status"] = native_recovery["native_task_status"]
                report["selected_success_episode_terminal"] = {
                    "status": "native_BDDL_success_after_exact_snapshot_recovery",
                    "episode_native_success": native_episode,
                    "snapshot_recovery_verification": native_recovery,
                    "safety_certified": False,
                }
                report["archive"] = str(output_dir)
                _json(output_dir / "run_report.json", report)
                return report
            if expert_report.get("success") is None:
                report["status"] = "bounded_expert_search_exhausted"
                report["failure_reason"] = (
                    expert_report.get("failure_classification")
                    or (expert_failure_reasons[0] if expert_failure_reasons else None)
                )
            else:
                success_snapshot = expert_report["success"].get("snapshot", {}).get("path")
                if not success_snapshot or not Path(success_snapshot).is_file():
                    raise RuntimeError("expert_stage_success_has_no_recoverable_runtime_snapshot")
                bridge = BridgeSession(("127.0.0.1", args.expert_port + 1), output_dir / "student_handoff_bridge.log", bridge_args)
                bridge.start()
                reset = bridge.request({"op": "reset"})
                if reset.get("scene_metadata", {}).get("model_geometry_sha256") != model_sha:
                    raise RuntimeError("student_handoff_scene_fingerprint_mismatch")
                restored = bridge.request({
                    "op": "restore_runtime_snapshot",
                    "payload": Path(success_snapshot).read_bytes(),
                })
                expected = expert_report["success"]["snapshot"].get("snapshot_state", {})
                actual_state = _integration_state(restored.get("snapshot_state", {}))
                expected_state = _integration_state(expected)
                if (
                    restored.get("status") != "restored"
                    or restored.get("model_geometry_sha256") != model_sha
                    or actual_state.shape != expected_state.shape
                    or not np.array_equal(actual_state, expected_state)
                ):
                    raise RuntimeError("expert_stage_runtime_snapshot_exact_restore_verification_failed")
                report["student_handoff_snapshot_restore"] = {
                    "status": "verified_exact",
                    "path": success_snapshot,
                    "mujoco_integration_state_sha256": _state_digest(actual_state),
                    "model_geometry_sha256": model_sha,
                }
                bridge.request({"op": "observe", "include_snapshot_state": True})
                native = bridge.request({"op": "native_task_status"})
                _, success = _task_values(native)
                if not success and not experience_first and not args.skip_student_first:
                    handoff = _student_segment(
                        bridge, student, max_actions=args.student_handoff_action_budget,
                        target_binding=target_binding, instruction=observed["instruction"], policy=policy,
                        output_dir=output_dir / "student_handoff", segment="after_expert_grasp_lift",
                    )
                    report["segments"].append({"type": "student_handoff_after_stage_goal", **handoff})
                    native = bridge.request({"op": "native_task_status"})
                    _, success = _task_values(native)
                elif not success:
                    report["student_handoff_skipped"] = (
                        "Stage_A_student_handoff_skipped_for_search_efficiency"
                        if args.skip_student_first
                        else "matched_successful_skill_transfer_keeps_expert_OSC_in_control_until_native_goal"
                    )
                if success:
                    report["status"] = "complete_task_success"
                    report["final_native_task_status"] = native
                else:
                    live_grasp = bridge.request({
                        "op": "grasp_state",
                        "carried_binding": target_binding,
                        "expected_instruction": observed["instruction"],
                    })
                    if live_grasp.get("grasp_confirmed") is not True:
                        restored = bridge.request({
                            "op": "restore_runtime_snapshot",
                            "payload": Path(success_snapshot).read_bytes(),
                        })
                        restored_state = _integration_state(restored.get("snapshot_state", {}))
                        expected_state = _integration_state(
                            expert_report["success"]["snapshot"].get("snapshot_state", {})
                        )
                        if (
                            restored.get("status") != "restored"
                            or restored.get("model_geometry_sha256") != model_sha
                            or restored_state.shape != expected_state.shape
                            or not np.array_equal(restored_state, expected_state)
                        ):
                            raise RuntimeError("failed_student_handoff_did_not_restore_verified_grasp_snapshot")
                        report["student_handoff_recovery"] = {
                            "status": "exact_expert_grasp_snapshot_restored_after_Student_lost_grasp",
                            "integration_state_sha256": _state_digest(restored_state),
                        }
                        live_grasp = bridge.request({
                            "op": "grasp_state",
                            "carried_binding": target_binding,
                            "expected_instruction": observed["instruction"],
                        })
                        if live_grasp.get("grasp_confirmed") is not True:
                            raise RuntimeError("restored_expert_snapshot_no_longer_has_verified_grasp")
                    controller = TaskPhaseController(_controller_config(target_id, predicate, expert_report["success"]))
                    controller.adopt_held_object(target_id)
                    report["student_handoff_failure_trigger"] = "native_BDDL_goal_not_reached_after_verified_grasp_lift_stage"
                    place_result = _native_place(
                        bridge, target_id=target_id, target_binding=target_binding,
                        predicate=predicate, instruction=observed["instruction"], model_sha=model_sha,
                        safety_policy=policy, controller_fields=controller_inventory["robots"][0]["fields"],
                        controller=controller, output_dir=output_dir / "expert_native_placement",
                        keypose_experiences=keypose_experiences,
                        maximum_physical_branches=args.maximum_physical_branches,
                        maximum_rrt_preflights=args.maximum_rrt_preflights,
                        keypose_preflight_report=keypose_preflight_report,
                        demonstration_horizon=args.demonstration_horizon,
                        rrt_time_budget_s=args.rrt_time_budget_s,
                        recovery_fallback_snapshot=recovery_fallback_snapshot,
                        transport_strategy=args.transport_strategy,
                        native_region_route_variant=args.native_region_route_variant,
                        native_region_candidate_offset=args.native_region_candidate_offset,
                        native_region_transit_clearance_mode=args.native_region_transit_clearance_mode,
                        native_region_transit_height_override_m=args.native_region_transit_height_override_m,
                        native_region_progress_max_step_m=args.native_region_progress_max_step_m,
                        native_region_entry_mode=args.native_region_entry_mode,
                        native_region_aperture_tilt_degrees=args.native_region_aperture_tilt_degrees,
                        native_region_entry_height_offset_m=args.native_region_entry_height_offset_m,
                        native_region_yaw_delta_degrees=args.native_region_yaw_delta_degrees,
                        expected_start_snapshot_sha256=(direct_resume.get("payload_sha256") if direct_resume else None),
                        expected_bddl_sha256=str(observed.get("scene_metadata", {}).get("bddl_sha256")),
                    )
                    report["segments"].append({"type": "expert_native_goal_stage_control", **place_result})
                    if not isinstance(place_result.get("final_native_task_status"), dict):
                        place_result["final_native_task_status"] = bridge.request({"op": "native_task_status"})
                    report["final_native_task_status"] = place_result["final_native_task_status"]
                    report["task_success"] = report["final_native_task_status"].get("task_success") is True
                    report["status"] = (
                        "complete_task_success" if report["task_success"] else place_result["status"]
                    )
                    learned_operation = _register_verified_operation_skill_from_report(
                        args.experience_library, report, output_dir / "run_report.json"
                    )
                    if learned_operation is not None:
                        report["operation_experience_learning"] = learned_operation
        report["archive"] = str(output_dir)
        _json(output_dir / "run_report.json", report)
        return report
    except Exception as exc:
        report["status"] = "executor_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["archive"] = str(output_dir)
        _json(output_dir / "run_report.json", report)
        raise
    finally:
        bridge.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark", default="obstacle_avoidance")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--target-predicate", nargs="+", default=None)
    parser.add_argument("--resume-from-grasp-report", type=Path, default=None)
    parser.add_argument("--resume-from-placement-report", type=Path, default=None)
    parser.add_argument("--resume-from-runtime-snapshot", type=Path, default=None)
    parser.add_argument("--candidate-id", action="append", default=[],
                        help="restrict the bounded grasp search to one generated geometry candidate")
    parser.add_argument(
        "--candidate-tilt-degrees", type=float, action="append", default=[],
        help="rotate selected grasp candidates about the jaw axis for a bounded physical orientation probe",
    )
    parser.add_argument(
        "--candidate-roll-degrees", type=float, action="append", default=[],
        help="rotate selected grasp candidates about the approach axis for a bounded physical orientation probe",
    )
    parser.add_argument(
        "--candidate-offset-object-z-mm", type=float, action="append", default=[],
        help=(
            "generate bounded grasp contact-height variants along object-local Z; "
            "use live MuJoCo contacts to verify any selected pose"
        ),
    )
    parser.add_argument(
        "--candidate-offset-object-y-mm", type=float, action="append", default=[],
        help=(
            "apply generated active-geometry contact-height offsets along object local Y; "
            "cold-start use is limited to same-episode Snapshot continuations"
        ),
    )
    parser.add_argument(
        "--candidate-offset-object-x-mm", type=float, action="append", default=[],
        help=(
            "apply generated active-geometry contact-patch offsets along object local X; "
            "cold-start use is limited to same-episode Snapshot continuations"
        ),
    )
    parser.add_argument(
        "--candidate-coupled-perturbations", action="store_true",
        help=(
            "combine one orientation family with one or more object-axis contact offsets "
            "for bounded physical validation"
        ),
    )
    parser.add_argument("--recovery-fallback-runtime-snapshot", type=Path, default=None,
                        help="identity-checked legal Runtime Snapshot used only if OSC joint-limit recovery fails")
    parser.add_argument("--resume-from-keypose-preflight-report", type=Path, default=None)
    parser.add_argument("--previous-search-report", type=Path, action="append", default=[])
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--demonstration-horizon", type=int, default=1000)
    parser.add_argument(
        "--contact-only-substep-audit",
        action="store_true",
        help=(
            "for Stage A, retain every MuJoCo-substep contact sample and native BDDL check "
            "while skipping exhaustive per-substep all-Geom clearance sweeps"
        ),
    )
    parser.add_argument(
        "--stage-a-near-goal-release-probe",
        action="store_true",
        help=(
            "when a verified bilateral grasp is near its unsatisfied native placement goal, "
            "physically release and let native BDDL determine whether placement succeeded"
        ),
    )
    parser.add_argument(
        "--stage-a-near-goal-release-probe-max-error-m",
        type=float,
        default=0.012,
        help=(
            "maximum object-root distance for physical release probes; values above 12 mm "
            "are honored only for In goals when the live collision footprint projects into "
            "the native region aperture above the candidate placement"
        ),
    )
    parser.add_argument(
        "--stage-a-unheld-in-freefall-settle-probe",
        action="store_true",
        help=(
            "on an exact same-episode Runtime Snapshot, open the gripper and allow a short "
            "real MuJoCo settle only when an unheld object already fits above its native In opening"
        ),
    )
    parser.add_argument(
        "--stage-a-unheld-in-freefall-settle-probe-actions",
        type=int,
        default=24,
        help="maximum real control actions in the unheld In-region freefall settle probe",
    )
    parser.add_argument(
        "--stage-a-near-aperture-release-probe",
        action="store_true",
        help=(
            "on an exact same-Episode Runtime Snapshot, physically open the gripper "
            "and observe passive MuJoCo settling when the object footprint is within "
            "15 cm of the active native In region"
        ),
    )
    parser.add_argument(
        "--stage-a-near-aperture-release-probe-actions",
        type=int,
        default=32,
        help="maximum physical open-gripper/settling actions in the near-aperture release probe",
    )
    parser.add_argument("--port", type=int, default=38569)
    parser.add_argument("--expert-port", type=int, default=38570)
    parser.add_argument("--student-action-budget", type=int, default=32)
    parser.add_argument(
        "--skip-student-first", action="store_true",
        help="start bounded Stage A physical grasp search from the verified legal t0 Snapshot",
    )
    parser.add_argument("--student-handoff-action-budget", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=16)
    parser.add_argument("--candidate-start-index", type=int, default=0)
    parser.add_argument("--geometry-candidate-pool-offset", type=int, default=0)
    parser.add_argument("--maximum-rrt-preflights", type=int, default=8)
    parser.add_argument("--local-replan-rrt-time-budget-s", type=float, default=2.0)
    parser.add_argument("--maximum-physical-branches", type=int, default=3)
    parser.add_argument("--branch-control-action-budget", type=int, default=440)
    parser.add_argument(
        "--stage-a-unilateral-close-probe-position-m", type=float, default=0.010,
        help=(
            "maximum candidate-pose error for a bounded physical close probe after a real unilateral target-pad contact; "
            "the live MuJoCo grasp monitor must still verify closure"
        ),
    )
    parser.add_argument(
        "--stage-a-pregrasp-staged-clearance-height-m", type=float, default=0.0,
        help=(
            "optional absolute world-Z clearance for the physical pregrasp route: "
            "lift with the live wrist orientation, translate above the candidate, "
            "rotate, then descend; 0 disables it"
        ),
    )
    parser.add_argument(
        "--stage-a-pregrasp-lateral-detour-m", type=float, default=0.0,
        help=(
            "at staged clearance, offset the transit corridor to one side of the "
            "direct wrist route, then return beyond the obstacle; signed meters"
        ),
    )
    parser.add_argument(
        "--stage-a-pregrasp-lift-before-lateral-detour", action="store_true",
        help=(
            "when a staged lateral detour is enabled, first lift vertically at the "
            "live wrist pose, then sidestep at clearance height"
        ),
    )
    parser.add_argument("--rrt-time-budget-s", type=float, default=20.0)
    parser.add_argument(
        "--approach-route-strategy",
        choices=("auto", "geometry_high_approach", "geometry_direct_approach"),
        default="auto",
        help=(
            "select normal scratch RRT routing, or an explicit generated-geometry high-lift/direct "
            "OSC approach after live-scene endpoint IK verification"
        ),
    )
    parser.add_argument(
        "--transport-strategy",
        choices=("native_region_progress_osc", "legacy_rrt"),
        default="native_region_progress_osc",
        help="task-success transport route; native region progress uses live object feedback with real OSC",
    )
    parser.add_argument(
        "--native-region-route-variant",
        choices=(
            "direct_region_progress", "keypose_side_region_progress",
            "keypose_side_region_progress_reverse",
            "obstacle_aware_xy_region_progress",
        ),
        default=None,
        help="force a native-region OSC route variant when resuming a physical transport failure",
    )
    parser.add_argument(
        "--native-region-candidate-offset", type=int, default=0,
        help="start placement search from this ranked native-region geometry layout",
    )
    parser.add_argument(
        "--native-region-transit-clearance-mode",
        choices=("compiled_obstacle", "reachability_biased", "surface_slide", "low_clearance"),
        default="compiled_obstacle",
        help="choose compiled-obstacle, reachability-biased, Stage A surface-slide, or low-clearance transit",
    )
    parser.add_argument(
        "--native-region-transit-height-override-m",
        type=float,
        default=None,
        help=(
            "Stage A physical-search override for world-Z object transit height; "
            "MuJoCo contacts and the native task predicate remain authoritative"
        ),
    )
    parser.add_argument(
        "--native-region-entry-mode",
        choices=("over_rim", "horizontal_aperture"),
        default="over_rim",
        help="track open-top In placement over the rim or descend outside and enter horizontally through an aperture",
    )
    parser.add_argument(
        "--native-region-aperture-tilt-degrees",
        type=float,
        default=0.0,
        help="tilt a carried object toward a compact tool posture for horizontal native In insertion",
    )
    parser.add_argument(
        "--native-region-entry-height-offset-m",
        type=float,
        default=None,
        help=(
            "minimum horizontal-aperture entry lift above the native In support height; "
            "use measured contact evidence to clear a lower lip before crossing"
        ),
    )
    parser.add_argument(
        "--native-region-yaw-delta-degrees",
        type=float,
        default=None,
        help=(
            "pin a relative yaw for native placement; In uses it for fit, while On may use "
            "any yaw to move the wrist away from a saturated joint configuration"
        ),
    )
    parser.add_argument(
        "--native-region-progress-max-step-m", type=float, default=0.008,
        help="per-action OSC target displacement limit for native-region transport; default 8 mm",
    )
    parser.add_argument("--experience-library", type=Path, default=None)
    parser.add_argument("--keypose-experience", type=Path, action="append", default=[])
    parser.add_argument("--keypose-start-after", default=None)
    parser.add_argument(
        "--cold-start", action="store_true",
        help="forbid success experiences, keyposes, candidate pins, prior reports, and partial-task snapshots",
    )
    parser.add_argument(
        "--cold-start-continuation", action="store_true",
        help="continue a cold-start Episode from its own V3-generated, identity-checked stage Snapshot or contact report",
    )
    parser.add_argument(
        "--cold-start-continuation-root", type=Path, default=None,
        help="V3 output root that must contain the same-run continuation source",
    )
    args = parser.parse_args()
    if (
        not np.isfinite(args.stage_a_near_goal_release_probe_max_error_m)
        or not 0.012 <= args.stage_a_near_goal_release_probe_max_error_m <= 0.5
    ):
        parser.error("--stage-a-near-goal-release-probe-max-error-m must be in 0.012..0.5")
    if not 1 <= args.stage_a_unheld_in_freefall_settle_probe_actions <= 120:
        parser.error("--stage-a-unheld-in-freefall-settle-probe-actions must be in 1..120")
    if not 1 <= args.stage_a_near_aperture_release_probe_actions <= 120:
        parser.error("--stage-a-near-aperture-release-probe-actions must be in 1..120")
    resume_sources = [
        args.resume_from_grasp_report,
        args.resume_from_placement_report,
        args.resume_from_runtime_snapshot,
    ]
    if sum(source is not None for source in resume_sources) > 1:
        parser.error("choose only one Runtime Snapshot resume source")
    if (
        args.resume_from_keypose_preflight_report is not None
        and args.resume_from_runtime_snapshot is None
        and args.resume_from_grasp_report is None
    ):
        parser.error(
            "--resume-from-keypose-preflight-report requires a matching Runtime Snapshot or grasp report"
        )
    for name in ("student_action_budget", "student_handoff_action_budget"):
        if not 1 <= getattr(args, name) <= 256:
            parser.error(f"--{name.replace('_', '-')} must be in 1..256")
    if not 1 <= args.maximum_candidates <= 16:
        parser.error("--maximum-candidates must be in 1..16")
    if not 0 <= args.candidate_start_index < args.maximum_candidates:
        parser.error("--candidate-start-index must select an index inside the candidate budget")
    if (
        args.geometry_candidate_pool_offset < 0
        or args.geometry_candidate_pool_offset + args.maximum_candidates > 48
    ):
        parser.error("--geometry-candidate-pool-offset plus --maximum-candidates must fit the generated geometry pool")
    if args.candidate_id and args.candidate_start_index != 0:
        parser.error("--candidate-id requires --candidate-start-index 0")
    cold_start_mode = args.cold_start or args.cold_start_continuation
    continuation_sources = (args.resume_from_runtime_snapshot, args.resume_from_placement_report)
    if args.cold_start_continuation and (
        not args.cold_start
        or sum(source is not None for source in continuation_sources) != 1
        or args.cold_start_continuation_root is None
    ):
        parser.error(
            "--cold-start-continuation requires --cold-start, exactly one resume source, "
            "and --cold-start-continuation-root"
        )
    if args.stage_a_unheld_in_freefall_settle_probe and (
        not args.cold_start_continuation
        or args.resume_from_runtime_snapshot is None
        or args.resume_from_placement_report is not None
    ):
        parser.error(
            "--stage-a-unheld-in-freefall-settle-probe requires an exact cold-start Runtime Snapshot continuation"
        )
    if args.stage_a_near_aperture_release_probe and (
        not args.cold_start_continuation
        or args.resume_from_runtime_snapshot is None
        or args.resume_from_placement_report is not None
    ):
        parser.error(
            "--stage-a-near-aperture-release-probe requires an exact cold-start Runtime Snapshot continuation"
        )
    if cold_start_mode and (
        args.experience_library is not None
        or args.keypose_experience
        or args.previous_search_report
        or args.resume_from_grasp_report is not None
        or (args.resume_from_placement_report is not None and not args.cold_start_continuation)
        or (args.resume_from_runtime_snapshot is not None and not args.cold_start_continuation)
        or args.resume_from_keypose_preflight_report is not None
        or args.recovery_fallback_runtime_snapshot is not None
        or args.candidate_start_index != 0
        # Keep geometry-derived contact-patch offsets available to a fresh-t0
        # cold start; the offsets do not import prior task experience.
    ):
        parser.error("--cold-start cannot use historical experience or partial-task inputs")
    if not 1 <= args.maximum_rrt_preflights <= 8:
        parser.error("--maximum-rrt-preflights must be in 1..8")
    if not 1 <= args.maximum_physical_branches <= 3:
        parser.error("--maximum-physical-branches must be in 1..3")
    if not 1 <= args.branch_control_action_budget <= 440:
        parser.error("--branch-control-action-budget must be in 1..440")
    if not np.isfinite(args.stage_a_unilateral_close_probe_position_m) or not (
        0.010 <= args.stage_a_unilateral_close_probe_position_m <= 0.080
    ):
        parser.error("--stage-a-unilateral-close-probe-position-m must be in 0.010..0.080")
    if (
        not np.isfinite(args.stage_a_pregrasp_staged_clearance_height_m)
        or args.stage_a_pregrasp_staged_clearance_height_m != 0.0
        and not 0.20 <= args.stage_a_pregrasp_staged_clearance_height_m <= 2.50
    ):
        parser.error(
            "--stage-a-pregrasp-staged-clearance-height-m must be 0 or in 0.20..2.50"
        )
    if (
        not np.isfinite(args.stage_a_pregrasp_lateral_detour_m)
        or not -0.40 <= args.stage_a_pregrasp_lateral_detour_m <= 0.40
    ):
        parser.error("--stage-a-pregrasp-lateral-detour-m must be in -0.40..0.40")
    if not 0 <= args.native_region_candidate_offset < 16:
        parser.error("--native-region-candidate-offset must be in 0..15")
    if not 0.5 <= args.rrt_time_budget_s <= 60:
        parser.error("--rrt-time-budget-s must be in 0.5..60")
    if not 0.5 <= args.local_replan_rrt_time_budget_s <= 60:
        parser.error("--local-replan-rrt-time-budget-s must be in 0.5..60")
    if not 0.001 <= args.native_region_progress_max_step_m <= 0.02:
        parser.error("--native-region-progress-max-step-m must be in 0.001..0.02")
    if args.native_region_transit_height_override_m is not None and (
        not np.isfinite(args.native_region_transit_height_override_m)
        or not 0.02 <= args.native_region_transit_height_override_m <= 2.50
    ):
        parser.error("--native-region-transit-height-override-m must be in 0.02..2.50")
    if not np.isfinite(args.native_region_aperture_tilt_degrees) or not 0.0 <= args.native_region_aperture_tilt_degrees <= 90.0:
        parser.error("--native-region-aperture-tilt-degrees must be in 0..90")
    if args.native_region_entry_height_offset_m is not None and (
        not np.isfinite(args.native_region_entry_height_offset_m)
        or not 0.0 <= args.native_region_entry_height_offset_m <= 0.15
        or args.native_region_entry_mode != "horizontal_aperture"
    ):
        parser.error(
            "--native-region-entry-height-offset-m must be in 0..0.15 and requires "
            "--native-region-entry-mode horizontal_aperture"
        )
    if args.native_region_yaw_delta_degrees is not None and (
        not np.isfinite(args.native_region_yaw_delta_degrees)
        or not -180.0 <= args.native_region_yaw_delta_degrees <= 180.0
    ):
        parser.error("--native-region-yaw-delta-degrees must be in -180..180")
    if args.native_region_entry_mode != "horizontal_aperture" and args.native_region_aperture_tilt_degrees != 0.0:
        parser.error("--native-region-aperture-tilt-degrees requires --native-region-entry-mode horizontal_aperture")
    if not 1000 <= args.demonstration_horizon <= MAX_DEMONSTRATION_HORIZON:
        parser.error(
            f"--demonstration-horizon must be in 1000..{MAX_DEMONSTRATION_HORIZON}"
        )
    if args.target_predicate is not None and len(args.target_predicate) not in (2, 3):
        parser.error("--target-predicate must contain two or three BDDL tokens")
    return args


def main() -> int:
    result = run(_parse_args())
    print(json.dumps({
        "status": result.get("status"),
        "archive": result.get("archive"),
        "task": result.get("task"),
        "segments": [
            {key: row.get(key) for key in ("type", "status", "actions_executed", "task_success", "archive", "recorded_actions") if key in row}
            for row in result.get("segments", [])
        ],
        "failure_reason": result.get("failure_reason") or result.get("error"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("status") == "complete_task_success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
