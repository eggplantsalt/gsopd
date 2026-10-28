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
FUTURE_OBJECT_GOAL_ACTION_RESERVE = 1200
MAX_JOINT_LIMIT_RECOVERY_ACTIONS = 8
NATIVE_PLACEMENT_RRT_BUDGET_S = 12.0
NATIVE_PLACEMENT_RRT_MAX_CONFIG_CHECKS = 8000
NATIVE_PLACEMENT_RRT_MAX_ITERATIONS = 2000


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
        "--maximum-physical-branches", str(args.maximum_physical_branches),
        "--rrt-time-budget-s", str(args.rrt_time_budget_s),
        "--demonstration-horizon", str(args.demonstration_horizon),
        "--target-predicate", *predicate,
    ]
    if resume_from_runtime_snapshot is not None:
        command.extend([
            "--resume-from-runtime-snapshot", str(resume_from_runtime_snapshot),
        ])
    candidate_id = getattr(args, "candidate_id", None)
    if candidate_id is not None:
        command.extend(["--candidate-id", str(candidate_id)])
    if args.resume_from_keypose_preflight_report is not None:
        command.extend([
            "--resume-from-keypose-preflight-report",
            str(args.resume_from_keypose_preflight_report.resolve()),
        ])
    if getattr(args, "cold_start", False):
        command.append("--cold-start")
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
    future_reserve = len(future_goals) * FUTURE_OBJECT_GOAL_ACTION_RESERVE
    available = max(0, demonstration_horizon - timestep - completion_reserve - future_reserve)
    if available == 0:
        raise RuntimeError("native_transport_action_budget_reserved_for_completion_and_pending_goals")
    estimated_actions = route_control_action_budget(
        route,
        observed_speed_m_s=observed_speed_m_s,
        minimum_actions=74,
        maximum_actions=10_000_000,
    )
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
        "timestep_at_budget_start": timestep,
        "demonstration_horizon": demonstration_horizon,
        "observed_speed_assumption_m_s": observed_speed_m_s,
        "termination_policy": "route_or_remaining_horizon_budget; no fixed per-transport action cap",
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
) -> dict[str, Any]:
    """Try a short physical OSC retreat from a near-limit state, then restore a verified hold."""
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _record_step,
        _snapshot_state_vector,
        _write_recording,
    )
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor
    from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action

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
                action = world_pose_tracking_action(
                    current_pos, current_quat, target_pos, current_quat,
                    controller_fields, gripper_action=0.0,
                ).astype(np.float32)
                before = _joint_limit_diagnostics(
                    current_robot.get("joint_qpos"), joint_names, joint_ranges,
                )
                _record_step(
                    bridge, action, "joint_limit_OSC_recovery", "native_goal_recovery",
                    action_index, target_binding, instruction, safety_policy, monitor,
                    records, frames, raise_on_done=False,
                    planning_tracking={
                        "tracking_mode": "bounded_native_OSC_pose_recovery_from_joint_limit_violation",
                        "target_eef_pose_world": np.r_[target_pos, current_quat].tolist(),
                        "joint_limits_before": before,
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


def _native_region_progress_route(
    start_object_world: np.ndarray,
    goal_object_world: np.ndarray,
    *,
    region: dict[str, Any],
    route_variant: str,
    keypose_side_region: np.ndarray | None = None,
    obstacle_clearance_height_region_m: float | None = None,
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
        or route_variant not in {
            "direct_region_progress",
            "keypose_side_region_progress",
            "keypose_side_region_progress_reverse",
        }
    ):
        raise ValueError("native_region_progress_route_requires_valid_poses_region_and_route_variant")
    rotation = region_rotation
    local_start = rotation.T @ (start[:3, 3] - region_position)
    local_goal = rotation.T @ (goal[:3, 3] - region_position)
    horizontal_delta = local_goal[:2] - local_start[:2]
    horizontal_distance = float(np.linalg.norm(horizontal_delta))
    clearance_delta = float(np.clip(
        max(0.35 * float(np.max(region_size[:2])), 0.20 * horizontal_distance),
        0.045, 0.16,
    ))
    transit_height = max(float(local_start[2]), float(local_goal[2]) + clearance_delta)
    if obstacle_clearance_height_region_m is not None:
        obstacle_height = float(obstacle_clearance_height_region_m)
        if not np.isfinite(obstacle_height):
            raise ValueError("obstacle_clearance_height_must_be_finite")
        transit_height = max(transit_height, obstacle_height)
    side = None
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
        midpoint_local = np.asarray([
            0.5 * (local_start[0] + local_goal[0]) + side[0] * offset,
            0.5 * (local_start[1] + local_goal[1]) + side[1] * offset,
            transit_height,
        ])
    else:
        midpoint_local = None

    rotation_start = Rotation.from_matrix(start[:3, :3])
    rotation_delta = (Rotation.from_matrix(goal[:3, :3]) * rotation_start.inv()).as_rotvec()
    points: list[tuple[np.ndarray, str]] = []
    needs_clearance = transit_height - float(local_start[2]) > 0.012
    combined_side_lift = bool(needs_clearance and midpoint_local is not None)
    if needs_clearance and not combined_side_lift:
        lifted = start.copy()
        lifted[:3, 3] = region_position + rotation @ np.asarray([
            local_start[0], local_start[1], transit_height,
        ])
        points.append((lifted, "adaptive_clearance"))
    if midpoint_local is not None:
        midpoint = np.eye(4, dtype=np.float64)
        midpoint[:3, :3] = start[:3, :3]
        midpoint[:3, 3] = region_position + rotation @ midpoint_local
        points.append((
            midpoint,
            "adaptive_clearance_with_keypose_side_detour"
            if combined_side_lift else "keypose_side_detour",
        ))

    transit_start = (midpoint_local if midpoint_local is not None else local_start.copy())
    transit_goal = local_goal.copy()
    transit_start[2] = transit_height
    transit_goal[2] = transit_height
    transit_distance = float(np.linalg.norm(transit_goal[:2] - transit_start[:2]))
    segment_count = max(1, int(np.ceil(transit_distance / 0.045)))
    for index in range(1, segment_count + 1):
        fraction = index / segment_count
        local = transit_start + fraction * (transit_goal - transit_start)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = region_position + rotation @ local
        pose[:3, :3] = (
            Rotation.from_rotvec(rotation_delta * fraction) * rotation_start
        ).as_matrix()
        points.append((pose, "region_progress"))
    final = goal.copy()
    points.append((final, "native_region_placement"))
    deduplicated: list[dict[str, Any]] = []
    previous = start
    for pose, role in points:
        if float(np.linalg.norm(pose[:3, 3] - previous[:3, 3])) < 0.006 and role != "native_region_placement":
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
        "adaptive_transit_height_region_m": transit_height,
        "adaptive_clearance_delta_m": clearance_delta,
        "obstacle_clearance_height_region_m": obstacle_clearance_height_region_m,
        "keypose_side_reference_region": (
            np.asarray(keypose_side_region, dtype=np.float64).tolist()
            if keypose_side_region is not None else None
        ),
        "no_fixed_preplace_pose": True,
        "waypoint_frame": "target_object_world_pose_retargeted_from_live_object_to_EEF_transform",
    }


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
    lowest_target_offset = _placement_lowest_geometry_offset(
        target_context, region_rotation, start[:3, :3],
    )
    required_root_height_world = (
        maximum_obstacle_top + float(clearance_margin_m) - lowest_target_offset
        if np.isfinite(maximum_obstacle_top) else None
    )
    required_region_height = None
    if required_root_height_world is not None:
        support_point_world = np.asarray([
            region_position[0], region_position[1], required_root_height_world,
        ], dtype=np.float64)
        required_region_height = float(
            (region_rotation.T @ (support_point_world - region_position))[2]
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
        "required_transit_height_region_m": required_region_height,
        "safety_certified": False,
    }


def _generate_native_region_placement_candidates(
    *,
    predicate: list[str],
    region: dict[str, Any],
    target_context: dict[str, Any],
    occupied_contexts: list[dict[str, Any]],
    maximum_candidates: int = 4,
) -> list[dict[str, Any]]:
    """Rank object-relative placements inside a native BDDL support/container site.

    Geometry is used to rank physical layouts, never to certify safety or reject a
    candidate solely for a conservative clearance estimate. Every selected target
    is still reached and settled by real OSC actions in MuJoCo.
    """
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
    if predicate[0].lower() == "on" and matching_occupants:
        matching_pose = matching_occupants[0].get("target_root_pose_world", {})
        matching_rotation = np.asarray(matching_pose.get("rotation"), dtype=np.float64)
        if matching_rotation.shape == (3, 3) and np.isfinite(matching_rotation).all():
            target_rotation = matching_rotation
    target_radius = _placement_footprint_radius(target_context, region_rotation, target_rotation)
    target_bottom = _placement_lowest_geometry_offset(target_context, region_rotation, target_rotation)
    verb = predicate[0].lower()
    support_z = (
        region_position[2] + region_size[2]
        if verb == "on" else region_position[2] - region_size[2]
    )
    support_local_z = float((region_rotation.T @ (np.asarray([region_position[0], region_position[1], support_z]) - region_position))[2])
    root_local_z = support_local_z - target_bottom
    matching_heights: list[float] = []
    if verb == "on":
        # Reuse a physically settled same-region height when available (for
        # example, the first member of a multi-object On goal).
        for context in occupied_contexts:
            pose = context.get("target_root_pose_world", {})
            position = np.asarray(pose.get("position_m"), dtype=np.float64)
            if position.shape == (3,) and np.isfinite(position).all():
                occupied_radius = _placement_footprint_radius(context, region_rotation)
                if abs(occupied_radius - target_radius) <= max(0.015, 0.30 * target_radius):
                    matching_heights.append(float((region_rotation.T @ (position - region_position))[2]))
        if matching_heights:
            root_local_z = float(np.median(matching_heights))

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
    for offset in sorted(offsets, key=lambda value: (value[0], value[1])):
        local_xy = np.asarray(offset, dtype=np.float64)
        position = region_position + region_rotation @ np.asarray([local_xy[0], local_xy[1], root_local_z])
        clearances = [
            float(np.linalg.norm(local_xy - other_xy) - target_radius - other_radius)
            for other_xy, other_radius, _ in occupied
        ]
        limiting_clearance = min(clearances, default=float("inf"))
        # Prefer more room from already completed objects, then the shortest
        # physical transfer from the actual held-object pose.
        travel = float(np.linalg.norm(position - target_root))
        support_centering_penalty = (
            2.0 * float(np.linalg.norm(local_xy))
            if region.get("support_geometry_kind") == "compiled_object_collision_envelope"
            else 0.0
        )
        clearance_score = limiting_clearance if np.isfinite(limiting_clearance) else 0.0
        score = clearance_score - 0.15 * travel - support_centering_penalty
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
            "rotation_world": target_rotation.tolist(),
            "region_local_xy_m": local_xy.tolist(),
            "entry_direction_world": entry.tolist(),
            "estimated_completed_object_clearance_m": limiting_clearance if np.isfinite(limiting_clearance) else None,
            "support_centering_penalty_m": support_centering_penalty,
            "travel_distance_m": travel,
            "ranking_score": score,
            "root_height_source": "settled_same_region_object" if matching_heights and verb == "on" else "compiled_collision_geometry_support_extent",
            "orientation_source": "compatible_completed_object_pose" if verb == "on" and matching_occupants else "current_object_pose",
            "geometry_candidate_only_not_safety_certified": True,
        })
    rows.sort(key=lambda row: (-float(row["ranking_score"]), float(row["travel_distance_m"]), row["candidate_id"]))
    selected = rows[:maximum_candidates]
    # Placement layouts and their entrances form one bounded search portfolio.
    # Rotate the object/region-relative entrance around the nearest completed
    # object so a blocked route can try a perpendicular or opposite side.
    direction_angles = (0.0, 90.0, -90.0, 180.0)
    for index, row in enumerate(selected):
        base = region_rotation.T @ np.asarray(row["entry_direction_world"], dtype=np.float64)
        angle = np.deg2rad(direction_angles[index % len(direction_angles)])
        c, s = float(np.cos(angle)), float(np.sin(angle))
        rotated = np.asarray([c * base[0] - s * base[1], s * base[0] + c * base[1], 0.0])
        row["entry_direction_world"] = (region_rotation @ rotated).tolist()
        row["entry_direction_variant_degrees"] = direction_angles[index % len(direction_angles)]
        row["route_strategy"] = "jointspace_rrt" if index % 2 == 0 else "elevated_osc_waypoints"
    return selected


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
) -> dict[str, Any]:
    """Physically push a native articulated fixture goal through OSC_POSE."""
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _advance_pose,
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
    current = observed["snapshot_state"]["robot_states"][0]
    quaternion = np.asarray(current["eef_quat"], dtype=np.float64)
    _advance_pose.controller_fields = controller_fields
    counter = [0]
    precontact = path[0] + outward * 0.075
    contact_offset = outward * 0.035
    waypoints = [precontact, path[0] + contact_offset]
    waypoints.extend(point + contact_offset for point in path[1:])
    for waypoint in waypoints:
        _advance_pose(
            bridge, waypoint, quaternion, f"articulated_fixture_{predicate[0].lower()}",
            f"native_{predicate[0].lower()}_{predicate[1]}", counter,
            max(1, 128 - counter[0]), target_binding, instruction, policy,
            monitor, records, frames,
        )
        native = bridge.request({"op": "native_task_status"})
        if any(row.get("predicate") == predicate and row.get("value") is True
               for row in native.get("native_goal_values", [])):
            break
        if counter[0] >= 128:
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
    return {
        "status": "complete_task_success" if final_native.get("task_success") is True
        else "partial_predicate_completed" if predicate_value is True
        else "articulated_fixture_goal_incomplete",
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
    demonstration_horizon: int = 1000,
    rrt_time_budget_s: float = NATIVE_PLACEMENT_RRT_BUDGET_S,
) -> dict[str, Any]:
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
                "require_safe_configurations": route_strategy == "jointspace_rrt", "seed": 74103,
            },
            "ik_seed_qpos": _placement_ik_seed_portfolio(robot),
        })
        if route_strategy == "jointspace_rrt":
            planned["planning_mode"] = "task_success_collision_aware_zero_extra_margin"
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
            "collision_constraint": "verified full robot and carried-object geometry; zero extra clearance; no penetration",
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
    durable_records_path = output_dir / "durable_physical_step_records.jsonl"
    durable_frames_dir = output_dir / "durable_dual_camera_frames"
    durable_frames_dir.mkdir(parents=True, exist_ok=True)
    contact_escape: dict[str, Any] | None = None

    def persist_step(record: dict[str, Any], frame: dict[str, Any]) -> None:
        # Preserve each executed action before the phase returns. The final
        # bundle still writes compressed records, NPZ and MP4; these files make
        # the physical prefix recoverable if a long RRT/OSC stage is interrupted.
        with durable_records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                record, ensure_ascii=False, separators=(",", ":"),
                default=lambda item: item.tolist() if isinstance(item, np.ndarray) else str(item),
            ))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        combined = np.concatenate((frame["agentview_rgb"], frame["wrist_rgb"]), axis=1)
        frame_path = durable_frames_dir / f"frame_{int(frame['step_index']):06d}.jpg"
        if not cv2.imwrite(str(frame_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("durable_dual_camera_frame_write_failed")
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
        transport_route_source = "verified_native_goal_snapshot_resume_at_release"
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
        region_progress_route = _native_region_progress_route(
            object_world,
            goal_object,
            region=region,
            route_variant=route_variant,
            keypose_side_region=keypose_side_reference_region,
            obstacle_clearance_height_region_m=(
                obstacle_clearance.get("required_transit_height_region_m")
            ),
        )
        region_progress_route["compiled_obstacle_clearance"] = obstacle_clearance
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
        )
    )
    transport_action_budget = int(transport_budget_details["actions"])

    try:
        if route_strategy == "native_region_progress_osc":
            if region_progress_route is None:
                raise RuntimeError("native_region_progress_route_was_not_generated")
            region_position = np.asarray(region["site_position_world_m"], dtype=np.float64)
            region_rotation = np.asarray(region["site_rotation_world"], dtype=np.float64)
            goal_local = np.asarray(region_progress_route["goal_region_local_m"], dtype=np.float64)
            flat_distance = lambda position: float(np.linalg.norm(
                (region_rotation.T @ (np.asarray(position, dtype=np.float64) - region_position))[:2]
                - goal_local[:2]
            ))
            stagnant_segments = 0
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
                target_object_world = np.asarray(
                    waypoint_row["object_pose_world"], dtype=np.float64,
                ).copy()
                role = str(waypoint_row["role"])
                distance_before = flat_distance(live_object_world[:3, 3])
                if role == "region_progress":
                    local_current = region_rotation.T @ (live_object_world[:3, 3] - region_position)
                    delta = goal_local[:2] - local_current[:2]
                    distance_to_goal = float(np.linalg.norm(delta))
                    if distance_to_goal <= 0.035:
                        continue
                    segment_m = min(0.045, distance_to_goal)
                    next_local = local_current.copy()
                    next_local[:2] += delta * (segment_m / distance_to_goal)
                    next_local[2] = float(region_progress_route["adaptive_transit_height_region_m"])
                    target_object_world[:3, 3] = region_position + region_rotation @ next_local
                target_eef_world = target_object_world @ live_object_to_eef
                action_budget_remaining = max(0, transport_action_budget - counter[0])
                if action_budget_remaining <= 0:
                    raise RuntimeError("native_region_progress_transport_horizon_budget_exhausted")
                # The short waypoint is bounded by the route-derived remaining
                # budget; OSC feedback stops at the waypoint or on real stagnation.
                segment_steps = action_budget_remaining
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
                    "soft_keypose_reference_id": (
                        keypose_experience.get("experience_id") if keypose_experience else None
                    ),
                    "keypose_is_a_soft_route_reference": keypose_experience is not None,
                    "fixed_high_preplace_required": False,
                    "RRT_or_IK_timeout_is_terminal": False,
                    "physical_collision_response_enabled": True,
                    "safety_certified": False,
                }
                _advance_pose(
                    bridge, target_eef_world[:3, 3], _wxyz(target_eef_world[:3, :3]),
                    f"transport_region_progress_{role}", "native_region_progress",
                    counter, segment_steps, target_binding, instruction,
                    safety_policy, monitor, records, frames,
                    step_callback=persist_step,
                    stagnation_window_actions=24,
                    position_tolerance_m=0.015,
                    rotation_tolerance_rad=0.12,
                    planning_tracking=progress_context,
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
                }
                region_progress_history.append(row)
                if not row["bilateral_grasp_confirmed"]:
                    raise RuntimeError("native_region_progress_transport_lost_bilateral_grasp")
                if role == "region_progress":
                    stagnant_segments = stagnant_segments + 1 if row["goal_distance_progress_m"] < 0.004 else 0
                    if stagnant_segments >= 3:
                        raise RuntimeError("native_region_progress_transport_object_goal_stagnated_across_three_segments")
                if row["goal_distance_progress_m"] > 0.003 or role in {
                    "adaptive_clearance", "keypose_side_detour",
                    "adaptive_clearance_with_keypose_side_detour", "native_region_placement",
                }:
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
        if route_strategy == "native_region_progress_osc":
            route = route[:1]
        for waypoint_index, waypoint in enumerate(route[1:]):
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
) -> dict[str, Any]:
    """Search distinct native-region layouts from one exact held-object Snapshot."""
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
            demonstration_horizon=demonstration_horizon,
            rrt_time_budget_s=rrt_time_budget_s,
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
        maximum_candidates=4,
    )
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
            candidate["route_variant"] = (
                native_region_route_variant
                or (
                    "direct_region_progress" if variant_index % 2 == 0
                    else "keypose_side_region_progress"
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
        candidates = candidates[:max(1, min(2, int(maximum_physical_branches)))]
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
        # the existing full-geometry MuJoCo 7D planner. Do not turn the second
        # branch into an obstacle-blind elevated OSC shortcut.
        for candidate in candidates:
            candidate["route_strategy"] = "jointspace_rrt"
        candidates = candidates[:max(1, min(2, int(maximum_physical_branches)))]
        keypose_preflight = [{
            "status": "native_region_layout_candidates_selected_for_full_geometry_RRT",
            "candidate_ids": [row["candidate_id"] for row in candidates],
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

    def restore_branch_start() -> None:
        try:
            restored = bridge.request({"op": "restore_runtime_snapshot", "payload": base_payload})
        except (EOFError, BrokenPipeError, ConnectionError, OSError):
            # A terminated Bridge loses only its live session. Restart the
            # same compiled scene, then recover the last verified physical
            # prefix from the exported Runtime Snapshot before retrying.
            bridge.close()
            bridge.start()
            reset = bridge.request({"op": "reset"})
            if reset.get("scene_metadata", {}).get("model_geometry_sha256") != model_sha:
                raise RuntimeError("restarted_placement_branch_model_fingerprint_mismatch")
            restored = bridge.request({"op": "restore_runtime_snapshot", "payload": base_payload})
        restored_state = _integration_state(restored.get("snapshot_state", {}))
        if (
            restored.get("status") != "restored"
            or restored.get("model_geometry_sha256") != model_sha
            or restored_state.shape != base_state.shape
            or not np.array_equal(restored_state, base_state)
        ):
            raise RuntimeError("placement_candidate_snapshot_restore_not_exact")
        restored_native = bridge.request({"op": "native_task_status"})
        restored_values, restored_success = _task_values(restored_native)
        if restored_success or restored_values != values:
            raise RuntimeError("placement_candidate_restore_native_predicates_changed")
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

    physical_branch_budget = max(1, min(3, int(maximum_physical_branches)))
    for ordinal, candidate in enumerate(candidates[:physical_branch_budget]):
        if ordinal:
            try:
                restore_branch_start()
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
                demonstration_horizon=demonstration_horizon,
                rrt_time_budget_s=rrt_time_budget_s,
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
            "keypose_RRT_waypoints": result.get("keypose_RRT_waypoints", []),
            "recorded_actions": result.get("recorded_actions", result.get("transport_actions", 0)),
            "native_task_success": native_final.get("task_success", result.get("native_task_success")),
            "target_predicate_value": next((
                row.get("value") for row in native_final.get("native_goal_values", [])
                if row.get("predicate") == predicate
            ), None),
            "terminal_snapshot": result.get("terminal_snapshot"),
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
    from research.gs_opd.repair.automatic_recovery_runtime import Pi05StudentPolicy
    from research.gs_opd.repair.task_success_first_grasp_search import (
        BridgeSession, _PythonObjectState, _build_safety_policy, _build_bddl_grasp_screen_policy, _restore_initial,
        _snapshot_state_vector, _verify_cold_start_reset,
    )

    cold_start = bool(getattr(args, "cold_start", False))
    if cold_start and (
        args.experience_library is not None
        or args.keypose_experience
        or args.previous_search_report
        or args.resume_from_grasp_report is not None
        or args.resume_from_placement_report is not None
        or args.resume_from_runtime_snapshot is not None
        or args.resume_from_keypose_preflight_report is not None
        or args.recovery_fallback_runtime_snapshot is not None
        or args.candidate_id is not None
        or args.candidate_start_index != 0
        or args.native_region_route_variant is not None
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
                "start_state": "identity_checked_legal_t0_snapshot",
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
                )
                report["segments"].append({"type": "expert_articulated_fixture_goal", **articulated})
                report["status"] = articulated["status"]
                report["final_native_task_status"] = articulated["final_native_task_status"]
                report["fixture_goal_failure_reason"] = (
                    None if articulated["native_goal_value"] is True
                    else "bounded_physical_OSC_fixture_push_did_not_satisfy_native_goal"
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
                    if experience_first else "native_BDDL_task_success_not_reached_within_student_action_budget"
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
                "safety_certified": False,
            })
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
                if not success and not experience_first:
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
                        "matched_successful_skill_transfer_keeps_expert_OSC_in_control_until_native_goal"
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
    parser.add_argument("--candidate-id", default=None,
                        help="restrict the bounded grasp search to one generated geometry candidate")
    parser.add_argument("--recovery-fallback-runtime-snapshot", type=Path, default=None,
                        help="identity-checked legal Runtime Snapshot used only if OSC joint-limit recovery fails")
    parser.add_argument("--resume-from-keypose-preflight-report", type=Path, default=None)
    parser.add_argument("--previous-search-report", type=Path, action="append", default=[])
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--demonstration-horizon", type=int, default=1000)
    parser.add_argument("--port", type=int, default=38569)
    parser.add_argument("--expert-port", type=int, default=38570)
    parser.add_argument("--student-action-budget", type=int, default=32)
    parser.add_argument("--student-handoff-action-budget", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=16)
    parser.add_argument("--candidate-start-index", type=int, default=0)
    parser.add_argument("--maximum-rrt-preflights", type=int, default=8)
    parser.add_argument("--maximum-physical-branches", type=int, default=3)
    parser.add_argument("--rrt-time-budget-s", type=float, default=20.0)
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
        ),
        default=None,
        help="force a native-region OSC route variant when resuming a physical transport failure",
    )
    parser.add_argument("--experience-library", type=Path, default=None)
    parser.add_argument("--keypose-experience", type=Path, action="append", default=[])
    parser.add_argument("--keypose-start-after", default=None)
    parser.add_argument(
        "--cold-start", action="store_true",
        help="forbid success experiences, keyposes, candidate pins, prior reports, and partial-task snapshots",
    )
    args = parser.parse_args()
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
    if args.candidate_id is not None and args.candidate_start_index != 0:
        parser.error("--candidate-id requires --candidate-start-index 0")
    if args.cold_start and (
        args.experience_library is not None
        or args.keypose_experience
        or args.previous_search_report
        or any(source is not None for source in resume_sources)
        or args.resume_from_keypose_preflight_report is not None
        or args.recovery_fallback_runtime_snapshot is not None
        or args.candidate_id is not None
        or args.candidate_start_index != 0
        or args.native_region_route_variant is not None
    ):
        parser.error("--cold-start cannot use historical experience or partial-task inputs")
    if not 1 <= args.maximum_rrt_preflights <= 8:
        parser.error("--maximum-rrt-preflights must be in 1..8")
    if not 1 <= args.maximum_physical_branches <= 3:
        parser.error("--maximum-physical-branches must be in 1..3")
    if not 0.5 <= args.rrt_time_budget_s <= 60:
        parser.error("--rrt-time-budget-s must be in 0.5..60")
    if not 1000 <= args.demonstration_horizon <= 3600:
        parser.error("--demonstration-horizon must be in 1000..3600")
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
