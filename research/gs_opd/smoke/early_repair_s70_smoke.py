"""Probe small smooth S70 XYZ detours without Student inference or A0_75.

All candidates are generated from the S70 bowl/bottle nearest-Geom direction,
branched from one Snapshot, checked by the existing Contact Compiler and Repair
constraints, and compared against A0 over the same five-action task horizon.
Clearance margins are exploratory filters, not safety guarantees or training
labels.
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.repair.verifier_guided_repair import (
    RepairConfig,
    _candidate_constraints,
    _controller_fields,
    build_repair_actions,
    evaluate_task_progress,
)
from research.gs_opd.smoke.contact_safety_compiler_smoke import (
    AUTHKEY,
    BRIDGE_PORT,
    _complete_monitoring,
    compile_forbid_contact,
)
from research.gs_opd.smoke.cross_chunk_contact_smoke import compare_snapshot_diagnostics
from research.gs_opd.smoke.near_far_gap_smoke import (
    DEFAULT_ARCHIVE as DEFAULT_ARCHIVE_NEAR_FAR,
    DEFAULT_BINDINGS as DEFAULT_BINDINGS_NEAR_FAR,
    DEFAULT_TEACHER as DEFAULT_TEACHER_NEAR_FAR,
    STEP_70,
    _read_json,
    _resolve_bound_geom_sets,
    _select_forbid_pair,
    _strict_state_check,
)
from research.gs_opd.smoke.replay_archive_contact_smoke import (
    _validate_current_bindings,
    load_archive_episode,
)
from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


DEFAULT_ARCHIVE = DEFAULT_ARCHIVE_NEAR_FAR
DEFAULT_TEACHER = DEFAULT_TEACHER_NEAR_FAR
DEFAULT_BINDINGS = DEFAULT_BINDINGS_NEAR_FAR
DEFAULT_OUT = Path("logs/gs_opd/early_repair_s70.json")
CHUNK_LENGTH = 5
STATE_ATOL = 1e-6
MAX_CANDIDATES = 36
AMPLITUDES_M = (0.008, 0.016, 0.024, 0.032)
WINDOWS = ((1, 3), (2, 2))  # geometry-centered smoothing envelopes; not collision times
GAP_THRESHOLDS_M = (0.005, 0.010, 0.020)
MIN_GAP_IMPROVEMENT_M = 1e-6
# This is a conservative heuristic for preserving an already demonstrated
# grasp across a short detour. It is evidence for task continuity, not a
# gripper-contact or task-success proof.
MAX_GRASP_RELATIVE_DISTANCE_CHANGE_M = 0.05
OBJECT_ID = "object_1"
OBSTACLE_ID = "object_4"

PROBE_CONFIG = RepairConfig(
    minimum_amplitude_m=0.002,
    maximum_amplitude_m_per_step=0.032,
    maximum_integrated_detour_m=0.08,
    windows=WINDOWS,
    max_candidate_evaluations=MAX_CANDIDATES,
)


class EarlyRepairInputError(ValueError):
    """Raised when the archived S70 decision cannot support the probe."""


def _request(connection: Any, payload: dict[str, Any]) -> dict[str, Any]:
    connection.send(payload)
    response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError(f"Bridge returned a non-object for {payload.get('op')!r}")
    return response


def load_s70_archive(archive_path: Path) -> dict[str, Any]:
    """Load only the S70 chunk and its executed prefix; never extract A0_75."""
    try:
        episode = load_archive_episode(archive_path, decision_step=STEP_70)
    except Exception as exc:
        raise EarlyRepairInputError(str(exc)) from exc
    actions = np.asarray(episode["action_chunk"], dtype=np.float32)
    if actions.shape != (CHUNK_LENGTH, 7) or not np.isfinite(actions).all():
        raise EarlyRepairInputError(f"S70 A0 must be finite with shape (5, 7), got {actions.shape}")
    return {
        "archive": str(archive_path.resolve()),
        "instruction": episode["instruction"],
        "episode": episode,
        "actions": actions,
        "archive_checks": {
            "decision_step": episode["decision_step"],
            "decision_index": episode["decision_index"],
            "a0_70_shape": list(actions.shape),
            "a0_70_matches_archived_executed_actions": True,
            "a0_75_actions_extracted_or_used": False,
        },
    }


def _branch_candidate(container: Any, candidate_id: str) -> dict[str, Any] | None:
    if not isinstance(container, dict) or not isinstance(container.get("candidates"), list):
        return None
    return next(
        (
            item for item in container["candidates"]
            if isinstance(item, dict) and item.get("candidate_id") == candidate_id
        ),
        None,
    )


def _compiler_candidate(container: Any, candidate_id: str) -> dict[str, Any] | None:
    if not isinstance(container, dict) or not isinstance(container.get("candidates"), list):
        return None
    return next(
        (
            item for item in container["candidates"]
            if isinstance(item, dict) and item.get("candidate_id") == candidate_id
        ),
        None,
    )


def _find_endpoint_pair(distance: dict[str, Any], subject_body_id: int, obstacle_body_id: int) -> dict[str, Any]:
    if (
        distance.get("query_complete") is not True
        or distance.get("distance_reliable") is not True
        or not isinstance(distance.get("minimum_pair"), dict)
    ):
        raise ValueError("A0 chunk endpoint distance query is unavailable or incomplete")
    pair = distance["minimum_pair"]
    geom_a, geom_b = pair.get("geom_a"), pair.get("geom_b")
    if not isinstance(geom_a, dict) or not isinstance(geom_b, dict):
        raise ValueError("Endpoint minimum Geom pair metadata is incomplete")
    if geom_a.get("body_id") == subject_body_id and geom_b.get("body_id") == obstacle_body_id:
        ordered = pair
    elif geom_b.get("body_id") == subject_body_id and geom_a.get("body_id") == obstacle_body_id:
        ordered = {
            **pair,
            "geom_a": geom_b,
            "geom_b": geom_a,
            "closest_segment_world": list(pair.get("closest_segment_world", [])[3:])
            + list(pair.get("closest_segment_world", [])[:3]),
        }
    else:
        raise ValueError("Nearest Geom pair does not match the validated subject/obstacle Body IDs")
    segment = np.asarray(ordered.get("closest_segment_world"), dtype=np.float64)
    if segment.shape != (6,) or not np.isfinite(segment).all():
        raise ValueError("Nearest Geom closest-point segment is malformed or non-finite")
    return {
        "distance_m": distance.get("minimum_distance_m"),
        "geom_a": ordered["geom_a"],
        "geom_b": ordered["geom_b"],
        "closest_point_on_subject_world": segment[:3].tolist(),
        "closest_point_on_obstacle_world": segment[3:].tolist(),
        "closest_segment_world": segment.tolist(),
        "mesh_approximation": bool(distance.get("mesh_approximation")),
    }


def derive_clearance_directions(
    distance: dict[str, Any],
    subject_body_id: int,
    obstacle_body_id: int,
    world_up: Any,
    baseline_eef_positions: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Get away and two lateral directions from the supplied A0 endpoint geometry."""
    pair = _find_endpoint_pair(distance, subject_body_id, obstacle_body_id)
    subject_point = np.asarray(pair["closest_point_on_subject_world"], dtype=np.float64)
    obstacle_point = np.asarray(pair["closest_point_on_obstacle_world"], dtype=np.float64)
    away_raw = subject_point - obstacle_point
    away_norm = float(np.linalg.norm(away_raw))
    if not np.isfinite(away_norm) or away_norm <= 1e-10:
        raise ValueError("Nearest-point segment has no resolvable away direction")
    away = away_raw / away_norm
    up_array = np.asarray(world_up, dtype=np.float64)
    if up_array.shape != (3,) or not np.isfinite(up_array).all() or np.linalg.norm(up_array) <= 1e-10:
        raise ValueError("Active MuJoCo world-up vector is unavailable")
    up = up_array / np.linalg.norm(up_array)
    lateral = np.cross(up, away)
    lateral_source = "cross(world_up, nearest_geom_away)"
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm <= 1e-10 and baseline_eef_positions is not None:
        positions = np.asarray(baseline_eef_positions, dtype=np.float64)
        if positions.ndim == 2 and positions.shape[1] == 3 and len(positions) >= 2:
            motion = positions[-1] - positions[0]
            lateral = np.cross(motion, away)
            lateral_norm = float(np.linalg.norm(lateral))
            lateral_source = "cross(A0 EEF motion, nearest_geom_away)"
    if lateral_norm <= 1e-10:
        basis = np.eye(3)[int(np.argmin(np.abs(away)))]
        lateral = np.cross(basis, away)
        lateral_norm = float(np.linalg.norm(lateral))
        lateral_source = "cross(least_aligned_world_axis, nearest_geom_away)"
    if not np.isfinite(lateral_norm) or lateral_norm <= 1e-10:
        raise ValueError("No stable lateral direction can be constructed")
    lateral /= lateral_norm
    directions = [
        {"name": "nearest_geom_away", "direction_world": away.tolist(), "basis_source": "subject closest point minus obstacle closest point"},
        {"name": "nearest_geom_lateral_positive", "direction_world": lateral.tolist(), "basis_source": lateral_source},
        {"name": "nearest_geom_lateral_negative", "direction_world": (-lateral).tolist(), "basis_source": lateral_source},
    ]
    return directions, {
        "source": "current A0 chunk endpoint nearest-point geometry; no future action or collision event used",
        "nearest_pair": pair,
        "nearest_point_separation_norm_m": away_norm,
        "mesh_approximation": pair["mesh_approximation"],
        "directions": directions,
        "collision_evidence_used": False,
    }


def _osc_input_bounds_check(actions: Any, controller_context: dict[str, Any]) -> dict[str, Any]:
    action_array = np.asarray(actions, dtype=np.float64)
    try:
        fields = controller_context["robots"][0]["fields"]
        low = np.asarray(fields["input_min"], dtype=np.float64).reshape(-1)
        high = np.asarray(fields["input_max"], dtype=np.float64).reshape(-1)
        if action_array.ndim != 2 or action_array.shape[1] != 7 or low.shape != (6,) or high.shape != (6,):
            raise ValueError("expected (T,7) actions and six OSC input bounds")
        finite = bool(np.isfinite(action_array).all() and np.isfinite(low).all() and np.isfinite(high).all())
        if not finite:
            raise ValueError("non-finite action or OSC bound")
        outside = np.argwhere(
            (action_array[:, :6] < low[None, :] - 1e-9)
            | (action_array[:, :6] > high[None, :] + 1e-9)
        )
        return {
            "status": "pass" if not len(outside) else "fail",
            "checked_dimensions": "OSC translation and rotation inputs (0:6)",
            "input_min": low.tolist(),
            "input_max": high.tolist(),
            "out_of_bounds_indices": outside.tolist(),
            "gripper_dimension_preserved_from_A0": True,
            "gripper_input_bounds_source": "not exposed by repair_context; candidate keeps A0 gripper value unchanged",
        }
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def generate_clearance_candidates(
    original_actions: Any,
    directions: list[dict[str, Any]],
    controller_context: dict[str, Any],
    config: RepairConfig = PROBE_CONFIG,
    *,
    max_candidates: int = MAX_CANDIDATES,
    candidate_prefix: str = "early-s70",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use the existing smooth Repair action builder for a bounded probe grid."""
    actions = np.asarray(original_actions, dtype=np.float64)
    if actions.shape != (CHUNK_LENGTH, 7) or not np.isfinite(actions).all():
        raise ValueError("A0 must be a finite (5,7) action chunk")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError(f"max_candidates must be in [1, {MAX_CANDIDATES}]")
    if not isinstance(candidate_prefix, str) or not candidate_prefix.strip():
        raise ValueError("candidate_prefix must be a non-empty string")
    bound_check = _osc_input_bounds_check(actions, controller_context)
    if bound_check.get("status") != "pass":
        raise ValueError(f"A0 does not satisfy active OSC input bounds: {bound_check}")
    if not directions:
        return [], [{"reason": "no_geometric_repair_directions"}]
    specs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counter = 0
    for direction in directions:
        for anchor, (lead, tail) in zip((1, 2), WINDOWS):
            for amplitude in AMPLITUDES_M:
                if counter >= max_candidates:
                    return specs, rejected
                candidate_id = f"{candidate_prefix}-{counter + 1:03d}"
                # build_repair_actions requires a field named violation_step to
                # center its envelope. Here the value is only a geometric
                # smoothing anchor; no violation event is invented or implied.
                repaired, diagnostics = build_repair_actions(
                    actions,
                    direction.get("direction_world"),
                    amplitude,
                    violation_step=anchor,
                    lead_steps=lead,
                    tail_steps=tail,
                    controller_context=controller_context,
                    config=config,
                )
                counter += 1
                window = diagnostics.get("window")
                if isinstance(window, dict) and "violation_step" in window:
                    window["window_anchor_step"] = window.pop("violation_step")
                diagnostics["window_semantics"] = (
                    "smooth geometry-probe window center; it is not a collision or violation step"
                )
                if repaired is None or diagnostics.get("valid") is not True:
                    rejected.append({
                        "candidate_id": candidate_id,
                        "direction": direction.get("name"),
                        "amplitude_m_per_control_step": amplitude,
                        "window": diagnostics.get("window"),
                        "reason": diagnostics.get("reason", "repair_action_builder_rejected_candidate"),
                        "action_diagnostics": diagnostics,
                    })
                    continue
                candidate_array = np.asarray(repaired, dtype=np.float64)
                candidate_bounds = _osc_input_bounds_check(candidate_array, controller_context)
                if candidate_bounds.get("status") != "pass":
                    rejected.append({
                        "candidate_id": candidate_id,
                        "direction": direction.get("name"),
                        "amplitude_m_per_control_step": amplitude,
                        "window": diagnostics.get("window"),
                        "reason": "candidate_exceeds_active_osc_input_bounds",
                        "input_bounds_check": candidate_bounds,
                    })
                    continue
                if not np.array_equal(candidate_array[:, 3:], actions[:, 3:]):
                    rejected.append({
                        "candidate_id": candidate_id,
                        "reason": "candidate_changed_rotation_or_gripper_action",
                    })
                    continue
                specs.append({
                    "candidate_id": candidate_id,
                    "actions": candidate_array,
                    "direction": direction,
                    "amplitude_m_per_control_step": amplitude,
                    "window": diagnostics.get("window"),
                    "action_diagnostics": diagnostics,
                    "input_bounds_check": candidate_bounds,
                })
    return specs, rejected


def _complete_five_steps(candidate: Any) -> bool:
    return bool(
        isinstance(candidate, dict)
        and candidate.get("status") == "ok"
        and candidate.get("requested_steps") == CHUNK_LENGTH
        and candidate.get("executed_steps") == CHUNK_LENGTH
        and isinstance(candidate.get("steps"), list)
        and len(candidate["steps"]) == CHUNK_LENGTH
    )


def evaluate_same_horizon_progress(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    original_actions: np.ndarray,
    controller_context: dict[str, Any],
    task_object_binding: dict[str, Any],
    config: RepairConfig = PROBE_CONFIG,
) -> dict[str, Any]:
    """Use the existing task-progress verifier over exactly all five actions."""
    if not _complete_five_steps(baseline) or not _complete_five_steps(candidate):
        return {
            "status": "unavailable",
            "reason": "A0_and_candidate_must_both_complete_the_same_five_action_horizon",
            "aligned_prefix_steps_compared": None,
            "comparison_horizon_steps": CHUNK_LENGTH,
        }
    result = evaluate_task_progress(
        baseline,
        candidate,
        original_actions,
        violation_step=CHUNK_LENGTH,
        controller_context=controller_context,
        task_object_binding=task_object_binding,
        config=config,
    )
    result["comparison_horizon_steps"] = CHUNK_LENGTH
    result["aligned_prefix_steps_compared"] = CHUNK_LENGTH
    result["comparison_semantics"] = "A0 and candidate both execute actions 0..4 from the same decision Snapshot"
    result["eef_displacement_is_not_task_success"] = True
    return result


def _task_object_eef_path(
    candidate: dict[str, Any], body_id: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, str]:
    """Read the bound object's and EEF's start plus five post-action states."""
    if not _complete_five_steps(candidate):
        return None, "candidate_did_not_complete_five_steps"
    start = candidate.get("start_state")
    steps = candidate.get("steps")
    checkpoints = [start, *[step.get("state") if isinstance(step, dict) else None for step in steps]]
    eef_positions: list[np.ndarray] = []
    object_positions: list[np.ndarray] = []
    for index, state in enumerate(checkpoints):
        if not isinstance(state, dict):
            return None, f"state_{index}_missing"
        robot_states = state.get("robot_states")
        object_states = state.get("object_states")
        if not isinstance(robot_states, list) or not robot_states or not isinstance(robot_states[0], dict):
            return None, f"eef_state_{index}_missing"
        if not isinstance(object_states, dict):
            return None, f"object_states_{index}_missing"
        matching_objects = [
            item
            for item in object_states.values()
            if isinstance(item, dict)
            and isinstance(item.get("body_id"), (int, np.integer))
            and not isinstance(item.get("body_id"), (bool, np.bool_))
            and int(item["body_id"]) == body_id
        ]
        if len(matching_objects) != 1:
            return None, f"bound_task_object_state_{index}_missing_or_ambiguous"
        try:
            eef = np.asarray(robot_states[0]["eef_pos"], dtype=np.float64)
            obj = np.asarray(matching_objects[0]["root_pos"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return None, f"eef_or_task_object_position_{index}_missing_or_invalid"
        if eef.shape != (3,) or obj.shape != (3,) or not np.isfinite(eef).all() or not np.isfinite(obj).all():
            return None, f"eef_or_task_object_position_{index}_missing_or_invalid"
        eef_positions.append(eef)
        object_positions.append(obj)
    return np.stack(eef_positions), np.stack(object_positions)


def _candidate_reward_and_done(candidate: dict[str, Any]) -> tuple[list[float], list[bool]] | tuple[None, str]:
    rewards: list[float] = []
    dones: list[bool] = []
    for index, step in enumerate(candidate["steps"]):
        if not isinstance(step, dict):
            return None, f"step_{index}_record_missing"
        try:
            reward = float(step["reward"])
        except (KeyError, TypeError, ValueError):
            return None, f"step_{index}_reward_missing_or_invalid"
        done = step.get("done")
        if not np.isfinite(reward) or not isinstance(done, (bool, np.bool_)):
            return None, f"step_{index}_reward_or_done_invalid"
        rewards.append(reward)
        dones.append(bool(done))
    return rewards, dones


def evaluate_early_repair_task_feasibility(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    task_object_binding: dict[str, Any],
    config: RepairConfig = PROBE_CONFIG,
    *,
    original_direction_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check same-horizon detour feasibility without requiring A0-direction progress.

    The former 70% root-position projection remains diagnostic only. Passing
    requires complete states, a bound task object, reward/termination checks,
    and baseline-supported EEF/object co-motion that remains within a bounded
    relative-distance change. This is a short-horizon grasp-preservation
    heuristic, not a success label.

    ``evaluation_complete`` means the required states and reward/done signals
    were available for assessment. A complete assessment may still return
    ``unavailable`` when those observations do not establish a reliable grasp.
    """
    result: dict[str, Any] = {
        "status": "unavailable",
        "evaluation_complete": False,
        "reason": None,
        "basis": "same_horizon_grasp_preservation_and_no_reward_loss",
        "aligned_prefix_steps_compared": CHUNK_LENGTH,
        "comparison_horizon_steps": CHUNK_LENGTH,
        "high_quality_training_label_created": False,
        "original_a0_direction_70_percent_diagnostic": {
            **(original_direction_diagnostic or {}),
            "used_as_hard_gate": False,
        },
        "grasp_evidence": {
            "status": "unavailable",
            "relative_distance_tolerance_m": MAX_GRASP_RELATIVE_DISTANCE_CHANGE_M,
            "limitation": "EEF/object kinematic co-motion is heuristic evidence and does not prove a physical grasp.",
        },
    }
    if not _complete_five_steps(baseline) or not _complete_five_steps(candidate):
        result["reason"] = "A0_and_candidate_must_both_complete_the_same_five_action_horizon"
        return result
    if (
        not isinstance(task_object_binding, dict)
        or isinstance(task_object_binding.get("body_id"), (bool, np.bool_))
        or not isinstance(task_object_binding.get("body_id"), (int, np.integer))
        or int(task_object_binding["body_id"]) < 0
        or task_object_binding.get("status") not in (None, "matched_by_iou")
    ):
        result["reason"] = "task_object_physical_binding_missing_or_invalid"
        return result
    body_id = int(task_object_binding["body_id"])
    result["task_object_binding"] = {
        "body_id": body_id,
        "object_id": task_object_binding.get("object_id"),
        "validation_status": task_object_binding.get("status"),
    }
    baseline_path = _task_object_eef_path(baseline, body_id)
    candidate_path = _task_object_eef_path(candidate, body_id)
    if baseline_path[0] is None or candidate_path[0] is None:
        result["reason"] = (
            "baseline_" + baseline_path[1]
            if baseline_path[0] is None
            else "candidate_" + candidate_path[1]
        )
        result["object_and_eef_states_complete"] = False
        return result
    result["object_and_eef_states_complete"] = True
    baseline_signals = _candidate_reward_and_done(baseline)
    candidate_signals = _candidate_reward_and_done(candidate)
    if baseline_signals[0] is None or candidate_signals[0] is None:
        result["reason"] = (
            "baseline_" + baseline_signals[1]
            if baseline_signals[0] is None
            else "candidate_" + candidate_signals[1]
        )
        return result
    result["evaluation_complete"] = True
    baseline_eef, baseline_object = baseline_path
    candidate_eef, candidate_object = candidate_path
    baseline_relative = np.linalg.norm(baseline_object - baseline_eef, axis=1)
    candidate_relative = np.linalg.norm(candidate_object - candidate_eef, axis=1)
    baseline_object_path_m = float(np.linalg.norm(np.diff(baseline_object, axis=0), axis=1).sum())
    baseline_eef_path_m = float(np.linalg.norm(np.diff(baseline_eef, axis=0), axis=1).sum())
    baseline_relative_range_m = float(np.max(baseline_relative) - np.min(baseline_relative))
    relative_error = np.abs(candidate_relative - baseline_relative)
    max_relative_error_m = float(np.max(relative_error))
    grasp_reliable = bool(
        baseline_object_path_m >= config.minimum_object_motion_m
        and baseline_eef_path_m >= config.minimum_eef_progress_m
        and baseline_relative_range_m <= MAX_GRASP_RELATIVE_DISTANCE_CHANGE_M
    )
    result["grasp_evidence"] = {
        "status": "reliable_heuristic" if grasp_reliable else "unreliable",
        "method": "same_horizon_baseline_object_eef_co_motion_and_relative_distance",
        "baseline_object_path_length_m": baseline_object_path_m,
        "baseline_eef_path_length_m": baseline_eef_path_m,
        "baseline_relative_distance_range_m": baseline_relative_range_m,
        "candidate_max_aligned_relative_distance_error_m": max_relative_error_m,
        "relative_distance_tolerance_m": MAX_GRASP_RELATIVE_DISTANCE_CHANGE_M,
        "baseline_motion_thresholds_m": {
            "task_object": config.minimum_object_motion_m,
            "eef": config.minimum_eef_progress_m,
        },
        "limitation": "This kinematic heuristic cannot distinguish every supported-object configuration from a grasp.",
    }
    if not grasp_reliable:
        result["reason"] = "baseline_grasp_evidence_unreliable"
        return result
    if max_relative_error_m > MAX_GRASP_RELATIVE_DISTANCE_CHANGE_M:
        result["status"] = "fail"
        result["reason"] = "candidate_task_object_eef_relative_position_changed_abnormally"
        return result

    baseline_rewards, _ = baseline_signals
    candidate_rewards, candidate_dones = candidate_signals
    baseline_total = float(sum(baseline_rewards))
    candidate_total = float(sum(candidate_rewards))
    baseline_prefix = np.cumsum(baseline_rewards)
    candidate_prefix = np.cumsum(candidate_rewards)
    reward_loss_steps = [
        index
        for index, (base_reward, candidate_reward) in enumerate(zip(baseline_prefix, candidate_prefix))
        if base_reward > 0.0 and candidate_reward + 1e-9 < base_reward
    ]
    baseline_positive_reward_preserved = bool(
        baseline_total <= 0.0 or candidate_total + 1e-9 >= baseline_total
    ) and not reward_loss_steps
    result["reward_preservation"] = {
        "status": "pass" if baseline_positive_reward_preserved else "fail",
        "baseline_reward_sum": baseline_total,
        "candidate_reward_sum": candidate_total,
        "baseline_positive_reward_exists": any(reward > 0.0 for reward in baseline_rewards),
        "aligned_prefix_reward_loss_steps": reward_loss_steps,
        "preserved": baseline_positive_reward_preserved,
    }
    first_done = next((index for index, done in enumerate(candidate_dones) if done), None)
    if first_done is not None:
        reward_through_done = float(sum(candidate_rewards[: first_done + 1]))
        result["termination"] = {
            "first_done_step": first_done,
            "reward_through_done": reward_through_done,
            "positive_success_reward_present": reward_through_done > 0.0,
        }
        if reward_through_done <= 0.0:
            result["status"] = "fail"
            result["reason"] = "candidate_terminated_without_positive_success_reward"
            return result
    else:
        result["termination"] = {"first_done_step": None, "positive_success_reward_present": False}
    if not baseline_positive_reward_preserved:
        result["status"] = "fail"
        result["reason"] = "candidate_did_not_preserve_existing_positive_task_reward"
        return result
    result["status"] = "pass"
    result["reason"] = None
    return result


def _restore_and_compare(connection: Any, snapshot_id: str, snapshot_state: dict[str, Any]) -> dict[str, Any]:
    response = _request(connection, {"op": "restore", "snapshot_id": snapshot_id})
    if response.get("error"):
        return {"matched": False, "error": response["error"]}
    if response.get("restored_snapshot_id") != snapshot_id:
        return {"matched": False, "error": "Bridge restored a different Snapshot ID"}
    actual = response.get("snapshot_state")
    if not isinstance(snapshot_state, dict) or not isinstance(actual, dict):
        return {"matched": False, "error": "Restore omitted Snapshot diagnostics"}
    return compare_snapshot_diagnostics(snapshot_state, actual, STATE_ATOL)


def _compare_mujoco_integration_state(
    expected_diagnostics: Any,
    actual_diagnostics: Any,
    atol: float = STATE_ATOL,
) -> dict[str, Any]:
    """Strictly compare the complete MuJoCo integration-state vectors."""
    field = "mujoco_integration_state"
    if not isinstance(expected_diagnostics, dict) or not isinstance(actual_diagnostics, dict):
        return {
            "matched": False,
            "field": field,
            "error": "diagnostics must be objects",
            "atol": atol,
        }
    if field not in expected_diagnostics or field not in actual_diagnostics:
        return {
            "matched": False,
            "field": field,
            "error": "MuJoCo integration state is missing from one or both diagnostics",
            "atol": atol,
        }
    try:
        expected = np.asarray(expected_diagnostics[field], dtype=np.float64)
        actual = np.asarray(actual_diagnostics[field], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return {
            "matched": False,
            "field": field,
            "error": f"integration state is not numeric: {type(exc).__name__}: {exc}",
            "atol": atol,
        }
    same_shape = expected.shape == actual.shape
    finite = bool(np.isfinite(expected).all() and np.isfinite(actual).all())
    max_abs_error = (
        float(np.max(np.abs(expected - actual)))
        if same_shape and expected.size
        else (0.0 if same_shape else None)
    )
    matched = (
        expected.ndim == 1
        and actual.ndim == 1
        and same_shape
        and finite
        and bool(np.allclose(expected, actual, rtol=0.0, atol=atol))
    )
    return {
        "matched": bool(matched),
        "field": field,
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "same_shape": bool(same_shape),
        "finite": finite,
        "max_abs_error": max_abs_error,
        "atol": atol,
        "rtol": 0.0,
    }


def _distance_response_summary(response: dict[str, Any]) -> dict[str, Any]:
    value = response.get("minimum_distance_m")
    available = bool(
        response.get("query_complete") is True
        and response.get("distance_reliable") is True
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    return {
        "distance_available": available,
        "minimum_distance_m": float(value) if available else None,
        "query_status": response.get("status", "unavailable"),
        "query_complete": response.get("query_complete") is True,
        "distance_reliable": response.get("distance_reliable") is True,
        "pair_count_requested": response.get("pair_count_requested"),
        "pair_count_attempted": response.get("pair_count_attempted"),
        "pair_count_censored_at_distmax": response.get("pair_count_censored_at_distmax"),
        "minimum_pair": response.get("minimum_pair"),
        "mesh_approximation": response.get("mesh_approximation"),
        "warnings": response.get("warnings", []),
        "errors": response.get("errors", response.get("pair_errors", [])[:10]),
        "not_a_safety_proof": True,
    }


def replay_and_measure_endpoint(
    connection: Any,
    snapshot_id: str,
    snapshot_state: dict[str, Any],
    actions: np.ndarray,
    branch_candidate: dict[str, Any],
    subject_geom_ids: list[int],
    obstacle_geom_ids: list[int],
    distmax_m: float = 10.0,
) -> dict[str, Any]:
    """Sequentially replay a complete branch only to query its endpoint gap."""
    result: dict[str, Any] = {
        "status": "incomplete",
        "requested_steps": CHUNK_LENGTH,
        "executed_steps": 0,
        "endpoint_distance": None,
        "restore_before_replay": None,
        "branch_vs_sequential_endpoint": None,
        "branch_vs_sequential_integration_state": None,
        "restore_after_replay": None,
    }
    final_restore_attempted = False
    try:
        result["restore_before_replay"] = _restore_and_compare(connection, snapshot_id, snapshot_state)
        if result["restore_before_replay"].get("matched") is not True:
            result["reason"] = "could_not_restore_snapshot_before_endpoint_replay"
            return result
        final_response = None
        for index, action in enumerate(actions):
            final_response = _request(
                connection,
                {
                    "op": "step",
                    "action": np.asarray(action, dtype=np.float64).tolist(),
                    "include_snapshot_state": index == CHUNK_LENGTH - 1,
                },
            )
            if final_response.get("error"):
                result["reason"] = f"sequential_endpoint_replay_step_{index}_failed:{final_response['error']}"
                break
            result["executed_steps"] += 1
            if final_response.get("done") is True and index + 1 < CHUNK_LENGTH:
                result["reason"] = f"sequential_endpoint_replay_terminated_at_step_{index}"
                break
        if result["executed_steps"] == CHUNK_LENGTH and final_response is not None:
            branch_steps = branch_candidate.get("steps")
            branch_state = branch_steps[-1].get("state") if isinstance(branch_steps, list) else None
            live_state = final_response.get("snapshot_state")
            result["branch_vs_sequential_endpoint"] = compare_snapshot_diagnostics(
                branch_state, live_state, STATE_ATOL
            )
            result["branch_vs_sequential_endpoint"]["use_for_endpoint_distance_gate"] = False
            result["branch_vs_sequential_endpoint"]["interpretation"] = (
                "full diagnostics retained; derived kinematics cache differences are informational"
            )
            result["branch_vs_sequential_integration_state"] = _compare_mujoco_integration_state(
                branch_state, live_state, STATE_ATOL
            )
            if result["branch_vs_sequential_integration_state"].get("matched") is True:
                response = _request(
                    connection,
                    {
                        "op": "geom_set_min_distance",
                        "geom_ids_a": subject_geom_ids,
                        "geom_ids_b": obstacle_geom_ids,
                        "distmax_m": distmax_m,
                    },
                )
                result["endpoint_distance"] = _distance_response_summary(response)
                result["status"] = "complete" if result["endpoint_distance"].get("distance_available") else "distance_unavailable"
            else:
                result["reason"] = "branch_and_sequential_mujoco_integration_states_differ_at_1e-6"
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        final_restore_attempted = True
        try:
            result["restore_after_replay"] = _restore_and_compare(connection, snapshot_id, snapshot_state)
        except Exception as exc:
            result["restore_after_replay"] = {"matched": False, "error": f"{type(exc).__name__}: {exc}"}
        result["final_restore_attempted"] = final_restore_attempted
        result["main_state_restored"] = result["restore_after_replay"].get("matched") is True
        if not result["main_state_restored"]:
            result["status"] = "restore_failed"
            result["reason"] = "Main-state restoration failed after endpoint replay"
    return result


def _monitor_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    monitor = candidate.get("contact_monitoring")
    if not isinstance(monitor, dict):
        return {"status": "unavailable"}
    return {
        "status": monitor.get("status"),
        "sampling_point": monitor.get("sampling_point"),
        "expected_physics_substeps_total": monitor.get("expected_physics_substeps_total"),
        "captured_physics_substeps": monitor.get("captured_physics_substeps"),
        "unobserved_expected_substeps": monitor.get("unobserved_expected_substeps"),
        "unavailable_contact_substeps": monitor.get("unavailable_contact_substeps"),
        "contact_substeps": monitor.get("contact_substeps"),
        "no_contact_substeps": monitor.get("no_contact_substeps"),
        "events_count": len(monitor.get("events", [])) if isinstance(monitor.get("events"), list) else None,
    }


def assess_candidate(
    spec: dict[str, Any],
    branch_candidate: dict[str, Any] | None,
    compiled_candidate: dict[str, Any] | None,
    baseline_candidate: dict[str, Any],
    controller_context: dict[str, Any],
    task_object_binding: dict[str, Any],
    config: RepairConfig,
    baseline_gap_m: float | None,
    endpoint_replay: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fail-closed gates for one full candidate; no train-label semantics."""
    result: dict[str, Any] = {
        "candidate_id": spec["candidate_id"],
        "actions": np.asarray(spec["actions"], dtype=np.float64).tolist(),
        "direction": spec["direction"],
        "amplitude_m_per_control_step": spec["amplitude_m_per_control_step"],
        "window": spec["window"],
        "window_semantics": spec["action_diagnostics"].get("window_semantics"),
        "action_diagnostics": spec["action_diagnostics"],
        "input_bounds_check": spec["input_bounds_check"],
        "training_label_valid": False,
        "final_a_plus": False,
        "rejection_reasons": [],
    }
    if branch_candidate is None:
        result["rejection_reasons"].append("branch_result_missing")
    else:
        result["branch_status"] = branch_candidate.get("status")
        result["requested_steps"] = branch_candidate.get("requested_steps")
        result["executed_steps"] = branch_candidate.get("executed_steps")
        result["contact_monitoring"] = _monitor_summary(branch_candidate)
        full = _complete_five_steps(branch_candidate)
        result["full_five_step_chunk"] = full
        monitor, monitor_error = _complete_monitoring(branch_candidate)
        result["contact_monitoring_complete"] = monitor_error is None
        if monitor_error:
            result["rejection_reasons"].append(monitor_error)
        if not full:
            result["rejection_reasons"].append("candidate_did_not_execute_all_five_actions")
        if full:
            actions = np.asarray(spec["actions"], dtype=np.float64)
            controller, controller_error = _controller_fields(controller_context)
            if controller is None:
                result["action_constraints"] = {"status": "unavailable", "reason": controller_error}
            else:
                result["action_constraints"] = _candidate_constraints(
                    np.asarray(spec["original_actions"], dtype=np.float64),
                    actions,
                    spec["action_diagnostics"],
                    branch_candidate,
                    controller,
                    config,
                )
                range_result = _osc_input_bounds_check(actions, controller_context)
                result["action_constraints"]["all_osc_input_bounds"] = range_result
                if range_result.get("status") != "pass" and result["action_constraints"].get("status") == "pass":
                    result["action_constraints"].update({"status": range_result.get("status"), "reason": "candidate_osc_input_bounds_invalid"})
            if result.get("action_constraints", {}).get("status") != "pass":
                result["rejection_reasons"].append(result.get("action_constraints", {}).get("reason", "action_constraints_not_passed"))
            original_direction_progress = evaluate_same_horizon_progress(
                baseline_candidate,
                branch_candidate,
                np.asarray(spec["original_actions"], dtype=np.float64),
                controller_context,
                task_object_binding,
                config,
            )
            legacy_object_progress = original_direction_progress.get("task_object_progress")
            result["task_progress"] = evaluate_early_repair_task_feasibility(
                baseline_candidate,
                branch_candidate,
                task_object_binding,
                config,
                original_direction_diagnostic={
                    "legacy_evaluator_status": original_direction_progress.get("status"),
                    "task_object_root_pos_projection": legacy_object_progress,
                    "required_progress_fraction": config.required_object_progress_fraction,
                },
            )
            if result["task_progress"].get("status") != "pass":
                result["rejection_reasons"].append(result["task_progress"].get("reason", "task_progress_not_passed"))
    verdict = compiled_candidate.get("verdict") if isinstance(compiled_candidate, dict) else None
    result["compiler"] = {
        "verdict": verdict or "unknown",
        "reason": compiled_candidate.get("reason") if isinstance(compiled_candidate, dict) else "compiler_candidate_missing",
        "object_contact_checks": compiled_candidate.get("object_contact_checks", []) if isinstance(compiled_candidate, dict) else [],
    }
    result["physical_safety_verdict"] = verdict or "unknown"
    if verdict != "safe":
        result["rejection_reasons"].append("contact_compiler_did_not_return_safe")
    result["endpoint_replay"] = endpoint_replay
    gap = endpoint_replay.get("endpoint_distance") if isinstance(endpoint_replay, dict) else None
    result["endpoint_gap_m"] = (
        gap.get("minimum_distance_m")
        if isinstance(gap, dict) and gap.get("distance_available") is True
        else None
    )
    result["baseline_a0_endpoint_gap_m"] = baseline_gap_m
    result["endpoint_gap_change_vs_a0_m"] = (
        float(result["endpoint_gap_m"]) - float(baseline_gap_m)
        if isinstance(result["endpoint_gap_m"], (int, float))
        and isinstance(baseline_gap_m, (int, float))
        else None
    )
    result["endpoint_gap_quality"] = (
        "mesh_approximation" if isinstance(gap, dict) and gap.get("mesh_approximation") else "MuJoCo geometric estimate"
    ) if isinstance(gap, dict) and gap.get("distance_available") is True else "unavailable"
    if not isinstance(endpoint_replay, dict) or endpoint_replay.get("status") != "complete" or endpoint_replay.get("main_state_restored") is not True:
        result["rejection_reasons"].append("candidate_endpoint_gap_or_snapshot_restore_unavailable")
    if result["endpoint_gap_change_vs_a0_m"] is None or result["endpoint_gap_change_vs_a0_m"] <= MIN_GAP_IMPROVEMENT_M:
        result["rejection_reasons"].append("candidate_did_not_increase_endpoint_gap_over_a0")
    result["minimum_required_gap_increase_m"] = MIN_GAP_IMPROVEMENT_M
    result["accepted_for_probe"] = not result["rejection_reasons"]
    if result["accepted_for_probe"]:
        result["acceptance_reason"] = "complete_contact_monitoring_safe_detour_feasibility_and_constraints_pass"
    return result


def clearance_margin_counts(candidate_reports: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [item for item in candidate_reports if isinstance(item.get("endpoint_gap_m"), (int, float))]
    probe_eligible = [item for item in measured if item.get("accepted_for_probe") is True]
    return {
        "thresholds_m": list(GAP_THRESHOLDS_M),
        "thresholds_mm": [threshold * 1000.0 for threshold in GAP_THRESHOLDS_M],
        "measured_candidate_count": len(measured),
        "counts_by_margin": {
            f"{threshold * 1000:.0f}mm": {
                "all_measured_candidates": sum(float(item["endpoint_gap_m"]) >= threshold for item in measured),
                "safe_progress_preserving_probe_candidates": sum(float(item["endpoint_gap_m"]) >= threshold for item in probe_eligible),
            }
            for threshold in GAP_THRESHOLDS_M
        },
        "margin_is_exploratory_not_a_safety_proof": True,
    }


def _compiler_candidate_summary(result: dict[str, Any], candidate_id: str) -> dict[str, Any] | None:
    selected = _compiler_candidate(result, candidate_id)
    if selected is None:
        return None
    return selected


def run_experiment(
    connection: Any,
    archive: dict[str, Any],
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    *,
    subject_object_id: str = OBJECT_ID,
    obstacle_object_id: str = OBSTACLE_ID,
    distmax_m: float = 10.0,
    max_candidates: int = MAX_CANDIDATES,
) -> dict[str, Any]:
    """Replay S70, validate A0 safety, then probe bounded geometric detours."""
    actions = np.asarray(archive["actions"], dtype=np.float64)
    report: dict[str, Any] = {
        "status": "failed",
        "archive": archive["archive"],
        "decision_step": STEP_70,
        "instruction": archive["instruction"],
        "original_a0_actions": actions.tolist(),
        "archive_checks": archive["archive_checks"],
        "forecast_source": "S70 A0 only; no A0_75 extracted or used",
        "candidate_limit": min(max_candidates, MAX_CANDIDATES),
        "gap_thresholds_m": list(GAP_THRESHOLDS_M),
        "training_label_valid": False,
        "final_a_plus": None,
    }
    snapshot_id = None
    snapshot_state = None
    try:
        report["semantic_rule"] = _select_forbid_pair(teacher, subject_object_id, obstacle_object_id)
        reset = _request(connection, {"op": "reset"})
        if reset.get("error"):
            report.update({"status": "reset_failed", "failure_reason": reset["error"]})
            return report
        if reset.get("instruction") != archive["instruction"]:
            report.update({"status": "task_instruction_mismatch", "failure_reason": "Bridge reset instruction differs from archive"})
            return report

        inventory = _request(connection, {"op": "geometry_inventory"})
        gripper_inventory = _request(connection, {"op": "gripper_geometry_inventory"})
        controller_context = _request(connection, {"op": "repair_context"})
        report["geometry_inventory_status"] = inventory.get("status")
        report["repair_context_status"] = controller_context.get("status")
        binding_validation = _validate_current_bindings(teacher, semantic_results, inventory)
        report["semantic_binding_validation"] = binding_validation
        if not binding_validation.get("valid"):
            report.update({"status": "binding_validation_failed", "failure_reason": "Teacher Body/Geom bindings do not match active Geometry Inventory"})
            return report
        resolved_objects, subject_geom_ids, obstacle_geom_ids = _resolve_bound_geom_sets(
            binding_validation, inventory, subject_object_id, obstacle_object_id
        )
        teacher_objects = {item["id"]: item for item in teacher.get("objects", []) if isinstance(item, dict)}
        for object_id, binding in resolved_objects.items():
            binding["teacher_name"] = teacher_objects.get(object_id, {}).get("name")
        report["resolved_objects"] = resolved_objects

        task_binding = dict(binding_validation["objects"][subject_object_id])
        task_binding.update({"status": "matched_by_iou", "role": teacher_objects[subject_object_id]["role"]})
        report["task_progress_binding"] = task_binding
        action_bounds = _osc_input_bounds_check(actions, controller_context)
        report["a0_controller_bounds"] = action_bounds
        if action_bounds.get("status") != "pass":
            report.update({"status": "a0_input_bounds_unavailable_or_invalid", "failure_reason": "A0 does not satisfy active OSC input bounds"})
            return report

        prefix = np.asarray(archive["episode"]["replay_actions"], dtype=np.float32)
        if prefix.shape != (STEP_70, 7) or not np.isfinite(prefix).all():
            report.update({"status": "archive_prefix_invalid", "failure_reason": "S70 must have exactly 70 finite replay actions"})
            return report
        for index, action in enumerate(prefix):
            response = _request(connection, {"op": "step", "action": action.tolist()})
            if response.get("error"):
                report.update({"status": "s70_replay_failed", "failure_reason": f"step {index}: {response['error']}"})
                return report
            if response.get("done") is True and index + 1 < STEP_70:
                report.update({"status": "s70_replay_terminated_early", "failure_reason": f"Bridge ended after step {index}"})
                return report
        state_check = _strict_state_check(archive["episode"]["expected_observation"], response)
        report["s70_state_check"] = state_check
        if not state_check["matched"]:
            report.update({"status": "s70_state_mismatch", "failure_reason": "S70 instruction or robot state differs from archive at 1e-6"})
            return report

        snapshot = _request(connection, {"op": "snapshot"})
        if snapshot.get("error") or not isinstance(snapshot.get("snapshot_id"), str):
            report.update({"status": "snapshot_failed", "failure_reason": snapshot.get("error", "Snapshot ID missing")})
            return report
        snapshot_id = snapshot["snapshot_id"]
        snapshot_state = snapshot.get("snapshot_state")
        if not isinstance(snapshot_state, dict):
            report.update({"status": "snapshot_failed", "failure_reason": "Snapshot state diagnostic missing"})
            return report
        report["snapshot_id"] = snapshot_id

        baseline_id = "baseline-a0-s70"
        baseline_branch_result = _request(connection, {
            "op": "branch_rollout",
            "snapshot_id": snapshot_id,
            "candidates": [{"candidate_id": baseline_id, "actions": actions.tolist()}],
        })
        baseline_compile = compile_forbid_contact(
            teacher,
            subject_object_id,
            semantic_results[subject_object_id],
            baseline_branch_result,
            inventory,
            gripper_inventory,
            object_semantic_results=semantic_results,
        )
        baseline_candidate = _branch_candidate(baseline_branch_result, baseline_id)
        baseline_compiled = _compiler_candidate(baseline_compile, baseline_id)
        report["baseline_a0"] = {
            "branch_restored_main_state": baseline_branch_result.get("main_state_restored") is True,
            "full_five_step_chunk": _complete_five_steps(baseline_candidate),
            "contact_monitoring": _monitor_summary(baseline_candidate) if baseline_candidate else None,
            "compiler": baseline_compiled,
        }
        if baseline_branch_result.get("main_state_restored") is not True:
            report.update({"status": "baseline_snapshot_restore_failed", "failure_reason": "A0 branch did not restore the S70 main state"})
            return report
        if not _complete_five_steps(baseline_candidate):
            report.update({"status": "baseline_chunk_incomplete", "failure_reason": "A0 must execute all five actions from S70"})
            return report
        _monitor, baseline_monitor_error = _complete_monitoring(baseline_candidate)
        report["baseline_a0"]["contact_monitoring_complete"] = baseline_monitor_error is None
        if baseline_monitor_error:
            report.update({"status": "baseline_contact_monitoring_incomplete", "failure_reason": baseline_monitor_error})
            return report
        if not isinstance(baseline_compiled, dict) or baseline_compiled.get("verdict") != "safe":
            report.update({"status": "baseline_a0_not_safe", "failure_reason": "A0 must be explicitly safe under the existing Contact Compiler"})
            return report

        baseline_endpoint = replay_and_measure_endpoint(
            connection, snapshot_id, snapshot_state, actions, baseline_candidate,
            subject_geom_ids, obstacle_geom_ids, distmax_m,
        )
        report["baseline_s75_endpoint"] = baseline_endpoint
        if baseline_endpoint.get("main_state_restored") is not True:
            report.update({"status": "baseline_endpoint_restore_failed", "failure_reason": baseline_endpoint.get("reason")})
            return report
        if baseline_endpoint.get("status") != "complete":
            report.update({"status": "baseline_endpoint_unavailable", "failure_reason": baseline_endpoint.get("reason", "A0 endpoint distance unavailable")})
            return report
        baseline_gap = baseline_endpoint["endpoint_distance"]
        try:
            directions, direction_basis = derive_clearance_directions(
                baseline_gap,
                binding_validation["objects"][subject_object_id]["body_id"],
                binding_validation["objects"][obstacle_object_id]["body_id"],
                controller_context.get("world_up"),
                [
                    baseline_candidate["start_state"]["robot_states"][0]["eef_pos"],
                    *[step["state"]["robot_states"][0]["eef_pos"] for step in baseline_candidate["steps"]],
                ],
            )
        except Exception as exc:
            report.update({"status": "clearance_direction_unavailable", "failure_reason": f"{type(exc).__name__}: {exc}"})
            return report
        report["direction_basis"] = direction_basis
        specs, generation_rejections = generate_clearance_candidates(
            actions, directions, controller_context, PROBE_CONFIG, max_candidates=max_candidates
        )
        report["candidate_generation_rejections"] = generation_rejections
        report["generated_candidate_count"] = len(specs)
        report["generated_candidate_limit_enforced"] = len(specs) <= MAX_CANDIDATES
        if not specs:
            report.update({"status": "no_executable_candidates", "failure_reason": "All geometric correction candidates failed action preflight constraints"})
            return report

        branch_result = _request(connection, {
            "op": "branch_rollout",
            "snapshot_id": snapshot_id,
            "candidates": [
                {"candidate_id": spec["candidate_id"], "actions": spec["actions"].tolist()}
                for spec in specs
            ],
        })
        report["candidate_branch_main_restore"] = branch_result.get("main_state_restored") is True
        if branch_result.get("main_state_restored") is not True:
            report.update({"status": "candidate_branch_restore_failed", "failure_reason": branch_result.get("main_restore_error", "Bridge did not confirm main-state restoration after candidate branches")})
            return report
        compiler_result = compile_forbid_contact(
            teacher,
            subject_object_id,
            semantic_results[subject_object_id],
            branch_result,
            inventory,
            gripper_inventory,
            object_semantic_results=semantic_results,
        )

        candidate_reports = []
        restore_failure = None
        for spec in specs:
            branch_candidate = _branch_candidate(branch_result, spec["candidate_id"])
            compiled_candidate = _compiler_candidate_summary(compiler_result, spec["candidate_id"])
            endpoint_replay = None
            if _complete_five_steps(branch_candidate):
                endpoint_replay = replay_and_measure_endpoint(
                    connection,
                    snapshot_id,
                    snapshot_state,
                    spec["actions"],
                    branch_candidate,
                    subject_geom_ids,
                    obstacle_geom_ids,
                    distmax_m,
                )
                if endpoint_replay.get("main_state_restored") is not True:
                    restore_failure = endpoint_replay.get("reason", "Candidate endpoint replay failed to restore S70")
            report_item = assess_candidate(
                {**spec, "original_actions": actions},
                branch_candidate,
                compiled_candidate,
                baseline_candidate,
                controller_context,
                task_binding,
                PROBE_CONFIG,
                baseline_gap.get("minimum_distance_m"),
                endpoint_replay,
            )
            candidate_reports.append(report_item)
            if restore_failure:
                break
        report["candidate_results"] = candidate_reports
        report["candidate_results_complete"] = len(candidate_reports) == len(specs)
        report["candidate_margin_counts"] = clearance_margin_counts(candidate_reports)
        report["accepted_probe_count"] = sum(item.get("accepted_for_probe") is True for item in candidate_reports)
        report["final_a_plus"] = None
        report["training_label_valid"] = False
        if restore_failure:
            report.update({"status": "candidate_endpoint_restore_failed", "failure_reason": restore_failure})
            return report
        report["status"] = "probe_complete" if report["accepted_probe_count"] else "no_candidate_passed"
        report["interpretation"] = (
            "Accepted candidates are only S70 feasibility probes. The clearance margins are exploratory, do not prove safety beyond the simulated five-step horizon, and are not final A+ training labels."
        )
        return report
    except Exception as exc:
        report.update({"status": "error", "failure_reason": f"{type(exc).__name__}: {exc}"})
        return report
    finally:
        if snapshot_id is not None and isinstance(snapshot_state, dict):
            try:
                final_restore = _restore_and_compare(connection, snapshot_id, snapshot_state)
            except Exception as exc:
                final_restore = {"matched": False, "error": f"{type(exc).__name__}: {exc}"}
            report["final_s70_restore"] = final_restore
            if final_restore.get("matched") is not True:
                report["status"] = "main_state_restore_failed"
                report["failure_reason"] = final_restore.get("error", "Final S70 restore mismatch")


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    print(f"Early S70 Repair feasibility: status={report.get('status')}")
    print(f"  A0 compiler verdict={report.get('baseline_a0', {}).get('compiler', {}).get('verdict')}")
    print(f"  generated candidates={report.get('generated_candidate_count', 0)}; accepted for probe={report.get('accepted_probe_count', 0)}")
    print(f"  margin counts={report.get('candidate_margin_counts', {}).get('counts_by_margin')}")
    print(f"  training_label_valid={report.get('training_label_valid')}; final_a_plus={report.get('final_a_plus')}")
    if report.get("failure_reason"):
        print(f"  failure: {report['failure_reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--teacher-json", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--semantic-results-json", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--subject-object-id", default=OBJECT_ID)
    parser.add_argument("--obstacle-object-id", default=OBSTACLE_ID)
    parser.add_argument("--distmax-m", type=float, default=10.0)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.max_candidates <= MAX_CANDIDATES:
            raise ValueError(f"--max-candidates must be in [1, {MAX_CANDIDATES}]")
        archive = load_s70_archive(args.archive)
        teacher = parse_teacher_output_v2(args.teacher_json.read_text(encoding="utf-8"))
        semantic_results = _read_json(args.semantic_results_json, "semantic bindings")
        if not isinstance(semantic_results, dict):
            raise ValueError("Semantic binding JSON must map object IDs to resolved bindings")
        connection = Client((args.host, args.port), authkey=AUTHKEY)
        try:
            report = run_experiment(
                connection,
                archive,
                teacher,
                semantic_results,
                subject_object_id=args.subject_object_id,
                obstacle_object_id=args.obstacle_object_id,
                distmax_m=args.distmax_m,
                max_candidates=args.max_candidates,
            )
        finally:
            connection.close()
    except Exception as exc:
        report = {
            "status": "input_or_connection_error",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "training_label_valid": False,
            "final_a_plus": None,
        }
    _write_json(args.out, report)
    print_summary(report)
    print(f"  report: {args.out.resolve()}")
    return 0 if report.get("status") in ("probe_complete", "no_candidate_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
