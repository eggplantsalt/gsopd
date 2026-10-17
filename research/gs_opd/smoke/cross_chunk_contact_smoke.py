"""Diagnose the archived LIBERO-Safety S70→S75 action-chunk boundary.

The Smoke replays the archived 70-step prefix and branches the two archived
five-action chunks from one Snapshot. It runs no policy, Teacher, or segmenter;
the two branches are diagnostics of archived actions, not a repair evaluation.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.smoke.contact_safety_compiler_smoke import (
    AUTHKEY,
    BRIDGE_PORT,
    compile_forbid_contact,
)
from research.gs_opd.smoke.replay_archive_contact_smoke import (
    _candidate_coverage,
    _compare_nested,
    _validate_current_bindings,
    compare_observation,
    load_archive_episode,
)
from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


DEFAULT_ARCHIVE = Path("logs/gs_opd/on_policy_safety_l0t0_object_v2_80step.npz")
DEFAULT_TEACHER = Path("logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt")
DEFAULT_BINDINGS = Path("logs/gs_opd/semantic_bindings_l0t0.json")
STEP_70 = 70
STEP_75 = 75
CHUNK_LENGTH = 5
STATE_ATOL = 1e-6
ARCHIVE_ROBOT_STATE_FIELDS = {
    "robot0_eef_pos": "eef_pos",
    "robot0_eef_quat": "eef_quat",
    "robot0_gripper_qpos": "gripper_qpos",
}
PREFIX_STEP_FIELDS = (
    "state",
    "reward",
    "done",
    "contacts",
    "substep_contact_coverage",
)
PREFIX_EVENT_FIELDS = (
    "geom_pair",
    "body_pair",
    "classification",
    "robot_involved",
    "first_control_step",
    "first_physics_substep",
    "first_candidate_physics_substep",
    "first_contact_position_world",
    "contact_position_status",
    "first_contact_normal_geom1_to_geom2_world",
    "contact_normal_status",
    "contact_normal_geom_ids",
    "first_contact_sample_recorded",
)


class CrossChunkArchiveError(ValueError):
    """Raised when the two archived decisions cannot form the requested probe."""


def load_cross_chunk_archive(archive_path: Path) -> dict[str, Any]:
    """Load S70 and S75 with offset-aware archive extraction and exact action checks."""
    try:
        step_70 = load_archive_episode(archive_path, decision_step=STEP_70)
        step_75 = load_archive_episode(archive_path, decision_step=STEP_75)
    except Exception as exc:
        raise CrossChunkArchiveError(str(exc)) from exc

    action_70 = np.asarray(step_70["action_chunk"], dtype=np.float32)
    action_75 = np.asarray(step_75["action_chunk"], dtype=np.float32)
    if action_70.shape != (CHUNK_LENGTH, 7) or action_75.shape != (CHUNK_LENGTH, 7):
        raise CrossChunkArchiveError(
            "S70 and S75 must each contain exactly five finite 7D actions; "
            f"got {action_70.shape} and {action_75.shape}"
        )
    if step_75["decision_step"] != step_70["decision_step"] + CHUNK_LENGTH:
        raise CrossChunkArchiveError("S75 is not the decision immediately after the S70 chunk")
    if step_70["instruction"] != step_75["instruction"]:
        raise CrossChunkArchiveError("Archived S70 and S75 task instructions differ")
    boundary_observation = compare_archive_step74_post_to_step75_decision(
        archive_path, step_75
    )
    return {
        "archive": str(archive_path.resolve()),
        "instruction": step_70["instruction"],
        "step_70": step_70,
        "step_75": step_75,
        "action_70": action_70,
        "action_75": action_75,
        "action_10": np.concatenate((action_70, action_75), axis=0),
        "archive_step74_to_step75_observation": boundary_observation,
        "archive_action_checks": {
            "step_70_prediction_matches_executed_actions": True,
            "step_75_prediction_matches_executed_actions": True,
            "step_70_shape": list(action_70.shape),
            "step_75_shape": list(action_75.shape),
            "step_70_decision_index": step_70["decision_index"],
            "step_75_decision_index": step_75["decision_index"],
            "source": "decision_step_indices + predicted_action_offsets; verified against step_indices/executed_actions",
        },
    }


def _compare_numeric_arrays(expected: Any, actual: Any, atol: float) -> dict[str, Any]:
    try:
        left = np.asarray(expected, dtype=np.float64)
        right = np.asarray(actual, dtype=np.float64)
        same_shape = left.shape == right.shape
        finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        max_abs_error = (
            float(np.max(np.abs(left - right)))
            if same_shape and left.size and right.size
            else (0.0 if same_shape else None)
        )
        matched = (
            same_shape
            and finite
            and bool(np.allclose(left, right, rtol=0.0, atol=atol))
        )
        return {
            "matched": bool(matched),
            "expected_shape": list(left.shape),
            "actual_shape": list(right.shape),
            "max_abs_error": max_abs_error,
            "atol": atol,
        }
    except Exception as exc:
        return {"matched": False, "error": f"{type(exc).__name__}: {exc}", "atol": atol}


def _compare_rgb_arrays(expected: Any, actual: Any) -> dict[str, Any]:
    try:
        left = np.asarray(expected)
        right = np.asarray(actual)
        same_shape = left.shape == right.shape
        same_dtype = left.dtype == right.dtype == np.uint8
        if same_shape and same_dtype and left.ndim == 3 and left.shape[-1] == 3:
            delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
            differing_pixels = int(np.count_nonzero(np.any(delta != 0, axis=-1)))
            max_channel_delta = int(delta.max(initial=0))
            matched = differing_pixels == 0
        else:
            differing_pixels = None
            max_channel_delta = None
            matched = False
        return {
            "matched_pixel_exactly": bool(matched),
            "expected_shape": list(left.shape),
            "actual_shape": list(right.shape),
            "same_dtype": bool(same_dtype),
            "differing_pixels": differing_pixels,
            "max_channel_delta": max_channel_delta,
        }
    except Exception as exc:
        return {"matched_pixel_exactly": False, "error": f"{type(exc).__name__}: {exc}"}


def compare_archive_step74_post_to_step75_decision(
    archive_path: Path, step_75: dict[str, Any]
) -> dict[str, Any]:
    """Check the stored post-observation after action 74 against decision S75."""
    state_keys = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    camera_keys = ("agentview_image", "robot0_eye_in_hand_image")
    fields: dict[str, Any] = {"state": {}, "images": {}}
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            step_indices = np.asarray(archive["step_indices"], dtype=np.int64)
            rows = np.flatnonzero(step_indices == STEP_75 - 1)
            if len(rows) != 1:
                raise ValueError(f"Expected one archived post step {STEP_75 - 1}, found {len(rows)}")
            post_row = int(rows[0])
            decision_index = int(step_75["decision_index"])
            for key in state_keys:
                fields["state"][key] = _compare_numeric_arrays(
                    archive[f"post_{key}"][post_row],
                    archive[f"decision_{key}"][decision_index],
                    STATE_ATOL,
                )
            for key in camera_keys:
                fields["images"][key] = _compare_rgb_arrays(
                    archive[f"post_{key}"][post_row],
                    archive[f"decision_{key}"][decision_index],
                )
    except Exception as exc:
        return {
            "matched": False,
            "available": False,
            "post_step_index": STEP_75 - 1,
            "decision_step_index": STEP_75,
            "error": f"{type(exc).__name__}: {exc}",
            "fields": fields,
        }

    decision_instruction_matches_metadata = (
        step_75["instruction"] == step_75["metadata"].get("instruction")
    )
    field_matches = [
        item.get("matched") is True for item in fields["state"].values()
    ] + [
        item.get("matched_pixel_exactly") is True for item in fields["images"].values()
    ]
    return {
        "matched": bool(all(field_matches) and decision_instruction_matches_metadata),
        "available": True,
        "post_step_index": STEP_75 - 1,
        "decision_step_index": STEP_75,
        "state_atol": STATE_ATOL,
        "instruction_matches_archive_metadata": bool(decision_instruction_matches_metadata),
        "post_observation_instruction": "not stored per transition in the current NPZ schema",
        "comparison_source": "TrajectoryWriter post-step arrays vs next decision arrays",
        "fields": fields,
    }


def compare_snapshot_diagnostics(
    expected: Any, actual: Any, atol: float = STATE_ATOL
) -> dict[str, Any]:
    """Compare all state fields numerically; report kinematics_source as metadata."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return {
            "matched": False,
            "physical_values": {"matched": False, "error": "diagnostics must be objects"},
            "diagnostic_metadata": {"field": "kinematics_source", "matched": None},
        }
    expected_metadata = expected.get("kinematics_source")
    actual_metadata = actual.get("kinematics_source")
    metadata_is_source_string = (
        isinstance(expected_metadata, str) and isinstance(actual_metadata, str)
    )
    expected_values = dict(expected)
    actual_values = dict(actual)
    if metadata_is_source_string:
        expected_values.pop("kinematics_source", None)
        actual_values.pop("kinematics_source", None)
    physical = _compare_nested(expected_values, actual_values, atol)
    if _has_non_finite_number(expected_values) or _has_non_finite_number(actual_values):
        physical["matched"] = False
        physical["mismatch_count"] = int(physical.get("mismatch_count", 0)) + 1
        physical["mismatch_paths_sample"] = list(physical.get("mismatch_paths_sample", [])) + [
            "$snapshot_state:non_finite_number"
        ]
    return {
        "matched": physical["matched"],
        "physical_values": physical,
        "diagnostic_metadata": {
            "field": "kinematics_source",
            "expected": expected_metadata,
            "actual": actual_metadata,
            "matched": expected_metadata == actual_metadata,
            "informational_only": True,
            "both_are_source_strings": metadata_is_source_string,
        },
    }


def _has_non_finite_number(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_non_finite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_non_finite_number(item) for item in value)
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        return not bool(np.isfinite(value))
    return False


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read {label} JSON {path}: {exc}") from exc


def _choose_compiler_object_id(
    teacher: dict[str, Any], requested: str | None
) -> str:
    rules = teacher.get("object_contact_rules")
    if not isinstance(rules, list):
        raise ValueError("Safety IR v2 has no object_contact_rules array")
    subjects = sorted(
        {
            rule.get("subject_id")
            for rule in rules
            if isinstance(rule, dict)
            and rule.get("decision") == "forbid"
            and rule.get("predicate") == "contact"
            and isinstance(rule.get("subject_id"), str)
        }
    )
    if requested is not None:
        if requested not in subjects:
            raise ValueError(
                f"--object-id {requested!r} is not a subject of an object-contact forbid rule"
            )
        return requested
    if len(subjects) != 1:
        raise ValueError(
            "Expected exactly one object-contact forbid subject; pass --object-id "
            "when the Teacher contains multiple subjects"
        )
    return subjects[0]


def _bridge_request(connection: Any, request: dict[str, Any]) -> dict[str, Any]:
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError(
            f"Bridge returned a non-object response for op={request.get('op')!r}"
        )
    return response


def _event_prefix_signature(monitor: Any, prefix_length: int) -> list[dict[str, Any]] | None:
    if not isinstance(monitor, dict) or not isinstance(monitor.get("events"), list):
        return None
    events = []
    for event in monitor["events"]:
        if not isinstance(event, dict):
            return None
        first_step = event.get("first_control_step")
        if not isinstance(first_step, int) or isinstance(first_step, bool):
            return None
        if first_step < prefix_length:
            events.append({key: event.get(key) for key in PREFIX_EVENT_FIELDS})
    return sorted(
        events,
        key=lambda item: (
            item.get("first_control_step", -1),
            tuple((part.get("id", -1) if isinstance(part, dict) else -1) for part in (item.get("geom_pair") or [])),
        ),
    )


def compare_shared_prefix(
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    prefix_actions: np.ndarray,
    atol: float = STATE_ATOL,
) -> dict[str, Any]:
    """Compare first-five actions, physical states, contacts, and substep records."""
    result: dict[str, Any] = {
        "prefix_length": int(len(prefix_actions)),
        "atol": atol,
        "actions_exact": False,
        "physical_and_contact_match": False,
        "fifth_step_end_state_match": False,
        "matched": False,
    }
    steps_a, steps_b = candidate_a.get("steps"), candidate_b.get("steps")
    if (
        not isinstance(steps_a, list)
        or not isinstance(steps_b, list)
        or len(steps_a) < len(prefix_actions)
        or len(steps_b) < len(prefix_actions)
    ):
        result["reason"] = "one_or_both_candidates_omit_the_shared_prefix"
        return result

    action_checks = []
    selected_a = []
    selected_b = []
    for index, expected_action in enumerate(prefix_actions):
        step_a, step_b = steps_a[index], steps_b[index]
        try:
            action_a = np.asarray(step_a["action"], dtype=np.float32)
            action_b = np.asarray(step_b["action"], dtype=np.float32)
            exact = (
                action_a.shape == (7,)
                and action_b.shape == (7,)
                and np.array_equal(action_a, expected_action)
                and np.array_equal(action_b, expected_action)
            )
        except Exception:
            exact = False
        action_checks.append(bool(exact))
        selected_a.append({key: step_a.get(key) for key in PREFIX_STEP_FIELDS})
        selected_b.append({key: step_b.get(key) for key in PREFIX_STEP_FIELDS})

    monitor_a, monitor_b = candidate_a.get("contact_monitoring"), candidate_b.get("contact_monitoring")
    control_a = monitor_a.get("control_steps") if isinstance(monitor_a, dict) else None
    control_b = monitor_b.get("control_steps") if isinstance(monitor_b, dict) else None
    if not isinstance(control_a, list) or not isinstance(control_b, list):
        result["reason"] = "contact_monitoring_control_steps_are_missing"
        return result
    if len(control_a) < len(prefix_actions) or len(control_b) < len(prefix_actions):
        result["reason"] = "contact_monitoring_does_not_cover_the_shared_prefix"
        return result

    prefix_a = {
        "steps": selected_a,
        "control_steps": control_a[: len(prefix_actions)],
        "first_prefix_events": _event_prefix_signature(monitor_a, len(prefix_actions)),
    }
    prefix_b = {
        "steps": selected_b,
        "control_steps": control_b[: len(prefix_actions)],
        "first_prefix_events": _event_prefix_signature(monitor_b, len(prefix_actions)),
    }
    if prefix_a["first_prefix_events"] is None or prefix_b["first_prefix_events"] is None:
        result["reason"] = "contact_monitoring_event_summary_is_malformed"
        return result
    physical_comparison = _compare_nested(prefix_a, prefix_b, atol)
    fifth_state = _compare_nested(
        steps_a[len(prefix_actions) - 1].get("state"),
        steps_b[len(prefix_actions) - 1].get("state"),
        atol,
    )
    result.update(
        {
            "actions_exact": all(action_checks),
            "per_step_action_exact": action_checks,
            "physical_and_contact_match": physical_comparison["matched"],
            "physical_and_contact_comparison": physical_comparison,
            "fifth_step_end_state_match": fifth_state["matched"],
            "fifth_step_end_state_comparison": fifth_state,
            "contact_comparison_scope": (
                "per-control-step contacts and substep coverage, control-step monitoring, "
                "and first-event identity/position/normal for contacts first seen in the prefix"
            ),
        }
    )
    result["matched"] = bool(
        result["actions_exact"]
        and result["physical_and_contact_match"]
        and result["fifth_step_end_state_match"]
    )
    return result


def compare_endpoint_to_step75(candidate: dict[str, Any], step_75: dict[str, Any]) -> dict[str, Any]:
    """Compare branch endpoint fields that the Bridge and NPZ both expose."""
    steps = candidate.get("steps")
    if not isinstance(steps, list) or len(steps) < CHUNK_LENGTH:
        return {
            "matched_available_fields": False,
            "complete": False,
            "error": "candidate_has_no_fifth_step_endpoint",
        }
    robot_states = steps[CHUNK_LENGTH - 1].get("state", {}).get("robot_states")
    if not isinstance(robot_states, list) or not robot_states or not isinstance(robot_states[0], dict):
        return {
            "matched_available_fields": False,
            "complete": False,
            "error": "candidate_fifth_step_has_no_robot_state",
        }

    expected = step_75["expected_observation"]["state"]
    observed = robot_states[0]
    comparisons: dict[str, Any] = {}
    unavailable = []
    for archive_key, bridge_key in ARCHIVE_ROBOT_STATE_FIELDS.items():
        target = np.asarray(expected[archive_key], dtype=np.float64)
        value = observed.get(bridge_key)
        if value is None:
            unavailable.append(archive_key)
            comparisons[archive_key] = {
                "matched": None,
                "reason": f"branch_rollout_state_does_not_expose_{bridge_key}",
                "expected_shape": list(target.shape),
            }
            continue
        try:
            actual = np.asarray(value, dtype=np.float64)
            same_shape = actual.shape == target.shape
            max_abs_error = (
                float(np.max(np.abs(actual - target)))
                if same_shape and target.size
                else (0.0 if same_shape else None)
            )
            matched = same_shape and np.allclose(actual, target, rtol=0.0, atol=STATE_ATOL)
        except Exception:
            actual, same_shape, max_abs_error, matched = None, False, None, False
        comparisons[archive_key] = {
            "matched": bool(matched),
            "expected_shape": list(target.shape),
            "actual_shape": list(actual.shape) if isinstance(actual, np.ndarray) else None,
            "max_abs_error": max_abs_error,
            "atol": STATE_ATOL,
        }

    matched = all(item.get("matched") is True for item in comparisons.values() if item.get("matched") is not None)
    return {
        "matched_available_fields": bool(matched),
        "complete": not unavailable,
        "unavailable_fields": unavailable,
        "state": comparisons,
        "images": {
            "matched": None,
            "reason": "branch_rollout_step_records_do_not_include_RGB_frames",
        },
        "task_instruction": {
            "matched": step_75["instruction"] == step_75["metadata"]["instruction"],
            "scope": "archive S75 instruction matched archive metadata; Bridge instruction was checked at S70",
            "reason": "branch step records do not carry per-step task instructions",
        },
    }


def _find_candidate(items: Any, candidate_id: str) -> dict[str, Any] | None:
    if not isinstance(items, list):
        return None
    matches = [
        item for item in items
        if isinstance(item, dict) and item.get("candidate_id") == candidate_id
    ]
    return matches[0] if len(matches) == 1 else None


def _first_object_violation(candidate_result: dict[str, Any]) -> dict[str, Any] | None:
    violations = []
    for check in candidate_result.get("object_contact_checks", []):
        if not isinstance(check, dict) or check.get("verdict") != "violation":
            continue
        for evidence in check.get("evidence", []):
            if isinstance(evidence, dict):
                violations.append(evidence)
    if not violations:
        return None
    return min(
        violations,
        key=lambda item: (
            item.get("first_control_step", float("inf")),
            item.get("first_physics_substep", float("inf")),
        ),
    )


def _candidate_report(
    candidate_id: str,
    requested_steps: int,
    branch_candidate: dict[str, Any] | None,
    compiler_candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    if branch_candidate is None:
        return {
            "candidate_id": candidate_id,
            "status": "missing_branch_candidate",
            "full_chunk_executed": False,
            "contact_monitoring_complete": False,
            "safety_verdict": "unknown",
        }
    coverage = _candidate_coverage(branch_candidate, requested_steps)
    compiler_candidate = compiler_candidate or {}
    first_violation = _first_object_violation(compiler_candidate)
    first_index = (
        first_violation.get("first_control_step")
        if isinstance(first_violation, dict)
        else None
    )
    return {
        "candidate_id": candidate_id,
        "branch_status": branch_candidate.get("status"),
        "branch_error": branch_candidate.get("error"),
        "full_chunk_executed": coverage["full_chunk_executed"],
        "contact_monitoring_complete": coverage["contact_monitoring_complete"],
        "coverage": coverage,
        "safety_verdict": compiler_candidate.get("verdict", "unknown"),
        "safety_reason": compiler_candidate.get("reason"),
        "object_contact_checks": compiler_candidate.get("object_contact_checks", []),
        "first_object_to_object_violation": first_violation,
        "first_violation_control_step_index": first_index,
        "first_violation_archive_step": (
            STEP_70 + first_index if isinstance(first_index, int) and not isinstance(first_index, bool) else None
        ),
        "first_violation_is_in_indices_5_to_9": (
            bool(5 <= first_index <= 9)
            if isinstance(first_index, int) and not isinstance(first_index, bool)
            else None
        ),
    }


def _stop(report: dict[str, Any], status: str, reason: str) -> dict[str, Any]:
    report["status"] = status
    report["failure_reason"] = reason
    return report


def restore_step_chunk_and_restore(
    connection: Any,
    snapshot_id: str,
    snapshot_state: Any,
    actions: np.ndarray,
    expected_step75: dict[str, Any],
    branch_candidate_a: dict[str, Any] | None,
) -> dict[str, Any]:
    """Replay A0_70 through normal step observations, always restoring S70 afterward."""
    result: dict[str, Any] = {
        "status": "incomplete",
        "strict_same_source_comparison": True,
        "image_mismatch_flag_applies": False,
        "requested_steps": int(len(actions)),
        "executed_steps": 0,
        "step_records": [],
        "final_restore_attempted": False,
    }
    last_response = None
    try:
        pre_restore = _bridge_request(
            connection, {"op": "restore", "snapshot_id": snapshot_id}
        )
        if pre_restore.get("error"):
            result["pre_step_restore"] = {"matched": False, "error": pre_restore["error"]}
        elif pre_restore.get("restored_snapshot_id") != snapshot_id:
            result["pre_step_restore"] = {
                "matched": False,
                "error": "Bridge restored a different Snapshot ID before sequential step replay",
            }
        elif not isinstance(snapshot_state, dict) or not isinstance(pre_restore.get("snapshot_state"), dict):
            result["pre_step_restore"] = {
                "matched": False,
                "error": "S70 Snapshot or pre-step Restore omitted diagnostics",
            }
        else:
            result["pre_step_restore"] = compare_snapshot_diagnostics(
                snapshot_state, pre_restore["snapshot_state"], STATE_ATOL
            )

        if result.get("pre_step_restore", {}).get("matched") is True:
            for index, action in enumerate(actions):
                last_response = _bridge_request(
                    connection,
                    {
                        "op": "step",
                        "action": action.tolist(),
                        "include_snapshot_state": True,
                    },
                )
                if last_response.get("error"):
                    result["step_records"].append(
                        {"step_index": index, "error": last_response["error"]}
                    )
                    break
                result["executed_steps"] += 1
                record = {
                    "step_index": index,
                    "reward": last_response.get("reward"),
                    "done": last_response.get("done"),
                    "snapshot_state_available": isinstance(last_response.get("snapshot_state"), dict),
                }
                result["step_records"].append(record)
                if last_response.get("done") is True and index + 1 < len(actions):
                    result["early_done_step"] = index
                    break

            if result["executed_steps"] == len(actions) and last_response is not None:
                result["same_source_environment_observation_vs_step75"] = compare_observation(
                    expected_step75["expected_observation"], last_response, STATE_ATOL
                )
                result["same_source_environment_observation_vs_step75"]["comparison_source"] = (
                    "Safety Bridge env.step observation vs NPZ Step75 decision observation"
                )
                result["same_source_environment_observation_vs_step75"]["strict_gate"] = True

                branch_steps = branch_candidate_a.get("steps") if isinstance(branch_candidate_a, dict) else None
                branch_state = (
                    branch_steps[CHUNK_LENGTH - 1].get("state")
                    if isinstance(branch_steps, list) and len(branch_steps) >= CHUNK_LENGTH
                    else None
                )
                step_snapshot_state = last_response.get("snapshot_state")
                result["same_physical_diagnostic_branch_vs_step"] = compare_snapshot_diagnostics(
                    branch_state, step_snapshot_state, STATE_ATOL
                )
                result["same_physical_diagnostic_branch_vs_step"]["comparison_source"] = (
                    "Branch fifth-step diagnostics vs env.step(include_snapshot_state=True) diagnostics"
                )
                result["same_physical_diagnostic_branch_vs_step"]["strict_numeric_gate"] = True
                result["same_physical_diagnostic_branch_vs_step"]["expected_kinematics_source"] = (
                    branch_state.get("kinematics_source") if isinstance(branch_state, dict) else None
                )
                result["same_physical_diagnostic_branch_vs_step"]["actual_kinematics_source"] = (
                    step_snapshot_state.get("kinematics_source")
                    if isinstance(step_snapshot_state, dict)
                    else None
                )
            else:
                result["step_chunk_complete"] = False
                if result.get("early_done_step") is None and last_response is None:
                    result["error"] = "No sequential step response was received"
        if result.get("executed_steps") == len(actions):
            result["step_chunk_complete"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["final_restore_attempted"] = True
        try:
            final_restore = _bridge_request(
                connection, {"op": "restore", "snapshot_id": snapshot_id}
            )
            if final_restore.get("error"):
                final_check = {"matched": False, "error": final_restore["error"]}
            elif final_restore.get("restored_snapshot_id") != snapshot_id:
                final_check = {"matched": False, "error": "Bridge restored a different Snapshot ID after step replay"}
            elif not isinstance(snapshot_state, dict) or not isinstance(final_restore.get("snapshot_state"), dict):
                final_check = {"matched": False, "error": "Final Restore omitted S70 state diagnostics"}
            else:
                final_check = compare_snapshot_diagnostics(
                    snapshot_state, final_restore["snapshot_state"], STATE_ATOL
                )
            result["final_main_state_restoration"] = final_check
        except Exception as exc:
            result["final_main_state_restoration"] = {
                "matched": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    same_observation = result.get("same_source_environment_observation_vs_step75", {})
    same_physics = result.get("same_physical_diagnostic_branch_vs_step", {})
    result["matched"] = bool(
        result.get("pre_step_restore", {}).get("matched") is True
        and result.get("step_chunk_complete") is True
        and same_observation.get("matched") is True
        and same_physics.get("matched") is True
        and result.get("final_main_state_restoration", {}).get("matched") is True
    )
    if result["matched"]:
        result["status"] = "matched"
    else:
        result["status"] = "mismatch_or_unavailable"
    return result


def run_experiment(
    connection: Any,
    archive_pair: dict[str, Any],
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    *,
    compiler_object_id: str | None = None,
    allow_image_mismatch: bool = False,
) -> dict[str, Any]:
    """Run the guarded replay, two-branch contact check, and restore diagnostics."""
    step_70 = archive_pair["step_70"]
    step_75 = archive_pair["step_75"]
    action_70 = archive_pair["action_70"]
    action_75 = archive_pair["action_75"]
    action_10 = archive_pair["action_10"]
    report: dict[str, Any] = {
        "archive": archive_pair["archive"],
        "status": "failed",
        "diagnostic_only": True,
        "not_a_repair_success_proof": True,
        "allow_image_mismatch": bool(allow_image_mismatch),
        "step_70": STEP_70,
        "step_75": STEP_75,
        "instruction": archive_pair["instruction"],
        "archive_step74_to_step75_observation": archive_pair.get(
            "archive_step74_to_step75_observation"
        ),
        "archive_action_checks": archive_pair["archive_action_checks"],
        "action_a_step_70": action_70.tolist(),
        "action_b_step_70_to_79": action_10.tolist(),
        "action_b_step_75_component": action_75.tolist(),
        "replay_requested_steps": len(step_70["replay_actions"]),
        "snapshot_created": False,
        "branch_called": False,
        "current_visual_reidentification_performed": False,
    }
    if not isinstance(report["archive_step74_to_step75_observation"], dict) or report[
        "archive_step74_to_step75_observation"
    ].get("matched") is not True:
        return _stop(
            report,
            "archive_boundary_observation_mismatch",
            "NPZ Step74 post-observation and Step75 decision-observation are missing or differ",
        )
    try:
        object_id = _choose_compiler_object_id(teacher, compiler_object_id)
        report["compiler_object_id"] = object_id

        reset = _bridge_request(connection, {"op": "reset"})
        if reset.get("error"):
            return _stop(report, "reset_failed", str(reset["error"]))
        if reset.get("instruction") != archive_pair["instruction"]:
            return _stop(report, "initial_task_mismatch", "Bridge reset instruction differs from archived task")

        last_response = reset
        for index, action in enumerate(step_70["replay_actions"]):
            last_response = _bridge_request(
                connection,
                {"op": "step", "action": action.tolist()},
            )
            if last_response.get("error"):
                report["replayed_steps"] = index
                return _stop(report, "replay_failed", f"Archived replay failed at prefix action {index}: {last_response['error']}")
            if last_response.get("done") is True:
                report["replayed_steps"] = index + 1
                return _stop(report, "replay_terminated_early", f"Bridge terminated during the 70-step replay at action {index}")
        report["replayed_steps"] = len(step_70["replay_actions"])

        observation = compare_observation(
            step_70["expected_observation"], last_response, STATE_ATOL
        )
        report["step_70_observation_comparison"] = observation
        report["step_70_historical_robot_state_matches"] = bool(
            observation["instruction_matched"] and observation["state_matched"]
        )
        report["step_70_historical_images_pixel_exact"] = observation["images_matched"]
        if not observation["instruction_matched"]:
            return _stop(report, "task_instruction_mismatch", "Current Bridge instruction differs from archived S70")
        if not observation["state_matched"]:
            return _stop(report, "step_70_robot_state_mismatch", "Replayed robot state differs from archived S70 at 1e-6")
        if not observation["images_available"]:
            return _stop(report, "step_70_image_unavailable", "Both current Bridge RGB images must be readable")
        if not observation["images_matched"] and not allow_image_mismatch:
            return _stop(report, "step_70_image_mismatch", "RGB differs from archive; pass --allow-image-mismatch to retain it as a diagnostic")

        geometry_inventory = _bridge_request(connection, {"op": "geometry_inventory"})
        gripper_inventory = _bridge_request(connection, {"op": "gripper_geometry_inventory"})
        binding_validation = _validate_current_bindings(
            teacher, semantic_results, geometry_inventory
        )
        report["geometry_inventory_summary"] = {
            "status": geometry_inventory.get("status"),
            "robot_classification_status": geometry_inventory.get("robot_classification_status"),
            "geom_count": geometry_inventory.get("geom_count"),
        }
        report["semantic_binding_validation"] = binding_validation
        if not binding_validation.get("valid"):
            return _stop(report, "binding_validation_failed", "Teacher object Body/Geom bindings do not match the current Geometry Inventory")

        snapshot = _bridge_request(connection, {"op": "snapshot"})
        if snapshot.get("error") or not isinstance(snapshot.get("snapshot_id"), str):
            return _stop(report, "snapshot_failed", str(snapshot.get("error", "Bridge omitted snapshot_id")))
        snapshot_id = snapshot["snapshot_id"]
        snapshot_state = snapshot.get("snapshot_state")
        report["snapshot_created"] = True
        report["snapshot_id"] = snapshot_id

        roundtrip = _bridge_request(connection, {"op": "restore", "snapshot_id": snapshot_id})
        if roundtrip.get("error"):
            report["snapshot_restore_roundtrip"] = {"matched": False, "error": roundtrip["error"]}
        elif roundtrip.get("restored_snapshot_id") != snapshot_id:
            report["snapshot_restore_roundtrip"] = {"matched": False, "error": "Bridge restored a different Snapshot ID"}
        elif not isinstance(snapshot_state, dict) or not isinstance(roundtrip.get("snapshot_state"), dict):
            report["snapshot_restore_roundtrip"] = {"matched": False, "error": "Snapshot/Restore omitted state diagnostics"}
        else:
            report["snapshot_restore_roundtrip"] = compare_snapshot_diagnostics(
                snapshot_state, roundtrip["snapshot_state"], STATE_ATOL
            )
        if not report["snapshot_restore_roundtrip"].get("matched"):
            return _stop(report, "snapshot_restore_roundtrip_failed", "Immediate Snapshot→Restore did not reproduce S70")

        candidate_a_id = "archived-s70-five-step"
        candidate_b_id = "archived-s70-plus-s75-ten-step"
        branch = _bridge_request(
            connection,
            {
                "op": "branch_rollout",
                "snapshot_id": snapshot_id,
                "candidates": [
                    {"candidate_id": candidate_a_id, "actions": action_70.tolist()},
                    {"candidate_id": candidate_b_id, "actions": action_10.tolist()},
                ],
            },
        )
        report["branch_called"] = True
        report["bridge_main_state_restored"] = branch.get("main_state_restored") is True
        branch_candidates = branch.get("candidates")
        branch_a = _find_candidate(branch_candidates, candidate_a_id)
        branch_b = _find_candidate(branch_candidates, candidate_b_id)
        report["branch_candidate_count"] = len(branch_candidates) if isinstance(branch_candidates, list) else None
        report["branch_error"] = branch.get("error")

        # Branch diagnostics and normal env.step observations are different sources.
        # Preserve the former comparison for analysis, but gate S75 against an actual
        # sequential Bridge step observation below.
        report["step_75_cross_source_branch_vs_archive_diagnostic"] = (
            compare_endpoint_to_step75(branch_a, step_75)
            if branch_a is not None
            else {"matched_available_fields": False, "complete": False, "error": "branch A is missing"}
        )
        report["step_75_cross_source_branch_vs_archive_diagnostic"].update(
            {
                "comparison_source": "branch state diagnostics vs archived environment observation",
                "classification": "cross-source diagnostic; representation differences are not same-source mismatches",
                "integrity_gate": False,
            }
        )

        try:
            post_branch_snapshot = _bridge_request(connection, {"op": "snapshot"})
            if post_branch_snapshot.get("error") or not isinstance(post_branch_snapshot.get("snapshot_state"), dict):
                restore_comparison = {
                    "matched": False,
                    "error": post_branch_snapshot.get("error", "Post-branch Snapshot omitted diagnostics"),
                }
            elif not isinstance(snapshot_state, dict):
                restore_comparison = {"matched": False, "error": "Pre-branch Snapshot omitted diagnostics"}
            else:
                restore_comparison = compare_snapshot_diagnostics(
                    snapshot_state, post_branch_snapshot["snapshot_state"], STATE_ATOL
                )
        except Exception as exc:
            # Do not let a diagnostic Snapshot read prevent the mandatory
            # restore→step→restore consistency probe below.
            restore_comparison = {
                "matched": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["branch_main_state_restoration"] = {
            "bridge_reported_restored": report["bridge_main_state_restored"],
            "snapshot_diagnostics_match": restore_comparison.get("matched") is True,
            "comparison": restore_comparison,
        }

        # Always attempt the sequential A0_70 replay and final S70 restore after a
        # branch response, even if other diagnostics already show a mismatch.
        report["sequential_step_replay"] = restore_step_chunk_and_restore(
            connection,
            snapshot_id,
            snapshot_state,
            action_70,
            step_75,
            branch_a,
        )
        report["main_state_restoration"] = {
            "bridge_reported_restored": report["bridge_main_state_restored"],
            "branch_snapshot_diagnostics_match": restore_comparison.get("matched") is True,
            "sequential_final_restore_attempted": report["sequential_step_replay"].get(
                "final_restore_attempted"
            ) is True,
            "snapshot_diagnostics_match": report["sequential_step_replay"]
            .get("final_main_state_restoration", {})
            .get("matched")
            is True,
            "comparison": report["sequential_step_replay"].get("final_main_state_restoration"),
        }

        if branch_a is None or branch_b is None:
            report["candidates"] = [
                _candidate_report(candidate_a_id, CHUNK_LENGTH, branch_a, None),
                _candidate_report(candidate_b_id, 2 * CHUNK_LENGTH, branch_b, None),
            ]
            return _stop(report, "branch_candidates_missing", "Bridge did not return exactly identifiable A and B candidates")

        coverage_a = _candidate_coverage(branch_a, CHUNK_LENGTH)
        coverage_b = _candidate_coverage(branch_b, 2 * CHUNK_LENGTH)
        start_state_checks = {}
        for candidate_id, candidate in ((candidate_a_id, branch_a), (candidate_b_id, branch_b)):
            candidate_start = candidate.get("start_state")
            start_state_checks[candidate_id] = (
                compare_snapshot_diagnostics(snapshot_state, candidate_start, STATE_ATOL)
                if isinstance(snapshot_state, dict) and isinstance(candidate_start, dict)
                else {"matched": False, "error": "Snapshot or candidate start diagnostics are missing"}
            )
        report["candidate_start_state_checks"] = start_state_checks
        report["candidate_action_echo"] = {
            candidate_a_id: _actions_echo_match(branch_a, action_70),
            candidate_b_id: _actions_echo_match(branch_b, action_10),
        }
        report["shared_first_five_comparison"] = compare_shared_prefix(
            branch_a, branch_b, action_70, STATE_ATOL
        )
        report["fifth_step_end_state_comparison"] = report["shared_first_five_comparison"].get(
            "fifth_step_end_state_comparison"
        )
        report["step_75_decision_observation_comparison"] = report[
            "step_75_cross_source_branch_vs_archive_diagnostic"
        ]

        compiler = compile_forbid_contact(
            teacher,
            object_id,
            semantic_results.get(object_id),
            branch,
            geometry_inventory,
            gripper_inventory,
            object_semantic_results=semantic_results,
        )
        report["compiler_result"] = compiler
        compiler_a = _find_candidate(compiler.get("candidates"), candidate_a_id)
        compiler_b = _find_candidate(compiler.get("candidates"), candidate_b_id)
        report["candidates"] = [
            _candidate_report(candidate_a_id, CHUNK_LENGTH, branch_a, compiler_a),
            _candidate_report(candidate_b_id, 2 * CHUNK_LENGTH, branch_b, compiler_b),
        ]
        report["raw_branch_candidates"] = [branch_a, branch_b]

        integrity_errors = []
        if not coverage_a["full_chunk_executed"] or not coverage_b["full_chunk_executed"]:
            integrity_errors.append("one_or_both_candidates_did_not_execute_the_complete_chunk")
        if report["branch_candidate_count"] != 2:
            integrity_errors.append("branch_rollout_did_not_return_exactly_two_candidates")
        if not all(item.get("matched") is True for item in start_state_checks.values()):
            integrity_errors.append("one_or_both_candidates_did_not_start_from_the_s70_snapshot")
        if not coverage_a["contact_monitoring_complete"] or not coverage_b["contact_monitoring_complete"]:
            integrity_errors.append("one_or_both_candidates_have_incomplete_physics_substep_contact_monitoring")
        if not all(report["candidate_action_echo"].values()):
            integrity_errors.append("branch_action_echo_differs_from_archived_actions")
        if not report["shared_first_five_comparison"].get("matched"):
            integrity_errors.append("first_five_branch_prefix_state_action_or_contact_results_differ")
        if not report["sequential_step_replay"].get("same_source_environment_observation_vs_step75", {}).get("matched"):
            integrity_errors.append("same_source_env_step_observation_differs_from_archived_step75")
        if not report["sequential_step_replay"].get("same_physical_diagnostic_branch_vs_step", {}).get("matched"):
            integrity_errors.append("branch_fifth_step_physical_diagnostics_differ_from_sequential_step")
        if not report["bridge_main_state_restored"] or not restore_comparison.get("matched"):
            integrity_errors.append("main_environment_was_not_confirmed_at_s70_after_branch")
        if not report["sequential_step_replay"].get("final_main_state_restoration", {}).get("matched"):
            integrity_errors.append("main_environment_was_not_confirmed_at_s70_after_sequential_step_replay")
        if compiler.get("main_state_restored") is not True:
            integrity_errors.append("Compiler_did_not_confirm_branch_main_state_restore")
        if integrity_errors:
            report["integrity_errors"] = integrity_errors
            return _stop(report, "diagnostic_integrity_failed", "; ".join(integrity_errors))

        report["status"] = "diagnostic_complete"
        report["interpretation"] = (
            "A and B are archived-action contact diagnostics. Even if B has no observed "
            "violation, that is not evidence that a repaired or safe action was found."
        )
        return report
    except Exception as exc:
        return _stop(report, "bridge_or_compiler_error", f"{type(exc).__name__}: {exc}")


def _actions_echo_match(candidate: dict[str, Any], expected: np.ndarray) -> bool:
    steps = candidate.get("steps")
    if not isinstance(steps, list) or len(steps) != len(expected):
        return False
    try:
        echoed = np.asarray([step["action"] for step in steps], dtype=np.float32)
    except Exception:
        return False
    return echoed.shape == expected.shape and np.array_equal(echoed, expected)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    print(f"Cross-chunk contact diagnostic: status={report.get('status')}")
    print(
        "  action source verified: "
        f"S70={report.get('archive_action_checks', {}).get('step_70_prediction_matches_executed_actions')} "
        f"S75={report.get('archive_action_checks', {}).get('step_75_prediction_matches_executed_actions')}"
    )
    if report.get("step_70_historical_images_pixel_exact") is not None:
        print(
            "  S70 replay: instruction/state="
            f"{report.get('step_70_historical_robot_state_matches')} "
            f"RGB_pixel_exact={report.get('step_70_historical_images_pixel_exact')}"
        )
    archive_boundary = report.get("archive_step74_to_step75_observation")
    if isinstance(archive_boundary, dict):
        print(
            "  archive S74 post → S75 decision: "
            f"matched={archive_boundary.get('matched')} available={archive_boundary.get('available')}"
        )
    shared = report.get("shared_first_five_comparison", {})
    if shared:
        print(
            "  shared first five: "
            f"matched={shared.get('matched')} actions_exact={shared.get('actions_exact')} "
            f"physical/contact={shared.get('physical_and_contact_match')} "
            f"fifth_end_state={shared.get('fifth_step_end_state_match')}"
        )
    for candidate in report.get("candidates", []):
        print(
            f"  {candidate.get('candidate_id')}: "
            f"steps={candidate.get('coverage', {}).get('executed_steps')}/"
            f"{candidate.get('coverage', {}).get('requested_steps')} "
            f"substep_monitoring={candidate.get('contact_monitoring_complete')} "
            f"verdict={candidate.get('safety_verdict')}"
        )
        evidence = candidate.get("first_object_to_object_violation")
        if evidence:
            print(
                "    first object-object violation: "
                f"control_index={evidence.get('first_control_step')} "
                f"physics_substep={evidence.get('first_physics_substep')} "
                f"geom_pair={evidence.get('geom_pair')} body_pair={evidence.get('body_pair')} "
                f"position={evidence.get('first_contact_position_world')} "
                f"normal={evidence.get('first_contact_normal_geom1_to_geom2_world')} "
                f"max_force_norm={evidence.get('max_force_norm')} "
                f"B_tail_index_5_to_9={candidate.get('first_violation_is_in_indices_5_to_9')}"
            )
        elif candidate.get("safety_verdict") == "unknown":
            print(f"    object-object evidence unavailable/unknown: {candidate.get('object_contact_checks')}")
        else:
            print("    no object-object violation evidence returned by the Compiler")
    restoration = report.get("main_state_restoration")
    if restoration:
        print(
            "  main state restored: "
            f"bridge={restoration.get('bridge_reported_restored')} "
            f"diagnostics={restoration.get('snapshot_diagnostics_match')}"
        )
    sequential = report.get("sequential_step_replay")
    if isinstance(sequential, dict):
        observation_check = sequential.get("same_source_environment_observation_vs_step75", {})
        physical_check = sequential.get("same_physical_diagnostic_branch_vs_step", {})
        print(
            "  sequential A0_70 env.step vs S75: "
            f"matched={observation_check.get('matched')} "
            f"branch/step physical diagnostics={physical_check.get('matched')} "
            f"final_restore={sequential.get('final_main_state_restoration', {}).get('matched')}"
        )
    if report.get("failure_reason"):
        print(f"  failure: {report['failure_reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--teacher-json", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--semantic-results-json", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--object-id", help="forbid object-contact subject when the Teacher has several")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT)
    parser.add_argument(
        "--allow-image-mismatch",
        action="store_true",
        help="keep S70 RGB differences diagnostic; instruction, robot state, bindings, and physics checks remain strict",
    )
    args = parser.parse_args(argv)

    try:
        archive_pair = load_cross_chunk_archive(args.archive)
        teacher = parse_teacher_output_v2(args.teacher_json.read_text(encoding="utf-8"))
        semantic_results = _read_json(args.semantic_results_json, "semantic results")
        if not isinstance(semantic_results, dict):
            raise ValueError("Semantic results JSON must map Teacher object IDs to binding objects")
        connection = Client((args.host, args.port), authkey=AUTHKEY)
        try:
            report = run_experiment(
                connection,
                archive_pair,
                teacher,
                semantic_results,
                compiler_object_id=args.object_id,
                allow_image_mismatch=args.allow_image_mismatch,
            )
        finally:
            connection.close()
    except Exception as exc:
        report = {
            "status": "input_or_connection_error",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "diagnostic_only": True,
            "not_a_repair_success_proof": True,
        }
    _write_json(args.out, report)
    print_summary(report)
    print(f"  report: {args.out.resolve()}")
    return 0 if report.get("status") == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
