"""Replay an archived episode prefix and verify one archived object-contact branch.

This Smoke does not invoke a policy or change the main rollout. It resets the
fixed Bridge environment, replays the archived executed actions up to one
decision, verifies the resulting observation against the archived decision,
then snapshots and branches the archived predicted Action Chunk exactly once.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from research.gs_opd.smoke.contact_safety_compiler_smoke import (
    AUTHKEY,
    BRIDGE_PORT,
    _resolve_target_binding,
    compile_forbid_contact,
)
from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


class ArchiveValidationError(ValueError):
    """Raised when the requested archived decision cannot be replayed safely."""


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read {label} JSON {path}: {exc}") from exc


def load_archive_episode(archive_path: Path, decision_step: int) -> dict[str, Any]:
    """Extract a contiguous executed prefix and one complete predicted chunk."""
    required = {
        "metadata_json",
        "decision_step_indices",
        "decision_instructions",
        "predicted_actions",
        "predicted_action_offsets",
        "step_indices",
        "executed_actions",
        "step_dones",
        *(f"decision_{key}" for key in CAMERA_KEYS),
        *(f"decision_{key}" for key in STATE_KEYS),
    }
    try:
        archive = np.load(archive_path, allow_pickle=False)
    except Exception as exc:
        raise ArchiveValidationError(f"Cannot load archive {archive_path}: {exc}") from exc

    with archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ArchiveValidationError(f"Archive is missing fields: {missing}")
        try:
            metadata = json.loads(str(archive["metadata_json"]))
        except Exception as exc:
            raise ArchiveValidationError(f"Archive metadata_json is invalid: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ArchiveValidationError("Archive metadata_json must contain an object")

        decision_steps = np.asarray(archive["decision_step_indices"], dtype=np.int64)
        matches = np.flatnonzero(decision_steps == decision_step)
        if len(matches) != 1:
            raise ArchiveValidationError(
                f"Expected one decision at step {decision_step}; found {len(matches)}"
            )
        decision_index = int(matches[0])

        offsets = np.asarray(archive["predicted_action_offsets"], dtype=np.int64)
        predicted = np.asarray(archive["predicted_actions"], dtype=np.float32)
        if offsets.shape != (len(decision_steps) + 1,) or offsets[0] != 0:
            raise ArchiveValidationError("predicted_action_offsets has an invalid shape or origin")
        if np.any(np.diff(offsets) < 0) or offsets[-1] != len(predicted):
            raise ArchiveValidationError("predicted_action_offsets do not cover predicted_actions")
        start, stop = int(offsets[decision_index]), int(offsets[decision_index + 1])
        action_chunk = predicted[start:stop].copy()
        if action_chunk.ndim != 2 or action_chunk.shape[1:] != (7,) or len(action_chunk) == 0:
            raise ArchiveValidationError("The selected predicted Action Chunk is not a non-empty (T, 7) array")
        if not np.isfinite(action_chunk).all():
            raise ArchiveValidationError("The selected predicted Action Chunk contains non-finite values")

        step_indices = np.asarray(archive["step_indices"], dtype=np.int64)
        executed_actions = np.asarray(archive["executed_actions"], dtype=np.float32)
        step_dones = np.asarray(archive["step_dones"], dtype=np.bool_)
        if step_indices.ndim != 1 or executed_actions.shape != (len(step_indices), 7):
            raise ArchiveValidationError("step_indices and executed_actions have inconsistent shapes")
        if step_dones.shape != step_indices.shape:
            raise ArchiveValidationError("step_dones and step_indices have inconsistent shapes")
        if not np.isfinite(executed_actions).all():
            raise ArchiveValidationError("executed_actions contains non-finite values")

        index_by_step: dict[int, int] = {}
        for row, step in enumerate(step_indices.tolist()):
            if step in index_by_step:
                raise ArchiveValidationError(f"Archive contains duplicate executed step index {step}")
            index_by_step[int(step)] = row
        required_prefix_steps = list(range(decision_step))
        absent = [step for step in required_prefix_steps if step not in index_by_step]
        if absent:
            raise ArchiveValidationError(f"Archive does not contain a contiguous prefix; missing steps {absent[:8]}")
        prefix_rows = [index_by_step[step] for step in required_prefix_steps]
        if any(step_dones[row] for row in prefix_rows):
            raise ArchiveValidationError("The archived episode terminates before the requested decision")
        chunk_step_numbers = list(range(decision_step, decision_step + len(action_chunk)))
        missing_chunk_steps = [step for step in chunk_step_numbers if step not in index_by_step]
        if missing_chunk_steps:
            raise ArchiveValidationError(
                "The selected predicted chunk was not fully executed in the archive; "
                f"missing steps {missing_chunk_steps[:8]}"
            )
        archived_chunk_execution = executed_actions[
            [index_by_step[step] for step in chunk_step_numbers]
        ]
        if not np.array_equal(archived_chunk_execution, action_chunk):
            raise ArchiveValidationError(
                "Predicted Action Chunk differs from its archived executed actions"
            )

        instruction = str(archive["decision_instructions"][decision_index])
        metadata_instruction = metadata.get("instruction")
        if not instruction or instruction != metadata_instruction:
            raise ArchiveValidationError("Decision instruction disagrees with archive metadata")

        observation: dict[str, Any] = {"instruction": instruction, "state": {}, "images": {}}
        for key in STATE_KEYS:
            value = np.asarray(archive[f"decision_{key}"][decision_index], dtype=np.float64).copy()
            if not np.isfinite(value).all():
                raise ArchiveValidationError(f"Archived decision state {key} contains non-finite values")
            observation["state"][key] = value
        for key in CAMERA_KEYS:
            value = np.asarray(archive[f"decision_{key}"][decision_index]).copy()
            if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
                raise ArchiveValidationError(f"Archived decision image {key} is not uint8 RGB")
            observation["images"][key] = value

        return {
            "archive": str(archive_path.resolve()),
            "metadata": metadata,
            "decision_index": decision_index,
            "decision_step": int(decision_step),
            "instruction": instruction,
            "replay_actions": executed_actions[prefix_rows].copy(),
            "action_chunk": action_chunk,
            "expected_observation": observation,
        }


def _decode_bridge_image(response: dict[str, Any], key: str) -> np.ndarray:
    image = response.get("images", {}).get(key)
    if not isinstance(image, dict):
        raise ValueError(f"Bridge response is missing image {key}")
    try:
        dtype = np.dtype(image["dtype"])
        shape = tuple(int(item) for item in image["shape"])
        value = np.frombuffer(image["data"], dtype=dtype).reshape(shape).copy()
    except Exception as exc:
        raise ValueError(f"Bridge image {key} has an invalid encoding: {exc}") from exc
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError(f"Bridge image {key} is not uint8 RGB")
    return value


def compare_observation(
    expected: dict[str, Any],
    actual: dict[str, Any],
    state_atol: float,
) -> dict[str, Any]:
    """Compare archived robot state and both raw RGB camera observations."""
    state_results: dict[str, Any] = {}
    state_match = True
    actual_state = actual.get("state")
    if not isinstance(actual_state, dict):
        actual_state = {}
    for key in STATE_KEYS:
        observed = None
        target = np.asarray(expected["state"][key], dtype=np.float64)
        try:
            observed = np.asarray(actual_state[key], dtype=np.float64)
            same_shape = observed.shape == target.shape
            max_abs_error = (
                float(np.max(np.abs(observed - target)))
                if same_shape and target.size
                else (0.0 if same_shape else None)
            )
            matched = same_shape and np.allclose(observed, target, rtol=0.0, atol=state_atol)
        except Exception:
            same_shape, max_abs_error, matched = False, None, False
        state_results[key] = {
            "matched": bool(matched),
            "expected_shape": list(target.shape),
            "actual_shape": list(observed.shape) if isinstance(observed, np.ndarray) else None,
            "max_abs_error": max_abs_error,
            "atol": state_atol,
        }
        state_match = state_match and bool(matched)

    image_results: dict[str, Any] = {}
    images_match = True
    images_available = True
    for key in CAMERA_KEYS:
        observed = None
        target = expected["images"][key]
        try:
            observed = _decode_bridge_image(actual, key)
            same_shape = observed.shape == target.shape
            same_dtype = observed.dtype == target.dtype
            if same_shape and same_dtype:
                delta = np.abs(observed.astype(np.int16) - target.astype(np.int16))
                differing_pixels = int(np.count_nonzero(np.any(delta != 0, axis=-1)))
                max_abs_error = int(delta.max(initial=0))
                matched = differing_pixels == 0
            else:
                differing_pixels = None
                max_abs_error = None
                matched = False
        except Exception as exc:
            same_shape = same_dtype = False
            images_available = False
            differing_pixels = None
            max_abs_error = None
            matched = False
            decode_error = f"{type(exc).__name__}: {exc}"
        else:
            decode_error = None
        image_results[key] = {
            "matched_pixel_exactly": bool(matched),
            "expected_shape": list(target.shape),
            "actual_shape": list(observed.shape) if isinstance(observed, np.ndarray) else None,
            "same_dtype": bool(same_dtype),
            "differing_pixels": differing_pixels,
            "max_abs_error": max_abs_error,
            "error": decode_error,
        }
        images_match = images_match and bool(matched)

    instruction_match = actual.get("instruction") == expected.get("instruction")
    return {
        "matched": state_match and images_match and instruction_match,
        "state_matched": state_match,
        "images_matched": images_match,
        "images_available": images_available,
        "instruction_matched": instruction_match,
        "state": state_results,
        "images": image_results,
        "expected_instruction": expected.get("instruction"),
        "actual_instruction": actual.get("instruction"),
    }


def _compare_nested(expected: Any, actual: Any, atol: float) -> dict[str, Any]:
    mismatches: list[str] = []
    max_abs_delta = 0.0

    def visit(left: Any, right: Any, path: str) -> None:
        nonlocal max_abs_delta
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                mismatches.append(path + ":keys")
                return
            for key in left:
                visit(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatches.append(path + ":length")
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
            delta = abs(float(left) - float(right))
            max_abs_delta = max(max_abs_delta, delta)
            if not math.isfinite(delta) or delta > atol:
                mismatches.append(path)
            return
        if type(left) is not type(right) or left != right:
            mismatches.append(path)

    visit(expected, actual, "$snapshot_state")
    return {
        "matched": not mismatches,
        "max_abs_delta": max_abs_delta,
        "atol": atol,
        "mismatch_count": len(mismatches),
        "mismatch_paths_sample": mismatches[:20],
    }


def _bridge_request(connection: Any, request: dict[str, Any]) -> dict[str, Any]:
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError(f"Bridge returned a non-object response for op={request.get('op')!r}")
    return response


def _geometry_summary(inventory: Any) -> dict[str, Any]:
    if not isinstance(inventory, dict):
        return {"status": "unavailable", "geom_count": None}
    return {
        "status": inventory.get("status"),
        "robot_classification_status": inventory.get("robot_classification_status"),
        "geom_count": inventory.get("geom_count"),
        "error": inventory.get("error"),
    }


def _validate_current_bindings(
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    geometry_inventory: Any,
) -> dict[str, Any]:
    """Check archived semantic bindings against the active model inventory."""
    teacher_objects = {
        item["id"]: item for item in teacher.get("objects", []) if isinstance(item, dict)
    }
    rules = teacher.get("object_contact_rules", [])
    required_ids = sorted(
        {
            object_id
            for rule in rules
            if isinstance(rule, dict)
            for object_id in (rule.get("subject_id"), rule.get("target_id"))
            if isinstance(object_id, str)
        }
    )
    result: dict[str, Any] = {
        "valid": False,
        "binding_source": "pre-existing semantic results; not re-run on current RGB",
        "current_visual_reidentification_performed": False,
        "geometry_inventory": _geometry_summary(geometry_inventory),
        "objects": {},
        "errors": [],
    }
    if not required_ids:
        result["errors"].append("teacher_has_no_object_contact_binding_targets")
        return result
    body_owners: dict[int, str] = {}
    for object_id in required_ids:
        teacher_object = teacher_objects.get(object_id)
        semantic = semantic_results.get(object_id)
        if teacher_object is None or not isinstance(semantic, dict):
            result["objects"][object_id] = {"status": "missing"}
            result["errors"].append(f"semantic_binding_missing:{object_id}")
            continue
        semantic_teacher_object = semantic.get("teacher_object")
        if (
            not isinstance(semantic_teacher_object, dict)
            or semantic_teacher_object.get("name") != teacher_object.get("name")
            or semantic_teacher_object.get("role") != teacher_object.get("role")
        ):
            result["objects"][object_id] = {"status": "teacher_semantics_mismatch"}
            result["errors"].append(f"semantic_binding_teacher_object_mismatch:{object_id}")
            continue
        binding, _geom_by_id, _body_by_id, error = _resolve_target_binding(
            semantic,
            object_id,
            geometry_inventory,
        )
        if error is not None or binding is None:
            result["objects"][object_id] = {
                "status": "geometry_binding_mismatch",
                "error": error or "binding_not_resolved",
            }
            result["errors"].append(f"current_geometry_binding_mismatch:{object_id}:{error}")
            continue
        prior_owner = body_owners.get(binding["body_id"])
        if prior_owner is not None and prior_owner != object_id:
            result["objects"][object_id] = {
                "status": "conflicting_body_binding",
                "body_id": binding["body_id"],
                "conflicts_with": prior_owner,
            }
            result["errors"].append(
                f"distinct_teacher_objects_share_current_body:{prior_owner}:{object_id}"
            )
            continue
        body_owners[binding["body_id"]] = object_id
        result["objects"][object_id] = {
            "status": "matches_current_geometry_inventory",
            "body_id": binding["body_id"],
            "body_name": binding["body_name"],
            "geom_ids": binding["geom_ids"],
            "geom_names": binding["geom_names"],
        }
    result["valid"] = not result["errors"] and len(result["objects"]) == len(required_ids)
    return result


def _save_visual_diagnostics(
    expected: dict[str, Any],
    actual: dict[str, Any],
    output_path: Path | None,
) -> dict[str, Any]:
    """Save current camera frames and per-pixel absolute-difference images."""
    if output_path is None:
        return {"saved": False, "reason": "output_path_not_provided", "cameras": {}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cameras: dict[str, Any] = {}
    for key in CAMERA_KEYS:
        try:
            current = _decode_bridge_image(actual, key)
        except Exception as exc:
            cameras[key] = {"saved": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        prefix = output_path.with_name(f"{output_path.stem}_{key}")
        current_path = prefix.with_name(prefix.name + "_current.png")
        Image.fromarray(current).save(current_path)
        expected_image = expected["images"][key]
        diff_path = None
        if expected_image.shape == current.shape:
            delta = np.abs(current.astype(np.int16) - expected_image.astype(np.int16)).astype(np.uint8)
            diff_path = prefix.with_name(prefix.name + "_absolute_difference.png")
            Image.fromarray(delta).save(diff_path)
        cameras[key] = {
            "current_image": str(current_path.resolve()),
            "absolute_difference_image": str(diff_path.resolve()) if diff_path else None,
            "current_shape": list(current.shape),
        }
    return {"saved": any(item.get("saved", True) for item in cameras.values()), "cameras": cameras}


def _candidate_coverage(candidate: dict[str, Any], expected_steps: int) -> dict[str, Any]:
    steps = candidate.get("steps")
    monitor = candidate.get("contact_monitoring")
    executed = candidate.get("executed_steps")
    requested = candidate.get("requested_steps")
    full = (
        candidate.get("status") == "ok"
        and requested == expected_steps
        and executed == expected_steps
        and isinstance(steps, list)
        and len(steps) == expected_steps
        and all(isinstance(step, dict) and step.get("step_index") == index for index, step in enumerate(steps))
    )
    control_steps = monitor.get("control_steps") if isinstance(monitor, dict) else None
    coverage = (
        isinstance(monitor, dict)
        and monitor.get("status") == "complete"
        and isinstance(control_steps, list)
        and len(control_steps) == expected_steps
        and monitor.get("unobserved_expected_substeps") == 0
        and monitor.get("unavailable_contact_substeps") == 0
        and monitor.get("expected_physics_substeps_total") == monitor.get("captured_physics_substeps")
        and all(
            isinstance(item, dict)
            and item.get("status") == "complete"
            and item.get("expected_physics_substeps") == item.get("captured_physics_substeps")
            and item.get("unobserved_expected_substeps") == 0
            and isinstance(item.get("contact_samples"), dict)
            and item["contact_samples"].get("unavailable") == 0
            for item in control_steps
        )
    )
    return {
        "full_chunk_executed": bool(full),
        "contact_monitoring_complete": bool(coverage),
        "requested_steps": requested,
        "executed_steps": executed,
        "step_record_count": len(steps) if isinstance(steps, list) else None,
        "contact_monitoring_status": monitor.get("status") if isinstance(monitor, dict) else None,
        "candidate_status": candidate.get("status"),
        "candidate_error": candidate.get("error"),
        "step_rewards": [step.get("reward") for step in steps] if isinstance(steps, list) else None,
        "step_dones": [step.get("done") for step in steps] if isinstance(steps, list) else None,
        "control_steps": control_steps,
        "expected_physics_substeps_total": monitor.get("expected_physics_substeps_total") if isinstance(monitor, dict) else None,
        "captured_physics_substeps": monitor.get("captured_physics_substeps") if isinstance(monitor, dict) else None,
        "unobserved_expected_substeps": monitor.get("unobserved_expected_substeps") if isinstance(monitor, dict) else None,
        "unavailable_contact_substeps": monitor.get("unavailable_contact_substeps") if isinstance(monitor, dict) else None,
    }


def _branch_coverage(branch_result: Any, expected_steps: int) -> dict[str, Any]:
    candidates = branch_result.get("candidates") if isinstance(branch_result, dict) else None
    if not isinstance(candidates, list) or len(candidates) != 2 or any(
        not isinstance(candidate, dict) for candidate in candidates
    ):
        return {
            "full_chunk_executed": False,
            "contact_monitoring_complete": False,
            "candidate_count": len(candidates) if isinstance(candidates, list) else None,
            "expected_candidate_count": 2,
            "reason": "expected_two_repeated_branch_candidates",
        }
    per_candidate = [
        _candidate_coverage(candidate, expected_steps) for candidate in candidates
    ]
    return {
        "full_chunk_executed": all(item["full_chunk_executed"] for item in per_candidate),
        "contact_monitoring_complete": all(
            item["contact_monitoring_complete"] for item in per_candidate
        ),
        "candidate_count": len(candidates),
        "expected_candidate_count": 2,
        "candidates": per_candidate,
    }


def _branch_repeat_consistency(branch_result: Any, atol: float) -> dict[str, Any]:
    candidates = branch_result.get("candidates") if isinstance(branch_result, dict) else None
    if not isinstance(candidates, list) or len(candidates) != 2 or any(
        not isinstance(candidate, dict) for candidate in candidates
    ):
        return {
            "consistent": False,
            "reason": "expected_two_repeated_branch_candidates",
            "candidate_count": len(candidates) if isinstance(candidates, list) else None,
            "atol": atol,
        }
    left, right = (copy.deepcopy(candidate) for candidate in candidates)
    left.pop("candidate_id", None)
    right.pop("candidate_id", None)
    result = _compare_nested(left, right, atol)
    return {
        "consistent": result["matched"],
        "candidate_ids": [candidate.get("candidate_id") for candidate in candidates],
        "max_abs_delta": result["max_abs_delta"],
        "atol": atol,
        "mismatch_count": result["mismatch_count"],
        "mismatch_paths_sample": result["mismatch_paths_sample"],
        "comparison_scope": "start/step states, actions, rewards, done, contacts, and contact monitoring",
    }


def _object_contact_evidence(compiler_result: Any) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if not isinstance(compiler_result, dict):
        return evidence
    for candidate in compiler_result.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        for check in candidate.get("object_contact_checks", []):
            if not isinstance(check, dict):
                continue
            for item in check.get("evidence", []):
                if isinstance(item, dict):
                    evidence.append(item)
    return evidence


def run_experiment(
    connection: Any,
    archive_episode: dict[str, Any],
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    object_id: str,
    state_atol: float = 1e-6,
    allow_image_mismatch: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the guarded reset/replay/compare/snapshot/branch/compiler sequence."""
    report: dict[str, Any] = {
        "archive": archive_episode["archive"],
        "decision_step": archive_episode["decision_step"],
        "decision_index": archive_episode["decision_index"],
        "replay_requested_steps": len(archive_episode["replay_actions"]),
        "action_chunk_length": len(archive_episode["action_chunk"]),
        "action_chunk": archive_episode["action_chunk"].tolist(),
        "object_id": object_id,
        "status": "failed",
        "snapshot_created": False,
        "branch_called": False,
        "allow_image_mismatch": bool(allow_image_mismatch),
        "current_visual_reidentification_performed": False,
    }
    try:
        reset_response = _bridge_request(connection, {"op": "reset"})
        if reset_response.get("error"):
            raise RuntimeError(f"Bridge reset failed: {reset_response['error']}")
        if reset_response.get("instruction") != archive_episode["instruction"]:
            report["status"] = "initial_task_mismatch"
            report["failure_reason"] = "Bridge reset instruction differs from archive decision instruction"
            return report

        last_response = reset_response
        for index, action in enumerate(archive_episode["replay_actions"]):
            last_response = _bridge_request(
                connection,
                {"op": "step", "action": action.tolist()},
            )
            if last_response.get("error"):
                report["status"] = "replay_failed"
                report["failure_reason"] = f"Bridge failed while replaying archived step {index}: {last_response['error']}"
                report["replayed_steps"] = index
                return report
            if last_response.get("done") is True and index + 1 < len(archive_episode["replay_actions"]):
                report["status"] = "replay_terminated_early"
                report["failure_reason"] = f"Bridge terminated at replay action {index} before the target decision"
                report["replayed_steps"] = index + 1
                return report

        report["replayed_steps"] = len(archive_episode["replay_actions"])
        observation_comparison = compare_observation(
            archive_episode["expected_observation"], last_response, state_atol
        )
        report["decision_observation_comparison"] = observation_comparison
        report["historical_decision_state_matches"] = (
            observation_comparison["instruction_matched"]
            and observation_comparison["state_matched"]
        )
        report["historical_decision_images_pixel_exact"] = observation_comparison["images_matched"]
        report["historical_decision_observation_matches"] = observation_comparison["matched"]
        report["visual_diagnostics"] = _save_visual_diagnostics(
            archive_episode["expected_observation"],
            last_response,
            output_path,
        )
        if not observation_comparison["instruction_matched"]:
            report["status"] = "initial_task_mismatch"
            report["failure_reason"] = "Bridge task instruction differs from the archived decision"
            return report
        if not observation_comparison["state_matched"]:
            report["status"] = "initial_state_mismatch"
            report["failure_reason"] = "Replayed robot state differs from the archived decision; no Snapshot or branch was created"
            return report
        if not observation_comparison["images_available"]:
            report["status"] = "current_camera_unavailable"
            report["failure_reason"] = "Could not decode both current Bridge camera images"
            return report
        if not observation_comparison["images_matched"] and not allow_image_mismatch:
            report["status"] = "initial_image_mismatch"
            report["failure_reason"] = "Raw camera images differ from archive; strict mode stopped before Snapshot/branch"
            return report

        geometry_inventory = _bridge_request(connection, {"op": "geometry_inventory"})
        gripper_inventory = _bridge_request(connection, {"op": "gripper_geometry_inventory"})
        report["geometry_inventory"] = _geometry_summary(geometry_inventory)
        binding_validation = _validate_current_bindings(
            teacher,
            semantic_results,
            geometry_inventory,
        )
        report["semantic_binding_validation"] = binding_validation
        if not binding_validation["valid"]:
            report["status"] = "binding_validation_failed"
            report["failure_reason"] = "Historical semantic bindings are missing, conflicting, or disagree with the active Geometry Inventory"
            return report

        snapshot_response = _bridge_request(connection, {"op": "snapshot"})
        if snapshot_response.get("error") or not isinstance(snapshot_response.get("snapshot_id"), str):
            report["status"] = "snapshot_failed"
            report["failure_reason"] = snapshot_response.get("error", "Bridge did not return a snapshot_id")
            return report
        report["snapshot_created"] = True
        snapshot_id = snapshot_response["snapshot_id"]
        snapshot_state_before = snapshot_response.get("snapshot_state")

        roundtrip_response = _bridge_request(
            connection,
            {"op": "restore", "snapshot_id": snapshot_id},
        )
        if roundtrip_response.get("error"):
            report["snapshot_restore_roundtrip"] = {
                "matched": False,
                "error": roundtrip_response["error"],
            }
        elif roundtrip_response.get("restored_snapshot_id") != snapshot_id:
            report["snapshot_restore_roundtrip"] = {
                "matched": False,
                "error": "Bridge restored a different Snapshot ID",
            }
        elif not isinstance(snapshot_state_before, dict) or not isinstance(
            roundtrip_response.get("snapshot_state"), dict
        ):
            report["snapshot_restore_roundtrip"] = {
                "matched": False,
                "error": "Snapshot or immediate Restore omitted state diagnostics",
            }
        else:
            report["snapshot_restore_roundtrip"] = _compare_nested(
                snapshot_state_before,
                roundtrip_response["snapshot_state"],
                state_atol,
            )
            report["snapshot_restore_roundtrip"]["restored_snapshot_id"] = snapshot_id
        if not report["snapshot_restore_roundtrip"].get("matched"):
            report["status"] = "snapshot_restore_roundtrip_failed"
            report["failure_reason"] = "Immediate Snapshot→Restore did not reproduce its canonical diagnostics"
            return report

        candidate_id = f"archived-decision-{archive_episode['decision_step']}"
        repeat_candidate_id = f"{candidate_id}-repeat"
        branch_result = _bridge_request(
            connection,
            {
                "op": "branch_rollout",
                "snapshot_id": snapshot_id,
                "candidates": [
                    {"candidate_id": candidate_id, "actions": archive_episode["action_chunk"].tolist()},
                    {"candidate_id": repeat_candidate_id, "actions": archive_episode["action_chunk"].tolist()},
                ],
            },
        )
        report["branch_called"] = True
        report["bridge_main_state_restored"] = branch_result.get("main_state_restored") is True
        coverage = _branch_coverage(branch_result, len(archive_episode["action_chunk"]))
        report["branch_coverage"] = coverage
        repeat_consistency = _branch_repeat_consistency(branch_result, state_atol)
        report["repeat_candidate_consistency"] = repeat_consistency

        compiler_result = compile_forbid_contact(
            teacher,
            object_id,
            semantic_results.get(object_id),
            branch_result,
            geometry_inventory,
            gripper_inventory,
            object_semantic_results=semantic_results,
        )
        report["compiler_result"] = compiler_result
        report["compiler_verdict"] = compiler_result.get("status")
        report["contact_evidence"] = _object_contact_evidence(compiler_result)
        if compiler_result.get("status") == "violation":
            report["current_scene_physical_collision"] = "detected"
        elif (
            compiler_result.get("status") == "safe"
            and coverage["full_chunk_executed"]
            and coverage["contact_monitoring_complete"]
            and repeat_consistency["consistent"]
        ):
            report["current_scene_physical_collision"] = "not_detected_in_this_chunk"
        else:
            report["current_scene_physical_collision"] = "unknown"

        # The Bridge's branch result flag is checked independently against the
        # same-state diagnostics captured before and after branch execution.
        post_snapshot = _bridge_request(connection, {"op": "snapshot"})
        if post_snapshot.get("error") or not isinstance(post_snapshot.get("snapshot_state"), dict):
            restoration_comparison = {
                "matched": False,
                "error": post_snapshot.get("error", "Post-branch snapshot has no diagnostics"),
            }
        elif not isinstance(snapshot_state_before, dict):
            restoration_comparison = {
                "matched": False,
                "error": "Pre-branch snapshot has no diagnostics",
            }
        else:
            restoration_comparison = _compare_nested(snapshot_state_before, post_snapshot["snapshot_state"], state_atol)
        report["main_state_restoration"] = {
            "bridge_reported_restored": report["bridge_main_state_restored"],
            "snapshot_diagnostics_match": restoration_comparison.get("matched") is True,
            "comparison": restoration_comparison,
        }

        if not coverage["full_chunk_executed"]:
            report["status"] = "incomplete_chunk"
            report["failure_reason"] = "Branch did not execute and record the complete archived Action Chunk"
        elif not coverage["contact_monitoring_complete"]:
            report["status"] = "incomplete_contact_monitoring"
            report["failure_reason"] = "Per-physics-substep contact coverage is incomplete"
        elif not report["bridge_main_state_restored"] or not restoration_comparison.get("matched"):
            report["status"] = "main_state_restore_failed"
            report["failure_reason"] = "Bridge did not restore the main environment to its pre-branch Snapshot"
        elif not repeat_consistency["consistent"]:
            report["status"] = "repeated_branch_result_mismatch"
            report["failure_reason"] = "Repeated identical candidates from one Snapshot produced different state/contact results"
        elif compiler_result.get("status") == "violation":
            report["status"] = "current_scene_physical_collision_detected"
        elif compiler_result.get("status") == "safe":
            report["status"] = "current_scene_no_collision_observed_in_chunk"
        else:
            report["status"] = "unknown"
            report["failure_reason"] = "Compiler could not establish a complete object-contact verdict"
        report["snapshot_id"] = snapshot_id
        report["branch_result_summary"] = {
            "error": branch_result.get("error"),
            "candidate_count": len(branch_result.get("candidates", [])) if isinstance(branch_result.get("candidates"), list) else None,
        }
        return report
    except Exception as exc:
        report["status"] = "bridge_or_compiler_error"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("logs/gs_opd/on_policy_safety_l0t0_osc_fixed.npz"),
    )
    parser.add_argument("--teacher-json", type=Path, required=True)
    parser.add_argument("--semantic-results-json", type=Path, required=True)
    parser.add_argument(
        "--object-id",
        help="semantic object used as the Compiler target; defaults to the first forbid-rule subject",
    )
    parser.add_argument("--decision-step", type=int, default=70)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT)
    parser.add_argument("--state-atol", type=float, default=1e-6)
    parser.add_argument(
        "--allow-image-mismatch",
        action="store_true",
        help=(
            "treat raw RGB pixel differences as diagnostics after instruction/state checks; "
            "this does not establish exact historical replay or current visual re-identification"
        ),
    )
    args = parser.parse_args(argv)

    if args.decision_step < 1:
        parser.error("--decision-step must be at least 1")
    if not math.isfinite(args.state_atol) or args.state_atol < 0:
        parser.error("--state-atol must be finite and non-negative")
    for label, path in (
        ("archive", args.archive),
        ("Safety IR v2", args.teacher_json),
        ("semantic binding map", args.semantic_results_json),
    ):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")

    try:
        archive_episode = load_archive_episode(args.archive, args.decision_step)
        teacher = parse_teacher_output_v2(args.teacher_json.read_text(encoding="utf-8"))
        semantic_results = _read_json(args.semantic_results_json, "semantic binding map")
        if not isinstance(semantic_results, dict):
            raise ValueError("semantic binding map must be an object keyed by Teacher object ID")
        object_id = args.object_id
        if object_id is None:
            object_rules = teacher.get("object_contact_rules", [])
            object_id = next(
                (
                    rule.get("subject_id")
                    for rule in object_rules
                    if isinstance(rule, dict)
                    and rule.get("decision") == "forbid"
                    and isinstance(rule.get("subject_id"), str)
                ),
                None,
            )
        if not isinstance(object_id, str):
            raise ValueError("Teacher JSON contains no forbid object-contact rule; pass --object-id only if the rule is present")
        if not teacher.get("object_contact_rules"):
            raise ValueError("Safety IR v2 has no object_contact_rules to compile")

        connection = Client((args.host, args.port), authkey=AUTHKEY)
        try:
            report = run_experiment(
                connection,
                archive_episode,
                teacher,
                semantic_results,
                object_id,
                args.state_atol,
                allow_image_mismatch=args.allow_image_mismatch,
                output_path=args.out,
            )
        finally:
            connection.close()
    except Exception as exc:
        report = {
            "status": "preflight_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "archive": str(args.archive.resolve()),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Replay contact Smoke: {report.get('status')}")
    if report.get("failure_reason"):
        print(f"Reason: {report['failure_reason']}")
    if report.get("decision_observation_comparison"):
        comparison = report["decision_observation_comparison"]
        print(
            "Step decision observation match:",
            comparison["matched"],
            "state:", comparison["state_matched"],
            "images:", comparison["images_matched"],
        )
    if report.get("compiler_verdict"):
        print("Compiler verdict:", report["compiler_verdict"])
    if report.get("snapshot_restore_roundtrip"):
        roundtrip = report["snapshot_restore_roundtrip"]
        print(
            "Immediate Snapshot→Restore:",
            roundtrip.get("matched"),
            "max_abs_delta:",
            roundtrip.get("max_abs_delta"),
        )
    if report.get("repeat_candidate_consistency"):
        repeated = report["repeat_candidate_consistency"]
        print(
            "Repeated identical branch:",
            repeated.get("consistent"),
            "max_abs_delta:",
            repeated.get("max_abs_delta"),
        )
    if report.get("historical_decision_state_matches") is not None:
        print("Historical decision task/state match:", report["historical_decision_state_matches"])
        print("Historical decision images pixel-exact:", report["historical_decision_images_pixel_exact"])
    if report.get("current_scene_physical_collision"):
        print("Current-scene physical collision:", report["current_scene_physical_collision"])
    if report.get("semantic_binding_validation"):
        print("Current Geometry Inventory bindings:", report["semantic_binding_validation"]["valid"])
    print(f"Result: {args.out.resolve()}")
    if report.get("status") in {
        "current_scene_physical_collision_detected",
    }:
        return 0
    if report.get("status") in {
        "current_scene_no_collision_observed_in_chunk",
    }:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
