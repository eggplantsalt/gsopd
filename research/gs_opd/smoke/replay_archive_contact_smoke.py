"""Replay an archived episode prefix and verify one archived object-contact branch.

This Smoke does not invoke a policy or change the main rollout. It resets the
fixed Bridge environment, replays the archived executed actions up to one
decision, verifies the resulting observation against the archived decision,
then snapshots and branches the archived predicted Action Chunk exactly once.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.smoke.contact_safety_compiler_smoke import (
    AUTHKEY,
    BRIDGE_PORT,
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


def _branch_coverage(branch_result: Any, expected_steps: int) -> dict[str, Any]:
    candidate = None
    if isinstance(branch_result, dict) and isinstance(branch_result.get("candidates"), list):
        if len(branch_result["candidates"]) == 1 and isinstance(branch_result["candidates"][0], dict):
            candidate = branch_result["candidates"][0]
    if candidate is None:
        return {"full_chunk_executed": False, "contact_monitoring_complete": False, "reason": "branch_candidate_missing_or_ambiguous"}

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
        "requested_steps": requested,
        "executed_steps": executed,
        "step_record_count": len(steps) if isinstance(steps, list) else None,
        "contact_monitoring_complete": bool(coverage),
        "contact_monitoring_status": monitor.get("status") if isinstance(monitor, dict) else None,
        "expected_physics_substeps_total": monitor.get("expected_physics_substeps_total") if isinstance(monitor, dict) else None,
        "captured_physics_substeps": monitor.get("captured_physics_substeps") if isinstance(monitor, dict) else None,
        "unobserved_expected_substeps": monitor.get("unobserved_expected_substeps") if isinstance(monitor, dict) else None,
        "unavailable_contact_substeps": monitor.get("unavailable_contact_substeps") if isinstance(monitor, dict) else None,
        "control_steps": control_steps,
        "candidate_status": candidate.get("status"),
        "candidate_error": candidate.get("error"),
        "step_rewards": [step.get("reward") for step in steps] if isinstance(steps, list) else None,
        "step_dones": [step.get("done") for step in steps] if isinstance(steps, list) else None,
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
        if not observation_comparison["matched"]:
            report["status"] = "initial_state_mismatch"
            report["failure_reason"] = "Replayed state or raw camera image differs from the archived decision; no Snapshot or branch was created"
            return report

        geometry_inventory = _bridge_request(connection, {"op": "geometry_inventory"})
        gripper_inventory = _bridge_request(connection, {"op": "gripper_geometry_inventory"})
        report["geometry_inventory"] = _geometry_summary(geometry_inventory)
        snapshot_response = _bridge_request(connection, {"op": "snapshot"})
        if snapshot_response.get("error") or not isinstance(snapshot_response.get("snapshot_id"), str):
            report["status"] = "snapshot_failed"
            report["failure_reason"] = snapshot_response.get("error", "Bridge did not return a snapshot_id")
            return report
        report["snapshot_created"] = True
        snapshot_id = snapshot_response["snapshot_id"]
        snapshot_state_before = snapshot_response.get("snapshot_state")

        candidate_id = f"archived-decision-{archive_episode['decision_step']}"
        branch_result = _bridge_request(
            connection,
            {
                "op": "branch_rollout",
                "snapshot_id": snapshot_id,
                "candidates": [
                    {"candidate_id": candidate_id, "actions": archive_episode["action_chunk"].tolist()}
                ],
            },
        )
        report["branch_called"] = True
        report["bridge_main_state_restored"] = branch_result.get("main_state_restored") is True
        coverage = _branch_coverage(branch_result, len(archive_episode["action_chunk"]))
        report["branch_coverage"] = coverage

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
        elif compiler_result.get("status") == "violation":
            report["status"] = "violation_reproduced"
        elif compiler_result.get("status") == "safe":
            report["status"] = "no_violation_observed"
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
    print(f"Result: {args.out.resolve()}")
    if report.get("status") == "violation_reproduced":
        return 0
    if report.get("status") == "no_violation_observed":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
