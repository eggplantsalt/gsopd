"""Replay archived Step 70 and probe verified object-to-object action repair.

This experiment does not run a policy or step the main environment after the
archived prefix. Candidate chunks are evaluated with Bridge branch_rollout from
one Snapshot, then checked by the existing Contact Safety Compiler and Repair
core.
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

from research.gs_opd.repair.verifier_guided_repair import (
    RepairConfig,
    locate_first_violation,
    run_verifier_guided_repair,
)
from research.gs_opd.smoke.contact_safety_compiler_smoke import (
    AUTHKEY,
    BRIDGE_PORT,
    _resolve_target_binding,
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


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read {label} JSON {path}: {exc}") from exc


def _bridge_request(connection: Any, request: dict[str, Any]) -> dict[str, Any]:
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError(f"Bridge returned a non-object response for op={request.get('op')!r}")
    return response


def _choose_compiler_object_id(teacher: dict[str, Any], requested: str | None) -> str:
    object_rules = teacher.get("object_contact_rules")
    if not isinstance(object_rules, list) or not object_rules:
        raise ValueError("Safety IR v2 has no object_contact_rules to compile")
    valid_subjects = sorted({
        rule.get("subject_id")
        for rule in object_rules
        if isinstance(rule, dict)
        and rule.get("decision") == "forbid"
        and isinstance(rule.get("subject_id"), str)
    })
    if requested is not None:
        if requested not in valid_subjects:
            raise ValueError(f"object_id {requested!r} is not a subject of a forbid object-contact rule")
        return requested
    if len(valid_subjects) != 1:
        raise ValueError(
            "Safety IR v2 must have exactly one forbid object-contact subject, "
            "or pass --object-id"
        )
    return valid_subjects[0]


def _resolve_task_object_binding(
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    geometry_inventory: dict[str, Any],
    requested_object_id: str | None,
) -> dict[str, Any]:
    objects = {
        item.get("id"): item
        for item in teacher.get("objects", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    target_ids = sorted(
        object_id for object_id, item in objects.items()
        if item.get("role") == "target_object"
    )
    if requested_object_id is None:
        if len(target_ids) != 1:
            raise ValueError(
                "Expected one Safety IR target_object for task-progress validation; "
                "pass --task-object-id when the Teacher lists multiple targets"
            )
        requested_object_id = target_ids[0]
    teacher_object = objects.get(requested_object_id)
    semantic = semantic_results.get(requested_object_id)
    if teacher_object is None or teacher_object.get("role") != "target_object":
        raise ValueError("Task-progress binding must reference a Safety IR target_object")
    if not isinstance(semantic, dict):
        raise ValueError("Semantic binding for the target_object is missing")
    semantic_teacher = semantic.get("teacher_object")
    if (
        not isinstance(semantic_teacher, dict)
        or semantic_teacher.get("object_id") != requested_object_id
        or semantic_teacher.get("name") != teacher_object.get("name")
        or semantic_teacher.get("role") != "target_object"
    ):
        raise ValueError("Target-object semantic binding does not match the current Teacher object")
    physical = semantic.get("physical_association")
    binding, _geom_by_id, _body_by_id, error = _resolve_target_binding(
        semantic, requested_object_id, geometry_inventory
    )
    if error is not None or binding is None:
        raise ValueError(f"Target-object Geometry Inventory binding is invalid: {error}")
    if not isinstance(physical, dict) or physical.get("status") != "matched_by_iou":
        raise ValueError("Target-object physical association is not a resolved IoU binding")
    if physical.get("body_id") != binding["body_id"]:
        raise ValueError("Target-object semantic Body ID conflicts with its active model binding")
    return {
        "object_id": requested_object_id,
        "status": physical["status"],
        "role": "target_object",
        "body_id": binding["body_id"],
        "body_name": binding["body_name"],
        "geom_ids": binding["geom_ids"],
        "geom_names": binding["geom_names"],
    }


def _branch_and_compile(
    connection: Any,
    snapshot_id: str,
    snapshot_state: dict[str, Any],
    specifications: list[dict[str, Any]],
    teacher: dict[str, Any],
    object_id: str,
    semantic_results: dict[str, Any],
    geometry_inventory: dict[str, Any],
    gripper_inventory: dict[str, Any],
    state_atol: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    branch = _bridge_request(connection, {
        "op": "branch_rollout",
        "snapshot_id": snapshot_id,
        "candidates": [
            {"candidate_id": item["candidate_id"], "actions": item["actions"]}
            for item in specifications
        ],
    })
    after = _bridge_request(connection, {"op": "snapshot"})
    if after.get("error") or not isinstance(after.get("snapshot_state"), dict):
        restoration = {
            "matched": False,
            "error": after.get("error", "post-branch Snapshot omitted state diagnostics"),
        }
    else:
        restoration = _compare_nested(snapshot_state, after["snapshot_state"], state_atol)
    bridge_reported_restored = branch.get("main_state_restored") is True
    restored = bridge_reported_restored and restoration.get("matched") is True
    branch = {**branch, "main_state_restored": restored}
    compiled = compile_forbid_contact(
        teacher,
        object_id,
        semantic_results.get(object_id),
        branch,
        geometry_inventory,
        gripper_inventory,
        object_semantic_results=semantic_results,
    )
    return branch, compiled, {
        "bridge_reported_restored": bridge_reported_restored,
        "snapshot_diagnostics_match": restoration.get("matched") is True,
        "comparison": restoration,
    }


def _candidate_result(branch: dict[str, Any], compiled: dict[str, Any], candidate_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    branch_candidate = next(
        (item for item in branch.get("candidates", [])
         if isinstance(item, dict) and item.get("candidate_id") == candidate_id),
        None,
    )
    compiled_candidate = next(
        (item for item in compiled.get("candidates", [])
         if isinstance(item, dict) and item.get("candidate_id") == candidate_id),
        None,
    )
    return (
        {**branch, "candidates": [branch_candidate] if branch_candidate is not None else []},
        {**compiled, "candidates": [compiled_candidate] if compiled_candidate is not None else []},
    )


def _candidate_attempt_summary(attempt: dict[str, Any], execution: dict[str, Any] | None) -> dict[str, Any]:
    summary = {
        key: value for key, value in attempt.items()
        if key not in ("branch_result", "compiled_result")
    }
    if execution is not None:
        summary["execution_validation"] = execution
    return summary


def run_experiment(
    connection: Any,
    archive_episode: dict[str, Any],
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    object_id: str,
    task_object_id: str | None,
    *,
    state_atol: float = 1e-6,
    allow_image_mismatch: bool = False,
    repair_config: RepairConfig = RepairConfig(),
) -> dict[str, Any]:
    """Run guarded archive replay, A0 verification, and same-snapshot repair search."""
    actions = archive_episode["action_chunk"]
    report: dict[str, Any] = {
        "experiment": "object_to_object_geometry_guided_repair_v1",
        "archive": archive_episode["archive"],
        "decision_step": archive_episode["decision_step"],
        "instruction": archive_episode["instruction"],
        "object_id": object_id,
        "requested_task_object_id": task_object_id,
        "action_chunk_A0": actions.tolist(),
        "action_chunk_length": int(len(actions)),
        "allow_image_mismatch": bool(allow_image_mismatch),
        "historical_decision_exact_replay": None,
        "current_visual_reidentification_performed": False,
        "snapshot_created": False,
        "status": "failed",
    }
    try:
        reset = _bridge_request(connection, {"op": "reset"})
        if reset.get("error"):
            raise RuntimeError(f"Bridge reset failed: {reset['error']}")
        if reset.get("instruction") != archive_episode["instruction"]:
            report.update({"status": "task_instruction_mismatch", "failure_reason": "reset_instruction_differs_from_archive"})
            return report

        last_response = reset
        for index, action in enumerate(archive_episode["replay_actions"]):
            last_response = _bridge_request(connection, {"op": "step", "action": action.tolist()})
            if last_response.get("error"):
                report.update({
                    "status": "archive_replay_failed",
                    "failure_reason": f"bridge_step_{index}:{last_response['error']}",
                })
                return report
            if last_response.get("done") is True and index + 1 < len(archive_episode["replay_actions"]):
                report.update({"status": "archive_replay_terminated_early", "failure_reason": f"replay_done_at_step_{index}"})
                return report
        observation = compare_observation(
            archive_episode["expected_observation"], last_response, state_atol
        )
        report["decision_observation_comparison"] = observation
        report["historical_decision_state_matches"] = observation["instruction_matched"] and observation["state_matched"]
        report["historical_decision_images_pixel_exact"] = observation["images_matched"]
        report["historical_decision_exact_replay"] = observation["matched"]
        if not observation["instruction_matched"] or not observation["state_matched"]:
            report.update({"status": "archive_decision_state_mismatch", "failure_reason": "instruction_or_robot_state_did_not_match_archived_decision"})
            return report
        if not observation["images_available"]:
            report.update({"status": "current_camera_unavailable", "failure_reason": "could_not_decode_both_current_rgb_images"})
            return report
        if not observation["images_matched"] and not allow_image_mismatch:
            report.update({"status": "archive_image_mismatch", "failure_reason": "pixel_mismatch_requires_--allow-image-mismatch"})
            return report

        geometry_inventory = _bridge_request(connection, {"op": "geometry_inventory"})
        gripper_inventory = _bridge_request(connection, {"op": "gripper_geometry_inventory"})
        binding_validation = _validate_current_bindings(teacher, semantic_results, geometry_inventory)
        report["semantic_binding_validation"] = binding_validation
        if not binding_validation["valid"]:
            report.update({"status": "semantic_binding_validation_failed", "failure_reason": "existing semantic bindings do not match active Geometry Inventory"})
            return report
        task_binding = _resolve_task_object_binding(
            teacher, semantic_results, geometry_inventory, task_object_id
        )
        report["task_object_binding"] = task_binding
        controller_context = _bridge_request(connection, {"op": "repair_context"})
        report["repair_context"] = controller_context
        if controller_context.get("status") != "available":
            report.update({"status": "repair_context_unavailable", "failure_reason": "Bridge did not return active controller bounds and state"})
            return report

        snapshot = _bridge_request(connection, {"op": "snapshot"})
        if snapshot.get("error") or not isinstance(snapshot.get("snapshot_id"), str):
            report.update({"status": "snapshot_failed", "failure_reason": snapshot.get("error", "Snapshot returned no ID")})
            return report
        snapshot_id = snapshot["snapshot_id"]
        snapshot_state = snapshot.get("snapshot_state")
        report["snapshot_id"] = snapshot_id
        report["snapshot_created"] = True
        report["snapshot_state_before"] = snapshot_state
        roundtrip = _bridge_request(connection, {"op": "restore", "snapshot_id": snapshot_id})
        if (
            roundtrip.get("error")
            or roundtrip.get("restored_snapshot_id") != snapshot_id
            or not isinstance(snapshot_state, dict)
            or not isinstance(roundtrip.get("snapshot_state"), dict)
        ):
            roundtrip_check = {"matched": False, "error": roundtrip.get("error", "immediate Restore omitted or changed Snapshot diagnostics")}
        else:
            roundtrip_check = _compare_nested(snapshot_state, roundtrip["snapshot_state"], state_atol)
        report["snapshot_restore_roundtrip"] = roundtrip_check
        if not roundtrip_check.get("matched"):
            report.update({"status": "snapshot_restore_roundtrip_failed", "failure_reason": "Snapshot→Restore diagnostics did not match"})
            return report

        baseline_id = f"repair-A0-step-{archive_episode['decision_step']}"
        baseline_branch, baseline_compiled, baseline_restore = _branch_and_compile(
            connection,
            snapshot_id,
            snapshot_state,
            [{"candidate_id": baseline_id, "actions": actions.tolist()}],
            teacher,
            object_id,
            semantic_results,
            geometry_inventory,
            gripper_inventory,
            state_atol,
        )
        report["baseline_main_state_restoration"] = baseline_restore
        coverage = _candidate_coverage(
            next((item for item in baseline_branch.get("candidates", []) if isinstance(item, dict)), {}),
            len(actions),
        )
        report["baseline_execution_coverage"] = coverage
        report["baseline_compiler_verdict"] = baseline_compiled.get("status")
        report["baseline_compiler_result"] = baseline_compiled
        baseline_candidate = next(
            (item for item in baseline_branch.get("candidates", [])
             if isinstance(item, dict) and item.get("candidate_id") == baseline_id),
            None,
        )
        baseline_compiled_candidate = next(
            (item for item in baseline_compiled.get("candidates", [])
             if isinstance(item, dict) and item.get("candidate_id") == baseline_id),
            None,
        )
        if not baseline_restore.get("bridge_reported_restored") or not baseline_restore.get("snapshot_diagnostics_match"):
            report.update({"status": "baseline_main_state_restore_failed", "failure_reason": "A0 branch did not restore the pre-branch main state"})
            return report
        if not coverage["full_chunk_executed"] or not coverage["contact_monitoring_complete"]:
            report.update({"status": "baseline_chunk_or_monitoring_incomplete", "failure_reason": "A0 must have full action and physics-substep contact coverage"})
            return report
        if baseline_compiled.get("status") != "violation":
            report.update({
                "status": "baseline_not_explicitly_violating",
                "failure_reason": "A0 must be explicitly violation under the current object-contact Compiler",
                "baseline_compiled_result": baseline_compiled,
            })
            return report
        baseline_violation, violation_error = locate_first_violation(baseline_branch, baseline_compiled)
        report["first_violation"] = baseline_violation
        if baseline_violation is None:
            report.update({"status": "baseline_violation_evidence_unresolved", "failure_reason": violation_error})
            return report
        if baseline_violation.get("contact_type") != "object_to_object":
            report.update({"status": "baseline_first_violation_not_object_contact", "failure_reason": "earliest compiled violation is not a joinable object-to-object contact"})
            return report
        if baseline_violation.get("moving_object_id") != task_binding["object_id"]:
            report.update({
                "status": "task_object_does_not_match_violation_subject",
                "failure_reason": "the current target_object binding is not the moving subject of the first object-contact violation",
            })
            return report
        report["baseline_branch_summary"] = {
            "candidate_id": baseline_id,
            "status": baseline_candidate.get("status") if baseline_candidate else None,
            "requested_steps": baseline_candidate.get("requested_steps") if baseline_candidate else None,
            "executed_steps": baseline_candidate.get("executed_steps") if baseline_candidate else None,
            "step_rewards": [item.get("reward") for item in baseline_candidate.get("steps", [])] if baseline_candidate else None,
            "step_dones": [item.get("done") for item in baseline_candidate.get("steps", [])] if baseline_candidate else None,
            "contact_monitoring": coverage,
        }
        report["baseline_compiled_evidence"] = baseline_compiled_candidate

        candidate_executions: dict[str, dict[str, Any]] = {}

        def evaluate_candidates(specifications: list[dict[str, Any]]) -> list[dict[str, Any]]:
            branch, compiled, restoration = _branch_and_compile(
                connection,
                snapshot_id,
                snapshot_state,
                specifications,
                teacher,
                object_id,
                semantic_results,
                geometry_inventory,
                gripper_inventory,
                state_atol,
            )
            results = []
            for item in specifications:
                candidate_id = item["candidate_id"]
                branch_subset, compiled_subset = _candidate_result(branch, compiled, candidate_id)
                candidate = branch_subset["candidates"][0] if branch_subset["candidates"] else {}
                candidate_compiled = compiled_subset["candidates"][0] if compiled_subset["candidates"] else {}
                candidate_coverage = _candidate_coverage(candidate, len(actions))
                candidate_executions[candidate_id] = {
                    "candidate_id": candidate_id,
                    "main_state_restoration": restoration,
                    "branch_coverage": candidate_coverage,
                    "compiler_verdict": candidate_compiled.get("verdict"),
                    "compiled_candidate": candidate_compiled,
                }
                results.append({
                    "candidate_id": candidate_id,
                    "branch_result": branch_subset,
                    "compiled_result": compiled_subset,
                })
            return results

        repair = run_verifier_guided_repair(
            actions,
            baseline_branch,
            baseline_compiled,
            controller_context,
            evaluate_candidates,
            task_object_binding=task_binding,
            config=repair_config,
        )
        report["repair_search"] = {
            key: value for key, value in repair.items()
            if key != "search_attempts"
        }
        report["candidate_search_records"] = [
            _candidate_attempt_summary(attempt, candidate_executions.get(attempt.get("candidate_id")))
            for attempt in repair.get("search_attempts", [])
        ]
        safe_candidates = [
            value for value in candidate_executions.values()
            if value.get("compiler_verdict") == "safe"
            and value.get("branch_coverage", {}).get("full_chunk_executed")
            and value.get("branch_coverage", {}).get("contact_monitoring_complete")
        ]
        task_rejected_safe = [
            attempt for attempt in repair.get("search_attempts", [])
            if attempt.get("safety_verdict") == "safe"
            and attempt.get("task_progress", {}).get("status") != "pass"
        ]
        if repair.get("status") == "repaired_candidate_found":
            report["status"] = "repaired_candidate_found"
            report["repaired_action_chunk_A_plus"] = repair.get("repaired_action_chunk_A_plus")
            report["final_candidate_id"] = repair.get("selected_candidate_id")
            report["candidate_outcome_classification"] = "verified_safe_progress_preserving_A_plus_found"
        elif safe_candidates and task_rejected_safe:
            report["status"] = "safe_candidates_failed_task_progress"
            report["repaired_action_chunk_A_plus"] = None
            report["candidate_outcome_classification"] = "safe_candidate_found_but_task_progress_not_preserved"
            report["failure_reason"] = "at least one complete compiler-safe candidate failed target-object task-progress validation"
        elif not safe_candidates:
            report["status"] = "no_safe_candidate_found"
            report["repaired_action_chunk_A_plus"] = None
            report["candidate_outcome_classification"] = "no_complete_compiler_safe_candidate_found"
            report["failure_reason"] = repair.get("failure_reason")
        else:
            report["status"] = repair.get("status", "unknown")
            report["repaired_action_chunk_A_plus"] = None
            report["candidate_outcome_classification"] = "safe_candidate_rejected_by_other_repair_gate"
            report["failure_reason"] = repair.get("failure_reason")
        report["candidate_counts"] = {
            "evaluated": len(candidate_executions),
            "complete_compiler_safe": len(safe_candidates),
            "safe_but_task_progress_failed": len(task_rejected_safe),
            "accepted_A_plus": int(report["repaired_action_chunk_A_plus"] is not None),
        }
        report["all_branch_main_state_restores_passed"] = all(
            item.get("main_state_restoration", {}).get("bridge_reported_restored") is True
            and item.get("main_state_restoration", {}).get("snapshot_diagnostics_match") is True
            for item in candidate_executions.values()
        ) and baseline_restore.get("bridge_reported_restored") is True and baseline_restore.get("snapshot_diagnostics_match") is True
        if report["all_branch_main_state_restores_passed"] is not True:
            report["status"] = "aborted_main_state_restore_failed"
            report["failure_reason"] = "one or more branch operations failed independent main-state Snapshot diagnostics"
            report["repaired_action_chunk_A_plus"] = None
        return report
    except Exception as exc:
        report["status"] = "experiment_error"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("logs/gs_opd/on_policy_safety_l0t0_osc_fixed.npz"))
    parser.add_argument("--teacher-json", type=Path, required=True)
    parser.add_argument("--semantic-results-json", type=Path, required=True)
    parser.add_argument("--object-id", help="forbid object-contact rule subject; inferred when unique")
    parser.add_argument("--task-object-id", help="target_object used for task-progress verification; inferred when unique")
    parser.add_argument("--decision-step", type=int, default=70)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT)
    parser.add_argument("--state-atol", type=float, default=1e-6)
    parser.add_argument(
        "--allow-image-mismatch",
        action="store_true",
        help="retain raw RGB mismatch as a diagnostic after exact instruction and robot-state checks",
    )
    parser.add_argument("--max-candidate-evaluations", type=int, default=96)
    args = parser.parse_args(argv)
    if args.decision_step < 1:
        parser.error("--decision-step must be positive")
    if not math.isfinite(args.state_atol) or args.state_atol < 0:
        parser.error("--state-atol must be finite and non-negative")
    if args.max_candidate_evaluations < 1:
        parser.error("--max-candidate-evaluations must be positive")
    for label, path in (
        ("archive", args.archive),
        ("Safety IR v2", args.teacher_json),
        ("semantic bindings", args.semantic_results_json),
    ):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    config = RepairConfig(max_candidate_evaluations=args.max_candidate_evaluations)
    try:
        episode = load_archive_episode(args.archive, args.decision_step)
        teacher = parse_teacher_output_v2(args.teacher_json.read_text(encoding="utf-8"))
        semantic_results = _read_json(args.semantic_results_json, "semantic results")
        if not isinstance(semantic_results, dict):
            raise ValueError("semantic results must be keyed by Teacher object ID")
        object_id = _choose_compiler_object_id(teacher, args.object_id)
        connection = Client((args.host, args.port), authkey=AUTHKEY)
        try:
            report = run_experiment(
                connection,
                episode,
                teacher,
                semantic_results,
                object_id,
                args.task_object_id,
                state_atol=args.state_atol,
                allow_image_mismatch=args.allow_image_mismatch,
                repair_config=config,
            )
        finally:
            connection.close()
    except Exception as exc:
        report = {
            "status": "preflight_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "archive": str(args.archive.resolve()),
            "decision_step": args.decision_step,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Object Geometry-Guided Repair Smoke: {report.get('status')}")
    print(f"A0 first violation: {json.dumps(report.get('first_violation'), ensure_ascii=False)}")
    print(f"Candidate counts: {report.get('candidate_counts')}")
    print(f"A+: {'present' if report.get('repaired_action_chunk_A_plus') is not None else 'none'}")
    if report.get("failure_reason"):
        print(f"Reason: {report['failure_reason']}")
    print(f"Result: {args.out.resolve()}")
    if report.get("status") in (
        "repaired_candidate_found",
        "no_safe_candidate_found",
        "safe_candidates_failed_task_progress",
        "safe_candidate_rejected_by_other_repair_gate",
    ):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
