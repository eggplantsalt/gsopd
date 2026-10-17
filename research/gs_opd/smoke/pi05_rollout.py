Warning: truncated output (original token count: 17009)
Total output lines: 1553

import argparse
import os
from pathlib import Path
from multiprocessing.connection import Client

os.environ["EMBODIED_PATH"] = "/workspace/zy_workspace/RLinf/examples/embodiment"

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rlinf.models import get_model
from rlinf.envs.sim.libero.utils import get_libero_image, get_libero_wrist_image, quat2axisangle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-out",
        type=Path,
        help="Optional MP4 path for the agentview rollout video.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=240,
        help="Maximum number of environment actions to execute (default: 240).",
    )
    parser.add_argument(
        "--trace-out",
        type=Path,
        help="Optional NPZ path for on-policy rollout transitions.",
    )
    parser.add_argument(
        "--contact-safety-probe",
        action="store_true",
        help="Probe the first predicted Action Chunk with Bridge branching, then stop before real actions.",
    )
    parser.add_argument(
        "--verifier-guided-repair",
        action="store_true",
        help="Search Snapshot-verified local XYZ repairs for the first predicted Action Chunk, then stop.",
    )
    parser.add_argument(
        "--on-policy-safety-collection",
        action="store_true",
        help="Verify every predicted chunk with Snapshot/Branch Rollout, collect labels, and continue the original Student actions.",
    )
    parser.add_argument("--teacher-json", type=Path)
    parser.add_argument("--object-id")
    parser.add_argument("--semantic-result", type=Path)
    parser.add_argument(
        "--semantic-results-json",
        type=Path,
        help="Optional mapping from Teacher object IDs to current SAM/IoU semantic bindings for On-Policy collection.",
    )
    parser.add_argument("--probe-out-dir", type=Path)
    parser.add_argument("--repair-out-dir", type=Path)
    parser.add_argument("--on-policy-safety-out", type=Path)
    parser.add_argument("--task-object-id")
    parser.add_argument("--task-object-semantic-result", type=Path)
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be zero or greater")
    if args.video_out is not None and args.video_out.suffix != ".mp4":
        parser.error("--video-out must use the .mp4 extension")
    if args.trace_out is not None and args.trace_out.suffix != ".npz":
        parser.error("--trace-out must use the .npz extension")
    probe_inputs = (
        args.teacher_json,
        args.object_id,
        args.semantic_result,
    )
    if args.contact_safety_probe and args.verifier_guided_repair:
        parser.error("--contact-safety-probe and --verifier-guided-repair are mutually exclusive")
    probe_enabled = args.contact_safety_probe or args.verifier_guided_repair
    collection_enabled = args.on_policy_safety_collection
    if probe_enabled and collection_enabled:
        parser.error("branch probe modes cannot be combined with --on-policy-safety-collection")
    if collection_enabled:
        if args.teacher_json is None or args.object_id is None:
            parser.error(
                "--on-policy-safety-collection requires --teacher-json and --object-id"
            )
        if args.semantic_result is not None and args.semantic_results_json is not None:
            parser.error("choose either --semantic-result or --semantic-results-json")
        if args.semantic_result is None and args.semantic_results_json is None:
            parser.error(
                "--on-policy-safety-collection requires --semantic-result or --semantic-results-json"
            )
        if args.on_policy_safety_out is None:
            parser.error("--on-policy-safety-collection requires --on-policy-safety-out")
        if args.on_policy_safety_out.suffix != ".npz":
            parser.error("--on-policy-safety-out must use the .npz extension")
        if args.probe_out_dir is not None or args.repair_out_dir is not None:
            parser.error("probe and repair output directories cannot be used for episode collection")
        if args.trace_out is not None:
            parser.error("--trace-out cannot be combined with --on-policy-safety-collection")
        if args.max_steps == 0:
            parser.error("--on-policy-safety-collection requires --max-steps greater than zero")
        if args.semantic_results_json is None and (
            (args.task_object_id is None) != (args.task_object_semantic_result is None)
        ):
            parser.error("--task-object-id and --task-object-semantic-result must be supplied together")
        if args.semantic_results_json is not None and args.task_object_semantic_result is not None:
            parser.error("--semantic-results-json already supplies the task-object binding; omit --task-object-semantic-result")
        if not args.teacher_json.is_file():
            parser.error(f"Safety IR v2 file does not exist: {args.teacher_json}")
        if args.semantic_result is not None and not args.semantic_result.is_file():
            parser.error(f"Semantic result file does not exist: {args.semantic_result}")
        if args.semantic_results_json is not None and not args.semantic_results_json.is_file():
            parser.error(f"Semantic results mapping does not exist: {args.semantic_results_json}")
        if args.task_object_semantic_result is not None and not args.task_object_semantic_result.is_file():
            parser.error(f"Task object semantic result does not exist: {args.task_object_semantic_result}")
    elif probe_enabled:
        if args.semantic_results_json is not None:
            parser.error("--semantic-results-json is currently supported only by --on-policy-safety-collection")
        if any(value is None for value in probe_inputs):
            parser.error(
                "a probe mode requires --teacher-json, --object-id, "
                "and --semantic-result"
            )
        if args.verifier_guided_repair and args.repair_out_dir is None:
            parser.error("--verifier-guided-repair requires --repair-out-dir")
        if args.on_policy_safety_out is not None:
            parser.error("--on-policy-safety-out requires --on-policy-safety-collection")
        if args.contact_safety_probe and args.probe_out_dir is None:
            parser.error("--contact-safety-probe requires --probe-out-dir")
        if args.verifier_guided_repair and args.probe_out_dir is not None:
            parser.error("--probe-out-dir is only used by --contact-safety-probe")
        if args.contact_safety_probe and args.repair_out_dir is not None:
            parser.error("--repair-out-dir requires --verifier-guided-repair")
        if args.max_steps == 0:
            parser.error("a probe mode requires --max-steps greater than zero")
        if args.trace_out is not None:
            parser.error("--trace-out cannot be combined with a branch probe mode")
        if not args.teacher_json.is_file():
            parser.error(f"Safety IR v2 file does not exist: {args.teacher_json}")
        if not args.semantic_result.is_file():
            parser.error(f"Semantic result file does not exist: {args.semantic_result}")
        if (args.task_object_id is None) != (args.task_object_semantic_result is None):
            parser.error("--task-object-id and --task-object-semantic-result must be supplied together")
        if args.task_object_id is not None and not args.verifier_guided_repair:
            parser.error("task-object binding options require --verifier-guided-repair")
        if args.task_object_semantic_result is not None and not args.task_object_semantic_result.is_file():
            parser.error(f"Task object semantic result does not exist: {args.task_object_semantic_result}")
    elif any(value is not None for value in probe_inputs) or any(
        value is not None for value in (
            args.probe_out_dir,
            args.repair_out_dir,
            args.on_policy_safety_out,
            args.semantic_results_json,
            args.task_object_id,
            args.task_object_semantic_result,
        )
    ):
        parser.error(
            "Teacher, binding, and output arguments require a probe mode"
        )
    return args


def run_first_chunk_contact_probe(
    conn,
    chunk,
    instruction,
    teacher_json_path,
    object_id,
    semantic_result_path,
    out_dir,
):
    """Branch the complete first predicted chunk and compile its contact evidence."""
    import json
    import sys

    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from research.gs_opd.smoke.contact_safety_compiler_smoke import (
        compile_forbid_contact,
    )
    from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2

    teacher = parse_teacher_output_v2(teacher_json_path.read_text(encoding="utf-8"))
    if not any(item["id"] == object_id for item in teacher["objects"]):
        raise ValueError(f"object_id {object_id!r} is absent from Safety IR v2")
    semantic_result = json.loads(semantic_result_path.read_text(encoding="utf-8"))

    def bridge_request(payload):
        conn.send(payload)
        response = conn.recv()
        if not isinstance(response, dict):
            return {"error": "Safety Bridge returned a non-object response"}
        return response

    actions = chunk.tolist()
    candidates = [{"candidate_id": "pi05-first-action-chunk", "actions": actions}]
    geometry_inventory = {"status": "unavailable"}
    gripper_inventory = {"status": "unavailable"}
    branch_result = {"error": "branch_rollout was not started", "main_state_restored": False}
    snapshot_id = None
    try:
        geometry_inventory = bridge_request({"op": "geometry_inventory"})
        gripper_inventory = bridge_request({"op": "gripper_geometry_inventory"})
        snapshot = bridge_request({"op": "snapshot"})
        snapshot_id = snapshot.get("snapshot_id")
        if isinstance(snapshot_id, str):
            branch_result = bridge_request(
                {
                    "op": "branch_rollout",
                    "snapshot_id": snapshot_id,
                    "candidates": candidates,
                }
            )
        else:
            branch_result = {
                "error": snapshot.get("error", "Snapshot operation returned no ID"),
                "main_state_restored": False,
                "candidates": [],
            }
    except (OSError, EOFError) as exc:
        branch_result = {
            "error": f"Safety Bridge request failed: {type(exc).__name__}: {exc}",
            "main_state_restored": False,
            "candidates": [],
        }

    compiled_result = compile_forbid_contact(
        teacher,
        object_id,
        semantic_result,
        branch_result,
        geometry_inventory,
        gripper_inventory,
    )
    report = {
        "mode": "first_action_chunk_contact_safety_probe",
        "instruction": instruction,
        "object_id": object_id,
        "predicted_action_chunk": actions,
        "snapshot_id": snapshot_id,
        "main_state_restored": branch_result.get("main_state_restored") is True,
        "verdict": compiled_result["status"],
        "compiled_result": compiled_result,
        "bridge_geometry_inventory": geometry_inventory,
        "bridge_gripper_geometry_inventory": gripper_inventory,
        "branch_rollout_result": branch_result,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "first_chunk_contact_safety_probe.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "FIRST CHUNK CONTACT SAFETY PROBE:",
        report["verdict"],
        "MAIN STATE RESTORED:",
        report["main_state_restored"],
        flush=True,
    )
    print("PROBE RESULT:", report_path, flush=True)
    return report


def _resolve_task_object_binding(teacher, object_id, semantic_result, geometry_inventory):
    """Require explicit target_object semantics and a model-verified Body binding."""
    teacher_object = next((item for item in teacher["objects"] if item["id"] == object_id), None)
    if teacher_object is None or teacher_object.get("role") != "target_object":
        raise ValueError("task progress object must have Safety IR role target_object")
    if not isinstance(semantic_result, dict):
        raise ValueError("task object semantic result is not a JSON object")
    decision = semantic_result.get("semantic_decision")
    semantic_object = semantic_result.get("teacher_object")
    association = semantic_result.get("physical_association")
    if (
        not isinstance(decision, dict)
        or decision.get("object_id") != object_id
        or decision.get("status") != "resolved"
        or not isinstance(semantic_object, dict)
        or semantic_object.get("object_id") != object_id
        or not isinstance(association, dict)
        or association.get("status") != "matched_by_iou"
    ):
        raise ValueError("task object semantic result is unresolved or lacks a physical association")
    body_id = association.get("body_id")
    body_name = association.get("body_name")
    if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id < 0:
        raise ValueError("task object semantic binding has an invalid Body ID")
    if not isinstance(geometry_inventory, dict) or geometry_inventory.get("status") != "complete":
        raise ValueError("active Bridge geometry inventory is incomplete")
    body_entries = {
        item.get("body_id"): item
        for item in geometry_inventory.get("geometries", [])
        if isinstance(item, dict) and item.get("body_id") == body_id
    }
    if not body_entries or any(item.get("category") != "scene" for item in body_entries.values()):
        raise ValueError("task object Body ID is absent from the active scene model")
    active_names = {item.get("body_name") for item in body_entries.values()}
    if not isinstance(body_name, str) or active_names != {body_name}:
        raise ValueError("task object Body name disagrees with the active model")
    return {
        "status": "matched_by_iou",
        "object_id": object_id,
        "role": "target_object",
        "body_id": body_id,
        "body_name": body_name,
        "physical_object_id": association.get("physical_object_id"),
    }


def run_first_chunk_verifier_guided_repair(
    conn,
    chunk,
    instruction,
    teacher_json_path,
    object_id,
    semantic_result_path,
    out_dir,
    task_object_id=None,
    task_object_semantic_result_path=None,
):
    """Branch A0 and adaptively search verified repairs; never step the main env."""
    import json
    import sys

    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from research.gs_opd.repair.verifier_guided_repair import run_verifier_guided_repair
    from research.gs_opd.smoke.contact_safety_compiler_smoke import compile_forbid_contact
    from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2

    def bridge_request(payload):
        conn.send(payload)
        response = conn.recv()
        if not isinstance(response, dict):
            return {"error": "Safety Bridge returned a non-object response"}
        return response

    actions = np.asarray(chunk, dtype=np.float32)
    report = {
        "mode": "first_chunk_verifier_guided_action_repair_probe",
        "instruction": instruction,
        "object_id": object_id,
        "task_object_id": task_object_id,
        "original_action_chunk_A0": actions.tolist(),
        "snapshot_id": None,
        "main_state_restored": None,
        "status": "unknown",
        "repaired_action_chunk_A_plus": None,
        "bridge_geometry_inventory": {"status": "unavailable"},
        "bridge_gripper_geometry_inventory": {"status": "unavailable"},
        "bridge_repair_context": {"status": "unavailable"},
        "baseline_branch_result": None,
        "baseline_compiled_result": None,
        "repair_search": None,
    }
    try:
        teacher = parse_teacher_output_v2(teacher_json_path.read_text(encoding="utf-8"))
        if not any(item["id"] == object_id for item in teacher["objects"]):
            raise ValueError(f"object_id {object_id!r} is absent from Safety IR v2")
        semantic_result = json.loads(semantic_result_path.read_text(encoding="utf-8"))
        task_semantic_result = (
            json.loads(task_object_semantic_result_path.read_text(encoding="utf-8"))
            if task_object_semantic_result_path is not None
            else None
        )
        report["bridge_geometry_inventory"] = bridge_request({"op": "geometry_inventory"})
        report["bridge_gripper_geometry_inventory"] = bridge_request({"op": "gripper_geometry_inventory"})
        report["bridge_repair_context"] = bridge_request({"op": "repair_context"})
        if report["bridge_repair_context"].get("status") != "available":
            raise RuntimeError("Safety Bridge did not provide active controller repair context")
        snapshot = bridge_request({"op": "snapshot"})
        report["snapshot_id"] = snapshot.get("snapshot_id")
        if not isinstance(report["snapshot_id"], str):
            raise RuntimeError(snapshot.get("error", "Safety Bridge Snapshot returned no ID"))
        if task_object_id is not None:
            report["task_object_binding"] = _resolve_task_object_binding(
                teacher,
                task_object_id,
                task_semantic_result,
                report["bridge_geometry_inventory"],
            )
        else:
            report["task_object_binding"] = None

        baseline_candidate_id = "repair-A0"
        baseline_branch = bridge_request({
            "op": "branch_rollout",
            "snapshot_id": report["snapshot_id"],
            "candidates": [{"candidate_id": baseline_candidate_id, "actions": actions.tolist()}],
        })
        report["baseline_branch_result"] = baseline_branch
        report["main_state_restored"] = baseline_branch.get("main_state_restored") is True
        baseline_compiled = compile_forbid_contact(
            teacher,
            object_id,
            semantic_result,
            baseline_branch,
            report["bridge_geometry_inventory"],
            report["bridge_gripper_geometry_inventory"],
        )
        report["baseline_compiled_result"] = baseline_compiled
        if not report["main_state_restored"]:
            report["status"] = "aborted_main_state_restore_failed"
            report["failure_reason"] = "baseline_branch_did_not_restore_the_preprobe_main_state"
        else:
            def evaluate_candidates(specifications):
                # One branch request per adaptive round; the Bridge restores the
                # same supplied Snapshot before each candidate and restores its
                # call-entry state in a finally block.
                report["main_state_restored"] = False
                branch = bridge_request({
                    "op": "branch_rollout",
                    "snapshot_id": report["snapshot_id"],
                    "candidates": [
                        {"candidate_id": item["candidate_id"], "actions": item["actions"]}
                        for item in specifications
                    ],
                })
                report["main_state_restored"] = branch.get("main_state_restored") is True
                compiled = compile_forbid_contact(
                    teacher,
                    object_id,
                    semantic_result,
                    branch,
                    report["bridge_geometry_inventory"],
                    report["bridge_gripper_geometry_inventory"],
                )
                branch_by_id = {
                    item.get("candidate_id"): item
                    for item in branch.get("candidates", [])
                    if isinstance(item, dict)
                }
                compiled_by_id = {
                    item.get("candidate_id"): item
                    for item in compiled.get("candidates", [])
                    if isinstance(item, dict)
                }
                results = []
                for item in specifications:
                    candidate_id = item["candidate_id"]
                    candidate_branch = branch_by_id.get(candidate_id)
                    candidate_compiled = compiled_by_id.get(candidate_id)
                    results.append({
                        "candidate_id": candidate_id,
                        "branch_result": {
                            **branch,
                            "candidates": [candidate_branch] if candidate_branch is not None else [],
                        },
                        "compiled_result": {
                            **compiled,
                            "candidates": [candidate_compiled] if candidate_compiled is not None else [],
                        },
                    })
                return results

            repair = run_verifier_guided_repair(
                actions,
                baseline_branch,
                baseline_compiled,
                report["bridge_repair_context"],
                evaluate_candidates,
                task_object_binding=report["task_object_binding"],
            )
            report["repair_search"] = repair
            report["status"] = repair.get("status", "unknown")
            report["repaired_action_chunk_A_plus"] = repair.get("repaired_action_chunk_A_plus")
            report["failure_reason"] = repair.get("failure_reason")
            report["main_state_restored"] = report["main_state_restored"] and repair.get("status") != "aborted_main_state_restore_failed"
    except Exception as exc:
        report["status"] = "aborted_unknown"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "verifier_guided_repair.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "VERIFIER-GUIDED REPAIR:", report["status"],
        "MAIN STATE RESTORED:", report["main_state_restored"],
        "HAS A+:", report["repaired_action_chunk_A_plus"] is not None,
        flush=True,
    )
    print("REPAIR RESULT:", report_path, flush=True)
    return report


def _compact_repair_report(repair_report):
    """Keep search decisions and actions while dropping duplicated branch payloads."""
    compact = dict(repair_report)

    def compact_candidate(candidate):
        if not isinstance(candidate, dict):
            return candidate
        return {
            key: value
            for key, value in candidate.items()
            if key not in ("branch_result", "compiled_result")
        }

    if isinstance(compact.get("search_attempts"), list):
        compact["search_attempts"] = [
            compact_candidate(item) for item in compact["search_attempts"]
        ]
    if "selected_candidate" in compact:
        compact["selected_candidate"] = compact_candidate(compact["selected_candidate"])
    return compact


def prepare_on_policy_safety_collection(
    conn,
    teacher_json_path,
    object_id,
    semantic_result_path=None,
    task_object_id=None,
    task_object_semantic_result_path=None,
    semantic_results_path=None,
):
    """Load fixed-task Teacher bindings and verify every referenced body/geom."""
    import json
    import sys

    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from research.gs_opd.smoke.contact_safety_compiler_smoke import compile_forbid_contact
    from research.gs_opd.smoke.replay_archive_contact_smoke import (
 …5009 tokens truncated…rt["training_action_chunk"] = None
            report["training_label_valid"] = False
            report["label_source"] = None
            add_failure("repair_branch_did_not_restore_the_decision_state")
        if report["status"] == "aborted_main_state_restore_failed":
            report["must_abort_episode"] = True
        return report
    except (ConnectionError, OSError, EOFError) as exc:
        report.update({
            "status": "aborted_main_state_restore_failed",
            "main_state_restored": False,
            "must_abort_episode": True,
        })
        add_failure(f"Safety_Bridge_request_failed:{type(exc).__name__}:{exc}")
        return report
    except Exception as exc:
        report.update({"status": "a0_unknown", "repair_status": "skipped_a0_unknown"})
        # A parse/compiler-side failure does not itself mutate MuJoCo, but if a
        # branch had started and its restore was unconfirmed, fail closed.
        if report["main_state_restored"] is None:
            report["main_state_restored"] = False
            report["must_abort_episode"] = True
        add_failure(f"{type(exc).__name__}:{exc}")
        return report


def select_verified_execution_chunk(decision, predicted_actions):
    """Return the only full Chunk that this decision has proved executable."""
    result = {
        "valid": False,
        "actions": None,
        "source": None,
        "candidate_id": None,
        "reason": None,
    }

    def reject(reason):
        result["reason"] = str(reason)
        return result

    try:
        predicted = np.asarray(predicted_actions, dtype=np.float32)
    except (TypeError, ValueError):
        return reject("student_A0_is_not_numeric")
    if predicted.ndim != 2 or predicted.shape[1:] != (7,) or not len(predicted):
        return reject("student_A0_must_have_shape_(T,7)")
    if not np.isfinite(predicted).all():
        return reject("student_A0_contains_non_finite_values")
    if not isinstance(decision, dict):
        return reject("safety_decision_is_not_an_object")
    if decision.get("main_state_restored") is not True:
        return reject("snapshot_restore_was_not_confirmed")

    branch_summary = decision.get("a0_branch_summary")
    if (
        not isinstance(branch_summary, dict)
        or branch_summary.get("full_action_chunk_executed") is not True
    ):
        return reject("A0_did_not_receive_full_chunk_validation")
    try:
        recorded_a0 = np.asarray(decision.get("original_action_chunk_A0"), dtype=np.float32)
    except (TypeError, ValueError):
        return reject("safety_record_A0_is_not_numeric")
    if recorded_a0.shape != predicted.shape or not np.isfinite(recorded_a0).all():
        return reject("safety_record_A0_shape_or_values_do_not_match_student_prediction")
    if not np.array_equal(recorded_a0, predicted):
        return reject("safety_record_A0_does_not_match_student_prediction")
    verdict = decision.get("a0_verdict")
    if verdict not in ("safe", "violation"):
        reasons = decision.get("failure_reasons")
        reason = (
            reasons[0]
            if isinstance(reasons, list) and reasons
            else "A0_verdict_is_not_safe_or_violation"
        )
        return reject(reason)
    if verdict == "violation" and (
        decision.get("status") != "violation_repaired"
        or decision.get("repair_attempted") is not True
        or decision.get("repair_status") != "repaired_candidate_found"
    ):
        return reject(
            decision.get("repair_failure_reason")
            or decision.get("repair_status")
            or "A0_violation_has_no_validated_repair"
        )
    if decision.get("training_label_valid") is not True:
        return reject("decision_has_no_validated_training_action")

    if verdict == "safe":
        if decision.get("status") != "a0_safe":
            return reject("safe_A0_decision_status_is_inconsistent")
        chosen = predicted
        try:
            labeled_a0 = np.asarray(decision.get("training_action_chunk"), dtype=np.float32)
        except (TypeError, ValueError):
            return reject("validated_safe_A0_label_is_not_numeric")
        if (
            labeled_a0.shape != predicted.shape
            or not np.isfinite(labeled_a0).all()
            or not np.array_equal(labeled_a0, predicted)
        ):
            return reject("validated_safe_A0_label_does_not_match_student_prediction")
        source = "compiler_verified_safe_A0"
    elif verdict == "violation":
        search = decision.get("repair_search")
        selected = search.get("selected_candidate") if isinstance(search, dict) else None
        candidate_id = (
            search.get("selected_candidate_id") if isinstance(search, dict) else None
        )
        supervision = search.get("supervision") if isinstance(search, dict) else None
        coverage = selected.get("execution_coverage") if isinstance(selected, dict) else None
        constraints = selected.get("action_constraints") if isinstance(selected, dict) else None
        progress = selected.get("task_progress") if isinstance(selected, dict) else None
        if (
            not isinstance(selected, dict)
            or selected.get("verdict") != "accepted"
            or selected.get("safety_verdict") != "safe"
            or not isinstance(candidate_id, str)
            or selected.get("candidate_id") != candidate_id
            or not isinstance(supervision, dict)
            or supervision.get("training_label_valid") is not True
            or supervision.get("valid_action_steps") != len(predicted)
            or not isinstance(coverage, dict)
            or coverage.get("training_label_valid") is not True
            or coverage.get("validated_action_steps") != len(predicted)
            or not isinstance(constraints, dict)
            or constraints.get("status") != "pass"
            or not isinstance(progress, dict)
            or progress.get("status") != "pass"
        ):
            return reject("selected_A_plus_lacks_full_safety_progress_or_constraint_validation")
        try:
            plus = np.asarray(decision.get("repaired_action_chunk_A_plus"), dtype=np.float32)
            labeled = np.asarray(decision.get("training_action_chunk"), dtype=np.float32)
            selected_actions = np.asarray(selected.get("actions"), dtype=np.float32)
        except (TypeError, ValueError):
            return reject("validated_A_plus_is_not_numeric")
        if any(
            values.shape != predicted.shape or not np.isfinite(values).all()
            for values in (plus, labeled, selected_actions)
        ):
            return reject("validated_A_plus_shape_or_values_are_invalid")
        if not (
            np.array_equal(plus, labeled)
            and np.array_equal(plus, selected_actions)
        ):
            return reject("A_plus_does_not_match_the_selected_validated_candidate")
        chosen = plus
        source = "compiler_verified_repaired_A_plus"
        result["candidate_id"] = candidate_id
    else:
        return reject("A0_verdict_is_not_safe_or_violation")

    if chosen.shape != predicted.shape or not np.isfinite(chosen).all():
        return reject("selected_action_chunk_shape_or_values_are_invalid")
    result.update({
        "valid": True,
        "actions": chosen.copy(),
        "source": source,
        "reason": None,
    })
    return result


def _new_safety_episode_summary():
    return {
        "a0_verdict_counts": {"safe": 0, "violation": 0, "unknown": 0},
        "repair_attempts": 0,
        "repair_status_counts": {},
        "valid_a_plus_count": 0,
        "valid_training_label_count": 0,
        "failure_reason_counts": {},
        "candidate_rejection_reason_counts": {},
    }


def _must_abort_safety_episode(decision):
    return (
        decision.get("must_abort_episode") is True
        or decision.get("main_state_restored") is not True
        or decision.get("execution_selection_valid") is False
    )


def _update_safety_episode_summary(summary, decision):
    verdict = decision.get("a0_verdict")
    if verdict not in summary["a0_verdict_counts"]:
        verdict = "unknown"
    summary["a0_verdict_counts"][verdict] += 1
    if decision.get("repair_attempted") is True:
        summary["repair_attempts"] += 1
        status = str(decision.get("repair_status") or "unknown")
        summary["repair_status_counts"][status] = summary["repair_status_counts"].get(status, 0) + 1
    if decision.get("repaired_action_chunk_A_plus") is not None and decision.get("training_label_valid") is True:
        summary["valid_a_plus_count"] += 1
    if decision.get("training_label_valid") is True:
        summary["valid_training_label_count"] += 1
    for reason in decision.get("failure_reasons", []):
        summary["failure_reason_counts"][str(reason)] = summary["failure_reason_counts"].get(str(reason), 0) + 1
    repair_search = decision.get("repair_search")
    attempts = repair_search.get("search_attempts", []) if isinstance(repair_search, dict) else []
    for item in attempts:
        if not isinstance(item, dict) or item.get("verdict") == "accepted":
            continue
        reason = item.get("reason")
        if reason is not None:
            reason = str(reason)
            summary["candidate_rejection_reason_counts"][reason] = (
                summary["candidate_rejection_reason_counts"].get(reason, 0) + 1
            )


def get_video_frame(data):
    image = data["images"]["agentview_image"]
    frame = np.frombuffer(image["data"], dtype=image["dtype"])
    frame = frame.reshape(image["shape"]).copy()
    return np.ascontiguousarray(get_libero_image({"agentview_image": frame}))


args = parse_args()

ckpt = Path("/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft")
assert (ckpt / "model.safetensors").is_file(), f"Checkpoint missing: {ckpt}"

with initialize_config_dir(config_dir="/workspace/zy_workspace/RLinf/evaluations/libero", version_base=None):
    cfg = compose(config_name="libero_spatial_openpi_pi05_smoke")

cfg.rollout.model.model_path = str(ckpt)
OmegaConf.update(cfg.rollout.model, "openpi.torch_compile", False, force_add=True)

print("MODEL PATH:", ckpt, flush=True)
print("NORM STATS EXISTS:", (ckpt / "norm_stats.json").is_file(), flush=True)
print("LOADING PI0.5...", flush=True)

model = get_model(cfg.rollout.model)
model = model.to("cuda:0").eval()
print("MODEL LOADED", flush=True)

import time
import json

conn = Client(("127.0.0.1", 38567), authkey=b"gs-opd-local")
steps = 0
chunks = 0
success = False
done = False
start = time.monotonic()

try:
    conn.send({"op": "reset"})
    data = conn.recv()
    trace_writer = None
    collection_enabled = args.on_policy_safety_collection
    trace_output_path = args.on_policy_safety_out if collection_enabled else args.trace_out
    if trace_output_path is not None:
        import sys

        repo_root = str(Path(__file__).resolve().parents[3])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from research.gs_opd.data.trajectory_writer import TrajectoryWriter

        trace_metadata = {
            "suite": "obstacle_avoidance",
            "level": 0,
            "task_id": 0,
            "instruction": data["instruction"],
            "checkpoint_path": str(ckpt),
            "checkpoint_file": str(ckpt / "model.safetensors"),
            "max_steps": args.max_steps,
            "camera_resolution": 256,
            "env_seed": 0,
            "initial_state_index": 0,
        }
        if collection_enabled:
            trace_metadata.update({
                "collection_mode": "continuous_on_policy_safety",
                "teacher_json_path": str(args.teacher_json),
                "safety_object_id": args.object_id,
                "semantic_result_path": (
                    str(args.semantic_result) if args.semantic_result is not None else None
                ),
                "semantic_results_json_path": (
                    str(args.semantic_results_json)
                    if args.semantic_results_json is not None
                    else None
                ),
                "task_object_id": args.task_object_id,
                "task_object_semantic_result_path": (
                    str(args.task_object_semantic_result)
                    if args.task_object_semantic_result is not None
                    else None
                ),
                "task_binding_policy": "per-run Safety IR binding validated against the active fixed-task MuJoCo model",
            })
        trace_writer = TrajectoryWriter(
            trace_output_path,
            trace_metadata,
        )
    safety_episode_summary = _new_safety_episode_summary() if collection_enabled else None
    collection_context = None
    collection_preflight_failure = None
    collection_abort_reason = None
    if collection_enabled:
        try:
            collection_context = prepare_on_policy_safety_collection(
                conn,
                args.teacher_json,
                args.object_id,
                args.semantic_result,
                task_object_id=args.task_object_id,
                task_object_semantic_result_path=args.task_object_semantic_result,
                semantic_results_path=args.semantic_results_json,
            )
            if trace_writer is not None:
                trace_writer.metadata["task_object_id"] = collection_context.get("task_object_id")
                trace_writer.metadata["semantic_binding_validation"] = collection_context.get(
                    "semantic_binding_validation"
                )
        except Exception as exc:
            collection_preflight_failure = f"{type(exc).__name__}:{exc}"
            safety_episode_summary["failure_reason_counts"][collection_preflight_failure] = 1
            print("SAFETY COLLECTION PREFLIGHT FAILED:", collection_preflight_failure, flush=True)
    video_frames = [] if args.video_out is not None else None
    if video_frames is not None:
        video_frames.append(get_video_frame(data))
    print("TASK:", data["instruction"], flush=True)
    contact_probe_result = None

    while (
        steps < args.max_steps
        and not done
        and not success
        and collection_preflight_failure is None
        and collection_abort_reason is None
    ):
        obs = {
            k: np.frombuffer(v["data"], dtype=v["dtype"]).reshape(v["shape"]).copy()
            for k, v in data["images"].items()
        }
        obs.update({
            k: np.asarray(v, dtype=np.float32)
            for k, v in data["state"].items()
        })

        state = np.concatenate([
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"].copy()),
            obs["robot0_gripper_qpos"],
        ]).astype(np.float32)

        env_obs = {
            "main_images": torch.from_numpy(np.ascontiguousarray(get_libero_image(obs))).unsqueeze(0),
            "wrist_images": torch.from_numpy(np.ascontiguousarray(get_libero_wrist_image(obs))).unsqueeze(0),
            "states": torch.from_numpy(state).unsqueeze(0),
            "task_descriptions": [data["instruction"]],
        }

        with torch.inference_mode():
            actions, _ = model.predict_action_batch(env_obs, mode="eval")

        chunk = actions[0].detach().float().cpu().numpy()
        if not collection_enabled:
            assert chunk.ndim == 2 and chunk.shape[1] == 7
            assert np.isfinite(chunk).all()
        chunk_index = chunks
        if trace_writer is not None and not collection_enabled:
            trace_writer.add_decision(chunk_index, steps, data, chunk)
        chunks += 1

        if args.contact_safety_probe:
            contact_probe_result = run_first_chunk_contact_probe(
                conn=conn,
                chunk=chunk,
                instruction=data["instruction"],
                teacher_json_path=args.teacher_json,
                object_id=args.object_id,
                semantic_result_path=args.semantic_result,
                out_dir=args.probe_out_dir,
            )
            break

        if args.verifier_guided_repair:
            contact_probe_result = run_first_chunk_verifier_guided_repair(
                conn=conn,
                chunk=chunk,
                instruction=data["instruction"],
                teacher_json_path=args.teacher_json,
                object_id=args.object_id,
                semantic_result_path=args.semantic_result,
                out_dir=args.repair_out_dir,
                task_object_id=args.task_object_id,
                task_object_semantic_result_path=args.task_object_semantic_result,
            )
            break

        if collection_enabled:
            try:
                safety_decision = run_on_policy_safety_collection_decision(
                    conn=conn,
                    chunk=chunk,
                    instruction=data["instruction"],
                    chunk_index=chunk_index,
                    decision_step=steps,
                    collection_context=collection_context,
                )
            except Exception as exc:
                safety_decision = {
                    "chunk_index": chunk_index,
                    "decision_step": steps,
                    "instruction": data["instruction"],
                    "original_action_chunk_A0": chunk.tolist(),
                    "a0_verdict": "unknown",
                    "status": "aborted_main_state_restore_failed",
                    "main_state_restored": False,
                    "must_abort_episode": True,
                    "training_action_chunk": None,
                    "training_label_valid": False,
                    "failure_reasons": [f"collection_decision_exception:{type(exc).__name__}:{exc}"],
                }
            execution_selection = select_verified_execution_chunk(safety_decision, chunk)
            safety_decision["execution_selection_valid"] = execution_selection["valid"]
            safety_decision["selected_action_source"] = execution_selection["source"]
            safety_decision["selected_candidate_id"] = execution_selection["candidate_id"]
            safety_decision["selected_action_chunk"] = (
                execution_selection["actions"].tolist()
                if execution_selection["valid"]
                else None
            )
            safety_decision["execution_selection_failure_reason"] = execution_selection["reason"]
            if not execution_selection["valid"]:
                safety_decision["must_abort_episode"] = True
                safety_decision["status"] = "aborted_before_unverified_action"
                failure_reason = execution_selection["reason"] or "no_verified_action_chunk"
                safety_decision.setdefault("failure_reasons", []).append(failure_reason)
                collection_abort_reason = failure_reason
            else:
                execution_chunk = execution_selection["actions"]
            if chunk.ndim == 2:
                trace_writer.add_decision(
                    chunk_index,
                    steps,
                    data,
                    chunk,
                    safety_record=safety_decision,
                )
            _update_safety_episode_summary(safety_episode_summary, safety_decision)
            if _must_abort_safety_episode(safety_decision):
                collection_abort_reason = (
                    collection_abort_reason
                    or safety_decision.get("status")
                    or "branch_restore_unconfirmed"
                )
                print(
                    "SAFETY COLLECTION ABORTED BEFORE MAIN ACTION:",
                    collection_abort_reason,
                    flush=True,
                )
                break
        else:
            execution_chunk = chunk

        for action_index, action in enumerate(execution_chunk):
            conn.send({"op": "step", "action": action.tolist()})
            data = conn.recv()
            steps += 1
            if trace_writer is not None:
                trace_writer.add_step(
                    step_index=steps - 1,
                    chunk_index=chunk_index,
                    action_index=action_index,
                    action=action,
                    bridge_data=data,
                )
            if video_frames is not None:
                video_frames.append(get_video_frame(data))
            success = success or data["reward"] > 0
            done = bool(data["done"])

            if steps % 20 == 0 or success or done:
                print(f"STEP={steps} REWARD={data['reward']} DONE={done} SUCCESS={success}", flush=True)

            if steps >= args.max_steps or done or success:
                break

    if args.video_out is not None:
        from rlinf.envs.utils import save_rollout_video

        save_rollout_video(
            video_frames,
            str(args.video_out.parent),
            args.video_out.stem,
        )
        print("VIDEO:", args.video_out, flush=True)

    safety_summary_path = None
    if collection_enabled:
        if collection_preflight_failure is not None:
            episode_status = "collection_preflight_failed"
        elif collection_abort_reason is not None:
            episode_status = "main_state_restore_failed"
        elif success:
            episode_status = "success"
        elif done:
            episode_status = "environment_done"
        else:
            episode_status = "max_steps_reached"
        safety_episode_summary.update({
            "scope": collection_context["scope"] if collection_context is not None else {
                "suite": "obstacle_avoidance", "level": 0, "task_id": 0,
                "binding_scope": "preflight failed; no scene-general binding assumed",
            },
            "instruction": data.get("instruction"),
            "episode_status": episode_status,
            "steps": steps,
            "chunks": chunks,
            "decision_count": chunks,
            "preflight_failure": collection_preflight_failure,
            "abort_reason": collection_abort_reason,
            "data_path": str(trace_output_path),
        })
        safety_summary_path = trace_output_path.with_suffix(".summary.json")
        safety_summary_path.parent.mkdir(parents=True, exist_ok=True)
        safety_summary_path.write_text(
            json.dumps(safety_episode_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        trace_writer.metadata["safety_episode_summary"] = safety_episode_summary
        print(
            "SAFETY COLLECTION SUMMARY:",
            json.dumps(safety_episode_summary, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    if trace_writer is not None:
        trace_writer.save(steps=steps, chunks=chunks)
        print("TRACE:" if not collection_enabled else "SAFETY DATA:", trace_output_path, flush=True)
        if safety_summary_path is not None:
            print("SAFETY SUMMARY:", safety_summary_path, flush=True)

    result = {
        "suite": "obstacle_avoidance",
        "level": 0,
        "task_id": 0,
        "steps": steps,
        "chunks": chunks,
        "success_once": success,
        "done": done,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "camera_resolution": 256,
    }
    if contact_probe_result is not None:
        if args.contact_safety_probe:
            result["contact_safety_probe"] = contact_probe_result["verdict"]
        elif args.verifier_guided_repair:
            result["verifier_guided_repair"] = contact_probe_result["status"]
    if collection_enabled:
        result["on_policy_safety_collection"] = safety_episode_summary

    with open("/tmp/gs_pi05_safety_l0t0_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("RESULT:", result, flush=True)
    print("===== PI05 SAFETY ROLLOUT FINISHED =====", flush=True)
finally:
    conn.close()
