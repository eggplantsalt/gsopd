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
        _validate_current_bindings,
    )
    from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2

    def bridge_request(payload):
        conn.send(payload)
        response = conn.recv()
        if not isinstance(response, dict):
            raise RuntimeError("Safety Bridge returned a non-object response")
        return response

    teacher = parse_teacher_output_v2(teacher_json_path.read_text(encoding="utf-8"))
    if not any(item["id"] == object_id for item in teacher["objects"]):
        raise ValueError(f"object_id {object_id!r} is absent from Safety IR v2")
    if semantic_results_path is not None:
        loaded_results = json.loads(semantic_results_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_results, dict) or any(
            not isinstance(key, str) or not isinstance(value, dict)
            for key, value in loaded_results.items()
        ):
            raise ValueError("semantic results JSON must map Teacher object IDs to result objects")
        semantic_results = loaded_results
        if object_id not in semantic_results:
            raise ValueError(f"semantic results mapping has no binding for {object_id!r}")
    else:
        if semantic_result_path is None:
            raise ValueError("a semantic result file or semantic results mapping is required")
        semantic_results = {
            object_id: json.loads(semantic_result_path.read_text(encoding="utf-8"))
        }

    semantic_result = semantic_results[object_id]
    geometry_inventory = bridge_request({"op": "geometry_inventory"})
    gripper_inventory = bridge_request({"op": "gripper_geometry_inventory"})
    repair_context = bridge_request({"op": "repair_context"})

    object_contact_rules = teacher.get("object_contact_rules", [])
    binding_validation = None
    if object_contact_rules:
        binding_validation = _validate_current_bindings(
            teacher,
            semantic_results,
            geometry_inventory,
        )
        if binding_validation.get("valid") is not True:
            errors = binding_validation.get("errors") or [
                "object_semantic_bindings_are_unverified"
            ]
            raise ValueError(
                "Object-to-Object semantic Body/Geom bindings do not match the active "
                f"Geometry Inventory: {errors}"
            )

    # A branch error is expected in this read-only preflight. All other compiler
    # prerequisites must resolve before collecting from the fixed task.
    preflight = compile_forbid_contact(
        teacher,
        object_id,
        semantic_result,
        {"main_state_restored": True, "candidates": []},
        geometry_inventory,
        gripper_inventory,
        object_semantic_results=semantic_results,
    )
    preflight_errors = [
        reason
        for reason in preflight.get("unknown_reasons", [])
        if reason != "branch_rollout_has_no_candidate_results"
    ]
    selected_contact_rules = [
        rule
        for rule in teacher.get("rules", [])
        if isinstance(rule, dict)
        and rule.get("target_id") == object_id
        and rule.get("predicate") == "contact"
    ]
    if not selected_contact_rules and not object_contact_rules:
        preflight_errors.append("teacher_has_no_supported_contact_rules_for_collection")
    if selected_contact_rules:
        # The current robot-part compiler accepts exactly one gripper rule for
        # this target. An explicit allow is valid without the forbid-only
        # compiled_rule field; all other shapes remain fail-closed.
        if len(selected_contact_rules) != 1:
            preflight_errors.append("multiple_or_conflicting_gripper_contact_rules")
        else:
            selected_rule = selected_contact_rules[0]
            supported_gripper_rule = (
                selected_rule.get("subject_parts") == ["gripper"]
                and selected_rule.get("phase") == "any"
                and selected_rule.get("decision") in ("allow", "forbid")
            )
            if not supported_gripper_rule:
                preflight_errors.append("unsupported_or_unknown_gripper_contact_rule")
            elif (
                selected_rule["decision"] == "forbid"
                and preflight.get("compiled_rule") is None
            ):
                preflight_errors.append("gripper_contact_rule_did_not_compile")
    if (
        preflight.get("target_binding") is None
        or not preflight.get("gripper_geom_ids")
        or preflight_errors
    ):
        raise ValueError(
            "Safety IR rule or semantic physical binding is not valid for the active "
            f"LIBERO-Safety task: {preflight_errors or preflight.get('unknown_reasons')}"
        )

    task_object_binding = None
    if semantic_results_path is not None and task_object_id is None:
        inferred_targets = [
            item["id"] for item in teacher["objects"]
            if item.get("role") == "target_object"
        ]
        if object_contact_rules:
            if len(inferred_targets) != 1:
                raise ValueError(
                    "Object-to-Object collection requires exactly one target_object role "
                    "or an explicit --task-object-id"
                )
            task_object_id = inferred_targets[0]
        elif len(inferred_targets) == 1 and inferred_targets[0] in semantic_results:
            task_object_id = inferred_targets[0]

    if task_object_id is not None:
        if semantic_results_path is not None:
            task_semantic_result = semantic_results.get(task_object_id)
            if not isinstance(task_semantic_result, dict):
                raise ValueError(
                    "semantic results mapping has no task-progress binding for "
                    f"{task_object_id!r}"
                )
        else:
            if task_object_semantic_result_path is None:
                raise ValueError("task-object ID requires a semantic result binding")
            task_semantic_result = json.loads(
                task_object_semantic_result_path.read_text(encoding="utf-8")
            )
            semantic_results[task_object_id] = task_semantic_result
        task_object_binding = _resolve_task_object_binding(
            teacher,
            task_object_id,
            task_semantic_result,
            geometry_inventory,
        )
    return {
        "scope": {
            "suite": "obstacle_avoidance",
            "level": 0,
            "task_id": 0,
            "bridge_source": "research/gs_opd/smoke/safety_bridge.py",
            "binding_scope": "validated against the active fixed-task MuJoCo geometry inventory",
        },
        "teacher": teacher,
        "object_id": object_id,
        "semantic_result": semantic_result,
        "semantic_results": semantic_results,
        "semantic_binding_validation": binding_validation,
        "object_contact_rules": object_contact_rules,
        "task_object_id": task_object_id,
        "safety_target_binding": preflight["target_binding"],
        "geometry_inventory": geometry_inventory,
        "gripper_inventory": gripper_inventory,
        "repair_context": repair_context,
        "task_object_binding": task_object_binding,
        "preflight_result": preflight,
    }


def _near_far_result_is_complete(
    result, snapshot, actions, target_geom_ids, obstacle_geom_ids, decision_step
):
    """Fail closed unless the forecast covers this Snapshot/A0 and restored it."""
    if not isinstance(result, dict) or not isinstance(snapshot, dict):
        return False
    if result.get("status") != "complete" or result.get("risk_verdict") not in ("warning", "no_warning"):
        return False
    if result.get("risk_warning") is not (result.get("risk_verdict") == "warning"):
        return False
    if result.get("snapshot_id") != snapshot.get("snapshot_id"):
        return False
    try:
        from research.gs_opd.repair.near_far import action_chunk_fingerprint

        if result.get("action_chunk_sha256") != action_chunk_fingerprint(actions):
            return False
    except Exception:
        return False
    if (
        result.get("simulation_complete") is not True
        or result.get("requested_action_steps") != 5
        or result.get("executed_action_steps") != 5
    ):
        return False
    samples = result.get("distance_samples")
    if not isinstance(samples, list) or len(samples) != 6:
        return False
    for index, item in enumerate(samples):
        distance = item.get("minimum_distance_m") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or item.get("distance_available") is not True
            or item.get("step_index") != decision_step + index
            or isinstance(distance, bool)
            or not isinstance(distance, (int, float, np.integer, np.floating))
            or not np.isfinite(distance)
        ):
            return False
    restore = result.get("snapshot_restore")
    if not isinstance(restore, dict) or any(
        not isinstance(restore.get(key), dict) or restore[key].get("matched") is not True
        for key in ("before", "after")
    ):
        return False
    if not isinstance(target_geom_ids, list) or not isinstance(obstacle_geom_ids, list):
        return False
    try:
        return (
            set(result.get("target_geom_ids", [])) == set(target_geom_ids)
            and set(result.get("obstacle_geom_ids", [])) == set(obstacle_geom_ids)
            and not (set(target_geom_ids) & set(obstacle_geom_ids))
        )
    except TypeError:
        return False


def _validated_early_repair_candidates(result, expected_shape):
    """Recheck Early Repair's executable contract before any candidate can run."""
    from research.gs_opd.smoke.early_repair_s70_smoke import MIN_GAP_IMPROVEMENT_M

    def invalid(reason):
        return {"valid": False, "reason": reason, "candidates": [], "rejected_candidates": []}

    def finite_number(value):
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float, np.integer, np.floating))
            and bool(np.isfinite(value))
        )

    if not isinstance(result, dict) or result.get("executable") is not True:
        return invalid("early_repair_did_not_report_executable")
    if not (
        result.get("verification_complete") is True
        and result.get("candidate_results_complete") is True
        and result.get("candidate_validation_complete") is True
        and result.get("candidate_branch_main_state_restored") is True
    ):
        return invalid("early_repair_candidate_verification_is_incomplete")
    restore = result.get("snapshot_restore")
    if not isinstance(restore, dict) or any(
        not isinstance(restore.get(key), dict) or restore[key].get("matched") is not True
        for key in ("before", "after")
    ):
        return invalid("early_repair_final_snapshot_restore_not_confirmed")
    if result.get("errors"):
        return invalid("early_repair_report_contains_errors")
    qualified = result.get("qualified_candidates")
    candidate_results = result.get("candidate_results")
    if not isinstance(qualified, list) or not qualified:
        return invalid("early_repair_has_no_qualified_candidates")
    if not isinstance(candidate_results, list) or not candidate_results:
        return invalid("early_repair_candidate_results_missing")
    by_id = {}
    for candidate in candidate_results:
        if not isinstance(candidate, dict):
            return invalid("early_repair_candidate_result_malformed")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in by_id:
            return invalid("early_repair_candidate_ids_missing_or_duplicate")
        by_id[candidate_id] = candidate
        if candidate.get("candidate_validation_complete") is not True:
            return invalid("early_repair_contains_incompletely_validated_candidate_results")

    validated = []
    rejected_candidates = []
    baseline_gaps = []
    seen = set()
    for item in qualified:
        if not isinstance(item, dict):
            return invalid("early_repair_qualified_candidate_malformed")
        candidate_id = item.get("candidate_id")
        verification = item.get("verification")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            return invalid("early_repair_qualified_candidate_ids_missing_or_duplicate")
        seen.add(candidate_id)
        candidate_result = by_id.get(candidate_id)
        if not isinstance(verification, dict) or candidate_result is None:
            return invalid("early_repair_qualified_candidate_has_no_matching_result")
        if not (
            verification.get("accepted_for_probe") is True
            and verification.get("candidate_validation_complete") is True
            and verification.get("full_five_step_chunk") is True
            and verification.get("contact_monitoring_complete") is True
            and verification.get("physical_safety_verdict") == "safe"
            and isinstance(verification.get("action_constraints"), dict)
            and verification["action_constraints"].get("status") == "pass"
            and isinstance(verification.get("task_progress"), dict)
            and verification["task_progress"].get("status") == "pass"
            and isinstance(verification.get("endpoint_replay"), dict)
            and verification["endpoint_replay"].get("status") == "complete"
            and verification["endpoint_replay"].get("main_state_restored") is True
        ):
            return invalid("early_repair_candidate_lacks_required_safety_progress_or_restore_checks")
        try:
            actions = np.asarray(item.get("actions"), dtype=np.float32)
            recorded = np.asarray(verification.get("actions"), dtype=np.float32)
            result_actions = np.asarray(candidate_result.get("actions"), dtype=np.float32)
        except (TypeError, ValueError):
            return invalid("early_repair_candidate_actions_are_not_numeric")
        if any(
            value.shape != expected_shape or not np.isfinite(value).all()
            for value in (actions, recorded, result_actions)
        ):
            return invalid("early_repair_candidate_action_shape_or_values_are_invalid")
        if not (
            np.array_equal(actions, recorded)
            and np.array_equal(actions, result_actions)
            and candidate_result.get("accepted_for_probe") is True
        ):
            return invalid("early_repair_candidate_actions_or_acceptance_do_not_match")
        endpoint_gap = verification.get("endpoint_gap_m")
        baseline_gap = verification.get("baseline_a0_endpoint_gap_m")
        reported_improvement = verification.get("endpoint_gap_change_vs_a0_m")
        result_gap = candidate_result.get("endpoint_gap_m")
        result_baseline_gap = candidate_result.get("baseline_a0_endpoint_gap_m")
        result_improvement = candidate_result.get("endpoint_gap_change_vs_a0_m")
        if not all(finite_number(value) for value in (
            endpoint_gap,
            baseline_gap,
            reported_improvement,
            result_gap,
            result_baseline_gap,
            result_improvement,
        )):
            rejected_candidates.append({
                "candidate_id": candidate_id,
                "reason": "endpoint_gap_or_a0_baseline_gap_is_missing_or_non_finite",
            })
            continue
        if not all(
            np.isclose(float(verified_value), float(result_value), rtol=1e-9, atol=1e-12)
            for verified_value, result_value in (
                (endpoint_gap, result_gap),
                (baseline_gap, result_baseline_gap),
                (reported_improvement, result_improvement),
            )
        ):
            rejected_candidates.append({
                "candidate_id": candidate_id,
                "reason": "qualified_candidate_gap_metadata_does_not_match_candidate_result",
            })
            continue
        gap_improvement = float(endpoint_gap) - float(baseline_gap)
        if (
            not np.isclose(float(reported_improvement), gap_improvement, rtol=1e-9, atol=1e-12)
            or gap_improvement <= MIN_GAP_IMPROVEMENT_M
        ):
            rejected_candidates.append({
                "candidate_id": candidate_id,
                "reason": "endpoint_gap_improvement_does_not_match_or_pass_existing_gate",
            })
            continue
        baseline_gaps.append(float(baseline_gap))
        validated.append({
            "candidate_id": candidate_id,
            "actions": actions,
            "endpoint_gap_m": float(endpoint_gap),
            "baseline_a0_endpoint_gap_m": float(baseline_gap),
            "gap_improvement_m": gap_improvement,
        })
    if not validated:
        return {
            "valid": False,
            "reason": "early_repair_has_no_candidates_with_finite_verified_gap_improvement",
            "candidates": [],
            "rejected_candidates": rejected_candidates,
        }
    baseline_gap = baseline_gaps[0]
    if any(not np.isclose(value, baseline_gap, rtol=1e-9, atol=1e-12) for value in baseline_gaps[1:]):
        return invalid("early_repair_candidates_disagree_on_a0_endpoint_gap")
    return {
        "valid": True,
        "reason": None,
        "candidates": validated,
        "rejected_candidates": rejected_candidates,
        "baseline_a0_endpoint_gap_m": baseline_gap,
    }


def run_on_policy_safety_collection_decision(
    conn,
    chunk,
    instruction,
    chunk_index,
    decision_step,
    collection_context,
):
    """Verify A0 from a fresh Snapshot and optionally search for a safe A+."""
    import sys

    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from research.gs_opd.repair.verifier_guided_repair import run_verifier_guided_repair
    from research.gs_opd.smoke.contact_safety_compiler_smoke import compile_forbid_contact

    actions = np.asarray(chunk, dtype=np.float32)
    object_id = collection_context["object_id"]
    teacher = collection_context["teacher"]
    semantic_result = collection_context["semantic_result"]
    semantic_results = collection_context.get("semantic_results") or {object_id: semantic_result}
    geometry_inventory = collection_context["geometry_inventory"]
    gripper_inventory = collection_context["gripper_inventory"]
    report = {
        "chunk_index": int(chunk_index),
        "decision_step": int(decision_step),
        "instruction": str(instruction),
        "teacher_context": teacher,
        "object_id": object_id,
        "rules": [rule for rule in teacher["rules"] if rule["target_id"] == object_id],
        "object_contact_rules": collection_context.get("object_contact_rules", []),
        "safety_target_binding": collection_context["safety_target_binding"],
        "semantic_binding_result": semantic_result,
        "semantic_binding_results": semantic_results,
        "semantic_binding_validation": collection_context.get("semantic_binding_validation"),
        "original_action_chunk_A0": actions.tolist(),
        "snapshot_id": None,
        "a0_verdict": "unknown",
        "a0_compiler_verdict": "unknown",
        "a0_compiled_result": None,
        "a0_branch_summary": None,
        "repair_attempted": False,
        "repair_status": None,
        "repair_failure_reason": None,
        "repair_search": None,
        "near_far_status": None,
        "near_far_risk_verdict": None,
        "near_far_result": None,
        "early_repair_status": None,
        "early_repair_search": None,
        "early_repair_selected_candidate_id": None,
        "selection_policy": "max_endpoint_gap_then_candidate_id_ascending",
        "selected_candidate_id": None,
        "selected_endpoint_gap_m": None,
        "baseline_a0_endpoint_gap_m": None,
        "selected_gap_improvement_m": None,
        "qualified_candidates_min_endpoint_gap_m": None,
        "qualified_candidates_max_endpoint_gap_m": None,
        "qualified_candidate_endpoint_gap_count": 0,
        "early_repair_candidate_gap_exclusions": [],
        "repaired_action_chunk_A_plus": None,
        "training_action_chunk": None,
        "training_label_valid": False,
        "label_source": None,
        "main_state_restored": None,
        "must_abort_episode": False,
        "failure_reasons": [],
    }

    def add_failure(reason):
        if reason is not None and str(reason) not in report["failure_reasons"]:
            report["failure_reasons"].append(str(reason))

    if actions.ndim != 2 or actions.shape[1:] != (7,) or not len(actions):
        report.update({
            "status": "invalid_A0_action_shape",
            "main_state_restored": True,
            "must_abort_episode": True,
        })
        add_failure("student_A0_must_have_shape_(T,7)")
        return report
    if not np.isfinite(actions).all():
        report.update({
            "status": "invalid_A0_action_values",
            "main_state_restored": True,
            "must_abort_episode": True,
        })
        add_failure("student_A0_contains_non_finite_values")
        return report

    def bridge_request(payload):
        try:
            conn.send(payload)
            response = conn.recv()
        except (OSError, EOFError) as exc:
            raise ConnectionError(
                f"Safety Bridge connection failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(response, dict):
            raise RuntimeError("Safety Bridge returned a non-object response")
        return response

    a0_candidate_id = f"on-policy-A0-{int(chunk_index):06d}"
    try:
        snapshot = bridge_request({"op": "snapshot"})
        report["snapshot_id"] = snapshot.get("snapshot_id")
        if not isinstance(report["snapshot_id"], str):
            # Snapshot creation is read-only; continue Student A0 as an
            # explicitly unknown sample because no branch changed the state.
            report["main_state_restored"] = True
            report["status"] = "unknown_snapshot_unavailable"
            add_failure(snapshot.get("error", "snapshot_operation_returned_no_id"))
            add_failure("original_action_chunk_not_branch_verified")
            return report

        baseline_branch = bridge_request({
            "op": "branch_rollout",
            "snapshot_id": report["snapshot_id"],
            "candidates": [{"candidate_id": a0_candidate_id, "actions": actions.tolist()}],
        })
        report["main_state_restored"] = baseline_branch.get("main_state_restored") is True
        baseline_candidate = next(
            (
                item for item in baseline_branch.get("candidates", [])
                if isinstance(item, dict) and item.get("candidate_id") == a0_candidate_id
            ),
            None,
        )
        if baseline_candidate is not None:
            branch_steps = baseline_candidate.get("steps")
            branch_step_count = len(branch_steps) if isinstance(branch_steps, list) else 0
            requested_steps = baseline_candidate.get("requested_steps")
            executed_steps = baseline_candidate.get("executed_steps")
            report["a0_branch_summary"] = {
                "status": baseline_candidate.get("status"),
                "requested_steps": requested_steps,
                "executed_steps": executed_steps,
                "step_record_count": branch_step_count,
                "full_action_chunk_executed": (
                    isinstance(requested_steps, int)
                    and not isinstance(requested_steps, bool)
                    and isinstance(executed_steps, int)
                    and not isinstance(executed_steps, bool)
                    and requested_steps == len(actions)
                    and executed_steps == len(actions)
                    and branch_step_count == len(actions)
                ),
                "step_rewards": [step.get("reward") for step in branch_steps or []],
                "step_dones": [step.get("done") for step in branch_steps or []],
                "contact_monitoring": baseline_candidate.get("contact_monitoring"),
            }
        compiled = compile_forbid_contact(
            teacher,
            object_id,
            semantic_result,
            baseline_branch,
            geometry_inventory,
            gripper_inventory,
            object_semantic_results=semantic_results,
        )
        report["a0_compiled_result"] = compiled
        report["a0_compiler_verdict"] = compiled.get("status", "unknown")
        report["a0_verdict"] = report["a0_compiler_verdict"]

        if report["main_state_restored"] is not True:
            report["status"] = "aborted_main_state_restore_failed"
            report["must_abort_episode"] = True
            add_failure("baseline_branch_did_not_restore_the_decision_state")
            return report

        a0_full_chunk = (
            isinstance(report["a0_branch_summary"], dict)
            and report["a0_branch_summary"].get("full_action_chunk_executed") is True
        )
        if report["a0_verdict"] == "safe" and not a0_full_chunk:
            report["a0_verdict"] = "unknown"
            report.update({"status": "a0_unknown", "repair_status": "skipped_a0_unknown"})
            add_failure("A0_safe_verdict_did_not_cover_the_requested_full_chunk")
            return report

        if report["a0_verdict"] == "safe":
            if not isinstance(report["a0_branch_summary"].get("contact_monitoring"), dict) or report["a0_branch_summary"]["contact_monitoring"].get("status") != "complete":
                report.update({
                    "status": "a0_contact_monitoring_unknown",
                    "a0_verdict": "unknown",
                    "repair_status": "skipped_a0_unknown",
                    "must_abort_episode": True,
                })
                add_failure("A0_contact_monitoring_was_not_complete")
                return report

            # Near-Far operates only on the unique object-to-object forbid pair
            # already present in the validated Safety IR and bound to this
            # active Geometry Inventory. Never infer a pair from object names.
            near_far_result = None
            pair_subject_id = None
            pair_obstacle_id = None
            forecast_started = False
            try:
                if not isinstance(snapshot.get("snapshot_state"), dict):
                    raise ValueError("snapshot_state_missing_for_near_far_restore_verification")
                binding_validation = collection_context.get("semantic_binding_validation")
                if not isinstance(binding_validation, dict) or binding_validation.get("valid") is not True:
                    raise ValueError("semantic_geometry_binding_validation_missing_or_invalid")
                from research.gs_opd.smoke.near_far_gap_smoke import (
                    _resolve_bound_geom_sets,
                    _select_forbid_pair,
                )
                from research.gs_opd.repair.near_far import (
                    forecast_near_far_risk,
                )

                object_rules = [
                    rule for rule in teacher.get("object_contact_rules", [])
                    if isinstance(rule, dict)
                    and rule.get("decision") == "forbid"
                    and rule.get("predicate") == "contact"
                    and rule.get("phase") == "any"
                ]
                pairs = {
                    (rule.get("subject_id"), rule.get("target_id"))
                    for rule in object_rules
                    if isinstance(rule.get("subject_id"), str)
                    and isinstance(rule.get("target_id"), str)
                }
                if len(pairs) != 1:
                    raise ValueError("near_far_requires_one_unambiguous_forbid_object_contact_pair")
                pair_subject_id, pair_obstacle_id = next(iter(pairs))
                _select_forbid_pair(teacher, pair_subject_id, pair_obstacle_id)
                task_binding = collection_context.get("task_object_binding")
                subject_binding = binding_validation.get("objects", {}).get(pair_subject_id)
                if (
                    not isinstance(task_binding, dict)
                    or task_binding.get("object_id") != pair_subject_id
                    or not isinstance(subject_binding, dict)
                    or task_binding.get("body_id") != subject_binding.get("body_id")
                ):
                    raise ValueError("near_far_subject_does_not_match_the_validated_task_object_binding")
                _, target_geom_ids, obstacle_geom_ids = _resolve_bound_geom_sets(
                    binding_validation,
                    geometry_inventory,
                    pair_subject_id,
                    pair_obstacle_id,
                )
                forecast_started = True
                near_far_result = forecast_near_far_risk(
                    conn,
                    snapshot,
                    actions,
                    target_geom_ids,
                    obstacle_geom_ids,
                    decision_step=int(decision_step),
                )
                report["near_far_geom_bindings"] = {
                    "subject_object_id": pair_subject_id,
                    "obstacle_object_id": pair_obstacle_id,
                    "target_geom_ids": target_geom_ids,
                    "obstacle_geom_ids": obstacle_geom_ids,
                    "source": "Safety IR pair, semantic binding, and active Geometry Inventory",
                }
                report["near_far_result"] = near_far_result
                report["near_far_status"] = near_far_result.get("status")
                report["near_far_risk_verdict"] = near_far_result.get("risk_verdict")
            except Exception as exc:
                if forecast_started:
                    report["main_state_restored"] = False
                near_far_result = {
                    "status": "unknown",
                    "risk_verdict": "unknown",
                    "risk_warning": None,
                    "unknown_reasons": [f"{type(exc).__name__}: {exc}"],
                    "snapshot_restore": {"before": None, "after": None},
                }
                report["near_far_result"] = near_far_result
                report["near_far_status"] = "unknown"
                report["near_far_risk_verdict"] = "unknown"

            restore = near_far_result.get("snapshot_restore") if isinstance(near_far_result, dict) else None
            restore_after = restore.get("after") if isinstance(restore, dict) else None
            if isinstance(restore_after, dict) and restore_after.get("matched") is not True:
                report["main_state_restored"] = False
            if not _near_far_result_is_complete(
                near_far_result,
                snapshot,
                actions,
                report["near_far_geom_bindings"].get("target_geom_ids") if isinstance(report.get("near_far_geom_bindings"), dict) else None,
                report["near_far_geom_bindings"].get("obstacle_geom_ids") if isinstance(report.get("near_far_geom_bindings"), dict) else None,
                int(decision_step),
            ):
                report.update({
                    "status": "near_far_unknown",
                    "a0_verdict": "unknown",
                    "repair_status": "skipped_near_far_unknown",
                    "must_abort_episode": True,
                })
                for reason in near_far_result.get("unknown_reasons", []):
                    add_failure(reason)
                if report["main_state_restored"] is not True:
                    report["status"] = "aborted_main_state_restore_failed"
                    add_failure("near_far_snapshot_restore_failed")
                if not report["failure_reasons"]:
                    add_failure("near_far_forecast_did_not_complete_with_resolved_verdict")
                return report

            if near_far_result.get("risk_verdict") == "no_warning":
                report.update({
                    "status": "a0_safe",
                    "repair_status": "skipped_a0_safe_no_near_far_warning",
                    "training_action_chunk": actions.tolist(),
                    "training_label_valid": True,
                    "label_source": "compiler_and_near_far_verified_safe_A0",
                })
                return report

            report["repair_attempted"] = True
            report["repair_status"] = "early_repair_searching"
            try:
                from research.gs_opd.repair.early_repair import run_early_repair_search
                from research.gs_opd.repair.near_far import action_chunk_fingerprint

                compiled_a0 = next(
                    item for item in compiled.get("candidates", [])
                    if isinstance(item, dict)
                    and item.get("candidate_id") == a0_candidate_id
                    and item.get("verdict") == "safe"
                )
                a0_physical_validation = {
                    "main_state_restored": True,
                    "snapshot_id": report["snapshot_id"],
                    "action_chunk_sha256": action_chunk_fingerprint(actions),
                    "branch_candidate": baseline_candidate,
                    "compiler_candidate": compiled_a0,
                }
                early = run_early_repair_search(
                    conn,
                    snapshot,
                    actions,
                    near_far_result,
                    teacher,
                    semantic_results,
                    collection_context["semantic_binding_validation"],
                    geometry_inventory,
                    gripper_inventory,
                    collection_context["repair_context"],
                    collection_context.get("task_object_binding"),
                    a0_physical_validation,
                    subject_object_id=pair_subject_id,
                    obstacle_object_id=pair_obstacle_id,
                )
            except Exception as exc:
                early = {
                    "status": "not_executable",
                    "executable": False,
                    "verification_complete": False,
                    "candidate_results_complete": False,
                    "candidate_validation_complete": False,
                    "qualified_candidates": [],
                    "candidate_results": [],
                    "snapshot_restore": {"before": None, "after": None},
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            report["early_repair_search"] = early
            report["early_repair_status"] = early.get("status")
            report["main_state_restored"] = (
                isinstance(early.get("snapshot_restore"), dict)
                and isinstance(early["snapshot_restore"].get("after"), dict)
                and early["snapshot_restore"]["after"].get("matched") is True
            )
            validated_candidates = _validated_early_repair_candidates(early, actions.shape)
            report["early_repair_candidate_gap_exclusions"] = validated_candidates.get(
                "rejected_candidates", []
            )
            if not validated_candidates.get("valid"):
                report.update({
                    "status": (
                        "aborted_main_state_restore_failed"
                        if report["main_state_restored"] is not True
                        else "early_repair_unavailable"
                    ),
                    "repair_status": "early_repair_failed",
                    "must_abort_episode": True,
                })
                report["repair_failure_reason"] = validated_candidates.get("reason")
                add_failure(validated_candidates.get("reason"))
                for reason in early.get("errors", []):
                    add_failure(reason)
                return report

            eligible = sorted(
                validated_candidates["candidates"],
                key=lambda item: (-item["endpoint_gap_m"], item["candidate_id"]),
            )
            selected = eligible[0]
            plus = selected["actions"]
            endpoint_gaps = [item["endpoint_gap_m"] for item in eligible]
            selected_id = selected["candidate_id"]
            report.update({
                "status": "a0_safe_early_repaired",
                "repair_status": "early_repair_candidate_selected",
                "early_repair_status": "qualified_candidate_selected",
                "early_repair_selected_candidate_id": selected_id,
                "selected_candidate_id": selected_id,
                "selected_endpoint_gap_m": selected["endpoint_gap_m"],
                "baseline_a0_endpoint_gap_m": selected["baseline_a0_endpoint_gap_m"],
                "selected_gap_improvement_m": selected["gap_improvement_m"],
                "qualified_candidates_min_endpoint_gap_m": min(endpoint_gaps),
                "qualified_candidates_max_endpoint_gap_m": max(endpoint_gaps),
                "qualified_candidate_endpoint_gap_count": len(eligible),
                "repaired_action_chunk_A_plus": plus.tolist(),
                "training_action_chunk": plus.tolist(),
                "training_label_valid": True,
                "label_source": "near_far_verified_geometry_guided_A_plus",
            })
            return report

        if report["a0_verdict"] != "violation":
            report.update({"status": "a0_unknown", "repair_status": "skipped_a0_unknown"})
            for reason in compiled.get("unknown_reasons", []):
                add_failure(reason)
            for item in compiled.get("candidates", []):
                if isinstance(item, dict):
                    add_failure(item.get("reason"))
            if not report["failure_reasons"]:
                add_failure("A0_compiler_verdict_is_unknown")
            return report

        report["repair_attempted"] = True
        last_restore_state = {"restored": True}

        def evaluate_candidates(specifications):
            last_restore_state["restored"] = False
            branch = bridge_request({
                "op": "branch_rollout",
                "snapshot_id": report["snapshot_id"],
                "candidates": [
                    {"candidate_id": item["candidate_id"], "actions": item["actions"]}
                    for item in specifications
                ],
            })
            last_restore_state["restored"] = branch.get("main_state_restored") is True
            compiled_candidates = compile_forbid_contact(
                teacher,
                object_id,
                semantic_result,
                branch,
                geometry_inventory,
                gripper_inventory,
                object_semantic_results=semantic_results,
            )
            branch_by_id = {
                item.get("candidate_id"): item
                for item in branch.get("candidates", [])
                if isinstance(item, dict)
            }
            compiled_by_id = {
                item.get("candidate_id"): item
                for item in compiled_candidates.get("candidates", [])
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
                        **compiled_candidates,
                        "candidates": [candidate_compiled] if candidate_compiled is not None else [],
                    },
                })
            return results

        try:
            repair = run_verifier_guided_repair(
                actions,
                baseline_branch,
                compiled,
                collection_context["repair_context"],
                evaluate_candidates,
                task_object_binding=collection_context.get("task_object_binding"),
            )
        except Exception as exc:
            repair = {
                "status": "repair_exception",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "repaired_action_chunk_A_plus": None,
                "supervision": {"training_label_valid": False},
            }
        report["main_state_restored"] = last_restore_state["restored"]
        report["repair_search"] = _compact_repair_report(repair)
        report["repair_status"] = repair.get("status", "unknown")
        report["repair_failure_reason"] = repair.get("failure_reason")
        report["repaired_action_chunk_A_plus"] = repair.get("repaired_action_chunk_A_plus")
        plus = report["repaired_action_chunk_A_plus"]
        try:
            plus_array = np.asarray(plus, dtype=np.float32)
        except (TypeError, ValueError):
            plus_array = np.empty((0, 0), dtype=np.float32)
        valid_repair_label = (
            report["repair_status"] == "repaired_candidate_found"
            and repair.get("supervision", {}).get("training_label_valid") is True
            and isinstance(plus, list)
            and plus_array.shape == actions.shape
            and np.isfinite(plus_array).all()
        )
        if valid_repair_label:
            report["training_action_chunk"] = plus
            report["training_label_valid"] = True
            report["label_source"] = "compiler_verified_repaired_A_plus"
            report["status"] = "violation_repaired"
        else:
            report["status"] = (
                "aborted_main_state_restore_failed"
                if report["main_state_restored"] is not True
                else "violation_unrepaired"
            )
            add_failure(report["repair_failure_reason"])
            for reason in repair.get("rejected_reasons", []):
                add_failure(reason)
            if not report["failure_reasons"]:
                add_failure(report["repair_status"])
        if report["main_state_restored"] is not True:
            report["status"] = "aborted_main_state_restore_failed"
            report["must_abort_episode"] = True
            report["training_action_chunk"] = None
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


def select_verified_execution_chunk(decision, predicted_actions):
    """Select A0 or A+ only when its exact chunk passed the required gates."""
    try:
        a0 = np.asarray(predicted_actions, dtype=np.float32)
    except (TypeError, ValueError):
        return {"valid": False, "reason": "student_A0_is_not_numeric"}
    if a0.ndim != 2 or a0.shape[1:] != (7,) or not len(a0):
        return {"valid": False, "reason": "student_A0_must_have_shape_(T,7)"}
    if not np.isfinite(a0).all():
        return {"valid": False, "reason": "student_A0_contains_non_finite_values"}
    if not isinstance(decision, dict) or decision.get("main_state_restored") is not True:
        return {"valid": False, "reason": "snapshot_restore_was_not_confirmed"}
    if not (
        isinstance(decision.get("a0_branch_summary"), dict)
        and decision["a0_branch_summary"].get("full_action_chunk_executed") is True
    ):
        return {"valid": False, "reason": "A0_did_not_receive_full_chunk_validation"}
    try:
        recorded_a0 = np.asarray(decision.get("original_action_chunk_A0"), dtype=np.float32)
    except (TypeError, ValueError):
        return {"valid": False, "reason": "safety_record_A0_is_not_numeric"}
    if recorded_a0.shape != a0.shape or not np.isfinite(recorded_a0).all() or not np.array_equal(recorded_a0, a0):
        return {"valid": False, "reason": "safety_record_A0_does_not_match_student_prediction"}

    verdict = decision.get("a0_verdict")
    if verdict == "safe":
        if decision.get("status") == "a0_safe_early_repaired":
            if not (
                decision.get("training_label_valid") is True
                and decision.get("repair_attempted") is True
                and decision.get("repair_status") == "early_repair_candidate_selected"
                and decision.get("early_repair_status") == "qualified_candidate_selected"
                and decision.get("main_state_restored") is True
            ):
                return {"valid": False, "reason": "early_repair_selection_or_restore_is_not_valid"}
            validated = _validated_early_repair_candidates(
                decision.get("early_repair_search"), a0.shape
            )
            if not validated.get("valid"):
                return {"valid": False, "reason": validated.get("reason")}
            candidate_id = decision.get("early_repair_selected_candidate_id")
            selected = next(
                (item for item in validated["candidates"] if item["candidate_id"] == candidate_id),
                None,
            )
            try:
                plus = np.asarray(decision.get("repaired_action_chunk_A_plus"), dtype=np.float32)
                labeled = np.asarray(decision.get("training_action_chunk"), dtype=np.float32)
            except (TypeError, ValueError):
                return {"valid": False, "reason": "early_repair_selected_actions_are_not_numeric"}
            if (
                selected is None
                or plus.shape != a0.shape
                or labeled.shape != a0.shape
                or not np.isfinite(plus).all()
                or not np.isfinite(labeled).all()
                or not np.array_equal(plus, labeled)
                or not np.array_equal(plus, selected["actions"])
            ):
                return {"valid": False, "reason": "early_repair_selected_candidate_actions_do_not_match"}
            return {
                "valid": True,
                "actions": plus.copy(),
                "source": "near_far_verified_geometry_guided_A_plus",
                "candidate_id": candidate_id,
                "reason": None,
            }
        if decision.get("status") != "a0_safe" or decision.get("training_label_valid") is not True:
            return {"valid": False, "reason": "safe_A0_is_not_marked_valid"}
        try:
            labeled = np.asarray(decision.get("training_action_chunk"), dtype=np.float32)
        except (TypeError, ValueError):
            return {"valid": False, "reason": "safe_A0_label_is_not_numeric"}
        if labeled.shape != a0.shape or not np.isfinite(labeled).all() or not np.array_equal(labeled, a0):
            return {"valid": False, "reason": "safe_A0_label_does_not_match_student_prediction"}
        return {
            "valid": True,
            "actions": a0.copy(),
            "source": "compiler_verified_safe_A0",
            "candidate_id": None,
            "reason": None,
        }

    if verdict != "violation":
        reasons = decision.get("failure_reasons")
        reason = reasons[0] if isinstance(reasons, list) and reasons else "A0_verdict_is_not_safe_or_violation"
        return {"valid": False, "reason": reason}
    if not (
        decision.get("status") == "violation_repaired"
        and decision.get("repair_attempted") is True
        and decision.get("repair_status") == "repaired_candidate_found"
        and decision.get("training_label_valid") is True
    ):
        return {
            "valid": False,
            "reason": decision.get("repair_failure_reason") or decision.get("repair_status") or "A0_violation_has_no_validated_repair",
        }
    search = decision.get("repair_search")
    selected = search.get("selected_candidate") if isinstance(search, dict) else None
    candidate_id = search.get("selected_candidate_id") if isinstance(search, dict) else None
    supervision = search.get("supervision") if isinstance(search, dict) else None
    if not (
        isinstance(selected, dict)
        and selected.get("candidate_id") == candidate_id
        and selected.get("verdict") == "accepted"
        and selected.get("safety_verdict") == "safe"
        and isinstance(supervision, dict)
        and supervision.get("training_label_valid") is True
        and supervision.get("valid_action_steps") == len(a0)
        and isinstance(selected.get("execution_coverage"), dict)
        and selected["execution_coverage"].get("training_label_valid") is True
        and selected["execution_coverage"].get("validated_action_steps") == len(a0)
        and isinstance(selected.get("action_constraints"), dict)
        and selected["action_constraints"].get("status") == "pass"
        and isinstance(selected.get("task_progress"), dict)
        and selected["task_progress"].get("status") == "pass"
    ):
        return {"valid": False, "reason": "selected_A_plus_lacks_full_validation"}
    try:
        plus = np.asarray(decision.get("repaired_action_chunk_A_plus"), dtype=np.float32)
        labeled = np.asarray(decision.get("training_action_chunk"), dtype=np.float32)
        candidate_actions = np.asarray(selected.get("actions"), dtype=np.float32)
    except (TypeError, ValueError):
        return {"valid": False, "reason": "validated_A_plus_is_not_numeric"}
    if any(value.shape != a0.shape or not np.isfinite(value).all() for value in (plus, labeled, candidate_actions)):
        return {"valid": False, "reason": "validated_A_plus_shape_or_values_are_invalid"}
    if not (np.array_equal(plus, labeled) and np.array_equal(plus, candidate_actions)):
        return {"valid": False, "reason": "A_plus_does_not_match_selected_validated_candidate"}
    if not isinstance(candidate_id, str) or not candidate_id:
        return {"valid": False, "reason": "selected_A_plus_candidate_id_is_missing"}
    return {
        "valid": True,
        "actions": plus.copy(),
        "source": "compiler_verified_repaired_A_plus",
        "candidate_id": candidate_id,
        "reason": None,
    }


def _classify_safety_abort(decision):
    """Classify a fail-closed stop using its actual safety/restore outcome."""
    if not isinstance(decision, dict) or decision.get("main_state_restored") is not True:
        return "main_state_restore_failed"
    status = decision.get("status")
    if status in ("invalid_A0_action_shape", "invalid_A0_action_values"):
        return "invalid_action"
    if decision.get("near_far_status") == "unknown":
        return "near_far_verification_unknown"
    if decision.get("a0_verdict") == "unknown":
        return "safety_verification_unknown"
    if decision.get("early_repair_status") not in (None, "qualified_candidate_selected"):
        return "early_repair_failed"
    if status in ("violation_unrepaired", "repair_exception"):
        return "repair_failed"
    if decision.get("a0_verdict") == "violation":
        return "repair_validation_failed"
    return "action_validation_failed"


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
    collection_abort_status = None
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
            selection = select_verified_execution_chunk(safety_decision, chunk)
            safety_decision["execution_selection_valid"] = selection["valid"]
            safety_decision["selected_action_source"] = selection.get("source")
            safety_decision["selected_candidate_id"] = selection.get("candidate_id")
            safety_decision["selected_action_chunk"] = (
                selection["actions"].tolist() if selection["valid"] else None
            )
            safety_decision["execution_selection_failure_reason"] = selection.get("reason")
            if not selection["valid"]:
                safety_decision.setdefault("failure_reasons", []).append(selection["reason"])
                collection_abort_reason = selection["reason"] or "no_verified_action_chunk"
                collection_abort_status = _classify_safety_abort(safety_decision)
            else:
                execution_chunk = selection["actions"]
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
                collection_abort_reason = collection_abort_reason or safety_decision.get("status") or "branch_restore_unconfirmed"
                collection_abort_status = collection_abort_status or _classify_safety_abort(safety_decision)
                print(
                    "SAFETY COLLECTION ABORTED BEFORE MAIN ACTION:",
                    collection_abort_status,
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
            episode_status = collection_abort_status or "safety_collection_stopped"
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
