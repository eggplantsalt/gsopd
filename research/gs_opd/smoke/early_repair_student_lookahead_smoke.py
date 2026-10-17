"""Compare S70 A0 and accepted early-repair routes with a fresh pi0.5 lookahead.

The first five actions for every route are checked from one S70 Snapshot. Each
route is then replayed through the live Bridge to obtain its own S75 observation;
pi0.5 predicts a new five-action chunk from that observation, which is checked
from a route-specific S75 Snapshot. This is a bounded diagnostic Smoke, not a
training or repair-label writer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Callable

import numpy as np

from research.gs_opd.smoke import early_repair_s70_smoke as early_repair
from research.gs_opd.smoke.contact_safety_compiler_smoke import (
    AUTHKEY,
    BRIDGE_PORT,
    _complete_monitoring,
    compile_forbid_contact,
)
from research.gs_opd.smoke.cross_chunk_contact_smoke import compare_snapshot_diagnostics
from research.gs_opd.smoke.near_far_gap_smoke import (
    DEFAULT_ARCHIVE,
    DEFAULT_BINDINGS,
    DEFAULT_TEACHER,
    STEP_70,
    STEP_75,
    _read_json,
    _resolve_bound_geom_sets,
    _select_forbid_pair,
    _strict_state_check,
)
from research.gs_opd.smoke.replay_archive_contact_smoke import (
    _decode_bridge_image,
    _validate_current_bindings,
)
from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


DEFAULT_EARLY_REPAIR = Path("logs/gs_opd/early_repair_s70.json")
DEFAULT_OUT = Path("logs/gs_opd/early_repair_student_lookahead.json")
DEFAULT_CHECKPOINT = Path("/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft")
OBJECT_ID = "object_1"
OBSTACLE_ID = "object_4"
STATE_ATOL = 1e-6
MIN_REPAIR_GAP_M = 0.010
MAX_REPAIR_CANDIDATES = 2
CHUNK_LENGTH = 5
PROGRESS_CONFIG = early_repair.PROBE_CONFIG


def _bridge_request(connection: Any, request: dict[str, Any]) -> dict[str, Any]:
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError(f"Safety Bridge returned a non-object for {request.get('op')!r}")
    return response


def select_repair_candidates(
    repair_document: Any,
    *,
    limit: int = MAX_REPAIR_CANDIDATES,
    minimum_gap_m: float = MIN_REPAIR_GAP_M,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select only accepted probe actions with at least the requested gap."""
    if not 0 <= limit <= MAX_REPAIR_CANDIDATES:
        raise ValueError(f"limit must be in [0, {MAX_REPAIR_CANDIDATES}]")
    entries = repair_document.get("candidate_results") if isinstance(repair_document, dict) else None
    if not isinstance(entries, list):
        entries = []
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    eligible_count = 0
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("accepted_for_probe") is not True:
            continue
        gap = entry.get("endpoint_gap_m")
        if (
            isinstance(gap, bool)
            or not isinstance(gap, (int, float))
            or not math.isfinite(float(gap))
            or float(gap) < minimum_gap_m
        ):
            rejected.append({"index": index, "reason": "accepted_candidate_gap_below_10mm_or_unavailable"})
            continue
        candidate_id = entry.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen_ids:
            rejected.append({"index": index, "candidate_id": candidate_id, "reason": "candidate_id_missing_or_duplicate"})
            continue
        try:
            actions = np.asarray(entry.get("actions"), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            rejected.append({"index": index, "candidate_id": candidate_id, "reason": f"actions_invalid:{exc}"})
            continue
        if actions.shape != (CHUNK_LENGTH, 7) or not np.isfinite(actions).all():
            rejected.append({
                "index": index,
                "candidate_id": candidate_id,
                "reason": f"actions_must_be_finite_shape_({CHUNK_LENGTH},7)",
                "shape": list(actions.shape),
            })
            continue
        seen_ids.add(candidate_id)
        eligible_count += 1
        if len(selected) < limit:
            selected.append({
                "candidate_id": candidate_id,
                "source_index": index,
                "source_endpoint_gap_m": float(gap),
                "actions": actions,
            })
    return selected, {
        "minimum_endpoint_gap_m": float(minimum_gap_m),
        "maximum_selected_candidates": limit,
        "accepted_and_valid_candidate_count": eligible_count,
        "selected_candidate_ids": [item["candidate_id"] for item in selected],
        "selected_count": len(selected),
        "no_eligible_repair_candidates": eligible_count == 0,
        "rejections": rejected,
        "selection_policy": "accepted_for_probe is true and saved endpoint_gap_m >= 0.010; source order; no threshold relaxation",
    }


def current_chunk_gate(
    *,
    branch_restored: bool,
    full_chunk: bool,
    monitor_error: str | None,
    compiler_verdict: Any,
    task_progress_status: Any,
    action_bounds_status: Any,
) -> dict[str, Any]:
    checks = {
        "branch_main_state_restored": branch_restored is True,
        "full_five_step_chunk": full_chunk is True,
        "contact_monitoring_complete": monitor_error is None,
        "contact_compiler_safe": compiler_verdict == "safe",
        "task_progress_pass": task_progress_status == "pass",
        "action_bounds_pass": action_bounds_status == "pass",
    }
    return {"passed": all(checks.values()), "checks": checks, "monitor_error": monitor_error}


def _blocked_lookahead(reason: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "s75_endpoint_gap": {"status": "not_queried", "reason": reason},
        "student_next_actions": None,
        "subsequent_five_step_check": {"status": "not_run", "reason": reason},
    }


def validate_student_chunk(actions: Any) -> np.ndarray:
    """Require the configured Student to return exactly one finite five-step chunk."""
    try:
        values = np.asarray(actions, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Student prediction is not a numeric action chunk: {exc}") from exc
    if values.shape != (CHUNK_LENGTH, 7):
        raise ValueError(f"Student next chunk must have shape (5,7), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Student next chunk contains non-finite actions")
    return values


def first_violation_summary(compiled_candidate: Any, segment_start_step: int) -> dict[str, Any]:
    """Extract the earliest compiler evidence without inferring safety."""
    if not isinstance(compiled_candidate, dict):
        return {"status": "unavailable", "reason": "compiler_candidate_missing"}
    evidence_items: list[dict[str, Any]] = []
    object_checks = compiled_candidate.get("object_contact_checks", [])
    checks = object_checks if isinstance(object_checks, list) else []
    for check in checks:
        if not isinstance(check, dict) or check.get("verdict") != "violation":
            continue
        evidence = check.get("evidence")
        if isinstance(evidence, list):
            evidence_items.extend(item for item in evidence if isinstance(item, dict))
    if compiled_candidate.get("verdict") == "violation" and not evidence_items:
        evidence = compiled_candidate.get("evidence")
        if isinstance(evidence, list):
            evidence_items.extend(item for item in evidence if isinstance(item, dict))
    positioned = []
    for event in evidence_items:
        control = event.get("first_control_step", event.get("control_step_index"))
        substep = event.get("first_physics_substep", event.get("physics_substep_index"))
        if isinstance(control, int) and not isinstance(control, bool):
            positioned.append((control, substep if isinstance(substep, int) else -1, event))
    if not positioned:
        return {
            "status": "unavailable" if compiled_candidate.get("verdict") == "violation" else "not_applicable",
            "reason": "compiler_violation_has_no_timed_evidence" if compiled_candidate.get("verdict") == "violation" else None,
        }
    control, substep, event = min(positioned, key=lambda item: (item[0], item[1]))
    return {
        "status": "found",
        "segment_control_step_index": control,
        "absolute_action_index": int(segment_start_step + control),
        "physics_substep_index": None if substep < 0 else substep,
        "evidence": event,
    }


def _restore_and_compare(connection: Any, snapshot_id: str, expected_state: dict[str, Any]) -> dict[str, Any]:
    response = _bridge_request(connection, {"op": "restore", "snapshot_id": snapshot_id})
    if response.get("error"):
        return {"matched": False, "error": response["error"]}
    if response.get("restored_snapshot_id") != snapshot_id:
        return {"matched": False, "error": "Bridge restored a different Snapshot ID"}
    actual = response.get("snapshot_state")
    if not isinstance(expected_state, dict) or not isinstance(actual, dict):
        return {"matched": False, "error": "Restore omitted state diagnostics"}
    return compare_snapshot_diagnostics(expected_state, actual, STATE_ATOL)


def _snapshot(connection: Any, label: str) -> tuple[str, dict[str, Any]]:
    response = _bridge_request(connection, {"op": "snapshot"})
    if response.get("error") or not isinstance(response.get("snapshot_id"), str):
        raise RuntimeError(f"{label} Snapshot failed: {response.get('error', 'snapshot_id missing')}")
    state = response.get("snapshot_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"{label} Snapshot omitted diagnostics")
    return response["snapshot_id"], state


def _distance_summary(response: dict[str, Any]) -> dict[str, Any]:
    return early_repair._distance_response_summary(response)


def _observation_summary(response: dict[str, Any]) -> dict[str, Any]:
    state = response.get("state")
    state_summary = {}
    if isinstance(state, dict):
        for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
            if key in state:
                state_summary[key] = np.asarray(state[key]).tolist()
    image_shapes = {}
    image_errors = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        try:
            image_shapes[key] = list(_decode_bridge_image(response, key).shape)
        except Exception as exc:
            image_errors.append(f"{key}:{type(exc).__name__}:{exc}")
    return {
        "instruction": response.get("instruction"),
        "state": state_summary,
        "image_shapes": image_shapes,
        "images_available": len(image_shapes) == 2,
        "image_errors": image_errors,
    }


def _load_student_model(checkpoint: Path) -> Any:
    """Load pi0.5 with the same RLinf smoke config without importing its runner."""
    checkpoint = checkpoint.expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"pi0.5 checkpoint is missing model.safetensors: {checkpoint}")
    repo_root = Path(__file__).resolve().parents[3]
    os.environ["EMBODIED_PATH"] = str(repo_root / "examples" / "embodiment")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from rlinf.models import get_model

    with initialize_config_dir(config_dir=str(repo_root / "evaluations" / "libero"), version_base=None):
        cfg = compose(config_name="libero_spatial_openpi_pi05_smoke")
    cfg.rollout.model.model_path = str(checkpoint)
    OmegaConf.update(cfg.rollout.model, "openpi.torch_compile", False, force_add=True)
    model = get_model(cfg.rollout.model)
    return model.to("cuda:0").eval()


def _predict_student_actions(model: Any, response: dict[str, Any], instruction: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the exact Pi0.5 LIBERO image/state batch used by pi05_rollout."""
    import torch
    from rlinf.envs.sim.libero.utils import (
        get_libero_image,
        get_libero_wrist_image,
        quat2axisangle,
    )

    if response.get("instruction") != instruction:
        raise ValueError("S75 observation instruction differs from the archived task")
    images = {
        "agentview_image": _decode_bridge_image(response, "agentview_image"),
        "robot0_eye_in_hand_image": _decode_bridge_image(response, "robot0_eye_in_hand_image"),
    }
    raw_state = response.get("state")
    if not isinstance(raw_state, dict):
        raise ValueError("S75 observation has no robot state mapping")
    state_values = {
        key: np.asarray(raw_state[key], dtype=np.float32)
        for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    }
    if any(not np.isfinite(value).all() for value in state_values.values()):
        raise ValueError("S75 robot state contains non-finite values")
    state = np.concatenate([
        state_values["robot0_eef_pos"],
        quat2axisangle(state_values["robot0_eef_quat"].copy()),
        state_values["robot0_gripper_qpos"],
    ]).astype(np.float32)
    obs = {**images, **state_values}
    model_input = {
        "main_images": torch.from_numpy(np.ascontiguousarray(get_libero_image(obs))).unsqueeze(0),
        "wrist_images": torch.from_numpy(np.ascontiguousarray(get_libero_wrist_image(obs))).unsqueeze(0),
        "states": torch.from_numpy(state).unsqueeze(0),
        "task_descriptions": [instruction],
    }
    with torch.inference_mode():
        actions, _ = model.predict_action_batch(model_input, mode="eval")
    if not hasattr(actions, "detach") or actions.ndim != 3 or actions.shape[0] != 1:
        raise ValueError(f"Student prediction must have shape (1,T,7), got {getattr(actions, 'shape', None)}")
    predicted = validate_student_chunk(actions[0].detach().float().cpu().numpy())
    return predicted.astype(np.float32, copy=False), {
        "inference_mode": "eval",
        "main_image_shape_after_preprocess": list(model_input["main_images"].shape),
        "wrist_image_shape_after_preprocess": list(model_input["wrist_images"].shape),
        "robot_state_encoding": "eef position + quat2axisangle(eef quaternion) + gripper qpos",
        "image_preprocessing": "get_libero_image/get_libero_wrist_image (180-degree rotation); contiguous RGB tensors",
    }


def _run_live_chunk(connection: Any, actions: np.ndarray) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    final_response = None
    transitions = []
    for index, action in enumerate(actions):
        try:
            final_response = _bridge_request(connection, {
                "op": "step",
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "include_snapshot_state": index == len(actions) - 1,
            })
        except Exception as exc:
            return None, transitions, f"step_{index}:{type(exc).__name__}:{exc}"
        if final_response.get("error"):
            return None, transitions, f"step_{index}:{final_response['error']}"
        transitions.append({"control_step_index": index, "reward": final_response.get("reward"), "done": final_response.get("done")})
        if final_response.get("done") is True:
            return final_response, transitions, f"environment_terminated_at_control_step_{index}"
    return final_response, transitions, None


def _route_restore(connection: Any, snapshot_id: str, snapshot_state: dict[str, Any]) -> dict[str, Any]:
    try:
        return _restore_and_compare(connection, snapshot_id, snapshot_state)
    except Exception as exc:
        return {"matched": False, "error": f"{type(exc).__name__}: {exc}"}


def run_experiment(
    connection: Any,
    archive: dict[str, Any],
    repair_document: dict[str, Any],
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    model_loader: Callable[[Path], Any] = _load_student_model,
    subject_object_id: str = OBJECT_ID,
    obstacle_object_id: str = OBSTACLE_ID,
) -> dict[str, Any]:
    selected, selection_report = select_repair_candidates(repair_document)
    actions_a0 = np.asarray(archive["actions"], dtype=np.float32)
    report: dict[str, Any] = {
        "mode": "s70_s75_student_closed_loop_lookahead",
        "status": "failed",
        "archive": archive.get("archive"),
        "early_repair_report": repair_document.get("report_path") if isinstance(repair_document, dict) else None,
        "checkpoint_path": str(checkpoint),
        "decision_step": STEP_70,
        "student_lookahead_start_action_index": STEP_75,
        "instruction": archive.get("instruction"),
        "candidate_selection": selection_report,
        "no_eligible_repair_candidates": selection_report["no_eligible_repair_candidates"],
        "original_a0_actions": actions_a0.tolist(),
        "training_label_written": False,
        "routes": [],
        "errors": [],
    }
    s70_snapshot_id: str | None = None
    s70_snapshot_state: dict[str, Any] | None = None
    try:
        _select_forbid_pair(teacher, subject_object_id, obstacle_object_id)
        reset = _bridge_request(connection, {"op": "reset"})
        if reset.get("error"):
            report["failure_reason"] = f"reset_failed:{reset['error']}"
            return report
        if reset.get("instruction") != archive["instruction"]:
            report["failure_reason"] = "initial_task_instruction_differs_from_archive"
            return report

        inventory = _bridge_request(connection, {"op": "geometry_inventory"})
        gripper_inventory = _bridge_request(connection, {"op": "gripper_geometry_inventory"})
        controller_context = _bridge_request(connection, {"op": "repair_context"})
        binding_validation = _validate_current_bindings(teacher, semantic_results, inventory)
        report["geometry_binding_validation"] = binding_validation
        if not binding_validation.get("valid"):
            report["failure_reason"] = "teacher_semantic_bindings_do_not_match_current_geometry_inventory"
            return report
        resolved_objects, subject_geom_ids, obstacle_geom_ids = _resolve_bound_geom_sets(
            binding_validation, inventory, subject_object_id, obstacle_object_id
        )
        report["resolved_objects"] = resolved_objects
        task_binding = dict(binding_validation["objects"][subject_object_id])
        teacher_objects = {item["id"]: item for item in teacher.get("objects", []) if isinstance(item, dict)}
        task_binding.update({"status": "matched_by_iou", "role": teacher_objects[subject_object_id]["role"]})
        report["task_progress_binding"] = task_binding

        last_response = reset
        replay_actions = np.asarray(archive["episode"]["replay_actions"], dtype=np.float32)
        if replay_actions.shape != (STEP_70, 7) or not np.isfinite(replay_actions).all():
            report["failure_reason"] = "archived_s70_prefix_must_be_finite_shape_70_by_7"
            return report
        for index, action in enumerate(replay_actions):
            last_response = _bridge_request(connection, {"op": "step", "action": action.tolist()})
            if last_response.get("error"):
                report["failure_reason"] = f"s70_replay_failed_at_action_{index}:{last_response['error']}"
                return report
            if last_response.get("done") is True:
                report["failure_reason"] = f"environment_terminated_during_s70_prefix_at_action_{index}"
                return report
        s70_check = _strict_state_check(archive["episode"]["expected_observation"], last_response)
        report["s70_archive_state_check"] = s70_check
        if not s70_check.get("matched"):
            report["failure_reason"] = "replayed_s70_instruction_or_robot_state_mismatch_at_1e-6"
            return report
        s70_snapshot_id, s70_snapshot_state = _snapshot(connection, "S70")
        report["s70_snapshot_id"] = s70_snapshot_id
        report["s70_snapshot_restore_check"] = _route_restore(
            connection, s70_snapshot_id, s70_snapshot_state
        )
        if report["s70_snapshot_restore_check"].get("matched") is not True:
            report["failure_reason"] = "s70_snapshot_immediate_restore_check_failed"
            return report

        routes_to_branch = [{
            "route_id": "a0_s70",
            "route_type": "original_a0",
            "source_candidate_id": None,
            "source_gap_m": None,
            "actions": actions_a0,
        }]
        routes_to_branch.extend({
            "route_id": f"repair_{index}",
            "route_type": "accepted_early_repair_probe",
            "source_candidate_id": item["candidate_id"],
            "source_gap_m": item["source_endpoint_gap_m"],
            "actions": item["actions"],
        } for index, item in enumerate(selected))

        route_preflight = {}
        branch_candidates = []
        for route in routes_to_branch:
            bounds = early_repair._osc_input_bounds_check(route["actions"], controller_context)
            route_preflight[route["route_id"]] = bounds
            if bounds.get("status") == "pass":
                branch_candidates.append({"candidate_id": route["route_id"], "actions": route["actions"].tolist()})
        report["s70_route_action_preflight"] = route_preflight
        if "a0_s70" not in {item["candidate_id"] for item in branch_candidates}:
            report["failure_reason"] = "archived_a0_fails_active_osc_input_bounds_preflight"
            return report

        s70_branch = _bridge_request(connection, {
            "op": "branch_rollout",
            "snapshot_id": s70_snapshot_id,
            "candidates": branch_candidates,
        })
        report["s70_branch_main_state_restored"] = s70_branch.get("main_state_restored") is True
        report["s70_branch_error"] = s70_branch.get("error")
        s70_compilation = compile_forbid_contact(
            teacher,
            subject_object_id,
            semantic_results[subject_object_id],
            s70_branch,
            inventory,
            gripper_inventory,
            object_semantic_results=semantic_results,
        )
        report["s70_compiler_overall_verdict"] = s70_compilation.get("verdict")
        report["s70_branch_restore_diagnostic"] = _route_restore(
            connection, s70_snapshot_id, s70_snapshot_state
        )
        if (
            s70_branch.get("main_state_restored") is not True
            or report["s70_branch_restore_diagnostic"].get("matched") is not True
        ):
            report["failure_reason"] = "s70_branch_did_not_restore_snapshot_state"
            return report

        baseline_candidate = early_repair._branch_candidate(s70_branch, "a0_s70")
        route_records: dict[str, dict[str, Any]] = {}
        for route in routes_to_branch:
            route_id = route["route_id"]
            preflight = route_preflight[route_id]
            if preflight.get("status") != "pass":
                record = {
                    "route_id": route_id,
                    "route_type": route["route_type"],
                    "source_candidate_id": route["source_candidate_id"],
                    "source_endpoint_gap_m": route["source_gap_m"],
                    "current_chunk": {"status": "blocked", "reason": "active_osc_input_bounds_preflight_failed", "action_bounds": preflight},
                    "lookahead": _blocked_lookahead("active_osc_input_bounds_preflight_failed"),
                }
                report["routes"].append(record)
                route_records[route_id] = record
                continue
            branch_candidate = early_repair._branch_candidate(s70_branch, route_id)
            compiled = early_repair._compiler_candidate(s70_compilation, route_id)
            _monitor, monitor_error = _complete_monitoring(branch_candidate)
            full_chunk = early_repair._complete_five_steps(branch_candidate)
            progress = early_repair.evaluate_same_horizon_progress(
                baseline_candidate,
                branch_candidate,
                actions_a0,
                controller_context,
                task_binding,
                PROGRESS_CONFIG,
            ) if baseline_candidate is not None and branch_candidate is not None else {
                "status": "unavailable", "reason": "baseline_or_route_branch_candidate_missing"
            }
            compiler_verdict = compiled.get("verdict") if isinstance(compiled, dict) else "unknown"
            gate = current_chunk_gate(
                branch_restored=(
                    s70_branch.get("main_state_restored") is True
                    and report["s70_branch_restore_diagnostic"].get("matched") is True
                ),
                full_chunk=full_chunk,
                monitor_error=monitor_error,
                compiler_verdict=compiler_verdict,
                task_progress_status=progress.get("status"),
                action_bounds_status=preflight.get("status"),
            )
            record = {
                "route_id": route_id,
                "route_type": route["route_type"],
                "source_candidate_id": route["source_candidate_id"],
                "source_endpoint_gap_m": route["source_gap_m"],
                "s70_actions": route["actions"].tolist(),
                "current_chunk": {
                    "status": "pass" if gate["passed"] else "blocked",
                    "gate": gate,
                    "executed_steps": branch_candidate.get("executed_steps") if isinstance(branch_candidate, dict) else None,
                    "contact_monitoring": early_repair._monitor_summary(branch_candidate) if branch_candidate else None,
                    "contact_monitoring_error": monitor_error,
                    "compiler": compiled,
                    "first_violation": first_violation_summary(compiled, STEP_70),
                    "task_progress_vs_a0_same_five_step_horizon": progress,
                    "active_action_bounds": preflight,
                },
            }
            report["routes"].append(record)
            route_records[route_id] = record
            if not gate["passed"]:
                record["lookahead"] = _blocked_lookahead("current_five_step_gates_did_not_pass")
        if not baseline_candidate or not early_repair._complete_five_steps(baseline_candidate):
            report["failure_reason"] = "a0_s70_branch_did_not_complete_all_five_steps"
            return report

        runnable_routes = [
            route for route in routes_to_branch
            if route_records[route["route_id"]].get("current_chunk", {}).get("gate", {}).get("passed") is True
        ]
        if not runnable_routes:
            report["failure_reason"] = "no_route_passed_s70_safety_progress_monitoring_and_action_gates"
            report["status"] = "no_route_passed_current_chunk_gates"
            return report
        model = model_loader(checkpoint)
        report["model_loaded"] = True
        report["model_configuration"] = {
            "checkpoint": str(checkpoint),
            "config": "evaluations/libero/libero_spatial_openpi_pi05_smoke.yaml",
            "model_type": "openpi pi0.5; predict_action_batch(mode='eval')",
            "torch_compile": False,
            "device": "cuda:0",
            "num_action_chunks": CHUNK_LENGTH,
        }

        future_progress_inputs: dict[str, dict[str, Any]] = {}
        for route in runnable_routes:
            if report.get("status") == "main_state_restore_failed":
                break
            record = route_records[route["route_id"]]
            route_result: dict[str, Any] = {
                "actual_s75_observation": None,
                "s75_endpoint_gap": None,
                "student_next_actions": None,
                "student_prediction": None,
                "subsequent_five_step_check": None,
                "errors": [],
            }
            s75_snapshot_id: str | None = None
            s75_snapshot_state: dict[str, Any] | None = None
            try:
                before_route_restore = _route_restore(connection, s70_snapshot_id, s70_snapshot_state)
                route_result["restore_s70_before_live_route"] = before_route_restore
                if before_route_restore.get("matched") is not True:
                    route_result["errors"].append("could_not_restore_s70_before_live_route")
                    continue
                live_response, transition_log, live_error = _run_live_chunk(connection, route["actions"])
                route_result["live_five_step_execution"] = {
                    "requested_steps": CHUNK_LENGTH,
                    "executed_steps": len(transition_log),
                    "transitions": transition_log,
                    "error": live_error,
                }
                if live_error is not None or live_response is None or len(transition_log) != CHUNK_LENGTH:
                    route_result["errors"].append(live_error or "live_s70_route_did_not_complete_five_actions")
                    continue
                route_result["actual_s75_observation"] = _observation_summary(live_response)
                if live_response.get("instruction") != archive["instruction"]:
                    route_result["errors"].append("actual_s75_instruction_differs_from_task")
                    continue
                branch_candidate = early_repair._branch_candidate(s70_branch, route["route_id"])
                branch_state = (
                    branch_candidate.get("steps", [])[-1].get("state")
                    if isinstance(branch_candidate, dict) and isinstance(branch_candidate.get("steps"), list) and branch_candidate.get("steps")
                    else None
                )
                live_state = live_response.get("snapshot_state")
                route_result["branch_vs_live_s75_full_diagnostic"] = compare_snapshot_diagnostics(
                    branch_state, live_state, STATE_ATOL
                )
                route_result["branch_vs_live_s75_full_diagnostic"]["use_as_physical_gate"] = False
                route_result["branch_vs_live_s75_integration_state"] = early_repair._compare_mujoco_integration_state(
                    branch_state, live_state, STATE_ATOL
                )
                if route_result["branch_vs_live_s75_integration_state"].get("matched") is not True:
                    route_result["errors"].append("live_s75_integration_state_differs_from_s70_branch_endpoint")
                    continue
                gap_response = _bridge_request(connection, {
                    "op": "geom_set_min_distance",
                    "geom_ids_a": subject_geom_ids,
                    "geom_ids_b": obstacle_geom_ids,
                    "distmax_m": 10.0,
                })
                route_result["s75_endpoint_gap"] = _distance_summary(gap_response)
                if gap_response.get("error"):
                    route_result["errors"].append(f"s75_gap_query:{gap_response['error']}")
                s75_snapshot_id, s75_snapshot_state = _snapshot(connection, f"{route['route_id']} S75")
                route_result["s75_snapshot_id"] = s75_snapshot_id
                if not _observation_summary(live_response).get("images_available"):
                    route_result["errors"].append("actual_s75_rgb_observation_unavailable")
                    continue

                predicted, prediction_info = _predict_student_actions(model, live_response, archive["instruction"])
                route_result["student_next_actions"] = predicted.tolist()
                future_bounds = early_repair._osc_input_bounds_check(predicted, controller_context)
                route_result["student_prediction"] = {
                    **prediction_info,
                    "shape": list(predicted.shape),
                    "finite": bool(np.isfinite(predicted).all()),
                    "action_bounds": future_bounds,
                    "not_training_data": True,
                }
                if future_bounds.get("status") != "pass":
                    route_result["errors"].append("student_next_actions_fail_active_osc_input_bounds")
                    continue

                future_candidate_id = f"{route['route_id']}_student_s75_chunk"
                future_branch = _bridge_request(connection, {
                    "op": "branch_rollout",
                    "snapshot_id": s75_snapshot_id,
                    "candidates": [{"candidate_id": future_candidate_id, "actions": predicted.tolist()}],
                })
                future_compilation = compile_forbid_contact(
                    teacher,
                    subject_object_id,
                    semantic_results[subject_object_id],
                    future_branch,
                    inventory,
                    gripper_inventory,
                    object_semantic_results=semantic_results,
                )
                future_candidate = early_repair._branch_candidate(future_branch, future_candidate_id)
                future_compiled = early_repair._compiler_candidate(future_compilation, future_candidate_id)
                _future_monitor, future_monitor_error = _complete_monitoring(future_candidate)
                future_full = early_repair._complete_five_steps(future_candidate)
                s75_branch_restore = _route_restore(connection, s75_snapshot_id, s75_snapshot_state)
                future_verdict = future_compiled.get("verdict") if isinstance(future_compiled, dict) else "unknown"
                route_result["subsequent_five_step_check"] = {
                    "branch_main_state_restored": future_branch.get("main_state_restored") is True,
                    "snapshot_restore_diagnostic": s75_branch_restore,
                    "requested_steps": future_candidate.get("requested_steps") if isinstance(future_candidate, dict) else None,
                    "executed_steps": future_candidate.get("executed_steps") if isinstance(future_candidate, dict) else None,
                    "full_five_step_chunk": future_full,
                    "contact_monitoring": early_repair._monitor_summary(future_candidate) if future_candidate else None,
                    "contact_monitoring_error": future_monitor_error,
                    "compiler": future_compiled,
                    "compiler_overall_verdict": future_compilation.get("verdict"),
                    "route_verdict": future_verdict,
                    "first_violation": first_violation_summary(future_compiled, STEP_75),
                    "action_bounds": future_bounds,
                    "safe_lookahead_passed": bool(
                        future_branch.get("main_state_restored") is True
                        and s75_branch_restore.get("matched") is True
                        and future_full
                        and future_monitor_error is None
                        and future_verdict == "safe"
                        and future_bounds.get("status") == "pass"
                    ),
                    "task_progress_vs_a0_student_policy": {"status": "pending_a0_lookahead_result"},
                }
                if future_candidate is not None:
                    future_progress_inputs[route["route_id"]] = {
                        "candidate": future_candidate,
                        "actions": predicted,
                    }
                if future_branch.get("main_state_restored") is not True or s75_branch_restore.get("matched") is not True:
                    route_result["errors"].append("s75_branch_restore_check_failed")
            except Exception as exc:
                route_result["errors"].append(f"{type(exc).__name__}:{exc}")
            finally:
                if s75_snapshot_id is not None and isinstance(s75_snapshot_state, dict):
                    route_result["restore_s75_after_lookahead"] = _route_restore(
                        connection, s75_snapshot_id, s75_snapshot_state
                    )
                s70_after_route = _route_restore(connection, s70_snapshot_id, s70_snapshot_state)
                route_result["restore_s70_after_route"] = s70_after_route
                if s70_after_route.get("matched") is not True:
                    route_result["status"] = "s70_restore_failed"
                elif route_result["errors"]:
                    route_result["status"] = "route_completed_with_errors"
                else:
                    future_check = route_result.get("subsequent_five_step_check")
                    if not isinstance(future_check, dict):
                        route_result["status"] = "student_lookahead_not_completed"
                    elif future_check.get("safe_lookahead_passed") is True:
                        route_result["status"] = "lookahead_safe"
                    elif future_check.get("route_verdict") == "violation":
                        route_result["status"] = "lookahead_violation"
                    else:
                        route_result["status"] = "lookahead_unknown_or_incomplete"
                record["lookahead"] = route_result
                if s70_after_route.get("matched") is not True:
                    report["failure_reason"] = f"s70_restore_failed_after_route_{route['route_id']}"
                    report["status"] = "main_state_restore_failed"
                    break

        baseline_future_input = future_progress_inputs.get("a0_s70")
        if baseline_future_input is not None:
            for record in report["routes"]:
                future_input = future_progress_inputs.get(record.get("route_id"))
                followup = record.get("lookahead", {}).get("subsequent_five_step_check")
                if future_input is None or not isinstance(followup, dict):
                    continue
                progress = early_repair.evaluate_same_horizon_progress(
                    baseline_future_input["candidate"],
                    future_input["candidate"],
                    np.asarray(baseline_future_input["actions"], dtype=np.float64),
                    controller_context,
                    task_binding,
                    PROGRESS_CONFIG,
                )
                followup["task_progress_vs_a0_student_policy"] = progress
            report["future_task_progress_comparison"] = {
                "baseline_route": "a0_s70",
                "horizon_steps": CHUNK_LENGTH,
                "basis": "existing time-aligned task-object and reward progress verifier",
            }
        else:
            report["future_task_progress_comparison"] = {
                "status": "unavailable",
                "reason": "original_a0_route_did_not_produce_a_complete_student_lookahead",
            }
        if report.get("status") != "main_state_restore_failed":
            report["status"] = "completed"
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}:{exc}")
        report["failure_reason"] = f"{type(exc).__name__}:{exc}"
    finally:
        if s70_snapshot_id is not None and isinstance(s70_snapshot_state, dict):
            try:
                final_restore = _route_restore(connection, s70_snapshot_id, s70_snapshot_state)
            except Exception as exc:
                final_restore = {"matched": False, "error": f"{type(exc).__name__}:{exc}"}
            report["final_s70_restore"] = final_restore
            if final_restore.get("matched") is not True:
                report["status"] = "main_state_restore_failed"
                report["failure_reason"] = "final_s70_snapshot_restore_failed"
    return report


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Arrays used internally for progress comparison are removed before serialization.
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--early-repair-json", type=Path, default=DEFAULT_EARLY_REPAIR)
    parser.add_argument("--teacher-json", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--semantic-results-json", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subject-object-id", default=OBJECT_ID)
    parser.add_argument("--obstacle-object-id", default=OBSTACLE_ID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT)
    args = parser.parse_args(argv)
    try:
        archive = early_repair.load_s70_archive(args.archive)
        repair_document = _read_json(args.early_repair_json, "early repair report")
        if not isinstance(repair_document, dict):
            raise ValueError("Early-repair report must be a JSON object")
        repair_document["report_path"] = str(args.early_repair_json.resolve())
        teacher = parse_teacher_output_v2(args.teacher_json.read_text(encoding="utf-8"))
        semantic_results = _read_json(args.semantic_results_json, "semantic bindings")
        if not isinstance(semantic_results, dict):
            raise ValueError("Semantic bindings must map Teacher object IDs to physical bindings")
        connection = Client((args.host, args.port), authkey=AUTHKEY)
        try:
            report = run_experiment(
                connection,
                archive,
                repair_document,
                teacher,
                semantic_results,
                checkpoint=args.checkpoint,
                subject_object_id=args.subject_object_id,
                obstacle_object_id=args.obstacle_object_id,
            )
        finally:
            connection.close()
    except Exception as exc:
        report = {
            "mode": "s70_s75_student_closed_loop_lookahead",
            "status": "input_or_connection_error",
            "failure_reason": f"{type(exc).__name__}:{exc}",
            "training_label_written": False,
        }
    _write_json(args.out, report)
    print(f"S70→S75′ Student lookahead: status={report.get('status')}")
    print(f"  no eligible >=10mm repair candidates={report.get('no_eligible_repair_candidates')}")
    for route in report.get("routes", []):
        lookahead = route.get("lookahead", {})
        followup = lookahead.get("subsequent_five_step_check", {})
        print(
            f"  {route.get('route_id')}: current={route.get('current_chunk', {}).get('status')}; "
            f"S75 gap={lookahead.get('s75_endpoint_gap', {}).get('minimum_distance_m')}; "
            f"student={followup.get('route_verdict', 'not_run')}; "
            f"first_violation={followup.get('first_violation', {}).get('absolute_action_index')}"
        )
    if report.get("failure_reason"):
        print(f"  failure: {report['failure_reason']}")
    print(f"  report: {args.out.resolve()}")
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
