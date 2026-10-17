"""Reusable Near-Far-triggered, geometry-guided local Repair search.

This module owns the decision-state preflight and candidate orchestration. It
delegates action construction, workspace/task-progress validation, endpoint
replay, and candidate assessment to the existing feasibility Smoke helpers so
the two paths do not drift. It never selects a candidate or creates a training
label.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.gs_opd.repair.near_far import CHUNK_LENGTH, action_chunk_fingerprint
from research.gs_opd.teacher.schema_v2 import validate_teacher_output_v2

CHUNK_LENGTH = 5
MAX_CANDIDATES = 36


def _object_inventory_binding(
    binding_validation: dict[str, Any], object_id: str
) -> dict[str, Any] | None:
    objects = binding_validation.get("objects")
    binding = objects.get(object_id) if isinstance(objects, dict) else None
    if not isinstance(binding, dict) or binding.get("status") != "matches_current_geometry_inventory":
        return None
    body_id = binding.get("body_id")
    geom_ids = binding.get("geom_ids")
    if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id < 0:
        return None
    if not isinstance(geom_ids, list) or not geom_ids or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in geom_ids
    ):
        return None
    if len(set(geom_ids)) != len(geom_ids):
        return None
    return binding


def _preflight(
    *,
    snapshot: dict[str, Any],
    actions: np.ndarray,
    near_far_result: dict[str, Any],
    safety_ir: dict[str, Any],
    semantic_bindings: dict[str, Any],
    semantic_binding_validation: dict[str, Any],
    geometry_inventory: dict[str, Any],
    gripper_inventory: dict[str, Any],
    repair_context: dict[str, Any],
    task_object_binding: dict[str, Any],
    a0_physical_validation: dict[str, Any],
    subject_object_id: str,
    obstacle_object_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from research.gs_opd.smoke import early_repair_s70_smoke as probe
    from research.gs_opd.smoke.near_far_gap_smoke import _resolve_bound_geom_sets, _select_forbid_pair
    from research.gs_opd.smoke.contact_safety_compiler_smoke import _complete_monitoring

    checks: dict[str, Any] = {"passed": False, "errors": []}

    def fail(code: str) -> tuple[dict[str, Any], None]:
        checks["errors"].append(code)
        return checks, None

    try:
        validate_teacher_output_v2(safety_ir)
    except Exception as exc:
        return fail(f"safety_ir_v2_invalid:{type(exc).__name__}:{exc}")
    checks["safety_ir_v2_valid"] = True

    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("snapshot_id"), str) or not snapshot["snapshot_id"]:
        return fail("snapshot_id_missing")
    if not isinstance(snapshot.get("snapshot_state"), dict):
        return fail("snapshot_state_missing")
    if not isinstance(near_far_result, dict):
        return fail("near_far_result_missing")
    if (
        near_far_result.get("status") != "complete"
        or near_far_result.get("risk_verdict") != "warning"
        or near_far_result.get("risk_warning") is not True
    ):
        return fail("near_far_not_complete_with_positive_warning")
    decision_step = near_far_result.get("decision_step")
    samples = near_far_result.get("distance_samples")
    if (
        isinstance(decision_step, bool)
        or not isinstance(decision_step, int)
        or decision_step < 0
        or near_far_result.get("simulation_complete") is not True
        or near_far_result.get("requested_action_steps") != CHUNK_LENGTH
        or near_far_result.get("executed_action_steps") != CHUNK_LENGTH
        or not isinstance(samples, list)
        or len(samples) != CHUNK_LENGTH + 1
    ):
        return fail("near_far_forecast_coverage_incomplete")
    for offset, sample in enumerate(samples):
        if (
            not isinstance(sample, dict)
            or sample.get("distance_available") is not True
            or isinstance(sample.get("step_index"), bool)
            or not isinstance(sample.get("step_index"), int)
            or sample.get("step_index") != decision_step + offset
        ):
            return fail(f"near_far_distance_sample_incomplete_or_misaligned:{offset}")
    if near_far_result.get("snapshot_id") != snapshot["snapshot_id"]:
        return fail("near_far_snapshot_id_mismatch")
    try:
        action_fingerprint = action_chunk_fingerprint(actions)
    except Exception as exc:
        return fail(f"student_a0_invalid:{type(exc).__name__}:{exc}")
    if near_far_result.get("action_chunk_sha256") != action_fingerprint:
        return fail("near_far_action_chunk_fingerprint_mismatch")
    restore_record = near_far_result.get("snapshot_restore")
    if (
        not isinstance(restore_record, dict)
        or not isinstance(restore_record.get("before"), dict)
        or restore_record["before"].get("matched") is not True
        or not isinstance(restore_record.get("after"), dict)
        or restore_record["after"].get("matched") is not True
    ):
        return fail("near_far_snapshot_restore_not_verified")
    checks["near_far_trigger_verified"] = True

    try:
        _select_forbid_pair(safety_ir, subject_object_id, obstacle_object_id)
    except Exception as exc:
        return fail(f"forbidden_object_pair_rule_unavailable:{type(exc).__name__}:{exc}")
    if semantic_binding_validation.get("valid") is not True:
        return fail("semantic_geometry_binding_validation_failed")
    if geometry_inventory.get("status") != "complete" or geometry_inventory.get("robot_classification_status") != "available":
        return fail("geometry_inventory_incomplete")
    if gripper_inventory.get("status") != "complete" or gripper_inventory.get("robot_classification_status") != "available":
        return fail("gripper_inventory_incomplete")
    if repair_context.get("status") != "available":
        return fail("repair_context_unavailable")
    if not isinstance(semantic_bindings, dict):
        return fail("semantic_bindings_missing")

    subject = _object_inventory_binding(semantic_binding_validation, subject_object_id)
    obstacle = _object_inventory_binding(semantic_binding_validation, obstacle_object_id)
    task_body = task_object_binding.get("body_id") if isinstance(task_object_binding, dict) else None
    if subject is None or obstacle is None:
        return fail("forbidden_pair_geometry_binding_missing")
    if subject["body_id"] == obstacle["body_id"]:
        return fail("forbidden_pair_resolves_to_same_body")
    if isinstance(task_body, bool) or not isinstance(task_body, int) or task_body != subject["body_id"]:
        return fail("task_object_binding_does_not_match_repair_subject")
    for object_id, binding in ((subject_object_id, subject), (obstacle_object_id, obstacle)):
        semantic = semantic_bindings.get(object_id)
        teacher_object = next(
            (item for item in safety_ir.get("objects", [])
             if isinstance(item, dict) and item.get("id") == object_id),
            None,
        )
        semantic_teacher = semantic.get("teacher_object") if isinstance(semantic, dict) else None
        if (
            not isinstance(teacher_object, dict)
            or not isinstance(semantic_teacher, dict)
            or semantic_teacher.get("name") != teacher_object.get("name")
            or semantic_teacher.get("role") != teacher_object.get("role")
        ):
            return fail(f"semantic_teacher_object_mismatch:{object_id}")

    try:
        _resolved, subject_geom_ids, obstacle_geom_ids = _resolve_bound_geom_sets(
            semantic_binding_validation, geometry_inventory, subject_object_id, obstacle_object_id
        )
    except Exception as exc:
        return fail(f"geometry_inventory_geom_set_resolution_failed:{type(exc).__name__}:{exc}")
    if set(near_far_result.get("target_geom_ids", [])) != set(subject_geom_ids):
        return fail("near_far_subject_geom_set_mismatch")
    if set(near_far_result.get("obstacle_geom_ids", [])) != set(obstacle_geom_ids):
        return fail("near_far_obstacle_geom_set_mismatch")
    checks["semantic_bindings_match_current_geometry_inventory"] = True

    if actions.shape != (CHUNK_LENGTH, 7) or not np.isfinite(actions).all():
        return fail(f"student_a0_must_be_finite_{CHUNK_LENGTH}_by_7")
    validation = a0_physical_validation
    if not isinstance(validation, dict) or validation.get("main_state_restored") is not True:
        return fail("a0_branch_snapshot_restore_not_confirmed")
    if validation.get("snapshot_id") != snapshot["snapshot_id"]:
        return fail("a0_physical_validation_snapshot_id_mismatch")
    if validation.get("action_chunk_sha256") != action_fingerprint:
        return fail("a0_physical_validation_action_chunk_mismatch")
    baseline = validation.get("branch_candidate")
    compiled = validation.get("compiler_candidate")
    if not isinstance(baseline, dict) or not probe._complete_five_steps(baseline):
        return fail("a0_branch_did_not_complete_five_steps")
    _monitor, monitor_error = _complete_monitoring(baseline)
    if monitor_error is not None:
        return fail(f"a0_contact_monitoring_incomplete:{monitor_error}")
    if not isinstance(compiled, dict) or compiled.get("verdict") != "safe":
        return fail("a0_contact_compiler_did_not_prove_safe")
    baseline_id = baseline.get("candidate_id")
    compiled_id = compiled.get("candidate_id")
    if not isinstance(baseline_id, str) or not baseline_id or compiled_id != baseline_id:
        return fail("a0_branch_and_compiler_candidate_ids_do_not_match")
    checks.update({
        "passed": True,
        "near_far_trigger_verified": True,
        "semantic_bindings_match_current_geometry_inventory": True,
        "a0_complete_five_step_branch": True,
        "a0_contact_monitoring_complete": True,
        "a0_contact_compiler_verdict": "safe",
        "a0_action_chunk_sha256": action_fingerprint,
        "subject_body_id": subject["body_id"],
        "obstacle_body_id": obstacle["body_id"],
        "target_geom_ids": subject_geom_ids,
        "obstacle_geom_ids": obstacle_geom_ids,
    })
    return checks, {
        "baseline_candidate": baseline,
        "baseline_compiled_candidate": compiled,
        "subject_binding": subject,
        "obstacle_binding": obstacle,
        "subject_geom_ids": subject_geom_ids,
        "obstacle_geom_ids": obstacle_geom_ids,
    }


def _candidate_validation_complete(candidate_result: dict[str, Any]) -> bool:
    """Whether a candidate received every required physical and task check."""
    endpoint = candidate_result.get("endpoint_replay")
    task_progress = candidate_result.get("task_progress")
    action_constraints = candidate_result.get("action_constraints")
    return bool(
        candidate_result.get("full_five_step_chunk") is True
        and candidate_result.get("contact_monitoring_complete") is True
        and candidate_result.get("physical_safety_verdict") in ("safe", "violation")
        and isinstance(task_progress, dict)
        and task_progress.get("status") in ("pass", "fail")
        and isinstance(action_constraints, dict)
        and action_constraints.get("status") in ("pass", "fail")
        and isinstance(endpoint, dict)
        and endpoint.get("status") == "complete"
        and endpoint.get("main_state_restored") is True
    )


def run_early_repair_search(
    connection: Any,
    snapshot: dict[str, Any],
    original_actions: Any,
    near_far_result: dict[str, Any],
    safety_ir: dict[str, Any],
    semantic_bindings: dict[str, Any],
    semantic_binding_validation: dict[str, Any],
    geometry_inventory: dict[str, Any],
    gripper_inventory: dict[str, Any],
    repair_context: dict[str, Any],
    task_object_binding: dict[str, Any],
    a0_physical_validation: dict[str, Any],
    *,
    subject_object_id: str,
    obstacle_object_id: str,
    max_candidates: int = MAX_CANDIDATES,
    distmax_m: float = 10.0,
) -> dict[str, Any]:
    """Search and validate geometry-guided candidates from one decision Snapshot.

    A0 physical validation has this shape::

        {"main_state_restored": True,
         "snapshot_id": <the supplied Snapshot ID>,
         "action_chunk_sha256": <fingerprint of this exact A0>,
         "branch_candidate": <full 5-step Bridge candidate>,
         "compiler_candidate": <Contact Compiler candidate verdict>}

    Near-Far provenance must identify the same Snapshot and exact A0 chunk.
    Candidate IDs/actions are returned only after all physical, contact,
    progress, action, clearance, and restoration checks. No candidate is
    selected and no training label is produced here.
    """
    from research.gs_opd.smoke import early_repair_s70_smoke as probe
    from research.gs_opd.smoke.contact_safety_compiler_smoke import compile_forbid_contact

    actions = np.asarray(original_actions, dtype=np.float64)
    report: dict[str, Any] = {
        "status": "not_executable",
        "executable": False,
        "candidate_results": [],
        "qualified_candidates": [],
        "verification_complete": False,
        "training_label_created": False,
        "candidate_selected": False,
        "original_a0_actions": actions.tolist() if actions.ndim == 2 else None,
        "candidate_limit": max_candidates,
        "decision_step": near_far_result.get("decision_step") if isinstance(near_far_result, dict) else None,
        "preflight": None,
        "snapshot_restore": {"before": None, "after": None},
        "errors": [],
    }
    snapshot_id = snapshot.get("snapshot_id") if isinstance(snapshot, dict) else None
    snapshot_state = snapshot.get("snapshot_state") if isinstance(snapshot, dict) else None
    can_restore = isinstance(snapshot_id, str) and bool(snapshot_id) and isinstance(snapshot_state, dict)
    try:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= MAX_CANDIDATES:
            report["errors"].append("max_candidates_must_be_in_1_to_36")
        elif isinstance(distmax_m, bool) or not isinstance(distmax_m, (int, float)) or not np.isfinite(distmax_m) or distmax_m <= 0:
            report["errors"].append("distmax_m_must_be_finite_and_positive")
        else:
            preflight, resolved = _preflight(
                snapshot=snapshot,
                actions=actions,
                near_far_result=near_far_result,
                safety_ir=safety_ir,
                semantic_bindings=semantic_bindings,
                semantic_binding_validation=semantic_binding_validation,
                geometry_inventory=geometry_inventory,
                gripper_inventory=gripper_inventory,
                repair_context=repair_context,
                task_object_binding=task_object_binding,
                a0_physical_validation=a0_physical_validation,
                subject_object_id=subject_object_id,
                obstacle_object_id=obstacle_object_id,
            )
            report["preflight"] = preflight
            if resolved is not None:
                report["snapshot_restore"]["before"] = probe._restore_and_compare(
                    connection, snapshot_id, snapshot_state
                )
                if report["snapshot_restore"]["before"].get("matched") is not True:
                    report["errors"].append("snapshot_restore_before_search_failed")
                else:
                    nearfar_samples = near_far_result.get("distance_samples", [])
                    endpoint_sample = nearfar_samples[-1] if isinstance(nearfar_samples, list) and nearfar_samples else {}
                    endpoint_distance = {
                        "query_complete": endpoint_sample.get("distance_available") is True,
                        "distance_reliable": endpoint_sample.get("distance_available") is True,
                        "minimum_distance_m": endpoint_sample.get("minimum_distance_m"),
                        "minimum_pair": near_far_result.get("endpoint_closest_geom_evidence"),
                        "mesh_approximation": endpoint_sample.get("mesh_approximation"),
                    }
                    try:
                        eef_path = [
                            resolved["baseline_candidate"]["start_state"]["robot_states"][0]["eef_pos"],
                            *[
                                step["state"]["robot_states"][0]["eef_pos"]
                                for step in resolved["baseline_candidate"]["steps"]
                            ],
                        ]
                        directions, direction_basis = probe.derive_clearance_directions(
                            endpoint_distance,
                            resolved["subject_binding"]["body_id"],
                            resolved["obstacle_binding"]["body_id"],
                            repair_context.get("world_up"),
                            eef_path,
                        )
                        specs, generation_rejections = probe.generate_clearance_candidates(
                            actions,
                            directions,
                            repair_context,
                            probe.PROBE_CONFIG,
                            max_candidates=max_candidates,
                            candidate_prefix="early",
                        )
                    except Exception as exc:
                        report["errors"].append(f"candidate_generation_failed:{type(exc).__name__}:{exc}")
                    else:
                        report["direction_basis"] = direction_basis
                        report["candidate_generation_rejections"] = generation_rejections
                        report["generated_candidate_count"] = len(specs)
                        if specs:
                            try:
                                branch_result = probe._request(connection, {
                                    "op": "branch_rollout",
                                    "snapshot_id": snapshot_id,
                                    "candidates": [
                                        {"candidate_id": item["candidate_id"], "actions": item["actions"].tolist()}
                                        for item in specs
                                    ],
                                })
                                report["candidate_branch_main_state_restored"] = (
                                    branch_result.get("main_state_restored") is True
                                )
                                if branch_result.get("main_state_restored") is not True:
                                    report["errors"].append("candidate_branch_main_snapshot_restore_failed")
                                elif branch_result.get("error"):
                                    report["errors"].append(f"candidate_branch_error:{branch_result['error']}")
                                else:
                                    compiler_result = compile_forbid_contact(
                                        safety_ir,
                                        subject_object_id,
                                        semantic_bindings[subject_object_id],
                                        branch_result,
                                        geometry_inventory,
                                        gripper_inventory,
                                        object_semantic_results=semantic_bindings,
                                    )
                                    restore_failure = None
                                    for spec in specs:
                                        branch_candidate = probe._branch_candidate(
                                            branch_result, spec["candidate_id"]
                                        )
                                        compiled_candidate = probe._compiler_candidate_summary(
                                            compiler_result, spec["candidate_id"]
                                        )
                                        endpoint_replay = None
                                        if probe._complete_five_steps(branch_candidate):
                                            endpoint_replay = probe.replay_and_measure_endpoint(
                                                connection,
                                                snapshot_id,
                                                snapshot_state,
                                                spec["actions"],
                                                branch_candidate,
                                                resolved["subject_geom_ids"],
                                                resolved["obstacle_geom_ids"],
                                                distmax_m,
                                            )
                                            if endpoint_replay.get("main_state_restored") is not True:
                                                restore_failure = endpoint_replay.get(
                                                    "reason", "candidate_endpoint_snapshot_restore_failed"
                                                )
                                        candidate_report = probe.assess_candidate(
                                            {**spec, "original_actions": actions},
                                            branch_candidate,
                                            compiled_candidate,
                                            resolved["baseline_candidate"],
                                            repair_context,
                                            task_object_binding,
                                            probe.PROBE_CONFIG,
                                            float(endpoint_sample["minimum_distance_m"]),
                                            endpoint_replay,
                                        )
                                        candidate_report.pop("training_label_valid", None)
                                        candidate_report.pop("final_a_plus", None)
                                        candidate_report["candidate_validation_complete"] = (
                                            _candidate_validation_complete(candidate_report)
                                        )
                                        report["candidate_results"].append(candidate_report)
                                        if restore_failure:
                                            break
                                    report["candidate_results_complete"] = (
                                        len(report["candidate_results"]) == len(specs)
                                    )
                                    if restore_failure:
                                        report["errors"].append(f"candidate_endpoint_restore_failed:{restore_failure}")
                            except Exception as exc:
                                report["errors"].append(f"candidate_verification_failed:{type(exc).__name__}:{exc}")
                        else:
                            report["candidate_results_complete"] = True
                            report["candidate_validation_complete"] = True

                        report["qualified_candidates"] = [
                            {
                                "candidate_id": item.get("candidate_id"),
                                "actions": item.get("actions"),
                                "verification": item,
                            }
                            for item in report["candidate_results"]
                            if item.get("accepted_for_probe") is True
                        ]
                        if specs:
                            results_complete = len(report["candidate_results"]) == len(specs)
                            validations_complete = all(
                                item.get("candidate_validation_complete") is True
                                for item in report["candidate_results"]
                            )
                            report["candidate_results_complete"] = results_complete
                            report["candidate_validation_complete"] = validations_complete
                            report["verification_complete"] = bool(
                                report.get("candidate_branch_main_state_restored") is True
                                and results_complete
                                and validations_complete
                            )
                            if not report["verification_complete"]:
                                report["errors"].append("candidate_validation_incomplete; no candidate is executable")
                        else:
                            report["verification_complete"] = True
    except Exception as exc:
        report["errors"].append(f"early_repair_search_failed:{type(exc).__name__}:{exc}")
    finally:
        if can_restore:
            try:
                report["snapshot_restore"]["after"] = probe._restore_and_compare(
                    connection, snapshot_id, snapshot_state
                )
            except Exception as exc:
                report["snapshot_restore"]["after"] = {
                    "matched": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        else:
            report["snapshot_restore"]["after"] = {
                "matched": False,
                "reason": "snapshot_id_or_snapshot_state_unavailable",
            }

    restore_ok = (
        isinstance(report["snapshot_restore"].get("before"), dict)
        and report["snapshot_restore"]["before"].get("matched") is True
        and isinstance(report["snapshot_restore"].get("after"), dict)
        and report["snapshot_restore"]["after"].get("matched") is True
    )
    if not restore_ok:
        report["errors"].append("snapshot_restore_unverified; candidates_are_not_executable")
    report["executable"] = bool(
        restore_ok
        and report["verification_complete"] is True
        and report["qualified_candidates"]
        and not report["errors"]
    )
    if report["executable"]:
        report["status"] = "qualified_candidates_available"
    elif restore_ok and report["verification_complete"] is True and not report["errors"]:
        report["status"] = "no_qualified_candidates"
    elif report["errors"]:
        report["status"] = "not_executable"
    else:
        report["status"] = "no_qualified_candidates"
    report["errors"] = list(dict.fromkeys(report["errors"]))
    return report
