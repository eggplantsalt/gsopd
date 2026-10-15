"""Compile one explicit forbid-contact Teacher rule against Bridge substep records."""

from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


AUTHKEY = b"gs-opd-local"
BRIDGE_PORT = 38567
SAMPLING_POINT = "immediately_after_each_successful_MjSim.step"


class UnknownEvidence(ValueError):
    """Raised when contact evidence or identity cannot support a verdict."""


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read {label} JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_int(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _resolve_forbid_rule(teacher: dict[str, Any], object_id: str) -> tuple[dict | None, str | None]:
    related = [
        rule
        for rule in teacher["rules"]
        if rule["target_id"] == object_id
        and rule["predicate"] == "contact"
        and "gripper" in rule["subject_parts"]
    ]
    if not related:
        return None, "no_contact_rule_for_gripper_and_target"
    if len(related) != 1:
        return None, "multiple_or_conflicting_gripper_contact_rules"
    rule = related[0]
    if rule["decision"] != "forbid":
        return None, f"unsupported_contact_decision:{rule['decision']}"
    if rule["subject_parts"] != ["gripper"]:
        return None, "rule_includes_subject_parts_outside_minimal_gripper_compiler"
    if rule["phase"] != "any":
        return None, "unsupported_contact_rule_phase"
    return rule, None


def _resolve_target_binding(
    semantic_result: Any,
    object_id: str,
    geometry_inventory: Any,
) -> tuple[dict | None, dict[int, dict], dict[int, dict], str | None]:
    if not isinstance(semantic_result, dict):
        return None, {}, {}, "semantic_disambiguation_result_is_not_an_object"
    decision = semantic_result.get("semantic_decision")
    teacher_object = semantic_result.get("teacher_object")
    association = semantic_result.get("physical_association")
    if (
        not isinstance(decision, dict)
        or decision.get("object_id") != object_id
        or decision.get("status") != "resolved"
        or not isinstance(teacher_object, dict)
        or teacher_object.get("object_id") != object_id
    ):
        return None, {}, {}, "semantic_disambiguation_is_unresolved_or_targets_another_object"
    candidate = semantic_result.get("selected_candidate")
    candidate_index = decision.get("candidate_index")
    if (
        not _is_int(candidate_index)
        or not isinstance(candidate, dict)
        or candidate.get("candidate_index") != candidate_index
    ):
        return None, {}, {}, "semantic_candidate_binding_is_inconsistent"
    if not isinstance(association, dict) or association.get("status") != "matched_by_iou":
        return None, {}, {}, "semantic_result_has_no_resolved_physical_association"
    body_id = association.get("body_id")
    geom_ids = association.get("geom_ids")
    if (
        not _is_int(body_id)
        or not isinstance(association.get("physical_object_id"), str)
        or not association["physical_object_id"]
        or not isinstance(geom_ids, list)
        or not geom_ids
        or any(not _is_int(geom_id) for geom_id in geom_ids)
        or len(set(geom_ids)) != len(geom_ids)
    ):
        return None, {}, {}, "semantic_physical_binding_has_invalid_body_or_geom_ids"
    if candidate.get("body_id") is not None and candidate["body_id"] != body_id:
        return None, {}, {}, "selected_candidate_body_id_disagrees_with_association"
    if candidate.get("geom_ids") is not None and candidate["geom_ids"] != geom_ids:
        return None, {}, {}, "selected_candidate_geom_ids_disagree_with_association"

    if (
        not isinstance(geometry_inventory, dict)
        or geometry_inventory.get("status") != "complete"
        or geometry_inventory.get("robot_classification_status") != "available"
    ):
        return None, {}, {}, "active_bridge_geometry_inventory_is_not_complete"
    raw_geometries = geometry_inventory.get("geometries")
    geom_count = geometry_inventory.get("geom_count")
    if not isinstance(raw_geometries, list) or not _is_int(geom_count, 1):
        return None, {}, {}, "active_bridge_geometry_inventory_is_malformed"
    if len(raw_geometries) != geom_count:
        return None, {}, {}, "active_bridge_geometry_inventory_count_mismatch"
    geom_by_id: dict[int, dict] = {}
    body_by_id: dict[int, dict] = {}
    for item in raw_geometries:
        if not isinstance(item, dict):
            return None, {}, {}, "active_bridge_geometry_inventory_has_invalid_entry"
        geom_id = item.get("geom_id")
        current_body_id = item.get("body_id")
        if (
            not _is_int(geom_id)
            or geom_id >= geom_count
            or not _is_int(current_body_id)
            or not isinstance(item.get("geom_name"), str)
            or not isinstance(item.get("body_name"), str)
            or item.get("category") not in ("robot", "scene")
            or geom_id in geom_by_id
        ):
            return None, {}, {}, "active_bridge_geometry_inventory_has_incomplete_or_duplicate_identity"
        geom_by_id[geom_id] = item
        prior_body = body_by_id.setdefault(
            current_body_id,
            {"body_id": current_body_id, "body_name": item["body_name"], "category": item["category"]},
        )
        if prior_body["body_name"] != item["body_name"] or prior_body["category"] != item["category"]:
            return None, {}, {}, "active_bridge_geometry_inventory_has_inconsistent_body_identity"
    if len(geom_by_id) != geom_count:
        return None, {}, {}, "active_bridge_geometry_inventory_does_not_cover_all_geoms"

    target_body = body_by_id.get(body_id)
    if target_body is None or target_body["category"] != "scene":
        return None, {}, {}, "semantic_target_body_is_missing_or_not_a_scene_body"
    expected_body_name = association.get("body_name")
    if expected_body_name is not None and expected_body_name != target_body["body_name"]:
        return None, {}, {}, "semantic_target_body_name_disagrees_with_active_model"
    expected_geom_names = association.get("geom_names")
    if expected_geom_names is not None and (
        not isinstance(expected_geom_names, list) or len(expected_geom_names) != len(geom_ids)
    ):
        return None, {}, {}, "semantic_target_geom_names_are_malformed"
    for position, geom_id in enumerate(geom_ids):
        geom = geom_by_id.get(geom_id)
        if geom is None or geom["body_id"] != body_id or geom["category"] != "scene":
            return None, {}, {}, "semantic_target_geom_does_not_match_active_scene_body"
        if expected_geom_names is not None and expected_geom_names[position] != geom["geom_name"]:
            return None, {}, {}, "semantic_target_geom_name_disagrees_with_active_model"

    binding = {
        "object_id": object_id,
        "candidate_index": candidate_index,
        "physical_object_id": association.get("physical_object_id"),
        "body_id": body_id,
        "body_name": target_body["body_name"],
        "geom_ids": geom_ids,
        "geom_names": [geom_by_id[geom_id]["geom_name"] for geom_id in geom_ids],
    }
    return binding, geom_by_id, body_by_id, None


def _resolve_gripper_geometries(
    inventory: Any,
    geom_by_id: dict[int, dict],
    body_by_id: dict[int, dict],
) -> tuple[dict[int, int] | None, set[int] | None, str | None]:
    if (
        not isinstance(inventory, dict)
        or inventory.get("status") != "complete"
        or inventory.get("robot_classification_status") != "available"
    ):
        return None, None, "active_bridge_gripper_geometry_inventory_is_incomplete"
    robot_ids = inventory.get("robot_body_ids")
    model_body_count = inventory.get("model_body_count")
    grippers = inventory.get("grippers")
    if (
        not isinstance(robot_ids, list)
        or not robot_ids
        or any(not _is_int(body_id) for body_id in robot_ids)
        or not _is_int(model_body_count, 1)
        or any(body_id >= model_body_count for body_id in robot_ids)
        or len(set(robot_ids)) != len(robot_ids)
        or not isinstance(grippers, list)
        or not grippers
    ):
        return None, None, "active_bridge_gripper_geometry_inventory_is_malformed"
    robot_body_ids = set(robot_ids)
    if any(
        body_id in body_by_id and body_by_id[body_id]["category"] != "robot"
        for body_id in robot_body_ids
    ):
        return None, None, "gripper_inventory_robot_bodies_disagree_with_geometry_inventory"
    if any(
        body["category"] == "robot" and body_id not in robot_body_ids
        for body_id, body in body_by_id.items()
    ):
        return None, None, "geometry_inventory_robot_bodies_are_not_fully_classified"

    gripper_geom_to_body: dict[int, int] = {}
    for gripper in grippers:
        if not isinstance(gripper, dict) or gripper.get("status") != "complete":
            return None, None, "one_or_more_grippers_have_unresolved_geometry_identity"
        geoms = gripper.get("geometries")
        if not isinstance(geoms, list) or not geoms:
            return None, None, "gripper_has_no_resolved_collision_geometries"
        for item in geoms:
            if not isinstance(item, dict):
                return None, None, "gripper_geometry_entry_is_malformed"
            geom_id, body_id = item.get("geom_id"), item.get("body_id")
            active_geom = geom_by_id.get(geom_id) if _is_int(geom_id) else None
            if (
                active_geom is None
                or not _is_int(body_id)
                or body_id not in robot_body_ids
                or active_geom["body_id"] != body_id
                or active_geom["category"] != "robot"
                or item.get("geom_name") != active_geom["geom_name"]
            ):
                return None, None, "gripper_geometry_does_not_match_active_robot_model"
            if geom_id in gripper_geom_to_body and gripper_geom_to_body[geom_id] != body_id:
                return None, None, "gripper_geom_maps_to_multiple_robot_bodies"
            gripper_geom_to_body[geom_id] = body_id
    if not gripper_geom_to_body:
        return None, None, "no_gripper_collision_geometries_resolved"
    return gripper_geom_to_body, robot_body_ids, None


def _complete_monitoring(candidate: Any) -> tuple[dict | None, str | None]:
    if not isinstance(candidate, dict) or candidate.get("status") != "ok":
        return None, "branch_candidate_failed"
    monitor = candidate.get("contact_monitoring")
    if not isinstance(monitor, dict) or monitor.get("status") != "complete":
        return None, "substep_contact_monitoring_not_complete"
    if (
        monitor.get("sampling_point") != SAMPLING_POINT
        or monitor.get("robot_classification_status") != "available"
    ):
        return None, "substep_sampling_or_robot_classification_is_unavailable"
    steps = candidate.get("steps")
    control_steps = monitor.get("control_steps")
    executed_steps = candidate.get("executed_steps")
    if (
        not isinstance(steps, list)
        or not _is_int(executed_steps)
        or len(steps) != executed_steps
        or not isinstance(control_steps, list)
        or len(control_steps) != executed_steps
        or executed_steps == 0
    ):
        return None, "branch_steps_and_contact_monitoring_length_do_not_match"
    expected_total = 0
    captured_total = 0
    unavailable_total = 0
    for position, control_step in enumerate(control_steps):
        if not isinstance(control_step, dict) or control_step.get("status") != "complete":
            return None, f"control_step_{position}_monitoring_is_incomplete"
        expected = control_step.get("expected_physics_substeps")
        captured = control_step.get("captured_physics_substeps")
        unobserved = control_step.get("unobserved_expected_substeps")
        samples = control_step.get("contact_samples")
        if (
            not _is_int(expected, 1)
            or not _is_int(captured)
            or captured != expected
            or not _is_int(unobserved)
            or unobserved != 0
            or not isinstance(samples, dict)
        ):
            return None, f"control_step_{position}_substep_counts_are_incomplete"
        if (
            not _is_int(control_step.get("control_step_index"))
            or control_step["control_step_index"] != position
            or not isinstance(steps[position], dict)
            or not _is_int(steps[position].get("step_index"))
            or steps[position]["step_index"] != position
        ):
            return None, f"control_step_{position}_does_not_match_branch_step"
        counts = [samples.get(key) for key in ("contact", "no_contact", "unavailable")]
        if any(not _is_int(count) for count in counts):
            return None, f"control_step_{position}_contact_sample_counts_are_invalid"
        if sum(counts) != captured or counts[2] != 0:
            return None, f"control_step_{position}_contains_unavailable_contact_samples"
        expected_total += expected
        captured_total += captured
        unavailable_total += counts[2]
    contact_substeps = monitor.get("contact_substeps")
    no_contact_substeps = monitor.get("no_contact_substeps")
    unavailable_contact_substeps = monitor.get("unavailable_contact_substeps")
    if (
        not _is_int(monitor.get("expected_physics_substeps_total"))
        or monitor["expected_physics_substeps_total"] != expected_total
        or not _is_int(monitor.get("captured_physics_substeps"))
        or monitor["captured_physics_substeps"] != captured_total
        or not _is_int(monitor.get("unobserved_expected_substeps"))
        or monitor["unobserved_expected_substeps"] != 0
        or not _is_int(unavailable_contact_substeps)
        or unavailable_contact_substeps != unavailable_total
        or not _is_int(contact_substeps)
        or not _is_int(no_contact_substeps)
        or contact_substeps + no_contact_substeps + unavailable_contact_substeps
        != captured_total
    ):
        return None, "aggregate_contact_monitoring_counts_are_inconsistent"
    events = monitor.get("events")
    if not isinstance(events, list):
        return None, "contact_monitor_events_are_missing"
    if (contact_substeps > 0) != bool(events):
        return None, "contact_substep_count_and_event_list_disagree"
    return monitor, None


def _int_pair_entry(pair: Any, index: int) -> tuple[int, str] | None:
    if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[index], dict):
        return None
    item = pair[index]
    item_id = item.get("id")
    name = item.get("name")
    if not _is_int(item_id) or not isinstance(name, str):
        return None
    return item_id, name


def _inspect_contact_event(
    event: Any,
    *,
    object_body_id: int,
    object_geom_ids: set[int],
    gripper_geom_to_body: dict[int, int],
    robot_body_ids: set[int],
    geom_by_id: dict[int, dict],
    body_by_id: dict[int, dict],
) -> tuple[dict | None, dict | None, str | None]:
    if not isinstance(event, dict):
        return None, None, "contact_event_is_not_an_object"
    geom_pair, body_pair = event.get("geom_pair"), event.get("body_pair")
    geoms = [_int_pair_entry(geom_pair, index) for index in (0, 1)]
    bodies = [_int_pair_entry(body_pair, index) for index in (0, 1)]
    if any(item is None for item in geoms + bodies):
        return None, None, "contact_event_pair_identity_is_malformed"
    geom_ids = [item[0] for item in geoms]
    body_ids = [item[0] for item in bodies]
    for index in (0, 1):
        active_geom = geom_by_id.get(geom_ids[index])
        active_body = body_by_id.get(body_ids[index])
        if (
            active_geom is None
            or active_body is None
            or active_geom["body_id"] != body_ids[index]
            or active_geom["geom_name"] != geoms[index][1]
            or active_body["body_name"] != bodies[index][1]
        ):
            return None, None, "contact_event_identity_disagrees_with_active_model"
        if geom_ids[index] in object_geom_ids and body_ids[index] != object_body_id:
            return None, None, "bound_target_geom_has_a_different_contact_body"
    expected_class = (
        "robot_robot" if body_ids[0] in robot_body_ids and body_ids[1] in robot_body_ids
        else "robot_scene" if body_ids[0] in robot_body_ids or body_ids[1] in robot_body_ids
        else "scene_scene"
    )
    if event.get("classification") != expected_class or event.get("robot_involved") != (
        expected_class != "scene_scene"
    ):
        return None, None, "contact_event_robot_classification_is_inconsistent"
    occurrence_count = event.get("occurrence_count")
    first_control = event.get("first_control_step")
    first_substep = event.get("first_physics_substep")
    if not _is_int(occurrence_count, 1) or not _is_int(first_control) or not _is_int(first_substep):
        return None, None, "contact_event_timing_or_occurrence_is_invalid"
    force = event.get("max_force_norm")
    contact_points = event.get("contact_point_count")
    force_samples = event.get("force_sample_count")
    force_unavailable = event.get("force_unavailable_count")
    if (
        not _is_int(contact_points)
        or not _is_int(force_samples)
        or not _is_int(force_unavailable)
        or force_samples + force_unavailable != contact_points
    ):
        return None, None, "contact_event_force_or_point_counts_are_invalid"
    if force is not None and (
        isinstance(force, bool)
        or not isinstance(force, (int, float))
        or not math.isfinite(force)
        or force < 0
    ):
        return None, None, "contact_event_force_is_invalid"

    target_side = next((index for index in (0, 1) if body_ids[index] == object_body_id), None)
    if target_side is None:
        if any(geom_id in gripper_geom_to_body for geom_id in geom_ids):
            return None, None, "gripper_contact_event_does_not_match_target_body"
        return None, None, None
    gripper_side = 1 - target_side
    gripper_geom_id = geom_ids[gripper_side]
    if gripper_geom_id in gripper_geom_to_body:
        expected_gripper_body = gripper_geom_to_body[gripper_geom_id]
        if body_ids[gripper_side] != expected_gripper_body:
            return None, None, "gripper_geom_contact_body_disagrees_with_model_inventory"
        evidence = {
            "classification": "gripper_target_contact",
            "geom_pair": [dict(item) for item in geom_pair],
            "body_pair": [dict(item) for item in body_pair],
            "first_control_step": first_control,
            "first_physics_substep": first_substep,
            "occurrence_count": occurrence_count,
            "contact_point_count": contact_points,
            "max_force_norm": force,
            "force_sample_count": force_samples,
            "force_unavailable_count": force_unavailable,
        }
        return evidence, None, None
    if body_ids[gripper_side] in robot_body_ids:
        other_part = {
            "classification": "other_robot_part_target_contact",
            "geom_pair": [dict(item) for item in geom_pair],
            "body_pair": [dict(item) for item in body_pair],
            "first_control_step": first_control,
            "first_physics_substep": first_substep,
            "occurrence_count": occurrence_count,
        }
        return None, other_part, None
    return None, None, None


def compile_forbid_contact(
    teacher: dict[str, Any],
    object_id: str,
    semantic_result: Any,
    branch_result: Any,
    geometry_inventory: Any,
    gripper_inventory: Any,
) -> dict[str, Any]:
    rule, rule_error = _resolve_forbid_rule(teacher, object_id)
    binding, geom_by_id, body_by_id, binding_error = _resolve_target_binding(
        semantic_result, object_id, geometry_inventory
    )
    gripper_geom_to_body, robot_body_ids, gripper_error = _resolve_gripper_geometries(
        gripper_inventory, geom_by_id, body_by_id
    ) if binding_error is None else (None, None, "target_physical_binding_is_unavailable")
    main_restored = isinstance(branch_result, dict) and branch_result.get("main_state_restored") is True
    branch_error = None
    candidates = branch_result.get("candidates") if isinstance(branch_result, dict) else None
    if not main_restored:
        branch_error = "branch_rollout_did_not_confirm_main_state_restore"
    elif branch_result.get("error"):
        branch_error = "branch_rollout_returned_error"
    elif not isinstance(candidates, list) or not candidates:
        branch_error = "branch_rollout_has_no_candidate_results"

    compiled_rule = None
    if rule is not None and binding is not None:
        compiled_rule = {
            "rule_id": rule["id"],
            "predicate": "contact",
            "decision": "forbid",
            "subject_part": "gripper",
            "target_object_id": object_id,
            "target_body_id": binding["body_id"],
            "target_body_name": binding["body_name"],
            "target_geom_ids": binding["geom_ids"],
        }

    common_error = rule_error or binding_error or gripper_error or branch_error
    if isinstance(branch_result, dict) and branch_result.get("error") and branch_error:
        branch_error = str(branch_result["error"])
        common_error = rule_error or binding_error or gripper_error or branch_error

    candidate_results = []
    if isinstance(candidates, list):
        for candidate in candidates:
            candidate_id = candidate.get("candidate_id") if isinstance(candidate, dict) else None
            base = {"candidate_id": candidate_id, "verdict": "unknown", "evidence": []}
            if common_error is not None:
                base["reason"] = common_error
                candidate_results.append(base)
                continue
            monitor, coverage_error = _complete_monitoring(candidate)
            if coverage_error is not None:
                base["reason"] = coverage_error
                candidate_results.append(base)
                continue

            violations = []
            other_robot_contacts = []
            event_error = None
            assert binding is not None
            assert gripper_geom_to_body is not None
            assert robot_body_ids is not None
            for event in monitor["events"]:
                violation, other_part, current_error = _inspect_contact_event(
                    event,
                    object_body_id=binding["body_id"],
                    object_geom_ids=set(binding["geom_ids"]),
                    gripper_geom_to_body=gripper_geom_to_body,
                    robot_body_ids=robot_body_ids,
                    geom_by_id=geom_by_id,
                    body_by_id=body_by_id,
                )
                if current_error is not None:
                    event_error = current_error
                    break
                if violation is not None:
                    violations.append(violation)
                if other_part is not None:
                    other_robot_contacts.append(other_part)
            if event_error is not None:
                base["reason"] = event_error
                candidate_results.append(base)
            elif violations:
                base["verdict"] = "violation"
                base["reason"] = "forbidden_gripper_contact_observed"
                base["evidence"] = violations
                base["contact_monitoring"] = {
                    "status": monitor["status"],
                    "captured_physics_substeps": monitor["captured_physics_substeps"],
                    "expected_physics_substeps_total": monitor["expected_physics_substeps_total"],
                }
                candidate_results.append(base)
            else:
                base["verdict"] = "safe"
                base["reason"] = "complete_substep_monitoring_found_no_gripper_target_contact"
                base["evidence"] = {
                    "complete_contact_substeps": monitor["captured_physics_substeps"],
                    "contact_substeps": monitor["contact_substeps"],
                    "no_contact_substeps": monitor["no_contact_substeps"],
                    "other_robot_part_target_contacts": other_robot_contacts,
                }
                candidate_results.append(base)

    verdicts = [item["verdict"] for item in candidate_results]
    if "violation" in verdicts:
        overall = "violation"
    elif verdicts and all(verdict == "safe" for verdict in verdicts):
        overall = "safe"
    else:
        overall = "unknown"
    return {
        "status": overall,
        "scope": "only the compiled forbid contact(gripper, target_object) rule",
        "object_id": object_id,
        "compiled_rule": compiled_rule,
        "target_binding": binding,
        "gripper_geom_ids": sorted(gripper_geom_to_body) if gripper_geom_to_body else None,
        "main_state_restored": main_restored,
        "candidates": candidate_results,
        "unknown_reasons": [reason for reason in (rule_error, binding_error, gripper_error, branch_error) if reason],
    }


def _bridge_request(connection, request: dict[str, Any]) -> Any:
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict):
        return {"error": "Safety Bridge returned a non-object response"}
    return response


def _capture_bridge_results(host: str, port: int, candidates: list[Any]) -> tuple[Any, Any, Any, Any]:
    connection = Client((host, port), authkey=AUTHKEY)
    try:
        geometry_inventory = _bridge_request(connection, {"op": "geometry_inventory"})
        gripper_inventory = _bridge_request(connection, {"op": "gripper_geometry_inventory"})
        snapshot = _bridge_request(connection, {"op": "snapshot"})
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("snapshot_id"), str):
            error = snapshot.get("error", "snapshot operation failed") if isinstance(snapshot, dict) else "snapshot operation failed"
            return geometry_inventory, gripper_inventory, {"error": error, "main_state_restored": False}, None
        branch_result = _bridge_request(
            connection,
            {
                "op": "branch_rollout",
                "snapshot_id": snapshot["snapshot_id"],
                "candidates": candidates,
            },
        )
        return geometry_inventory, gripper_inventory, branch_result, snapshot["snapshot_id"]
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-json", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--semantic-result", type=Path, required=True)
    parser.add_argument(
        "--candidates-json",
        type=Path,
        required=True,
        help="JSON list accepted by Bridge branch_rollout: candidate_id plus 7-D action sequences",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT)
    args = parser.parse_args(argv)
    for label, path in (
        ("Safety IR v2", args.teacher_json),
        ("VLM semantic disambiguation result", args.semantic_result),
        ("branch candidate actions", args.candidates_json),
    ):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    try:
        teacher = parse_teacher_output_v2(args.teacher_json.read_text(encoding="utf-8"))
        if not any(item["id"] == args.object_id for item in teacher["objects"]):
            parser.error(f"object_id {args.object_id!r} is absent from Safety IR v2")
        semantic_result = _load_json(args.semantic_result, "semantic result")
        candidates = _load_json(args.candidates_json, "branch candidates")
        if not isinstance(candidates, list) or not candidates:
            parser.error("branch candidates JSON must be a non-empty list")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            geometry_inventory, gripper_inventory, branch_result, snapshot_id = _capture_bridge_results(
                args.host, args.port, candidates
            )
        except (OSError, EOFError) as exc:
            geometry_inventory = {"status": "unavailable"}
            gripper_inventory = {"status": "unavailable"}
            branch_result = {"error": f"Cannot query Safety Bridge: {type(exc).__name__}: {exc}", "main_state_restored": False}
            snapshot_id = None
        result = compile_forbid_contact(
            teacher,
            args.object_id,
            semantic_result,
            branch_result,
            geometry_inventory,
            gripper_inventory,
        )
        result.update(
            {
                "snapshot_id": snapshot_id,
                "teacher_json": str(args.teacher_json.resolve()),
                "semantic_result": str(args.semantic_result.resolve()),
                "candidates_json": str(args.candidates_json.resolve()),
                "bridge_host": args.host,
                "bridge_port": args.port,
            }
        )
        _write_json(args.out_dir / "bridge_geometry_inventory.json", geometry_inventory)
        _write_json(args.out_dir / "bridge_gripper_geometry_inventory.json", gripper_inventory)
        _write_json(args.out_dir / "branch_rollout_result.json", branch_result)
        result["geometry_inventory_file"] = "bridge_geometry_inventory.json"
        result["gripper_geometry_inventory_file"] = "bridge_gripper_geometry_inventory.json"
        result["branch_rollout_result_file"] = "branch_rollout_result.json"
        _write_json(args.out_dir / "contact_safety_compiler_results.json", result)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Contact Safety Compiler Smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Contact safety verdict: {result['status']}")
    print(json.dumps(result["compiled_rule"], ensure_ascii=False, indent=2))
    for candidate in result["candidates"]:
        print(
            f"candidate={candidate['candidate_id']} verdict={candidate['verdict']} "
            f"reason={candidate.get('reason')} evidence_count="
            f"{len(candidate['evidence']) if isinstance(candidate.get('evidence'), list) else 0}"
        )
        if candidate["verdict"] == "violation":
            print(json.dumps(candidate["evidence"], ensure_ascii=False, indent=2))
    print(f"Results saved under: {args.out_dir.resolve()}")
    return 0 if result["status"] != "unknown" else 2


if __name__ == "__main__":
    raise SystemExit(main())
