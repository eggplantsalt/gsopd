"""Snapshot-based, verifier-guided local repair for decoded 7-D Pi0.5 actions.

This module contains deterministic search and evidence checks. The caller supplies
an evaluator that runs each candidate from the same Bridge snapshot and invokes
the existing Contact Safety Compiler; no model, environment, or dependency is
loaded here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class RepairConfig:
    """Bounded adaptive search settings; amplitudes are world meters per step."""

    minimum_amplitude_m: float = 0.002
    maximum_amplitude_m_per_step: float = 0.04
    maximum_integrated_detour_m: float = 0.08
    amplitude_growth: float = 2.0
    amplitude_refinement_iterations: int = 3
    minimum_refinement_m: float = 0.001
    windows: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (3, 3))
    max_candidate_evaluations: int = 96
    minimum_eef_progress_m: float = 0.005
    required_baseline_progress_fraction: float = 0.70
    minimum_object_motion_m: float = 0.002
    required_object_progress_fraction: float = 0.70
    maximum_action_correction_slew_m: float = 0.025
    workspace_tolerance: float = 1e-5


def _vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite vector of length {size}")
    return result


def _unit(value: Any, label: str) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-10:
        return None
    return vector / norm


def _candidate_with_id(container: Any, candidate_id: Any = None) -> dict | None:
    if not isinstance(container, dict) or not isinstance(container.get("candidates"), list):
        return None
    candidates = container["candidates"]
    for candidate in candidates:
        if isinstance(candidate, dict) and (
            candidate_id is None or candidate.get("candidate_id") == candidate_id
        ):
            return candidate
    return None


def _compiled_candidate(compiled: Any, candidate_id: Any = None) -> dict | None:
    if not isinstance(compiled, dict) or not isinstance(compiled.get("candidates"), list):
        return None
    for candidate in compiled["candidates"]:
        if isinstance(candidate, dict) and (
            candidate_id is None or candidate.get("candidate_id") == candidate_id
        ):
            return candidate
    return None


def _pair_ids(pair: Any) -> tuple[int, int] | None:
    if not isinstance(pair, list) or len(pair) != 2:
        return None
    try:
        ids = tuple(int(item["id"]) for item in pair)
    except Exception:
        return None
    if any(isinstance(item.get("id"), bool) for item in pair if isinstance(item, dict)):
        return None
    return ids


def _pair_names(pair: Any) -> tuple[str, str] | None:
    if not isinstance(pair, list) or len(pair) != 2:
        return None
    names = tuple(item.get("name") for item in pair if isinstance(item, dict))
    if len(names) != 2 or any(not isinstance(name, str) or not name for name in names):
        return None
    return names


def _is_id_list(value: Any) -> list[int] | None:
    if not isinstance(value, list) or not value:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    if len(set(value)) != len(value):
        return None
    return value


def _object_contact_violation(
    evidence: Any,
    check: dict[str, Any],
    compiled_rules: list[Any],
    events: list[Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve object/object evidence against its active binding and raw event."""
    if not isinstance(evidence, dict):
        return None, "object_contact_evidence_is_malformed"
    rule = next(
        (
            item for item in compiled_rules
            if isinstance(item, dict)
            and item.get("rule_id") == check.get("rule_id")
            and item.get("subject_object_id") == check.get("subject_id")
            and item.get("target_object_id") == check.get("target_id")
        ),
        None,
    )
    if rule is None or rule.get("decision") != "forbid" or rule.get("predicate") != "contact":
        return None, "compiled_object_contact_rule_binding_is_missing_or_conflicting"
    try:
        subject_body_id = int(rule["subject_body_id"])
        target_body_id = int(rule["target_body_id"])
    except (KeyError, TypeError, ValueError):
        return None, "compiled_object_contact_body_binding_is_missing"
    if subject_body_id == target_body_id:
        return None, "compiled_object_contact_bodies_are_not_distinct"
    subject_geom_ids = _is_id_list(rule.get("subject_geom_ids"))
    target_geom_ids = _is_id_list(rule.get("target_geom_ids"))
    if subject_geom_ids is None or target_geom_ids is None:
        return None, "compiled_object_contact_geom_binding_is_missing"
    if (
        evidence.get("classification") != "forbidden_object_object_contact"
        or evidence.get("subject_object_id") != check.get("subject_id")
        or evidence.get("target_object_id") != check.get("target_id")
    ):
        return None, "object_contact_evidence_semantic_identity_conflicts_with_rule"
    try:
        evidence_subject_body = int(evidence["subject_body_id"])
        evidence_target_body = int(evidence["target_body_id"])
        control_step = int(evidence["first_control_step"])
        physics_substep = int(evidence["first_physics_substep"])
    except (KeyError, TypeError, ValueError):
        return None, "object_contact_evidence_body_or_timing_is_missing"
    if (
        evidence_subject_body != subject_body_id
        or evidence_target_body != target_body_id
        or control_step < 0
        or physics_substep < 0
    ):
        return None, "object_contact_evidence_disagrees_with_current_semantic_body_binding"
    geom_ids = _pair_ids(evidence.get("geom_pair"))
    body_ids = _pair_ids(evidence.get("body_pair"))
    geom_names = _pair_names(evidence.get("geom_pair"))
    body_names = _pair_names(evidence.get("body_pair"))
    if geom_ids is None or body_ids is None or geom_names is None or body_names is None:
        return None, "object_contact_evidence_pair_identity_is_malformed"
    if set(body_ids) != {subject_body_id, target_body_id}:
        return None, "object_contact_evidence_pair_does_not_match_bound_bodies"

    # The compiler's semantic geom IDs are visual anchors and need not be the
    # collision geom that contacted. Bind the raw collision geoms by their
    # corresponding body IDs, preserving the event's geom/body ordering.
    subject_pair_sides = [index for index, body_id in enumerate(body_ids) if body_id == subject_body_id]
    target_pair_sides = [index for index, body_id in enumerate(body_ids) if body_id == target_body_id]
    if len(subject_pair_sides) != 1 or len(target_pair_sides) != 1:
        return None, "object_contact_evidence_has_ambiguous_bound_body_side"
    subject_side = subject_pair_sides[0]
    target_side = target_pair_sides[0]
    subject_geom_id = geom_ids[subject_side]
    target_geom_id = geom_ids[target_side]
    if subject_geom_id in target_geom_ids or target_geom_id in subject_geom_ids:
        return None, "object_contact_collision_geom_conflicts_with_opposite_semantic_binding"

    matching_events = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if (
            _pair_ids(event.get("geom_pair")) == geom_ids
            and _pair_ids(event.get("body_pair")) == body_ids
            and _pair_names(event.get("geom_pair")) == geom_names
            and _pair_names(event.get("body_pair")) == body_names
            and event.get("first_control_step") == control_step
            and event.get("first_physics_substep") == physics_substep
        ):
            matching_events.append(event)
    if len(matching_events) != 1:
        return None, "object_contact_evidence_has_no_unique_matching_raw_monitor_event"
    event = matching_events[0]
    if event.get("classification") != "scene_scene" or event.get("robot_involved") is not False:
        return None, "object_contact_raw_event_is_not_a_scene_scene_contact"
    if event.get("contact_position_status") != "available":
        return None, "object_contact_world_position_is_unavailable"
    if event.get("contact_normal_status") != "available":
        return None, "object_contact_world_normal_is_unavailable"
    try:
        position = _vector(event.get("first_contact_position_world"), 3, "object contact position")
        normal = _unit(
            event.get("first_contact_normal_geom1_to_geom2_world"),
            "object contact normal",
        )
        normal_geom_ids = tuple(int(value) for value in event["contact_normal_geom_ids"])
    except Exception:
        return None, "object_contact_world_position_or_normal_is_malformed"
    if normal is None or normal_geom_ids not in (geom_ids, geom_ids[::-1]):
        return None, "object_contact_normal_geom_order_disagrees_with_raw_event"
    # MuJoCo's contact frame first axis points from the first geom in
    # contact_normal_geom_ids toward the second. Move the rule subject away
    # from the target object, independent of event pair sort order.
    subject_is_normal_geom1 = normal_geom_ids[0] == subject_geom_id
    if not subject_is_normal_geom1 and normal_geom_ids[1] != subject_geom_id:
        return None, "object_contact_normal_does_not_reference_bound_subject_geom"
    away = -normal if subject_is_normal_geom1 else normal
    return {
        "violation_source": "object_contact_check",
        "contact_type": "object_to_object",
        "rule_id": check.get("rule_id"),
        "moving_object_id": check.get("subject_id"),
        "obstacle_object_id": check.get("target_id"),
        "moving_object_body_id": subject_body_id,
        "obstacle_object_body_id": target_body_id,
        "moving_object_semantic_geom_ids": subject_geom_ids,
        "obstacle_object_semantic_geom_ids": target_geom_ids,
        "moving_object_contact_geom_id": subject_geom_id,
        "obstacle_contact_geom_id": target_geom_id,
        "control_step": control_step,
        "physics_substep": physics_substep,
        "geom_pair": evidence["geom_pair"],
        "body_pair": evidence["body_pair"],
        "contact_position_world": position.tolist(),
        "contact_normal_geom1_to_geom2_world": normal.tolist(),
        "normal_geom_ids_order": list(normal_geom_ids),
        "normal_convention": "MuJoCo contact.frame first axis, first contact_normal_geom_ids geom toward second",
        "normal_away_from_obstacle": away.tolist(),
        "max_force_norm": evidence.get("max_force_norm"),
        "source_event": event,
    }, None


def locate_first_violation(
    branch_result: dict[str, Any],
    compiled_result: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Join compiler evidence to the richer substep contact record."""
    compiled = _compiled_candidate(compiled_result)
    if compiled is None or compiled.get("verdict") != "violation":
        return None, "baseline_has_no_explicit_compiled_violation"
    baseline = _candidate_with_id(branch_result, compiled.get("candidate_id"))
    if baseline is None:
        return None, "baseline_branch_candidate_is_missing"
    evidence = compiled.get("evidence")
    object_checks = compiled.get("object_contact_checks")
    has_object_violation = isinstance(object_checks, list) and any(
        isinstance(check, dict) and check.get("verdict") == "violation"
        and isinstance(check.get("evidence"), list) and check["evidence"]
        for check in object_checks
    )
    if (not isinstance(evidence, list) or not evidence) and not has_object_violation:
        return None, "compiled_violation_has_no_contact_evidence"
    gripper_ids = compiled_result.get("gripper_geom_ids")
    monitor = baseline.get("contact_monitoring")
    events = monitor.get("events") if isinstance(monitor, dict) else None
    if not isinstance(events, list):
        return None, "baseline_substep_contact_events_are_missing"
    violation_items: list[tuple[int, int, str, Any, Any]] = []
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                try:
                    violation_items.append((
                        int(item.get("first_control_step", 1 << 30)),
                        int(item.get("first_physics_substep", 1 << 30)),
                        "gripper",
                        item,
                        None,
                    ))
                except (TypeError, ValueError):
                    continue
    compiled_rules = compiled_result.get("compiled_object_contact_rules")
    if not isinstance(compiled_rules, list):
        compiled_rules = []
    if isinstance(object_checks, list):
        for check in object_checks:
            if not isinstance(check, dict) or check.get("verdict") != "violation":
                continue
            object_evidence = check.get("evidence")
            if not isinstance(object_evidence, list):
                continue
            for item in object_evidence:
                if not isinstance(item, dict):
                    continue
                try:
                    violation_items.append((
                        int(item.get("first_control_step", 1 << 30)),
                        int(item.get("first_physics_substep", 1 << 30)),
                        "object_object",
                        item,
                        check,
                    ))
                except (TypeError, ValueError):
                    continue
    if not violation_items:
        return None, "compiled_violation_has_no_parseable_contact_evidence"
    violation_items.sort(key=lambda item: (item[0], item[1]))
    if not isinstance(gripper_ids, list) or not gripper_ids:
        gripper_ids_set = set()
    else:
        try:
            gripper_ids_set = {int(item) for item in gripper_ids}
        except (TypeError, ValueError):
            gripper_ids_set = set()
    for evidence_control_step, evidence_physics_substep, evidence_kind, item, check in violation_items:
        if evidence_kind == "object_object":
            object_violation, object_error = _object_contact_violation(
                item, check, compiled_rules, events
            )
            if object_violation is not None:
                return object_violation, None
            return None, object_error or "object_contact_evidence_could_not_be_joined"
        if not gripper_ids_set:
            return None, "compiled_result_has_no_gripper_geometry_identity"
        geom_pair = item.get("geom_pair")
        ids = _pair_ids(geom_pair)
        if ids is None or sum(geom_id in gripper_ids_set for geom_id in ids) != 1:
            return None, "gripper_violation_evidence_has_invalid_geom_pair_or_identity"
        event = None
        for raw_event in events:
            if not isinstance(raw_event, dict) or _pair_ids(raw_event.get("geom_pair")) != ids:
                continue
            if raw_event.get("first_control_step") != evidence_control_step:
                continue
            if raw_event.get("first_physics_substep") == evidence_physics_substep:
                event = raw_event
                break
        if event is None:
            return None, "gripper_violation_evidence_has_no_matching_raw_contact_event"
        position = event.get("first_contact_position_world")
        normal = event.get("first_contact_normal_geom1_to_geom2_world")
        normal_geom_ids = event.get("contact_normal_geom_ids")
        if event.get("contact_position_status") != "available":
            return None, "gripper_violation_contact_position_is_unavailable"
        if event.get("contact_normal_status") != "available":
            return None, "gripper_violation_contact_normal_is_unavailable"
        try:
            position = _vector(position, 3, "contact position")
            normal = _unit(normal, "contact normal")
            normal_geom_ids = tuple(int(value) for value in normal_geom_ids)
        except Exception:
            return None, "gripper_violation_contact_position_or_normal_is_malformed"
        if normal is None or normal_geom_ids not in (ids, ids[::-1]):
            return None, "gripper_violation_normal_geom_order_disagrees_with_raw_event"
        robot_geom_id = next(geom_id for geom_id in ids if geom_id in gripper_ids_set)
        robot_is_geom1_in_normal = normal_geom_ids[0] == robot_geom_id
        # MuJoCo contact.frame's first axis points from raw geom1 toward raw geom2.
        away = -normal if robot_is_geom1_in_normal else normal
        return {
            "control_step": int(item["first_control_step"]),
            "physics_substep": int(item["first_physics_substep"]),
            "geom_pair": geom_pair,
            "body_pair": item.get("body_pair"),
            "robot_geom_id": robot_geom_id,
            "violation_source": "gripper_contact_evidence",
            "contact_type": "gripper_to_object",
            "contact_position_world": position.tolist(),
            "contact_normal_geom1_to_geom2_world": normal.tolist(),
            "normal_geom_ids_order": list(normal_geom_ids),
            "normal_convention": "MuJoCo contact.frame first axis, raw geom1 toward raw geom2",
            "normal_away_from_obstacle": away.tolist(),
            "max_force_norm": item.get("max_force_norm"),
            "source_event": event,
        }, None
    return None, "violation_lacks_joinable_world_contact_position_or_normal"


def _eef_positions(branch_candidate: Any) -> list[np.ndarray] | None:
    if not isinstance(branch_candidate, dict):
        return None
    start = branch_candidate.get("start_state")
    steps = branch_candidate.get("steps")
    try:
        first = _vector(start["robot_states"][0]["eef_pos"], 3, "start eef_pos")
        if not isinstance(steps, list):
            return None
        positions = [first]
        for step in steps:
            positions.append(
                _vector(step["state"]["robot_states"][0]["eef_pos"], 3, "step eef_pos")
            )
        return positions
    except Exception:
        return None


def _controller_fields(context: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(context, dict) or context.get("status") != "available":
        return None, "active_controller_context_is_unavailable"
    robots = context.get("robots")
    if not isinstance(robots, list) or len(robots) != 1 or not isinstance(robots[0], dict):
        return None, "repair_requires_one_resolved_robot_controller"
    robot = robots[0]
    if robot.get("status") != "available" or not isinstance(robot.get("fields"), dict):
        return None, "active_robot_controller_fields_are_unavailable"
    fields = robot["fields"]
    try:
        control_dim = int(fields["control_dim"])
        input_min = np.asarray(fields["input_min"], dtype=np.float64).reshape(-1)
        input_max = np.asarray(fields["input_max"], dtype=np.float64).reshape(-1)
        output_min = np.asarray(fields["output_min"], dtype=np.float64).reshape(-1)
        output_max = np.asarray(fields["output_max"], dtype=np.float64).reshape(-1)
        if (
            control_dim < 3
            or min(map(len, (input_min, input_max, output_min, output_max))) < 3
            or not all(np.isfinite(value).all() for value in (input_min, input_max, output_min, output_max))
            or np.any(input_max[:3] <= input_min[:3])
            or np.any(output_max[:3] <= output_min[:3])
        ):
            raise ValueError("controller bounds or dimensions are invalid")
        if fields.get("control_delta") is not True or fields.get("use_delta") is not True:
            raise ValueError("active OSC must use delta position commands")
        gain = (output_max[:3] - output_min[:3]) / (input_max[:3] - input_min[:3])
        if not np.isfinite(gain).all() or np.any(gain <= 0):
            raise ValueError("controller XYZ action scale is not positive and finite")
        up = _unit(context.get("world_up"), "world up")
        if up is None:
            raise ValueError("MuJoCo world up vector is unavailable")
    except Exception as exc:
        return None, f"active_controller_context_is_invalid:{type(exc).__name__}:{exc}"
    return {
        "robot": robot,
        "fields": fields,
        "input_min": input_min[:3],
        "input_max": input_max[:3],
        "output_min": output_min[:3],
        "output_max": output_max[:3],
        "gain_m_per_env_action": gain,
        "world_up": up,
    }, None


def _motion_direction(
    branch_candidate: dict[str, Any],
    actions: np.ndarray,
    violation_step: int,
    controller: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    positions = _eef_positions(branch_candidate)
    if positions is not None and violation_step > 0:
        # Use state checkpoints strictly before the violating control action;
        # the Bridge does not expose EEF pose at the exact physics substep.
        end_index = min(max(violation_step, 1), len(positions) - 1)
        start_index = max(0, end_index - 3)
        delta = positions[end_index] - positions[start_index]
        direction = _unit(delta, "observed EEF movement")
        if direction is not None:
            return direction, {
                "source": "observed_eef_motion_before_or_at_first_violation",
                "delta_world_m": delta.tolist(),
                "sample_start_step": start_index,
                "sample_end_step": end_index - 1,
            }
    lo = controller["input_min"]
    hi = controller["input_max"]
    end = min(max(violation_step, 0), len(actions) - 1)
    start = max(0, end - 2)
    commands = np.clip(actions[start : end + 1, :3], lo, hi) * controller[
        "gain_m_per_env_action"
    ]
    delta = commands.sum(axis=0) if len(commands) else np.zeros(3)
    direction = _unit(delta, "commanded EEF movement")
    if direction is None:
        return None, {"source": "unavailable", "reason": "no_observed_or_commanded_xyz_motion"}
    return direction, {
        "source": "decoded_action_and_active_controller_scale",
        "delta_world_m": delta.tolist(),
        "sample_start_step": start,
        "sample_end_step": end,
    }


def derive_repair_directions(
    violation: dict[str, Any],
    motion_direction: Any,
    world_up: Any,
    robot_eef_position_world: Any = None,
) -> list[dict[str, Any]]:
    """Build a normal, two surface tangents, and a gravity-derived lift vector."""
    away = _unit(violation.get("normal_away_from_obstacle"), "away normal")
    motion = _unit(motion_direction, "motion direction")
    up = _unit(world_up, "world up")
    if away is None or motion is None:
        return []
    candidates = [("contact_normal_away", away, "contact normal")]
    tangent = None
    tangent_source = None
    try:
        contact_position = _vector(
            violation.get("contact_position_world"), 3, "contact position"
        )
        eef_position = _vector(robot_eef_position_world, 3, "robot EEF position")
        radial = eef_position - contact_position
        radial -= float(np.dot(radial, away)) * away
        tangent = _unit(radial, "contact-point tangent")
        if tangent is not None:
            tangent_source = "EEF-to-contact-point offset projected onto contact tangent plane"
    except Exception:
        pass
    if tangent is None:
        tangent = _unit(np.cross(motion, away), "surface tangent")
        if tangent is not None:
            tangent_source = "cross product of observed motion and contact normal"
    if tangent is None and up is not None:
        tangent = _unit(np.cross(up, away), "fallback surface tangent")
        if tangent is not None:
            tangent_source = "cross product of gravity up and contact normal"
    if tangent is not None:
        candidates.extend((
            ("tangent_side_a", tangent, tangent_source),
            ("tangent_side_b", -tangent, tangent_source),
        ))
    if up is not None:
        candidates.append(("gravity_derived_lift", up, "negative normalized MuJoCo gravity"))
    result: list[dict[str, Any]] = []
    for name, direction, basis_source in candidates:
        if any(float(np.dot(direction, _unit(item["direction_world"], "prior direction"))) > 0.995 for item in result):
            continue
        result.append({"name": name, "direction_world": direction.tolist(), "basis_source": basis_source})
    return result


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def _window_envelope(length: int, violation_step: int, lead: int, tail: int) -> tuple[int, int, np.ndarray]:
    """Build a contact-centered envelope that starts steering before contact.

    The first action in the lead window must already have a non-zero weight;
    otherwise early violations can leave no anticipatory action to repair.
    """
    start = max(0, violation_step - max(0, int(lead)))
    end = min(length - 1, violation_step + max(0, int(tail)))
    envelope = np.zeros(length, dtype=np.float64)
    if start == end:
        envelope[start] = 1.0
        return start, end, envelope
    for index in range(start, end + 1):
        if index <= violation_step:
            span = violation_step - start
            # Include the lead action in the ramp. For a one-step lead this
            # gives weights [0.5, 1.0], rather than [0.0, 1.0].
            progress = 1.0 if span == 0 else (index - start + 1) / (span + 1)
            envelope[index] = _smoothstep(progress)
        else:
            span = end - violation_step
            progress = (index - violation_step) / span
            envelope[index] = 1.0 - _smoothstep(progress)
    if envelope[violation_step] == 0:
        envelope[violation_step] = 1.0
    return start, end, envelope


def build_repair_actions(
    original_actions: Any,
    direction_world: Any,
    amplitude_m_per_step: float,
    violation_step: int,
    lead_steps: int,
    tail_steps: int,
    controller_context: Any,
    config: RepairConfig = RepairConfig(),
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Add a smooth world-space XYZ velocity bias over a contact-centered window."""
    actions = np.asarray(original_actions, dtype=np.float64)
    direction = _unit(direction_world, "repair direction")
    controller, error = _controller_fields(controller_context)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        return None, {"valid": False, "reason": "original_actions_must_be_finite_(T,7)"}
    if direction is None or controller is None:
        return None, {"valid": False, "reason": error or "repair_direction_is_invalid"}
    if not np.isfinite(amplitude_m_per_step) or amplitude_m_per_step <= 0:
        return None, {"valid": False, "reason": "amplitude_must_be_positive_and_finite"}
    if amplitude_m_per_step > config.maximum_amplitude_m_per_step:
        return None, {"valid": False, "reason": "per_step_amplitude_exceeds_configured_bound"}
    if not 0 <= violation_step < len(actions):
        return None, {"valid": False, "reason": "violation_step_is_outside_action_chunk"}

    start, end, envelope = _window_envelope(len(actions), violation_step, lead_steps, tail_steps)
    active_steps = np.flatnonzero(envelope > 1e-8)
    if len(active_steps) < 2:
        return None, {
            "valid": False,
            "reason": "repair_window_must_modify_multiple_consecutive_actions",
            "window": {"start_step": start, "violation_step": violation_step, "end_step": end},
            "active_action_steps": active_steps.tolist(),
        }
    bias = envelope[:, None] * amplitude_m_per_step * direction[None, :]
    integrated = float(np.linalg.norm(bias, axis=1).sum())
    if integrated > config.maximum_integrated_detour_m + 1e-9:
        return None, {
            "valid": False,
            "reason": "integrated_detour_exceeds_configured_bound",
            "integrated_detour_m": integrated,
        }

    repaired = actions.copy()
    low, high = controller["input_min"], controller["input_max"]
    gain = controller["gain_m_per_env_action"]
    correction_action = np.zeros((len(actions), 3), dtype=np.float64)
    for step_index in range(start, end + 1):
        correction_action[step_index] = bias[step_index] / gain
        for axis in range(3):
            if abs(correction_action[step_index, axis]) <= 1e-12:
                continue
            effective_base = float(np.clip(actions[step_index, axis], low[axis], high[axis]))
            corrected = effective_base + correction_action[step_index, axis]
            if corrected < low[axis] - 1e-9 or corrected > high[axis] + 1e-9:
                return None, {
                    "valid": False,
                    "reason": "xyz_repair_would_saturate_controller_input",
                    "step_index": step_index,
                    "axis": axis,
                    "corrected_input": corrected,
                    "input_bounds": [float(low[axis]), float(high[axis])],
                    "integrated_detour_m": integrated,
                }
            repaired[step_index, axis] = corrected

    non_xyz_unchanged = bool(np.array_equal(repaired[:, 3:], actions[:, 3:]))
    effective_delta_m = (repaired[:, :3] - np.clip(actions[:, :3], low, high)) * gain
    max_slew = float(
        np.max(np.linalg.norm(np.diff(effective_delta_m, axis=0), axis=1))
    ) if len(actions) > 1 else 0.0
    diagnostics = {
        "valid": non_xyz_unchanged and max_slew <= config.maximum_action_correction_slew_m + 1e-9,
        "reason": None if non_xyz_unchanged and max_slew <= config.maximum_action_correction_slew_m + 1e-9 else (
            "non_xyz_action_changed" if not non_xyz_unchanged else "repair_action_correction_is_not_smooth"
        ),
        "direction_world": direction.tolist(),
        "amplitude_m_per_control_step": float(amplitude_m_per_step),
        "control_gain_m_per_env_action": gain.tolist(),
        "window": {"start_step": start, "violation_step": violation_step, "end_step": end},
        "active_action_steps": active_steps.tolist(),
        "envelope": envelope.tolist(),
        "integrated_detour_m": integrated,
        "physical_xyz_bias_by_step_m": bias.tolist(),
        "max_correction_slew_m": max_slew,
        "only_xyz_changed": non_xyz_unchanged,
        "requested_input_saturation_checked": True,
    }
    if not diagnostics["valid"]:
        return None, diagnostics
    return repaired, diagnostics


def _object_state_changes(
    branch_candidate: dict[str, Any], prefix_steps: int | None = None
) -> dict[str, dict[str, Any]]:
    start_state = branch_candidate.get("start_state", {}).get("object_states")
    steps = branch_candidate.get("steps")
    if not isinstance(start_state, dict) or not isinstance(steps, list):
        return {}
    if prefix_steps is None:
        if not steps:
            return {}
        end_checkpoint = steps[-1]
    else:
        if isinstance(prefix_steps, bool) or prefix_steps < 0 or prefix_steps > len(steps):
            return {}
        # A prefix of N actions ends at the state after action N-1; zero
        # actions means the initial Snapshot state.
        end_checkpoint = branch_candidate.get("start_state") if prefix_steps == 0 else steps[prefix_steps - 1]
    if prefix_steps == 0:
        end_state = end_checkpoint.get("object_states")
    else:
        end_state = end_checkpoint.get("state", {}).get("object_states")
    if not isinstance(end_state, dict):
        return {}
    changes = {}
    for key, start in start_state.items():
        end = end_state.get(key)
        if not isinstance(start, dict) or not isinstance(end, dict) or start.get("body_id") != end.get("body_id"):
            continue
        try:
            start_pos = _vector(start["root_pos"], 3, f"{key} start position")
            end_pos = _vector(end["root_pos"], 3, f"{key} end position")
            delta = end_pos - start_pos
            start_q = np.asarray(start.get("joint_qpos", []), dtype=np.float64).reshape(-1)
            end_q = np.asarray(end.get("joint_qpos", []), dtype=np.float64).reshape(-1)
            if start_q.shape != end_q.shape or not np.isfinite(start_q).all() or not np.isfinite(end_q).all():
                start_q = end_q = np.empty(0, dtype=np.float64)
            changes[key] = {
                "body_id": int(start["body_id"]),
                "root_translation_delta_world_m": delta.tolist(),
                "root_translation_distance_m": float(np.linalg.norm(delta)),
                "joint_qpos_delta": (end_q - start_q).tolist(),
                "joint_qpos_delta_norm": float(np.linalg.norm(end_q - start_q)),
            }
        except Exception:
            continue
    return changes


def _task_object_progress(
    baseline: dict[str, Any], candidate: dict[str, Any], task_object_binding: Any,
    config: RepairConfig, prefix_steps: int,
) -> dict[str, Any]:
    if not isinstance(task_object_binding, dict) or not isinstance(task_object_binding.get("body_id"), int):
        return {"status": "unavailable", "reason": "task_object_physical_binding_not_supplied"}
    body_id = task_object_binding["body_id"]
    base_changes = _object_state_changes(baseline, prefix_steps)
    candidate_changes = _object_state_changes(candidate, prefix_steps)
    base = next((value for value in base_changes.values() if value["body_id"] == body_id), None)
    actual = next((value for value in candidate_changes.values() if value["body_id"] == body_id), None)
    if base is None or actual is None:
        return {
            "status": "unavailable",
            "reason": "task_object_body_state_missing_from_aligned_prefix",
            "prefix_steps_compared": prefix_steps,
        }
    base_pos = _vector(base["root_translation_delta_world_m"], 3, "baseline task object delta")
    candidate_pos = _vector(actual["root_translation_delta_world_m"], 3, "candidate task object delta")
    baseline_distance = float(np.linalg.norm(base_pos))
    base_joint = np.asarray(base["joint_qpos_delta"], dtype=np.float64)
    candidate_joint = np.asarray(actual["joint_qpos_delta"], dtype=np.float64)
    joint_progress = None
    joint_distance = float(np.linalg.norm(base_joint))
    if base_joint.shape == candidate_joint.shape and joint_distance >= 1e-6:
        joint_progress = float(np.dot(candidate_joint, base_joint / joint_distance))
    joint_progress_fraction = (
        joint_progress / joint_distance if joint_progress is not None else None
    )
    if baseline_distance < config.minimum_object_motion_m:
        return {
            "status": "no_baseline_task_object_motion",
            "reason": "insufficient_baseline_task_object_translation",
            "body_id": body_id,
            "prefix_steps_compared": prefix_steps,
            "baseline_translation_m": baseline_distance,
            "baseline_joint_qpos_delta_norm": joint_distance,
            "candidate_joint_progress": joint_progress,
            "candidate_joint_progress_fraction": joint_progress_fraction,
            "joint_qpos_role": "diagnostic_only_not_used_for_progress_acceptance",
        }
    position_progress = float(np.dot(candidate_pos, base_pos / baseline_distance))
    required_translation_progress = config.required_object_progress_fraction * baseline_distance
    return {
        "status": "pass" if position_progress >= required_translation_progress else "fail",
        "body_id": body_id,
        "prefix_steps_compared": prefix_steps,
        "baseline_translation_m": baseline_distance,
        "candidate_translation_progress_m": position_progress,
        "candidate_translation_progress_fraction": position_progress / baseline_distance,
        "required_translation_progress_m": required_translation_progress,
        "baseline_joint_qpos_delta_norm": joint_distance,
        "candidate_joint_progress": joint_progress,
        "candidate_joint_progress_fraction": joint_progress_fraction,
        "joint_qpos_role": "diagnostic_only_not_used_for_progress_acceptance",
        "required_progress_fraction": config.required_object_progress_fraction,
    }


def evaluate_task_progress(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    original_actions: np.ndarray,
    violation_step: int,
    controller_context: Any,
    task_object_binding: Any = None,
    config: RepairConfig = RepairConfig(),
) -> dict[str, Any]:
    """Require task-specific evidence and compare motion/reward at equal prefixes."""
    baseline_steps = baseline.get("steps", [])
    candidate_steps = candidate.get("steps", [])
    prefix_steps = max(0, min(int(violation_step), len(original_actions)))
    baseline_reward = float(sum(float(step.get("reward", 0.0)) for step in baseline_steps))
    candidate_reward = float(sum(float(step.get("reward", 0.0)) for step in candidate_steps))
    baseline_prefix_reward = float(sum(float(step.get("reward", 0.0)) for step in baseline_steps[:prefix_steps]))
    candidate_prefix_reward = float(sum(float(step.get("reward", 0.0)) for step in candidate_steps[:prefix_steps]))
    baseline_positive_reward = any(float(step.get("reward", 0.0)) > 0 for step in baseline_steps)
    candidate_positive_reward = any(float(step.get("reward", 0.0)) > 0 for step in candidate_steps)
    done_steps = [bool(step.get("done", False)) for step in candidate_steps]
    positions_base = _eef_positions(baseline)
    positions_candidate = _eef_positions(candidate)
    controller, controller_error = _controller_fields(controller_context)
    eef_metric: dict[str, Any] = {"status": "unavailable", "reason": "eef_state_or_controller_missing"}
    if positions_base and positions_candidate and controller is not None:
        lo, hi = controller["input_min"], controller["input_max"]
        gain = controller["gain_m_per_env_action"]
        intent = np.clip(original_actions[:prefix_steps, :3], lo, hi) * gain
        intent_vector = intent.sum(axis=0) if len(intent) else np.zeros(3)
        intent_direction = _unit(intent_vector, "pre-violation intent")
        if intent_direction is None:
            baseline_end_index = min(prefix_steps, len(positions_base) - 1)
            intent_direction = _unit(positions_base[baseline_end_index] - positions_base[0], "baseline EEF movement")
        if intent_direction is not None:
            baseline_end_index = min(prefix_steps, len(positions_base) - 1)
            baseline_progress = float(
                np.dot(positions_base[baseline_end_index] - positions_base[0], intent_direction)
            )
            candidate_end_index = min(prefix_steps, len(positions_candidate) - 1)
            candidate_progress = float(
                np.dot(positions_candidate[candidate_end_index] - positions_candidate[0], intent_direction)
            )
            candidate_distance = float(np.linalg.norm(positions_candidate[candidate_end_index] - positions_candidate[0]))
            required = max(
                config.minimum_eef_progress_m,
                max(0.0, baseline_progress) * config.required_baseline_progress_fraction,
            )
            eef_metric = {
                "status": (
                    "not_required" if baseline_progress < config.minimum_eef_progress_m
                    else "pass" if candidate_progress >= required else "fail"
                ),
                "intent_direction_world": intent_direction.tolist(),
                "prefix_steps_compared": prefix_steps,
                "baseline_previolation_forward_progress_m": baseline_progress,
                "candidate_forward_progress_m": candidate_progress,
                "candidate_net_eef_displacement_m": candidate_distance,
                "required_forward_progress_m": required,
                "required_baseline_progress_fraction": config.required_baseline_progress_fraction,
                "role": "local trajectory diagnostic only; never sufficient evidence of task progress",
                "proxy_limitation": "Tracks same-prefix displacement along A0's pre-violation XYZ intent, not task completion semantics.",
            }
        else:
            eef_metric = {
                "status": "not_required",
                "reason": "no_previolation_eef_motion_to_preserve",
                "prefix_steps_compared": prefix_steps,
                "role": "local trajectory diagnostic only; never sufficient evidence of task progress",
            }
    elif controller_error:
        eef_metric = {"status": "unavailable", "reason": controller_error}

    task_object_metric = _task_object_progress(
        baseline, candidate, task_object_binding, config, prefix_steps
    )
    baseline_object_changes = _object_state_changes(baseline)
    all_object_changes = _object_state_changes(candidate)
    if baseline_positive_reward:
        passed = candidate_reward + 1e-9 >= baseline_reward
        basis = "preserved_positive_libero_reward"
        reason = None if passed else "candidate_reward_below_baseline_success_reward"
    elif candidate_positive_reward:
        passed = True
        basis = "candidate_positive_libero_reward"
        reason = None
    elif any(done_steps):
        passed = False
        basis = "terminal_without_success_reward"
        reason = "candidate_terminated_without_positive_success_reward"
    else:
        object_pass = task_object_metric.get("status") == "pass"
        if object_pass:
            passed = True
            basis = "time_aligned_task_object_state_proxy"
            reason = None
        elif task_object_binding is not None and task_object_metric.get("status") in ("fail", "unavailable"):
            passed = False
            basis = "insufficient_task_object_progress_evidence"
            reason = "task_object_progress_failed_or_unavailable_in_aligned_prefix"
        else:
            passed = False
            basis = "no_task_specific_progress_evidence"
            reason = "no_task_specific_progress_evidence"
    prefix_reward_pass = (
        baseline_prefix_reward <= 0.0
        or candidate_prefix_reward + 1e-9 >= baseline_prefix_reward
    )
    # A positive reward in the A0 prefix must be preserved at the same
    # control-step horizon; matching only the full-chunk total is insufficient.
    if not prefix_reward_pass:
        passed = False
        reason = "candidate_did_not_preserve_aligned_prefix_reward"
    return {
        "status": "pass" if passed else "fail",
        "reason": reason,
        "basis": basis,
        "baseline_reward_sum": baseline_reward,
        "candidate_reward_sum": candidate_reward,
        "aligned_prefix_steps_compared": prefix_steps,
        "baseline_aligned_prefix_reward_sum": baseline_prefix_reward,
        "candidate_aligned_prefix_reward_sum": candidate_prefix_reward,
        "aligned_prefix_reward_preserved": prefix_reward_pass,
        "reward_alignment": "A0 and candidate prefix rewards both cover actions [0, first_violation_control_step); full-chunk reward is reported separately.",
        "candidate_done_by_step": done_steps,
        "eef_forward_progress": eef_metric,
        "task_object_progress": task_object_metric,
        "observed_baseline_object_state_changes_by_key": baseline_object_changes,
        "observed_candidate_object_state_changes_by_key": all_object_changes,
        "task_object_binding": task_object_binding,
        "limitations": [
            "Short-chunk LIBERO reward is sparse; done is reported but is not treated alone as success.",
            "EEF progress is only a local same-prefix motion-preservation constraint and never task-success evidence.",
            "Object-state progress is usable only when a target_object is physically bound, and compares equal-length pre-violation prefixes.",
        ],
    }


def _workspace_and_state_constraints(
    repaired_actions: np.ndarray,
    branch_candidate: dict[str, Any],
    controller: dict[str, Any],
    config: RepairConfig,
) -> dict[str, Any]:
    robot = controller["robot"]
    fields = controller["fields"]
    positions = _eef_positions(branch_candidate)
    steps = branch_candidate.get("steps")
    if positions is None or not isinstance(steps, list) or len(positions) != len(steps) + 1 or len(steps) > len(repaired_actions):
        return {"status": "fail", "reason": "candidate_eef_state_path_is_incomplete"}
    for position in positions:
        if not np.isfinite(position).all():
            return {"status": "fail", "reason": "candidate_eef_position_is_non_finite"}

    position_limits_raw = fields.get("position_limits")
    position_limits = None
    action_low, action_high = controller["input_min"], controller["input_max"]
    gain = controller["gain_m_per_env_action"]
    if position_limits_raw is not None:
        try:
            position_limits = np.asarray(position_limits_raw, dtype=np.float64)
            if position_limits.shape != (2, 3) or not np.isfinite(position_limits).all():
                raise ValueError("OSC position_limits must have shape (2, 3)")
        except Exception as exc:
            return {"status": "fail", "reason": f"active_position_limits_are_invalid:{exc}"}

    target_violations = []
    if position_limits is not None:
        low, high = position_limits
        for index, action in enumerate(repaired_actions[: len(steps)]):
            effective = np.clip(action[:3], action_low, action_high) * gain
            target = positions[index] + effective
            if np.any(target < low - config.workspace_tolerance) or np.any(target > high + config.workspace_tolerance):
                target_violations.append({"step_index": index, "requested_goal_world": target.tolist()})

    joint_ranges = robot.get("joint_ranges")
    joint_limited = robot.get("joint_limited")
    joint_violations = []
    if joint_ranges is None or joint_limited is None:
        return {
            "status": "unavailable",
            "reason": "active_robot_joint_limits_are_unavailable",
            "controller_position_limits": position_limits.tolist() if position_limits is not None else None,
        }
    joint_ranges = np.asarray(joint_ranges, dtype=np.float64)
    joint_limited = np.asarray(joint_limited, dtype=bool)
    for step_index, step in enumerate(steps):
        try:
            qpos = np.asarray(step["state"]["robot_states"][0]["joint_qpos"], dtype=np.float64)
            if qpos.shape != (len(joint_ranges),) or not np.isfinite(qpos).all():
                raise ValueError("joint_qpos shape or values are invalid")
            for joint_index, is_limited in enumerate(joint_limited):
                if not is_limited:
                    continue
                low, high = joint_ranges[joint_index]
                if qpos[joint_index] < low - config.workspace_tolerance or qpos[joint_index] > high + config.workspace_tolerance:
                    joint_violations.append({
                        "step_index": step_index,
                        "joint_index": joint_index,
                        "qpos": float(qpos[joint_index]),
                        "range": [float(low), float(high)],
                    })
        except Exception as exc:
            return {"status": "unavailable", "reason": f"candidate_joint_state_is_invalid:{exc}"}
    return {
        "status": "fail" if target_violations or joint_violations else "pass",
        "reason": "workspace_or_joint_limit_violated" if target_violations or joint_violations else None,
        "controller_position_limits": position_limits.tolist() if position_limits is not None else None,
        "controller_goal_limit_violations": target_violations,
        "robot_joint_limit_violations": joint_violations,
        "joint_limit_source": "active compiled MuJoCo model",
        "cartesian_limit_source": "active OSC controller" if position_limits is not None else "none_configured; actual EEF path checked by branch simulation",
    }


def _candidate_constraints(
    original_actions: np.ndarray,
    repaired_actions: np.ndarray,
    diagnostics: dict[str, Any],
    branch_candidate: dict[str, Any],
    controller: dict[str, Any],
    config: RepairConfig,
) -> dict[str, Any]:
    if repaired_actions.shape != original_actions.shape or not np.isfinite(repaired_actions).all():
        return {"status": "fail", "reason": "candidate_action_shape_or_finiteness_invalid"}
    if not diagnostics.get("only_xyz_changed") or not np.array_equal(
        repaired_actions[:, 3:], original_actions[:, 3:]
    ):
        return {"status": "fail", "reason": "rotation_or_gripper_actions_changed"}
    low, high = controller["input_min"], controller["input_max"]
    original_effective = np.clip(original_actions[:, :3], low, high)
    repaired_effective = np.clip(repaired_actions[:, :3], low, high)
    physical_corrections = (repaired_effective - original_effective) * controller[
        "gain_m_per_env_action"
    ]
    expected_corrections = np.asarray(
        diagnostics.get("physical_xyz_bias_by_step_m"), dtype=np.float64
    )
    if expected_corrections.shape != physical_corrections.shape or not np.allclose(
        physical_corrections, expected_corrections, atol=1e-8, rtol=1e-6
    ):
        return {"status": "fail", "reason": "controller_input_clipping_changed_requested_xyz_repair"}
    changed = np.abs(repaired_actions[:, :3] - original_actions[:, :3]) > 1e-10
    if np.any(changed & ((repaired_actions[:, :3] < low - 1e-9) | (repaired_actions[:, :3] > high + 1e-9))):
        return {"status": "fail", "reason": "repaired_xyz_action_outside_controller_input_bounds"}
    max_change = float(np.max(np.linalg.norm(physical_corrections, axis=1)))
    integrated = float(np.linalg.norm(physical_corrections, axis=1).sum())
    slew = float(np.max(np.linalg.norm(np.diff(physical_corrections, axis=0), axis=1))) if len(repaired_actions) > 1 else 0.0
    if max_change > config.maximum_amplitude_m_per_step + 1e-8:
        return {"status": "fail", "reason": "per_step_xyz_repair_exceeds_bound", "maximum_step_m": max_change}
    if integrated > config.maximum_integrated_detour_m + 1e-8:
        return {"status": "fail", "reason": "integrated_xyz_repair_exceeds_bound", "integrated_detour_m": integrated}
    if slew > config.maximum_action_correction_slew_m + 1e-8:
        return {"status": "fail", "reason": "xyz_repair_is_not_smooth", "correction_slew_m": slew}
    workspace = _workspace_and_state_constraints(
        repaired_actions, branch_candidate, controller, config
    )
    if workspace.get("status") != "pass":
        return {"status": workspace.get("status", "fail"), "reason": workspace.get("reason"), "workspace": workspace}
    return {
        "status": "pass",
        "max_xyz_correction_m_per_step": max_change,
        "integrated_xyz_correction_m": integrated,
        "xyz_correction_slew_m": slew,
        "only_xyz_changed": True,
        "workspace": workspace,
    }


def _initial_amplitude(
    baseline: dict[str, Any],
    violation_step: int,
    actions: np.ndarray,
    controller: dict[str, Any],
    config: RepairConfig,
) -> float:
    positions = _eef_positions(baseline)
    observed = []
    if positions is not None and violation_step > 0:
        end = min(max(violation_step, 1), len(positions) - 1)
        start = max(0, end - 3)
        observed = [float(np.linalg.norm(positions[index + 1] - positions[index])) for index in range(start, end)]
    observed_scale = float(np.median(observed)) if observed else 0.0
    if observed_scale > 1e-5:
        scale = observed_scale
    else:
        end = min(max(violation_step, 0), len(actions) - 1)
        commands = np.clip(
            actions[max(0, end - 2) : end + 1, :3],
            controller["input_min"],
            controller["input_max"],
        ) * controller["gain_m_per_env_action"]
        scale = float(np.median(np.linalg.norm(commands, axis=1))) if len(commands) else 0.0
    return float(np.clip(scale * 0.5, config.minimum_amplitude_m, min(0.012, config.maximum_amplitude_m_per_step)))


def _task_binding_error(binding: Any) -> str | None:
    if binding is None:
        return None
    if not isinstance(binding, dict) or isinstance(binding.get("body_id"), bool) or not isinstance(binding.get("body_id"), int):
        return "task_object_binding_has_no_valid_body_id"
    if binding.get("status") != "matched_by_iou" or binding.get("role") != "target_object":
        return "task_object_must_have_resolved_target_object_semantics_and_physical_binding"
    return None


def _config_error(config: RepairConfig) -> str | None:
    numeric_positive = (
        config.minimum_amplitude_m,
        config.maximum_amplitude_m_per_step,
        config.maximum_integrated_detour_m,
        config.minimum_refinement_m,
        config.minimum_eef_progress_m,
        config.maximum_action_correction_slew_m,
    )
    if any(not np.isfinite(value) or value <= 0 for value in numeric_positive):
        return "repair_config_positive_bounds_must_be_finite_and_greater_than_zero"
    if config.maximum_amplitude_m_per_step < config.minimum_amplitude_m:
        return "repair_config_maximum_amplitude_is_below_minimum"
    if not np.isfinite(config.amplitude_growth) or config.amplitude_growth <= 1.0:
        return "repair_config_amplitude_growth_must_be_greater_than_one"
    if config.max_candidate_evaluations < 1 or config.amplitude_refinement_iterations < 0:
        return "repair_config_candidate_budget_or_refinement_count_is_invalid"
    if not config.windows or any(lead < 0 or tail < 0 for lead, tail in config.windows):
        return "repair_config_windows_must_be_nonempty_and_nonnegative"
    if not 0 <= config.required_baseline_progress_fraction <= 1:
        return "repair_config_baseline_progress_fraction_must_be_in_[0,1]"
    if not 0 <= config.required_object_progress_fraction <= 1:
        return "repair_config_object_progress_fraction_must_be_in_[0,1]"
    return None


def run_verifier_guided_repair(
    original_actions: Any,
    baseline_branch_result: dict[str, Any],
    baseline_compiled_result: dict[str, Any],
    controller_context: dict[str, Any],
    evaluate_candidates: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    task_object_binding: dict[str, Any] | None = None,
    config: RepairConfig = RepairConfig(),
) -> dict[str, Any]:
    """Run baseline gates, adaptive expansion, then amplitude refinement.

    ``evaluate_candidates`` must execute every supplied action chunk from the
    same Snapshot and return records keyed by ``candidate_id`` containing the
    Bridge branch result and Contact Compiler result for that candidate.
    """
    actions = np.asarray(original_actions, dtype=np.float64)
    report: dict[str, Any] = {
        "algorithm": "verifier_guided_world_xyz_contact_repair_v1",
        "config": asdict(config),
        "original_action_chunk_A0": actions.tolist() if actions.ndim == 2 else original_actions,
        "repaired_action_chunk_A_plus": None,
        "status": "unknown",
        "search_attempts": [],
        "rejected_reasons": [],
        "supervision": {
            "requested_action_steps": int(len(actions)) if actions.ndim == 2 else None,
            "valid_action_steps": 0,
            "training_label_valid": False,
            "reason": "no_fully_validated_candidate",
        },
    }
    invalid_config = _config_error(config)
    if invalid_config is not None:
        report["status"] = "rejected"
        report["failure_reason"] = invalid_config
        return report
    if actions.ndim != 2 or actions.shape[1:] != (7,) or len(actions) == 0 or not np.isfinite(actions).all():
        report["status"] = "rejected"
        report["failure_reason"] = "original_action_chunk_must_be_finite_(T,7)"
        return report
    if baseline_branch_result.get("main_state_restored") is not True:
        report["status"] = "aborted_main_state_restore_failed"
        report["failure_reason"] = "baseline_branch_did_not_confirm_main_state_restore"
        return report
    baseline = _candidate_with_id(baseline_branch_result)
    compiled_baseline = _compiled_candidate(baseline_compiled_result)
    if baseline is None or compiled_baseline is None:
        report["status"] = "unknown"
        report["failure_reason"] = "baseline_branch_or_compiler_result_missing"
        return report
    report["baseline_verdict"] = compiled_baseline.get("verdict")
    if compiled_baseline.get("verdict") == "safe":
        report["status"] = "skipped_original_safe"
        report["failure_reason"] = "A0_is_already_safe_under_complete_compiler_monitoring"
        return report
    if compiled_baseline.get("verdict") != "violation":
        report["status"] = "skipped_unknown"
        report["failure_reason"] = compiled_baseline.get("reason", "baseline_compiler_verdict_is_unknown")
        return report
    violation, violation_error = locate_first_violation(
        baseline_branch_result, baseline_compiled_result
    )
    report["first_violation"] = violation
    if violation is None:
        report["status"] = "skipped_unknown"
        report["failure_reason"] = violation_error
        return report
    controller, controller_error = _controller_fields(controller_context)
    if controller is None:
        report["status"] = "skipped_unknown"
        report["failure_reason"] = controller_error
        return report
    binding_error = _task_binding_error(task_object_binding)
    if binding_error:
        report["status"] = "skipped_unknown"
        report["failure_reason"] = binding_error
        return report

    motion, motion_info = _motion_direction(
        baseline, actions, violation["control_step"], controller
    )
    report["robot_motion_direction"] = motion_info
    baseline_positions = _eef_positions(baseline)
    eef_at_violation = (
        baseline_positions[min(violation["control_step"] + 1, len(baseline_positions) - 1)]
        if baseline_positions
        else None
    )
    if eef_at_violation is not None:
        violation["robot_eef_position_world_after_violation_control_step"] = eef_at_violation.tolist()
        violation["contact_to_eef_vector_world"] = (
            eef_at_violation - np.asarray(violation["contact_position_world"], dtype=np.float64)
        ).tolist()
    directions = derive_repair_directions(
        violation, motion, controller_context.get("world_up"), eef_at_violation
    )
    report["repair_directions"] = directions
    if not directions:
        report["status"] = "skipped_unknown"
        report["failure_reason"] = "no_geometrically_supported_repair_direction"
        return report

    initial = _initial_amplitude(baseline, violation["control_step"], actions, controller, config)
    report["initial_amplitude_m_per_control_step"] = initial
    report["amplitude_semantics"] = "world-space OSC XYZ displacement bias per affected control step"
    report["search"] = {"expansion_factor": config.amplitude_growth, "amplitude_rounds": [], "refinement_rounds": []}

    configurations = []
    seen = set()
    for direction in directions:
        for lead, tail in config.windows:
            start, end, envelope = _window_envelope(
                len(actions), violation["control_step"], lead, tail
            )
            key = (direction["name"], start, end)
            if key in seen or end <= start:
                continue
            seen.add(key)
            configurations.append({
                "direction": direction,
                "lead_steps": lead,
                "tail_steps": tail,
                "start_step": start,
                "end_step": end,
                "envelope": envelope,
                "last_unsafe_amplitude": 0.0,
                "safe_amplitude": None,
                "latest_safe_record": None,
                "expansion_round": 0,
                "refinement_round": 0,
            })

    candidate_evaluations = 0
    candidate_counter = 0
    restore_failed = False

    def evaluate_specs(specs: list[tuple[dict[str, Any], float, np.ndarray, dict[str, Any]]]) -> list[dict[str, Any]]:
        nonlocal candidate_evaluations, candidate_counter, restore_failed
        executable = []
        local_records = []
        for config_item, amplitude, candidate_actions, diagnostics in specs:
            candidate_counter += 1
            candidate_id = f"repair-{candidate_counter:04d}"
            local = {
                "candidate_id": candidate_id,
                "direction": config_item["direction"]["name"],
                "direction_world": config_item["direction"]["direction_world"],
                "amplitude_m_per_control_step": amplitude,
                "window": diagnostics.get("window"),
                "action_diagnostics": diagnostics,
                "actions": candidate_actions.tolist(),
            }
            executable.append({
                "candidate_id": candidate_id,
                "actions": candidate_actions.tolist(),
                "metadata": local,
            })
            local_records.append(local)

        if not executable:
            return []
        if candidate_evaluations + len(executable) > config.max_candidate_evaluations:
            executable = executable[: max(0, config.max_candidate_evaluations - candidate_evaluations)]
            local_records = local_records[: len(executable)]
        if not executable:
            return []
        candidate_evaluations += len(executable)
        try:
            results = evaluate_candidates(executable)
        except Exception as exc:
            restore_failed = True
            report["rejected_reasons"].append(f"candidate_evaluator_error:{type(exc).__name__}:{exc}")
            for local in local_records:
                local["verdict"] = "unknown"
                local["reason"] = "candidate_evaluator_error"
                report["search_attempts"].append(local)
            return []
        result_by_id = {
            result.get("candidate_id"): result
            for result in results
            if isinstance(result, dict) and result.get("candidate_id") is not None
        } if isinstance(results, list) else {}
        accepted_records = []
        for local in local_records:
            result = result_by_id.get(local["candidate_id"])
            if not isinstance(result, dict):
                local.update({"verdict": "unknown", "reason": "candidate_evaluator_omitted_result"})
                report["search_attempts"].append(local)
                continue
            branch_result = result.get("branch_result")
            compiled_result = result.get("compiled_result")
            local["branch_result"] = branch_result
            local["compiled_result"] = compiled_result
            if not isinstance(branch_result, dict) or branch_result.get("main_state_restored") is not True:
                restore_failed = True
                local.update({"verdict": "unknown", "reason": "candidate_main_state_restore_failed"})
                report["search_attempts"].append(local)
                continue
            candidate_branch = _candidate_with_id(branch_result, local["candidate_id"])
            compiled_branch = _compiled_candidate(compiled_result, local["candidate_id"])
            if candidate_branch is None or compiled_branch is None:
                local.update({"verdict": "unknown", "reason": "candidate_branch_or_compiler_result_missing"})
                report["search_attempts"].append(local)
                continue
            branch_steps = candidate_branch.get("steps")
            branch_steps_len = len(branch_steps) if isinstance(branch_steps, list) else 0
            requested_steps = candidate_branch.get("requested_steps")
            executed_steps = candidate_branch.get("executed_steps")
            if isinstance(requested_steps, bool) or not isinstance(requested_steps, int):
                requested_steps = -1
            if isinstance(executed_steps, bool) or not isinstance(executed_steps, int):
                executed_steps = -1
            observed_steps = min(len(actions), max(0, executed_steps), branch_steps_len)
            local["execution_coverage"] = {
                "requested_action_steps": int(len(actions)),
                "bridge_requested_steps": requested_steps,
                "bridge_executed_steps": executed_steps,
                "branch_step_records": branch_steps_len,
                "observed_action_steps": observed_steps,
                "compiler_verdict": compiled_branch.get("verdict"),
                "training_label_valid": False,
            }
            if (
                requested_steps != len(actions)
                or executed_steps != len(actions)
                or branch_steps_len != len(actions)
            ):
                local.update({
                    "verdict": "rejected_incomplete_chunk",
                    "reason": "candidate_executed_incomplete_action_chunk",
                    "candidate_steps_with_state_records": observed_steps,
                    "training_label_valid": False,
                })
                report["search_attempts"].append(local)
                continue
            local["safety_verdict"] = compiled_branch.get("verdict")
            if compiled_branch.get("verdict") != "safe":
                local.update({"verdict": "rejected", "reason": "candidate_not_compiler_safe"})
                report["search_attempts"].append(local)
                continue
            candidate_actions = np.asarray(local["actions"], dtype=np.float64)
            local["action_constraints"] = _candidate_constraints(
                actions, candidate_actions, local["action_diagnostics"],
                candidate_branch, controller, config
            )
            if local["action_constraints"].get("status") != "pass":
                local.update({"verdict": "rejected", "reason": "candidate_action_constraints_failed"})
                report["search_attempts"].append(local)
                continue
            local["task_progress"] = evaluate_task_progress(
                baseline,
                candidate_branch,
                actions,
                violation["control_step"],
                controller_context,
                task_object_binding,
                config,
            )
            if local["task_progress"].get("status") != "pass":
                local.update({"verdict": "rejected", "reason": local["task_progress"].get("reason")})
                report["search_attempts"].append(local)
                continue
            local["verdict"] = "accepted"
            local["reason"] = "complete_contact_monitoring_safe_progress_preserved_constraints_pass"
            local["execution_coverage"].update({
                "validated_action_steps": len(actions),
                "training_label_valid": True,
            })
            report["search_attempts"].append(local)
            accepted_records.append(local)
        return accepted_records

    def create_specs(active: list[dict[str, Any]], amplitude_for) -> list[tuple[dict[str, Any], float, np.ndarray, dict[str, Any]]]:
        specs = []
        for item in active:
            amplitude = float(amplitude_for(item))
            repaired, diagnostics = build_repair_actions(
                actions,
                item["direction"]["direction_world"],
                amplitude,
                violation["control_step"],
                item["lead_steps"],
                item["tail_steps"],
                controller_context,
                config,
            )
            if repaired is None:
                report["search_attempts"].append({
                    "verdict": "rejected_before_branch",
                    "reason": diagnostics.get("reason"),
                    "direction": item["direction"]["name"],
                    "direction_world": item["direction"]["direction_world"],
                    "amplitude_m_per_control_step": amplitude,
                    "window": {"start_step": item["start_step"], "end_step": item["end_step"]},
                    "action_diagnostics": diagnostics,
                })
                continue
            specs.append((item, amplitude, repaired, diagnostics))
        return specs

    active = configurations[:]
    expansion_round = 0
    while active and candidate_evaluations < config.max_candidate_evaluations:
        # Clamp the geometric schedule to the legal maximum so a sequence
        # such as 0.012, 0.024, 0.048 still evaluates the 0.040 upper bound.
        scale = min(
            initial * (config.amplitude_growth ** expansion_round),
            config.maximum_amplitude_m_per_step,
        )
        report["search"]["amplitude_rounds"].append(scale)
        specs = create_specs(active, lambda _item: scale)
        accepted = evaluate_specs(specs)
        accepted_by_key = {
            (record["direction"], record["window"]["start_step"], record["window"]["end_step"]): record
            for record in accepted
        }
        next_active = []
        for item in active:
            key = (item["direction"]["name"], item["start_step"], item["end_step"])
            record = accepted_by_key.get(key)
            if record is not None:
                item["safe_amplitude"] = scale
                item["latest_safe_record"] = record
                item["expansion_round"] = expansion_round
            else:
                item["last_unsafe_amplitude"] = scale
                if scale < config.maximum_amplitude_m_per_step - 1e-9:
                    next_active.append(item)
        active = next_active
        expansion_round += 1
        if restore_failed:
            report["status"] = "aborted_main_state_restore_failed"
            report["failure_reason"] = "branch_rollout_failed_to_restore_main_state_during_search"
            return report

    refinable = [item for item in configurations if item["safe_amplitude"] is not None]
    for refinement_index in range(config.amplitude_refinement_iterations):
        if candidate_evaluations >= config.max_candidate_evaluations:
            break
        refinable = [
            item for item in refinable
            if item["safe_amplitude"] - item["last_unsafe_amplitude"] > config.minimum_refinement_m
        ]
        if not refinable:
            break
        specs = create_specs(
            refinable,
            lambda item: (item["safe_amplitude"] + item["last_unsafe_amplitude"]) / 2.0,
        )
        report["search"]["refinement_rounds"].append(refinement_index + 1)
        accepted = evaluate_specs(specs)
        accepted_by_key = {
            (record["direction"], record["window"]["start_step"], record["window"]["end_step"]): record
            for record in accepted
        }
        tested_by_key = {
            (spec[0]["direction"]["name"], spec[0]["start_step"], spec[0]["end_step"]): spec[1]
            for spec in specs
        }
        for item in refinable:
            key = (item["direction"]["name"], item["start_step"], item["end_step"])
            tested_amplitude = tested_by_key.get(key)
            if tested_amplitude is None:
                continue
            record = accepted_by_key.get(key)
            if record is not None:
                item["safe_amplitude"] = tested_amplitude
                item["latest_safe_record"] = record
            else:
                item["last_unsafe_amplitude"] = tested_amplitude
        if restore_failed:
            report["status"] = "aborted_main_state_restore_failed"
            report["failure_reason"] = "branch_rollout_failed_to_restore_main_state_during_refinement"
            return report

    safe_records = [item["latest_safe_record"] for item in configurations if item["latest_safe_record"] is not None]
    if not safe_records:
        report["status"] = "no_safe_progress_preserving_candidate"
        report["failure_reason"] = "adaptive_direction_window_and_amplitude_search_found_no_acceptable_candidate"
        return report

    def correction_cost(record: dict[str, Any]) -> tuple[float, float, float]:
        candidate_actions = np.asarray(record["actions"], dtype=np.float64)
        baseline_effective = np.clip(actions[:, :3], controller["input_min"], controller["input_max"])
        candidate_effective = np.clip(candidate_actions[:, :3], controller["input_min"], controller["input_max"])
        delta_m = (candidate_effective - baseline_effective) * controller["gain_m_per_env_action"]
        return (
            float(np.linalg.norm(delta_m)),
            float(np.linalg.norm(record["task_progress"]["eef_forward_progress"].get("candidate_forward_progress_m", 0.0))),
            float(record["amplitude_m_per_control_step"]),
        )

    chosen = min(safe_records, key=correction_cost)
    report.update({
        "status": "repaired_candidate_found",
        "failure_reason": None,
        "repaired_action_chunk_A_plus": chosen["actions"],
        "selected_candidate_id": chosen["candidate_id"],
        "selected_candidate": chosen,
        "selection_objective": "minimum L2 world-XYZ action correction among safety-, progress-, and constraint-passing branches",
        "candidate_evaluations": candidate_evaluations,
        "supervision": {
            "requested_action_steps": len(actions),
            "valid_action_steps": len(actions),
            "training_label_valid": True,
            "candidate_id": chosen["candidate_id"],
            "reason": "complete_chunk_safety_task_progress_and_action_constraints_validated",
        },
    })
    return report

