"""Measure black-bowl/perfume clearance along the archived S70 action chunk.

The S70 forecast uses only the current S70 state and archived A0_70. The S75
On-Policy compiler record is read afterward as an independent historical label.
MuJoCo distances are exploratory geometry estimates, never safety proofs.
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.smoke.contact_safety_compiler_smoke import AUTHKEY, BRIDGE_PORT
from research.gs_opd.smoke.replay_archive_contact_smoke import (
    CAMERA_KEYS,
    STATE_KEYS,
    _decode_bridge_image,
    _validate_current_bindings,
    compare_observation,
    load_archive_episode,
)
from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


DEFAULT_ARCHIVE = Path("logs/gs_opd/on_policy_safety_l0t0_object_v2_80step.npz")
DEFAULT_TEACHER = Path("logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt")
DEFAULT_BINDINGS = Path("logs/gs_opd/semantic_bindings_l0t0.json")
DEFAULT_OUT = Path("logs/gs_opd/near_far_gap_s65_s75.json")
STEP_65 = 65
STEP_70 = 70
STEP_75 = 75
CHUNK_LENGTH = STEP_75 - STEP_70
STATE_ATOL = 1e-6
GAP_THRESHOLDS_M = (0.005, 0.010, 0.020)
TREND_EPSILON_M = 1e-9


class NearFarArchiveError(ValueError):
    """Raised when archived actions or observations cannot support this probe."""


def _bridge_request(connection: Any, request: dict[str, Any]) -> dict[str, Any]:
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError(f"Bridge returned a non-object response for {request.get('op')!r}")
    return response


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read {label} JSON {path}: {exc}") from exc


def _load_decision_observation(archive_path: Path, decision_step: int) -> dict[str, Any]:
    with np.load(archive_path, allow_pickle=False) as archive:
        decision_steps = np.asarray(archive["decision_step_indices"], dtype=np.int64)
        indexes = np.flatnonzero(decision_steps == decision_step)
        if len(indexes) != 1:
            raise NearFarArchiveError(
                f"Expected exactly one decision observation at S{decision_step}; found {len(indexes)}"
            )
        index = int(indexes[0])
        metadata = json.loads(str(archive["metadata_json"]))
        instruction = str(archive["decision_instructions"][index])
        if not instruction or instruction != metadata.get("instruction"):
            raise NearFarArchiveError(f"S{decision_step} instruction disagrees with archive metadata")
        state = {
            key: np.asarray(archive[f"decision_{key}"][index], dtype=np.float64).copy()
            for key in STATE_KEYS
        }
        images = {
            key: np.asarray(archive[f"decision_{key}"][index]).copy()
            for key in CAMERA_KEYS
        }
        for key, value in state.items():
            if not np.isfinite(value).all():
                raise NearFarArchiveError(f"S{decision_step} field {key} contains non-finite values")
        for key, value in images.items():
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise NearFarArchiveError(f"S{decision_step} field {key} is not uint8 RGB")
        return {
            "decision_index": index,
            "decision_step": decision_step,
            "metadata": metadata,
            "instruction": instruction,
            "expected_observation": {"instruction": instruction, "state": state, "images": images},
        }


def load_near_far_archive(archive_path: Path) -> dict[str, Any]:
    """Load S65/S70 actions with the existing exact prediction/execution checks."""
    try:
        step_65 = load_archive_episode(archive_path, decision_step=STEP_65)
        step_70 = load_archive_episode(archive_path, decision_step=STEP_70)
        step_75_observation = _load_decision_observation(archive_path, STEP_75)
    except Exception as exc:
        raise NearFarArchiveError(str(exc)) from exc
    if step_65["instruction"] != step_70["instruction"]:
        raise NearFarArchiveError("Archived S65 and S70 task instructions differ")
    if step_70["instruction"] != step_75_observation["instruction"]:
        raise NearFarArchiveError("Archived S70 and S75 task instructions differ")
    replay_65 = np.asarray(step_65["replay_actions"], dtype=np.float32)
    replay_70_prefix = np.asarray(step_70["replay_actions"][:STEP_65], dtype=np.float32)
    if replay_65.shape != (STEP_65, 7) or not np.array_equal(replay_65, replay_70_prefix):
        raise NearFarArchiveError("S65/S70 archived replay prefixes are not exactly aligned")
    action_70 = np.asarray(step_70["action_chunk"], dtype=np.float32)
    if action_70.shape != (CHUNK_LENGTH, 7) or not np.isfinite(action_70).all():
        raise NearFarArchiveError(f"A0_70 must be a finite ({CHUNK_LENGTH}, 7) array")
    return {
        "archive": str(archive_path.resolve()),
        "instruction": step_70["instruction"],
        "step_65": step_65,
        "step_70": step_70,
        "step_75_observation": step_75_observation,
        "action_70": action_70,
        "archive_checks": {
            "s65_s70_replay_prefix_exact": True,
            "a0_70_prediction_matches_executed_actions": True,
            "a0_70_shape": list(action_70.shape),
            "historical_a0_75_actions_loaded_for_forecast": False,
        },
    }


def _select_forbid_pair(
    teacher: dict[str, Any], subject_object_id: str, obstacle_object_id: str
) -> dict[str, Any]:
    objects = {item["id"]: item for item in teacher.get("objects", [])}
    subject = objects.get(subject_object_id)
    obstacle = objects.get(obstacle_object_id)
    if not isinstance(subject, dict) or not isinstance(obstacle, dict):
        raise ValueError("Requested semantic object IDs are absent from Safety IR v2")
    if subject.get("role") != "target_object":
        raise ValueError(f"{subject_object_id} is not marked as target_object in Safety IR v2")
    if obstacle.get("role") not in ("obstacle", "avoid_object"):
        raise ValueError(f"{obstacle_object_id} is not marked as an obstacle/avoid_object")
    rules = [
        rule
        for rule in teacher.get("object_contact_rules", [])
        if isinstance(rule, dict)
        and rule.get("decision") == "forbid"
        and rule.get("predicate") == "contact"
        and rule.get("subject_id") == subject_object_id
        and rule.get("target_id") == obstacle_object_id
    ]
    if len(rules) != 1 or rules[0].get("phase") != "any":
        raise ValueError("Expected exactly one any-phase forbid contact rule for the selected objects")
    return rules[0]


def _resolve_bound_geom_sets(
    binding_validation: dict[str, Any],
    geometry_inventory: dict[str, Any],
    subject_object_id: str,
    obstacle_object_id: str,
) -> tuple[dict[str, Any], list[int], list[int]]:
    geometries = geometry_inventory.get("geometries")
    if not isinstance(geometries, list):
        raise ValueError("Current Geometry Inventory has no geometries array")
    resolved = {}
    geom_sets = []
    for object_id in (subject_object_id, obstacle_object_id):
        binding = binding_validation.get("objects", {}).get(object_id)
        if not isinstance(binding, dict) or binding.get("status") != "matches_current_geometry_inventory":
            raise ValueError(f"No validated current Geometry Inventory binding for {object_id}")
        body_id = binding.get("body_id")
        if not isinstance(body_id, int) or isinstance(body_id, bool):
            raise ValueError(f"Validated body_id is missing for {object_id}")
        entries = [
            entry
            for entry in geometries
            if isinstance(entry, dict) and entry.get("body_id") == body_id
        ]
        if not entries or any(entry.get("category") != "scene" for entry in entries):
            raise ValueError(f"Current model cannot resolve a complete scene Geom set for {object_id}")
        ids = sorted({entry.get("geom_id") for entry in entries})
        if not ids or any(not isinstance(item, int) or isinstance(item, bool) for item in ids):
            raise ValueError(f"Current model Geom IDs are incomplete for {object_id}")
        bound_ids = binding.get("geom_ids")
        if not isinstance(bound_ids, list) or not set(bound_ids).issubset(set(ids)):
            raise ValueError(f"Semantic Geom binding is inconsistent with the current Body Geoms for {object_id}")
        resolved[object_id] = {
            "teacher_object_id": object_id,
            "teacher_name": binding.get("teacher_name"),
            "body_id": body_id,
            "body_name": binding.get("body_name"),
            "semantic_binding_geom_ids": bound_ids,
            "distance_query_geom_ids": ids,
            "distance_query_geom_count": len(ids),
            "source": "all active Geometry Inventory Geoms whose body_id matches the validated semantic Body binding",
        }
        geom_sets.append(ids)
    return resolved, geom_sets[0], geom_sets[1]


def _strict_state_check(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    comparison = compare_observation(expected, actual, STATE_ATOL)
    return {
        "matched": bool(comparison["instruction_matched"] and comparison["state_matched"]),
        "atol": STATE_ATOL,
        "instruction_matched": comparison["instruction_matched"],
        "state_matched": comparison["state_matched"],
        "state": comparison["state"],
        "rgb_diagnostic_only": {
            "matched": comparison["images_matched"],
            "available": comparison["images_available"],
            "cameras": comparison["images"],
        },
    }


def _query_gap(
    connection: Any,
    state_label: str,
    global_step: int,
    subject_geom_ids: list[int],
    obstacle_geom_ids: list[int],
    distmax_m: float,
) -> dict[str, Any]:
    response = _bridge_request(
        connection,
        {
            "op": "geom_set_min_distance",
            "geom_ids_a": subject_geom_ids,
            "geom_ids_b": obstacle_geom_ids,
            "distmax_m": distmax_m,
        },
    )
    value = response.get("minimum_distance_m")
    valid = bool(
        response.get("query_complete") is True
        and response.get("distance_reliable") is True
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    return {
        "state": state_label,
        "global_step": global_step,
        "distance_available": valid,
        "minimum_distance_m": float(value) if valid else None,
        "query_status": response.get("status", "unavailable"),
        "query_complete": response.get("query_complete") is True,
        "distance_reliable": response.get("distance_reliable") is True,
        "distmax_m": response.get("distmax_m", distmax_m),
        "pair_count_requested": response.get("pair_count_requested"),
        "pair_count_attempted": response.get("pair_count_attempted"),
        "pair_count_censored_at_distmax": response.get("pair_count_censored_at_distmax"),
        "pair_error_count": response.get("pair_error_count", len(response.get("pair_errors", []))),
        "minimum_pair": response.get("minimum_pair"),
        "mesh_approximation": response.get("mesh_approximation"),
        "warnings": response.get("warnings", []),
        "errors": (
            response.get("errors")
            if response.get("errors") is not None
            else response.get("pair_errors", [])[:10]
        ),
        "not_a_safety_proof": True,
    }


def analyze_gap_trajectory(
    samples: list[dict[str, Any]],
    historical_collision: dict[str, Any] | None,
    thresholds_m: tuple[float, ...] = GAP_THRESHOLDS_M,
) -> dict[str, Any]:
    """Compute fixed-threshold alerts only from S70 and the A0_70 replay horizon."""
    by_label = {item.get("state"): item for item in samples}
    s70_to_s75 = [by_label.get(f"S{step}") for step in range(STEP_70, STEP_75 + 1)]
    valid_window = all(
        isinstance(item, dict) and item.get("distance_available") is True
        for item in s70_to_s75
    )
    deltas = []
    for previous, current in zip(s70_to_s75, s70_to_s75[1:]):
        if (
            isinstance(previous, dict)
            and isinstance(current, dict)
            and previous.get("distance_available") is True
            and current.get("distance_available") is True
        ):
            delta = float(current["minimum_distance_m"]) - float(previous["minimum_distance_m"])
            elapsed = int(current["global_step"]) - int(previous["global_step"])
            deltas.append(
                {
                    "from_state": previous["state"],
                    "to_state": current["state"],
                    "gap_change_m": delta,
                    "control_steps_elapsed": elapsed,
                    "gap_change_m_per_control_step": delta / elapsed if elapsed > 0 else None,
                }
            )
        else:
            deltas.append({"from_state": previous.get("state") if previous else None,
                           "to_state": current.get("state") if current else None,
                           "gap_change_m": None,
                           "control_steps_elapsed": None,
                           "gap_change_m_per_control_step": None,
                           "available": False})
    sustained = bool(
        valid_window
        and len(deltas) == CHUNK_LENGTH
        and all(item["gap_change_m"] < -TREND_EPSILON_M for item in deltas)
    )
    collision = historical_collision or {}
    collision_valid = collision.get("status") == "violation" and collision.get("validated") is True
    collision_step = collision.get("absolute_control_step") if collision_valid else None
    expected_substeps = collision.get("physics_substeps_per_control_step") if collision_valid else None
    collision_substep = collision.get("first_physics_substep") if collision_valid else None

    alerts = []
    for threshold in thresholds_m:
        observed = [item for item in samples if item.get("distance_available") is True]
        first_any = next(
            (item for item in observed if float(item["minimum_distance_m"]) <= threshold),
            None,
        )
        first_any_index = next(
            (
                index
                for index, item in enumerate(samples)
                if item.get("distance_available") is True
                and float(item["minimum_distance_m"]) <= threshold
            ),
            None,
        )
        first_any_known = bool(
            first_any_index is not None
            and all(item.get("distance_available") is True for item in samples[:first_any_index])
        )
        s70_first_index = next(
            (
                index
                for index, item in enumerate(s70_to_s75)
                if isinstance(item, dict)
                and item.get("distance_available") is True
                and float(item["minimum_distance_m"]) <= threshold
            ),
            None,
        )
        missing_before_s70_trigger = (
            s70_first_index is not None
            and any(
                not isinstance(item, dict) or item.get("distance_available") is not True
                for item in s70_to_s75[:s70_first_index]
            )
        )
        s70_distance = by_label.get("S70")
        if s70_first_index is not None:
            trigger = s70_to_s75[s70_first_index]
            prediction_triggered: bool | None = True
            earliest_step = None if missing_before_s70_trigger else trigger["global_step"]
        elif valid_window:
            trigger = None
            prediction_triggered = False
            earliest_step = None
        else:
            trigger = None
            prediction_triggered = None
            earliest_step = None
        lead = None
        if trigger is not None and collision_step is not None:
            lead = {
                "control_steps_to_historical_first_contact": collision_step - trigger["global_step"],
                "historical_first_contact_control_step": collision_step,
                "historical_first_contact_physics_substep": collision_substep,
                "physics_substeps_per_control_step": expected_substeps,
                "nominal_physics_substeps_to_contact": (
                    (collision_step - trigger["global_step"]) * expected_substeps
                    + collision_substep
                    if isinstance(expected_substeps, int)
                    and isinstance(collision_substep, int)
                    else None
                ),
                "substep_index_is_zero_based": True,
            }
        alerts.append(
            {
                "threshold_m": float(threshold),
                "threshold_mm": float(threshold * 1000.0),
                "s70_currently_within_threshold": (
                    float(s70_distance["minimum_distance_m"]) <= threshold
                    if isinstance(s70_distance, dict) and s70_distance.get("distance_available") is True
                    else None
                ),
                "s70_prediction_triggered_using_s70_plus_a0_70_only": prediction_triggered,
                "first_trigger_step_in_s70_forecast": earliest_step,
                "first_observed_trigger_step_in_s70_forecast": trigger.get("global_step") if trigger else None,
                "first_crossing_step_from_s65_observation": (
                    first_any.get("global_step") if first_any and first_any_known else None
                ),
                "first_observed_crossing_step_from_s65_observation": first_any.get("global_step") if first_any else None,
                "earliest_s65_crossing_is_known": first_any_known,
                "earliest_trigger_is_known": bool(trigger is not None and not missing_before_s70_trigger),
                "trigger_distance_m": trigger.get("minimum_distance_m") if trigger else None,
                "lead_to_historical_s75_contact": lead,
            }
        )
    return {
        "fixed_thresholds_m": [float(item) for item in thresholds_m],
        "s70_prediction_data_scope": "S70 geometry distance and states S71-S75 generated only by archived A0_70; A0_75 is not used",
        "s70_to_s75_all_distance_queries_valid": valid_window,
        "adjacent_gap_changes": deltas,
        "sustained_approach_over_s70_a0_70": sustained if valid_window else None,
        "approach_definition": f"all five sampled gap changes are < {-TREND_EPSILON_M:g} m",
        "threshold_alerts": alerts,
        "historical_collision_used_only_for_posthoc_lead": collision_valid,
        "not_a_safety_proof": True,
    }


def _unknown_history(reason: str, decision_step: int = STEP_75) -> dict[str, Any]:
    return {
        "source": "NPZ decision_safety_json post-hoc record",
        "decision_step": decision_step,
        "status": "unknown",
        "validated": False,
        "reason": reason,
        "not_used_for_s70_forecast": True,
    }


def validate_historical_violation_record(
    record: Any,
    decision_step: int,
    rule: dict[str, Any],
    subject_binding: dict[str, Any],
    obstacle_binding: dict[str, Any],
    geometry_inventory: dict[str, Any],
) -> dict[str, Any]:
    """Validate the archived S75 compiler label against current semantic identities."""
    if not isinstance(record, dict) or record.get("decision_step") != decision_step:
        return _unknown_history("S75 safety record is missing or points at a different decision", decision_step)
    raw_verdict = record.get("a0_verdict")
    compiler_verdict = record.get("a0_compiler_verdict")
    if raw_verdict not in ("safe", "violation", "unknown") or compiler_verdict not in (
        "safe", "violation", "unknown"
    ):
        return _unknown_history("Archived S75 A0/Compiler verdict is missing or invalid", decision_step)
    if raw_verdict != "violation" or compiler_verdict != "violation":
        return {
            "source": "NPZ decision_safety_json post-hoc record",
            "decision_step": decision_step,
            "status": raw_verdict if raw_verdict == compiler_verdict else "unknown",
            "validated": raw_verdict == compiler_verdict,
            "recorded_a0_verdict": raw_verdict,
            "recorded_compiler_verdict": compiler_verdict,
            "not_used_for_s70_forecast": True,
        }

    summary = record.get("a0_branch_summary")
    monitor = summary.get("contact_monitoring") if isinstance(summary, dict) else None
    if (
        not isinstance(summary, dict)
        or summary.get("full_action_chunk_executed") is not True
        or summary.get("requested_steps") != CHUNK_LENGTH
        or summary.get("executed_steps") != CHUNK_LENGTH
        or not isinstance(monitor, dict)
        or monitor.get("status") != "complete"
        or monitor.get("unobserved_expected_substeps") != 0
        or monitor.get("unavailable_contact_substeps") != 0
    ):
        return _unknown_history("Archived S75 branch or substep contact monitoring is incomplete", decision_step)
    controls = monitor.get("control_steps")
    if not isinstance(controls, list) or len(controls) != CHUNK_LENGTH:
        return _unknown_history("Archived S75 contact monitoring does not cover the full Chunk", decision_step)
    expected_total = 0
    captured_total = 0
    for index, control in enumerate(controls):
        if not isinstance(control, dict):
            return _unknown_history(f"Archived S75 control step {index} record is malformed", decision_step)
        samples = control.get("contact_samples")
        counts = [samples.get(key) for key in ("contact", "no_contact", "unavailable")] if isinstance(samples, dict) else []
        if (
            control.get("control_step_index") != index
            or control.get("status") != "complete"
            or control.get("unobserved_expected_substeps") != 0
            or control.get("captured_physics_substeps") != control.get("expected_physics_substeps")
            or not isinstance(samples, dict)
            or samples.get("unavailable") != 0
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts)
            or sum(counts) != control.get("captured_physics_substeps")
        ):
            return _unknown_history(f"Archived S75 control step {index} monitoring is incomplete", decision_step)
        expected_total += control["expected_physics_substeps"]
        captured_total += control["captured_physics_substeps"]
    if (
        monitor.get("expected_physics_substeps_total") != expected_total
        or monitor.get("captured_physics_substeps") != captured_total
    ):
        return _unknown_history("Archived S75 aggregate physics-substep counts are inconsistent", decision_step)

    subject_body = subject_binding.get("body_id")
    obstacle_body = obstacle_binding.get("body_id")
    inventory = {
        item.get("geom_id"): item
        for item in geometry_inventory.get("geometries", [])
        if isinstance(item, dict)
    }
    body_names = {
        item.get("body_id"): item.get("body_name")
        for item in geometry_inventory.get("geometries", [])
        if isinstance(item, dict)
    }
    current_subject_geoms = {
        geom_id for geom_id, item in inventory.items() if item.get("body_id") == subject_body
    }
    current_obstacle_geoms = {
        geom_id for geom_id, item in inventory.items() if item.get("body_id") == obstacle_body
    }
    compiler = record.get("a0_compiled_result")
    if not isinstance(compiler, dict):
        return _unknown_history("Archived S75 has no compiler result", decision_step)
    compiled_rules = [
        item
        for item in compiler.get("compiled_object_contact_rules", [])
        if isinstance(item, dict)
        and item.get("rule_id") == rule.get("id")
        and item.get("subject_object_id") == rule.get("subject_id")
        and item.get("target_object_id") == rule.get("target_id")
    ]
    if len(compiled_rules) != 1:
        return _unknown_history("Archived compiler result does not uniquely identify the selected rule", decision_step)
    compiled_rule = compiled_rules[0]
    compiled_subject_geoms = compiled_rule.get("subject_geom_ids")
    compiled_obstacle_geoms = compiled_rule.get("target_geom_ids")
    semantic_subject_geoms = subject_binding.get("geom_ids")
    semantic_obstacle_geoms = obstacle_binding.get("geom_ids")
    if (
        compiled_rule.get("subject_body_id") != subject_body
        or compiled_rule.get("target_body_id") != obstacle_body
        or not isinstance(compiled_subject_geoms, list)
        or not isinstance(compiled_obstacle_geoms, list)
        or not isinstance(semantic_subject_geoms, list)
        or not isinstance(semantic_obstacle_geoms, list)
        or not compiled_subject_geoms
        or not compiled_obstacle_geoms
        or set(compiled_subject_geoms) != set(semantic_subject_geoms)
        or set(compiled_obstacle_geoms) != set(semantic_obstacle_geoms)
        or not set(compiled_subject_geoms).issubset(current_subject_geoms)
        or not set(compiled_obstacle_geoms).issubset(current_obstacle_geoms)
    ):
        return _unknown_history("Archived compiler rule bindings differ from current semantic Body/Geom bindings", decision_step)

    matching_candidates = []
    for candidate in compiler.get("candidates", []):
        if not isinstance(candidate, dict) or candidate.get("verdict") != "violation":
            continue
        for check in candidate.get("object_contact_checks", []):
            if (
                isinstance(check, dict)
                and check.get("rule_id") == rule.get("id")
                and check.get("subject_id") == rule.get("subject_id")
                and check.get("target_id") == rule.get("target_id")
                and check.get("verdict") == "violation"
            ):
                for evidence in check.get("evidence", []):
                    if isinstance(evidence, dict):
                        matching_candidates.append((candidate, check, evidence))
    validated = []
    for candidate, check, evidence in matching_candidates:
        geom_pair = evidence.get("geom_pair")
        body_pair = evidence.get("body_pair")
        if not isinstance(geom_pair, list) or len(geom_pair) != 2 or not isinstance(body_pair, list) or len(body_pair) != 2:
            continue
        body_ids = [item.get("id") if isinstance(item, dict) else None for item in body_pair]
        geom_ids = [item.get("id") if isinstance(item, dict) else None for item in geom_pair]
        if body_ids != [subject_body, obstacle_body]:
            continue
        if evidence.get("subject_body_id") != subject_body or evidence.get("target_body_id") != obstacle_body:
            continue
        if geom_ids[0] not in current_subject_geoms or geom_ids[1] not in current_obstacle_geoms:
            continue
        if any(
            inventory[geom_id].get("geom_name") != geom_pair[index].get("name")
            for index, geom_id in enumerate(geom_ids)
        ):
            continue
        if any(
            body_names.get(body_id) != body_pair[index].get("name")
            for index, body_id in enumerate(body_ids)
        ):
            continue
        validated.append((candidate, check, evidence))
    if not validated:
        return _unknown_history("No S75 violation evidence matches the current object Body/Geom identities", decision_step)
    candidate, check, first = min(
        validated,
        key=lambda item: (item[2].get("first_control_step", math.inf), item[2].get("first_physics_substep", math.inf)),
    )
    step_controls = controls
    local_step = first.get("first_control_step")
    substep = first.get("first_physics_substep")
    if (
        not isinstance(local_step, int)
        or not 0 <= local_step < CHUNK_LENGTH
        or not isinstance(substep, int)
        or substep < 0
        or substep >= step_controls[local_step].get("expected_physics_substeps", 0)
    ):
        return _unknown_history("Archived S75 first-contact time index is invalid", decision_step)
    return {
        "source": "NPZ decision_safety_json post-hoc record",
        "decision_step": decision_step,
        "status": "violation",
        "validated": True,
        "recorded_a0_verdict": raw_verdict,
        "recorded_compiler_verdict": compiler_verdict,
        "candidate_id": candidate.get("candidate_id"),
        "rule_id": check.get("rule_id"),
        "subject_object_id": rule.get("subject_id"),
        "target_object_id": rule.get("target_id"),
        "absolute_control_step": decision_step + local_step,
        "first_control_step_in_chunk": local_step,
        "first_physics_substep": substep,
        "physics_substeps_per_control_step": step_controls[local_step].get("expected_physics_substeps"),
        "geom_pair": first.get("geom_pair"),
        "body_pair": first.get("body_pair"),
        "first_contact_position_world": first.get("first_contact_position_world"),
        "first_contact_normal_geom1_to_geom2_world": first.get("first_contact_normal_geom1_to_geom2_world"),
        "max_force_norm": first.get("max_force_norm"),
        "reason": check.get("reason"),
        "not_used_for_s70_forecast": True,
    }


def load_posthoc_s75_ground_truth(
    archive_path: Path,
    step75_observation: dict[str, Any],
    rule: dict[str, Any],
    subject_binding: dict[str, Any],
    obstacle_binding: dict[str, Any],
    geometry_inventory: dict[str, Any],
) -> dict[str, Any]:
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            if "decision_safety_json" not in archive.files:
                return _unknown_history("Archive has no decision_safety_json field")
            index = int(step75_observation["decision_index"])
            record = json.loads(str(archive["decision_safety_json"][index]))
    except Exception as exc:
        return _unknown_history(f"Cannot read archived S75 safety record: {type(exc).__name__}: {exc}")
    return validate_historical_violation_record(
        record,
        STEP_75,
        rule,
        subject_binding,
        obstacle_binding,
        geometry_inventory,
    )


def _run_steps(connection: Any, actions: np.ndarray, start: int, stop: int) -> dict[str, Any]:
    response = {}
    for index in range(start, stop):
        response = _bridge_request(connection, {"op": "step", "action": actions[index].tolist()})
        if response.get("error"):
            return {"ok": False, "failed_step": index, "error": response["error"], "response": response}
        if response.get("done") is True and index + 1 < stop:
            return {"ok": False, "failed_step": index, "error": "Bridge terminated before the requested prefix ended", "response": response}
    return {"ok": True, "executed_steps": stop - start, "response": response}


def run_experiment(
    connection: Any,
    archive: dict[str, Any],
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    *,
    subject_object_id: str = "object_1",
    obstacle_object_id: str = "object_4",
    distmax_m: float = 10.0,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "failed",
        "archive": archive["archive"],
        "instruction": archive["instruction"],
        "archive_checks": archive["archive_checks"],
        "fixed_thresholds_m": [float(item) for item in GAP_THRESHOLDS_M],
        "distance_query_distmax_m": distmax_m,
        "forecast_actions": "archived A0_70 only",
        "historical_A0_75_used_for_forecast": False,
        "not_a_safety_proof": True,
        "distance_samples": [],
    }
    try:
        rule = _select_forbid_pair(teacher, subject_object_id, obstacle_object_id)
        report["semantic_rule"] = rule
        reset = _bridge_request(connection, {"op": "reset"})
        if reset.get("error"):
            report["status"] = "reset_failed"
            report["failure_reason"] = str(reset["error"])
            return report
        if reset.get("instruction") != archive["instruction"]:
            report["status"] = "task_instruction_mismatch"
            report["failure_reason"] = "Bridge reset instruction differs from archived task"
            return report

        inventory = _bridge_request(connection, {"op": "geometry_inventory"})
        report["geometry_inventory_summary"] = {
            "status": inventory.get("status"),
            "robot_classification_status": inventory.get("robot_classification_status"),
            "geom_count": inventory.get("geom_count"),
        }
        binding_validation = _validate_current_bindings(teacher, semantic_results, inventory)
        report["semantic_binding_validation"] = binding_validation
        if not binding_validation.get("valid"):
            report["status"] = "binding_validation_failed"
            report["failure_reason"] = "Teacher semantic bindings do not match current Geometry Inventory"
            return report
        bindings, subject_geom_ids, obstacle_geom_ids = _resolve_bound_geom_sets(
            binding_validation,
            inventory,
            subject_object_id,
            obstacle_object_id,
        )
        teacher_objects = {item["id"]: item for item in teacher.get("objects", [])}
        for object_id, binding in bindings.items():
            binding["teacher_name"] = teacher_objects.get(object_id, {}).get("name")
        report["resolved_objects"] = bindings

        actions = archive["step_70"]["replay_actions"]
        s65 = archive["step_65"]
        replay_65 = _run_steps(connection, actions, 0, STEP_65)
        report["replay_to_s65"] = {key: value for key, value in replay_65.items() if key != "response"}
        if not replay_65.get("ok"):
            report["status"] = "replay_to_s65_failed"
            report["failure_reason"] = replay_65.get("error")
            return report
        observation = replay_65["response"]
        s65_check = _strict_state_check(s65["expected_observation"], observation)
        report["historical_state_checks"] = {"S65": s65_check}
        if not s65_check["matched"]:
            report["status"] = "s65_robot_state_mismatch"
            report["failure_reason"] = "Replayed S65 instruction or robot state differs from archive at 1e-6"
            return report
        report["distance_samples"].append(
            _query_gap(connection, "S65", STEP_65, subject_geom_ids, obstacle_geom_ids, distmax_m)
        )

        replay_s70 = _run_steps(connection, actions, STEP_65, STEP_70)
        report["replay_s65_to_s70"] = {key: value for key, value in replay_s70.items() if key != "response"}
        if not replay_s70.get("ok"):
            report["status"] = "replay_to_s70_failed"
            report["failure_reason"] = replay_s70.get("error")
            return report
        observation = replay_s70["response"]
        s70_check = _strict_state_check(archive["step_70"]["expected_observation"], observation)
        report["historical_state_checks"]["S70"] = s70_check
        if not s70_check["matched"]:
            report["status"] = "s70_robot_state_mismatch"
            report["failure_reason"] = "Replayed S70 instruction or robot state differs from archive at 1e-6"
            return report
        report["distance_samples"].append(
            _query_gap(connection, "S70", STEP_70, subject_geom_ids, obstacle_geom_ids, distmax_m)
        )

        action_70 = archive["action_70"]
        for offset, action in enumerate(action_70):
            response = _bridge_request(
                connection,
                {"op": "step", "action": action.tolist()},
            )
            if response.get("error"):
                report["status"] = "a0_70_execution_failed"
                report["failure_reason"] = f"A0_70 failed at action index {offset}: {response['error']}"
                return report
            state_label = f"S{STEP_70 + offset + 1}"
            report["distance_samples"].append(
                _query_gap(
                    connection,
                    state_label,
                    STEP_70 + offset + 1,
                    subject_geom_ids,
                    obstacle_geom_ids,
                    distmax_m,
                )
            )
            if response.get("done") is True and offset + 1 < len(action_70):
                report["status"] = "a0_70_terminated_early"
                report["failure_reason"] = f"Bridge terminated after A0_70 action index {offset}"
                return report

        s75_check = _strict_state_check(
            archive["step_75_observation"]["expected_observation"], response
        )
        report["historical_state_checks"]["S75"] = s75_check
        report["historical_robot_state_checks_passed"] = all(
            report["historical_state_checks"][label]["matched"]
            for label in ("S65", "S70", "S75")
        )
        report["rgb_differences_are_diagnostic_only"] = True

        # Freeze the S70-only forecast before reading the archived S75 label.
        pre_ground_truth_analysis = analyze_gap_trajectory(
            report["distance_samples"], None
        )
        report["s70_forecast"] = pre_ground_truth_analysis
        historical = load_posthoc_s75_ground_truth(
            Path(archive["archive"]),
            archive["step_75_observation"],
            rule,
            binding_validation["objects"][subject_object_id],
            binding_validation["objects"][obstacle_object_id],
            inventory,
        )
        report["historical_s75_ground_truth"] = historical
        report["gap_analysis_with_posthoc_lead"] = analyze_gap_trajectory(
            report["distance_samples"], historical
        )
        if not report["historical_robot_state_checks_passed"]:
            report["status"] = "historical_robot_state_mismatch"
            report["failure_reason"] = "S65, S70, or S75 robot state differs from the archive at 1e-6"
        elif any(item.get("distance_available") is not True for item in report["distance_samples"]):
            report["status"] = "distance_unavailable_or_unreliable"
            report["failure_reason"] = "At least one signed-distance query was incomplete or unreliable"
        elif historical.get("validated") is not True:
            report["status"] = "historical_ground_truth_unavailable"
            report["failure_reason"] = historical.get("reason", "S75 historical violation could not be validated")
        else:
            report["status"] = "diagnostic_complete"
        report["interpretation"] = (
            "Near/far thresholds and gap trends are exploratory signals. The separate archived S75 label is post-hoc only; neither a threshold crossing nor a distance estimate proves collision safety."
        )
        return report
    except Exception as exc:
        report["status"] = "bridge_or_diagnostic_error"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return report


def _write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    print(f"Near-Far gap diagnostic: status={report.get('status')}")
    print(f"  forecast source: {report.get('forecast_actions')}; S75 history used for forecast={report.get('historical_A0_75_used_for_forecast')}")
    for sample in report.get("distance_samples", []):
        distance = sample.get("minimum_distance_m")
        print(
            f"  {sample.get('state')}: gap_m={distance if distance is not None else 'unavailable'} "
            f"query={sample.get('query_status')} mesh_approx={sample.get('mesh_approximation')}"
        )
    analysis = report.get("gap_analysis_with_posthoc_lead", report.get("s70_forecast", {}))
    print(f"  sustained approach S70→S75: {analysis.get('sustained_approach_over_s70_a0_70')}")
    for item in analysis.get("threshold_alerts", []):
        print(
            f"  {item.get('threshold_mm')}mm: S70 forecast={item.get('s70_prediction_triggered_using_s70_plus_a0_70_only')} "
            f"first_step={item.get('first_trigger_step_in_s70_forecast')} "
            f"lead={item.get('lead_to_historical_s75_contact')}"
        )
    history = report.get("historical_s75_ground_truth", {})
    print(
        f"  historical S75 label={history.get('status')} validated={history.get('validated')} "
        f"first_contact={history.get('absolute_control_step')}+substep{history.get('first_physics_substep')}"
    )
    if report.get("failure_reason"):
        print(f"  failure: {report['failure_reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--teacher-json", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--semantic-results-json", type=Path, default=DEFAULT_BINDINGS)
    parser.add_argument("--subject-object-id", default="object_1")
    parser.add_argument("--obstacle-object-id", default="object_4")
    parser.add_argument("--distmax-m", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=BRIDGE_PORT)
    args = parser.parse_args(argv)

    try:
        archive = load_near_far_archive(args.archive)
        teacher = parse_teacher_output_v2(args.teacher_json.read_text(encoding="utf-8"))
        semantic_results = _read_json(args.semantic_results_json, "semantic bindings")
        if not isinstance(semantic_results, dict):
            raise ValueError("Semantic results must map Teacher Object IDs to binding objects")
        connection = Client((args.host, args.port), authkey=AUTHKEY)
        try:
            report = run_experiment(
                connection,
                archive,
                teacher,
                semantic_results,
                subject_object_id=args.subject_object_id,
                obstacle_object_id=args.obstacle_object_id,
                distmax_m=args.distmax_m,
            )
        finally:
            connection.close()
    except Exception as exc:
        report = {
            "status": "input_or_connection_error",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "not_a_safety_proof": True,
        }
    _write_json(args.out, report)
    print_summary(report)
    print(f"  report: {args.out.resolve()}")
    return 0 if report.get("status") == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
