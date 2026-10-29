# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pure scheduling helpers for bounded task-success search."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from typing import Any, Iterable

import numpy as np

MAX_DEMONSTRATION_HORIZON = 5000


def inspect_native_goal_coverage(native_status: dict[str, Any]) -> dict[str, Any]:
    """Classify every native conjunct without silently discarding operators."""
    rows = native_status.get("native_goal_values")
    if native_status.get("status") != "complete" or not isinstance(rows, list) or not rows:
        raise ValueError("native_goal_values_are_required_for_task_scheduling")
    arities = {"on": 3, "in": 3, "open": 2, "close": 2, "turnon": 2, "turnoff": 2}
    executable = {"on", "in", "open", "close", "turnon", "turnoff"}
    classified = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("value")) is not bool:
            raise ValueError("native_goal_value_must_be_an_explicit_boolean")
        predicate = row.get("predicate")
        if (not isinstance(predicate, list) or not predicate
                or not isinstance(predicate[0], str) or not predicate[0]):
            raise ValueError("native_goal_predicate_must_have_an_operator")
        operator = predicate[0].lower()
        if operator in arities and len(predicate) != arities[operator]:
            raise ValueError(f"native_goal_predicate_arity_mismatch:{operator}")
        if operator in arities and any(not isinstance(token, str) or not token for token in predicate):
            raise ValueError(f"native_goal_predicate_operand_invalid:{operator}")
        classified.append({
            "predicate": list(predicate),
            "value": row["value"],
            "operator": operator,
            "operation_supported": operator in executable,
            "unsupported_reason": None if operator in executable else "unrecognized_native_goal_operator",
        })
    return {
        "goals": classified,
        "pending_unsupported_goals": [row for row in classified
                                      if not row["value"] and not row["operation_supported"]],
    }


def pending_object_goals(native_status: dict[str, Any]) -> list[list[str]]:
    """Return unsatisfied native object placement goals in their BDDL order."""
    rows = native_status.get("native_goal_values")
    if native_status.get("status") != "complete" or not isinstance(rows, list):
        raise ValueError("native_goal_values_are_required_for_task_scheduling")
    return [
        [str(token) for token in row["predicate"]]
        for row in rows
        if isinstance(row, dict)
        and row.get("value") is False
        and isinstance(row.get("predicate"), list)
        and len(row["predicate"]) == 3
        and str(row["predicate"][0]).lower() in {"in", "on"}
    ]


def native_goal_counts(native_status: dict[str, Any]) -> tuple[int, int]:
    """Count all native goals and the currently satisfied subset."""
    rows = native_status.get("native_goal_values")
    if native_status.get("status") != "complete" or not isinstance(rows, list):
        raise ValueError("native_goal_values_are_required_for_task_scheduling")
    return len(rows), sum(row.get("value") is True for row in rows if isinstance(row, dict))


def route_signature(poses: Iterable[Iterable[float]], *, quantum_m: float = 0.025) -> str:
    """Hash a quantized EEF route to suppress near-identical physical retries."""
    route = np.asarray(list(poses), dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 7 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError("route_signature_requires_finite_N_by_7_EEF_poses")
    if not np.isfinite(quantum_m) or quantum_m <= 0:
        raise ValueError("route_signature_quantum_must_be_positive")
    segment_lengths = np.linalg.norm(np.diff(route[:, :3], axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    if cumulative[-1] <= 1e-9:
        raise ValueError("route_signature_requires_nonzero_translation")
    sample_lengths = np.linspace(0.0, cumulative[-1], min(12, max(2, int(np.ceil(cumulative[-1] / quantum_m)) + 1)))
    normalized = np.empty((len(sample_lengths), 7), dtype=np.float64)
    for index, distance in enumerate(sample_lengths):
        upper = int(np.searchsorted(cumulative, distance, side="right"))
        upper = min(max(1, upper), len(route) - 1)
        lower = upper - 1
        span = max(float(cumulative[upper] - cumulative[lower]), 1e-12)
        fraction = float(np.clip((distance - cumulative[lower]) / span, 0.0, 1.0))
        normalized[index, :3] = route[lower, :3] * (1.0 - fraction) + route[upper, :3] * fraction
        quaternion = route[lower, 3:] * (1.0 - fraction) + route[upper, 3:] * fraction
        normalized[index, 3:] = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
    normalized[:, :3] = np.round(normalized[:, :3] / quantum_m).astype(np.int64)
    quaternions = normalized[:, 3:].copy()
    quaternions /= np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12)
    # q and -q represent the same orientation.
    quaternions[quaternions[:, 0] < 0] *= -1
    normalized[:, 3:] = np.round(quaternions / 0.12).astype(np.int64)
    return hashlib.sha256(np.ascontiguousarray(normalized).tobytes()).hexdigest()[:20]


def planner_seed(base_seed: int, target_geom_id: int, candidate_ordinal: int) -> int:
    """Return a stable non-negative seed for geometry and experience candidates.

    Object-relative historical candidates have no single source-scene MuJoCo
    geom id and use ``-1``. Keep them deterministic without passing a negative
    value into NumPy's Generator constructor.
    """
    values = (base_seed, target_geom_id, candidate_ordinal)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise ValueError("planner_seed_inputs_must_be_integers")
    if base_seed < 0 or candidate_ordinal < 0:
        raise ValueError("planner_seed_base_and_candidate_ordinal_must_be_non_negative")
    return int((int(base_seed) + max(0, int(target_geom_id)) * 31 + int(candidate_ordinal)) % (2**32))


def route_control_action_budget(
    poses: Iterable[Iterable[float]],
    *,
    control_timestep_s: float = 0.05,
    observed_speed_m_s: float = 0.025,
    settling_multiplier: float = 1.2,
    minimum_actions: int = 74,
    maximum_actions: int = 320,
) -> int:
    """Bound OSC route execution from its length and measured control speed."""
    route = np.asarray(list(poses), dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 7 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError("route_budget_requires_finite_N_by_7_EEF_poses")
    if (
        not np.isfinite([control_timestep_s, observed_speed_m_s, settling_multiplier]).all()
        or control_timestep_s <= 0
        or observed_speed_m_s <= 0
        or settling_multiplier < 1.0
    ):
        raise ValueError("route_budget_timing_and_speed_parameters_are_invalid")
    if (
        isinstance(minimum_actions, bool)
        or isinstance(maximum_actions, bool)
        or not isinstance(minimum_actions, int)
        or not isinstance(maximum_actions, int)
        or not 1 <= minimum_actions <= maximum_actions
    ):
        raise ValueError("route_budget_action_bounds_are_invalid")
    distance_m = float(np.linalg.norm(np.diff(route[:, :3], axis=0), axis=1).sum())
    requested = int(np.ceil(distance_m / (observed_speed_m_s * control_timestep_s) * settling_multiplier))
    return min(maximum_actions, max(minimum_actions, requested))


def target_finger_contact_sides(grasp_state: dict[str, Any]) -> tuple[str, ...]:
    """Return Panda finger sides with measured contact on the active target."""
    if not isinstance(grasp_state, dict):
        return ()
    contacts = grasp_state.get("finger_contacts", {})
    if not isinstance(contacts, dict):
        return ()
    return tuple(side for side in ("left", "right") if contacts.get(side) is True)


def unilateral_target_contact_sides(grasp_state: dict[str, Any]) -> tuple[str, ...]:
    """Return the one Panda finger side with measured target contact, if any."""
    sides = target_finger_contact_sides(grasp_state)
    return sides if len(sides) == 1 else ()


def high_approach_route(
    start_pose_world: Iterable[float],
    target_pose_world: Iterable[float],
    *,
    elevation_m: float = 0.10,
) -> list[list[float]]:
    """Create a generic high-then-descend OSC route for an IK-reachable pose."""
    start = np.asarray(start_pose_world, dtype=np.float64)
    target = np.asarray(target_pose_world, dtype=np.float64)
    if (
        start.shape != (7,)
        or target.shape != (7,)
        or not np.isfinite(start).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("high_approach_route_requires_finite_world_poses")
    if not np.isfinite(elevation_m) or elevation_m <= 0:
        raise ValueError("high_approach_route_elevation_must_be_positive")
    for pose in (start, target):
        norm = float(np.linalg.norm(pose[3:]))
        if norm <= 1e-12:
            raise ValueError("high_approach_route_pose_has_zero_quaternion")
        pose[3:] /= norm
    high = target.copy()
    high[2] += float(elevation_m)
    return [start.tolist(), high.tolist(), target.tolist()]


def robot_contact_escape_direction(
    contacts: Iterable[dict[str, Any]], *, ignore_object_id: str | None = None
) -> dict[str, Any] | None:
    """Estimate an outward direction from actual MuJoCo robot-scene contacts."""
    vectors: list[tuple[float, np.ndarray, str]] = []
    for contact in contacts:
        if not isinstance(contact, dict):
            continue
        geom1, geom2 = contact.get("geom1", {}), contact.get("geom2", {})
        body1, body2 = contact.get("body1", {}), contact.get("body2", {})
        names1 = [str(geom1.get("name", "")), str(body1.get("name", ""))]
        names2 = [str(geom2.get("name", "")), str(body2.get("name", ""))]
        if ignore_object_id and any(
            name.startswith(str(ignore_object_id)) for name in names1 + names2
        ):
            continue

        def is_robot(names: list[str]) -> bool:
            return any(name.startswith(("robot0_", "gripper0_")) for name in names)

        robot1, robot2 = is_robot(names1), is_robot(names2)
        if robot1 == robot2 or any("mount0_pedestal_col" in name for name in names1 + names2):
            continue
        normal = np.asarray(contact.get("normal_geom1_to_geom2_world", []), dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            continue
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        direction = normal / norm if robot2 else -normal / norm
        force = np.asarray(contact.get("force_and_torque", [])[:3], dtype=np.float64)
        force_norm = float(np.linalg.norm(force)) if force.shape == (3,) and np.isfinite(force).all() else 0.0
        distance = abs(float(contact.get("signed_contact_distance_m", 0.0) or 0.0))
        weight = max(1.0, force_norm) * (1.0 + min(distance, 0.02) * 10.0)
        blocker = str((geom1 if robot2 else geom2).get("name", "unknown_scene_geom"))
        vectors.append((weight, direction, blocker))
    if not vectors:
        return None
    resultant = sum((weight * direction for weight, direction, _ in vectors), np.zeros(3))
    norm = float(np.linalg.norm(resultant))
    if norm <= 1e-9:
        _, direction, _ = max(vectors, key=lambda row: row[0])
    else:
        direction = resultant / norm
    return {
        "direction_world": direction.tolist(),
        "contact_count": len(vectors),
        "blocker_geom_names": sorted({name for _, _, name in vectors}),
        "source": "MuJoCo_robot_scene_contact_normals",
        "task_success_search_diagnostic_only": True,
    }


def object_relative_grasp_targets(
    object_pose_world: Any,
    object_to_eef: Any,
    approach_direction_object: Any,
    approach_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a grasp and its pregrasp point from the target object's frame.

    The grasp pose carries the EEF orientation, while the approach vector is
    expressed in the object frame. Keeping those frames separate is essential
    when migrating a grasp between layouts with different object rotations.
    """
    root = np.asarray(object_pose_world, dtype=np.float64)
    grasp_offset = np.asarray(object_to_eef, dtype=np.float64)
    approach_local = np.asarray(approach_direction_object, dtype=np.float64)
    if (
        root.shape != (4, 4)
        or grasp_offset.shape != (4, 4)
        or approach_local.shape != (3,)
        or not np.isfinite(root).all()
        or not np.isfinite(grasp_offset).all()
        or not np.isfinite(approach_local).all()
        or not np.isfinite(approach_distance_m)
        or approach_distance_m < 0
    ):
        raise ValueError("object_relative_grasp_target_inputs_are_invalid")
    norm = float(np.linalg.norm(approach_local))
    if norm <= 1e-12:
        raise ValueError("object_relative_approach_direction_is_zero")
    grasp_pose_world = root @ grasp_offset
    approach_world = root[:3, :3] @ (approach_local / norm)
    pregrasp_pose_world = grasp_pose_world.copy()
    pregrasp_pose_world[:3, 3] -= approach_world * float(approach_distance_m)
    return grasp_pose_world, pregrasp_pose_world


def filter_mechanically_infeasible_grasp_candidates(
    candidates: Iterable[dict[str, Any]], *, tolerance_m: float = 1e-6
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove grasps whose required jaw opening exceeds the active gripper limit.

    This is a task-feasibility check, not a safety clearance rule. Candidates
    without width metadata remain eligible so older verified templates keep
    their existing behavior.
    """
    if not np.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("grasp_opening_tolerance_must_be_finite_and_non_negative")
    feasible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("grasp_candidate_must_be_a_mapping")
        width = candidate.get("estimated_clamp_width_m")
        maximum = candidate.get("maximum_opening_m")
        if width is None or maximum is None:
            feasible.append(candidate)
            continue
        try:
            width_m, maximum_m = float(width), float(maximum)
        except (TypeError, ValueError) as exc:
            raise ValueError("grasp_opening_metadata_must_be_numeric") from exc
        if (
            not np.isfinite([width_m, maximum_m]).all()
            or width_m <= 0.0
            or maximum_m <= 0.0
        ):
            raise ValueError("grasp_opening_metadata_must_be_positive_and_finite")
        if width_m <= maximum_m + tolerance_m:
            feasible.append(candidate)
            continue
        rejected.append({
            "candidate_id": candidate.get("candidate_id"),
            "reason": "estimated_clamp_width_exceeds_maximum_pad_separation",
            "estimated_clamp_width_m": width_m,
            "maximum_opening_m": maximum_m,
            "opening_overrun_m": width_m - maximum_m,
        })
    return feasible, rejected


def task_approach_feasible_candidate_ids(screen_report: dict[str, Any]) -> set[str]:
    """Validate and return candidates with a scratch-only task approach path.

    The report must attest only its own screen scope and unchanged live state;
    this helper makes no safety or physical-grasp claim.
    """
    if (
        not isinstance(screen_report, dict)
        or screen_report.get("operation") != "screen_grasp_candidates"
        or screen_report.get("status") != "complete"
        or screen_report.get("task_approach_only") is not True
        or screen_report.get("live_state_unchanged") is not True
    ):
        raise ValueError("task_approach_screen_report_is_incomplete_or_out_of_scope")
    rows = screen_report.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("task_approach_screen_candidate_rows_are_missing")
    seen: set[str] = set()
    feasible: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("task_approach_screen_candidate_row_is_malformed")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("task_approach_screen_candidate_id_is_missing")
        if candidate_id in seen:
            raise ValueError("task_approach_screen_candidate_ids_are_not_unique")
        seen.add(candidate_id)
        if row.get("status") == "offline_feasible_candidate":
            feasible.add(candidate_id)
    return feasible


def diversify_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order ranked experience templates first, diversifying routes within each rank."""
    remaining = list(candidates)
    if len(remaining) < 2:
        return remaining

    def family(row: dict[str, Any]) -> tuple[Any, ...]:
        route_family = row.get("approach_route_family")
        if route_family is not None:
            return ("experience_approach", str(route_family))
        return (
            int(row.get("jaw_axis_index_in_geom") if row.get("jaw_axis_index_in_geom") is not None else -1),
            int(row.get("approach_axis_index_in_geom") if row.get("approach_axis_index_in_geom") is not None else -1),
            int(row.get("approach_sign") if row.get("approach_sign") is not None else 0),
            int(row.get("jaw_sign") if row.get("jaw_sign") is not None else 0),
        )

    experience = [
        row for row in remaining
        if row.get("historical_GS_OPD_object_relative_waypoint_prior_used")
        or row.get("experience_object_relative_grasp_prior_used")
    ]

    def experience_rank(row: dict[str, Any]) -> int:
        rank = row.get("experience_template_rank")
        if rank is not None:
            return max(0, int(rank))
        # Older candidate records have no explicit match rank. Preserve their
        # historical source-time ordering, with missing metadata treated as 0.
        return max(0, int(row.get("experience_source_global_timestep") or 0))

    if experience:
        best_rank = min(experience_rank(row) for row in experience)
        first = min(
            (row for row in experience if experience_rank(row) == best_rank),
            key=lambda row: (
                int(row.get("experience_source_global_timestep") or 0),
                row["candidate_id"],
            ),
        )
    else:
        first = min(remaining, key=lambda row: (
            float(row.get("pregrasp_eef_distance_m") or 0.0), row["candidate_id"]
        ))
    ordered = [first]
    remaining.remove(ordered[0])
    while remaining:
        selected_families = [family(row) for row in ordered]
        experience_remaining = [
            row for row in remaining
            if row.get("historical_GS_OPD_object_relative_waypoint_prior_used")
            or row.get("experience_object_relative_grasp_prior_used")
        ]
        if experience_remaining:
            best_rank = min(experience_rank(row) for row in experience_remaining)
            pool = [row for row in experience_remaining if experience_rank(row) == best_rank]
        else:
            pool = remaining
        next_row = max(
            pool,
            key=lambda row: (
                min(sum(a != b for a, b in zip(family(row), prior)) for prior in selected_families),
                float(row.get("opening_overrun_m_record_only") or 0.0) <= 1e-9,
                -float(row.get("pregrasp_eef_distance_m") or 0.0),
                row["candidate_id"],
            ),
        )
        ordered.append(next_row)
        remaining.remove(next_row)
    return ordered


def candidate_route_preflight_admitted(
    rrt_status: str | None,
    live_state_unchanged: bool | None,
    *,
    direct_osc_fallback_added: bool = False,
    live_contact_probe: bool = False,
) -> bool:
    """Admit physically executable task-search routes after unchanged scratch preflight.

    RRT feasibility is one route source. A bounded object-relative OSC fallback
    is also admissible after IK succeeds and scratch planning leaves the live
    state unchanged; task-success search records its contacts instead of using
    a safety clearance as a route gate.
    """
    if live_state_unchanged is not True:
        return False
    return bool(
        rrt_status == "offline_feasible"
        or direct_osc_fallback_added
        or live_contact_probe
    )


def generated_geometry_high_approach_admitted(
    candidate: dict[str, Any], planner_response: dict[str, Any]
) -> bool:
    """Admit one generic high-approach fallback after a failed scratch RRT."""
    return bool(
        candidate.get("candidate_source")
        == "active_BDDL_target_compiled_collision_geom_and_live_Panda_pad_centers"
        and planner_response.get("status") != "offline_feasible"
        and planner_response.get("live_state_unchanged") is True
        and planner_response.get("ik_candidates")
    )


def classify_failure(reason: str | None, contact_pairs: Iterable[str] = ()) -> str:
    """Map physical evidence to a stable scheduler failure class."""
    text = (reason or "").lower()
    if any(contact_pairs):
        return "contact_blocked"
    if (
        "stagnation" in text
        or "did_not_reach_pose" in text
        or "pregrasp_route_action" in text
        or "approach_action_budget_exhausted" in text
        or "route_action_budget_exhausted" in text
    ):
        return "route_stalled"
    if "ik" in text or "joint_limit" in text or "reach" in text and "route" in text:
        return "reachability"
    if "grasp" in text or "lift" in text:
        return "grasp_or_lift"
    if "unstable" in text or "nonfinite" in text or "non_finite" in text:
        return "simulation_instability"
    return "other_physical_failure"


def blocked_candidate_ids_from_report(
    report: dict[str, Any], *, goal_predicate: str
) -> tuple[set[str], list[dict[str, Any]]]:
    """Extract failed candidates for one native goal, reclassifying old reports.

    Search reports can predate a failure-classifier fix, so the reason string is
    authoritative when importing memory from an earlier run.
    """
    memory = report.get("search_failure_memory", {})
    events = memory.get("events", []) if isinstance(memory, dict) else []
    blocked: set[str] = set()
    imported: list[dict[str, Any]] = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict) or str(event.get("goal_predicate")) != goal_predicate:
            continue
        candidate_id = str(event.get("candidate_id", ""))
        if not candidate_id:
            continue
        failure_reason = str(event.get("failure_reason", ""))
        failure_class = classify_failure(failure_reason, event.get("contact_pairs", []))
        blocked.add(candidate_id)
        imported.append({
            "goal_predicate": goal_predicate,
            "candidate_id": candidate_id,
            "route_id": event.get("route_id"),
            "failure_reason": failure_reason,
            "failure_class": failure_class,
            "reclassified_from": event.get("failure_class"),
            "next_strategy": (
                "skip_same_migrated_route_and_try_distinct_geometry_or_route_family"
                if failure_class in {"contact_blocked", "route_stalled"}
                else "avoid_exact_failed_candidate_route"
            ),
        })
    return blocked, imported


def candidate_matches_failure_memory(candidate_id: str, blocked_candidate_ids: Iterable[str]) -> bool:
    """Match a library template ID to its display-prefixed search candidate ID."""
    candidate = str(candidate_id)
    blocked = {str(value) for value in blocked_candidate_ids}
    if candidate.startswith("experience__"):
        return candidate in blocked or candidate[len("experience__"):] in blocked
    prefix = f"experience__{candidate}"
    return any(
        value == candidate or value == prefix or value.startswith(f"{prefix}__alt_")
        for value in blocked
    )


@dataclass
class SearchFailureMemory:
    """Remember attempted candidate/route pairs and repeated physical blockers."""

    attempted: set[tuple[str, str, str]] = field(default_factory=set)
    failure_counts: Counter[str] = field(default_factory=Counter)
    contact_counts: Counter[str] = field(default_factory=Counter)
    events: list[dict[str, Any]] = field(default_factory=list)

    def admit(self, goal: str, candidate_id: str, route_id: str) -> bool:
        key = (str(goal), str(candidate_id), str(route_id))
        if key in self.attempted:
            return False
        self.attempted.add(key)
        return True

    def record(
        self,
        *,
        goal: str,
        candidate_id: str,
        route_id: str,
        reason: str | None,
        contact_pairs: Iterable[str] = (),
    ) -> dict[str, Any]:
        pairs = sorted(set(str(item) for item in contact_pairs))
        category = classify_failure(reason, pairs)
        self.failure_counts[category] += 1
        for pair in pairs:
            self.contact_counts[pair] += 1
        switch = "keep_candidate_family"
        if category == "contact_blocked":
            switch = "change_approach_side_or_height_and_route"
        elif category == "route_stalled":
            switch = "change_waypoint_route_and_replan"
        elif category in {"reachability", "grasp_or_lift"}:
            switch = "change_grasp_or_approach_family"
        event = {
            "goal_predicate": str(goal),
            "candidate_id": str(candidate_id),
            "route_id": str(route_id),
            "failure_class": category,
            "failure_reason": reason,
            "contact_pairs": pairs,
            "recommended_next_strategy": switch,
        }
        self.events.append(event)
        return event

    def summary(self) -> dict[str, Any]:
        return {
            "attempted_candidate_route_pairs": len(self.attempted),
            "failure_counts": dict(sorted(self.failure_counts.items())),
            "repeated_contact_pairs": [
                {"pair": pair, "count": count}
                for pair, count in self.contact_counts.most_common()
            ],
            "events": list(self.events),
        }


def progress_label(native_status: dict[str, Any], milestones: dict[str, bool]) -> str:
    """Describe the strongest observed stage without upgrading partial success."""
    if native_status.get("status") == "complete" and native_status.get("task_success") is True:
        return "native_full_task_success"
    for key in (
        "transported", "stable_grasp", "partial_predicate", "reachable", "target_reached"
    ):
        if milestones.get(key):
            return {
                "partial_predicate": "partial_predicate_completed",
                "transported": "object_transported",
                "stable_grasp": "stable_grasp_verified",
                "reachable": "goal_reachable_without_grasp",
                "target_reached": "target_not_reached",
            }[key]
    return "target_not_reached"


__all__ = [
    "SearchFailureMemory",
    "classify_failure",
    "blocked_candidate_ids_from_report",
    "candidate_matches_failure_memory",
    "diversify_candidates",
    "high_approach_route",
    "native_goal_counts",
    "planner_seed",
    "route_control_action_budget",
    "target_finger_contact_sides",
    "unilateral_target_contact_sides",
    "pending_object_goals",
    "progress_label",
    "robot_contact_escape_direction",
    "route_signature",
]
