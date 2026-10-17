"""Snapshot-local Near-Far geometric risk forecast helpers.

The forecast replays one five-action Student chunk on the Bridge main
environment, queries MuJoCo geom-set clearance at the current state and after
each action, then restores and verifies the supplied Snapshot. Distances are
exploratory evidence, not a collision-safety proof.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

GAP_THRESHOLDS_M = (0.005, 0.010, 0.020)
TREND_EPSILON_M = 1e-9
CHUNK_LENGTH = 5
SNAPSHOT_ATOL = 1e-6


def _bridge_request(connection: Any, request: dict[str, Any]) -> dict[str, Any]:
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError(f"Bridge returned a non-object response for {request.get('op')!r}")
    return response


def _valid_geom_ids(values: Any, name: str) -> list[int]:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a non-empty sequence of Geom IDs")
    result: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must contain only integer Geom IDs")
        geom_id = int(value)
        if geom_id < 0:
            raise ValueError(f"{name} must contain non-negative Geom IDs")
        result.append(geom_id)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and contain unique Geom IDs")
    return result


def query_geom_gap(
    connection: Any,
    *,
    subject_geom_ids: list[int],
    obstacle_geom_ids: list[int],
    distmax_m: float = 10.0,
    step_index: int | None = None,
) -> dict[str, Any]:
    """Query one complete signed minimum distance between verified Geom sets."""
    try:
        response = _bridge_request(connection, {
            "op": "geom_set_min_distance",
            "geom_ids_a": subject_geom_ids,
            "geom_ids_b": obstacle_geom_ids,
            "distmax_m": distmax_m,
        })
    except Exception as exc:
        return {
            "step_index": step_index,
            "distance_available": False,
            "query_status": "unavailable",
            "query_complete": False,
            "distance_reliable": False,
            "minimum_distance_m": None,
            "minimum_pair": None,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "not_a_safety_proof": True,
        }
    value = response.get("minimum_distance_m")
    valid_value = (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )
    valid = bool(
        response.get("query_complete") is True
        and response.get("distance_reliable") is True
        and valid_value
    )
    errors = response.get("errors")
    if errors is None:
        errors = response.get("pair_errors", [])[:10]
    return {
        "step_index": step_index,
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
        "minimum_pair": response.get("minimum_pair") if valid else None,
        "mesh_approximation": response.get("mesh_approximation"),
        "warnings": response.get("warnings", []),
        "errors": errors,
        "not_a_safety_proof": True,
    }


def evaluate_gap_samples(
    samples: list[dict[str, Any] | None],
    thresholds_m: tuple[float, ...] | list[float] = GAP_THRESHOLDS_M,
    trend_epsilon_m: float = TREND_EPSILON_M,
) -> dict[str, Any]:
    """Apply the existing fixed-threshold and sustained-approach criteria.

    Missing or unreliable samples make a negative result unknown. A confirmed
    threshold crossing remains a warning if a later sample is missing.
    """
    thresholds = [float(value) for value in thresholds_m]
    if not thresholds or any(not math.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("thresholds_m must contain finite, non-negative distances")
    if not math.isfinite(float(trend_epsilon_m)) or trend_epsilon_m < 0:
        raise ValueError("trend_epsilon_m must be finite and non-negative")

    def get_distance(sample: Any) -> float | None:
        if not isinstance(sample, dict) or sample.get("distance_available") is not True:
            return None
        value = sample.get("minimum_distance_m")
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
        ):
            return None
        return float(value)

    def get_step(sample: Any) -> Any:
        if not isinstance(sample, dict):
            return None
        return sample.get("step_index", sample.get("global_step"))

    distances = [get_distance(sample) for sample in samples]
    complete = bool(samples) and all(value is not None for value in distances)
    deltas: list[dict[str, Any]] = []
    for index, (previous, current) in enumerate(zip(samples, samples[1:])):
        before, after = distances[index], distances[index + 1]
        from_step = get_step(previous)
        to_step = get_step(current)
        if before is None or after is None:
            deltas.append({
                "from_step": from_step, "to_step": to_step,
                "gap_change_m": None, "gap_change_m_per_control_step": None,
                "available": False,
            })
            continue
        elapsed = (
            int(to_step) - int(from_step)
            if isinstance(from_step, int) and not isinstance(from_step, bool)
            and isinstance(to_step, int) and not isinstance(to_step, bool)
            else 1
        )
        delta = after - before
        deltas.append({
            "from_step": from_step, "to_step": to_step,
            "gap_change_m": delta, "control_steps_elapsed": elapsed,
            "gap_change_m_per_control_step": delta / elapsed if elapsed > 0 else None,
            "available": elapsed > 0,
        })
    sustained: bool | None = None
    if complete:
        sustained = bool(
            deltas and all(
                item.get("available") is True
                and item.get("gap_change_m") is not None
                and item["gap_change_m"] < -float(trend_epsilon_m)
                for item in deltas
            )
        )

    alerts: list[dict[str, Any]] = []
    for threshold in thresholds:
        observed_index = next(
            (index for index, value in enumerate(distances)
             if value is not None and value <= threshold), None
        )
        known = observed_index is not None and all(
            value is not None for value in distances[:observed_index + 1]
        )
        observed = samples[observed_index] if observed_index is not None else None
        alerts.append({
            "threshold_m": threshold,
            "triggered": True if observed_index is not None else (False if complete else None),
            "first_trigger_step": get_step(observed) if known else None,
            "first_observed_trigger_step": get_step(observed),
            "trigger_distance_m": distances[observed_index] if observed_index is not None else None,
            "earliest_trigger_is_known": bool(observed_index is not None and known),
        })
    return {
        "all_distance_queries_valid": complete,
        "adjacent_gap_changes": deltas,
        "sustained_approach": sustained,
        "approach_definition": f"all {len(deltas)} adjacent gap changes are < {-float(trend_epsilon_m):g} m",
        "threshold_alerts": alerts,
    }


def _compare_snapshot_state(expected: Any, actual: Any, atol: float) -> dict[str, Any]:
    mismatches: list[str] = []
    max_abs_delta = 0.0
    source_metadata: dict[str, Any] = {"informational_only": True}

    def visit(left: Any, right: Any, path: str) -> None:
        nonlocal max_abs_delta
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                mismatches.append(path + ":keys")
                return
            for key in left:
                if key == "kinematics_source" and isinstance(left[key], str) and isinstance(right[key], str):
                    source_metadata.update({"expected": left[key], "actual": right[key], "matched": left[key] == right[key]})
                    continue
                visit(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, np.ndarray):
            left = left.tolist()
        if isinstance(right, np.ndarray):
            right = right.tolist()
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                mismatches.append(path + ":length")
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        numeric_left = isinstance(left, (int, float, np.integer, np.floating)) and not isinstance(left, (bool, np.bool_))
        numeric_right = isinstance(right, (int, float, np.integer, np.floating)) and not isinstance(right, (bool, np.bool_))
        if numeric_left and numeric_right:
            delta = abs(float(left) - float(right))
            max_abs_delta = max(max_abs_delta, delta) if math.isfinite(delta) else math.inf
            if not math.isfinite(delta) or delta > atol:
                mismatches.append(path)
            return
        if type(left) is not type(right) or left != right:
            mismatches.append(path)

    if not isinstance(expected, dict) or not isinstance(actual, dict):
        mismatches.append("$snapshot_state:missing_or_non_object")
    else:
        visit(expected, actual, "$snapshot_state")
    return {
        "matched": not mismatches,
        "max_abs_delta": max_abs_delta,
        "atol": atol,
        "mismatch_count": len(mismatches),
        "mismatch_paths_sample": mismatches[:20],
        "diagnostic_metadata": source_metadata,
    }


def _restore_and_compare(connection: Any, snapshot_id: str, snapshot_state: dict[str, Any], atol: float) -> dict[str, Any]:
    try:
        response = _bridge_request(connection, {"op": "restore", "snapshot_id": snapshot_id})
    except Exception as exc:
        return {"matched": False, "reason": f"{type(exc).__name__}: {exc}"}
    if response.get("error"):
        return {"matched": False, "reason": f"restore_error: {response['error']}", "response": response}
    if response.get("restored_snapshot_id") != snapshot_id:
        return {"matched": False, "reason": "restored_snapshot_id_mismatch", "response": response}
    comparison = _compare_snapshot_state(snapshot_state, response.get("snapshot_state"), atol)
    comparison["restored_snapshot_id"] = response.get("restored_snapshot_id")
    return comparison


def forecast_near_far_risk(
    connection: Any,
    snapshot: dict[str, Any],
    original_actions: Any,
    target_geom_ids: list[int],
    obstacle_geom_ids: list[int],
    thresholds_m: tuple[float, ...] | list[float] = GAP_THRESHOLDS_M,
    *,
    decision_step: int = 0,
    distmax_m: float = 10.0,
    snapshot_atol: float = SNAPSHOT_ATOL,
) -> dict[str, Any]:
    """Forecast one current five-step A0 chunk and always restore its Snapshot.

    Snapshot is a Bridge response with snapshot_id and snapshot_state.
    decision_step is an arbitrary caller-supplied global index; no archive,
    future action, or fixed episode step is consulted.
    """
    actions = np.asarray(original_actions, dtype=np.float64)
    if actions.shape != (CHUNK_LENGTH, 7) or not np.isfinite(actions).all():
        raise ValueError(f"original_actions must be finite with shape ({CHUNK_LENGTH}, 7)")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("snapshot_id"), str) or not snapshot["snapshot_id"]:
        raise ValueError("snapshot must contain a non-empty snapshot_id")
    if not isinstance(snapshot.get("snapshot_state"), dict):
        raise ValueError("snapshot must contain snapshot_state for strict restore verification")
    if isinstance(decision_step, bool) or not isinstance(decision_step, int) or decision_step < 0:
        raise ValueError("decision_step must be a non-negative integer")
    if isinstance(distmax_m, bool) or not isinstance(distmax_m, (int, float)) or not math.isfinite(float(distmax_m)) or distmax_m <= 0:
        raise ValueError("distmax_m must be finite and positive")
    if isinstance(snapshot_atol, bool) or not isinstance(snapshot_atol, (int, float)) or not math.isfinite(float(snapshot_atol)) or snapshot_atol < 0:
        raise ValueError("snapshot_atol must be finite and non-negative")
    target_ids = _valid_geom_ids(target_geom_ids, "target_geom_ids")
    obstacle_ids = _valid_geom_ids(obstacle_geom_ids, "obstacle_geom_ids")
    if set(target_ids) & set(obstacle_ids):
        raise ValueError("target and obstacle Geom ID sets must be disjoint")
    evaluate_gap_samples([], thresholds_m)

    snapshot_id = snapshot["snapshot_id"]
    snapshot_state = snapshot["snapshot_state"]
    samples: list[dict[str, Any]] = []
    unknown_reasons: list[str] = []
    executed_steps = 0
    simulation_complete = False
    restore_before: dict[str, Any] = {"matched": False, "reason": "not_attempted"}
    restore_after: dict[str, Any] = {"matched": False, "reason": "not_attempted"}
    try:
        restore_before = _restore_and_compare(connection, snapshot_id, snapshot_state, float(snapshot_atol))
        if not restore_before.get("matched"):
            unknown_reasons.append("snapshot_restore_before_forecast_failed")
        else:
            first = query_geom_gap(
                connection, subject_geom_ids=target_ids, obstacle_geom_ids=obstacle_ids,
                distmax_m=float(distmax_m), step_index=decision_step,
            )
            samples.append(first)
            if not first.get("distance_available"):
                unknown_reasons.append(f"distance_query_incomplete_at_step_{decision_step}")
            for offset, action in enumerate(actions):
                global_step = decision_step + offset + 1
                try:
                    step_response = _bridge_request(connection, {"op": "step", "action": action.tolist()})
                except Exception as exc:
                    unknown_reasons.append(f"bridge_step_failed_at_chunk_index_{offset}: {type(exc).__name__}: {exc}")
                    break
                if step_response.get("error"):
                    unknown_reasons.append(f"bridge_step_failed_at_chunk_index_{offset}: {step_response['error']}")
                    break
                executed_steps += 1
                sample = query_geom_gap(
                    connection, subject_geom_ids=target_ids, obstacle_geom_ids=obstacle_ids,
                    distmax_m=float(distmax_m), step_index=global_step,
                )
                samples.append(sample)
                if not sample.get("distance_available"):
                    unknown_reasons.append(f"distance_query_incomplete_at_step_{global_step}")
                if step_response.get("done") is True and offset + 1 < CHUNK_LENGTH:
                    unknown_reasons.append(f"environment_terminated_before_chunk_end_at_step_{global_step}")
                    break
            simulation_complete = executed_steps == CHUNK_LENGTH
            if not simulation_complete and not any(item.startswith("environment_terminated") for item in unknown_reasons):
                unknown_reasons.append(f"action_chunk_incomplete:{executed_steps}/{CHUNK_LENGTH}")
    except Exception as exc:
        unknown_reasons.append(f"forecast_failed: {type(exc).__name__}: {exc}")
    finally:
        restore_after = _restore_and_compare(connection, snapshot_id, snapshot_state, float(snapshot_atol))
        if not restore_after.get("matched"):
            unknown_reasons.append("snapshot_restore_after_forecast_failed")

    analysis = evaluate_gap_samples(samples, thresholds_m)
    triggered_alerts = [item for item in analysis["threshold_alerts"] if item.get("triggered") is True]
    threshold_steps = [item.get("first_trigger_step") for item in triggered_alerts]
    threshold_steps = [item for item in threshold_steps if isinstance(item, int) and not isinstance(item, bool)]
    trend_trigger_step = samples[-1].get("step_index") if analysis.get("sustained_approach") is True and samples else None
    trigger_steps = threshold_steps + ([trend_trigger_step] if isinstance(trend_trigger_step, int) else [])
    first_trigger_step = min(trigger_steps) if trigger_steps else None
    observed_threshold_steps = [item.get("first_observed_trigger_step") for item in triggered_alerts]
    observed_threshold_steps = [
        item for item in observed_threshold_steps
        if isinstance(item, int) and not isinstance(item, bool)
    ]
    observed_warning_step = min(observed_threshold_steps) if observed_threshold_steps else trend_trigger_step
    observed_warning = bool(triggered_alerts) or analysis.get("sustained_approach") is True
    integrity_ok = bool(
        simulation_complete
        and analysis.get("all_distance_queries_valid") is True
        and restore_before.get("matched") is True
        and restore_after.get("matched") is True
    )
    if observed_warning:
        risk_verdict, risk_warning = "warning", True
    elif integrity_ok:
        risk_verdict, risk_warning = "no_warning", False
    else:
        risk_verdict, risk_warning = "unknown", None
    for alert in analysis["threshold_alerts"]:
        if alert.get("triggered") is None:
            unknown_reasons.append(f"threshold_{alert['threshold_m']:g}_m_not_resolved_by_complete_samples")
    unknown_reasons = list(dict.fromkeys(unknown_reasons))
    endpoint = samples[-1] if len(samples) == CHUNK_LENGTH + 1 else None
    return {
        "status": "complete" if integrity_ok else "unknown",
        "risk_verdict": risk_verdict,
        "risk_warning": risk_warning,
        "decision_step": decision_step,
        "forecast_scope": "current Snapshot plus the supplied five-step A0 only",
        "historical_or_future_actions_used": False,
        "thresholds_m": [float(value) for value in thresholds_m],
        "target_geom_ids": target_ids,
        "obstacle_geom_ids": obstacle_ids,
        "requested_action_steps": CHUNK_LENGTH,
        "executed_action_steps": executed_steps,
        "simulation_complete": simulation_complete,
        "distance_samples": samples,
        "adjacent_gap_changes": analysis["adjacent_gap_changes"],
        "sustained_approach": analysis["sustained_approach"],
        "approach_definition": analysis["approach_definition"],
        "threshold_alerts": analysis["threshold_alerts"],
        "first_trigger_step": first_trigger_step,
        "first_observed_warning_step": observed_warning_step,
        "endpoint_closest_geom_evidence": endpoint.get("minimum_pair") if endpoint else None,
        "snapshot_restore": {"before": restore_before, "after": restore_after},
        "unknown_reasons": unknown_reasons,
        "not_a_safety_proof": True,
    }
