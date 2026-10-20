"""Deterministic Far/Mid/Near Episode scheduling on existing Near-Far thresholds."""

from __future__ import annotations

import math
from typing import Any

from research.gs_opd.repair.near_far import GAP_THRESHOLDS_M

RISK_LAYER_ORDER = ("far", "mid", "near")
RISK_LAYER_THRESHOLD_M = {
    "far": GAP_THRESHOLDS_M[2],
    "mid": GAP_THRESHOLDS_M[1],
    "near": GAP_THRESHOLDS_M[0],
}
RISK_LAYER_LOWER_EXCLUSIVE_M = {
    "far": GAP_THRESHOLDS_M[1],
    "mid": GAP_THRESHOLDS_M[0],
    "near": None,
}


def layer_for_episode_index(episode_index: int) -> str:
    """Assign Far, Mid, Near in a reproducible cycle across Episode indices."""
    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError("episode_index must be a non-negative integer")
    return RISK_LAYER_ORDER[episode_index % len(RISK_LAYER_ORDER)]


def classify_gap_risk_layer(distance_m: float) -> str | None:
    """Map one finite signed gap into the existing 5/10/20 mm bands."""
    if isinstance(distance_m, bool) or not isinstance(distance_m, (int, float)):
        return None
    distance = float(distance_m)
    if not math.isfinite(distance):
        return None
    if distance <= GAP_THRESHOLDS_M[0]:
        return "near"
    if distance <= GAP_THRESHOLDS_M[1]:
        return "mid"
    if distance <= GAP_THRESHOLDS_M[2]:
        return "far"
    return None


def new_episode_risk_layer_record(episode_index: int) -> dict[str, Any]:
    layer = layer_for_episode_index(episode_index)
    return {
        "episode_index": episode_index,
        "episode_id": f"gs-opd-risk-layer-{episode_index:06d}",
        "schedule": "episode_index_modulo_far_mid_near",
        "selected_layer": layer,
        "trigger_threshold_m": float(RISK_LAYER_THRESHOLD_M[layer]),
        "trigger_lower_exclusive_m": (
            float(RISK_LAYER_LOWER_EXCLUSIVE_M[layer])
            if RISK_LAYER_LOWER_EXCLUSIVE_M[layer] is not None
            else None
        ),
        "threshold_source": "research.gs_opd.repair.near_far.GAP_THRESHOLDS_M",
        "trigger_semantics": "first stable-grasp state observed inside the selected existing 5/10/20 mm risk band; skipped deeper bands are not backfilled",
        "distance_is_safety_proof": False,
        "observations": [],
        "trigger": None,
        "workflow_status": "waiting_for_stable_grasp_and_layer",
    }


def record_episode_risk_distance(
    record: dict[str, Any], *, step_index: int, distance_m: float | None,
    distance_available: bool, query_status: str | None = None,
    minimum_pair: Any = None, errors: Any = None,
) -> dict[str, Any] | None:
    """Record one live gap query and return the first threshold trigger, if any."""
    if not isinstance(record, dict) or record.get("selected_layer") not in RISK_LAYER_THRESHOLD_M:
        raise ValueError("record must come from new_episode_risk_layer_record")
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index must be a non-negative integer")
    valid_distance = (
        distance_available is True
        and isinstance(distance_m, (int, float))
        and not isinstance(distance_m, bool)
        and math.isfinite(float(distance_m))
    )
    sample = {
        "step_index": step_index,
        "distance_available": bool(valid_distance),
        "minimum_distance_m": float(distance_m) if valid_distance else None,
        "observed_layer": classify_gap_risk_layer(float(distance_m)) if valid_distance else None,
        "query_status": query_status,
        "minimum_pair": minimum_pair,
        "errors": list(errors) if isinstance(errors, (list, tuple)) else [],
    }
    observations = record.setdefault("observations", [])
    if observations and observations[-1].get("step_index") == step_index:
        observations[-1] = sample
    else:
        observations.append(sample)

    if record.get("trigger") is None and valid_distance:
        threshold = float(record["trigger_threshold_m"])
        if sample["observed_layer"] == record["selected_layer"]:
            trigger = {
                "step_index": step_index,
                "layer": record["selected_layer"],
                "threshold_m": threshold,
                "lower_exclusive_m": record["trigger_lower_exclusive_m"],
                "observed_distance_m": float(distance_m),
                "minimum_pair": minimum_pair,
            }
            record["trigger"] = trigger
            record["workflow_status"] = "risk_layer_reached"
            return trigger
    elif record.get("trigger") is None and not valid_distance:
        record["workflow_status"] = "waiting_after_unavailable_distance_query"
    return None

