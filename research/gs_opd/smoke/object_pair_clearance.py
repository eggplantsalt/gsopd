# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Evaluate explicitly forbidden object pairs at a measured physical state."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable


def measure_forbidden_object_pairs(
    pairs: Iterable[dict[str, Any]],
    measure: Callable[[Iterable[int], Iterable[int]], dict[str, Any]],
    *,
    minimum_clearance_m: float = 0.010,
) -> dict[str, Any]:
    """Measure each explicitly forbidden pair, failing closed on missing evidence.

    ``measure`` must use the complete bound collision geometry sets in the
    current MuJoCo state. Support relations are not implicitly forbidden; a
    support that is also explicitly forbidden remains subject to the margin.
    """
    margin = float(minimum_clearance_m)
    if not math.isfinite(margin) or margin < 0.010:
        raise ValueError("forbidden_object_pair_margin_must_be_at_least_10mm")
    results = []
    has_incomplete = False
    has_violation = False
    for pair in pairs:
        name_a, name_b = pair.get("object_id_a"), pair.get("object_id_b")
        geoms_a, geoms_b = pair.get("geom_ids_a"), pair.get("geom_ids_b")
        if not isinstance(name_a, str) or not name_a or not isinstance(name_b, str) or not name_b:
            raise ValueError("forbidden_object_pair_identity_missing")
        if not geoms_a or not geoms_b:
            raise ValueError("forbidden_object_pair_geometry_missing")
        try:
            measurement = measure(geoms_a, geoms_b)
            distance = measurement.get("minimum_distance_m")
            complete = (measurement.get("status") == "complete"
                        and isinstance(distance, (int, float))
                        and not isinstance(distance, bool)
                        and math.isfinite(float(distance)))
        except Exception as exc:
            measurement = {"status": "incomplete", "error": f"{type(exc).__name__}: {exc}"}
            distance = None
            complete = False
        violation = bool(complete and float(distance) < margin)
        has_incomplete |= not complete
        has_violation |= violation
        raw_pair = measurement.get("pair") if isinstance(measurement, dict) else None
        geometry_pair = None
        if isinstance(raw_pair, dict):
            geometry_pair = {
                "geom_a_id": raw_pair.get("robot_geom_id"),
                "geom_a_name": raw_pair.get("robot_geom_name"),
                "geom_b_id": raw_pair.get("object_geom_id"),
                "geom_b_name": raw_pair.get("object_geom_name"),
                "closest_points_world": raw_pair.get("closest_points_world"),
            }
        results.append({
            "object_id_a": name_a, "object_id_b": name_b,
            "minimum_required_m": margin,
            "geometry_pair": geometry_pair,
            "measurement": measurement,
            "verdict": "violation" if violation else "pass" if complete else "incomplete",
        })
    return {
        "status": "unsafe" if has_violation else "incomplete" if has_incomplete else "complete",
        "measurement_complete": not has_incomplete,
        "pairs": results,
    }
