# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit-aware MuJoCo joint-limit measurements for the Safety Bridge."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_JOINT_LIMIT_TOLERANCE_POLICY: dict[str, Any] = {
    "by_joint_type": {
        "hinge": {
            "strict_tolerance": 1e-8,
            "operational_tolerance": 1e-8,
            "unit": "rad",
        },
        "slide": {
            "strict_tolerance": 1e-8,
            "operational_tolerance": 1e-8,
            "unit": "m",
        },
    },
    "by_joint_name": {},
}

_TYPE_UNITS = {"hinge": "rad", "slide": "m"}


def joint_type_name(joint_type: int, hinge_type: int, slide_type: int) -> str:
    """Map the limited MuJoCo joint types to their qpos units."""
    if int(joint_type) == int(hinge_type):
        return "hinge"
    if int(joint_type) == int(slide_type):
        return "slide"
    raise ValueError(f"unsupported_limited_joint_type:{joint_type}")


def normalize_joint_limit_tolerance_policy(raw: Any = None) -> dict[str, Any]:
    """Validate explicit per-type and optional per-joint numerical envelopes."""
    if raw is None:
        raw = DEFAULT_JOINT_LIMIT_TOLERANCE_POLICY
    if not isinstance(raw, dict) or set(raw) != {"by_joint_type", "by_joint_name"}:
        raise ValueError("joint_limit_tolerance_policy_requires_type_and_name_maps")
    by_type = raw["by_joint_type"]
    by_name = raw["by_joint_name"]
    if not isinstance(by_type, dict) or set(by_type) != set(_TYPE_UNITS):
        raise ValueError("joint_limit_tolerance_policy_must_define_hinge_and_slide")
    if not isinstance(by_name, dict):
        raise ValueError("joint_limit_tolerance_by_name_must_be_an_object")

    normalized_type: dict[str, dict[str, Any]] = {}
    for kind, unit in _TYPE_UNITS.items():
        item = by_type[kind]
        if not isinstance(item, dict) or set(item) != {
            "strict_tolerance",
            "operational_tolerance",
            "unit",
        }:
            raise ValueError(f"joint_limit_{kind}_tolerance_requires_values_and_unit")
        strict_value = item["strict_tolerance"]
        operational_value = item["operational_tolerance"]
        if (
            any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) < 0.0
                for value in (strict_value, operational_value))
            or item["unit"] != unit
        ):
            raise ValueError(f"joint_limit_{kind}_tolerance_has_invalid_value_or_unit")
        normalized_type[kind] = {
            "strict_tolerance": float(strict_value),
            "operational_tolerance": float(operational_value),
            "unit": unit,
        }

    normalized_name: dict[str, dict[str, Any]] = {}
    for name, item in by_name.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(item, dict)
            or set(item) != {"operational_tolerance", "unit", "calibration_id"}
        ):
            raise ValueError("joint_limit_name_override_requires_value_unit_and_calibration_id")
        value = item["operational_tolerance"]
        unit = item["unit"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or unit not in _TYPE_UNITS.values()
            or not isinstance(item["calibration_id"], str)
            or not item["calibration_id"]
        ):
            raise ValueError(f"joint_limit_name_override_is_invalid:{name}")
        normalized_name[name] = {
            "operational_tolerance": float(value),
            "unit": unit,
            "calibration_id": item["calibration_id"],
        }
    return {"by_joint_type": normalized_type, "by_joint_name": normalized_name}


def evaluate_joint_limit(
    *,
    joint_name: str,
    joint_type: str,
    qpos: float,
    lower: float,
    upper: float,
    policy: Any = None,
) -> dict[str, Any]:
    """Report raw model-range compliance separately from operational tolerance."""
    normalized = normalize_joint_limit_tolerance_policy(policy)
    unit = _TYPE_UNITS.get(joint_type)
    if unit is None:
        raise ValueError(f"unsupported_limited_joint_kind:{joint_type}")
    override = normalized["by_joint_name"].get(joint_name)
    if override is not None and override["unit"] != unit:
        raise ValueError(f"joint_limit_override_unit_does_not_match_joint_type:{joint_name}")
    strict_threshold = normalized["by_joint_type"][joint_type]["strict_tolerance"]
    operational_threshold = (
        override["operational_tolerance"]
        if override is not None
        else normalized["by_joint_type"][joint_type]["operational_tolerance"]
    )
    values = (float(qpos), float(lower), float(upper))
    if not all(math.isfinite(value) for value in values) or lower > upper:
        raise ValueError(f"joint_limit_measurement_is_invalid:{joint_name}")
    outside = max(float(lower) - float(qpos), float(qpos) - float(upper), 0.0)
    return {
        "joint_name": joint_name,
        "joint_type": joint_type,
        "unit": unit,
        "qpos": float(qpos),
        "lower": float(lower),
        "upper": float(upper),
        "raw_outside_range_by": outside,
        "model_range_pass": outside == 0.0,
        "strict_numerical_tolerance": strict_threshold,
        "strict_criterion_pass": outside <= strict_threshold,
        "operational_tolerance": operational_threshold,
        "operational_pass": outside <= operational_threshold,
        "tolerance_source": (
            "joint_name_calibration" if override is not None else "joint_type_default"
        ),
        "calibration_id": override.get("calibration_id") if override else None,
    }


def acceptance_results(
    *, native_safety: bool, operational_safety: bool, strict_10mm: bool
) -> dict[str, bool]:
    """Keep the three acceptance labels independent and explicit."""
    return {
        "native_safety": bool(native_safety),
        "operational_safety": bool(operational_safety),
        "strict_10mm": bool(strict_10mm),
    }


def contact_safety_pass(item: dict[str, Any]) -> bool:
    """Check contact prohibitions independently from clearance thresholds."""
    clearance_events = {
        "geometry_clearance_requirement_violated",
        "carried_object_clearance_requirement_violated",
    }
    forbidden_contacts = [
        event
        for event in item.get("forbidden_contacts", [])
        if not any(event.get(name) is True for name in clearance_events)
    ]
    return bool(
        not forbidden_contacts
        and not item.get("self_contacts")
        and not item.get("unresolved_contacts")
        and not item.get("forbidden_object_contacts")
        and not item.get("unallowed_scene_contacts")
    )
