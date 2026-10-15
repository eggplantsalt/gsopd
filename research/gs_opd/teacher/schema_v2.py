"""Independent strict Schema and Validator for GS-OPD Safety IR v2."""

from __future__ import annotations

import json
import math
import re
from typing import Any

SCHEMA_VERSION = "gs_opd.safety_teacher.v2"

# Keep these semantic labels aligned with the v1 Teacher vocabulary without
# importing or changing the v1 parser or its call path.
CAMERA_NAMES = ("agentview_image", "robot0_eye_in_hand_image")
OBJECT_ROLES = (
    "target_object",
    "placement_target",
    "avoid_object",
    "tool",
    "obstacle",
    "other",
    "unknown",
)
VISIBILITY_VALUES = ("visible", "partially_visible", "not_visible", "uncertain")
ROBOT_PARTS = (
    "gripper",
    "left_finger",
    "right_finger",
    "end_effector",
    "wrist",
    "arm",
    "base",
    "other",
    "unknown",
)
MOBILITY_VALUES = ("likely_movable", "likely_fixed", "unknown")
PREDICATES = ("contact",)
DECISIONS = ("allow", "forbid", "unknown")
PHASES = ("any",)

_STRING_SCHEMA = {"type": "string", "minLength": 1}
_FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["camera", "statement"],
    "properties": {
        "camera": {"enum": list(CAMERA_NAMES)},
        "statement": _STRING_SCHEMA,
    },
}
_INFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["statement", "confidence"],
    "properties": {
        "statement": _STRING_SCHEMA,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

SAFETY_OUTPUT_V2_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "GS-OPD Safety IR v2",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "objects", "rules", "global_uncertainties"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "objects": {
            "type": "array",
            "description": "Object ids must be unique within this array.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "name",
                    "role",
                    "visibility",
                    "mobility",
                    "visual_facts",
                    "inferences",
                    "uncertainties",
                ],
                "properties": {
                    "id": _STRING_SCHEMA,
                    "name": _STRING_SCHEMA,
                    "role": {"enum": list(OBJECT_ROLES)},
                    "visibility": {"enum": list(VISIBILITY_VALUES)},
                    "mobility": {"enum": list(MOBILITY_VALUES)},
                    "visual_facts": {
                        "type": "array",
                        "items": _FACT_SCHEMA,
                    },
                    "inferences": {
                        "type": "array",
                        "items": _INFERENCE_SCHEMA,
                    },
                    "uncertainties": {
                        "type": "array",
                        "items": _STRING_SCHEMA,
                    },
                },
            },
        },
        "rules": {
            "type": "array",
            "description": (
                "Rule ids must be unique within this array; target_id must reference "
                "an object id."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "subject_parts",
                    "target_id",
                    "predicate",
                    "decision",
                    "phase",
                    "reason",
                ],
                "properties": {
                    "id": _STRING_SCHEMA,
                    "subject_parts": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"enum": list(ROBOT_PARTS)},
                    },
                    "target_id": _STRING_SCHEMA,
                    "predicate": {"enum": list(PREDICATES)},
                    "decision": {"enum": list(DECISIONS)},
                    "phase": {"enum": list(PHASES)},
                    "reason": _STRING_SCHEMA,
                },
            },
        },
        "global_uncertainties": {
            "type": "array",
            "items": _STRING_SCHEMA,
        },
    },
}

_FORBIDDEN_PHYSICAL_ID_TEXT = re.compile(
    r"\b(?:geom|body)(?:[\s_-]*id|[#:\s_-]*\d+)\b",
    flags=re.IGNORECASE,
)


class TeacherOutputV2Error(ValueError):
    """Raised when Safety IR v2 JSON or its semantic references are invalid."""


def _fail(path: str, expected: str) -> None:
    raise TeacherOutputV2Error(f"{path}: expected {expected}")


def _check_keys(value: Any, expected: set[str], path: str) -> None:
    if not isinstance(value, dict):
        _fail(path, "an object")
    if any(not isinstance(key, str) for key in value):
        _fail(path, "an object with string field names")
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if extra:
            details.append(f"unexpected fields {extra}")
        raise TeacherOutputV2Error(f"{path}: " + "; ".join(details))


def _check_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "a non-empty string")
    return value


def _check_enum(value: Any, options: tuple[str, ...], path: str) -> str:
    if not isinstance(value, str) or value not in options:
        _fail(path, f"one of {list(options)}")
    return value


def _check_string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        _fail(path, "an array of non-empty strings")
    for index, item in enumerate(value):
        _check_text(item, f"{path}[{index}]")
    return value


def _walk_strings(value: Any, path: str = "$output"):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")


def validate_teacher_output_v2(value: Any) -> dict[str, Any]:
    """Validate v2 fields, unique identifiers, references, and text constraints."""
    top_fields = {"schema_version", "objects", "rules", "global_uncertainties"}
    _check_keys(value, top_fields, "$")
    if value["schema_version"] != SCHEMA_VERSION:
        _fail("$.schema_version", repr(SCHEMA_VERSION))

    objects = value["objects"]
    if not isinstance(objects, list):
        _fail("$.objects", "an array")
    object_ids: set[str] = set()
    object_fields = {
        "id",
        "name",
        "role",
        "visibility",
        "mobility",
        "visual_facts",
        "inferences",
        "uncertainties",
    }
    for object_index, item in enumerate(objects):
        path = f"$.objects[{object_index}]"
        _check_keys(item, object_fields, path)
        object_id = _check_text(item["id"], f"{path}.id")
        if object_id in object_ids:
            raise TeacherOutputV2Error(f"{path}.id: duplicate object id {object_id!r}")
        object_ids.add(object_id)
        _check_text(item["name"], f"{path}.name")
        _check_enum(item["role"], OBJECT_ROLES, f"{path}.role")
        _check_enum(item["visibility"], VISIBILITY_VALUES, f"{path}.visibility")
        _check_enum(item["mobility"], MOBILITY_VALUES, f"{path}.mobility")

        visual_facts = item["visual_facts"]
        if not isinstance(visual_facts, list):
            _fail(f"{path}.visual_facts", "an array")
        for fact_index, fact in enumerate(visual_facts):
            fact_path = f"{path}.visual_facts[{fact_index}]"
            _check_keys(fact, {"camera", "statement"}, fact_path)
            _check_enum(fact["camera"], CAMERA_NAMES, f"{fact_path}.camera")
            _check_text(fact["statement"], f"{fact_path}.statement")

        inferences = item["inferences"]
        if not isinstance(inferences, list):
            _fail(f"{path}.inferences", "an array")
        for inference_index, inference in enumerate(inferences):
            inference_path = f"{path}.inferences[{inference_index}]"
            _check_keys(inference, {"statement", "confidence"}, inference_path)
            _check_text(inference["statement"], f"{inference_path}.statement")
            confidence = inference["confidence"]
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or (isinstance(confidence, float) and not math.isfinite(confidence))
                or not 0 <= confidence <= 1
            ):
                _fail(f"{inference_path}.confidence", "a finite number from 0 to 1")

        _check_string_list(item["uncertainties"], f"{path}.uncertainties")

    rules = value["rules"]
    if not isinstance(rules, list):
        _fail("$.rules", "an array")
    rule_ids: set[str] = set()
    rule_fields = {
        "id",
        "subject_parts",
        "target_id",
        "predicate",
        "decision",
        "phase",
        "reason",
    }
    for rule_index, rule in enumerate(rules):
        path = f"$.rules[{rule_index}]"
        _check_keys(rule, rule_fields, path)
        rule_id = _check_text(rule["id"], f"{path}.id")
        if rule_id in rule_ids:
            raise TeacherOutputV2Error(f"{path}.id: duplicate rule id {rule_id!r}")
        rule_ids.add(rule_id)

        subject_parts = rule["subject_parts"]
        if not isinstance(subject_parts, list) or not subject_parts:
            _fail(f"{path}.subject_parts", "a non-empty array of robot-part labels")
        checked_parts = [
            _check_enum(part, ROBOT_PARTS, f"{path}.subject_parts[{part_index}]")
            for part_index, part in enumerate(subject_parts)
        ]
        if len(set(checked_parts)) != len(checked_parts):
            _fail(f"{path}.subject_parts", "unique robot-part labels")

        target_id = _check_text(rule["target_id"], f"{path}.target_id")
        if target_id not in object_ids:
            raise TeacherOutputV2Error(
                f"{path}.target_id: no object with id {target_id!r}"
            )
        _check_enum(rule["predicate"], PREDICATES, f"{path}.predicate")
        _check_enum(rule["decision"], DECISIONS, f"{path}.decision")
        _check_enum(rule["phase"], PHASES, f"{path}.phase")
        _check_text(rule["reason"], f"{path}.reason")

    _check_string_list(value["global_uncertainties"], "$.global_uncertainties")
    for path, text in _walk_strings(value):
        if _FORBIDDEN_PHYSICAL_ID_TEXT.search(text):
            raise TeacherOutputV2Error(
                f"{path}: MuJoCo geom/body IDs are forbidden; use semantic labels"
            )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token} is not allowed")


def parse_teacher_output_v2(raw_output: str) -> dict[str, Any]:
    """Parse strict JSON and validate it as GS-OPD Safety IR v2."""
    if not isinstance(raw_output, str):
        _fail("$raw_output", "a string containing JSON")
    try:
        value = json.loads(
            raw_output,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise TeacherOutputV2Error(
            f"invalid JSON at character {exc.pos}: {exc.msg}"
        ) from exc
    except ValueError as exc:
        raise TeacherOutputV2Error(f"invalid JSON: {exc}") from exc
    return validate_teacher_output_v2(value)
