"""Strict JSON schema and validation for task-conditioned safety semantics."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "gs_opd.safety_teacher.v1"
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
CONTACT_RULES = ("allowed", "forbidden", "unknown")
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

_STRING_SCHEMA = {"type": "string", "minLength": 1}
SAFETY_OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "GS-OPD VLM Safety Teacher Output",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "objects", "global_uncertainties"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "description",
                    "role",
                    "visibility",
                    "contact_rule",
                    "robot_parts",
                    "reason",
                    "visual_facts",
                    "inferences",
                    "uncertainties",
                ],
                "properties": {
                    "name": _STRING_SCHEMA,
                    "description": _STRING_SCHEMA,
                    "role": {"enum": list(OBJECT_ROLES)},
                    "visibility": {"enum": list(VISIBILITY_VALUES)},
                    "contact_rule": {"enum": list(CONTACT_RULES)},
                    "robot_parts": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"enum": list(ROBOT_PARTS)},
                    },
                    "reason": _STRING_SCHEMA,
                    "visual_facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["camera", "statement"],
                            "properties": {
                                "camera": {"enum": list(CAMERA_NAMES)},
                                "statement": _STRING_SCHEMA,
                            },
                        },
                    },
                    "inferences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["statement", "confidence"],
                            "properties": {
                                "statement": _STRING_SCHEMA,
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                        },
                    },
                    "uncertainties": {
                        "type": "array",
                        "items": _STRING_SCHEMA,
                    },
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


class TeacherOutputError(ValueError):
    """Raised when a model response is not valid GS-OPD Teacher JSON."""


def _fail(path: str, expected: str) -> None:
    raise TeacherOutputError(f"{path}: expected {expected}")


def _check_keys(value: Mapping[str, Any], required: set[str], path: str) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if extra:
            details.append(f"unexpected fields {extra}")
        raise TeacherOutputError(f"{path}: " + "; ".join(details))


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
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")


def validate_teacher_output(value: Any) -> dict[str, Any]:
    """Validate the complete output shape and semantic safety constraints."""
    if not isinstance(value, dict):
        _fail("$", "a JSON object")
    _check_keys(value, {"schema_version", "objects", "global_uncertainties"}, "$")
    if value["schema_version"] != SCHEMA_VERSION:
        _fail("$.schema_version", f"{SCHEMA_VERSION!r}")

    objects = value["objects"]
    if not isinstance(objects, list):
        _fail("$.objects", "an array")
    for object_index, item in enumerate(objects):
        path = f"$.objects[{object_index}]"
        if not isinstance(item, dict):
            _fail(path, "an object")
        _check_keys(
            item,
            {
                "name",
                "description",
                "role",
                "visibility",
                "contact_rule",
                "robot_parts",
                "reason",
                "visual_facts",
                "inferences",
                "uncertainties",
            },
            path,
        )
        for field in ("name", "description", "reason"):
            _check_text(item[field], f"{path}.{field}")
        _check_enum(item["role"], OBJECT_ROLES, f"{path}.role")
        _check_enum(item["visibility"], VISIBILITY_VALUES, f"{path}.visibility")
        _check_enum(item["contact_rule"], CONTACT_RULES, f"{path}.contact_rule")

        robot_parts = item["robot_parts"]
        if not isinstance(robot_parts, list) or not robot_parts:
            _fail(f"{path}.robot_parts", "a non-empty array of robot-part labels")
        for part_index, part in enumerate(robot_parts):
            _check_enum(part, ROBOT_PARTS, f"{path}.robot_parts[{part_index}]")
        if len(set(robot_parts)) != len(robot_parts):
            _fail(f"{path}.robot_parts", "unique labels")

        visual_facts = item["visual_facts"]
        if not isinstance(visual_facts, list):
            _fail(f"{path}.visual_facts", "an array")
        for fact_index, fact in enumerate(visual_facts):
            fact_path = f"{path}.visual_facts[{fact_index}]"
            if not isinstance(fact, dict):
                _fail(fact_path, "an object")
            _check_keys(fact, {"camera", "statement"}, fact_path)
            _check_enum(fact["camera"], CAMERA_NAMES, f"{fact_path}.camera")
            _check_text(fact["statement"], f"{fact_path}.statement")

        inferences = item["inferences"]
        if not isinstance(inferences, list):
            _fail(f"{path}.inferences", "an array")
        for inference_index, inference in enumerate(inferences):
            inference_path = f"{path}.inferences[{inference_index}]"
            if not isinstance(inference, dict):
                _fail(inference_path, "an object")
            _check_keys(inference, {"statement", "confidence"}, inference_path)
            _check_text(inference["statement"], f"{inference_path}.statement")
            confidence = inference["confidence"]
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                _fail(f"{inference_path}.confidence", "a finite number from 0 to 1")

        _check_string_list(item["uncertainties"], f"{path}.uncertainties")

    _check_string_list(value["global_uncertainties"], "$.global_uncertainties")
    for path, text in _walk_strings(value):
        if _FORBIDDEN_PHYSICAL_ID_TEXT.search(text):
            raise TeacherOutputError(
                f"{path}: physical geom/body IDs are forbidden; use semantic labels only"
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


def parse_teacher_output(raw_output: str) -> dict[str, Any]:
    """Parse strict JSON, reject duplicate/non-finite values, then validate."""
    if not isinstance(raw_output, str):
        _fail("$raw_output", "a string containing JSON")
    try:
        value = json.loads(
            raw_output,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise TeacherOutputError(
            f"invalid JSON at character {exc.pos}: {exc.msg}"
        ) from exc
    except ValueError as exc:
        raise TeacherOutputError(f"invalid JSON: {exc}") from exc
    return validate_teacher_output(value)
