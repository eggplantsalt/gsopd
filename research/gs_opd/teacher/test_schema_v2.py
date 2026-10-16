"""Standard-library tests for the independent Safety IR v2 validator."""

from __future__ import annotations

import copy
import json
import unittest

from research.gs_opd.teacher.schema_v2 import (
    SAFETY_OUTPUT_V2_JSON_SCHEMA,
    SCHEMA_VERSION,
    TeacherOutputV2Error,
    parse_teacher_output_v2,
    validate_teacher_output_v2,
)


def _valid_output() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "objects": [
            {
                "id": "object-1",
                "name": "red cup",
                "role": "target_object",
                "visibility": "visible",
                "mobility": "likely_movable",
                "visual_facts": [
                    {
                        "camera": "agentview_image",
                        "statement": "A red cup is on the table.",
                    }
                ],
                "inferences": [
                    {"statement": "The cup may be the task target.", "confidence": 0.8}
                ],
                "uncertainties": [],
            }
        ],
        "rules": [
            {
                "id": "rule-1",
                "subject_parts": ["gripper", "left_finger"],
                "target_id": "object-1",
                "predicate": "contact",
                "decision": "unknown",
                "phase": "any",
                "reason": "The task does not specify whether contact is permitted.",
            }
        ],
        "global_uncertainties": [],
    }


class SafetyIRV2SchemaTests(unittest.TestCase):
    def test_valid_v2_payload_parses(self) -> None:
        payload = _valid_output()
        parsed = parse_teacher_output_v2(json.dumps(payload))
        self.assertEqual(parsed, payload)
        self.assertEqual(
            SAFETY_OUTPUT_V2_JSON_SCHEMA["properties"]["schema_version"]["const"],
            SCHEMA_VERSION,
        )

    def test_unknown_values_are_preserved_without_rule_generation(self) -> None:
        payload = _valid_output()
        payload["objects"][0]["role"] = "unknown"
        payload["objects"][0]["visibility"] = "uncertain"
        payload["objects"][0]["mobility"] = "unknown"
        payload["rules"][0]["decision"] = "unknown"

        validated = validate_teacher_output_v2(payload)

        self.assertEqual(validated["objects"][0]["role"], "unknown")
        self.assertEqual(validated["objects"][0]["visibility"], "uncertain")
        self.assertEqual(validated["objects"][0]["mobility"], "unknown")
        self.assertEqual(validated["rules"][0]["decision"], "unknown")

        payload["objects"][0]["role"] = "target_object"
        payload["rules"] = []
        self.assertEqual(validate_teacher_output_v2(payload)["rules"], [])

    def test_legacy_v2_without_object_contact_rules_remains_valid(self) -> None:
        payload = _valid_output()
        self.assertNotIn("object_contact_rules", payload)
        self.assertEqual(parse_teacher_output_v2(json.dumps(payload)), payload)
        self.assertIn(
            "object_contact_rules",
            SAFETY_OUTPUT_V2_JSON_SCHEMA["required"],
        )

    def test_object_contact_rule_is_validated_and_unknown_is_preserved(self) -> None:
        payload = _valid_output()
        second = copy.deepcopy(payload["objects"][0])
        second["id"] = "object-2"
        second["name"] = "small bottle"
        payload["objects"].append(second)
        payload["rules"] = []
        payload["object_contact_rules"] = [
            {
                "id": "object-rule-1",
                "subject_id": "object-1",
                "target_id": "object-2",
                "predicate": "contact",
                "decision": "unknown",
                "phase": "any",
                "reason": "The route relation is visually unclear.",
            }
        ]
        parsed = parse_teacher_output_v2(json.dumps(payload))
        self.assertEqual(parsed["object_contact_rules"][0]["decision"], "unknown")
        self.assertEqual(parsed["object_contact_rules"], payload["object_contact_rules"])

    def test_object_contact_rule_references_and_ids_are_strict(self) -> None:
        payload = _valid_output()
        second = copy.deepcopy(payload["objects"][0])
        second["id"] = "object-2"
        payload["objects"].append(second)
        payload["object_contact_rules"] = [
            {
                "id": "rule-1",
                "subject_id": "object-1",
                "target_id": "object-2",
                "predicate": "contact",
                "decision": "forbid",
                "phase": "any",
                "reason": "The carried object should clear this obstacle.",
            }
        ]
        with self.assertRaisesRegex(TeacherOutputV2Error, "duplicate rule id"):
            validate_teacher_output_v2(payload)

        payload["rules"] = []
        payload["object_contact_rules"][0]["target_id"] = "missing"
        with self.assertRaisesRegex(TeacherOutputV2Error, "no object with id"):
            validate_teacher_output_v2(payload)

        payload["object_contact_rules"][0]["target_id"] = "object-1"
        with self.assertRaisesRegex(TeacherOutputV2Error, "different object id"):
            validate_teacher_output_v2(payload)

        payload["object_contact_rules"][0]["target_id"] = "object-2"
        payload["object_contact_rules"][0]["decision"] = "safe"
        with self.assertRaises(TeacherOutputV2Error):
            validate_teacher_output_v2(payload)

    def test_extra_or_missing_fields_are_rejected(self) -> None:
        payload = _valid_output()
        payload["objects"][0]["unexpected"] = "value"
        with self.assertRaisesRegex(TeacherOutputV2Error, "unexpected fields"):
            validate_teacher_output_v2(payload)

        payload = _valid_output()
        del payload["rules"][0]["reason"]
        with self.assertRaisesRegex(TeacherOutputV2Error, "missing fields"):
            validate_teacher_output_v2(payload)

    def test_invalid_target_id_is_rejected(self) -> None:
        payload = _valid_output()
        payload["rules"][0]["target_id"] = "missing-object"
        with self.assertRaisesRegex(TeacherOutputV2Error, "no object with id"):
            validate_teacher_output_v2(payload)

    def test_duplicate_object_and_rule_ids_are_rejected(self) -> None:
        payload = _valid_output()
        payload["objects"].append(copy.deepcopy(payload["objects"][0]))
        with self.assertRaisesRegex(TeacherOutputV2Error, "duplicate object id"):
            validate_teacher_output_v2(payload)

        payload = _valid_output()
        duplicate_rule = copy.deepcopy(payload["rules"][0])
        duplicate_rule["target_id"] = "object-1"
        payload["rules"].append(duplicate_rule)
        with self.assertRaisesRegex(TeacherOutputV2Error, "duplicate rule id"):
            validate_teacher_output_v2(payload)

    def test_empty_subject_parts_are_rejected(self) -> None:
        payload = _valid_output()
        payload["rules"][0]["subject_parts"] = []
        with self.assertRaisesRegex(TeacherOutputV2Error, "non-empty array"):
            validate_teacher_output_v2(payload)

    def test_duplicate_subject_parts_are_rejected_by_validator_and_parser(self) -> None:
        payload = _valid_output()
        payload["rules"][0]["subject_parts"] = ["gripper", "gripper"]
        with self.assertRaisesRegex(TeacherOutputV2Error, "unique robot-part labels"):
            validate_teacher_output_v2(payload)
        with self.assertRaisesRegex(TeacherOutputV2Error, "unique robot-part labels"):
            parse_teacher_output_v2(json.dumps(payload))

    def test_illegal_enums_are_rejected(self) -> None:
        invalid_values = (
            ("objects", "role", "person"),
            ("objects", "visibility", "hidden"),
            ("objects", "mobility", "static"),
            ("rules", "predicate", "near"),
            ("rules", "decision", "safe"),
            ("rules", "phase", "before_contact"),
        )
        for section, field, invalid_value in invalid_values:
            with self.subTest(section=section, field=field):
                payload = _valid_output()
                payload[section][0][field] = invalid_value
                with self.assertRaises(TeacherOutputV2Error):
                    validate_teacher_output_v2(payload)

        payload = _valid_output()
        payload["rules"][0]["subject_parts"] = ["camera"]
        with self.assertRaises(TeacherOutputV2Error):
            validate_teacher_output_v2(payload)

    def test_confidence_must_be_finite_numeric_and_in_range(self) -> None:
        for invalid_confidence in (-0.1, 1.1, True, float("inf")):
            with self.subTest(confidence=invalid_confidence):
                payload = _valid_output()
                payload["objects"][0]["inferences"][0][
                    "confidence"
                ] = invalid_confidence
                with self.assertRaisesRegex(TeacherOutputV2Error, "confidence"):
                    validate_teacher_output_v2(payload)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        raw = (
            '{"schema_version":"gs_opd.safety_teacher.v2",'
            '"schema_version":"gs_opd.safety_teacher.v2",'
            '"objects":[],"rules":[],"global_uncertainties":[]}'
        )
        with self.assertRaisesRegex(TeacherOutputV2Error, "duplicate JSON key"):
            parse_teacher_output_v2(raw)

    def test_mujoco_geom_and_body_ids_are_rejected(self) -> None:
        payload = _valid_output()
        payload["objects"][0]["name"] = "geom_12"
        with self.assertRaisesRegex(TeacherOutputV2Error, "geom/body IDs"):
            validate_teacher_output_v2(payload)


if __name__ == "__main__":
    unittest.main()
