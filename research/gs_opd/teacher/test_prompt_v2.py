"""Lightweight standard-library tests for the Safety IR v2 prompt."""

from __future__ import annotations

import unittest

from research.gs_opd.teacher.prompt_v2 import (
    INFERENCE_FIELDS,
    OBJECT_FIELDS,
    RULE_FIELDS,
    SYSTEM_PROMPT,
    TOP_LEVEL_FIELDS,
    VISUAL_FACT_FIELDS,
    build_user_prompt,
)
from research.gs_opd.teacher.schema_v2 import SAFETY_OUTPUT_V2_JSON_SCHEMA


class SafetyIRV2PromptTests(unittest.TestCase):
    def test_prompt_fields_match_v2_json_schema(self) -> None:
        properties = SAFETY_OUTPUT_V2_JSON_SCHEMA["properties"]
        self.assertEqual(set(TOP_LEVEL_FIELDS), set(properties) - {"$schema"})
        self.assertEqual(
            set(OBJECT_FIELDS),
            set(properties["objects"]["items"]["properties"]),
        )
        self.assertEqual(
            set(VISUAL_FACT_FIELDS),
            set(
                properties["objects"]["items"]["properties"]["visual_facts"]
                ["items"]["properties"]
            ),
        )
        self.assertEqual(
            set(INFERENCE_FIELDS),
            set(
                properties["objects"]["items"]["properties"]["inferences"]
                ["items"]["properties"]
            ),
        )
        self.assertEqual(
            set(RULE_FIELDS),
            set(properties["rules"]["items"]["properties"]),
        )
        for field in (*TOP_LEVEL_FIELDS, *OBJECT_FIELDS, *VISUAL_FACT_FIELDS,
                      *INFERENCE_FIELDS, *RULE_FIELDS):
            with self.subTest(field=field):
                self.assertIn(field, SYSTEM_PROMPT)

    def test_prompt_covers_uncertainty_contact_and_identity_rules(self) -> None:
        for phrase in (
            "Do not list the robot",
            "local semantic IDs",
            "visual_facts",
            "inferences",
            "uncertainties",
            "decision=unknown",
            "unexpected collision",
            "do not infer a prohibition from role alone",
            "non-target status alone is not evidence of danger",
            "A placement goal does not by itself authorize",
            "MuJoCo",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, SYSTEM_PROMPT)

    def test_user_prompt_includes_instruction_and_two_camera_names(self) -> None:
        user_prompt = build_user_prompt(
            "pick up the red cup",
            ("agentview_image", "robot0_eye_in_hand_image"),
        )
        self.assertIn("pick up the red cup", user_prompt)
        self.assertIn("agentview_image", user_prompt)
        self.assertIn("robot0_eye_in_hand_image", user_prompt)
        self.assertIn("two attached RGB camera images", user_prompt)


if __name__ == "__main__":
    unittest.main()
