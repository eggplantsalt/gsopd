"""Lightweight tests for Teacher-derived SAM 3 text prompting."""

from __future__ import annotations

import unittest

from research.gs_opd.grounding.sam3_smoke import _teacher_text_prompt


class Sam3TeacherPromptTests(unittest.TestCase):
    def test_default_prompt_remains_the_teacher_object_name(self) -> None:
        teacher_object = {
            "name": "akita black bowl",
            "visual_facts": [
                {"camera": "agentview_image", "statement": "A dark patterned bowl is visible."}
            ],
        }
        self.assertEqual(_teacher_text_prompt(teacher_object, False), "akita black bowl")

    def test_optional_prompt_adds_only_agentview_visual_facts(self) -> None:
        teacher_object = {
            "name": "akita black bowl",
            "visual_facts": [
                {"camera": "agentview_image", "statement": "A dark patterned bowl is visible."},
                {"camera": "robot0_eye_in_hand_image", "statement": "A patterned bowl fills the view."},
            ],
        }
        prompt = _teacher_text_prompt(teacher_object, True)
        self.assertIn("akita black bowl", prompt)
        self.assertIn("A dark patterned bowl is visible.", prompt)
        self.assertNotIn("fills the view", prompt)


if __name__ == "__main__":
    unittest.main()
