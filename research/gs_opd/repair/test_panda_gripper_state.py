# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression tests for the verified robosuite PandaGripper action mapping."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from research.gs_opd.repair.panda_gripper_state import (
    parse_panda_gripper_closure,
    validate_panda_gripper_closure_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PANDA_SOURCE = REPO_ROOT.parent / "LIBERO-Safety/third_party/robosuite-1.4/robosuite/models/grippers/panda_gripper.py"


class PandaGripperStateTests(unittest.TestCase):
    @unittest.skipUnless(PANDA_SOURCE.is_file(), "local LIBERO-Safety robosuite source is not present")
    def test_vendored_format_action_emits_verified_close_and_open_pairs(self):
        tree = ast.parse(PANDA_SOURCE.read_text(encoding="utf-8"))
        panda_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PandaGripper"
        )
        format_action = next(
            node
            for node in panda_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "format_action"
        )
        namespace = {"np": np}
        exec(
            compile(ast.Module(body=[format_action], type_ignores=[]), str(PANDA_SOURCE), "exec"),
            namespace,
        )

        class Gripper:
            speed = 0.01
            dof = 1

            def __init__(self):
                self.current_action = np.zeros(self.dof)

        closing = Gripper()
        close_action = np.array([1.0])
        for _ in range(100):
            namespace["format_action"](closing, close_action)
        self.assertEqual(closing.current_action.shape, (2,))
        self.assertEqual(closing.current_action.tolist(), [-1.0, 1.0])
        self.assertTrue(parse_panda_gripper_closure(closing.current_action)["closed"])

        opening = Gripper()
        namespace["format_action"](opening, np.array([-1.0]))
        self.assertEqual(opening.current_action.tolist(), [0.01, -0.01])
        self.assertFalse(parse_panda_gripper_closure(opening.current_action)["closed"])

    def test_initial_single_dof_zero_is_valid_but_not_closed(self):
        parsed = parse_panda_gripper_closure([0.0])
        self.assertEqual(parsed["status"], "pass")
        self.assertEqual(parsed["representation"], "robosuite_initial_single_dof_state")
        self.assertEqual(parsed["raw_actuator_action"], [0.0])
        self.assertEqual(parsed["close_command_fraction"], 0.0)
        self.assertFalse(parsed["closed"])

    def test_verified_negative_positive_pair_is_closed(self):
        parsed = parse_panda_gripper_closure([-1.0, 1.0])
        self.assertEqual(parsed["status"], "pass")
        self.assertEqual(parsed["raw_actuator_action"], [-1.0, 1.0])
        self.assertEqual(parsed["close_command_fraction"], 1.0)
        self.assertTrue(parsed["closed"])

    def test_opening_pair_is_not_misread_as_closed(self):
        parsed = parse_panda_gripper_closure([1.0, -1.0])
        self.assertEqual(parsed["status"], "pass")
        self.assertEqual(parsed["state"], "opening")
        self.assertEqual(parsed["close_command_fraction"], 0.0)
        self.assertFalse(parsed["closed"])

    def test_partial_close_retains_fraction_and_fails_threshold(self):
        parsed = parse_panda_gripper_closure([-0.4, 0.4])
        self.assertEqual(parsed["status"], "pass")
        self.assertAlmostEqual(parsed["close_command_fraction"], 0.4)
        self.assertFalse(parsed["closed"])
        self.assertEqual(parsed["reason"], "panda_gripper_close_command_below_threshold")

    def test_threshold_is_applied_to_verified_close_direction_only(self):
        self.assertTrue(parse_panda_gripper_closure([-0.9, 0.9])["closed"])
        self.assertFalse(parse_panda_gripper_closure([0.9, -0.9])["closed"])

    def test_invalid_shapes_direction_and_values_fail_closed(self):
        cases = (
            ([0.2], "unexpected_nonzero_single_dof_panda_state"),
            ([0.1, 0.1], "panda_actuator_pair_is_not_equal_and_opposite"),
            ([0.0, 0.2], "panda_actuator_pair_is_not_equal_and_opposite"),
            ([0.0, 0.0, 0.0], "invalid_panda_current_action_shape"),
            ([float("nan"), 0.0], "gripper_current_action_contains_non_finite_values"),
            ([-1.01, 1.01], "gripper_actuator_value_outside_normalized_range"),
        )
        for raw, expected_reason in cases:
            with self.subTest(raw=raw):
                parsed = parse_panda_gripper_closure(raw)
                self.assertEqual(parsed["status"], "unavailable")
                self.assertFalse(parsed["closed"])
                self.assertIn(expected_reason, parsed["reason"])

    def test_branch_acceptance_reparses_raw_actuators_and_requires_matching_report(self):
        raw = [-0.95, 0.95]
        report = parse_panda_gripper_closure(raw)
        accepted, parsed, reason = validate_panda_gripper_closure_evidence(raw, report)
        self.assertTrue(accepted)
        self.assertIsNone(reason)
        self.assertTrue(parsed["closed"])

        accepted, _, reason = validate_panda_gripper_closure_evidence(
            [0.95, -0.95], parse_panda_gripper_closure([0.95, -0.95])
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "panda_gripper_close_command_below_threshold")

        corrupt_report = dict(report, close_command_fraction=0.5)
        accepted, _, reason = validate_panda_gripper_closure_evidence(raw, corrupt_report)
        self.assertFalse(accepted)
        self.assertEqual(reason, "gripper_closure_report_conflicts_with_raw_actuator_action")

    def test_bridge_and_tracker_share_the_same_parser_contract(self):
        bridge = (REPO_ROOT / "research/gs_opd/smoke/safety_bridge.py").read_text(
            encoding="utf-8"
        )
        tracker = (REPO_ROOT / "research/gs_opd/repair/trajectory_tracking.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def _parse_tracking_gripper_closure(current_action):", bridge)
        self.assertIn("return parse_panda_gripper_closure(current_action)", bridge)
        self.assertIn("_parse_tracking_gripper_closure(getattr(gripper, \"current_action\", None))", bridge)
        self.assertIn("validate_panda_gripper_closure_evidence", tracker)
        self.assertIn("step_grasp.get(\"gripper_closure\")", tracker)


if __name__ == "__main__":
    unittest.main()
