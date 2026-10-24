# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression tests for unit-aware joint-limit diagnostics."""

from __future__ import annotations

import unittest

from research.gs_opd.smoke.joint_limit_policy import (
    acceptance_results,
    contact_safety_pass,
    evaluate_joint_limit,
    joint_type_name,
    normalize_joint_limit_tolerance_policy,
)


CALIBRATED_GRIPPER_POLICY = {
    "by_joint_type": {
        "hinge": {"strict_tolerance": 1e-8, "operational_tolerance": 1e-8, "unit": "rad"},
        "slide": {"strict_tolerance": 1e-8, "operational_tolerance": 1e-8, "unit": "m"},
    },
    "by_joint_name": {
        "gripper0_finger_joint1": {
            "operational_tolerance": 8e-6,
            "unit": "m",
            "calibration_id": "episode9_control_endpoint_max_excursion_7.374um",
        },
        "gripper0_finger_joint2": {
            "operational_tolerance": 8e-6,
            "unit": "m",
            "calibration_id": "episode9_control_endpoint_max_excursion_7.374um",
        },
    },
}


class JointLimitPolicyTests(unittest.TestCase):
    def test_mujoco_hinge_and_slide_use_radians_and_meters(self):
        self.assertEqual(joint_type_name(3, 3, 2), "hinge")
        self.assertEqual(joint_type_name(2, 3, 2), "slide")
        with self.assertRaisesRegex(ValueError, "unsupported_limited_joint_type"):
            joint_type_name(1, 3, 2)

    def test_calibrated_gripper_excursion_is_operational_but_not_strict(self):
        result = evaluate_joint_limit(
            joint_name="gripper0_finger_joint2",
            joint_type="slide",
            qpos=-0.04000417971758497,
            lower=-0.04,
            upper=0.0,
            policy=CALIBRATED_GRIPPER_POLICY,
        )
        self.assertAlmostEqual(result["raw_outside_range_by"], 4.17971758497e-6)
        self.assertFalse(result["model_range_pass"])
        self.assertFalse(result["strict_criterion_pass"])
        self.assertTrue(result["operational_pass"])
        self.assertEqual(result["unit"], "m")
        self.assertEqual(result["tolerance_source"], "joint_name_calibration")

    def test_slide_excursion_over_calibrated_envelope_fails_operational_check(self):
        result = evaluate_joint_limit(
            joint_name="gripper0_finger_joint2",
            joint_type="slide",
            qpos=-0.04000801,
            lower=-0.04,
            upper=0.0,
            policy=CALIBRATED_GRIPPER_POLICY,
        )
        self.assertFalse(result["operational_pass"])

    def test_hinge_tolerance_remains_radians_and_does_not_inherit_slide_override(self):
        result = evaluate_joint_limit(
            joint_name="robot0_joint1",
            joint_type="hinge",
            qpos=-2.0 - 2e-8,
            lower=-2.0,
            upper=2.0,
            policy=CALIBRATED_GRIPPER_POLICY,
        )
        self.assertEqual(result["unit"], "rad")
        self.assertFalse(result["operational_pass"])

    def test_joint_override_rejects_a_unit_mismatch(self):
        bad = {
            **CALIBRATED_GRIPPER_POLICY,
            "by_joint_name": {
                "robot0_joint1": {
                    "operational_tolerance": 8e-6,
                    "unit": "m",
                    "calibration_id": "incorrect-unit",
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "unit_does_not_match"):
            evaluate_joint_limit(
                joint_name="robot0_joint1",
                joint_type="hinge",
                qpos=0.0,
                lower=-2.0,
                upper=2.0,
                policy=bad,
            )

    def test_acceptance_results_are_reported_independently(self):
        self.assertEqual(
            acceptance_results(
                native_safety=True,
                operational_safety=True,
                strict_10mm=False,
            ),
            {
                "native_safety": True,
                "operational_safety": True,
                "strict_10mm": False,
            },
        )

    def test_native_contact_verdict_does_not_conflate_clearance_with_contact(self):
        self.assertTrue(contact_safety_pass({
            "forbidden_contacts": [{
                "geometry_clearance_requirement_violated": True,
                "object_id": "moka_pot_2",
                "distance_m": 0.004,
                "minimum_required_m": 0.010,
            }],
        }))
        self.assertFalse(contact_safety_pass({
            "forbidden_contacts": [{
                "allowed_pair_penetration_exceeded": True,
                "geom_ids": [77, 116],
                "distance_m": -0.0021,
            }],
        }))

    def test_tolerance_policy_rejects_incorrect_unit_label(self):
        bad = {
            "by_joint_type": {
                "hinge": {
                    "strict_tolerance": 1e-8,
                    "operational_tolerance": 1e-8,
                    "unit": "m",
                },
                "slide": {
                    "strict_tolerance": 1e-8,
                    "operational_tolerance": 0.0,
                    "unit": "m",
                },
            },
            "by_joint_name": {},
        }
        with self.assertRaisesRegex(ValueError, "hinge_tolerance_has_invalid_value_or_unit"):
            normalize_joint_limit_tolerance_policy(bad)


if __name__ == "__main__":
    unittest.main()
