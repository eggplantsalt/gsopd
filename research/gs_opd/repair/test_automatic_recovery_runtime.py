# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Runtime safety-branch evidence classification tests."""

from __future__ import annotations

import unittest

from research.gs_opd.repair.automatic_recovery_runtime import (
    risk_branch_coverage_is_complete,
)


def _monitor(*, unsafe=False, missing_substeps=False):
    expected = 25
    count = 24 if missing_substeps else expected
    geometry_status = "unsafe" if unsafe else "complete"
    substep_status = "unsafe" if unsafe else "complete"
    return {
        "expected_physics_substeps_per_control_step": expected,
        "captured_physics_substeps": count,
        "contact_substeps": count,
        "control_steps": [
            {
                "expected_physics_substeps": expected,
                "captured_physics_substeps": count,
                "unobserved_expected_substeps": expected - count,
                "contact_samples": {"contact": count, "no_contact": 0, "unavailable": 0},
                "capture_error": "substep_geometry_safety_unsafe" if unsafe else None,
                "status": "incomplete" if missing_substeps or unsafe else "complete",
            }
        ],
        "geometry_monitoring": {
            "status": geometry_status,
            "substeps": [{"status": substep_status} for _ in range(count)],
        },
    }


class AutomaticRecoveryRuntimeCoverageTests(unittest.TestCase):
    def test_complete_safe_branch_is_complete(self):
        self.assertTrue(risk_branch_coverage_is_complete(_monitor(), 1))

    def test_complete_unsafe_branch_is_known_unsafe_not_unknown(self):
        self.assertTrue(risk_branch_coverage_is_complete(_monitor(unsafe=True), 1))

    def test_missing_physical_substep_fails_coverage(self):
        self.assertFalse(
            risk_branch_coverage_is_complete(_monitor(missing_substeps=True), 1)
        )


if __name__ == "__main__":
    unittest.main()
