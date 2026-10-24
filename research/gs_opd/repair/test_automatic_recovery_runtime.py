# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Runtime safety-branch evidence classification tests."""

from __future__ import annotations

import unittest
import numpy as np

from research.gs_opd.repair.automatic_recovery import GoalSpec, Observation
from research.gs_opd.repair.automatic_recovery_runtime import (
    AutomaticRecoveryBridgeBackend,
    risk_branch_coverage_is_complete,
)
from research.gs_opd.repair.task_phase_controller import TaskPhaseController
from research.gs_opd.repair.mujoco_jointspace_planning import validate_finger_only_contact_geoms


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

    def test_recovery_goal_check_uses_approach_phase_tolerance(self):
        controller = TaskPhaseController({
            "objects": [{
                "object_id": "pot_2",
                "predicate": ["on", "pot_2", "support_region"],
                "goal_source": "verified_prior",
            }],
            "phase_pose_tolerances": {
                "approach": {"position_m": 0.008, "orientation_rad": 0.06}
            },
        })
        backend = object.__new__(AutomaticRecoveryBridgeBackend)
        backend.task_controller = controller
        backend.config_data = {
            "goal_position_tolerance_m": 0.012,
            "goal_orientation_tolerance_rad": 0.08,
        }
        goal = GoalSpec(
            goal_id="pot2-approach",
            phase="approach",
            source="test",
            source_artifacts=(),
            position_world_m=(0.0, 0.0, 0.0),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        observation = Observation(
            timestep=236,
            qpos=np.zeros(7),
            eef_position_world_m=np.array([0.0112673916, 0.0, 0.0]),
            eef_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            minimum_forbidden_clearance_m=0.02,
            clearance_complete=True,
            contacts_complete=True,
        )
        self.assertFalse(backend.is_recovery_goal_reached(observation, goal))
        observation.eef_position_world_m[:] = [0.0079, 0.0, 0.0]
        self.assertTrue(backend.is_recovery_goal_reached(observation, goal))

    def test_candidate_contact_exception_rejects_palm_or_hand_geometry(self):
        allowed = validate_finger_only_contact_geoms([76, 77], [79, 80], [76, 77, 79, 80])
        self.assertEqual(allowed, {76, 77, 79, 80})
        with self.assertRaisesRegex(ValueError, "only_the_verified_finger_geoms"):
            validate_finger_only_contact_geoms(
                [76, 77], [79, 80], [74, 76, 77, 79, 80]
            )


if __name__ == "__main__":
    unittest.main()
