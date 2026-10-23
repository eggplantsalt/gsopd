# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Tests for the reusable 7D RRT-Connect contract."""

from __future__ import annotations

import unittest

import numpy as np

from research.gs_opd.repair.jointspace_rrt import (
    ConfigurationCheck,
    JointSpacePlannerConfig,
    JointSpaceRRTConnect,
    KinematicSample,
)


class JointSpaceRRTTests(unittest.TestCase):
    def test_returns_dense_seven_dimensional_path_with_full_checks(self):
        checked = []

        def check(qpos):
            checked.append(qpos.copy())
            return ConfigurationCheck(
                status="complete",
                minimum_forbidden_clearance_m=0.025,
                clearances_by_object_m={"forbidden": 0.025},
            )

        def kinematics(qpos):
            return KinematicSample(
                eef_position_world_m=qpos[:3].copy(),
                eef_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
                robot_geom_centers_world_m=np.asarray([qpos[:3]]),
            )

        planner = JointSpaceRRTConnect(
            np.zeros(7),
            np.ones(7),
            check,
            kinematics,
            config=JointSpacePlannerConfig(
                step_rad=0.2,
                goal_bias=0.8,
                max_iterations=40,
                max_config_checks=5000,
                max_candidate_routes=1,
                shortcut_attempts=8,
                edge_max_joint_delta_rad=0.02,
                edge_max_eef_translation_m=0.005,
                edge_max_geom_center_step_m=0.004,
                minimum_clearance_m=0.01,
                seed=13,
            ),
        )
        start = np.zeros(7)
        goal = np.full(7, 0.01)

        result = planner.plan(start, goal)

        self.assertEqual(result.status, "offline_feasible")
        np.testing.assert_allclose(result.qpos_path[0], start, atol=1e-8)
        np.testing.assert_allclose(result.qpos_path[-1], goal, atol=1e-8)
        self.assertGreater(len(result.qpos_path), 2)
        self.assertGreater(len(checked), 0)
        self.assertGreaterEqual(result.minimum_forbidden_clearance_m, 0.01)

    def test_hard_clearance_cannot_be_lowered(self):
        with self.assertRaisesRegex(ValueError, "cannot be below 10 mm"):
            JointSpacePlannerConfig(minimum_clearance_m=0.009).validate()


if __name__ == "__main__":
    unittest.main()
