# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for live-pose object-frame grasp retargeting."""

from __future__ import annotations

import unittest

import numpy as np

from research.gs_opd.repair.grasp_retargeting import (
    GraspRetargetingTemplate,
    lateral_direction_away_from_nearest_obstacle_object_frame,
)


def _template():
    return GraspRetargetingTemplate.from_dict({
        "schema_version": "gs_opd_grasp_retargeting_template.v1",
        "source_trace_path": "archived_trace.npz",
        "source_trace_sha256": "a" * 64,
        "source_object_name": "archived_object",
        "source_body_name": "archived_object_main",
        "source_body_id_resolved_from_trace": 1,
        "stable_source_steps_1based": [10, 11],
        "source_first_finger_contact_step_1based": 10,
        "source_close_start_step_1based": 9,
        "source_stable_confirmation_step_1based": 11,
        "object_to_eef": [
            [1, 0, 0, 0.02],
            [0, 1, 0, 0.00],
            [0, 0, 1, 0.00],
            [0, 0, 0, 1],
        ],
        "approach_direction_object": [0, 0, -1],
        "approach_distance_m": 0.06,
        "open_command": -1.0,
        "close_schedule": [0.5, 1.0],
        "source_stable_contact_evidence": [{"step_1based": 10}, {"step_1based": 11}],
    })


class GraspRetargetingTests(unittest.TestCase):
    def test_lateral_candidate_points_away_from_nearest_live_obstacle(self):
        approach = np.asarray([0.1, -0.2, -0.97], dtype=np.float64)
        approach /= np.linalg.norm(approach)
        obstacle = np.asarray([-1.0, -1.0, 0.8], dtype=np.float64)
        direction = lateral_direction_away_from_nearest_obstacle_object_frame(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [obstacle],
            approach,
        )
        expected_away = -obstacle
        expected_lateral = expected_away - np.dot(expected_away, approach) * approach
        self.assertGreater(float(np.dot(direction, expected_lateral)), 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0)

    def test_away_offset_fails_closed_without_verified_obstacle_positions(self):
        with self.assertRaisesRegex(ValueError, "positions_are_unavailable"):
            lateral_direction_away_from_nearest_obstacle_object_frame(
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [],
                [0.0, 0.0, -1.0],
            )

    def test_pregrasp_uses_live_object_pose_and_verified_approach_distance(self):
        template = _template()
        first = template.retarget([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
        moved = template.retarget([1.1, 2.2, 3.3], [1.0, 0.0, 0.0, 0.0])

        np.testing.assert_allclose(
            first.pregrasp_position_world_m,
            [1.02, 2.0, 3.06],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            moved.pregrasp_position_world_m - first.pregrasp_position_world_m,
            [0.1, 0.2, 0.3],
            atol=1e-9,
        )
        self.assertAlmostEqual(
            np.linalg.norm(first.eef_position_world_m - first.pregrasp_position_world_m),
            0.06,
        )


if __name__ == "__main__":
    unittest.main()
