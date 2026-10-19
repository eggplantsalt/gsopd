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

import unittest

import numpy as np

from research.gs_opd.smoke.mink_osc_snapshot_smoke import (
    _snapshot_comparison,
    _validated_mink_trace,
    quaternion_angle_rad,
)


class LiveMinkOscSnapshotSmokeTests(unittest.TestCase):
    def _plan(self):
        return {
            "joint_trajectory_rad": [[0.0] * 7, [0.1] * 7],
            "sphere_trajectory_world_m": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            "eef_trajectory_world_m": [[0.2, 0.0, 0.5], [0.3, 0.0, 0.5]],
            "eef_quaternion_world_wxyz": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }

    def test_mink_trace_keeps_full_position_and_orientation_for_each_qpos(self):
        joints, spheres, poses = _validated_mink_trace(self._plan())
        self.assertEqual(joints.shape, (2, 7))
        self.assertEqual(spheres.shape, (2, 3))
        self.assertEqual(poses.shape, (2, 7))
        np.testing.assert_allclose(poses[1, :3], [0.3, 0.0, 0.5])
        np.testing.assert_allclose(poses[1, 3:], [0.0, 0.0, 0.0, 1.0])

    def test_mink_trace_rejects_missing_orientation_or_misaligned_arrays(self):
        plan = self._plan()
        plan.pop("eef_quaternion_world_wxyz")
        with self.assertRaisesRegex(ValueError, "quaternions"):
            _validated_mink_trace(plan)
        plan = self._plan()
        plan["sphere_trajectory_world_m"].pop()
        with self.assertRaisesRegex(ValueError, "shape"):
            _validated_mink_trace(plan)

    def test_mink_trace_rejects_nonfinite_and_nonunit_quaternions(self):
        plan = self._plan()
        plan["eef_quaternion_world_wxyz"][1] = [float("nan"), 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "quaternions"):
            _validated_mink_trace(plan)
        plan = self._plan()
        plan["eef_quaternion_world_wxyz"][1] = [2.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "unit"):
            _validated_mink_trace(plan)

    def test_snapshot_restore_comparison_is_strict_and_reports_numeric_error(self):
        expected = {
            "mujoco_integration_state": [0.0, 1.0],
            "robot_states": [{"eef_pos": [0.1, 0.2, 0.3]}],
        }
        self.assertTrue(_snapshot_comparison(expected, expected)["matched"])
        changed = {
            "mujoco_integration_state": [0.0, 1.00001],
            "robot_states": [{"eef_pos": [0.1, 0.2, 0.3]}],
        }
        result = _snapshot_comparison(expected, changed)
        self.assertFalse(result["matched"])
        self.assertAlmostEqual(result["maximum_numeric_error"], 1e-5)

    def test_quaternion_comparison_is_sign_invariant(self):
        self.assertAlmostEqual(
            quaternion_angle_rad([1, 0, 0, 0], [-1, 0, 0, 0]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
