# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Small contract tests; these are not Panda physical success evidence."""

import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from research.gs_opd.smoke.mink_panda_physical_smoke import (
    JointReferenceSlewLimit,
    aggregate_contacts,
    assess_bypass_intent,
    classify_contact,
    resample_guide,
    restore,
    state_vector,
    torque_command,
)


class IsolatedPhysicsContracts(unittest.TestCase):
    def test_bypass_requires_requested_side_and_far_side_exit(self):
        self.assertEqual(
            assess_bypass_intent(
                [[0, 0, 0], [1, 1, 0], [2, 0, 0]], [0, 0, 0], [1, 0, 0], [0, 1, 0]
            )["status"],
            "pass",
        )
        self.assertEqual(
            assess_bypass_intent(
                [[0, 0, 0], [1, -1, 0], [2, 0, 0]], [0, 0, 0], [1, 0, 0], [0, 1, 0]
            )["status"],
            "fail",
        )
        self.assertEqual(
            assess_bypass_intent(
                [[0, 0, 0], [0.5, 1, 0]], [0, 0, 0], [1, 0, 0], [0, 1, 0]
            )["status"],
            "fail",
        )

    def test_reference_acceleration_constraint_bounds_the_next_velocity(self):
        limit = JointReferenceSlewLimit(SimpleNamespace(nv=1), [0], [0.2], 2.0)
        constraint = limit.compute_qp_inequalities(None, 0.05)
        self.assertTrue(
            np.all(constraint.G @ np.array([0.3 * 0.05]) <= constraint.h + 1e-12)
        )
        self.assertFalse(
            np.all(constraint.G @ np.array([0.31 * 0.05]) <= constraint.h + 1e-12)
        )

    def test_guide_discretization_keeps_endpoints_and_speed(self):
        points = resample_guide([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0]], 0.02)
        np.testing.assert_array_equal(points[0], [0, 0, 0])
        np.testing.assert_array_equal(points[-1], [0.1, 0.1, 0])
        self.assertLessEqual(
            np.linalg.norm(np.diff(points, axis=0), axis=1).max(), 0.001 + 1e-12
        )

    def test_contact_policy_distinguishes_grasp_obstacle_and_self(self):
        context = {
            "model": SimpleNamespace(geom_bodyid=np.array([1, 2, 3, 4, 5])),
            "carried_geoms": {1},
            "gripper_geoms": {0},
            "obstacle_geoms": {2},
            "robot_bodies": {1, 4},
        }
        self.assertEqual(classify_contact(0, 1, context), "allowed_teacher_grasp")
        self.assertEqual(
            classify_contact(1, 2, context), "teacher_forbidden_payload_obstacle"
        )
        self.assertEqual(classify_contact(0, 3, context), "robot_self_collision")
        self.assertEqual(classify_contact(2, 4, context), "other_scene_support")

    def test_contact_aggregation_counts_substeps_and_contact_points(self):
        def record(force):
            return {
                "geom_pair": [{"id": 1}, {"id": 2}],
                "occurrence_count": 1,
                "contact_point_count": 1,
                "force_sample_count": 1,
                "max_force_norm": force,
            }

        events = {}
        aggregate_contacts(events, [record(2), record(4)])
        aggregate_contacts(events, [record(3)])
        event = events[(1, 2)]
        self.assertEqual(event["occurrence_count"], 2)
        self.assertEqual(event["contact_point_count"], 3)
        self.assertEqual(event["max_force_norm"], 4)

    def make_motor(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco>
          <option gravity="0 0 0"/>
          <worldbody><body><joint name="j" type="hinge" range="-170 170"/>
          <geom type="sphere" size="0.1" mass="1" pos="0.2 0 0"/></body></worldbody>
          <actuator><motor joint="j" ctrllimited="true" ctrlrange="-2 2"/></actuator>
          </mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        context = {
            "qidx": np.array([0]),
            "didx": np.array([0]),
            "actuators": np.array([0]),
        }
        return model, data, context

    def test_real_torque_actuator_advances_physics(self):
        model, data, context = self.make_motor()
        data.ctrl[0] = torque_command(
            model, data, context, np.array([0.1]), np.array([0.0])
        )[0]
        mujoco.mj_step(model, data)
        self.assertGreater(data.time, 0)
        self.assertGreater(data.qvel[0], 0)

    def test_torque_overflow_rejected_without_clipping(self):
        model, data, context = self.make_motor()
        with self.assertRaisesRegex(ValueError, "outside_actual_motor_limits"):
            torque_command(model, data, context, np.array([100.0]), np.array([0.0]))
        self.assertEqual(data.ctrl[0], 0.0)

    def test_full_integration_state_restores_after_real_steps(self):
        model, data, _ = self.make_motor()
        initial = state_vector(model, data)
        data.ctrl[0] = 1.0
        for _ in range(10):
            mujoco.mj_step(model, data)
        restored = restore(model, initial)
        np.testing.assert_array_equal(state_vector(model, restored), initial)


if __name__ == "__main__":
    unittest.main()
