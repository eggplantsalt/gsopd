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
    plan_route,
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

    def test_plan_route_emits_aligned_pose_and_joint_trajectories(self):
        from unittest.mock import patch

        import research.gs_opd.smoke.mink_panda_physical_smoke as smoke

        model = mujoco.MjModel.from_xml_string(
            f"""<mujoco><worldbody>
              <body name="arm">
                <joint name="arm_joint" type="hinge" axis="0 0 1" range="-3 3"/>
                <geom type="sphere" size="0.01" mass="0.01"/>
                <site name="{smoke.FRAME_NAME}" pos="0 0 0.5"/>
              </body>
              <body name="payload" pos="0 0 0.5">
                <freejoint name="payload_free"/>
                <geom type="sphere" size="0.005" mass="0.01"/>
              </body>
              <body name="primary_obstacle" pos="0.0005 0 0.5">
                <geom type="sphere" size="0.001"/>
              </body>
            </worldbody></mujoco>"""
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        initial = state_vector(model, data)
        route = {
            "candidate_id": "left-regression-route",
            "direction": "left",
            "reference_path": [[0.0, 0.0, 0.5], [0.001, 0.005, 0.5]],
            "grasp_evidence": {
                "initial_eef_sphere_center_relative_position_body_local_m": [
                    0.0,
                    0.0,
                    0.0,
                ],
                "initial_eef_body_relative_quaternion_local_wxyz": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            },
            "sphere_center_body_local_offset_m": [0.0, 0.0, 0.0],
            "geometric_candidate": {"direction_world": [0.0, 1.0, 0.0]},
        }
        obstacle_id = model.body("primary_obstacle").id
        ctx = {
            "model": model,
            "site": model.site(smoke.FRAME_NAME).id,
            "carried_id": model.body("payload").id,
            "carried_qadr": int(model.jnt_qposadr[model.joint("payload_free").id]),
            "didx": np.array([int(model.jnt_dofadr[model.joint("arm_joint").id])]),
            "qidx": np.array([int(model.jnt_qposadr[model.joint("arm_joint").id])]),
            "names": ["arm_joint"],
            "vmax": np.array([2.0]),
            "joint_ranges": np.array([[-3.0, 3.0]]),
            "obstacle_definitions": [
                {"object_id": "primary", "body_id": obstacle_id}
            ],
            "primary_obstacle_id": "primary",
        }
        args = SimpleNamespace(
            speed=10.0,
            max_steps=1,
            guide_tolerance=0.02,
            orientation_cost=0.0,
            joint_reference_speed=2.0,
            joint_reference_acceleration=10.0,
        )
        with (
            patch.object(smoke, "_make_mink_limits", return_value=[None, None]),
            patch.object(smoke, "collision_pair_pool", return_value=np.empty((0, 2), dtype=int)),
            patch.object(smoke, "active_collision_limit", return_value=None),
            patch.object(smoke, "scan_contacts", return_value=([], {})),
            patch.object(smoke.mink, "solve_ik", return_value=np.zeros(model.nv)),
        ):
            plan = plan_route(model, initial, route, ctx, args)

        trajectory_lengths = [
            len(plan["joint_trajectory_rad"]),
            len(plan["eef_trajectory_world_m"]),
            len(plan["eef_quaternion_world_wxyz"]),
            len(plan["sphere_trajectory_world_m"]),
        ]
        self.assertEqual(trajectory_lengths, [2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
