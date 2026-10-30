import unittest

import numpy as np

from research.gs_opd.repair.mujoco_jointspace_planning import (
    _bilateral_finger_contact_result,
    _candidate_gripper_close_samples,
    _candidate_closed_gripper_qpos,
    _candidate_eef_to_live_object_transform,
    _candidate_robot_contact_violation,
    _minimum_distance,
    _verified_static_pairs_for_scene_group,
)


class GraspScreenContactPolicyTests(unittest.TestCase):
    def test_candidate_width_derives_its_own_panda_close_posture(self):
        qpos, source = _candidate_closed_gripper_qpos(
            {"estimated_clamp_width_m": 0.01868, "maximum_opening_m": 0.08},
            [0.001507, -0.002057],
            [[0.0, 0.04], [-0.04, 0.0]],
            pad_gap_joint_offset_m=0.001,
        )
        np.testing.assert_allclose(qpos, [0.00984, -0.00984])
        self.assertIn("active_MuJoCo_pad_gap_calibration", source)

    def test_calibrated_narrow_width_matches_physical_successful_grasp_joint_gap(self):
        qpos, _ = _candidate_closed_gripper_qpos(
            {"estimated_clamp_width_m": 0.00256},
            [0.001507067627357956, -0.002056929114271362],
            [[0.0, 0.04], [-0.04, 0.0]],
            pad_gap_joint_offset_m=0.001,
        )
        observed_stable_joint_gap = 0.001507067627357956 - (-0.002056929114271362)
        self.assertAlmostEqual(float(qpos[0] - qpos[1]), observed_stable_joint_gap, places=5)

    def test_candidate_close_posture_falls_back_only_without_width_metadata(self):
        fallback = [0.001507, -0.002057]
        qpos, source = _candidate_closed_gripper_qpos(
            {"candidate_id": "legacy"},
            fallback,
            [[0.0, 0.04], [-0.04, 0.0]],
            pad_gap_joint_offset_m=0.001,
        )
        np.testing.assert_array_equal(qpos, fallback)
        self.assertIn("fallback_missing_candidate_width", source)

    def test_candidate_close_posture_rejects_invalid_or_unreachable_width(self):
        limits = [[0.0, 0.04], [-0.04, 0.0]]
        fallback = [0.001507, -0.002057]
        for candidate in (
            {"estimated_clamp_width_m": 0.0},
            {"estimated_clamp_width_m": float("nan")},
            {"estimated_clamp_width_m": 0.05, "maximum_opening_m": 0.04},
            {"estimated_clamp_width_m": 0.09},
        ):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                _candidate_closed_gripper_qpos(candidate, fallback, limits)
        with self.assertRaisesRegex(ValueError, "active_pad_gap_joint_offset_m"):
            _candidate_closed_gripper_qpos(
                {"estimated_clamp_width_m": 0.02},
                fallback,
                limits,
                pad_gap_joint_offset_m=float("nan"),
            )

    def test_minimum_distance_can_return_exact_geom_pair_diagnostics(self):
        class FakeModel:
            geom_rbound = np.zeros(4)
            geom_bodyid = np.asarray([4, 5, 6, 7])

        class FakeData:
            geom_xpos = np.zeros((4, 3))

        class FakeMujoco:
            class mjtObj:
                mjOBJ_GEOM = 0
                mjOBJ_BODY = 1

            distances = {(0, 2): 0.004, (0, 3): 0.008, (1, 2): 0.006, (1, 3): 0.005}
            names = {0: "panda_finger1", 1: "panda_hand", 2: "moka_handle", 3: "moka_lid"}
            body_names = {4: "finger_body", 5: "hand_body", 6: "handle_body", 7: "lid_body"}

            @classmethod
            def mj_geomDistance(cls, model, data, geom_a, geom_b, cutoff, segment):
                segment[:] = [geom_a, 0.0, 0.0, geom_b, 0.0, 0.0]
                return cls.distances[(geom_a, geom_b)]

            @classmethod
            def mj_id2name(cls, model, object_type, geom_id):
                return cls.names[geom_id] if object_type == 0 else cls.body_names[geom_id]

        result = _minimum_distance(
            FakeMujoco, FakeModel(), FakeData(), [0, 1], [2, 3], return_detail=True
        )
        self.assertAlmostEqual(result["distance_m"], 0.004)
        self.assertEqual(result["pair"]["robot_geom_name"], "panda_finger1")
        self.assertEqual(result["pair"]["robot_body_name"], "finger_body")
        self.assertEqual(result["pair"]["object_geom_name"], "moka_handle")
        self.assertEqual(result["pair"]["object_body_name"], "handle_body")
        self.assertEqual(result["pair"]["closest_points_world"], [0.0, 0.0, 0.0, 2.0, 0.0, 0.0])

    def test_historical_lifted_object_pose_cannot_offset_candidate_close_contact(self):
        candidate_eef = np.eye(4)
        candidate_eef[:3, 3] = [-0.2758, -0.0964, 0.9981]
        object_at_candidate_close = np.eye(4)
        object_at_candidate_close[:3, 3] = [-0.2703, -0.1434, 0.9289]
        historical_lifted_object = object_at_candidate_close.copy()
        historical_lifted_object[:3, 3] += [-0.0026, 0.0007, 0.0374]

        live_held_transform = _candidate_eef_to_live_object_transform(
            candidate_eef, object_at_candidate_close
        )
        wrongly_offset_transform = _candidate_eef_to_live_object_transform(
            candidate_eef, historical_lifted_object
        )

        np.testing.assert_allclose(
            live_held_transform,
            np.linalg.inv(candidate_eef) @ object_at_candidate_close,
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(
                wrongly_offset_transform[:3, 3] - live_held_transform[:3, 3]
            )),
            float(np.linalg.norm(historical_lifted_object[:3, 3] - object_at_candidate_close[:3, 3])),
            places=10,
        )

    def test_planner_static_installation_exceptions_are_pair_and_scene_group_exact(self):
        verified = {(14, 90), (15, 91)}
        self.assertEqual(
            _verified_static_pairs_for_scene_group({90, 91}, {14}, verified),
            {(14, 90)},
        )
        self.assertEqual(
            _verified_static_pairs_for_scene_group({90, 91}, {15, 16}, verified),
            {(15, 91)},
        )
        self.assertEqual(
            _verified_static_pairs_for_scene_group({90, 91}, {14, 15}, {(90, 91)}),
            set(),
        )

    def test_close_sweep_stays_within_both_finger_joint_limits(self):
        samples = _candidate_gripper_close_samples(
            [0.035, -0.035],
            [0.01434, -0.01434],
            [[0.0, 0.04], [-0.04, 0.0]],
            sample_count=9,
        )
        self.assertEqual(len(samples), 9)
        self.assertEqual(samples[0]["alpha"], 0.0)
        self.assertEqual(samples[-1]["alpha"], 1.0)
        for sample in samples:
            qpos = np.asarray(sample["qpos"])
            self.assertTrue(np.all(qpos >= [0.0, -0.04]))
            self.assertTrue(np.all(qpos <= [0.04, 0.0]))

    def test_single_finger_contact_does_not_pass_as_a_grasp(self):
        result = _bilateral_finger_contact_result({"left": 0.0004, "right": 0.008})
        self.assertFalse(result["pass"])
        self.assertEqual(result["failed_sides"], {"right": "finger_side_does_not_reach_target"})

    def test_hand_contact_with_target_remains_illegal_during_grasp_check(self):
        violation = _candidate_robot_contact_violation(
            70, 100, 6, 22, -0.0001,
            robot_body_ids={6}, target_body_ids={22}, gripper_geom_ids={10, 11, 12, 13},
            forbidden_scene_body_ids={30}, allowed_target_body_ids={22},
            other_scene_body_ids={1}, allowed_static_installation_pair_ids=set(),
            allow_target_gripper=True,
        )
        self.assertIsNotNone(violation)
        self.assertIn("robot_scene_contact_geoms", violation)

    def test_strict_target_contact_policy_allows_only_authorized_pad_geometry(self):
        common = dict(
            robot_body_ids={6}, target_body_ids={22}, gripper_geom_ids={10, 11, 12, 13},
            forbidden_scene_body_ids={30}, allowed_target_body_ids={22},
            other_scene_body_ids={1}, allowed_static_installation_pair_ids=set(),
            allow_target_gripper=True, allowed_target_contact_geom_ids={11, 13},
        )
        self.assertIsNotNone(
            _candidate_robot_contact_violation(10, 100, 6, 22, -0.0001, **common)
        )
        self.assertIsNone(
            _candidate_robot_contact_violation(11, 100, 6, 22, -0.0019, **common)
        )
        self.assertIsNotNone(
            _candidate_robot_contact_violation(11, 100, 6, 22, -0.0021, **common)
        )

    def test_robot_contact_with_other_table_geometry_remains_illegal(self):
        violation = _candidate_robot_contact_violation(
            70, 14, 6, 1, -0.001,
            robot_body_ids={6}, target_body_ids={22}, gripper_geom_ids={10, 11, 12, 13},
            forbidden_scene_body_ids={30}, allowed_target_body_ids={22},
            other_scene_body_ids={1}, allowed_static_installation_pair_ids={(14, 90)},
            allow_target_gripper=True,
        )
        self.assertIsNotNone(violation)
        self.assertIn("robot_scene_contact_geoms", violation)

    def test_only_the_exact_audited_installation_pair_is_exempt(self):
        common = dict(
            robot_body_ids={6}, target_body_ids={22}, gripper_geom_ids={10, 11, 12, 13},
            forbidden_scene_body_ids={30}, allowed_target_body_ids={22},
            other_scene_body_ids={1}, allowed_static_installation_pair_ids={(14, 90)},
            allow_target_gripper=True,
        )
        self.assertIsNone(_candidate_robot_contact_violation(90, 14, 6, 1, -0.09946, **common))
        self.assertIsNotNone(_candidate_robot_contact_violation(90, 15, 6, 1, -0.001, **common))


if __name__ == "__main__":
    unittest.main()
