import unittest

import numpy as np

from research.gs_opd.repair.mujoco_jointspace_planning import (
    _bilateral_finger_contact_result,
    _candidate_gripper_close_samples,
    _candidate_robot_contact_violation,
)


class GraspScreenContactPolicyTests(unittest.TestCase):
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
