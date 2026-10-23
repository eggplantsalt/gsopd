# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for composing a verified grasp transform along a candidate EEF path."""

from __future__ import annotations

import unittest

import numpy as np
import mujoco

from research.gs_opd.repair.mujoco_jointspace_planning import (
    apply_carried_object_pose_to_scratch,
    carried_object_world_transform,
)


class CarriedObjectGeometryTests(unittest.TestCase):
    def test_object_pose_tracks_candidate_eef_pose(self):
        world_eef = np.eye(4)
        world_eef[:3, 3] = [0.4, -0.2, 0.8]
        eef_object = np.eye(4)
        eef_object[:3, 3] = [0.0, 0.0, 0.12]
        world_object = carried_object_world_transform(world_eef, eef_object)
        np.testing.assert_allclose(world_object[:3, 3], [0.4, -0.2, 0.92])

        moved_eef = np.eye(4)
        moved_eef[:3, 3] = [0.6, 0.1, 0.7]
        moved_object = carried_object_world_transform(moved_eef, eef_object)
        np.testing.assert_allclose(moved_object[:3, 3], [0.6, 0.1, 0.82])

    def test_malformed_or_nonrigid_transform_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "rigid_4x4"):
            carried_object_world_transform(np.eye(3), np.eye(4))
        reflected = np.eye(4)
        reflected[0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "rigid_4x4"):
            carried_object_world_transform(reflected, np.eye(4))

    def test_mujoco_scratch_free_root_tracks_eef_without_touching_main_data(self):
        model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="eef_body">
              <joint name="x" type="slide" axis="1 0 0"/>
              <joint name="y" type="slide" axis="0 1 0"/>
              <joint name="z" type="slide" axis="0 0 1"/>
              <geom type="sphere" size="0.01" mass="0.1" contype="0" conaffinity="0"/>
              <site name="eef" size="0.005"/>
            </body>
            <body name="held_object">
              <freejoint name="held_free"/>
              <geom type="sphere" size="0.02"/>
            </body>
          </worldbody>
        </mujoco>
        """)
        main = mujoco.MjData(model)
        scratch = mujoco.MjData(model)
        main.qpos[:3] = [0.1, 0.2, 0.3]
        main.qpos[3:6] = [0.1, 0.2, 0.4]
        mujoco.mj_forward(model, main)
        main_state = main.qpos.copy()

        scratch.qpos[:] = main.qpos
        scratch.qpos[:3] = [0.4, -0.1, 0.5]
        mujoco.mj_forward(model, scratch)
        relative = np.eye(4)
        relative[:3, 3] = [0.0, 0.0, 0.1]
        carried = {"eef_to_object_transform": relative, "free_joint_qpos_adr": 3}
        expected = apply_carried_object_pose_to_scratch(
            mujoco, scratch, int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef")), carried
        )

        object_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "held_object"))
        np.testing.assert_allclose(scratch.xpos[object_body_id], expected[:3, 3], atol=1e-8)
        np.testing.assert_allclose(main.qpos, main_state, atol=0.0)
        self.assertFalse(np.allclose(scratch.xpos[object_body_id], main.xpos[object_body_id]))


if __name__ == "__main__":
    unittest.main()
