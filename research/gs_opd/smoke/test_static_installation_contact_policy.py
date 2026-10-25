"""Regression coverage for exact static-installation contact exceptions."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

try:
    import mujoco
except ImportError:  # The Bridge's LIBERO-Safety environment provides this dependency.
    mujoco = None


BRIDGE_PATH = Path(__file__).with_name("safety_bridge.py")


def _load_helpers():
    if mujoco is None:
        raise unittest.SkipTest("MuJoCo is only available in the LIBERO-Safety environment")
    tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
    wanted = {
        "_fixed_body_ancestor_chain",
        "_minimum_geom_clearance",
        "_static_installation_pairs_for_group",
        "_is_allowed_static_installation_contact",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"mujoco": mujoco, "np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BRIDGE_PATH), "exec"), namespace)
    return namespace


@unittest.skipIf(mujoco is None, "requires the LIBERO-Safety MuJoCo environment")
class StaticInstallationContactPolicyTests(unittest.TestCase):
    def setUp(self):
        self.helpers = _load_helpers()
        xml = """<mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 0.1" />
            <body name="fixed_mount" pos="0 0 0.05">
              <geom name="mount_collision" type="cylinder" size="0.1 0.2" />
            </body>
            <body name="moving_link" pos="0.4 0 0.03">
              <joint name="moving_slide" type="slide" axis="0 0 1" range="-0.1 0.1" />
              <geom name="moving_link_collision" type="sphere" size="0.1" />
            </body>
          </worldbody>
        </mujoco>"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.floor_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.mount_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "mount_collision"
        )
        self.moving_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_link_collision"
        )
        self.robot_geom_ids = {self.mount_geom, self.moving_geom}

    def test_exact_static_pair_does_not_hide_an_illegal_moving_link_floor_contact(self):
        policy = {
            "robot_geom_ids": self.robot_geom_ids,
            "allowed_static_installation_contact_pair_ids": {
                tuple(sorted((self.mount_geom, self.floor_geom)))
            },
        }
        floor_group = {"geom_ids": {self.floor_geom}}

        excluded = self.helpers["_static_installation_pairs_for_group"](
            policy, floor_group
        )
        self.assertEqual(
            excluded, {tuple(sorted((self.mount_geom, self.floor_geom)))}
        )
        clearance = self.helpers["_minimum_geom_clearance"](
            self.model,
            self.data,
            self.robot_geom_ids,
            floor_group["geom_ids"],
            excluded,
        )
        self.assertEqual(clearance["status"], "complete")
        self.assertLess(clearance["minimum_distance_m"], 0.0)
        self.assertEqual(clearance["pair"]["robot_geom_id"], self.moving_geom)
        self.assertEqual(clearance["pair"]["object_geom_id"], self.floor_geom)
        self.assertTrue(
            self.helpers["_is_allowed_static_installation_contact"](
                policy, (self.floor_geom, self.mount_geom)
            )
        )
        self.assertFalse(
            self.helpers["_is_allowed_static_installation_contact"](
                policy, (self.floor_geom, self.moving_geom)
            )
        )

    def test_jointed_body_cannot_be_classified_as_a_static_installation(self):
        moving_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_link"
        )
        with self.assertRaisesRegex(ValueError, "jointed_ancestor"):
            self.helpers["_fixed_body_ancestor_chain"](self.model, moving_body)


if __name__ == "__main__":
    unittest.main()
