"""MuJoCo-level regression tests for Bridge Snapshot state synchronization."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import mujoco
except ImportError:  # The Bridge's LIBERO-Safety environment provides this dependency.
    mujoco = None


BRIDGE_PATH = Path(__file__).with_name("safety_bridge.py")


def _load_snapshot_helpers():
    if mujoco is None:
        raise unittest.SkipTest("MuJoCo is only available in the LIBERO-Safety environment")
    tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
    wanted = {
        "_mujoco_integration_state",
        "_set_mujoco_integration_state",
        "_diagnostic_data_from_integration_state",
        "_snapshot_diagnostics",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"mujoco": mujoco, "np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BRIDGE_PATH), "exec"), namespace)
    return namespace


class MujocoSimFacade:
    def __init__(self, model, data):
        self.model = SimpleNamespace(_model=model)
        self.data = SimpleNamespace(_data=data)

    def forward(self):
        mujoco.mj_forward(self.model._model, self.data._data)


@unittest.skipIf(mujoco is None, "requires the LIBERO-Safety MuJoCo environment")
class SafetyBridgeSnapshotStateTests(unittest.TestCase):
    def setUp(self):
        helpers = _load_snapshot_helpers()
        self.get_state = helpers["_mujoco_integration_state"]
        self.set_state = helpers["_set_mujoco_integration_state"]
        self.diagnostic_data = helpers["_diagnostic_data_from_integration_state"]
        self.snapshot_diagnostics = helpers["_snapshot_diagnostics"]
        xml = """<mujoco>
          <option timestep="0.002" integrator="Euler" />
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1" />
            <body name="free_body" pos="0 0 0.095">
              <freejoint />
              <geom name="sphere" type="sphere" size="0.1" mass="1" />
              <site name="eef_site" pos="0.2 0 0" />
            </body>
          </worldbody>
        </mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for _ in range(5):
            mujoco.mj_step(model, data)
        self.sim = MujocoSimFacade(model, data)
        self.sim.data.qpos = data.qpos
        self.sim.data.qvel = data.qvel
        model_facade = SimpleNamespace(
            _model=model,
            body_name2id=lambda name: mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name
            ),
        )
        self.sim.model = model_facade
        self.robot = SimpleNamespace(
            _ref_joint_pos_indexes=np.asarray([0, 1, 2], dtype=np.int32),
            _ref_joint_vel_indexes=np.asarray([0, 1, 2], dtype=np.int32),
            eef_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_site"),
            has_gripper=False,
        )
        object_model = SimpleNamespace(root_body="free_body", joints=())
        self.task_env = SimpleNamespace(
            objects_dict={"free_object": object_model},
            fixtures_dict={},
        )
        self.helpers_env = SimpleNamespace(sim=self.sim, env=self.task_env, robots=[self.robot])
        helpers["_snapshot_diagnostics"].__globals__["env"] = self.helpers_env

    def test_roundtrip_restores_integration_and_refreshes_derived_kinematics(self):
        saved_state = self.get_state(self.sim)
        active_data = self.sim.data._data
        raw_cache_xpos = active_data.xpos.copy()
        canonical_data = self.diagnostic_data(self.sim)
        canonical_xpos = canonical_data.xpos.copy()

        # Euler stepping leaves forward-derived positions at the last forward
        # pass, while qpos already contains the integrated end-of-step state.
        self.assertGreater(float(np.max(np.abs(raw_cache_xpos - canonical_xpos))), 1e-8)
        np.testing.assert_array_equal(active_data.xpos, raw_cache_xpos)
        canonical_snapshot = self.snapshot_diagnostics(canonicalize_kinematics=True)
        np.testing.assert_allclose(
            canonical_snapshot["robot_states"][0]["eef_pos"],
            canonical_data.site_xpos[self.robot.eef_site_id],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(active_data.xpos, raw_cache_xpos)

        active_data.qpos[0] += 0.1
        self.sim.forward()
        self.set_state(self.sim, saved_state)

        np.testing.assert_array_equal(self.get_state(self.sim), saved_state)
        expected_data = self.diagnostic_data(self.sim)
        np.testing.assert_allclose(active_data.xpos, expected_data.xpos, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(active_data.site_xpos, expected_data.site_xpos, rtol=0.0, atol=1e-12)

    def test_identical_physics_step_sequence_from_snapshot_repeats_state_and_contacts(self):
        saved_state = self.get_state(self.sim)

        def rollout():
            states = []
            contacts = []
            for _ in range(3):
                mujoco.mj_step(self.sim.model._model, self.sim.data._data)
                states.append(self.get_state(self.sim).copy())
                step_contacts = []
                for index in range(self.sim.data._data.ncon):
                    contact = self.sim.data._data.contact[index]
                    step_contacts.append(
                        (
                            int(contact.geom1),
                            int(contact.geom2),
                            tuple(float(value) for value in contact.pos),
                            float(contact.dist),
                        )
                    )
                contacts.append(step_contacts)
            return states, contacts

        first_states, first_contacts = rollout()
        self.set_state(self.sim, saved_state)
        second_states, second_contacts = rollout()

        self.assertTrue(any(first_contacts), "MuJoCo fixture should record a floor contact")
        for first, second in zip(first_states, second_states):
            np.testing.assert_array_equal(first, second)
        self.assertEqual(first_contacts, second_contacts)


if __name__ == "__main__":
    unittest.main()
