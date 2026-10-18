"""Lightweight regression tests for the Bridge's Pi0.5 OSC_POSE setup."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np


BRIDGE_PATH = Path(__file__).with_name("safety_bridge.py")


def _load_controller_setup():
    tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
    wanted = {
        "_PI05_LIBERO_OSC_POS_OUTPUT",
        "_PI05_LIBERO_OSC_ROT_OUTPUT",
        "_PI05_LIBERO_OSC_KP",
        "_configure_pi05_libero_osc_pose",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if names & wanted:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            nodes.append(node)
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BRIDGE_PATH), "exec"), namespace)
    return namespace["_configure_pi05_libero_osc_pose"]


def _load_action_bound_helpers():
    tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
    wanted = {"_active_environment_action_bounds", "_validate_tracking_action"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BRIDGE_PATH), "exec"), namespace)
    return namespace["_active_environment_action_bounds"], namespace["_validate_tracking_action"]


class FakeOSCController:
    name = "OSC_POSE"

    def __init__(self):
        self.control_dim = 6
        self.use_delta = True
        self.use_ori = True
        self.impedance_mode = "fixed"
        self.input_min = np.full(6, -1.0)
        self.input_max = np.full(6, 1.0)
        self.output_min = np.full(6, -2.0)
        self.output_max = np.full(6, 2.0)
        self.kp = np.full(6, 750.0)
        self.kd = np.full(6, 2.0 * np.sqrt(750.0))
        self.kp_min = np.zeros(6)
        self.kp_max = np.full(6, 1000.0)
        self.damping_ratio_min = np.zeros(6)
        self.damping_ratio_max = np.full(6, 10.0)
        self.position_limits = None
        self.orientation_limits = None
        self.uncoupling = True
        self.action_scale = None
        self.action_input_transform = None
        self.action_output_transform = None

    def scale_action(self, action):
        if self.action_scale is None:
            self.action_scale = np.abs(self.output_max - self.output_min) / np.abs(
                self.input_max - self.input_min
            )
            self.action_input_transform = (self.input_max + self.input_min) / 2.0
            self.action_output_transform = (self.output_max + self.output_min) / 2.0
        clipped = np.clip(action, self.input_min, self.input_max)
        return (
            (clipped - self.action_input_transform) * self.action_scale
            + self.action_output_transform
        )


class SafetyBridgeControllerSetupTests(unittest.TestCase):
    def setUp(self):
        self.configure = _load_controller_setup()

    def test_runtime_parameters_and_affine_scale_match_libero_pi05(self):
        controller = FakeOSCController()

        report = self.configure(controller)

        np.testing.assert_allclose(
            controller.output_min, [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5]
        )
        np.testing.assert_allclose(
            controller.output_max, [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
        )
        np.testing.assert_allclose(controller.kp, np.full(6, 150.0))
        np.testing.assert_allclose(controller.kd, np.full(6, 2.0 * np.sqrt(150.0)))
        np.testing.assert_allclose(controller.kp_min, np.zeros(6))
        np.testing.assert_allclose(controller.kp_max, np.full(6, 300.0))
        np.testing.assert_allclose(
            controller.scale_action(np.full(6, 0.5)),
            [0.025, 0.025, 0.025, 0.25, 0.25, 0.25],
        )
        self.assertEqual(report["control_frame"], "world")
        self.assertEqual(report["control_delta"], True)
        self.assertEqual(report["damping_ratio"], 1.0)
        self.assertTrue(report["uncouple_pos_ori"])
        self.assertEqual(report["source"], "upstream robosuite v1.4.0 OSC_POSE defaults")

    def test_rejects_unexpected_controller_or_action_convention(self):
        controller = FakeOSCController()
        controller.name = "OSC_POSITION"
        with self.assertRaisesRegex(RuntimeError, "requires OSC_POSE"):
            self.configure(controller)

        controller = FakeOSCController()
        controller.use_delta = False
        with self.assertRaisesRegex(RuntimeError, "must use delta commands"):
            self.configure(controller)

    def test_bridge_reapplies_configuration_after_each_hard_reset(self):
        tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
        reset_handler = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Subscript)
            and isinstance(node.test.left.value, ast.Name)
            and node.test.left.value.id == "request"
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "reset"
        )
        reset_call = next(
            node for node in ast.walk(reset_handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "env"
            and node.func.attr == "reset"
        )
        configure_call = next(
            node for node in ast.walk(reset_handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_configure_pi05_libero_osc_pose"
        )
        self.assertLess(reset_call.lineno, configure_call.lineno)


class FakePandaActionSource:
    def __init__(self, low=None, high=None):
        self.action_limits = (
            np.full(7, -1.0) if low is None else np.asarray(low, dtype=np.float64),
            np.full(7, 1.0) if high is None else np.asarray(high, dtype=np.float64),
        )


class FakeRobosuiteActionEnv:
    def __init__(self, robot, action_spec=None):
        self.robot = robot
        self._action_spec = action_spec

    @property
    def action_spec(self):
        if self._action_spec is not None:
            return self._action_spec
        return self.robot.action_limits


class FakeOffScreenRenderEnv:
    """Mirror LIBERO: only the nested robosuite task exposes action_spec."""

    def __init__(self, task_env):
        self.env = task_env


class SafetyBridgeActionBoundsTests(unittest.TestCase):
    def setUp(self):
        self.get_bounds, self.validate_action = _load_action_bound_helpers()

    def test_reads_action_spec_from_nested_robosuite_env_and_matches_panda_limits(self):
        robot = FakePandaActionSource()
        task_env = FakeRobosuiteActionEnv(robot)
        wrapper = FakeOffScreenRenderEnv(task_env)
        self.assertFalse(hasattr(wrapper, "action_spec"))

        low, high = self.get_bounds(wrapper, robot)

        np.testing.assert_array_equal(low, robot.action_limits[0])
        np.testing.assert_array_equal(high, robot.action_limits[1])

    def test_rejects_missing_action_spec(self):
        class MissingActionSpec:
            pass

        with self.assertRaisesRegex(ValueError, "action bounds unavailable"):
            self.get_bounds(FakeOffScreenRenderEnv(MissingActionSpec()), FakePandaActionSource())

    def test_rejects_non_finite_action_bounds(self):
        robot = FakePandaActionSource()
        low = np.full(7, -1.0)
        high = np.full(7, 1.0)
        high[2] = np.nan
        wrapper = FakeOffScreenRenderEnv(FakeRobosuiteActionEnv(robot, (low, high)))
        with self.assertRaisesRegex(ValueError, "finite 7-D"):
            self.get_bounds(wrapper, robot)

    def test_rejects_illegal_action_bound_shape(self):
        robot = FakePandaActionSource()
        wrapper = FakeOffScreenRenderEnv(
            FakeRobosuiteActionEnv(robot, (np.full(6, -1.0), np.full(6, 1.0)))
        )
        with self.assertRaisesRegex(ValueError, "finite 7-D"):
            self.get_bounds(wrapper, robot)

    def test_rejects_environment_bounds_that_disagree_with_panda_limits(self):
        robot = FakePandaActionSource()
        low = np.full(7, -1.0)
        high = np.full(7, 1.0)
        high[-1] = 0.5
        wrapper = FakeOffScreenRenderEnv(FakeRobosuiteActionEnv(robot, (low, high)))
        with self.assertRaisesRegex(ValueError, "disagrees with Panda"):
            self.get_bounds(wrapper, robot)

    def test_action_validation_rejects_bad_shape_non_finite_and_out_of_range(self):
        low, high = np.full(7, -1.0), np.full(7, 1.0)
        with self.assertRaisesRegex(ValueError, "finite_7d"):
            self.validate_action(np.zeros(6), low, high)
        bad = np.zeros(7)
        bad[0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite_7d"):
            self.validate_action(bad, low, high)
        bad = np.zeros(7)
        bad[3] = 1.01
        with self.assertRaisesRegex(ValueError, "outside_active_environment_bounds"):
            self.validate_action(bad, low, high)


if __name__ == "__main__":
    unittest.main()
