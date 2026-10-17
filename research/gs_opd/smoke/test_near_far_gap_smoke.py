"""Lightweight distance API and Near-Far forecast tests; no LIBERO env starts."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import skipIf

import numpy as np

from research.gs_opd.smoke import near_far_gap_smoke as smoke

try:
    import mujoco
except ImportError:
    mujoco = None


BRIDGE_PATH = Path(__file__).with_name("safety_bridge.py")


def _load_bridge_geom_helpers(sim, mujoco_module=None):
    tree = ast.parse(BRIDGE_PATH.read_text(encoding="utf-8"))
    names = {
        "_mujoco_integration_state",
        "_diagnostic_data_from_integration_state",
        "_validated_geom_id_list",
        "_geom_set_min_distance",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    wrapper = SimpleNamespace(
        _model=sim.model._model,
        geom_id2name=lambda geom_id: mujoco.mj_id2name(
            sim.model._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        ),
        body_id2name=lambda body_id: mujoco.mj_id2name(
            sim.model._model, mujoco.mjtObj.mjOBJ_BODY, body_id
        ),
    )
    sim.model = wrapper
    namespace = {"np": np, "mujoco": mujoco_module or mujoco, "env": SimpleNamespace(sim=sim)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BRIDGE_PATH), "exec"), namespace)
    return namespace


def _packed_image(image):
    arr = np.ascontiguousarray(image)
    return {"shape": arr.shape, "dtype": str(arr.dtype), "data": arr.tobytes()}


class NearFarAnalysisTests(unittest.TestCase):
    @staticmethod
    def _samples(distances):
        return [
            {
                "state": f"S{step}",
                "global_step": step,
                "distance_available": value is not None,
                "minimum_distance_m": value,
            }
            for step, value in zip((65, 70, 71, 72, 73, 74, 75), distances)
        ]

    def test_fixed_thresholds_trigger_from_s70_a0_70_and_posthoc_lead_is_separate(self):
        samples = self._samples([0.04, 0.03, 0.019, 0.009, 0.004, 0.003, 0.002])
        forecast = smoke.analyze_gap_trajectory(samples, None)
        historical = {
            "status": "violation",
            "validated": True,
            "absolute_control_step": 75,
            "first_physics_substep": 8,
            "physics_substeps_per_control_step": 25,
        }
        with_ground_truth = smoke.analyze_gap_trajectory(samples, historical)
        self.assertEqual(with_ground_truth["fixed_thresholds_m"], [0.005, 0.01, 0.02])
        self.assertEqual(
            [item["s70_prediction_triggered_using_s70_plus_a0_70_only"] for item in forecast["threshold_alerts"]],
            [True, True, True],
        )
        self.assertEqual(
            [item["first_trigger_step_in_s70_forecast"] for item in forecast["threshold_alerts"]],
            [73, 72, 71],
        )
        self.assertEqual(
            [item["first_trigger_step_in_s70_forecast"] for item in with_ground_truth["threshold_alerts"]],
            [73, 72, 71],
        )
        self.assertEqual(
            with_ground_truth["threshold_alerts"][2]["lead_to_historical_s75_contact"][
                "nominal_physics_substeps_to_contact"
            ],
            108,
        )
        self.assertTrue(forecast["sustained_approach_over_s70_a0_70"])
        self.assertFalse(forecast["historical_collision_used_only_for_posthoc_lead"])

    def test_unavailable_distances_do_not_turn_into_no_alert_or_safe(self):
        samples = self._samples([0.04, 0.03, None, None, None, None, None])
        analysis = smoke.analyze_gap_trajectory(samples, None)
        self.assertIsNone(analysis["sustained_approach_over_s70_a0_70"])
        for alert in analysis["threshold_alerts"]:
            self.assertIsNone(alert["s70_prediction_triggered_using_s70_plus_a0_70_only"])

    def test_rgb_difference_is_diagnostic_while_robot_state_remains_strict(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        changed = image.copy()
        changed[0, 0, 0] = 1
        state = {
            "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.asarray([0.0, 0.0]),
        }
        expected = {
            "instruction": "move object",
            "state": state,
            "images": {key: image for key in CAMERA_KEYS_TEST},
        }
        actual = {
            "instruction": "move object",
            "state": {key: value.tolist() for key, value in state.items()},
            "images": {key: _packed_image(changed if key == "agentview_image" else image)
                       for key in CAMERA_KEYS_TEST},
        }
        check = smoke._strict_state_check(expected, actual)
        self.assertTrue(check["matched"])
        self.assertFalse(check["rgb_diagnostic_only"]["matched"])

    def test_historical_s75_violation_requires_current_body_and_geom_identity(self):
        rule = {
            "id": "rule-x",
            "subject_id": "object_carried",
            "target_id": "object_obstacle",
            "predicate": "contact",
            "decision": "forbid",
            "phase": "any",
        }
        subject = {"body_id": 12, "geom_ids": [100]}
        obstacle = {"body_id": 14, "geom_ids": [205]}
        inventory = {
            "geometries": [
                {"geom_id": 100, "geom_name": "payload_collider", "body_id": 12, "body_name": "payload"},
                {"geom_id": 205, "geom_name": "obstacle_collider", "body_id": 14, "body_name": "obstacle"},
            ]
        }
        controls = [
            {
                "control_step_index": i,
                "status": "complete",
                "expected_physics_substeps": 20,
                "captured_physics_substeps": 20,
                "unobserved_expected_substeps": 0,
                "contact_samples": {"contact": 0, "no_contact": 20, "unavailable": 0},
            }
            for i in range(5)
        ]
        evidence = {
            "subject_object_id": "object_carried",
            "target_object_id": "object_obstacle",
            "subject_body_id": 12,
            "target_body_id": 14,
            "first_control_step": 0,
            "first_physics_substep": 3,
            "geom_pair": [{"id": 100, "name": "payload_collider"}, {"id": 205, "name": "obstacle_collider"}],
            "body_pair": [{"id": 12, "name": "payload"}, {"id": 14, "name": "obstacle"}],
            "first_contact_position_world": [0.0, 0.0, 0.0],
            "first_contact_normal_geom1_to_geom2_world": [1.0, 0.0, 0.0],
            "max_force_norm": 0.5,
        }
        record = {
            "decision_step": 75,
            "a0_verdict": "violation",
            "a0_compiler_verdict": "violation",
            "a0_branch_summary": {
                "full_action_chunk_executed": True,
                "requested_steps": 5,
                "executed_steps": 5,
                "contact_monitoring": {
                    "status": "complete",
                    "unobserved_expected_substeps": 0,
                    "unavailable_contact_substeps": 0,
                    "expected_physics_substeps_total": 100,
                    "captured_physics_substeps": 100,
                    "control_steps": controls,
                },
            },
            "a0_compiled_result": {
                "compiled_object_contact_rules": [
                    {
                        "rule_id": "rule-x",
                        "subject_object_id": "object_carried",
                        "target_object_id": "object_obstacle",
                        "subject_body_id": 12,
                        "target_body_id": 14,
                        "subject_geom_ids": [100],
                        "target_geom_ids": [205],
                    }
                ],
                "candidates": [
                    {
                        "candidate_id": "archive-candidate",
                        "verdict": "violation",
                        "object_contact_checks": [
                            {
                                "rule_id": "rule-x",
                                "subject_id": "object_carried",
                                "target_id": "object_obstacle",
                                "verdict": "violation",
                                "reason": "forbidden contact",
                                "evidence": [evidence],
                            }
                        ],
                    }
                ],
            },
        }
        report = smoke.validate_historical_violation_record(record, 75, rule, subject, obstacle, inventory)
        self.assertTrue(report["validated"], report)
        self.assertEqual(report["status"], "violation")
        self.assertEqual(report["absolute_control_step"], 75)
        self.assertEqual(report["first_physics_substep"], 3)

        broken = json.loads(json.dumps(record))
        broken["a0_compiled_result"]["candidates"][0]["object_contact_checks"][0]["evidence"][0]["body_pair"][1]["id"] = 99
        invalid = smoke.validate_historical_violation_record(broken, 75, rule, subject, obstacle, inventory)
        self.assertFalse(invalid["validated"])
        self.assertEqual(invalid["status"], "unknown")


CAMERA_KEYS_TEST = ("agentview_image", "robot0_eye_in_hand_image")


@skipIf(mujoco is None, "requires the LIBERO-Safety MuJoCo environment")
class BridgeGeomDistanceTests(unittest.TestCase):
    def setUp(self):
        xml = """<mujoco><worldbody>
          <body name="left_body" pos="0 0 0"><geom name="left_geom" type="sphere" size="0.1"/></body>
          <body name="right_body" pos="0.25 0 0"><geom name="right_geom" type="sphere" size="0.1"/></body>
        </worldbody></mujoco>"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.geom_a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_geom")
        self.geom_b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_geom")

        class Sim:
            pass

        self.sim = Sim()
        self.sim.model = SimpleNamespace(_model=self.model)
        self.sim.data = SimpleNamespace(_data=self.data)
        self.namespace = _load_bridge_geom_helpers(self.sim)
        self.query = self.namespace["_geom_set_min_distance"]

    def test_signed_minimum_distance_uses_scratch_data_and_preserves_live_state(self):
        before = self.namespace["_mujoco_integration_state"](self.sim).copy()
        report = self.query([self.geom_a], [self.geom_b], 1.0)
        after = self.namespace["_mujoco_integration_state"](self.sim)
        self.assertTrue(report["query_complete"], report)
        self.assertAlmostEqual(report["minimum_distance_m"], 0.05, places=10)
        self.assertEqual(report["minimum_pair"]["geom_a"]["id"], self.geom_a)
        self.assertEqual(report["minimum_pair"]["geom_b"]["id"], self.geom_b)
        self.assertTrue(report["active_integration_state_unchanged"])
        np.testing.assert_array_equal(before, after)

    def test_distmax_censoring_is_not_reported_as_a_complete_gap(self):
        report = self.query([self.geom_a], [self.geom_b], 0.01)
        self.assertFalse(report["query_complete"])
        self.assertFalse(report["distance_reliable"])
        self.assertEqual(report["pair_count_censored_at_distmax"], 1)
        self.assertIsNone(report["minimum_distance_m"])

    def test_invalid_geom_sets_fail_closed(self):
        report = self.query([], [self.geom_b], 1.0)
        self.assertFalse(report["query_complete"])
        self.assertIsNone(report["minimum_distance_m"])
        self.assertTrue(report["errors"])

    def test_nonfinite_distance_is_incomplete_not_a_numeric_gap(self):
        real_mujoco = mujoco

        class Proxy:
            def __getattr__(self, name):
                if name == "mj_geomDistance":
                    return lambda *_args: float("nan")
                return getattr(real_mujoco, name)

        namespace = _load_bridge_geom_helpers(self.sim, Proxy())
        report = namespace["_geom_set_min_distance"]([self.geom_a], [self.geom_b], 1.0)
        self.assertFalse(report["query_complete"])
        self.assertFalse(report["distance_reliable"])
        self.assertIsNone(report["minimum_distance_m"])
        self.assertEqual(report["pair_error_count"], 1)

    def test_missing_geom_distance_api_is_explicitly_unavailable(self):
        real_mujoco = mujoco

        class Proxy:
            def __getattr__(self, name):
                if name == "mj_geomDistance":
                    raise AttributeError(name)
                return getattr(real_mujoco, name)

        namespace = _load_bridge_geom_helpers(self.sim, Proxy())
        report = namespace["_geom_set_min_distance"]([self.geom_a], [self.geom_b], 1.0)
        self.assertEqual(report["status"], "unavailable")
        self.assertFalse(report["query_complete"])
        self.assertTrue(any("does not expose" in item for item in report["errors"]))


if __name__ == "__main__":
    unittest.main()
