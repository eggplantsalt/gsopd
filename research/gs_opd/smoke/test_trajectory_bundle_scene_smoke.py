# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static synthetic MuJoCo geometry and protocol tests; no environment steps."""

import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.gs_opd.smoke import trajectory_bundle_scene_smoke as smoke
from research.gs_opd.smoke.test_pi05_on_policy_safety_collection import (
    _semantic_binding,
    _teacher_v2_object_only,
)

try:
    import mujoco
except ImportError:
    mujoco = None


BRIDGE = Path(__file__).with_name("safety_bridge.py")


def _helpers(sim, task_env, robot_body):
    tree = ast.parse(BRIDGE.read_text())
    names = {
        "_mujoco_integration_state",
        "_diagnostic_data_from_integration_state",
        "_build_geometry_inventory",
        "_geometry_inventory",
        "_validated_geom_id_list",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and (node.name in names or node.name.startswith("_planning_"))
    ]
    namespace = {
        "np": np,
        "mujoco": mujoco,
        "copy": copy,
        "env": SimpleNamespace(sim=sim, env=task_env),
        "task": SimpleNamespace(language="carry the payload to the receiver"),
        "_robot_body_ids": lambda: {robot_body},
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(BRIDGE), "exec"), namespace
    )
    return namespace


class ReadOnlyBridge:
    """A process-edge adapter running the real read-only Bridge helper on a toy model."""

    def __init__(self, helpers):
        self.helpers = helpers
        self.requests = []
        self.pending = None
        self.change_after_first = False
        self.query_count = 0

    def send(self, request):
        self.requests.append(copy.deepcopy(request))
        self.pending = request

    def recv(self):
        if self.pending["op"] == "geometry_inventory":
            return self.helpers["_geometry_inventory"]()
        if self.pending["op"] != "planning_geometry":
            raise AssertionError(
                "Smoke must not reset, step, branch, restore, or execute actions"
            )
        self.query_count += 1
        if self.change_after_first and self.query_count == 2:
            self.helpers["env"].sim.data._data.time += 0.01
        result = self.helpers["_planning_geometry"](
            self.pending["carried_binding"],
            self.pending["obstacle_bindings"],
            self.pending["expected_instruction"],
        )
        return result


@unittest.skipIf(
    mujoco is None, "requires existing MuJoCo package for static compilation only"
)
class PlanningGeometryTests(unittest.TestCase):
    def setUp(self):
        xml = """<mujoco>
          <asset><mesh name="tiny_mesh" vertex="-.02 -.01 -.03  .02 -.01 -.03  .02 .01 -.03  -.02 .01 -.03
             -.02 -.01 .03  .02 -.01 .03  .02 .01 .03  -.02 .01 .03"/></asset>
          <worldbody>
            <body name="robot"><geom name="robot_geom" type="sphere" size=".01" contype="0" conaffinity="0"/></body>
            <body name="payload" pos="-.4 0 .5"><freejoint/>
              <geom name="payload_collision" type="box" size=".03 .02 .015"/>
              <geom name="payload_visual" type="sphere" size=".5" pos="1 0 0" contype="0" conaffinity="0" mass=".001"/>
              <body name="payload_child" pos=".08 0 0">
                <geom name="child_collision" type="sphere" size=".02"/>
                <geom name="pair_only" type="sphere" size=".01" pos=".05 0 0" contype="0" conaffinity="0"/>
              </body>
            </body>
            <body name="blocker" pos="0 0 .5">
              <geom name="blocker_collision" type="capsule" size=".03 .04"/>
              <geom name="blocker_mesh" type="mesh" mesh="tiny_mesh" pos="0 .06 0"/>
              <geom name="blocker_visual" type="sphere" size=".06" contype="0" conaffinity="0"/>
            </body>
            <body name="receiver" pos=".5 0 .45">
              <geom name="receiver_collision" type="box" size=".1 .1 .01"/>
              <site name="receiver_region" type="box" size=".08 .08 .02"/>
            </body>
          </worldbody>
          <contact><pair geom1="pair_only" geom2="blocker_collision" margin=".004"/></contact>
        </mujoco>"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        wrapper = SimpleNamespace(
            _model=self.model,
            geom_bodyid=self.model.geom_bodyid,
            body_name2id=lambda name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            ),
            geom_name2id=lambda name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name
            ),
            body_id2name=lambda key: mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, key
            ),
            geom_id2name=lambda key: mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, key
            ),
            site_id2name=lambda key: mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_SITE, key
            ),
        )
        self.sim = SimpleNamespace(model=wrapper, data=SimpleNamespace(_data=self.data))
        self.task_env = SimpleNamespace(
            objects_dict={
                key: SimpleNamespace(root_body=key, contact_geoms=geoms)
                for key, geoms in (
                    ("payload", ["payload_collision", "child_collision", "pair_only"]),
                    ("blocker", ["blocker_collision", "blocker_mesh"]),
                    ("receiver", ["receiver_collision"]),
                )
            },
            fixtures_dict={},
            parsed_problem={"goal_state": [["on", "payload", "receiver"]]},
            bddl_file_name="synthetic.bddl",
        )
        self.h = _helpers(self.sim, self.task_env, wrapper.body_name2id("robot"))
        self.teacher = _teacher_v2_object_only()
        self.semantic = {
            "object_1": _semantic_binding(
                "object_1",
                wrapper.body_name2id("payload"),
                "payload",
                wrapper.geom_name2id("payload_collision"),
            ),
            "object_2": _semantic_binding(
                "object_2",
                wrapper.body_name2id("blocker"),
                "blocker",
                wrapper.geom_name2id("blocker_visual"),
            ),
        }
        self.semantic["object_1"]["task_instruction"] = (
            "carry the payload to the receiver"
        )
        self.inventory = self.h["_geometry_inventory"]()
        # The fixture helper uses generated names; replace them with actual names.
        for semantic in self.semantic.values():
            association = semantic["physical_association"]
            association["geom_names"] = [
                wrapper.geom_id2name(g) for g in association["geom_ids"]
            ]
        self.payload, _ = smoke._selected_bindings(
            self.teacher, self.semantic, self.inventory, None, None
        )

    def query(self):
        return self.h["_planning_geometry"](
            self.payload["carried_binding"],
            self.payload["obstacle_bindings"],
            self.payload["expected_instruction"],
        )

    def test_full_collision_subtree_includes_hidden_geoms_and_explicit_pairs(self):
        result = self.query()
        self.assertEqual(result["status"], "complete", result.get("error"))
        carried = result["objects"]["object_1"]
        names = {item["geom_name"] for item in carried["components"]}
        self.assertEqual(names, {"payload_collision", "child_collision", "pair_only"})
        pair = next(
            item for item in carried["components"] if item["geom_name"] == "pair_only"
        )
        self.assertGreaterEqual(pair["contact_margin_m"], 0.004)
        self.assertIn(
            self.sim.model.geom_name2id("payload_visual"),
            carried["excluded_visual_only_geom_ids"],
        )
        self.assertEqual(
            result["objects"]["object_2"]["semantic_binding_geom_ids"],
            [self.sim.model.geom_name2id("blocker_visual")],
        )
        self.assertNotIn(
            self.sim.model.geom_name2id("blocker_visual"),
            result["objects"]["object_2"]["geom_ids"],
        )
        for item in result["objects"].values():
            for component in item["components"]:
                self.assertLessEqual(
                    np.linalg.norm(
                        np.asarray(component["center_world_m"]) - item["center_world_m"]
                    )
                    + component["radius_m"],
                    item["radius_m"],
                )
        self.assertTrue(result["physical_state_unchanged"])

    def test_mesh_bound_covers_every_compiled_vertex(self):
        result = self.query()
        component = next(
            item
            for item in result["objects"]["object_2"]["components"]
            if item["geom_name"] == "blocker_mesh"
        )
        geom_id = component["geom_id"]
        mesh_id = self.model.geom_dataid[geom_id]
        address, count = (
            self.model.mesh_vertadr[mesh_id],
            self.model.mesh_vertnum[mesh_id],
        )
        vertices = np.asarray(
            self.model.mesh_vert[address : address + count], dtype=np.float64
        )
        self.assertLessEqual(
            float(np.linalg.norm(vertices, axis=1).max()), component["radius_m"]
        )
        self.assertIn("all compiled mesh vertices", component["source"])

    def test_canonical_scratch_positions_do_not_touch_stale_live_cache(self):
        self.data.qpos[0] += 0.07
        before = self.h["_mujoco_integration_state"](self.sim).copy()
        live = self.data.geom_xpos.copy()
        result = self.query()
        self.assertEqual(result["status"], "complete", result.get("error"))
        self.assertAlmostEqual(
            result["objects"]["object_1"]["center_world_m"][0], -0.33
        )
        np.testing.assert_array_equal(live, self.data.geom_xpos)
        np.testing.assert_array_equal(
            before, self.h["_mujoco_integration_state"](self.sim)
        )
        self.assertEqual(result["state_id"], self.query()["state_id"])
        self.data.qpos[0] += 0.01
        self.assertNotEqual(result["state_id"], self.query()["state_id"])

    def test_goal_body_and_site_are_diagnostics_not_a_verified_center_goal(self):
        result = self.query()
        self.assertEqual(result["goal"]["status"], "goal_unavailable")
        self.assertIsNone(result["goal"]["position_world_m"])
        target = result["goal"]["targets"][0]
        self.assertEqual(target["body_name"], "receiver")
        self.assertFalse(target["body_origin_is_carried_center_goal"])
        self.assertTrue(target["sites_diagnostic_only"])
        self.assertFalse(
            target["sites_diagnostic_only"][0]["safe_center_semantics_verified"]
        )

    def test_incomplete_binding_or_wrong_task_fails_closed(self):
        for key in ("geom_names", "body_name"):
            bad = copy.deepcopy(self.payload)
            bad["carried_binding"][key] = [] if key == "geom_names" else "wrong-body"
            result = self.h["_planning_geometry"](
                bad["carried_binding"],
                bad["obstacle_bindings"],
                bad["expected_instruction"],
            )
            self.assertEqual(result["status"], "unavailable")
            self.assertIn("error", result)
        result = self.h["_planning_geometry"](
            self.payload["carried_binding"],
            self.payload["obstacle_bindings"],
            "another task",
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertTrue(result["physical_state_unchanged"])

    def test_nonfinite_and_unsupported_geom_bounds_are_rejected(self):
        geom_id = self.sim.model.geom_name2id("payload_collision")
        original = self.model.geom_rbound[geom_id]
        self.model.geom_rbound[geom_id] = float("nan")
        with self.assertRaises(ValueError):
            self.h["_planning_component_sphere"](self.model, self.data, geom_id, {})
        self.model.geom_rbound[geom_id] = original
        self.model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_SDF
        with self.assertRaises(ValueError):
            self.h["_planning_component_sphere"](self.model, self.data, geom_id, {})

    def test_query_detects_an_accidental_live_state_mutation(self):
        original = self.h["_diagnostic_data_from_integration_state"]

        def corrupt(sim):
            data = original(sim)
            self.data.time += 0.001
            return data

        self.h["_diagnostic_data_from_integration_state"] = corrupt
        result = self.query()
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["physical_state_unchanged"])

    def test_scene_smoke_generates_local_bypass_bundle_from_current_query(self):
        bridge = ReadOnlyBridge(self.h)
        with tempfile.TemporaryDirectory() as directory:
            result = smoke.run_scene_smoke(
                bridge, self.teacher, self.semantic, directory
            )
            self.assertEqual(result["status"], "complete", result)
            self.assertEqual(result["goal_status"], "local_bypass_goal_constructed")
            self.assertFalse(result["final_placement_goal_verified"])
            self.assertTrue(result["geometry_query_succeeded"])
            self.assertTrue(result["state_consistency"]["matched"])
            self.assertEqual(result["candidate_count"], 18)
            self.assertTrue((Path(directory) / "planning_geometry.json").is_file())
            self.assertTrue((Path(directory) / "scene.json").is_file())
            self.assertTrue((Path(directory) / "trajectory_bundle.json").is_file())
            self.assertTrue((Path(directory) / "trajectory_bundle.svg").is_file())
            self.assertEqual(result["scene"]["goal"]["goal_type"], "local_bypass_goal")
            saved = json.loads(
                (Path(directory) / "scene_smoke_report.json").read_text()
            )
            self.assertFalse(saved["trajectory_executability_verified"])
        self.assertEqual(
            [item["op"] for item in bridge.requests],
            ["geometry_inventory", "planning_geometry", "planning_geometry"],
        )

    def test_local_bypass_uses_bddl_body_only_for_forward_direction(self):
        query = self.query()
        scene = smoke._build_local_bypass_scene(query, self.payload)
        start = np.asarray(scene["carried"]["center_world_m"])
        target = np.asarray(
            scene["goal"]["construction"]["placement_target_direction_reference"][
                "body_origin_world_m"
            ]
        )
        forward = (target - start) / np.linalg.norm(target - start)
        construction = scene["goal"]["construction"]
        obstacle = np.asarray(construction["obstacle_center_world_m"])
        endpoint = np.asarray(scene["goal"]["position_world_m"])
        self.assertEqual(
            construction["placement_target_direction_reference"]["native_predicate"],
            ["on", "payload", "receiver"],
        )
        self.assertFalse(
            construction["placement_target_direction_reference"]["used_as_final_goal"]
        )
        np.testing.assert_allclose(construction["forward_world"], forward, atol=1e-12)
        self.assertGreater(float(np.dot(endpoint - obstacle, forward)), 0)
        self.assertGreaterEqual(
            np.linalg.norm(endpoint - obstacle)
            - scene["carried"]["radius_m"]
            - query["objects"][construction["obstacle_object_id"]]["radius_m"],
            0.01,
        )
        self.assertNotEqual(endpoint.tolist(), target.tolist())

    def test_local_bypass_reports_degenerate_missing_and_misdirected_geometry(self):
        query = self.query()
        cases = []
        degenerate = copy.deepcopy(query)
        degenerate["goal"]["targets"][0]["body_origin_world_m_diagnostic_only"] = (
            degenerate["objects"]["object_1"]["center_world_m"]
        )
        cases.append(("placement_direction_degenerate", degenerate))
        missing = copy.deepcopy(query)
        missing["goal"]["targets"] = []
        cases.append(("compiled_bddl_target_body_missing_or_ambiguous", missing))
        wrong_direction = copy.deepcopy(query)
        wrong_direction["objects"]["object_2"]["center_world_m"] = [-0.8, 0, 0.5]
        cases.append(("no_forbidden_obstacle_in_forward_corridor", wrong_direction))
        outside_corridor = copy.deepcopy(query)
        outside_corridor["objects"]["object_2"]["center_world_m"] = [0, 0.5, 0.5]
        cases.append(("no_forbidden_obstacle_in_forward_corridor", outside_corridor))
        for expected, value in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    smoke._build_local_bypass_scene(value, self.payload)
        with self.assertRaises(smoke._LocalGoalUnavailable) as context:
            smoke._build_local_bypass_scene(wrong_direction, self.payload)
        self.assertIn(
            "obstacle_not_between_carried_state_and_bddl_target",
            context.exception.diagnostics[0]["rejection_reasons"],
        )

    def test_scene_change_prevents_saving_any_candidate_bundle(self):
        bridge = ReadOnlyBridge(self.h)
        bridge.change_after_first = True
        with tempfile.TemporaryDirectory() as directory:
            result = smoke.run_scene_smoke(
                bridge, self.teacher, self.semantic, directory
            )
            self.assertEqual(result["status"], "state_consistency_failed", result)
            self.assertFalse(result["geometry_query_succeeded"])
            self.assertFalse((Path(directory) / "trajectory_bundle.json").exists())

    def test_semantic_or_instruction_conflict_stops_before_geometry_request(self):
        for change in ("instruction", "binding"):
            bridge = ReadOnlyBridge(self.h)
            semantic = copy.deepcopy(self.semantic)
            if change == "instruction":
                semantic["object_2"]["task_instruction"] = "another task"
            else:
                semantic["object_2"]["physical_association"]["body_name"] = "wrong"
            with tempfile.TemporaryDirectory() as directory:
                result = smoke.run_scene_smoke(
                    bridge, self.teacher, semantic, directory
                )
                self.assertEqual(result["status"], "unavailable")
            self.assertEqual(
                [item["op"] for item in bridge.requests], ["geometry_inventory"]
            )

    def test_missing_instruction_requires_an_explicit_task_check(self):
        semantic = copy.deepcopy(self.semantic)
        del semantic["object_1"]["task_instruction"]
        with tempfile.TemporaryDirectory() as directory:
            bridge = ReadOnlyBridge(self.h)
            result = smoke.run_scene_smoke(bridge, self.teacher, semantic, directory)
            self.assertEqual(result["status"], "unavailable")
            self.assertIn("expected_instruction_unavailable", result["error"])
        with tempfile.TemporaryDirectory() as directory:
            result = smoke.run_scene_smoke(
                ReadOnlyBridge(self.h),
                self.teacher,
                semantic,
                directory,
                expected_instruction=self.payload["expected_instruction"],
            )
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["geometry_query_succeeded"])

    def test_preexisting_outputs_are_preserved_and_not_reused(self):
        bridge = ReadOnlyBridge(self.h)
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "trajectory_bundle.json"
            existing.write_text("old output")
            with self.assertRaises(FileExistsError):
                smoke.run_scene_smoke(bridge, self.teacher, self.semantic, directory)
            self.assertEqual(existing.read_text(), "old output")
        self.assertFalse(bridge.requests)

    def test_visual_only_object_is_not_a_complete_collision_enclosure(self):
        geom_id = self.sim.model.geom_name2id("receiver_collision")
        self.model.geom_contype[geom_id] = 0
        self.model.geom_conaffinity[geom_id] = 0
        binding = {
            "object_id": "receiver-query",
            "body_id": self.sim.model.body_name2id("receiver"),
            "body_name": "receiver",
            "geom_ids": [geom_id],
            "geom_names": ["receiver_collision"],
        }
        result = self.h["_planning_geometry"](
            binding,
            self.payload["obstacle_bindings"],
            self.payload["expected_instruction"],
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("no confirmed collision Geoms", result["error"])

    def test_custom_callback_is_rejected_before_scratch_forward(self):
        previous = mujoco.get_mjcb_passive()
        try:
            mujoco.set_mjcb_passive(lambda model, data: None)
            result = self.query()
        finally:
            mujoco.set_mjcb_passive(previous)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("custom callback", result["error"])
        self.assertTrue(result["physical_state_unchanged"])


if __name__ == "__main__":
    unittest.main()
