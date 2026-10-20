# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synthetic regressions for the live-grasp-gated one-shot acceptance flow."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.gs_opd.repair import trajectory_bundle_acceptance as acceptance

REPO_ROOT = Path(__file__).resolve().parents[3]
ROLLOUT_PATH = REPO_ROOT / "research/gs_opd/smoke/pi05_rollout.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeConnection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def send(self, payload):
        self.requests.append(payload)

    def recv(self):
        return self.responses.pop(0)


class VerifiedCandidateSelectionTests(unittest.TestCase):
    @staticmethod
    def _candidate(candidate_id, direction, points):
        return {
            "candidate_id": candidate_id,
            "direction": direction,
            "path_length_m": 0.4,
            "executable_safe": True,
            "physical_safety_verified": True,
            "physical_tracking_performed": True,
            "physical_tracking_status": "complete",
            "main_snapshot_restore_verified": True,
            "contact_compiler": {"verdict": "safe"},
            "geometric_candidate": {
                "goal_world_m": points[-1],
                "points_world_m": points,
            },
        }

    @classmethod
    def _tracking(cls):
        return {
            "status": "complete",
            "main_state_restored": True,
            "snapshot_restore_status": "verified",
            "executable_safe_candidate_ids": ["route-2", "route-1"],
            "candidates": [
                cls._candidate(
                    "route-1", "left", [[0, 0, 0], [0.1, 0.1, 0], [0.2, 0.2, 0]]
                ),
                cls._candidate(
                    "route-2", "right", [[0, 0, 0], [0.1, -0.1, 0], [0.2, -0.2, 0]]
                ),
            ],
        }

    def test_seeded_choice_is_repeatable_and_varies_across_verified_routes(self):
        tracking = self._tracking()
        self.assertEqual(
            acceptance.select_verified_candidate(tracking, 71),
            acceptance.select_verified_candidate(tracking, 71),
        )
        result = acceptance.build_candidate_selection_smoke(tracking, 71)
        self.assertTrue(result["same_seed_reproducible"])
        self.assertEqual(result["selection_pool_candidate_ids"], ["route-1", "route-2"])
        self.assertEqual(len(result["ten_seed_trials"]), 10)
        self.assertGreaterEqual(result["ten_seed_unique_candidate_count"], 2)
        self.assertTrue(result["different_verified_route_selected"])
        self.assertGreater(
            result["ten_seed_trials"][1]["route_difference_from_first_selection"][
                "normalized_path_max_separation_m"
            ],
            0.1,
        )

    def test_selection_fails_closed_without_complete_restore_or_safe_compiler(self):
        tracking = self._tracking()
        tracking["main_state_restored"] = False
        self.assertIsNone(acceptance.select_verified_candidate(tracking, 0))
        tracking = self._tracking()
        tracking["candidates"][0]["contact_compiler"]["verdict"] = "unknown"
        selected = acceptance.select_verified_candidate(tracking, 0)
        self.assertIn(selected["candidate_id"], {"route-2"})

    def test_one_verified_candidate_remains_selectable(self):
        tracking = self._tracking()
        tracking["executable_safe_candidate_ids"] = ["route-1"]
        tracking["candidates"] = tracking["candidates"][:1]
        selected = acceptance.select_verified_candidate(tracking, 0)
        self.assertEqual(selected["candidate_id"], "route-1")
        self.assertEqual(selected["selection_pool_candidate_ids"], ["route-1"])

    def test_selection_seed_must_be_integer(self):
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            acceptance.select_verified_candidate(self._tracking(), True)

    def test_empty_verified_pool_records_all_ten_seeded_rejections(self):
        result = acceptance.build_candidate_selection_smoke(
            {
                "status": "complete",
                "main_state_restored": True,
                "snapshot_restore_status": "verified",
                "executable_safe_candidate_ids": [],
                "candidates": [],
            },
            71,
        )
        self.assertIsNone(result["primary_selection"])
        self.assertIsNone(result["same_seed_reproducible"])
        self.assertEqual(len(result["ten_seed_trials"]), 10)
        self.assertEqual(
            [item["seed"] for item in result["ten_seed_trials"]], list(range(10))
        )
        self.assertTrue(
            all(item["candidate_id"] is None for item in result["ten_seed_trials"])
        )


def _finger_contact(finger_geom, finger_body, object_geom, object_body):
    return {
        "geom1": {"id": finger_geom, "name": f"finger-{finger_geom}"},
        "geom2": {"id": object_geom, "name": f"object-{object_geom}"},
        "body1": {"id": finger_body, "name": f"finger-body-{finger_body}"},
        "body2": {"id": object_body, "name": f"object-body-{object_body}"},
        "position": [0.0, 0.0, 0.0],
    }


def _snapshot_state():
    return {
        "mujoco_integration_state": [0.0, 1.0],
        "robot_states": [{"eef_pos": [0.1, 0.2, 0.3], "joint_qpos": [0.0]}],
        "object_states": {"object": {"body_id": 10, "root_pos": [0.4, 0.5, 0.6]}},
        "kinematics_source": "forwarded_copy_of_mujoco_integration_state",
    }


def _success_responses(*, restore_error=False, geometry_state=None):
    state = _snapshot_state()
    query = {
        "operation": "planning_geometry",
        "status": "complete",
        "query_complete": True,
        "physical_state_unchanged": True,
        "scratch_forward_used": True,
        "frame": "world",
        "units": "m",
        "instruction": "move the object",
        "state_id": "state-1",
        "model_geometry_sha256": "model-1",
        "mujoco_integration_state": [0.0, 1.0]
        if geometry_state is None
        else geometry_state,
        "geometry_inventory": {"status": "complete", "geometries": []},
        "objects": {"carry": {"body_id": 10}},
        "goal": {},
    }
    restore = (
        {"error": "Restore failed: injected"}
        if restore_error
        else {"restored_snapshot_id": "snapshot-1", "snapshot_state": state}
    )
    common = [
        {"snapshot_id": "snapshot-1", "snapshot_state": state},
        {"status": "complete", "geometries": []},
        {"status": "pass", "grasp_confirmed": True, "state_id": "state-1"},
        query,
    ]
    if restore_error:
        return [*common, {"status": "available"}, {"status": "complete"}, restore]
    return [
        *common,
        {"status": "available", "robots": [{"controller_name": "OSC_POSE"}]},
        {"status": "complete", "grippers": []},
        restore,
        query,
    ]


class OpposingFingerContactTests(unittest.TestCase):
    def test_requires_live_contacts_from_both_classified_fingers(self):
        contacts = {
            "status": "contact",
            "contacts": [
                _finger_contact(1, 11, 20, 10),
                _finger_contact(2, 12, 21, 10),
            ],
        }
        result = acceptance.classify_opposing_finger_contacts(
            contacts,
            object_geom_ids={20, 21},
            object_body_ids={10},
            gripper_geom_groups={"left": {1}, "right": {2}},
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["left"])
        self.assertTrue(result["right"])
        self.assertEqual(result["contact_count"], 2)

    def test_one_finger_or_wrong_object_body_does_not_confirm_grasp(self):
        contacts = {
            "status": "contact",
            "contacts": [_finger_contact(1, 11, 20, 99)],
        }
        result = acceptance.classify_opposing_finger_contacts(
            contacts,
            object_geom_ids={20},
            object_body_ids={10},
            gripper_geom_groups={"left": {1}, "right": {2}},
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["left"])
        self.assertFalse(result["right"])

    def test_unavailable_contact_data_is_not_no_contact_or_grasp(self):
        result = acceptance.classify_opposing_finger_contacts(
            {"status": "unavailable", "contacts": []},
            object_geom_ids={20},
            object_body_ids={10},
            gripper_geom_groups={"left": {1}, "right": {2}},
        )
        self.assertEqual(result["status"], "unavailable")

    def test_bridge_grasp_operation_uses_active_binding_and_close_state(self):
        bridge_path = REPO_ROOT / "research/gs_opd/smoke/safety_bridge.py"
        tree = ast.parse(bridge_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_live_grasp_state"
        )
        sides_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_tracking_gripper_sides"
        )
        closure_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_parse_tracking_gripper_closure"
        )

        class Model:
            nbody = 20
            ngeom = 4
            npair = 0
            geom_bodyid = [10, 11, 12, 10]
            geom_contype = [1, 1, 1, 1]
            geom_conaffinity = [1, 1, 1, 1]

        class ModelWrapper:
            _model = Model()

            @staticmethod
            def body_id2name(body_id):
                return f"body-{body_id}"

            @staticmethod
            def geom_name2id(name):
                return {"left": 1, "left-pad": 1, "right": 2, "right-pad": 2}[name]

        class Sim:
            model = ModelWrapper()
            data = type("Data", (), {"_data": object()})()

        class PandaGripper:
            current_action = [-0.95, 0.95]
            _important_geoms = {
                "left_finger": ["left"],
                "left_fingerpad": ["left-pad"],
                "right_finger": ["right"],
                "right_fingerpad": ["right-pad"],
            }

            @staticmethod
            def correct_naming(names):
                return names

        robot = type("Robot", (), {"gripper": PandaGripper()})()
        fake_env = type(
            "Env",
            (),
            {
                "sim": Sim(),
                "env": object(),
                "robots": [robot],
            },
        )()
        contacts = {
            "status": "contact",
            "contacts": [
                _finger_contact(1, 1, 0, 10),
                _finger_contact(2, 2, 3, 10),
            ],
        }
        inventory = {
            "status": "complete",
            "geometries": [
                {
                    "geom_id": 0,
                    "geom_name": "object-a",
                    "body_id": 10,
                    "body_name": "body-10",
                    "category": "scene",
                },
                {
                    "geom_id": 1,
                    "geom_name": "left-finger",
                    "body_id": 1,
                    "body_name": "body-1",
                    "category": "robot",
                },
                {
                    "geom_id": 2,
                    "geom_name": "right-finger",
                    "body_id": 2,
                    "body_name": "body-2",
                    "category": "robot",
                },
                {
                    "geom_id": 3,
                    "geom_name": "object-b",
                    "body_id": 10,
                    "body_name": "body-10",
                    "category": "scene",
                },
            ],
        }
        namespace = {
            "env": fake_env,
            "task": type("Task", (), {"language": "move the object"})(),
            "np": __import__("numpy"),
            "hashlib": __import__("hashlib"),
            "_mujoco_integration_state": lambda _sim: __import__("numpy").array(
                [1.0, 2.0]
            ),
            "_planning_model_signature": lambda _model: "model-hash",
            "_planning_cache_signature": lambda _data: "cache-hash",
            "_planning_registered_objects": lambda _wrapper, _task: {
                10: {"body_name": "body-10"}
            },
            "_planning_body_subtree": lambda _model, _body: {10},
            "_geometry_inventory": lambda: inventory,
            "_collect_contacts": lambda: contacts,
            "_tracking_snapshot_state_id": lambda _snapshot: "state-current",
        }
        exec(
            compile(
                ast.Module(body=[sides_function, closure_function, function], type_ignores=[]),
                str(bridge_path),
                "exec",
            ),
            namespace,
        )
        response = namespace["_live_grasp_state"](
            {
                "object_id": "carry",
                "body_id": 10,
                "body_name": "body-10",
                "geom_ids": [0],
                "geom_names": ["object-a"],
            },
            "move the object",
        )
        self.assertEqual(response["status"], "pass")
        self.assertTrue(response["grasp_confirmed"])
        self.assertEqual(response["gripper_current_action"], [-0.95, 0.95])
        self.assertAlmostEqual(
            response["gripper_closure"]["close_command_fraction"], 0.95
        )
        self.assertTrue(response["gripper_closed"])
        self.assertTrue(response["finger_contacts"]["left"])
        self.assertTrue(response["finger_contacts"]["right"])
        self.assertEqual(response["state_id"], "state-current")
        self.assertTrue(response["physical_state_unchanged"])

        robot.gripper.current_action = [0.0]
        neutral_response = namespace["_live_grasp_state"](
            {
                "object_id": "carry",
                "body_id": 10,
                "body_name": "body-10",
                "geom_ids": [0],
                "geom_names": ["object-a"],
            },
            "move the object",
        )
        self.assertEqual(neutral_response["status"], "not_grasped")
        self.assertEqual(neutral_response["gripper_current_action"], [0.0])
        self.assertEqual(neutral_response["gripper_closure"]["status"], "pass")
        self.assertEqual(neutral_response["gripper_closure"]["close_command_fraction"], 0.0)
        self.assertFalse(neutral_response["grasp_confirmed"])
        self.assertTrue(neutral_response["finger_contacts"]["left"])
        self.assertTrue(neutral_response["finger_contacts"]["right"])

        robot.gripper.current_action = [0.95, -0.95]
        opening_response = namespace["_live_grasp_state"](
            {
                "object_id": "carry",
                "body_id": 10,
                "body_name": "body-10",
                "geom_ids": [0],
                "geom_names": ["object-a"],
            },
            "move the object",
        )
        self.assertEqual(opening_response["status"], "not_grasped")
        self.assertFalse(opening_response["grasp_confirmed"])
        self.assertEqual(
            opening_response["reason"], "panda_gripper_close_command_below_threshold"
        )
        self.assertTrue(opening_response["finger_contacts"]["left"])
        self.assertTrue(opening_response["finger_contacts"]["right"])


class GraspStabilityTests(unittest.TestCase):
    @staticmethod
    def _pass(state_id):
        return {
            "status": "pass",
            "grasp_confirmed": True,
            "gripper_closed": True,
            "physical_state_unchanged": True,
            "state_id": state_id,
            "finger_contacts": {
                "status": "pass",
                "left": True,
                "right": True,
            },
        }

    def test_stability_requires_distinct_consecutive_live_states(self):
        monitor = acceptance.GraspStabilityMonitor(required_samples=3)
        monitor.update(self._pass("s0"), rollout_step=0)
        monitor.update(self._pass("s0"), rollout_step=0)
        self.assertFalse(monitor.stable)
        monitor.update(self._pass("s1"), rollout_step=1)
        self.assertFalse(monitor.stable)
        monitor.update(self._pass("s2"), rollout_step=2)
        self.assertTrue(monitor.stable)

    def test_contact_or_close_state_loss_resets_stability(self):
        monitor = acceptance.GraspStabilityMonitor(required_samples=2)
        monitor.update(self._pass("s0"), rollout_step=0)
        failed = self._pass("s1")
        failed["gripper_closed"] = False
        failed["status"] = "not_grasped"
        monitor.update(failed, rollout_step=1)
        self.assertEqual(monitor.consecutive_samples, 0)
        monitor.update(self._pass("s2"), rollout_step=2)
        self.assertFalse(monitor.stable)


class RolloutIntegrityGateTests(unittest.TestCase):
    def test_failed_restore_or_post_restore_state_check_stops_next_student_action(self):
        self.assertTrue(
            acceptance.must_stop_rollout_after_acceptance(
                {"status": "snapshot_restore_failed", "main_state_restored": False}
            )
        )
        self.assertTrue(
            acceptance.must_stop_rollout_after_acceptance(
                {"status": "state_consistency_failed", "main_state_restored": True}
            )
        )
        self.assertFalse(
            acceptance.must_stop_rollout_after_acceptance(
                {"status": "tracking_incomplete", "main_state_restored": True}
            )
        )
        self.assertFalse(
            acceptance.must_stop_rollout_after_acceptance(
                {"status": "blocked", "main_state_restored": None}
            )
        )


class SnapshotAcceptanceTests(unittest.TestCase):
    def _context(self):
        return {
            "teacher": {"objects": [], "object_contact_rules": []},
            "semantic_results": {},
            "instruction": "move the object",
            "subject_id": "carry",
            "binding_validation": {"valid": True},
        }

    def _patch_dependencies(self, tracking_override=None):
        payload = {
            "op": "planning_geometry",
            "carried_binding": {
                "object_id": "carry",
                "body_id": 10,
                "body_name": "carry_body",
                "geom_ids": [2],
                "geom_names": ["carry_geom"],
            },
            "obstacle_bindings": [],
            "expected_instruction": "move the object",
        }
        bundle = {
            "status": "complete",
            "candidate_count": 18,
            "screen_pass_count": 2,
            "candidates": [],
        }
        tracking = tracking_override or {
            "status": "complete",
            "tracked_candidate_count": 2,
            "main_state_restored": True,
            "executable_safe_candidate_ids": ["path-1"],
            "candidates": [
                {
                    "candidate_id": "path-1",
                    "executed_steps": 12,
                    "status": "verified_safe_tracking",
                },
                {
                    "candidate_id": "path-2",
                    "executed_steps": 4,
                    "status": "rejected",
                    "rejection_reasons": ["contact_compiler_violation"],
                },
            ],
            "response_errors": [],
        }
        return (
            patch.object(
                acceptance,
                "_selected_bindings",
                return_value=(payload, {"valid": True}),
            ),
            patch.object(acceptance, "_check_query", return_value={"valid": True}),
            patch.object(
                acceptance,
                "_build_local_bypass_scene",
                return_value={"state_id": "state-1"},
            ),
            patch.object(acceptance, "generate_trajectory_bundle", return_value=bundle),
            patch.object(
                acceptance,
                "save_trajectory_bundle",
                return_value={"json": "bundle.json", "svg": "bundle.svg"},
            ),
            patch.object(acceptance, "track_trajectory_bundle", return_value=tracking),
            patch.object(
                acceptance, "_compare_queries", return_value={"matched": True}
            ),
        )

    def test_snapshot_is_not_created_without_stable_grasp_evidence(self):
        connection = FakeConnection([])
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "not-stable.json"
            result = acceptance.run_snapshot_trajectory_acceptance(
                connection,
                self._context(),
                {"status": "pass", "counted_as_stable_sample": False},
                output,
                decision_step=4,
                grasp_stability={
                    "stable_grasp_observed": False,
                    "current_consecutive_states": 1,
                    "required_distinct_consecutive_states": 3,
                },
            )
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["main_state_restored"])
        self.assertEqual(connection.requests, [])
        self.assertIn(
            "live_stable_grasp_evidence_precondition_not_met",
            result["failure_reasons"][0],
        )

    def test_one_connection_runs_all_candidate_branches_then_restores(self):
        connection = FakeConnection(_success_responses())
        patches = self._patch_dependencies()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "acceptance.json"
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5] as tracker,
                patches[6],
            ):
                result = acceptance.run_snapshot_trajectory_acceptance(
                    connection,
                    self._context(),
                    {"status": "pass", "counted_as_stable_sample": True},
                    output,
                    decision_step=7,
                    grasp_stability={
                        "stable_grasp_observed": True,
                        "current_consecutive_states": 3,
                        "required_distinct_consecutive_states": 3,
                    },
                )
            self.assertEqual(
                result["status"], "complete_with_verified_tracking_candidate"
            )
            self.assertEqual(result["branch_candidate_count"], 2)
            self.assertTrue(result["main_state_restored"])
            self.assertFalse(result["training_label_created"])
            self.assertEqual(
                result["candidates"][1]["rejection_reasons"],
                ["contact_compiler_violation"],
            )
            tracker.assert_called_once()
            self.assertIs(tracker.call_args.kwargs["connection"], connection)
            self.assertEqual(connection.requests[0]["op"], "snapshot")
            self.assertEqual(connection.requests[-2]["op"], "restore")
            self.assertEqual(connection.requests[-1]["op"], "planning_geometry")
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["main_environment_candidate_execution"], False)

    def test_snapshot_geometry_mismatch_blocks_tracking_and_still_restores(self):
        responses = _success_responses(geometry_state=[0.0, 1.01])[:4]
        responses.extend(
            [
                {
                    "restored_snapshot_id": "snapshot-1",
                    "snapshot_state": _snapshot_state(),
                },
                _success_responses()[-1],
            ]
        )
        connection = FakeConnection(responses)
        patches = self._patch_dependencies()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "mismatch.json"
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5] as tracker,
                patches[6],
            ):
                result = acceptance.run_snapshot_trajectory_acceptance(
                    connection,
                    self._context(),
                    {"status": "pass", "counted_as_stable_sample": True},
                    output,
                    decision_step=9,
                    grasp_stability={
                        "stable_grasp_observed": True,
                        "current_consecutive_states": 3,
                        "required_distinct_consecutive_states": 3,
                    },
                )
            self.assertEqual(result["status"], "blocked")
            self.assertIn(
                "planning_geometry_does_not_match_captured_snapshot",
                result["failure_reasons"][0],
            )
            self.assertTrue(result["main_state_restored"])
            tracker.assert_not_called()

    def test_restore_failure_overrides_completed_branch_status(self):
        connection = FakeConnection(_success_responses(restore_error=True))
        patches = self._patch_dependencies()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "restore-fail.json"
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
            ):
                result = acceptance.run_snapshot_trajectory_acceptance(
                    connection,
                    self._context(),
                    {"status": "pass", "counted_as_stable_sample": True},
                    output,
                    decision_step=11,
                    grasp_stability={
                        "stable_grasp_observed": True,
                        "current_consecutive_states": 3,
                        "required_distinct_consecutive_states": 3,
                    },
                )
            self.assertEqual(result["status"], "snapshot_restore_failed")
            self.assertFalse(result["main_state_restored"])
            self.assertIn("final_snapshot_restore_failed", result["failure_reasons"])

    def test_post_restore_geometry_failure_invalidates_candidate_result(self):
        responses = _success_responses()
        responses[-1] = {"error": "planning geometry unavailable"}
        connection = FakeConnection(responses)
        patches = self._patch_dependencies()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "post-restore-geometry-error.json"
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
            ):
                result = acceptance.run_snapshot_trajectory_acceptance(
                    connection,
                    self._context(),
                    {"status": "pass", "counted_as_stable_sample": True},
                    output,
                    decision_step=13,
                    grasp_stability={
                        "stable_grasp_observed": True,
                        "current_consecutive_states": 3,
                        "required_distinct_consecutive_states": 3,
                    },
                )
            self.assertEqual(result["status"], "state_consistency_failed")
            self.assertTrue(result["main_state_restored"])
            self.assertIn(
                "post_restore_geometry_query_incomplete", result["failure_reasons"]
            )

    def test_complete_physical_branches_with_no_qualified_candidate_report_blocker(
        self,
    ):
        connection = FakeConnection(_success_responses())
        rejected_tracking = {
            "status": "complete",
            "tracked_candidate_count": 1,
            "main_state_restored": True,
            "executable_safe_candidate_ids": [],
            "candidates": [
                {
                    "candidate_id": "path-unsafe",
                    "executed_steps": 3,
                    "status": "rejected",
                    "rejection_reasons": ["contact_compiler_violation"],
                }
            ],
            "response_errors": [],
        }
        patches = self._patch_dependencies(tracking_override=rejected_tracking)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "no-qualified.json"
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
            ):
                result = acceptance.run_snapshot_trajectory_acceptance(
                    connection,
                    self._context(),
                    {"status": "pass", "counted_as_stable_sample": True},
                    output,
                    decision_step=12,
                    grasp_stability={
                        "stable_grasp_observed": True,
                        "current_consecutive_states": 3,
                        "required_distinct_consecutive_states": 3,
                    },
                )
            self.assertEqual(result["status"], "complete_no_candidate_passed")
            self.assertEqual(
                result["candidate_rejection_reason_counts"],
                {"contact_compiler_violation": 1},
            )
            self.assertIn(
                "no_candidate_passed_complete_tracking_contact_compiler_and_task_segment_checks",
                result["failure_reasons"],
            )


class RolloutCLIRegressionTests(unittest.TestCase):
    def test_explicit_mode_accepts_complete_teacher_and_mapping_inputs(self):
        tree = ast.parse(ROLLOUT_PATH.read_text(encoding="utf-8"))
        parse_args = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
        )
        namespace = {"argparse": argparse, "Path": Path}
        exec(
            compile(
                ast.Module(body=[parse_args], type_ignores=[]),
                str(ROLLOUT_PATH),
                "exec",
            ),
            namespace,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            teacher = Path(temp_dir) / "teacher.json"
            semantics = Path(temp_dir) / "bindings.json"
            teacher.write_text("{}", encoding="utf-8")
            semantics.write_text("{}", encoding="utf-8")
            argv = [
                "pi05_rollout.py",
                "--max-steps",
                "80",
                "--trajectory-bundle-physical-validation",
                "--teacher-json",
                str(teacher),
                "--semantic-results-json",
                str(semantics),
                "--trajectory-validation-out",
                str(Path(temp_dir) / "result.json"),
            ]
            with patch.object(sys, "argv", argv):
                args = namespace["parse_args"]()
        self.assertTrue(args.trajectory_bundle_physical_validation)
        self.assertEqual(args.max_steps, 80)


if __name__ == "__main__":
    unittest.main()
