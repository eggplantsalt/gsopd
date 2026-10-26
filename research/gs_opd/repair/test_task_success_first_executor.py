# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for exact, provenance-checked task-success prefix resume."""

from __future__ import annotations

import json
import hashlib
import tempfile
import threading
import unittest
import pickle
from pathlib import Path
from multiprocessing.connection import Client, Listener

import numpy as np

from research.gs_opd.repair.task_success_first_executor import (
    _phase_evidence,
    _finger_contacts_clear,
    _generate_native_region_placement_candidates,
    _keypose_rrt_call_count,
    _native_region_entry_offset_m,
    _native_placement_planner_target,
    _object_contact_escape_direction,
    _placement_ik_seed_portfolio,
    _register_verified_operation_skill_from_report,
    _release_retreat_target,
    _select_active_target_predicate,
    _verified_grasp_resume_descriptor,
    _verified_placement_resume_descriptor,
    _verified_runtime_snapshot_descriptor,
    _verify_planner_live_state,
    _transport_exception_outcome,
)
from research.gs_opd.repair.task_success_first_grasp_search import BridgeSession
from research.gs_opd.repair.task_success_first_grasp_search import (
    _record_step,
    _verified_resume_runtime_snapshot,
    _pose_tracking_error_within_tolerance,
    _pose_tracking_window_stagnated,
)


class TaskSuccessFirstExecutorTests(unittest.TestCase):
    def test_keypose_rrt_call_count_ignores_budget_placeholders(self):
        candidates = [
            {
                "preplace_RRT": {
                    "initial_RRT_attempt": {"status": "no_safe_path", "planning_seconds": 12.0},
                    "bounded_extended_RRT_attempt": {"status": "offline_feasible", "planning_seconds": 9.5},
                },
                "terminal_pose_IK_RRT_diagnostic": {
                    "status": "budget_exhausted", "planning_seconds": None,
                },
            },
            {
                "preplace_RRT": {"status": "budget_exhausted", "planning_seconds": None},
                "terminal_pose_IK_RRT_diagnostic": {
                    "status": "no_safe_path", "planning_seconds": 4.0,
                },
            },
            {
                "preplace_RRT": {"status": "offline_feasible", "planning_seconds": 3.0},
                "terminal_pose_IK_RRT_diagnostic": {
                    "status": "not_run_after_unreachable_preplace_route", "planning_seconds": None,
                },
            },
        ]
        self.assertEqual(_keypose_rrt_call_count(candidates), 4)

    def test_terminal_physical_step_can_be_recorded_before_done_is_raised(self):
        class DoneBridge:
            def request(self, request):
                if request["op"] == "step_monitored":
                    return {
                        "done": True, "task_success": True, "reward": 1.0,
                        "images": {
                            "agentview_image": {"data": bytes([1, 2, 3]), "shape": [1, 1, 3]},
                            "robot0_eye_in_hand_image": {"data": bytes([4, 5, 6]), "shape": [1, 1, 3]},
                        },
                    }
                return {"status": "complete", "finger_contacts": {"left": True, "right": True}}

        class Monitor:
            def update(self, *_args, **_kwargs):
                return {"status": "recorded"}

        bridge = DoneBridge()
        args = (
            bridge, np.zeros(7, dtype=np.float32), "release", "candidate", 0,
            {"body_id": 1}, "instruction", {}, Monitor(), [], [],
        )
        _record_step(*args, raise_on_done=False)
        self.assertEqual(len(args[-2]), 1)
        self.assertTrue(args[-2][0]["done"])
        self.assertTrue(args[-2][0]["task_success"])
        with self.assertRaisesRegex(RuntimeError, "environment_done"):
            _record_step(
                bridge, np.zeros(7, dtype=np.float32), "release", "candidate", 1,
                {"body_id": 1}, "instruction", {}, Monitor(), [], [],
            )

    def test_terminal_bridge_exception_preserves_native_success(self):
        completed = _transport_exception_outcome(
            RuntimeError("environment_done_during_grasp_only_branch"),
            {"task_success": True},
        )
        self.assertEqual(completed["status"], "complete_task_success")
        self.assertIsNone(completed["failure_reason"])
        failed = _transport_exception_outcome(
            TimeoutError("waypoint_not_reached"),
            {"task_success": False},
        )
        self.assertEqual(failed["status"], "native_goal_transport_failed")
        self.assertIn("waypoint_not_reached", failed["failure_reason"])

    def test_bridge_reconnect_retries_only_read_only_observation(self):
        with Listener(("127.0.0.1", 0), authkey=b"gs-opd-local") as listener:
            address = listener.address
            received = []

            def serve_reconnect():
                first = listener.accept()
                received.append(first.recv())
                first.close()
                second = listener.accept()
                received.append(second.recv())
                second.send({"status": "complete", "op": "observe"})
                second.close()

            server = threading.Thread(target=serve_reconnect, daemon=True)
            server.start()
            bridge = BridgeSession(address, Path("/tmp/unused-bridge.log"), None)
            bridge.connection = Client(address, authkey=b"gs-opd-local")
            try:
                reply = bridge.request({"op": "observe"})
            finally:
                bridge.close()
            server.join(timeout=2.0)
            self.assertEqual(reply, {"status": "complete", "op": "observe"})
            self.assertEqual([row["op"] for row in received], ["observe", "observe"])
        self.assertNotIn("step_monitored", BridgeSession._READ_ONLY_RECONNECT_OPS)
        self.assertNotIn("restore_runtime_snapshot", BridgeSession._READ_ONLY_RECONNECT_OPS)

    @staticmethod
    def _placement_context(object_id: str, position: list[float]) -> dict:
        return {
            "target_object_id": object_id,
            "target_root_pose_world": {
                "position_m": position,
                "rotation": np.eye(3).tolist(),
            },
            "target_collision_geometries": [{
                "center_root_m": [0.0, 0.0, 0.0],
                "rotation_root": np.eye(3).tolist(),
                "size_m": [0.02, 0.02, 0.05],
            }],
        }

    def test_native_region_candidates_separate_completed_objects_and_reuse_support_height(self):
        occupied = self._placement_context("object_a", [-0.03, 0.01, 0.997])
        target = self._placement_context("object_b", [0.0, 0.0, 1.14])
        candidates = _generate_native_region_placement_candidates(
            predicate=["on", "object_b", "region"],
            region={
                "site_position_world_m": [0.0, 0.0, 0.905],
                "site_rotation_world": np.eye(3).tolist(),
                "site_size_m": [0.075, 0.075, 0.0025],
            },
            target_context=target,
            occupied_contexts=[occupied],
            maximum_candidates=4,
        )
        self.assertEqual(len(candidates), 4)
        offsets = [tuple(np.round(row["region_local_xy_m"], 6)) for row in candidates]
        self.assertEqual(len(offsets), len(set(offsets)))
        self.assertTrue(all(abs(row["position_world_m"][2] - 0.997) < 1e-9 for row in candidates))
        self.assertTrue(all(row["estimated_completed_object_clearance_m"] > 0 for row in candidates))
        self.assertTrue(all(row["geometry_candidate_only_not_safety_certified"] for row in candidates))
        directions = [tuple(np.round(row["entry_direction_world"], 6)) for row in candidates]
        self.assertEqual(len(directions), len(set(directions)))
        self.assertEqual(
            [row["entry_direction_variant_degrees"] for row in candidates],
            [0.0, 90.0, -90.0, 180.0],
        )
        self.assertEqual(
            [row["route_strategy"] for row in candidates],
            ["jointspace_rrt", "elevated_osc_waypoints", "jointspace_rrt", "elevated_osc_waypoints"],
        )

    def test_native_region_placement_reuses_orientation_for_similar_compiled_envelopes(self):
        occupied = self._placement_context("object_a", [-0.03, 0.01, 0.997])
        occupied["target_root_pose_world"]["rotation"] = [
            [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        ]
        occupied["target_collision_geometries"][0]["center_root_m"] = [0.0005, 0.0, 0.0]
        target = self._placement_context("object_b", [0.0, 0.0, 1.14])
        candidates = _generate_native_region_placement_candidates(
            predicate=["on", "object_b", "region"],
            region={
                "site_position_world_m": [0.0, 0.0, 0.905],
                "site_rotation_world": np.eye(3).tolist(),
                "site_size_m": [0.075, 0.075, 0.0025],
            },
            target_context=target,
            occupied_contexts=[occupied],
            maximum_candidates=1,
        )
        self.assertEqual(candidates[0]["orientation_source"], "compatible_completed_object_pose")
        np.testing.assert_allclose(candidates[0]["rotation_world"], occupied["target_root_pose_world"]["rotation"])

    def test_placement_waypoint_stagnation_detection_keeps_progressing_routes(self):
        self.assertFalse(_pose_tracking_window_stagnated(
            [(0.10, 0.2), (0.09, 0.18), (0.08, 0.16), (0.07, 0.14)],
            window_actions=3,
        ))
        self.assertTrue(_pose_tracking_window_stagnated(
            [(0.10, 0.2), (0.099, 0.199), (0.101, 0.201), (0.100, 0.200)],
            window_actions=3,
        ))
        self.assertFalse(_pose_tracking_error_within_tolerance(0.006, 0.05, 0.004, 0.045))
        self.assertTrue(_pose_tracking_error_within_tolerance(0.006, 0.05, 0.010, 0.10))

    def test_planner_exception_is_not_misclassified_as_live_state_mutation(self):
        result = _verify_planner_live_state(
            {"status": "unavailable", "live_state_unchanged": None}, "same-hash", "same-hash",
        )
        self.assertTrue(result["live_state_unchanged"])
        self.assertEqual(result["status"], "exact_match")
        changed = _verify_planner_live_state(
            {"status": "unavailable", "live_state_unchanged": None}, "different", "same-hash",
        )
        self.assertFalse(changed["live_state_unchanged"])

    def test_native_region_placement_uses_object_relative_lateral_staging(self):
        self.assertEqual(_native_region_entry_offset_m(["on", "pot", "stove_region"]), 0.06)
        self.assertEqual(_native_region_entry_offset_m(["in", "cup", "microwave_region"]), 0.06)

    def test_placement_IK_seed_portfolio_is_deterministic_and_arm_bounded(self):
        robot = {"joint_qpos": [0.0, 0.1, -0.2, -1.0, 0.3, 1.1, 0.2, 0.01, -0.01]}
        seeds = _placement_ik_seed_portfolio(robot)
        self.assertEqual(len(seeds), 6)
        self.assertTrue(all(len(seed) == 7 for seed in seeds))
        self.assertEqual(seeds, _placement_ik_seed_portfolio(robot))
        self.assertTrue(all(seed != robot["joint_qpos"][:7] for seed in seeds))

    def test_RRT_targets_elevated_native_region_entries(self):
        goal = np.eye(4); goal[:3, 3] = [0.1, 0.2, 0.3]
        high = np.eye(4); high[:3, 3] = [0.4, 0.5, 0.6]
        selected, role = _native_placement_planner_target(["on", "pot", "stove"], goal, high)
        np.testing.assert_array_equal(selected, high)
        self.assertEqual(role, "open_support_high_entry_pose")
        selected, role = _native_placement_planner_target(["in", "cup", "microwave"], goal, high)
        np.testing.assert_array_equal(selected, high)
        self.assertEqual(role, "container_entry_high_pose")

    def test_direct_runtime_snapshot_resume_requires_matching_scene_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "held_object.pkl"
            snapshot.write_bytes(pickle.dumps({
                "schema_version": "gs_opd_runtime_snapshot.v1",
                "scene_metadata": {
                    "benchmark": "obstacle_avoidance", "level": 2, "task_id": 2,
                    "environment_seed": 0, "init_state_index": 0,
                    "bddl_sha256": "bddl-hash", "instruction": "put both pots on stove",
                },
                "model_geometry_sha256": "model-hash",
                "snapshot": {"mujoco_state": [0.0, 0.1]},
            }))
            descriptor = _verified_runtime_snapshot_descriptor(
                snapshot, benchmark="obstacle_avoidance", level=2, task_id=2,
                seed=0, init_state=0, model_sha="model-hash",
                bddl_sha="bddl-hash", instruction="put both pots on stove",
            )
            self.assertEqual(descriptor["payload_sha256"], hashlib.sha256(snapshot.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "model_fingerprint"):
                _verified_runtime_snapshot_descriptor(
                    snapshot, benchmark="obstacle_avoidance", level=2, task_id=2,
                    seed=0, init_state=0, model_sha="different-model",
                    bddl_sha="bddl-hash", instruction="put both pots on stove",
                )

    def test_grasp_search_resume_preserves_a_partial_task_snapshot_with_identity_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "pot1_complete_pot2_pending.pkl"
            snapshot.write_bytes(pickle.dumps({
                "schema_version": "gs_opd_runtime_snapshot.v1",
                "model_geometry_sha256": "model-hash",
                "scene_metadata": {
                    "benchmark": "obstacle_avoidance", "level": 1, "task_id": 2,
                    "environment_seed": 0, "init_state_index": 0,
                    "bddl_sha256": "bddl-hash", "instruction": "put both moka pots on the stove",
                },
                "snapshot": {"mujoco_state": [1.0, 2.0, 3.0]},
            }))
            descriptor = _verified_resume_runtime_snapshot(
                snapshot, benchmark="obstacle_avoidance", level=1, task_id=2,
                seed=0, init_state=0, model_sha="model-hash", bddl_sha="bddl-hash",
                instruction="put both moka pots on the stove",
            )
            self.assertEqual(descriptor["path"], snapshot)
            np.testing.assert_array_equal(
                descriptor["snapshot_state"]["mujoco_integration_state"], [1.0, 2.0, 3.0],
            )
            with self.assertRaisesRegex(ValueError, "identity_mismatch"):
                _verified_resume_runtime_snapshot(
                    snapshot, benchmark="obstacle_avoidance", level=2, task_id=2,
                    seed=0, init_state=0, model_sha="model-hash", bddl_sha="bddl-hash",
                    instruction="put both moka pots on the stove",
                )

    def test_empty_target_contact_is_clear_even_when_grasp_check_status_is_fail(self):
        self.assertTrue(_finger_contacts_clear({
            "status": "fail", "left": False, "right": False,
            "contact_count": 0, "matched_contacts": {"left": [], "right": []},
        }))
        self.assertFalse(_finger_contacts_clear({
            "status": "fail", "left": False, "right": True,
            "contact_count": 1, "matched_contacts": {"left": [], "right": [{"geom": "target"}]},
        }))

    def test_object_contact_normal_proposes_outward_scene_escape_from_real_contact(self):
        result = _object_contact_escape_direction({
            "status": "contact",
            "contacts": [{
                "geom1": {"name": "target_g0"},
                "geom2": {"name": "white_place_box_g2"},
                "normal_geom1_to_geom2": [0.0, -1.0, 0.0],
                "position": [0.0, 0.05, 0.0],
                "force_and_torque": [0.0, 3.0, 0.0, 0, 0, 0],
            }],
        }, {"geom_names": ["target_g0"]}, [0.0, 0.1, 0.0])
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["direction_world"], [0.0, 1.0, 0.0])
        self.assertEqual(result["blocker_geom_names"], ["white_place_box_g2"])
        self.assertAlmostEqual(result["support_radius_estimate_m"], 0.05)
        self.assertIsNone(_object_contact_escape_direction({
            "status": "contact",
            "contacts": [{
                "geom1": {"name": "gripper0_finger1_pad_collision"},
                "geom2": {"name": "target_g0"},
                "normal_geom1_to_geom2": [1.0, 0.0, 0.0],
                "position": [0, 0, 0],
            }],
        }, {"geom_names": ["target_g0"]}, [0, 0, 0]))
        self.assertFalse(_finger_contacts_clear({
            "status": "unavailable", "left": False, "right": False,
            "contact_count": 0, "matched_contacts": {"left": [], "right": []},
        }))

    def test_phase_evidence_uses_complete_verified_object_binding(self):
        predicate = ["on", "object_a", "support_region"]
        binding = {
            "object_id": "object_a",
            "body_id": 3,
            "body_name": "object_a_main",
            "geom_ids": [10, 11],
            "geom_names": ["object_a_geom_0", "object_a_geom_1"],
        }

        class Bridge:
            def __init__(self):
                self.grasp_binding = None

            def request(self, request):
                if request["op"] == "observe":
                    return {
                        "instruction": "place object a",
                        "timestep": 1,
                        "snapshot_state": {
                            "object_states": {
                                "objects_dict:object_a": {
                                    "body_id": 3, "body_name": "object_a_main",
                                    "root_pos": [0.0, 0.0, 0.1],
                                    "root_quat": [1.0, 0.0, 0.0, 0.0],
                                    "joint_qvel": [0.0] * 6,
                                },
                            },
                            "robot_states": [{
                                "eef_pos": [0.0, 0.0, 0.2],
                                "eef_quat": [1.0, 0.0, 0.0, 0.0],
                            }],
                        },
                    }
                if request["op"] == "native_task_status":
                    return {
                        "status": "complete", "task_success": False,
                        "native_goal_values": [{"predicate": predicate, "value": True}],
                    }
                if request["op"] == "grasp_state":
                    self.grasp_binding = request["carried_binding"]
                    return {
                        "status": "not_grasped", "grasp_confirmed": False,
                        "finger_contacts": {"status": "pass", "left": False, "right": True},
                    }
                raise AssertionError(request["op"])

        bridge = Bridge()
        evidence, _ = _phase_evidence(
            bridge, None, "object_a", 3, predicate, target_binding=binding,
        )
        self.assertEqual(bridge.grasp_binding["geom_ids"], [10, 11])
        self.assertEqual(
            bridge.grasp_binding["geom_names"],
            ["object_a_geom_0", "object_a_geom_1"],
        )
        self.assertFalse(evidence.grasp_confirmed)
        self.assertFalse(evidence.finger_contacts_clear)

    def test_release_retreat_moves_horizontally_away_from_live_object(self):
        eef = np.asarray([0.0, 0.0, 0.2])
        obj = np.asarray([-0.03, -0.04, 0.1])
        target = _release_retreat_target(eef, obj, retreat_distance_m=0.003)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(float(np.linalg.norm(target[:2] - eef[:2])), 0.003, places=8)
        self.assertEqual(float(target[2]), float(eef[2]))
        self.assertGreater(float(np.dot(target[:2] - eef[:2], eef[:2] - obj[:2])), 0.0)

    def test_release_retreat_rejects_invalid_or_vertical_only_direction(self):
        self.assertIsNone(_release_retreat_target([0, 0, 0], [0, 0, 1]))
        self.assertIsNone(_release_retreat_target([float("nan"), 0, 0], [0, 0, 0]))

    def test_target_selection_uses_first_pending_native_object_goal_in_order(self):
        native = {
            "status": "complete",
            "native_goal_values": [
                {"predicate": ["on", "object_a", "region_a"], "value": True},
                {"predicate": ["in", "object_b", "region_b"], "value": False},
                {"predicate": ["on", "object_c", "region_c"], "value": False},
                {"predicate": ["turnon", "fixture"], "value": False},
            ],
        }
        selected, source, pending = _select_active_target_predicate(native, None)
        self.assertEqual(selected, ["in", "object_b", "region_b"])
        self.assertEqual(source, "first_pending_native_object_goal_bddl_order")
        self.assertEqual(pending, [
            ["in", "object_b", "region_b"], ["on", "object_c", "region_c"]
        ])
        selected, source, _ = _select_active_target_predicate(
            native, ["on", "object_c", "region_c"]
        )
        self.assertEqual(selected, ["on", "object_c", "region_c"])
        self.assertEqual(source, "explicit_cli")

    def _report(self, snapshot: Path) -> dict:
        return {
            "schema_version": "gs_opd_task_success_first_grasp_search.v1",
            "status": "task_success_candidate_created",
            "scope": {
                "benchmark": "obstacle_avoidance",
                "level": 2,
                "task_id": 2,
                "environment_seed": 0,
                "init_state": 0,
                "model_geometry_sha256": "model-hash",
                "bddl_sha256": "bddl-hash",
            },
            "route_constraints": {"official_safety_demonstrations_used": False},
            "active_target": {"native_goal_predicate": [
                "on", "moka_pot_1", "flat_stove_1_cook_region"
            ]},
            "success": {
                "target_object_id": "moka_pot_1",
                "task_success_candidate": True,
                "safety_certified": False,
                "snapshot": {
                    "path": str(snapshot),
                    "snapshot_state": {"mujoco_integration_state": [0.0]},
                },
            },
        }

    def _kwargs(self) -> dict:
        return {
            "benchmark": "obstacle_avoidance",
            "level": 2,
            "task_id": 2,
            "seed": 0,
            "init_state": 0,
            "model_sha": "model-hash",
            "bddl_sha": "bddl-hash",
            "target_id": "moka_pot_1",
            "predicate": ["on", "moka_pot_1", "flat_stove_1_cook_region"],
        }

    def test_exact_grasp_prefix_is_accepted_for_same_task_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "lifted.pkl"
            snapshot.write_bytes(b"runtime snapshot")
            result = _verified_grasp_resume_descriptor(self._report(snapshot), **self._kwargs())
            self.assertEqual(result["path"], snapshot)
            self.assertEqual(result["success"]["target_object_id"], "moka_pot_1")

    def test_resume_rejects_other_task_fingerprint_or_official_data(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "lifted.pkl"
            snapshot.write_bytes(b"runtime snapshot")
            report = self._report(snapshot)
            with self.assertRaisesRegex(ValueError, "bddl_sha256_mismatch"):
                _verified_grasp_resume_descriptor(
                    report, **{**self._kwargs(), "bddl_sha": "another-bddl"}
                )
            report["route_constraints"]["official_safety_demonstrations_used"] = True
            with self.assertRaisesRegex(ValueError, "official_demo"):
                _verified_grasp_resume_descriptor(report, **self._kwargs())

    def test_partial_native_goal_snapshot_is_accepted_for_release_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "placement_failed.pkl"
            snapshot.write_bytes(pickle.dumps({
                "schema_version": "gs_opd_runtime_snapshot.v1",
                "scene_metadata": {
                    "benchmark": "obstacle_avoidance", "level": 2, "task_id": 2,
                    "environment_seed": 0, "init_state_index": 0,
                    "bddl_sha256": "bddl-hash",
                },
                "model_geometry_sha256": "model-hash",
                "snapshot": {"mujoco_state": [0.0]},
            }))
            predicate = ["on", "moka_pot_1", "flat_stove_1_cook_region"]
            native = {
                "status": "complete", "task_success": False,
                "native_goal_values": [
                    {"predicate": predicate, "value": True},
                    {"predicate": ["on", "moka_pot_2", "flat_stove_1_cook_region"], "value": False},
                ],
            }
            report = {
                "status": "native_goal_precise_placement_failed",
                "scope": {
                    "benchmark": "obstacle_avoidance", "level": 2, "task_id": 2,
                    "seed": 0, "init_state": 0, "model_geometry_sha256": "model-hash",
                },
                "policy": {
                    "official_safety_demonstration_data_used": False,
                    "object_or_robot_state_written_directly": False,
                    "all_actions_are_real_MuJoCo_steps": True,
                },
                "task": {
                    "BDDL_sha256": "bddl-hash",
                    "target_object_from_active_BDDL": "moka_pot_1",
                    "native_goal_predicate": predicate,
                },
                "segments": [{
                    "failure_phase": "precise_placement",
                    "native_goal_predicate": predicate,
                    "final_native_task_status": native,
                    "terminal_snapshot": str(snapshot),
                }],
            }
            result = _verified_placement_resume_descriptor(report, **self._kwargs())
            self.assertEqual(result["path"], snapshot)
            self.assertEqual(result["native_task_status"]["task_success"], False)

            report["segments"][0]["final_native_task_status"]["native_goal_values"][0]["value"] = False
            with self.assertRaisesRegex(ValueError, "current_native_goal_to_be_true"):
                _verified_placement_resume_descriptor(report, **self._kwargs())

    def test_transport_timeout_with_verified_native_contact_can_resume_for_release(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "transport_timeout_contact.pkl"
            predicate = ["on", "moka_pot_1", "flat_stove_1_cook_region"]
            snapshot.write_bytes(pickle.dumps({
                "schema_version": "gs_opd_runtime_snapshot.v1",
                "scene_metadata": {
                    "benchmark": "obstacle_avoidance", "level": 2, "task_id": 2,
                    "environment_seed": 0, "init_state_index": 0,
                    "bddl_sha256": "bddl-hash",
                },
                "model_geometry_sha256": "model-hash",
                "snapshot": {"mujoco_state": [0.0]},
            }))
            native = {
                "status": "complete", "task_success": False,
                "native_goal_values": [
                    {"predicate": predicate, "value": True},
                    {"predicate": ["on", "moka_pot_2", "flat_stove_1_cook_region"], "value": False},
                ],
            }
            report = {
                "status": "native_goal_transport_failed",
                "scope": {
                    "benchmark": "obstacle_avoidance", "level": 2, "task_id": 2,
                    "seed": 0, "init_state": 0, "model_geometry_sha256": "model-hash",
                },
                "policy": {
                    "official_safety_demonstration_data_used": False,
                    "object_or_robot_state_written_directly": False,
                    "all_actions_are_real_MuJoCo_steps": True,
                },
                "task": {
                    "BDDL_sha256": "bddl-hash",
                    "target_object_from_active_BDDL": "moka_pot_1",
                    "native_goal_predicate": predicate,
                },
                "final_native_task_status": native,
                "segments": [{
                    "type": "expert_native_goal_stage_control",
                    "placement_search": {"candidate_attempts": [{
                        "candidate": {"candidate_id": "layout-1", "predicate": predicate},
                        "status": "native_goal_transport_failed",
                        "target_predicate_value": True,
                        "terminal_snapshot": str(snapshot),
                    }]},
                }],
            }
            result = _verified_placement_resume_descriptor(report, **self._kwargs())
            self.assertEqual(result["path"], snapshot)
            self.assertEqual(result["segment"]["failure_phase"], "transport_after_native_goal_contact")
            self.assertTrue(result["native_task_status"]["native_goal_values"][0]["value"])

    def test_physically_completed_partial_cycle_is_added_to_skill_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library.json"
            library.write_text(json.dumps({
                "schema_version": "gs_opd_task_skill_experience_library.v1",
                "source_policy": {"official_safety_demonstration_data_used": False},
                "operation_skill_templates": [{
                    "skill_id": "l0-cycle", "object_family": "moka_pot",
                    "operation": "pick_transport_place_release",
                    "phase_sequence": ["approach", "stable_grasp", "lift", "transport",
                                       "precise_placement", "release", "withdrawal", "settle"],
                    "historical_actions_replayed": False,
                }],
            }), encoding="utf-8")
            grasp_path = root / "grasp.json"
            grasp_path.write_text(json.dumps({
                "status": "task_success_candidate_created",
                "route_constraints": {"official_safety_demonstrations_used": False},
                "success": {"stable_bilateral_grasp": {"stable_grasp_observed": True}},
                "physical_branches": [{"branch_control_actions": 7}],
            }), encoding="utf-8")
            placement_path = root / "placement.json"
            predicate = ["on", "moka_pot_1", "flat_stove"]
            native = {"status": "complete", "task_success": False,
                      "native_goal_values": [{"predicate": predicate, "value": True}]}
            placement_path.write_text(json.dumps({
                "status": "native_goal_precise_placement_failed",
                "resumed_from_grasp_skill": {"source_report": str(grasp_path)},
                "segments": [{"failure_phase": "precise_placement", "recorded_actions": 10}],
            }), encoding="utf-8")
            report_path = root / "release.json"
            report = {
                "status": "partial_predicate_completed",
                "scope": {"benchmark": "obstacle_avoidance", "level": 2,
                          "task_id": 2, "seed": 0, "init_state": 0},
                "task": {"instruction": "place both moka pots", "target_object_from_active_BDDL": "moka_pot_1",
                         "native_goal_predicate": predicate},
                "final_native_task_status": native,
                "resumed_from_native_goal_contact_snapshot": {"source_report": str(placement_path)},
                "segments": [{"type": "resumed_expert_release_withdrawal", "recorded_actions": 5,
                              "partial_goal_settled": True}],
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            result = _register_verified_operation_skill_from_report(library, report, report_path)
            self.assertEqual(result["status"], "skill_added")
            self.assertEqual(result["physical_actions_in_source_prefix"], 22)


if __name__ == "__main__":
    unittest.main()
