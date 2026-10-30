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
    _run_articulated_fixture_goal,
    ARTICULATED_FIXTURE_PRECONTACT_ACTION_BUDGET,
    ARTICULATED_FIXTURE_CONTACT_ACTION_BUDGET,
    ARTICULATED_FIXTURE_TOTAL_ACTION_BUDGET,
    ARTICULATED_FIXTURE_CONTACT_POSITION_TOLERANCE_M,
    ARTICULATED_FIXTURE_FOLLOW_POSITION_TOLERANCE_M,
    ARTICULATED_FIXTURE_HINGE_FOLLOW_LEAD_RAD,
    ARTICULATED_FIXTURE_SLIDE_CONTACT_OUTWARD_OFFSET_M,
    ARTICULATED_FIXTURE_SLIDE_HANDLE_APPROACH_OUTWARD_M,
    ARTICULATED_FIXTURE_SLIDE_HANDLE_EEF_HEIGHT_OFFSET_M,
    ARTICULATED_FIXTURE_SLIDE_PRECONTACT_OUTWARD_M,
    ARTICULATED_FIXTURE_SLIDE_TOOL_DOWN_ANGLE_RAD,
    ARTICULATED_FIXTURE_SLIDE_UNDER_TOP_HEIGHT_OFFSET_M,
    _articulated_fixture_follow_pose,
    _articulated_fixture_press_quaternion,
    _articulated_fixture_slide_approach_waypoints,
    _articulated_fixture_slide_grasp_quaternion,
    _articulated_fixture_robot_contact_pairs,
    _articulated_fixture_terminal_outcome,
    _thin_articulated_fixture_rrt_route,
    MAX_NATIVE_PLACEMENT_ACTIONS,
    _make_precomputed_rrt_route_cache,
    _validate_precomputed_rrt_route,
    _joint_limit_diagnostics,
    _live_joint_limit_specification,
    _transport_grasp_loss_reason,
    _native_transport_action_budget,
    _phase_evidence,
    _finger_contacts_clear,
    _generate_native_region_placement_candidates,
    _native_region_progress_route,
    _native_region_entry_transit_poses,
    _native_region_progress_waypoint_target,
    _native_object_relative_eef_target_world,
    _native_region_corridor_obstacle_clearance,
    _native_region_progress_tracking_tolerances,
    _keypose_rrt_call_count,
    _native_region_entry_offset_m,
    _native_placement_planner_target,
    _native_goal_insertion_rrt_path_arrays,
    _object_contact_escape_direction,
    _placement_ik_seed_portfolio,
    _preflight_row_is_task_success_admissible,
    _register_verified_operation_skill_from_report,
    _release_retreat_target,
    _select_active_target_predicate,
    _verified_grasp_resume_descriptor,
    _verified_placement_resume_descriptor,
    _verified_runtime_snapshot_descriptor,
    _verify_planner_live_state,
    _transport_exception_outcome,
    _native_episode_terminated,
    _task_success_waypoint_fallback_reason,
    _task_success_collision_route_policy,
    _task_success_first_grasp_search_command,
    _parse_args as _parse_task_success_executor_args,
)
from research.gs_opd.repair.task_success_first_grasp_search import BridgeSession
from research.gs_opd.repair.task_success_first_grasp_search import (
    _record_step,
    _compress_eef_route_with_indices,
    _verified_resume_runtime_snapshot,
    _pose_tracking_error_within_tolerance,
    _live_contact_pose_is_close_to_candidate,
    _object_relative_target_pose,
    _pose_tracking_window_stagnated,
)


class TaskSuccessFirstExecutorTests(unittest.TestCase):
    def test_articulated_success_done_is_recorded_as_termination_not_failure(self):
        failure, stop = _articulated_fixture_terminal_outcome(
            "RuntimeError: environment_done_during_grasp_only_branch",
            {"task_success": True},
        )
        self.assertIsNone(failure)
        self.assertEqual(stop, "native_task_success_terminated_environment")

        failure, stop = _articulated_fixture_terminal_outcome(
            "RuntimeError: articulated_fixture_live_follow_joint_progress_stalled",
            {"task_success": False},
        )
        self.assertEqual(failure, "RuntimeError: articulated_fixture_live_follow_joint_progress_stalled")
        self.assertIsNone(stop)

    def test_articulated_live_follow_can_close_the_gripper_after_switch_contact(self):
        from unittest.mock import patch

        captured = {"gripper_actions": []}

        class FakeBridge:
            def request(self, request):
                if request.get("op") == "observe":
                    return {"snapshot_state": {"robot_states": [{
                        "eef_pos": [0.2, 0.0, 1.0],
                        "eef_quat": [1.0, 0.0, 0.0, 0.0],
                        "joint_qpos": [0.0] * 7,
                    }]}}
                raise AssertionError(f"unexpected Bridge operation: {request.get('op')}")

        def fake_tracking_action(*_args, gripper_action=0.0, **_kwargs):
            captured["gripper_actions"].append(gripper_action)
            return np.zeros(7, dtype=np.float32)

        def fake_record_step(_bridge, _action, *_args, **_kwargs):
            records.append({"done": False, "snapshot_state": {"robot_states": [{
                "eef_pos": [0.2, 0.0, 1.0],
            }]}})
            frames.append({})

        from research.gs_opd.repair.task_success_first_grasp_search import _advance_pose

        records, frames = [], []
        _advance_pose.controller_fields = {}
        with patch("research.gs_opd.repair.trajectory_tracking.world_pose_tracking_action",
                   fake_tracking_action), \
             patch("research.gs_opd.repair.task_success_first_grasp_search._record_step",
                   fake_record_step):
            with self.assertRaisesRegex(TimeoutError, "test_pinch_follow_did_not_reach_pose"):
                _advance_pose(
                    FakeBridge(), np.asarray([0.19, 0.0, 1.0]),
                    np.asarray([1.0, 0.0, 0.0, 0.0]),
                    "test_pinch_follow", "test-switch", [0], 1,
                    {"body_id": 1}, "turn off switch", {}, None,
                    records, frames, gripper_action=1.0,
                )
        self.assertEqual(captured["gripper_actions"], [1.0])

    def test_articulated_fixture_press_orientation_aligns_fingers_with_hinge_contact(self):
        from scipy.spatial.transform import Rotation

        outward = np.asarray([-0.6, -0.8, 0.0])
        quaternion = _articulated_fixture_press_quaternion(
            np.asarray([1.0, 0.0, 0.0, 0.0]), outward,
        )
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        inward = -outward / np.linalg.norm(outward)
        self.assertTrue(np.allclose(rotation[:, 2], inward, atol=1e-8))
        self.assertAlmostEqual(float(np.dot(rotation[:, 0], inward)), 0.0, places=8)
        self.assertAlmostEqual(float(rotation[:, 0][2]), 0.0, places=8)
        self.assertGreater(float(np.linalg.det(rotation)), 0.999999)

    def test_articulated_fixture_slide_orientation_faces_handle_and_closes_across_bar(self):
        from scipy.spatial.transform import Rotation

        outward = np.asarray([0.0, 1.0, 0.0])
        quaternion = _articulated_fixture_slide_grasp_quaternion(outward)
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        expected_down = float(np.sin(ARTICULATED_FIXTURE_SLIDE_TOOL_DOWN_ANGLE_RAD))
        expected_inward = float(np.cos(ARTICULATED_FIXTURE_SLIDE_TOOL_DOWN_ANGLE_RAD))
        np.testing.assert_allclose(rotation[:, 0], [-1.0, 0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(
            rotation[:, 2], [0.0, -expected_inward, -expected_down], atol=1e-8,
        )
        self.assertGreater(float(np.linalg.det(rotation)), 0.999999)

    def test_articulated_slide_approach_enters_below_top_before_handle_height(self):
        contact = np.asarray([0.008558, -0.211354, 0.946140])
        outside, inward, handle_height = _articulated_fixture_slide_approach_waypoints(
            contact, np.asarray([0.0, 4.0, 0.0]),
        )
        under_top_height = contact[2] + ARTICULATED_FIXTURE_SLIDE_UNDER_TOP_HEIGHT_OFFSET_M
        self.assertAlmostEqual(outside[1], contact[1] + ARTICULATED_FIXTURE_SLIDE_PRECONTACT_OUTWARD_M)
        self.assertAlmostEqual(inward[1], contact[1] + ARTICULATED_FIXTURE_SLIDE_HANDLE_APPROACH_OUTWARD_M)
        self.assertAlmostEqual(handle_height[1], inward[1])
        self.assertAlmostEqual(outside[2], under_top_height)
        self.assertAlmostEqual(inward[2], under_top_height)
        self.assertAlmostEqual(
            handle_height[2], contact[2] + ARTICULATED_FIXTURE_SLIDE_HANDLE_EEF_HEIGHT_OFFSET_M,
        )
        self.assertAlmostEqual(
            contact[1] + ARTICULATED_FIXTURE_SLIDE_CONTACT_OUTWARD_OFFSET_M,
            contact[1] + 0.010,
        )
        self.assertGreater(outside[1], inward[1])

    def test_articulated_fixture_follow_pose_keeps_a_bounded_contact_side_offset(self):
        target_position, target_quaternion = _articulated_fixture_follow_pose(
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            {
                "joint": {"joint_type": "hinge", "position": np.pi / 2, "target_position": 0.0},
                "joint_axis_world": [0.0, 0.0, 1.0],
                "joint_origin_world_m": [0.0, 0.0, 0.0],
            },
            joint_position_at_anchor=np.pi / 2,
            follow_lead=0.1,
        )
        self.assertTrue(np.allclose(target_position, [np.cos(-0.1), np.sin(-0.1), 0.0], atol=1e-8))
        self.assertTrue(np.allclose(
            target_quaternion, [np.cos(0.05), 0.0, 0.0, -np.sin(0.05)], atol=1e-8,
        ))

    def test_articulated_fixture_follow_pose_uses_fixed_contact_anchor_with_bounded_lead(self):
        target_position, _target_quaternion = _articulated_fixture_follow_pose(
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            {
                "joint": {"joint_type": "hinge", "position": 0.8, "target_position": 0.0},
                "joint_axis_world": [0.0, 0.0, 1.0],
                "joint_origin_world_m": [0.0, 0.0, 0.0],
            },
            joint_position_at_anchor=1.0,
            follow_lead=0.1,
        )
        self.assertTrue(np.allclose(target_position, [np.cos(-0.3), np.sin(-0.3), 0.0], atol=1e-8))

    def test_turnoff_follow_can_track_the_native_closed_hinge_stop(self):
        target_position, _target_quaternion = _articulated_fixture_follow_pose(
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            {
                "joint": {
                    "joint_type": "hinge", "position": 0.0325,
                    "target_position": -0.0025, "range": [-0.005, 2.1],
                },
                "joint_axis_world": [0.0, 0.0, 1.0],
                "joint_origin_world_m": [0.0, 0.0, 0.0],
            },
            joint_position_at_anchor=1.0,
            follow_lead=0.04,
            allow_terminal_stop_follow=True,
        )
        self.assertTrue(np.allclose(
            target_position, [np.cos(-1.005), np.sin(-1.005), 0.0], atol=1e-8,
        ))

    def test_articulated_fixture_contact_requires_moving_fixture_geometry(self):
        record = {"contacts": {"contacts": [
            {
                "geom1": {"name": "gripper0_finger2_collision"},
                "geom2": {"name": "flat_stove_1_g19"},
                "body1": {"name": "gripper0_right_inner_finger"},
                "body2": {"name": "flat_stove_1_button"},
            },
            {
                "geom1": {"name": "robot0_link6_collision"},
                "geom2": {"name": "flat_stove_1_g2"},
                "body1": {"name": "robot0_link6"},
                "body2": {"name": "flat_stove_1_main"},
            },
        ]}}
        pairs = _articulated_fixture_robot_contact_pairs(
            record, "flat_stove_1", ["flat_stove_1_g19"],
        )
        self.assertEqual(pairs, [{
            "robot_geom": "gripper0_finger2_collision",
            "fixture_geom": "flat_stove_1_g19",
        }])

    def test_articulated_fixture_rrt_route_thinning_preserves_endpoint(self):
        route = np.column_stack([
            np.arange(5, dtype=np.float64),
            np.zeros((5, 6), dtype=np.float64),
        ])
        thinned, indices = _thin_articulated_fixture_rrt_route(route, stride=2)
        self.assertEqual(indices, [0, 2, 4])
        np.testing.assert_array_equal(thinned, route[[0, 2, 4]])

    def test_articulated_fixture_rrt_route_thinning_rejects_invalid_stride(self):
        with self.assertRaisesRegex(ValueError, "stride_must_be_a_positive_integer"):
            _thin_articulated_fixture_rrt_route(np.zeros((3, 7)), stride=0)

    def test_articulated_fixture_timeout_persists_actions_and_qpos_feedback(self):
        from unittest.mock import patch

        saved = {}
        context = {
            "status": "complete", "read_only": True,
            "fixture_id_from_active_BDDL": "flat_stove_1",
            "contact_point_path_world_m": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]],
            "outward_direction_world": [0.0, -1.0, 0.0],
            "joint": {"joint_name": "stove_button", "position": 0.98,
                      "target_position": 0.0, "range": [-0.005, 2.1]},
        }
        native = {"status": "complete", "task_success": False,
                  "native_goal_values": [{"predicate": ["turnoff", "flat_stove_1"], "value": False}]}
        rrt_requests = []

        class FakeBridge:
            def request(self, request):
                operation = request["op"]
                if operation == "observe":
                    return {"snapshot_state": {
                        "mujoco_integration_state": [0.0, 1.0, 2.0],
                        "robot_states": [{
                            "eef_pos": [0.2, 0.0, 1.0],
                            "eef_quat": [1.0, 0.0, 0.0, 0.0],
                            "joint_qpos": [0.0] * 7,
                        }],
                    }}
                if operation == "task_success_first_rrt":
                    rrt_requests.append(request)
                    return {
                        "status": "offline_feasible", "live_state_unchanged": True,
                        "planner_source": "test_rrt",
                        "require_safe_configurations": True,
                        "planner_minimum_clearance_m": 0.0,
                        "minimum_forbidden_clearance_m": 0.015,
                        "config_checks": 42,
                        "qpos_path": [[0.0] * 7, [0.1] * 7],
                        "eef_reference_poses_world": [
                            [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -0.05, 0.04, 1.0, 0.0, 0.0, 0.0],
                        ],
                    }
                if operation == "task_success_first_articulation_context":
                    return {**context, "native_goal_value": False}
                if operation == "native_task_status":
                    return native
                if operation == "export_runtime_snapshot":
                    return {"status": "complete", "payload": b"terminal-snapshot"}
                raise AssertionError(f"unexpected Bridge operation: {operation}")

        def fake_advance_pose(*args, **kwargs):
            records, frames = args[-2], args[-1]
            saved["absolute_action_limit"] = kwargs.get("absolute_action_limit")
            saved["position_tolerance_m"] = kwargs.get("position_tolerance_m")
            record = {"snapshot_state": {}, "planning_tracking": kwargs["planning_tracking"]}
            records.append(record)
            frames.append({})
            kwargs["step_callback"](record, frames[-1])
            raise TimeoutError("fixture pose budget exhausted")

        def fake_write_recording(_output_dir, records, _frames):
            saved["records"] = records

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            with patch("research.gs_opd.repair.task_success_first_grasp_search._advance_pose", fake_advance_pose), \
                 patch("research.gs_opd.repair.task_success_first_grasp_search._write_recording", fake_write_recording):
                result = _run_articulated_fixture_goal(
                    FakeBridge(), predicate=["turnoff", "flat_stove_1"],
                    target_binding={"body_id": 25}, instruction="turn off the stove",
                    policy={
                        "source": "unit_test",
                        "minimum_clearance_m": 0.0,
                        "forbidden_objects": [],
                        "allowed_target_objects": [
                            {"object_id": "flat_stove_1", "minimum_clearance_m": 0.0},
                        ],
                        "other_scene_objects": [],
                    }, articulation_context=context, controller_fields={},
                    output_dir=output_dir,
                    model_sha="test-model-sha", rrt_time_budget_s=2.0,
                )

            self.assertEqual(result["status"], "articulated_fixture_goal_incomplete")
            self.assertEqual(result["failure_reason"], "TimeoutError: fixture pose budget exhausted")
            self.assertEqual(result["recorded_actions"], 1)
            self.assertEqual(
                saved["absolute_action_limit"],
                ARTICULATED_FIXTURE_TOTAL_ACTION_BUDGET,
            )
            self.assertEqual(
                saved["position_tolerance_m"],
                ARTICULATED_FIXTURE_CONTACT_POSITION_TOLERANCE_M,
            )
            self.assertEqual(result["precontact_route"]["route_source"],
                             "MuJoCo_7D_RRT_full_geometry_EEF_path_tracked_by_native_OSC")
            self.assertEqual(len(rrt_requests), 1)
            self.assertEqual(
                rrt_requests[0]["goal_pose_world"]["position_world_m"],
                [0.0, -0.180, 0.0],
            )
            self.assertEqual(
                rrt_requests[0]["safety_geometry_policy"]["allowed_target_objects"][0]["minimum_clearance_m"],
                0.010,
            )
            self.assertIs(rrt_requests[0]["planner_config"]["require_safe_configurations"], True)
            self.assertEqual(result["precontact_route"]["config_checks"], 42)
            self.assertEqual(result["precontact_route"]["qpos_path_rad"], [[0.0] * 7, [0.1] * 7])
            self.assertEqual(saved["records"][0]["articulation_tracking"]["joint_position_after"], 0.98)
            self.assertTrue((output_dir / "articulated_fixture_terminal_runtime_snapshot.pkl").is_file())

    def test_articulated_fixture_rejects_route_below_forbidden_clearance_before_motion(self):
        from unittest.mock import patch

        context = {
            "status": "complete", "read_only": True,
            "fixture_id_from_active_BDDL": "cabinet_1",
            "contact_point_path_world_m": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]],
            "outward_direction_world": [0.0, -1.0, 0.0],
            "joint": {"joint_name": "bottom_slide", "joint_type": "slide",
                      "position": 0.0, "target_position": -0.15, "range": [-0.16, 0.01]},
        }
        native = {"status": "complete", "task_success": False,
                  "native_goal_values": [{"predicate": ["open", "cabinet_1"], "value": False}]}
        advance_calls = []

        class FakeBridge:
            def request(self, request):
                operation = request["op"]
                if operation == "observe":
                    return {"snapshot_state": {
                        "mujoco_integration_state": [0.0, 1.0, 2.0],
                        "robot_states": [{
                            "eef_pos": [0.2, 0.0, 1.0],
                            "eef_quat": [1.0, 0.0, 0.0, 0.0],
                            "joint_qpos": [0.0] * 7,
                        }],
                    }}
                if operation == "task_success_first_rrt":
                    self.planner_config = request["planner_config"]
                    return {
                        "status": "offline_feasible", "live_state_unchanged": True,
                        "planner_source": "test_rrt",
                        "require_safe_configurations": True,
                        "planner_minimum_clearance_m": 0.0,
                        "minimum_forbidden_clearance_m": 0.005,
                        "config_checks": 12,
                        "qpos_path": [[0.0] * 7, [0.1] * 7],
                        "eef_reference_poses_world": [
                            [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -0.05, 0.04, 1.0, 0.0, 0.0, 0.0],
                        ],
                    }
                if operation == "task_success_first_articulation_context":
                    return {**context, "native_goal_value": False}
                if operation == "native_task_status":
                    return native
                if operation == "export_runtime_snapshot":
                    return {"status": "complete", "payload": b"terminal-snapshot"}
                raise AssertionError(f"unexpected Bridge operation: {operation}")

        def forbidden_advance(*_args, **_kwargs):
            advance_calls.append(True)
            raise AssertionError("physical OSC motion must not run after a rejected RRT route")

        policy = {
            "source": "unit_test", "minimum_clearance_m": 0.010,
            "forbidden_objects": [{"object_id": "bowl_1", "minimum_clearance_m": 0.010}],
            "allowed_target_objects": [{"object_id": "cabinet_1", "minimum_clearance_m": 0.010}],
            "other_scene_objects": [],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            with patch("research.gs_opd.repair.task_success_first_grasp_search._advance_pose", forbidden_advance), \
                 patch("research.gs_opd.repair.task_success_first_grasp_search._write_recording"):
                result = _run_articulated_fixture_goal(
                    FakeBridge(), predicate=["open", "cabinet_1"],
                    target_binding={"body_id": 25}, instruction="open the bottom drawer",
                    policy=policy, articulation_context=context, controller_fields={},
                    output_dir=output_dir, model_sha="test-model-sha", rrt_time_budget_s=2.0,
                )

        self.assertEqual(result["status"], "articulated_fixture_goal_incomplete")
        self.assertIn("forbidden_clearance_below_policy", result["failure_reason"])
        self.assertEqual(result["recorded_actions"], 0)
        self.assertEqual(advance_calls, [])

    def test_articulated_fixture_follow_retargets_in_chunks_until_native_goal(self):
        from unittest.mock import patch

        saved = {"dynamic_tolerances": [], "dynamic_targets": [], "dynamic_gripper_actions": []}

        class FakeBridge:
            def __init__(self):
                self.joint_position = 0.98

            def context(self):
                reached = self.joint_position <= 0.5
                return {
                    "status": "complete", "read_only": True,
                    "fixture_id_from_active_BDDL": "flat_stove_1",
                    "contact_point_path_world_m": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]],
                    "outward_direction_world": [0.0, -1.0, 0.0],
                    "joint": {"joint_name": "stove_button", "joint_type": "hinge",
                              "position": self.joint_position, "target_position": 0.0,
                              "range": [-0.005, 2.1]},
                    "joint_axis_world": [0.0, 0.0, 1.0],
                    "joint_origin_world_m": [0.0, 0.0, 0.0],
                    "articulated_moving_collision_geom_names": ["flat_stove_1_g1"],
                    "native_goal_value": reached,
                }

            def request(self, request):
                operation = request["op"]
                if operation == "observe":
                    return {"snapshot_state": {
                        "mujoco_integration_state": [0.0, 1.0, 2.0],
                        "robot_states": [{
                            "eef_pos": [0.2, 0.0, 1.0],
                            "eef_quat": [1.0, 0.0, 0.0, 0.0],
                            "joint_qpos": [0.0] * 7,
                        }],
                    }}
                if operation == "task_success_first_rrt":
                    return {
                        "status": "offline_feasible", "live_state_unchanged": True,
                        "planner_source": "test_rrt",
                        "require_safe_configurations": True,
                        "planner_minimum_clearance_m": 0.0,
                        "minimum_forbidden_clearance_m": 10.0,
                        "config_checks": 24,
                        "qpos_path": [[0.0] * 7, [0.1] * 7],
                        "eef_reference_poses_world": [
                            [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, -0.05, 0.04, 1.0, 0.0, 0.0, 0.0],
                        ],
                    }
                if operation == "task_success_first_articulation_context":
                    return self.context()
                if operation == "native_task_status":
                    reached = self.joint_position <= 0.5
                    return {
                        "status": "complete", "task_success": reached,
                        "native_goal_values": [{"predicate": ["turnoff", "flat_stove_1"], "value": reached}],
                    }
                if operation == "export_runtime_snapshot":
                    return {"status": "complete", "payload": b"terminal-snapshot"}
                raise AssertionError(f"unexpected Bridge operation: {operation}")

        bridge = FakeBridge()
        initial_context = bridge.context()

        def fake_advance_pose(*args, **kwargs):
            counter = args[5]
            records, frames = args[-2], args[-1]
            phase = args[3]
            snapshot_state = {
                "robot_states": [{
                    "eef_pos": [0.2, 0.0, 1.0],
                    "eef_quat": [1.0, 0.0, 0.0, 0.0],
                    "joint_qpos": [0.0] * 7,
                }],
            }
            contacts = []
            if "live_contact_follow" not in phase:
                bridge.joint_position = 0.98
                contacts = [{
                    "geom1": {"name": "gripper0_finger2_collision"},
                    "geom2": {"name": "flat_stove_1_g1"},
                    "body1": {"name": "gripper0_finger2"},
                    "body2": {"name": "flat_stove_1_button"},
                }]
            else:
                saved["dynamic_tolerances"].append(kwargs.get("position_tolerance_m"))
                saved["dynamic_gripper_actions"].append(kwargs.get("gripper_action"))
                saved["dynamic_targets"].append(
                    kwargs["target_pose_provider"](snapshot_state)[0].tolist()
                )
                bridge.joint_position = 0.65 if len(saved["dynamic_targets"]) == 1 else 0.4
            record = {
                "snapshot_state": snapshot_state,
                "contacts": {"status": "contact" if contacts else "clear", "contacts": contacts},
                "planning_tracking": kwargs["planning_tracking"],
            }
            records.append(record)
            frames.append({})
            counter[0] += 1
            kwargs["step_callback"](record, frames[-1])
            return False

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            with patch("research.gs_opd.repair.task_success_first_grasp_search._advance_pose", fake_advance_pose), \
                 patch("research.gs_opd.repair.task_success_first_grasp_search._write_recording", lambda *_args: None):
                result = _run_articulated_fixture_goal(
                    bridge, predicate=["turnoff", "flat_stove_1"],
                    target_binding={"body_id": 25}, instruction="turn off the stove",
                    policy={
                        "source": "unit_test", "minimum_clearance_m": 0.0,
                        "forbidden_objects": [],
                        "allowed_target_objects": [{"object_id": "flat_stove_1", "minimum_clearance_m": 0.0}],
                        "other_scene_objects": [],
                    }, articulation_context=initial_context, controller_fields={},
                    output_dir=output_dir, model_sha="test-model-sha", rrt_time_budget_s=2.0,
                )

        self.assertEqual(result["status"], "complete_task_success")
        self.assertEqual(result["recorded_actions"], 3)
        self.assertEqual(saved["dynamic_tolerances"], [ARTICULATED_FIXTURE_FOLLOW_POSITION_TOLERANCE_M] * 2)
        self.assertEqual(saved["dynamic_gripper_actions"], [1.0, 1.0])
        self.assertEqual(len(result["fixture_contact_tracking"]["follow_chunks"]), 2)
        self.assertTrue(result["fixture_contact_tracking"]["switch_gripper_close_commanded"])
        self.assertEqual(
            result["fixture_contact_tracking"]["switch_bilateral_contact_observed"], False,
        )
        self.assertEqual(
            [chunk["follow_lead"] for chunk in result["fixture_contact_tracking"]["follow_chunks"]],
            [ARTICULATED_FIXTURE_HINGE_FOLLOW_LEAD_RAD,
             ARTICULATED_FIXTURE_HINGE_FOLLOW_LEAD_RAD - 0.015],
        )
        self.assertTrue(all(np.isfinite(target).all() for target in saved["dynamic_targets"]))

    def test_executor_cli_accepts_extended_demonstration_horizon_for_multigoal_episode(self):
        from unittest.mock import patch

        with patch("sys.argv", [
            "task_success_first_executor.py", "--demonstration-horizon", "5000",
        ]):
            args = _parse_task_success_executor_args()
        self.assertEqual(args.demonstration_horizon, 5000)

    def test_candidate_start_index_is_validated_by_executor_cli(self):
        from unittest.mock import patch

        with patch("sys.argv", ["task_success_first_executor.py", "--candidate-start-index", "3"]):
            args = _parse_task_success_executor_args()
        self.assertEqual(args.candidate_start_index, 3)

    def test_resume_and_exact_grasp_candidate_cli(self):
        from unittest.mock import patch

        with patch("sys.argv", [
            "task_success_first_executor.py", "--candidate-start-index", "0",
            "--resume-from-runtime-snapshot", "failed.pkl",
            "--candidate-id", "geom111_jaw1_approach2_-1_flip+1",
        ]):
            args = _parse_task_success_executor_args()
        self.assertEqual(args.resume_from_runtime_snapshot, Path("failed.pkl"))
        self.assertEqual(args.candidate_id, "geom111_jaw1_approach2_-1_flip+1")

    def test_cold_start_cli_rejects_resume_snapshots(self):
        from unittest.mock import patch

        with patch("sys.argv", [
            "task_success_first_executor.py", "--cold-start",
            "--resume-from-runtime-snapshot", "partial.pkl",
        ]), self.assertRaises(SystemExit):
            _parse_task_success_executor_args()

    def test_cold_start_continuation_cli_requires_and_accepts_its_exact_resume_input(self):
        from unittest.mock import patch

        with patch("sys.argv", [
            "task_success_first_executor.py", "--cold-start", "--cold-start-continuation",
            "--resume-from-runtime-snapshot", "same-run-stage.pkl",
            "--cold-start-continuation-root", ".",
        ]):
            args = _parse_task_success_executor_args()
        self.assertTrue(args.cold_start)
        self.assertTrue(args.cold_start_continuation)
        self.assertEqual(args.resume_from_runtime_snapshot, Path("same-run-stage.pkl"))

        with patch("sys.argv", [
            "task_success_first_executor.py", "--cold-start-continuation",
            "--resume-from-runtime-snapshot", "same-run-stage.pkl",
        ]), self.assertRaises(SystemExit):
            _parse_task_success_executor_args()

        with patch("sys.argv", [
            "task_success_first_executor.py", "--cold-start", "--cold-start-continuation",
            "--resume-from-placement-report", "same-run-source-report.json",
            "--cold-start-continuation-root", ".",
        ]):
            args = _parse_task_success_executor_args()
        self.assertEqual(args.resume_from_placement_report, Path("same-run-source-report.json"))

    def test_native_region_route_variant_cli(self):
        from unittest.mock import patch

        with patch("sys.argv", [
            "task_success_first_executor.py", "--cold-start",
            "--native-region-route-variant", "keypose_side_region_progress",
        ]):
            args = _parse_task_success_executor_args()
        self.assertEqual(args.native_region_route_variant, "keypose_side_region_progress")

    def test_bounded_grasp_search_forwards_candidate_window_and_failure_reports(self):
        from argparse import Namespace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous_reports = [root / "attempt8.json", root / "attempt9.json"]
            args = Namespace(
                benchmark="obstacle_avoidance",
                level=0,
                task_id=0,
                init_state=0,
                seed=0,
                camera_resolution=256,
                expert_port=38570,
                maximum_candidates=16,
                candidate_start_index=0,
                maximum_rrt_preflights=3,
                maximum_physical_branches=3,
                rrt_time_budget_s=12.0,
                demonstration_horizon=1000,
                resume_from_keypose_preflight_report=None,
                experience_library=root / "experience.json",
                previous_search_report=previous_reports,
                candidate_id="geom111_jaw1_approach2_-1_flip+1",
            )
            command = _task_success_first_grasp_search_command(
                args,
                snapshot_root=root / "snapshots",
                expert_root=root / "attempt10",
                predicate=["on", "akita_black_bowl_1", "plate_1"],
                resume_from_runtime_snapshot=root / "failed.pkl",
            )

        self.assertEqual(command[command.index("--candidate-start-index") + 1], "0")
        self.assertEqual(command[command.index("--resume-from-runtime-snapshot") + 1], str(root / "failed.pkl"))
        self.assertEqual(command[command.index("--candidate-id") + 1], "geom111_jaw1_approach2_-1_flip+1")
        self.assertEqual(
            [command[index + 1] for index, value in enumerate(command) if value == "--previous-search-report"],
            [str(path.resolve()) for path in previous_reports],
        )

    def test_cold_start_grasp_search_has_no_experience_or_candidate_answer(self):
        from argparse import Namespace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                benchmark="obstacle_avoidance", level=0, task_id=0,
                init_state=0, seed=0, camera_resolution=256, expert_port=38570,
                maximum_candidates=16, candidate_start_index=0,
                maximum_rrt_preflights=1, maximum_physical_branches=1,
                rrt_time_budget_s=12.0, demonstration_horizon=1000,
                resume_from_keypose_preflight_report=None,
                experience_library=None, previous_search_report=[],
                candidate_id=None, cold_start=True,
            )
            command = _task_success_first_grasp_search_command(
                args,
                snapshot_root=root / "snapshots",
                expert_root=root / "attempt1",
                predicate=["on", "akita_black_bowl_1", "plate_1"],
            )

        self.assertIn("--cold-start", command)
        self.assertNotIn("--experience-library", command)
        self.assertNotIn("--previous-search-report", command)
        self.assertNotIn("--candidate-id", command)

    def test_collision_route_policy_removes_only_extra_clearance_margin(self):
        strict = {
            "source": "strict",
            "minimum_clearance_m": 0.010,
            "measure_forbidden_object_pair_clearances": True,
            "forbidden_objects": [{"object_id": "bottle", "minimum_clearance_m": 0.010}],
            "allowed_target_objects": [{"object_id": "pot", "minimum_clearance_m": 0.010}],
            "other_scene_objects": [{"object_id": "world", "minimum_clearance_m": 0.0}],
            "allowed_contact_pairs": [{"robot_geom_id": 5, "scene_geom_id": 9, "minimum_clearance_m": -0.002}],
            "forbidden_object_pairs": [{"object_id_a": "pot", "object_id_b": "bottle"}],
        }
        route_policy = _task_success_collision_route_policy(strict)
        self.assertEqual(route_policy["minimum_clearance_m"], 0.0)
        self.assertFalse(route_policy["measure_forbidden_object_pair_clearances"])
        self.assertTrue(strict["measure_forbidden_object_pair_clearances"])
        self.assertEqual(route_policy["forbidden_objects"][0]["minimum_clearance_m"], 0.0)
        self.assertEqual(route_policy["allowed_target_objects"][0]["minimum_clearance_m"], 0.0)
        self.assertEqual(route_policy["other_scene_objects"][0]["minimum_clearance_m"], 0.0)
        self.assertEqual(route_policy["allowed_contact_pairs"], strict["allowed_contact_pairs"])
        self.assertEqual(route_policy["forbidden_object_pairs"], strict["forbidden_object_pairs"])
        self.assertEqual(strict["minimum_clearance_m"], 0.010)
        self.assertEqual(strict["forbidden_objects"][0]["minimum_clearance_m"], 0.010)

    def test_region_progress_routes_are_object_relative_and_use_keypose_only_as_a_soft_side_hint(self):
        start = np.eye(4, dtype=np.float64)
        start[:3, 3] = [0.30, -0.18, 0.94]
        goal = np.eye(4, dtype=np.float64)
        goal[:3, 3] = [0.02, 0.01, 0.91]
        region = {
            "site_position_world_m": [0.0, 0.0, 0.90],
            "site_rotation_world": np.eye(3).tolist(),
            "site_size_m": [0.16, 0.12, 0.01],
        }
        direct = _native_region_progress_route(
            start, goal, region=region, route_variant="direct_region_progress",
        )
        side = _native_region_progress_route(
            start, goal, region=region,
            route_variant="keypose_side_region_progress",
            keypose_side_region=np.asarray([0.0, 1.0]),
        )
        positive = _native_region_progress_route(
            start, goal, region=region,
            route_variant="keypose_side_region_progress",
        )
        reverse = _native_region_progress_route(
            start, goal, region=region,
            route_variant="keypose_side_region_progress_reverse",
        )
        in_region = _native_region_progress_route(
            start, goal, region=region,
            route_variant="keypose_side_region_progress",
            enable_entry_alignment=True,
        )
        self.assertTrue(direct["no_fixed_preplace_pose"])
        self.assertGreater(direct["adaptive_transit_height_world_m"], goal[2, 3])
        self.assertEqual(direct["waypoints"][0]["role"], "adaptive_clearance")
        self.assertEqual(
            side["waypoints"][0]["role"],
            "adaptive_clearance_with_keypose_side_detour",
        )
        np.testing.assert_allclose(side["waypoints"][-1]["object_pose_world"], goal)
        midpoint_xy = (start[:2, 3] + goal[:2, 3]) * 0.5
        line_xy = goal[:2, 3] - start[:2, 3]
        positive_offset = np.asarray(positive["waypoints"][0]["object_pose_world"])[:2, 3] - midpoint_xy
        reverse_offset = np.asarray(reverse["waypoints"][0]["object_pose_world"])[:2, 3] - midpoint_xy
        positive_cross = line_xy[0] * positive_offset[1] - line_xy[1] * positive_offset[0]
        reverse_cross = line_xy[0] * reverse_offset[1] - line_xy[1] * reverse_offset[0]
        self.assertGreater(positive_cross, 0.0)
        self.assertLess(reverse_cross, 0.0)
        np.testing.assert_allclose(direct["waypoints"][-1]["object_pose_world"], goal)
        self.assertEqual(in_region["waypoints"][-2]["role"], "native_region_entry_alignment")
        self.assertEqual(in_region["waypoints"][-1]["role"], "native_region_placement")
        entry_pose = np.asarray(in_region["waypoints"][-2]["object_pose_world"])
        np.testing.assert_allclose(entry_pose[:2, 3], goal[:2, 3])
        self.assertGreater(entry_pose[2, 3], goal[2, 3])

    def test_native_in_rrt_transit_poses_use_boundary_footprint_and_obstacle_height(self):
        start = np.eye(4, dtype=np.float64)
        start[:3, 3] = [0.0, -0.40, 1.0]
        goal = np.eye(4, dtype=np.float64)
        goal[:3, 3] = [0.0, 0.0, 0.90]
        region = {
            "site_position_world_m": [0.0, 0.0, 0.90],
            "site_rotation_world": np.eye(3).tolist(),
            "site_size_m": [0.20, 0.10, 0.01],
        }

        transit = _native_region_entry_transit_poses(
            start,
            goal,
            region=region,
            entry_direction_world=np.asarray([0.0, 1.0, 0.0]),
            entry_clearance_radius_m=0.05,
            obstacle_clearance_height_world_m=1.30,
        )

        outside = transit["outside_entry_object_world"]
        inside = transit["inside_alignment_object_world"]
        self.assertAlmostEqual(transit["route"]["entry_offset_m"], 0.17)
        np.testing.assert_allclose(outside[:3, 3], [0.0, 0.17, 1.30])
        np.testing.assert_allclose(inside[:3, 3], [0.0, 0.0, 1.30])

    def test_native_goal_insertion_rrt_requires_verified_aligned_full_paths(self):
        eef_path = np.zeros((2, 7), dtype=np.float64)
        qpos_path = np.zeros((2, 7), dtype=np.float64)
        result = {
            "status": "offline_feasible",
            "live_state_unchanged": True,
            "eef_reference_poses_world": eef_path.tolist(),
            "qpos_path": qpos_path.tolist(),
        }
        actual_eef, actual_qpos = _native_goal_insertion_rrt_path_arrays(result)
        np.testing.assert_array_equal(actual_eef, eef_path)
        np.testing.assert_array_equal(actual_qpos, qpos_path)

        with self.assertRaisesRegex(RuntimeError, "not_feasible_before_OSC"):
            _native_goal_insertion_rrt_path_arrays({
                **result, "live_state_unchanged": False,
            })
        with self.assertRaisesRegex(RuntimeError, "path_alignment_invalid"):
            _native_goal_insertion_rrt_path_arrays({
                **result, "qpos_path": np.zeros((1, 7)).tolist(),
            })
        invalid_eef = eef_path.copy()
        invalid_eef[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "path_alignment_invalid"):
            _native_goal_insertion_rrt_path_arrays({
                **result, "eef_reference_poses_world": invalid_eef.tolist(),
            })

    def test_final_native_region_alignment_uses_tight_tracking_tolerances(self):
        self.assertEqual(
            _native_region_progress_tracking_tolerances(
                "region_progress", 0.095, native_goal_operator="in"
            ),
            (0.006, 0.05),
        )
        self.assertEqual(
            _native_region_progress_tracking_tolerances(
                "region_progress", 0.12, native_goal_operator="in"
            ),
            (0.015, 0.12),
        )
        self.assertEqual(
            _native_region_progress_tracking_tolerances(
                "native_region_placement", 0.02, native_goal_operator="in"
            ),
            (0.015, 0.12),
        )
        self.assertEqual(
            _native_region_progress_tracking_tolerances(
                "region_progress", 0.02, native_goal_operator="on"
            ),
            (0.015, 0.12),
        )

    def test_joint_limit_diagnostic_reports_the_measured_small_overage(self):
        ranges = np.asarray([[-2.9, 2.9]] * 6 + [[-2.8973, 2.8973]], dtype=np.float64)
        qpos = np.asarray([0.0] * 6 + [2.897506367677], dtype=np.float64)
        result = _joint_limit_diagnostics(qpos, [f"robot0_joint{i + 1}" for i in range(7)], ranges)
        self.assertFalse(result["within_limits"])
        self.assertAlmostEqual(result["maximum_excess_rad"], 0.000206367677, places=11)
        self.assertEqual(result["violating_joint_count"], 1)
        self.assertEqual(result["joints"][-1]["joint_name"], "robot0_joint7")

    def test_joint_limit_specification_accepts_bridge_available_status(self):
        names, ranges = _live_joint_limit_specification({
            "status": "available",
            "robots": [{
                "status": "available",
                "joint_names": ["robot0_joint1", "robot0_joint7"],
                "joint_ranges": [[-2.9, 2.9], [-2.8973, 2.8973]],
            }],
        })
        self.assertEqual(names, ["robot0_joint1", "robot0_joint7"])
        self.assertEqual(np.asarray(ranges).shape, (2, 2))

    def test_transport_loss_stops_before_false_grasp_can_be_misread_as_release(self):
        loss = {"phase": "transport_preplace", "grasp_state": {"grasp_confirmed": False}}
        release = {"phase": "release", "grasp_state": {"grasp_confirmed": False}}
        self.assertEqual(
            _transport_grasp_loss_reason(loss),
            "bilateral_grasp_lost_during_native_goal_transport",
        )
        self.assertIsNone(_transport_grasp_loss_reason(release))

    def test_terminal_native_success_stops_before_withdrawal_or_settling_actions(self):
        self.assertTrue(_native_episode_terminated({"done": True, "task_success": True}))
        self.assertTrue(_native_episode_terminated({"done": True, "task_success": False}))
        self.assertFalse(_native_episode_terminated({"done": False, "task_success": True}))
        self.assertFalse(_native_episode_terminated({"task_success": True}))
        self.assertFalse(_native_episode_terminated(None))

    def test_transport_budget_scales_past_legacy_360_cap_and_reserves_next_object(self):
        route = np.asarray([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        native = {
            "status": "complete",
            "timestep": 500,
            "native_goal_values": [
                {"predicate": ["on", "moka_pot_1", "stove"], "value": False},
                {"predicate": ["on", "moka_pot_2", "stove"], "value": False},
            ],
        }
        result = _native_transport_action_budget(
            route, native, predicate=["on", "moka_pot_1", "stove"],
            demonstration_horizon=3000,
        )
        self.assertGreater(result["actions"], 360)
        self.assertEqual(result["future_pending_object_goal_count"], 1)
        self.assertEqual(
            result["available_after_reserves"],
            3000 - 500 - result["current_goal_completion_reserve"]
            - result["future_goal_reserve_each"],
        )
        self.assertLessEqual(result["actions"], result["available_after_reserves"])

    def test_single_terminal_region_goal_reserves_only_post_route_placement(self):
        route = np.asarray([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        native = {
            "status": "complete",
            "timestep": 246,
            "native_goal_values": [
                {"predicate": ["in", "book", "caddy"], "value": False},
            ],
        }
        result = _native_transport_action_budget(
            route, native, predicate=["in", "book", "caddy"],
            demonstration_horizon=1000,
            native_region_route_includes_final_placement=True,
        )
        self.assertEqual(result["current_goal_completion_reserve"], MAX_NATIVE_PLACEMENT_ACTIONS)
        self.assertEqual(
            result["available_after_reserves"],
            1000 - 246 - MAX_NATIVE_PLACEMENT_ACTIONS,
        )
        self.assertTrue(result["native_region_route_includes_final_placement"])

    def test_task_success_search_can_physically_try_valid_ik_after_rrt_timeout(self):
        row = {
            "admitted_to_physical_search": False,
            "preplace_RRT": {
                "status": "no_safe_path",
                "failure_reason": "planning_time_budget_exhausted",
                "ik_candidate_count": 3,
            },
            "terminal_pose_IK_RRT_diagnostic": {
                "status": "not_run_after_unreachable_preplace_route",
                "ik_candidate_count": 0,
            },
        }
        self.assertIsNotNone(_task_success_waypoint_fallback_reason(row["preplace_RRT"]))
        self.assertTrue(_preflight_row_is_task_success_admissible(row))
        blocked_ik = {
            **row,
            "preplace_RRT": {**row["preplace_RRT"], "ik_candidate_count": 0},
        }
        self.assertIsNone(_task_success_waypoint_fallback_reason(blocked_ik["preplace_RRT"]))
        self.assertFalse(_preflight_row_is_task_success_admissible(blocked_ik))
        collision_rejection = {
            **row,
            "preplace_RRT": {
                "status": "no_safe_path",
                "failure_reason": "IK_solution_collides_with_hard_joint_limit",
                "ik_candidate_count": 3,
            },
        }
        self.assertFalse(_preflight_row_is_task_success_admissible(collision_rejection))

    def test_precomputed_rrt_route_requires_exact_planning_identity_and_intact_paths(self):
        goal = {
            "position_world_m": [0.1, -0.2, 0.3],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        policy = {"carried_object": {"object_id": "pot_2"}, "forbidden": [5, 9]}
        config = {"seed": 83, "search_time_budget_s": 12.0, "require_safe_configurations": False}
        result = {
            "status": "offline_feasible",
            "qpos_path": np.arange(21, dtype=np.float64).reshape(3, 7).tolist(),
            "eef_reference_poses_world": [
                [0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                [0.11, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                [0.12, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            ],
            "planning_seconds": 1.25,
            "config_checks": 17,
            "iterations": 8,
        }
        cache = _make_precomputed_rrt_route_cache(
            result=result, model_sha256="model", bddl_sha256="bddl",
            start_state_sha256="start", goal_pose_world=goal,
            safety_geometry_policy=policy, planner_config=config,
        )
        self.assertIsNotNone(cache)
        accepted = _validate_precomputed_rrt_route(
            cache, model_sha256="model", bddl_sha256="bddl",
            start_state_sha256="start", goal_pose_world=goal,
            safety_geometry_policy=policy, planner_config=config,
        )
        self.assertTrue(accepted["reusable"])
        self.assertEqual(accepted["qpos_path"].shape, (3, 7))
        self.assertEqual(accepted["eef_reference_poses_world"].shape, (3, 7))

        mismatches = (
            {"model_sha256": "other"}, {"bddl_sha256": "other"},
            {"start_state_sha256": "other"},
            {"goal_pose_world": {**goal, "position_world_m": [0.100001, -0.2, 0.3]}},
            {"safety_geometry_policy": {**policy, "forbidden": [5, 10]}},
            {"planner_config": {**config, "seed": 84}},
        )
        for change in mismatches:
            kwargs = {
                "model_sha256": "model", "bddl_sha256": "bddl",
                "start_state_sha256": "start", "goal_pose_world": goal,
                "safety_geometry_policy": policy, "planner_config": config,
            }
            kwargs.update(change)
            with self.subTest(change=next(iter(change))):
                rejected = _validate_precomputed_rrt_route(cache, **kwargs)
                self.assertFalse(rejected["reusable"])
                self.assertIn("mismatch", rejected["reason"])

        damaged = json.loads(json.dumps(cache))
        damaged["paths"]["qpos_path_rad"][1][0] += 0.01
        rejected = _validate_precomputed_rrt_route(
            damaged, model_sha256="model", bddl_sha256="bddl",
            start_state_sha256="start", goal_pose_world=goal,
            safety_geometry_policy=policy, planner_config=config,
        )
        self.assertEqual(rejected["reason"], "route_cache_path_integrity_mismatch")

    def test_precomputed_rrt_route_rejects_unaligned_or_malformed_paths(self):
        with self.assertRaisesRegex(ValueError, "unaligned_joint_and_EEF_paths"):
            _make_precomputed_rrt_route_cache(
                result={
                    "status": "offline_feasible",
                    "qpos_path": [[0.0] * 7, [1.0] * 7],
                    "eef_reference_poses_world": [[0.0] * 7],
                },
                model_sha256="model", bddl_sha256="bddl",
                start_state_sha256="start",
                goal_pose_world={"position_world_m": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
                safety_geometry_policy={}, planner_config={},
            )

    def test_eef_route_compression_preserves_source_index_mapping(self):
        poses = np.zeros((5, 7), dtype=np.float64)
        poses[:, 0] = [0.0, 0.004, 0.021, 0.024, 0.050]
        poses[:, 3] = 1.0
        compressed, source_indices = _compress_eef_route_with_indices(poses)
        self.assertEqual(source_indices, [0, 2, 4])
        np.testing.assert_allclose(compressed, poses[source_indices])

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
                        "snapshot_state": {"robot_states": [{"joint_qpos": [0.1] * 7}]},
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
        _record_step(
            *args,
            raise_on_done=False,
            planning_tracking={
                "rrt_reference_qpos_rad": [0.0] * 7,
                "osc_command_target_eef_pose_world": [0.0] * 7,
            },
        )
        self.assertEqual(len(args[-2]), 1)
        self.assertTrue(args[-2][0]["done"])
        self.assertTrue(args[-2][0]["task_success"])
        self.assertEqual(args[-2][0]["planning_tracking"]["rrt_reference_joint_error_rad"], np.sqrt(0.07))
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

    def test_native_in_candidates_align_compiled_footprint_and_keep_root_inside_inverted_site(self):
        from scipy.spatial.transform import Rotation

        region_rotation = np.diag([1.0, -1.0, -1.0])
        target_rotation = Rotation.from_euler("z", 60.0, degrees=True).as_matrix()
        target = {
            "target_object_id": "book",
            "target_root_pose_world": {
                "position_m": [0.1, 0.0, 0.9],
                "rotation": target_rotation.tolist(),
            },
            "target_collision_geometries": [{
                "center_root_m": [-0.00047, -0.01215, 0.06937],
                "rotation_root": np.diag([-1.0, -1.0, 1.0]).tolist(),
                "size_m": [0.01434, 0.05519, 0.0672],
            }],
        }
        region = {
            "site_position_world_m": [-0.4, -0.1, 0.97],
            "site_rotation_world": region_rotation.tolist(),
            "site_size_m": [0.02775, 0.06216, 0.06046],
        }
        candidates = _generate_native_region_placement_candidates(
            predicate=["in", "book", "caddy_region"],
            region=region,
            target_context=target,
            occupied_contexts=[],
            maximum_candidates=2,
        )
        self.assertEqual(len(candidates), 2)
        candidate = candidates[0]
        self.assertEqual(candidate["orientation_source"], "region_collision_envelope_fit_yaw_search")
        np.testing.assert_allclose(candidate["rotation_world"], np.eye(3), atol=1e-7)
        np.testing.assert_allclose(candidate["footprint_half_extents_region_m"], [0.01434, 0.05519], atol=1e-7)
        self.assertLess(candidate["estimated_region_fit_cost"], 1.2)
        local_root = region_rotation.T @ (
            np.asarray(candidate["position_world_m"]) - np.asarray(region["site_position_world_m"])
        )
        self.assertLess(abs(local_root[0]), region["site_size_m"][0])
        self.assertLess(abs(local_root[1]), region["site_size_m"][1])
        self.assertLess(abs(local_root[2]), region["site_size_m"][2])

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

    def test_object_support_region_ranks_centered_placement_first(self):
        target = self._placement_context("object_b", [0.11, 0.02, 1.14])
        candidates = _generate_native_region_placement_candidates(
            predicate=["on", "object_b", "plate"],
            region={
                "site_position_world_m": [0.0, 0.0, 0.92],
                "site_rotation_world": np.eye(3).tolist(),
                "site_size_m": [0.07, 0.07, 0.0],
                "support_geometry_kind": "compiled_object_collision_envelope",
            },
            target_context=target,
            occupied_contexts=[],
            maximum_candidates=4,
        )
        self.assertEqual(len(candidates), 4)
        np.testing.assert_allclose(candidates[0]["region_local_xy_m"], [0.0, 0.0])
        self.assertEqual(candidates[0]["support_centering_penalty_m"], 0.0)

    def test_native_region_route_raises_above_measured_corridor_obstacles(self):
        start = np.eye(4, dtype=np.float64)
        goal = np.eye(4, dtype=np.float64)
        goal[0, 3] = 1.0
        region = {
            "site_position_world_m": [0.0, 0.0, 0.0],
            "site_rotation_world": np.eye(3).tolist(),
            "site_size_m": [0.1, 0.1, 0.0],
        }
        obstacle = {
            "object_id": "bottle",
            "mapping_name": "objects_dict",
            "root_position_world_m": [0.0, 0.0, 0.0],
            "root_rotation_world": np.eye(3).tolist(),
            "envelope_lower_root_m": [0.45, -0.05, 0.0],
            "envelope_upper_root_m": [0.55, 0.05, 0.5],
            "collision_geom_count": 1,
        }
        scene = {
            "status": "complete",
            "model_geometry_sha256": "model",
            "objects": [obstacle],
            "unavailable_objects": [],
        }
        clearance = _native_region_corridor_obstacle_clearance(
            start_object_world=start,
            goal_object_world=goal,
            region=region,
            target_context=self._placement_context("bowl", [0.0, 0.0, 0.0]),
            object_to_eef=np.eye(4),
            scene_envelopes=scene,
            excluded_object_ids={"bowl", "plate"},
            expected_model_sha256="model",
        )
        self.assertEqual([row["object_id"] for row in clearance["intersecting_objects"]], ["bottle"])
        self.assertAlmostEqual(clearance["maximum_intersecting_obstacle_top_world_m"], 0.5)
        self.assertAlmostEqual(clearance["required_transit_height_world_m"], 0.6)
        reduced_route_margin = _native_region_corridor_obstacle_clearance(
            start_object_world=start,
            goal_object_world=goal,
            region=region,
            target_context=self._placement_context("bowl", [0.0, 0.0, 0.0]),
            object_to_eef=np.eye(4),
            scene_envelopes=scene,
            excluded_object_ids={"bowl", "plate"},
            expected_model_sha256="model",
            clearance_margin_m=0.02,
        )
        self.assertAlmostEqual(
            clearance["required_transit_height_world_m"]
            - reduced_route_margin["required_transit_height_world_m"],
            0.03,
        )
        route = _native_region_progress_route(
            start,
            goal,
            region=region,
            route_variant="direct_region_progress",
            obstacle_clearance_height_world_m=clearance["required_transit_height_world_m"],
        )
        self.assertGreaterEqual(route["adaptive_transit_height_world_m"], 0.6)

    def test_native_region_route_uses_world_up_when_region_z_axis_is_inverted(self):
        start = np.eye(4, dtype=np.float64)
        start[:3, 3] = [-0.0104, 0.0008, 1.1244]
        goal = np.eye(4, dtype=np.float64)
        goal[:3, 3] = [-0.4036, -0.1181, 0.7729]
        region = {
            "site_position_world_m": [-0.4253, -0.1424, 0.9726],
            "site_rotation_world": np.diag([1.0, -1.0, -1.0]).tolist(),
            "site_size_m": [0.16, 0.12, 0.01],
        }
        route = _native_region_progress_route(
            start,
            goal,
            region=region,
            route_variant="keypose_side_region_progress",
            keypose_side_region=[0.0, 1.0],
            obstacle_clearance_height_world_m=1.4438,
        )
        self.assertGreaterEqual(route["adaptive_transit_height_world_m"], 1.4438)
        self.assertEqual(route["waypoints"][0]["role"], "adaptive_clearance")
        first_lift = np.asarray(route["waypoints"][0]["object_pose_world"])
        np.testing.assert_allclose(first_lift[:2, 3], start[:2, 3])
        self.assertEqual(route["waypoints"][1]["role"], "keypose_side_detour")
        for waypoint in route["waypoints"][:-1]:
            pose = np.asarray(waypoint["object_pose_world"], dtype=np.float64)
            self.assertAlmostEqual(pose[2, 3], route["adaptive_transit_height_world_m"], places=8)
        np.testing.assert_allclose(route["waypoints"][-1]["object_pose_world"], goal)

    def test_corridor_clearance_uses_world_up_with_inverted_native_region_frame(self):
        start = np.eye(4, dtype=np.float64)
        start[:3, 3] = [0.0, 0.0, 0.8]
        goal = np.eye(4, dtype=np.float64)
        goal[:3, 3] = [1.0, 0.0, 0.8]
        region = {
            "site_position_world_m": [0.0, 0.0, 0.9],
            "site_rotation_world": np.diag([1.0, -1.0, -1.0]).tolist(),
            "site_size_m": [0.1, 0.1, 0.1],
        }
        target = {
            "target_object_id": "book",
            "target_root_pose_world": {
                "position_m": [0.0, 0.0, 0.8],
                "rotation": np.eye(3).tolist(),
            },
            "target_collision_geometries": [{
                "center_root_m": [0.0, 0.0, 0.07],
                "rotation_root": np.eye(3).tolist(),
                "size_m": [0.01, 0.01, 0.02],
            }],
        }
        scene = {
            "status": "complete",
            "model_geometry_sha256": "model",
            "objects": [{
                "object_id": "bottle",
                "mapping_name": "objects_dict",
                "root_position_world_m": [0.5, 0.0, 0.0],
                "root_rotation_world": np.eye(3).tolist(),
                "envelope_lower_root_m": [-0.05, -0.05, 0.0],
                "envelope_upper_root_m": [0.05, 0.05, 0.5],
                "collision_geom_count": 1,
            }],
        }

        clearance = _native_region_corridor_obstacle_clearance(
            start_object_world=start,
            goal_object_world=goal,
            region=region,
            target_context=target,
            object_to_eef=np.eye(4),
            scene_envelopes=scene,
            excluded_object_ids={"book"},
            expected_model_sha256="model",
        )

        self.assertAlmostEqual(clearance["maximum_intersecting_obstacle_top_world_m"], 0.5)
        self.assertAlmostEqual(clearance["required_transit_height_world_m"], 0.5)

    def test_native_in_route_uses_geometric_lateral_entry_before_final_pose(self):
        start = np.eye(4, dtype=np.float64)
        start[:3, 3] = [0.0, -0.30, 1.10]
        goal = np.eye(4, dtype=np.float64)
        goal[:3, 3] = [0.01, 0.02, 0.90]
        region = {
            "site_position_world_m": [0.0, 0.0, 0.90],
            "site_rotation_world": np.eye(3).tolist(),
            "site_size_m": [0.10, 0.10, 0.06],
        }
        route = _native_region_progress_route(
            start,
            goal,
            region=region,
            route_variant="direct_region_progress",
            entry_direction_world=[0.0, -1.0, 0.0],
            entry_clearance_radius_m=0.04,
        )
        roles = [row["role"] for row in route["waypoints"]]
        self.assertIn("native_region_entry_approach", roles)
        self.assertEqual(roles[-2], "native_region_entry_alignment")
        self.assertEqual(roles[-1], "native_region_placement")
        outside_approach = np.asarray(route["waypoints"][-3]["object_pose_world"])
        inside_alignment = np.asarray(route["waypoints"][-2]["object_pose_world"])
        self.assertLess(outside_approach[1, 3], -0.10)
        self.assertAlmostEqual(outside_approach[2, 3], route["adaptive_transit_height_world_m"], places=8)
        np.testing.assert_allclose(inside_alignment[:2, 3], goal[:2, 3])
        self.assertAlmostEqual(
            inside_alignment[2, 3], route["adaptive_transit_height_world_m"], places=8,
        )
        np.testing.assert_allclose(route["waypoints"][-1]["object_pose_world"], goal)

    def test_native_region_progress_tracks_the_active_side_waypoint(self):
        region = {
            "site_position_world_m": [0.0, 0.0, 0.0],
            "site_rotation_world": np.eye(3).tolist(),
            "site_size_m": [0.2, 0.2, 0.1],
        }
        current = np.eye(4, dtype=np.float64)
        current[:3, 3] = [0.0, 0.0, 1.0]
        waypoint = np.eye(4, dtype=np.float64)
        waypoint[:3, 3] = [0.20, 0.10, 1.0]
        target, distance = _native_region_progress_waypoint_target(
            region, current, waypoint, 1.0,
        )
        self.assertAlmostEqual(distance, np.hypot(0.20, 0.10), places=8)
        np.testing.assert_allclose(target[:3, 3], [0.20 * 0.045 / distance, 0.10 * 0.045 / distance, 1.0])
        self.assertIsNone(_native_region_progress_waypoint_target(
            region, waypoint, waypoint, 1.0,
        )[0])

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

    def test_placement_waypoint_stagnation_uses_held_object_progress(self):
        stagnant_eef = [(0.10, 0.2), (0.099, 0.199), (0.101, 0.201), (0.100, 0.200)]
        improving_object = [(0.08, 0.1), (0.06, 0.09), (0.04, 0.08), (0.02, 0.07)]
        stagnant_object = [(0.08, 0.1), (0.079, 0.099), (0.081, 0.101), (0.080, 0.100)]
        self.assertFalse(_pose_tracking_window_stagnated(
            stagnant_eef, window_actions=3, object_progress_history=improving_object,
        ))
        self.assertTrue(_pose_tracking_window_stagnated(
            stagnant_eef, window_actions=3, object_progress_history=stagnant_object,
        ))

    def test_placement_waypoint_stagnation_allows_small_object_progress(self):
        stagnant_eef = [(0.10, 0.2), (0.099, 0.199), (0.101, 0.201), (0.100, 0.200)]
        small_object_progress = [(0.050, 0.08), (0.049, 0.07), (0.0485, 0.06), (0.0475, 0.055)]
        self.assertTrue(_pose_tracking_window_stagnated(
            stagnant_eef, window_actions=3, object_progress_history=small_object_progress,
        ))
        self.assertFalse(_pose_tracking_window_stagnated(
            stagnant_eef,
            window_actions=3,
            object_progress_history=small_object_progress,
            object_minimum_position_progress_m=0.002,
            object_minimum_rotation_progress_rad=0.02,
        ))
        still_stagnant_object = [(0.050, 0.08), (0.0498, 0.078), (0.0495, 0.076), (0.049, 0.075)]
        self.assertTrue(_pose_tracking_window_stagnated(
            stagnant_eef,
            window_actions=3,
            object_progress_history=still_stagnant_object,
            object_minimum_position_progress_m=0.002,
            object_minimum_rotation_progress_rad=0.02,
        ))

    def test_native_object_relative_eef_target_tracks_live_grasp_transform(self):
        desired_object = np.eye(4, dtype=np.float64)
        desired_object[:3, 3] = [0.5, -0.2, 0.8]
        live_object = np.eye(4, dtype=np.float64)
        live_object[:3, 3] = [0.1, 0.2, 0.3]
        live_object[:3, :3] = [
            [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        ]
        live_eef = live_object.copy()
        live_eef[:3, 3] += [0.03, 0.0, 0.02]
        target = _native_object_relative_eef_target_world(
            desired_object, live_object, live_eef,
        )
        np.testing.assert_allclose(
            np.linalg.inv(desired_object) @ target,
            np.linalg.inv(live_object) @ live_eef,
            atol=1e-10,
        )

    def test_live_contact_close_requires_candidate_pose_alignment(self):
        current = np.eye(4, dtype=np.float64)
        aligned_target = np.eye(4, dtype=np.float64)
        aligned_target[:3, 3] = [0.003, 0.0, 0.0]
        angle = 0.03
        aligned_target[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        aligned, position_error, rotation_error = _live_contact_pose_is_close_to_candidate(
            current, aligned_target,
        )
        self.assertTrue(aligned)
        self.assertAlmostEqual(position_error, 0.003)
        self.assertAlmostEqual(rotation_error, angle)

        incidental_contact_pose = np.eye(4, dtype=np.float64)
        incidental_contact_pose[:3, 3] = [0.1032, 0.0, 0.0]
        incidental_contact_pose[:3, :3] = [
            [np.cos(1.86), -np.sin(1.86), 0.0],
            [np.sin(1.86), np.cos(1.86), 0.0],
            [0.0, 0.0, 1.0],
        ]
        aligned, position_error, rotation_error = _live_contact_pose_is_close_to_candidate(
            current, incidental_contact_pose,
        )
        self.assertFalse(aligned)
        self.assertAlmostEqual(position_error, 0.1032)
        self.assertAlmostEqual(rotation_error, 1.86)

    def test_object_relative_target_pose_tracks_live_object_state(self):
        relative = np.eye(4, dtype=np.float64)
        relative[:3, 3] = [0.02, 0.0, 0.0]
        first_state = {
            "object_states": {
                "target": {
                    "body_id": 23,
                    "root_pos": [0.0, 0.0, 0.0],
                    "root_quat": [1.0, 0.0, 0.0, 0.0],
                },
            },
        }
        second_state = {
            "object_states": {
                "target": {
                    "body_id": 23,
                    "root_pos": [0.01, 0.0, 0.0],
                    "root_quat": [np.cos(0.05), 0.0, 0.0, np.sin(0.05)],
                },
            },
        }
        first_position, _ = _object_relative_target_pose(first_state, 23, relative)
        second_position, _ = _object_relative_target_pose(second_state, 23, relative)
        np.testing.assert_allclose(first_position, [0.02, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(
            second_position,
            [0.01 + 0.02 * np.cos(0.1), 0.02 * np.sin(0.1), 0.0],
            atol=1e-9,
        )

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

    def test_target_selection_handles_fixture_and_switch(self):
        native = {"status": "complete", "native_goal_values": [
            {"predicate": ["on", "pot", "stove"], "value": True},
            {"predicate": ["close", "door"], "value": False},
            {"predicate": ["turnon", "stove"], "value": False},
        ]}
        selected, source, _ = _select_active_target_predicate(native, None)
        self.assertEqual(selected, ["close", "door"])
        self.assertEqual(source, "first_pending_supported_fixture_goal_bddl_order")
        native["native_goal_values"][1]["value"] = True
        selected, source, _ = _select_active_target_predicate(native, None)
        self.assertEqual(selected, ["turnon", "stove"])
        selected, source, _ = _select_active_target_predicate(native, ["turnon", "stove"])
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
