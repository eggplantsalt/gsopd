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
"""Unit tests for fail-closed trajectory tracking verification (no MuJoCo)."""

from __future__ import annotations

import unittest

import numpy as np

from research.gs_opd.repair.trajectory_tracking import (
    TrackingConfig,
    body_local_offset_from_world_point,
    body_local_point_to_world,
    check_workspace_target,
    inverse_osc_world_delta,
    plan_bundle_reference_paths,
    quaternion_to_matrix,
    resample_polyline_by_distance,
    track_trajectory_bundle,
    world_pose_tracking_action,
)

STATE_ID = "decision-state-test"
FINGERPRINT = "a" * 64


def _teacher():
    def object_item(object_id, name, role):
        return {
            "id": object_id,
            "name": name,
            "role": role,
            "visibility": "visible",
            "mobility": "likely_movable",
            "visual_facts": [],
            "inferences": [],
            "uncertainties": [],
        }

    return {
        "schema_version": "gs_opd.safety_teacher.v2",
        "objects": [
            object_item("carry", "carried object", "target_object"),
            object_item("obstacle", "obstacle", "obstacle"),
        ],
        "rules": [
            {
                "id": "allow-grasp",
                "subject_parts": ["gripper"],
                "target_id": "carry",
                "predicate": "contact",
                "decision": "allow",
                "phase": "any",
                "reason": "The object must be grasped for transport.",
            }
        ],
        "object_contact_rules": [
            {
                "id": "avoid-obstacle",
                "subject_id": "carry",
                "target_id": "obstacle",
                "predicate": "contact",
                "decision": "forbid",
                "phase": "any",
                "reason": "Avoid the obstacle while carrying the object.",
            }
        ],
        "global_uncertainties": [],
    }


def _semantic_result(object_id, body_id, body_name, geom_id, geom_name):
    return {
        "teacher_object": {"object_id": object_id},
        "semantic_decision": {
            "object_id": object_id,
            "status": "resolved",
            "candidate_index": 0,
        },
        "selected_candidate": {
            "candidate_index": 0,
            "body_id": body_id,
            "geom_ids": [geom_id],
        },
        "physical_association": {
            "status": "matched_by_iou",
            "physical_object_id": f"physical-{body_id}",
            "body_id": body_id,
            "body_name": body_name,
            "geom_ids": [geom_id],
            "geom_names": [geom_name],
        },
    }


def _fixtures(candidate_count=1):
    semantic_results = {
        "carry": _semantic_result("carry", 10, "carry_body", 0, "carry_geom"),
        "obstacle": _semantic_result(
            "obstacle", 20, "obstacle_body", 1, "obstacle_geom"
        ),
    }
    geometry_inventory = {
        "status": "complete",
        "robot_classification_status": "available",
        "geom_count": 3,
        "model_body_count": 30,
        "robot_body_ids": [1],
        "geometries": [
            {
                "geom_id": 0,
                "geom_name": "carry_geom",
                "body_id": 10,
                "body_name": "carry_body",
                "category": "scene",
            },
            {
                "geom_id": 1,
                "geom_name": "obstacle_geom",
                "body_id": 20,
                "body_name": "obstacle_body",
                "category": "scene",
            },
            {
                "geom_id": 2,
                "geom_name": "finger_geom",
                "body_id": 1,
                "body_name": "robot_body",
                "category": "robot",
            },
        ],
    }
    gripper_inventory = {
        "status": "complete",
        "robot_classification_status": "available",
        "robot_body_ids": [1],
        "model_body_count": 30,
        "grippers": [
            {
                "status": "complete",
                "geometries": [
                    {"geom_id": 2, "geom_name": "finger_geom", "body_id": 1}
                ],
            }
        ],
    }
    carried_binding = {
        "object_id": "carry",
        "body_id": 10,
        "body_name": "carry_body",
        "body_ids": [10],
        "geom_ids": [0],
        "geom_names": ["carry_geom"],
        "semantic_binding_geom_ids": [0],
        "center_world_m": [0.0, 0.0, 0.0],
        "state_id": STATE_ID,
        "binding_verified": True,
        "covers_all_associated_collision_geoms": True,
    }
    bundle = {
        "status": "complete",
        "scene": {
            "frame": "world",
            "units": "m",
            "state_id": STATE_ID,
            "carried": {
                "object_id": "carry",
                "center_world_m": [0.0, 0.0, 0.0],
                "radius_m": 0.04,
                "state_id": STATE_ID,
            },
        },
        "candidates": [
            {
                "candidate_id": f"candidate-{index}",
                "passes_geometric_screen": True,
                "executable": False,
                "physical_safety_verified": False,
                "goal_world_m": [0.01, 0.0, 0.0],
                "points_world_m": [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
            }
            for index in range(candidate_count)
        ],
    }
    repair_context = {
        "status": "available",
        "robots": [
            {
                "status": "available",
                "controller_name": "OSC_POSE",
                "action_dim": 7,
                "fields": {
                    "input_min": [-1.0] * 6,
                    "input_max": [1.0] * 6,
                    "output_min": [-0.05] * 3 + [-0.5] * 3,
                    "output_max": [0.05] * 3 + [0.5] * 3,
                    "input_ref_frame": "world",
                },
            }
        ],
    }
    return (
        bundle,
        carried_binding,
        repair_context,
        _teacher(),
        semantic_results,
        geometry_inventory,
        gripper_inventory,
    )


def _event(kind):
    if kind == "grasp":
        geoms = [{"id": 0, "name": "carry_geom"}, {"id": 2, "name": "finger_geom"}]
        bodies = [{"id": 10, "name": "carry_body"}, {"id": 1, "name": "robot_body"}]
        classification, robot_involved = "robot_scene", True
    elif kind == "gripper-obstacle":
        geoms = [
            {"id": 1, "name": "obstacle_geom"},
            {"id": 2, "name": "finger_geom"},
        ]
        bodies = [
            {"id": 20, "name": "obstacle_body"},
            {"id": 1, "name": "robot_body"},
        ]
        classification, robot_involved = "robot_scene", True
    else:
        geoms = [{"id": 0, "name": "carry_geom"}, {"id": 1, "name": "obstacle_geom"}]
        bodies = [{"id": 10, "name": "carry_body"}, {"id": 20, "name": "obstacle_body"}]
        classification, robot_involved = "scene_scene", False
    return {
        "geom_pair": geoms,
        "body_pair": bodies,
        "classification": classification,
        "robot_involved": robot_involved,
        "first_control_step": 0,
        "first_physics_substep": 0,
        "occurrence_count": 1,
        "contact_point_count": 1,
        "max_force_norm": 1.0,
        "force_sample_count": 1,
        "force_unavailable_count": 0,
    }


def _bridge_candidate(
    candidate_id,
    *,
    grasp=True,
    tracking=True,
    collision=False,
    fingerprint=FINGERPRINT,
    sphere_offset=(0.0, 0.0, 0.0),
    object_quaternion=(1.0, 0.0, 0.0, 0.0),
    step_quaternions=None,
):
    sphere_offset = np.asarray(sphere_offset, dtype=np.float64)
    object_quaternion = np.asarray(object_quaternion, dtype=np.float64)

    def root_from_sphere(center, quaternion):
        rotation = quaternion_to_matrix(quaternion)
        return (
            np.asarray(center, dtype=np.float64) - rotation @ sphere_offset
        ).tolist()

    events = [_event("grasp")] if grasp else []
    if collision:
        events.append(_event("collision"))
    contact = bool(events)
    steps = []
    grasp_steps = []
    control_steps = []
    control_step_count = 5
    for index in range(control_step_count):
        sphere_center = [0.01 * (index + 1) / control_step_count, 0.0, 0.0]
        step_quaternion = (
            np.asarray(
                step_quaternions[min(index, len(step_quaternions) - 1)],
                dtype=np.float64,
            )
            if step_quaternions is not None
            else object_quaternion
        )
        steps.append(
            {
                "step_index": index,
                "reference_sphere_center_world_m": sphere_center,
                "actual_object_root_pos_world_m": root_from_sphere(
                    sphere_center, step_quaternion
                ),
                "actual_object_sphere_center_world_m": sphere_center,
                "sphere_center_body_local_offset_m": sphere_offset.tolist(),
                "actual_object_root_quat_world_wxyz": step_quaternion.tolist(),
                "actual_eef_position_world_m": [0.0, 0.0, 0.0],
                "actual_eef_quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
                "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "requested_osc_output_world_delta": [0.0] * 6,
                "controller_scaled_osc_output": [0.0] * 6,
                "reward": 0.0,
                "done": False,
                "state": {
                    "mujoco_integration_state": [0.0],
                    "robot_states": [
                        {"eef_pos": [0.0, 0.0, 0.0], "eef_quat": [1.0, 0.0, 0.0, 0.0]}
                    ],
                    "object_states": {
                        "carried": {
                            "body_id": 10,
                            "root_pos": root_from_sphere(
                                sphere_center, step_quaternion
                            ),
                            "root_quat": step_quaternion.tolist(),
                        }
                    },
                },
                "contacts": {
                    "status": "contact" if contact and index == 0 else "no_contact",
                    "contacts": [],
                },
                "tracking_status": "pass" if tracking else "fail",
                "sphere_center_tracking_error_m": 0.0 if tracking else 0.1,
                "eef_tracking_error_m": 0.0,
                "joint_limits": {"status": "pass"},
                "workspace": {"status": "pass"},
            }
        )
        grasp_steps.append(
            {
                "status": "pass" if grasp else "fail",
                "left": grasp,
                "right": grasp,
                "eef_sphere_center_relative_position_body_local_m": [0.0, 0.0, 0.0],
                "eef_body_relative_quaternion_local_wxyz": [1.0, 0.0, 0.0, 0.0],
                "relative_position_error_m": 0.0,
                "relative_rotation_error_rad": 0.0,
                "gripper_current_action": [1.0],
            }
        )
        step_contact = contact and index == 0
        control_steps.append(
            {
                "control_step_index": index,
                "status": "complete",
                "expected_physics_substeps": 1,
                "captured_physics_substeps": 1,
                "unobserved_expected_substeps": 0,
                "contact_samples": {
                    "contact": 1 if step_contact else 0,
                    "no_contact": 0 if step_contact else 1,
                    "unavailable": 0,
                },
            }
        )
    return {
        "candidate_id": candidate_id,
        "status": "ok",
        "snapshot_start_state_match": True,
        "start_state_fingerprint": fingerprint,
        "execution_complete": True,
        "sphere_center_body_local_offset_m": sphere_offset.tolist(),
        "start_object_root_pos_world_m": root_from_sphere(
            [0.0, 0.0, 0.0], object_quaternion
        ),
        "start_object_root_quat_world_wxyz": object_quaternion.tolist(),
        "start_object_sphere_center_world_m": [0.0, 0.0, 0.0],
        "requested_steps": control_step_count,
        "executed_steps": control_step_count,
        "goal_sphere_center_world_m": [0.01, 0.0, 0.0],
        "reference_path": [
            [0.01 * index / control_step_count, 0.0, 0.0]
            for index in range(control_step_count + 1)
        ],
        "start_state": {
            "mujoco_integration_state": [0.0],
            "robot_states": [
                {"eef_pos": [0.0, 0.0, 0.0], "eef_quat": [1.0, 0.0, 0.0, 0.0]}
            ],
            "object_states": {
                "carried": {
                    "body_id": 10,
                    "root_pos": root_from_sphere([0.0, 0.0, 0.0], object_quaternion),
                    "root_quat": object_quaternion.tolist(),
                }
            },
        },
        "task_feasibility": {
            "status": "pass",
            "final_task_success_confirmed": False,
            "complete_path_executed": True,
            "no_terminal_done": True,
            "finite_reward_trace": True,
            "opposing_finger_grasp_maintained": True,
            "endpoint_within_tracking_tolerance": True,
        },
        "steps": steps,
        "closed_loop_tracking": {
            "status": "pass" if tracking else "fail",
            "maximum_sphere_center_tracking_error_m": 0.0 if tracking else 0.1,
            "maximum_eef_tracking_error_m": 0.0,
        },
        "grasp_evidence": {
            "status": "pass" if grasp else "fail",
            "reason": None if grasp else "opposing contact lost",
            "physical_grasp_confirmed_by_distance_only": False,
            "initial_finger_contacts": {
                "status": "pass" if grasp else "fail",
                "left": grasp,
                "right": grasp,
            },
            "initial_eef_sphere_center_relative_position_body_local_m": [
                0.0,
                0.0,
                0.0,
            ],
            "initial_eef_body_relative_quaternion_local_wxyz": [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            "per_step": grasp_steps,
        },
        "contact_monitoring": {
            "status": "complete",
            "sampling_point": "immediately_after_each_successful_MjSim.step",
            "robot_classification_status": "available",
            "control_steps": control_steps,
            "expected_physics_substeps_total": control_step_count,
            "captured_physics_substeps": control_step_count,
            "unobserved_expected_substeps": 0,
            "contact_substeps": 1 if contact else 0,
            "no_contact_substeps": control_step_count - 1
            if contact
            else control_step_count,
            "unavailable_contact_substeps": 0,
            "events": events,
        },
    }


class FakeConnection:
    def __init__(self, candidates, *, restored=True):
        self.response = {
            "status": "complete",
            "main_state_restored": restored,
            "candidates": candidates,
        }
        self.request = None

    def send(self, request):
        self.request = request

    def recv(self):
        return self.response


class TrajectoryTrackingTests(unittest.TestCase):
    def _run(self, candidates, *, restored=True):
        fixtures = _fixtures(len(candidates))
        fixtures[0]["candidates"] = [
            {
                "candidate_id": item["candidate_id"],
                "passes_geometric_screen": True,
                "executable": False,
                "physical_safety_verified": False,
                "goal_world_m": [0.01, 0.0, 0.0],
                "points_world_m": [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
            }
            for item in candidates
        ]
        connection = FakeConnection(candidates, restored=restored)
        result = track_trajectory_bundle(
            connection,
            "snapshot-1",
            *fixtures,
            config=TrackingConfig(),
        )
        return result, connection

    def test_normal_opposing_finger_grasp_is_accepted_as_branch_safe(self):
        result, connection = self._run([_bridge_candidate("candidate-0")])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["executable_safe_candidate_ids"], ["candidate-0"])
        self.assertEqual(result["candidates"][0]["status"], "verified_safe_tracking")
        categories = {
            item["category"]
            for item in result["candidates"][0]["contact_classification"]["events"]
        }
        self.assertEqual(categories, {"allowed_grasp_contact"})
        self.assertEqual(connection.request["op"], "track_trajectory_bundle")
        self.assertEqual(
            connection.request["carried_binding"]["semantic_binding_geom_ids"], [0]
        )
        self.assertEqual(
            connection.request["options"]["max_control_steps_per_candidate"], 480
        )
        self.assertEqual(connection.request["options"]["max_total_control_steps"], 7200)

    def test_ungrasped_start_is_rejected(self):
        result, _ = self._run([_bridge_candidate("candidate-0", grasp=False)])
        self.assertEqual(result["executable_safe_candidate_ids"], [])
        self.assertIn(
            "initial_opposing_finger_contact_evidence_missing",
            result["candidates"][0]["rejection_reasons"],
        )

    def test_bundle_and_verified_binding_sphere_centers_must_match(self):
        fixtures = _fixtures(1)
        fixtures[1]["center_world_m"] = [0.001, 0.0, 0.0]
        connection = FakeConnection([_bridge_candidate("candidate-0")])
        with self.assertRaisesRegex(ValueError, "sphere center differs"):
            track_trajectory_bundle(connection, "snapshot-1", *fixtures)
        self.assertIsNone(connection.request)

    def test_reported_sphere_offset_must_match_snapshot_binding(self):
        candidate = _bridge_candidate("offset-mismatch")
        candidate["sphere_center_body_local_offset_m"] = [0.001, 0.0, 0.0]
        result, _ = self._run([candidate])
        self.assertEqual(result["executable_safe_candidate_ids"], [])
        self.assertTrue(
            any(
                reason.startswith("candidate_start_state_invalid:")
                for reason in result["candidates"][0]["rejection_reasons"]
            )
        )

    def test_unknown_or_unmapped_robot_part_rule_stops_preflight(self):
        for subject_parts, decision in ((["gripper"], "unknown"), (["arm"], "forbid")):
            fixtures = _fixtures(1)
            fixtures[3]["rules"].append(
                {
                    "id": "unsupported-rule",
                    "subject_parts": subject_parts,
                    "target_id": "obstacle",
                    "predicate": "contact",
                    "decision": decision,
                    "phase": "any",
                    "reason": "Unresolved or unmapped safety rule.",
                }
            )
            with self.assertRaisesRegex(ValueError, "teacher_"):
                track_trajectory_bundle(FakeConnection([]), "snapshot-1", *fixtures)

    def test_target_outside_workspace_is_rejected(self):
        result = check_workspace_target([0.0, 0.0, 2.0], [0.0, 0.0, 0.0], 1.0)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(
            result["reason"], "target_outside_conservative_Panda_reach_bound"
        )
        controller_limit = check_workspace_target(
            [0.6, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        )
        self.assertEqual(
            controller_limit["reason"], "active_controller_workspace_limit_exceeded"
        )

    def test_osc_delta_outside_active_input_bounds_is_rejected_without_clipping(self):
        fields = {
            "input_min": [-1.0] * 6,
            "input_max": [1.0] * 6,
            "output_min": [-0.05] * 3 + [-0.5] * 3,
            "output_max": [0.05] * 3 + [0.5] * 3,
            "input_ref_frame": "world",
        }
        with self.assertRaisesRegex(ValueError, "outside normalized input bounds"):
            inverse_osc_world_delta([0.06, 0.0, 0.0], [0.0] * 3, fields)

    def test_feedback_pose_delta_maps_to_active_world_frame_osc_action(self):
        fields = {
            "input_min": [-1.0] * 6,
            "input_max": [1.0] * 6,
            "output_min": [-0.05] * 3 + [-0.5] * 3,
            "output_max": [0.05] * 3 + [0.5] * 3,
            "input_ref_frame": "world",
        }
        target_quaternion = [np.cos(0.05), 0.0, 0.0, np.sin(0.05)]
        action = world_pose_tracking_action(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            target_quaternion,
            fields,
        )
        np.testing.assert_allclose(
            action, [0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 1.0], atol=1e-12
        )

    def test_teacher_forbidden_object_collision_is_rejected(self):
        result, _ = self._run([_bridge_candidate("candidate-0", collision=True)])
        self.assertEqual(result["executable_safe_candidate_ids"], [])
        self.assertEqual(
            result["candidates"][0]["contact_compiler"]["verdict"], "violation"
        )
        classified = result["candidates"][0]["contact_classification"]["events"]
        self.assertIn(
            "teacher_forbidden_contact", {item["category"] for item in classified}
        )

    def test_teacher_forbidden_gripper_contact_for_any_bound_object_is_compiled(self):
        fixtures = _fixtures(1)
        teacher = fixtures[3]
        teacher["rules"].append(
            {
                "id": "forbid-obstacle-gripper",
                "subject_parts": ["gripper"],
                "target_id": "obstacle",
                "predicate": "contact",
                "decision": "forbid",
                "phase": "any",
                "reason": "Do not let the gripper touch this obstacle.",
            }
        )
        candidate = _bridge_candidate("candidate-0")
        candidate["contact_monitoring"]["events"].append(_event("gripper-obstacle"))
        connection = FakeConnection([candidate])
        result = track_trajectory_bundle(connection, "snapshot-1", *fixtures)
        self.assertEqual(result["executable_safe_candidate_ids"], [])
        self.assertEqual(
            result["candidates"][0]["contact_compiler"]["verdict"], "violation"
        )

    def test_unclassified_other_contact_is_not_treated_as_safe(self):
        fixtures = _fixtures(1)
        geometry_inventory = fixtures[5]
        geometry_inventory["geom_count"] = 4
        geometry_inventory["geometries"].append(
            {
                "geom_id": 3,
                "geom_name": "other_scene_geom",
                "body_id": 30,
                "body_name": "other_scene_body",
                "category": "scene",
            }
        )
        candidate = _bridge_candidate("candidate-0")
        other_event = {
            "geom_pair": [
                {"id": 1, "name": "obstacle_geom"},
                {"id": 3, "name": "other_scene_geom"},
            ],
            "body_pair": [
                {"id": 20, "name": "obstacle_body"},
                {"id": 30, "name": "other_scene_body"},
            ],
            "classification": "scene_scene",
            "robot_involved": False,
            "first_control_step": 0,
            "first_physics_substep": 0,
            "occurrence_count": 1,
            "contact_point_count": 1,
            "max_force_norm": 1.0,
            "force_sample_count": 1,
            "force_unavailable_count": 0,
        }
        candidate["contact_monitoring"]["events"].append(other_event)
        connection = FakeConnection([candidate])
        result = track_trajectory_bundle(connection, "snapshot-1", *fixtures)
        self.assertEqual(result["executable_safe_candidate_ids"], [])
        self.assertEqual(
            result["candidates"][0]["contact_classification"]["status"], "unknown"
        )

    def test_lost_grasp_and_tracking_failure_are_rejected(self):
        for candidate in (
            _bridge_candidate("lost-grasp", grasp=False),
            _bridge_candidate("tracking-failed", tracking=False),
        ):
            result, _ = self._run([candidate])
            self.assertEqual(result["executable_safe_candidate_ids"], [])
            self.assertEqual(result["candidates"][0]["status"], "rejected")

    def test_incomplete_path_or_task_feasibility_evidence_is_rejected(self):
        candidate = _bridge_candidate("candidate-0")
        candidate["execution_complete"] = False
        candidate["task_feasibility"]["status"] = "unavailable"
        result, _ = self._run([candidate])
        self.assertEqual(result["executable_safe_candidate_ids"], [])
        self.assertIn(
            "local_task_feasibility_evidence_incomplete",
            result["candidates"][0]["rejection_reasons"],
        )

    def test_snapshot_restore_failure_rejects_every_candidate(self):
        result, _ = self._run([_bridge_candidate("candidate-0")], restored=False)
        self.assertFalse(result["main_state_restored"])
        self.assertEqual(result["executable_safe_candidate_ids"], [])
        self.assertIn(
            "main_snapshot_restore_unverified",
            result["candidates"][0]["rejection_reasons"],
        )

    def test_candidates_must_prove_same_snapshot_start(self):
        candidates = [
            _bridge_candidate("candidate-a", fingerprint="a" * 64),
            _bridge_candidate("candidate-b", fingerprint="b" * 64),
        ]
        result, _ = self._run(candidates)
        self.assertEqual(result["executable_safe_candidate_ids"], [])
        self.assertIn(
            "candidate_snapshot_start_fingerprints_missing_or_mismatched",
            result["response_errors"],
        )

    def test_multiple_candidates_with_same_snapshot_are_assessed_independently(self):
        candidates = [
            _bridge_candidate("route-left", fingerprint=FINGERPRINT),
            _bridge_candidate("route-above", fingerprint=FINGERPRINT),
        ]
        result, _ = self._run(candidates)
        self.assertEqual(
            result["executable_safe_candidate_ids"], ["route-left", "route-above"]
        )
        self.assertEqual(
            [item["start_state_fingerprint"] for item in result["candidates"]],
            [FINGERPRINT, FINGERPRINT],
        )

    def test_nonzero_body_local_sphere_offset_tracks_after_body_rotation(self):
        half_turn = np.sqrt(0.5)
        candidate = _bridge_candidate(
            "rotating-offset",
            sphere_offset=(0.03, 0.0, 0.0),
            object_quaternion=(1.0, 0.0, 0.0, 0.0),
            step_quaternions=[
                (half_turn, 0.0, 0.0, half_turn),
                (1.0, 0.0, 0.0, 0.0),
            ],
        )
        result, _ = self._run([candidate])
        self.assertEqual(result["executable_safe_candidate_ids"], ["rotating-offset"])
        self.assertEqual(result["candidates"][0]["status"], "verified_safe_tracking")
        np.testing.assert_allclose(
            candidate["start_object_root_pos_world_m"],
            [-0.03, 0.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            candidate["steps"][0]["actual_object_root_pos_world_m"],
            [0.002, -0.03, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            candidate["steps"][0]["actual_object_sphere_center_world_m"],
            [0.002, 0.0, 0.0],
        )

    def test_body_local_sphere_offset_transforms_and_zero_offset(self):
        quarter_turn = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
        local = body_local_offset_from_world_point(
            [0.2, 0.3, 0.0], [0.2, 0.2, 0.0], quarter_turn
        )
        np.testing.assert_allclose(local, [0.1, 0.0, 0.0], atol=1e-12)
        world = body_local_point_to_world([0.2, 0.2, 0.0], quarter_turn, local)
        np.testing.assert_allclose(world, [0.2, 0.3, 0.0], atol=1e-12)
        rotated_center = body_local_point_to_world(
            [0.2, 0.2, 0.0], quarter_turn, [0.1, 0.0, 0.0]
        )
        unrotated_center = body_local_point_to_world(
            [0.2, 0.2, 0.0], [1.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0]
        )
        self.assertFalse(np.allclose(rotated_center, unrotated_center))
        zero_local = body_local_offset_from_world_point(
            [0.2, 0.2, 0.0], [0.2, 0.2, 0.0], quarter_turn
        )
        np.testing.assert_allclose(zero_local, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(
            body_local_point_to_world([0.2, 0.2, 0.0], quarter_turn, zero_local),
            [0.2, 0.2, 0.0],
            atol=1e-12,
        )

    def test_resampling_is_bounded_to_20_hz_reference_speed(self):
        result = resample_polyline_by_distance(
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]],
            speed_m_s=0.04,
            control_timestep_s=0.05,
            maximum_steps=20,
        )
        self.assertEqual(result["requested_steps"], 10)
        self.assertLessEqual(result["maximum_reference_displacement_m"], 0.002 + 1e-12)

    def test_realistic_long_path_exceeds_old_240_step_cap_within_new_budget(self):
        config = TrackingConfig()
        plans, total = plan_bundle_reference_paths(
            [[[0.0, 0.0, 0.0], [0.567, 0.0, 0.0]]],
            speed_m_s=config.reference_speed_m_s,
            control_timestep_s=0.05,
            maximum_steps_per_candidate=config.max_control_steps_per_candidate,
            maximum_total_steps=config.max_total_control_steps,
        )
        self.assertEqual(plans[0]["requested_steps"], 284)
        self.assertGreater(plans[0]["requested_steps"], 240)
        self.assertEqual(total, 284)
        self.assertLessEqual(plans[0]["maximum_reference_displacement_m"], 0.002)

    def test_true_per_candidate_and_bundle_over_budget_paths_are_rejected(self):
        config = TrackingConfig()
        with self.assertRaisesRegex(ValueError, "exceeding_limit_480"):
            plan_bundle_reference_paths(
                [[[0.0, 0.0, 0.0], [0.962, 0.0, 0.0]]],
                speed_m_s=config.reference_speed_m_s,
                control_timestep_s=0.05,
                maximum_steps_per_candidate=config.max_control_steps_per_candidate,
                maximum_total_steps=config.max_total_control_steps,
            )
        with self.assertRaisesRegex(ValueError, "exceeding_total_budget_300"):
            plan_bundle_reference_paths(
                [
                    [[0.0, 0.0, 0.0], [0.31, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.31, 0.0, 0.0]],
                ],
                speed_m_s=config.reference_speed_m_s,
                control_timestep_s=0.05,
                maximum_steps_per_candidate=config.max_control_steps_per_candidate,
                maximum_total_steps=300,
            )

    def test_reference_speed_cannot_be_increased_to_bypass_duration_budget(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed 0.04 m/s"):
            TrackingConfig(reference_speed_m_s=0.05).validate()


if __name__ == "__main__":
    unittest.main()
