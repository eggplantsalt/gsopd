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

from research.gs_opd.repair.panda_gripper_state import parse_panda_gripper_closure
from research.gs_opd.repair.trajectory_tracking import (
    TrackingConfig,
    _parse_eef_reference_poses,
    _validate_carried_binding,
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
        "carry": _semantic_result("carry", 10, "carry_body", 3, "carry_visual"),
        "obstacle": _semantic_result(
            "obstacle", 20, "obstacle_body", 1, "obstacle_geom"
        ),
    }
    geometry_inventory = {
        "status": "complete",
        "robot_classification_status": "available",
        "geom_count": 5,
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
            {
                "geom_id": 3,
                "geom_name": "carry_visual",
                "body_id": 10,
                "body_name": "carry_body",
                "category": "scene",
            },
            {
                "geom_id": 4,
                "geom_name": "carry_geom_2",
                "body_id": 10,
                "body_name": "carry_body",
                "category": "scene",
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
        "geom_ids": [0, 4],
        "geom_names": ["carry_geom", "carry_geom_2"],
        "semantic_binding_geom_ids": [3],
        "semantic_binding_geom_names": ["carry_visual"],
        "components": [
            {
                "geom_id": 0,
                "geom_name": "carry_geom",
                "body_id": 10,
                "body_name": "carry_body",
            },
            {
                "geom_id": 4,
                "geom_name": "carry_geom_2",
                "body_id": 10,
                "body_name": "carry_body",
            },
        ],
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
                "gripper_current_action": [-0.95, 0.95],
                "gripper_closure": parse_panda_gripper_closure([-0.95, 0.95]),
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
            "initial_gripper_current_action": [-0.95, 0.95],
            "initial_gripper_closure": parse_panda_gripper_closure([-0.95, 0.95]),
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
            "robot_classification_error": None,
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
    def test_explicit_mink_eef_reference_requires_finite_unit_full_poses(self):
        valid = [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]]
        parsed = _parse_eef_reference_poses(valid, expected_count=2)
        self.assertEqual(parsed.shape, (2, 7))
        np.testing.assert_allclose(parsed[1, 3:], [0.0, 0.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "finite_N_by_7"):
            _parse_eef_reference_poses(valid[:1], expected_count=2)
        with self.assertRaisesRegex(ValueError, "unit_wxyz"):
            _parse_eef_reference_poses(
                [[0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]],
                expected_count=2,
            )

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
            connection.request["carried_binding"]["semantic_binding_geom_ids"], [3]
        )
        self.assertEqual(connection.request["carried_binding"]["geom_ids"], [0, 4])
        self.assertEqual(
            connection.request["carried_binding"]["geom_names"],
            ["carry_geom", "carry_geom_2"],
        )
        self.assertEqual(
            connection.request["carried_binding"]["semantic_binding_geom_names"],
            ["carry_visual"],
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

    def test_visual_only_semantic_geom_is_independent_of_collision_coverage(self):
        fixtures = _fixtures(1)
        bundle, binding, _context, _teacher_data, semantic, inventory, _gripper = fixtures
        self.assertEqual(binding["semantic_binding_geom_ids"], [3])
        self.assertNotIn(3, binding["geom_ids"])
        self.assertEqual(_validate_carried_binding(bundle, binding, semantic, inventory), ("carry", [0, 4]))

    def test_semantic_geom_can_also_be_a_collision_geom(self):
        fixtures = _fixtures(1)
        bundle, binding, _context, _teacher_data, semantic, inventory, _gripper = fixtures
        binding["semantic_binding_geom_ids"] = [0]
        binding["semantic_binding_geom_names"] = ["carry_geom"]
        semantic["carry"]["selected_candidate"]["geom_ids"] = [0]
        semantic["carry"]["physical_association"]["geom_ids"] = [0]
        semantic["carry"]["physical_association"]["geom_names"] = ["carry_geom"]
        self.assertEqual(_validate_carried_binding(bundle, binding, semantic, inventory), ("carry", [0, 4]))

    def test_binding_rejects_missing_or_reordered_collision_geom_names(self):
        for mutation in ("missing", "reordered"):
            with self.subTest(mutation=mutation):
                fixtures = _fixtures(1)
                bundle, binding, _context, _teacher_data, semantic, inventory, _gripper = fixtures
                if mutation == "missing":
                    binding["geom_names"] = []
                else:
                    binding["geom_names"] = list(reversed(binding["geom_names"]))
                with self.assertRaises(ValueError):
                    _validate_carried_binding(bundle, binding, semantic, inventory)

    def test_binding_rejects_cross_object_semantic_geom_and_incomplete_coverage(self):
        fixtures = _fixtures(1)
        bundle, binding, _context, _teacher_data, semantic, inventory, _gripper = fixtures
        binding["semantic_binding_geom_ids"] = [1]
        binding["semantic_binding_geom_names"] = ["obstacle_geom"]
        semantic["carry"]["selected_candidate"]["geom_ids"] = [1]
        semantic["carry"]["physical_association"]["geom_ids"] = [1]
        semantic["carry"]["physical_association"]["geom_names"] = ["obstacle_geom"]
        with self.assertRaises(ValueError):
            _validate_carried_binding(bundle, binding, semantic, inventory)

        fixtures = _fixtures(1)
        bundle, binding, _context, _teacher_data, semantic, inventory, _gripper = fixtures
        binding["geom_ids"] = [0]
        binding["geom_names"] = ["carry_geom"]
        with self.assertRaises(ValueError):
            _validate_carried_binding(bundle, binding, semantic, inventory)

    def test_binding_rejects_candidate_identity_conflicting_with_semantic_binding(self):
        fixtures = _fixtures(1)
        bundle, binding, _context, _teacher_data, semantic, inventory, _gripper = fixtures
        semantic["carry"]["selected_candidate"]["geom_ids"] = [4]
        with self.assertRaisesRegex(ValueError, "semantic binding"):
            _validate_carried_binding(bundle, binding, semantic, inventory)

    def test_binding_rejects_unsupported_child_body_geometry(self):
        fixtures = _fixtures(1)
        bundle, binding, _context, _teacher_data, semantic, inventory, _gripper = fixtures
        inventory["geometries"][4]["body_id"] = 11
        inventory["geometries"][4]["body_name"] = "carry_child"
        with self.assertRaisesRegex(ValueError, "active root Body"):
            _validate_carried_binding(bundle, binding, semantic, inventory)

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
        self.assertNotIn(
            "out_of_scope_background_contact",
            {item["status"] for item in classified},
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
        geometry_inventory["geom_count"] = 6
        geometry_inventory["geometries"].append(
            {
                "geom_id": 5,
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

    def test_preexisting_unrelated_scene_contact_is_out_of_scope_with_raw_evidence(self):
        fixtures = _fixtures(1)
        geometry_inventory = fixtures[5]
        geometry_inventory["geom_count"] = 7
        geometry_inventory["model_body_count"] = 32
        geometry_inventory["geometries"].extend(
            [
                {
                    "geom_id": 5,
                    "geom_name": "background_geom_a",
                    "body_id": 30,
                    "body_name": "background_body_a",
                    "category": "scene",
                },
                {
                    "geom_id": 6,
                    "geom_name": "background_geom_b",
                    "body_id": 31,
                    "body_name": "background_body_b",
                    "category": "scene",
                },
            ]
        )
        other_event = {
            "geom_pair": [
                {"id": 5, "name": "background_geom_a"},
                {"id": 6, "name": "background_geom_b"},
            ],
            "body_pair": [
                {"id": 30, "name": "background_body_a"},
                {"id": 31, "name": "background_body_b"},
            ],
            "classification": "scene_scene",
            "robot_involved": False,
            "first_control_step": 0,
            "first_physics_substep": 0,
            "occurrence_count": 5,
            "contact_point_count": 5,
            "max_force_norm": 0.25,
            "force_sample_count": 5,
            "force_unavailable_count": 0,
        }
        candidate = _bridge_candidate("candidate-0")
        candidate["contact_monitoring"]["events"].append(other_event)
        candidate["start_contacts"] = {
            "status": "contact",
            "contacts": [
                {
                    "geom1": other_event["geom_pair"][0],
                    "geom2": other_event["geom_pair"][1],
                    "body1": other_event["body_pair"][0],
                    "body2": other_event["body_pair"][1],
                }
            ],
        }
        connection = FakeConnection([candidate])
        result = track_trajectory_bundle(connection, "snapshot-1", *fixtures)
        self.assertEqual(result["executable_safe_candidate_ids"], ["candidate-0"])
        classified = result["candidates"][0]["contact_classification"]
        background = next(
            item
            for item in classified["events"]
            if item.get("category") == "out_of_scope_background_contact"
        )
        self.assertEqual(classified["status"], "pass")
        self.assertEqual(background["status"], "out_of_scope_background_contact")
        self.assertEqual(background["raw_event"], other_event)
        self.assertEqual(
            background["preexisting_start_contact_evidence"][0]["geom1"],
            other_event["geom_pair"][0],
        )
        candidate["snapshot_start_state_match"] = False
        mismatched_snapshot_result, _ = self._run([candidate])
        self.assertEqual(
            mismatched_snapshot_result["candidates"][0]["contact_classification"][
                "status"
            ],
            "unknown",
        )
        candidate["snapshot_start_state_match"] = True
        candidate["contact_monitoring"].pop("robot_classification_error")
        incomplete_classification_result, _ = self._run([candidate])
        self.assertEqual(
            incomplete_classification_result["candidates"][0][
                "contact_classification"
            ]["status"],
            "unknown",
        )

    def test_preexisting_robot_contact_is_not_out_of_scope(self):
        fixtures = _fixtures(1)
        geometry_inventory = fixtures[5]
        geometry_inventory["geom_count"] = 6
        geometry_inventory["geometries"].append(
            {
                "geom_id": 5,
                "geom_name": "robot_arm_geom",
                "body_id": 1,
                "body_name": "robot_body",
                "category": "robot",
            }
        )
        robot_event = {
            "geom_pair": [
                {"id": 5, "name": "robot_arm_geom"},
                {"id": 1, "name": "obstacle_geom"},
            ],
            "body_pair": [
                {"id": 1, "name": "robot_body"},
                {"id": 20, "name": "obstacle_body"},
            ],
            "classification": "robot_scene",
            "robot_involved": True,
            "first_control_step": 0,
            "first_physics_substep": 0,
            "occurrence_count": 1,
            "contact_point_count": 1,
            "max_force_norm": 0.1,
            "force_sample_count": 1,
            "force_unavailable_count": 0,
        }
        candidate = _bridge_candidate("candidate-0")
        candidate["contact_monitoring"]["events"].append(robot_event)
        candidate["start_contacts"] = {
            "status": "contact",
            "contacts": [
                {
                    "geom1": robot_event["geom_pair"][0],
                    "geom2": robot_event["geom_pair"][1],
                    "body1": robot_event["body_pair"][0],
                    "body2": robot_event["body_pair"][1],
                }
            ],
        }
        result, _ = self._run([candidate])
        self.assertEqual(result["executable_safe_candidate_ids"], [])
        classified = result["candidates"][0]["contact_classification"]
        self.assertEqual(classified["status"], "unknown")
        robot_record = next(
            item for item in classified["events"] if item.get("raw_event") == robot_event
        )
        self.assertEqual(robot_record["status"], "unknown")

    def test_preexisting_carried_object_contact_is_not_out_of_scope(self):
        fixtures = _fixtures(1)
        geometry_inventory = fixtures[5]
        geometry_inventory["geom_count"] = 6
        geometry_inventory["geometries"].append(
            {
                "geom_id": 5,
                "geom_name": "background_geom",
                "body_id": 30,
                "body_name": "background_body",
                "category": "scene",
            }
        )
        carried_event = {
            "geom_pair": [
                {"id": 0, "name": "carry_geom"},
                {"id": 5, "name": "background_geom"},
            ],
            "body_pair": [
                {"id": 10, "name": "carry_body"},
                {"id": 30, "name": "background_body"},
            ],
            "classification": "scene_scene",
            "robot_involved": False,
            "first_control_step": 0,
            "first_physics_substep": 0,
            "occurrence_count": 1,
            "contact_point_count": 1,
            "max_force_norm": 0.1,
            "force_sample_count": 1,
            "force_unavailable_count": 0,
        }
        candidate = _bridge_candidate("candidate-0")
        candidate["contact_monitoring"]["events"].append(carried_event)
        candidate["start_contacts"] = {
            "status": "contact",
            "contacts": [
                {
                    "geom1": carried_event["geom_pair"][0],
                    "geom2": carried_event["geom_pair"][1],
                    "body1": carried_event["body_pair"][0],
                    "body2": carried_event["body_pair"][1],
                }
            ],
        }
        result, _ = self._run([candidate])
        self.assertEqual(result["executable_safe_candidate_ids"], [])
        classified = result["candidates"][0]["contact_classification"]
        self.assertEqual(classified["status"], "unknown")
        carried_record = next(
            item
            for item in classified["events"]
            if item.get("raw_event") == carried_event
        )
        self.assertEqual(carried_record["status"], "unknown")

    def test_initial_contact_with_unverified_geometry_identity_stays_unknown(self):
        fixtures = _fixtures(1)
        geometry_inventory = fixtures[5]
        geometry_inventory["geom_count"] = 6
        geometry_inventory["geometries"].append(
            {
                "geom_id": 5,
                "geom_name": "verified_background_geom",
                "body_id": 30,
                "body_name": "background_body",
                "category": "scene",
            }
        )
        event = {
            "geom_pair": [
                {"id": 1, "name": "obstacle_geom"},
                {"id": 5, "name": "unverified_background_geom"},
            ],
            "body_pair": [
                {"id": 20, "name": "obstacle_body"},
                {"id": 30, "name": "background_body"},
            ],
            "classification": "scene_scene",
            "robot_involved": False,
            "first_control_step": 0,
            "first_physics_substep": 0,
            "occurrence_count": 1,
            "contact_point_count": 1,
            "max_force_norm": 0.1,
            "force_sample_count": 1,
            "force_unavailable_count": 0,
        }
        candidate = _bridge_candidate("candidate-0")
        candidate["contact_monitoring"]["events"].append(event)
        candidate["start_contacts"] = {
            "status": "contact",
            "contacts": [
                {
                    "geom1": event["geom_pair"][0],
                    "geom2": event["geom_pair"][1],
                    "body1": event["body_pair"][0],
                    "body2": event["body_pair"][1],
                }
            ],
        }
        result, _ = self._run([candidate])
        self.assertEqual(result["executable_safe_candidate_ids"], [])
        classified = result["candidates"][0]["contact_classification"]
        self.assertEqual(classified["status"], "unknown")
        identity_record = next(
            item for item in classified["events"] if item.get("raw_event") == event
        )
        self.assertEqual(identity_record["status"], "unknown")

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
        plans, total = plan_bundle_reference_paths(
            [[[0.0, 0.0, 0.0], [0.962, 0.0, 0.0]]],
            speed_m_s=config.reference_speed_m_s,
            control_timestep_s=0.05,
            maximum_steps_per_candidate=config.max_control_steps_per_candidate,
            maximum_total_steps=config.max_total_control_steps,
        )
        self.assertEqual(plans[0]["requested_steps"], 481)
        self.assertEqual(plans[0]["budget_status"], "rejected")
        self.assertEqual(
            plans[0]["budget_rejection_reason"],
            "trajectory_requires_481_steps_exceeding_limit_480",
        )
        self.assertEqual(total, 0)

        plans, total = plan_bundle_reference_paths(
            [
                [[0.0, 0.0, 0.0], [0.31, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.31, 0.0, 0.0]],
            ],
            speed_m_s=config.reference_speed_m_s,
            control_timestep_s=0.05,
            maximum_steps_per_candidate=config.max_control_steps_per_candidate,
            maximum_total_steps=300,
        )
        self.assertEqual([item["budget_status"] for item in plans], ["accepted", "rejected"])
        self.assertEqual(plans[1]["requested_steps"], 155)
        self.assertEqual(
            plans[1]["budget_rejection_reason"],
            "trajectory_requires_155_steps_exceeding_remaining_bundle_budget_145",
        )
        self.assertEqual(total, 155)
        self.assertLessEqual(total, 300)

    def test_bundle_budget_selection_round_robins_directions_deterministically(self):
        config = TrackingConfig()
        directions = ["left"] * 6 + ["right"] * 6 + ["above"] * 6
        paths = [
            [[0.0, 0.0, 0.0], [0.848, 0.0, 0.0]]
            for _ in directions
        ]
        plans, total = plan_bundle_reference_paths(
            paths,
            speed_m_s=config.reference_speed_m_s,
            control_timestep_s=0.05,
            maximum_steps_per_candidate=config.max_control_steps_per_candidate,
            maximum_total_steps=config.max_total_control_steps,
            directions=directions,
        )
        admitted = [
            direction
            for direction, plan in zip(directions, plans)
            if plan["budget_status"] == "accepted"
        ]
        self.assertEqual(total, 16 * 424)
        self.assertLessEqual(total, config.max_total_control_steps)
        self.assertEqual(
            {direction: admitted.count(direction) for direction in set(directions)},
            {"left": 6, "right": 5, "above": 5},
        )

    def test_v3_candidate_budget_rejection_is_independent_and_traceable(self):
        fixtures = _fixtures(18)
        bundle = fixtures[0]
        directions = ("left", "right", "above")
        step_counts = [424, 424, 424, *range(498, 513)]
        bundle["candidates"] = []
        for index, steps in enumerate(step_counts):
            length = steps * 0.04 * 0.05
            candidate_id = f"v3-{index:02d}"
            bundle["candidates"].append(
                {
                    "candidate_id": candidate_id,
                    "direction": directions[index % len(directions)],
                    "passes_geometric_screen": True,
                    "executable": False,
                    "physical_safety_verified": False,
                    "goal_world_m": [length, 0.0, 0.0],
                    "points_world_m": [[0.0, 0.0, 0.0], [length, 0.0, 0.0]],
                    "length_m": length,
                }
            )
        branch_candidates = [_bridge_candidate(f"v3-{index:02d}") for index in range(3)]
        connection = FakeConnection(branch_candidates)
        result = track_trajectory_bundle(connection, "snapshot-1", *fixtures)

        submitted = connection.request["candidates"]
        self.assertEqual(
            [item["candidate_id"] for item in submitted],
            ["v3-00", "v3-01", "v3-02"],
        )
        self.assertEqual({item["direction"] for item in submitted}, set(directions))
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["main_state_restored"])
        self.assertEqual(result["requested_candidate_count"], 18)
        self.assertEqual(result["budget_qualified_candidate_count"], 3)
        self.assertEqual(result["budget_rejected_candidate_count"], 15)
        self.assertEqual(result["preflight_total_control_steps"], 3 * 424)
        self.assertLessEqual(
            result["preflight_total_control_steps"],
            result["maximum_total_control_steps"],
        )
        self.assertEqual(
            [item["candidate_id"] for item in result["candidates"]],
            [f"v3-{index:02d}" for index in range(18)],
        )
        for item in result["candidates"][:3]:
            self.assertEqual(item["requested_steps"], 424)
            self.assertEqual(item["budget_status"], "accepted")
        for index, item in enumerate(result["candidates"][3:], start=3):
            requested_steps = step_counts[index]
            self.assertEqual(item["requested_steps"], requested_steps)
            self.assertEqual(item["budget_status"], "rejected")
            self.assertEqual(
                item["budget_rejection_reason"],
                f"trajectory_requires_{requested_steps}_steps_exceeding_limit_480",
            )
            self.assertFalse(item["executable_safe"])
            self.assertFalse(item["physical_safety_verified"])
            self.assertFalse(item["training_label_created"])

    def test_total_budget_caps_bridge_request_without_losing_direction_groups(self):
        fixtures = _fixtures(18)
        bundle = fixtures[0]
        directions = ("left", "right", "above")
        bundle["candidates"] = []
        for index in range(18):
            length = 424 * 0.04 * 0.05
            bundle["candidates"].append(
                {
                    "candidate_id": f"total-{index:02d}",
                    "direction": directions[index // 6],
                    "passes_geometric_screen": True,
                    "executable": False,
                    "physical_safety_verified": False,
                    "goal_world_m": [length, 0.0, 0.0],
                    "points_world_m": [[0.0, 0.0, 0.0], [length, 0.0, 0.0]],
                }
            )
        connection = FakeConnection(
            [
                _bridge_candidate(f"total-{index:02d}")
                for index in (*range(6), *range(6, 11), *range(12, 17))
            ]
        )
        result = track_trajectory_bundle(connection, "snapshot-1", *fixtures)

        self.assertEqual(len(connection.request["candidates"]), 16)
        self.assertEqual(
            {item["direction"] for item in connection.request["candidates"]},
            set(directions),
        )
        self.assertEqual(result["preflight_total_control_steps"], 16 * 424)
        self.assertLessEqual(
            result["preflight_total_control_steps"],
            result["maximum_total_control_steps"],
        )
        self.assertEqual(result["candidate_count"], 18)
        self.assertEqual(len(result["candidates"]), 18)
        self.assertEqual(
            [item["candidate_id"] for item in result["candidates"]],
            [f"total-{index:02d}" for index in range(18)],
        )
        rejected = [
            item for item in result["candidates"] if item["budget_status"] == "rejected"
        ]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(
            all(
                item["budget_rejection_reason"]
                == "trajectory_requires_424_steps_exceeding_remaining_bundle_budget_416"
                for item in rejected
            )
        )

    def test_reference_speed_cannot_be_increased_to_bypass_duration_budget(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed 0.04 m/s"):
            TrackingConfig(reference_speed_m_s=0.05).validate()


if __name__ == "__main__":
    unittest.main()
