# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for exact, provenance-checked task-success prefix resume."""

from __future__ import annotations

import json
import tempfile
import unittest
import pickle
from pathlib import Path

import numpy as np

from research.gs_opd.repair.task_success_first_executor import (
    _phase_evidence,
    _finger_contacts_clear,
    _object_contact_escape_direction,
    _register_verified_operation_skill_from_report,
    _release_retreat_target,
    _select_active_target_predicate,
    _verified_grasp_resume_descriptor,
    _verified_placement_resume_descriptor,
)


class TaskSuccessFirstExecutorTests(unittest.TestCase):
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
