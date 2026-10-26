# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for cross-scene task-skill experience lookup and geometry migration."""

from __future__ import annotations

import json
import tempfile
import unittest

import numpy as np

from research.gs_opd.repair.task_skill_experience import (
    alternate_experience_approach_candidates,
    append_verified_search_operation_skill,
    append_verified_search_grasp_template,
    load_experience_library,
    matching_grasp_template,
    matching_grasp_templates,
    matching_operation_skill,
    merge_verified_search_experiences,
    migrate_object_relative_pose,
)
from pathlib import Path


class TaskSkillExperienceTests(unittest.TestCase):
    def test_instance_suffix_is_removed_for_cross_scene_object_lookup(self):
        library = {
            "schema_version": "gs_opd_task_skill_experience_library.v1",
            "source_policy": {"official_safety_demonstration_data_used": False},
            "grasp_templates": [{"object_family": "moka_pot", "template_id": "prior"}],
        }
        self.assertEqual(matching_grasp_template(library, "moka_pot_2")["template_id"], "prior")
        self.assertIsNone(matching_grasp_template(library, "black_book_1"))

    def test_operation_skill_lookup_matches_family_and_requires_phase_reuse_policy(self):
        library = {
            "operation_skill_templates": [{
                "skill_id": "object_transfer_cycle:moka_pot:t248",
                "object_family": "moka_pot",
                "operation": "pick_transport_place_release",
                "historical_actions_replayed": False,
                "phase_sequence": ["approach", "stable_grasp", "lift", "transport", "release"],
            }],
        }
        skill = matching_operation_skill(library, "moka_pot_1")
        self.assertEqual(skill["skill_id"], "object_transfer_cycle:moka_pot:t248")
        self.assertIsNone(matching_operation_skill(library, "black_book_1"))

    def test_complete_L0_native_success_outranks_partial_L2_skill_with_more_validations(self):
        library = {
            "grasp_templates": [
                {"template_id": "l0t2-pot2", "object_family": "moka_pot",
                 "source_global_timestep": 248},
                {"template_id": "l2t2-partial", "object_family": "moka_pot",
                 "cross_scene_physical_validation_count": 3},
            ],
            "operation_skill_templates": [
                {"skill_id": "l0t2-complete", "grasp_template_id": "l0t2-pot2",
                 "object_family": "moka_pot", "operation": "pick_transport_place_release",
                 "source_native_task_success": True,
                 "phase_sequence": ["approach", "stable_grasp", "lift", "transport",
                                    "precise_placement", "release", "withdrawal", "settle"],
                 "historical_actions_replayed": False},
                {"skill_id": "l2t2-partial", "grasp_template_id": "l2t2-partial",
                 "object_family": "moka_pot", "operation": "pick_transport_place_release",
                 "source_native_task_success": False, "native_object_goal_satisfied": True,
                 "cross_scene_physical_validation_count": 3,
                 "phase_sequence": ["approach", "stable_grasp", "lift"],
                 "historical_actions_replayed": False},
            ],
        }
        self.assertEqual(matching_grasp_template(library, "moka_pot_2")["template_id"], "l0t2-pot2")
        self.assertEqual(
            [item["template_id"] for item in matching_grasp_templates(library, "moka_pot_2")],
            ["l0t2-pot2", "l2t2-partial"],
        )
        self.assertEqual(matching_operation_skill(library, "moka_pot_2")["skill_id"], "l0t2-complete")

    def test_verified_native_object_cycle_is_appended_and_prioritized(self):
        library = {
            "schema_version": "gs_opd_task_skill_experience_library.v1",
            "source_policy": {"official_safety_demonstration_data_used": False},
            "operation_skill_templates": [{
                "skill_id": "l0-source-cycle", "object_family": "moka_pot",
                "operation": "pick_transport_place_release",
                "historical_actions_replayed": False,
            }],
        }
        template = {
            "skill_id": "l2-verified-cycle",
            "object_family": "moka_pot",
            "operation": "pick_transport_place_release",
            "phase_sequence": ["approach", "stable_grasp", "lift", "transport",
                               "precise_placement", "release", "withdrawal", "settle"],
            "skill_success": True,
            "native_object_goal_satisfied": True,
            "released_and_stable": True,
            "stable_bilateral_grasp_verified": True,
            "native_goal_predicate": ["on", "moka_pot_1", "flat_stove"],
            "physical_actions": 420,
            "source_execution_reports": ["run5.json", "run7.json", "run8.json"],
            "safety_certified": False,
            "official_safety_demonstrations_used": False,
            "historical_actions_replayed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(library), encoding="utf-8")
            self.assertTrue(append_verified_search_operation_skill(path, template))
            self.assertFalse(append_verified_search_operation_skill(path, template))
            updated = load_experience_library(path)
            self.assertEqual(
                matching_operation_skill(updated, "moka_pot_2")["skill_id"],
                template["skill_id"],
            )

    def test_operation_skill_rejects_partial_or_safe_labeled_cycle(self):
        library = {
            "schema_version": "gs_opd_task_skill_experience_library.v1",
            "source_policy": {"official_safety_demonstration_data_used": False},
            "operation_skill_templates": [],
        }
        template = {
            "skill_id": "bad-cycle", "object_family": "moka_pot",
            "operation": "pick_transport_place_release",
            "phase_sequence": ["approach", "stable_grasp", "lift", "transport",
                               "precise_placement", "release", "withdrawal", "settle"],
            "skill_success": True, "native_object_goal_satisfied": False,
            "released_and_stable": False, "stable_bilateral_grasp_verified": True,
            "native_goal_predicate": ["on", "moka_pot_1", "flat_stove"],
            "physical_actions": 20, "source_execution_reports": ["run.json"],
            "safety_certified": True, "official_safety_demonstrations_used": False,
            "historical_actions_replayed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(library), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "physical_native_goal_cycle"):
                append_verified_search_operation_skill(path, template)

    def test_object_relative_pose_is_relocalized_into_live_object_frame(self):
        object_world = np.eye(4)
        object_world[:3, 3] = [0.5, -0.2, 0.8]
        object_relative = np.eye(4)
        object_relative[:3, 3] = [0.0, 0.08, 0.04]
        migrated = migrate_object_relative_pose(object_world, object_relative)
        np.testing.assert_allclose(migrated[:3, 3], [0.5, -0.12, 0.84])

    def test_failed_historical_route_keeps_grasp_transform_and_generates_distinct_approaches(self):
        base = {
            "candidate_id": "experience__moka_pot:stable_grasp:source",
            "object_to_eef": np.eye(4).tolist(),
            "approach_direction_object": [0.0, 0.0, 1.0],
            "approach_distance_m": 0.06,
            "historical_approach_waypoints_object_xyz_quaternion_wxyz": [[0, 0, 0, 1, 0, 0, 0]],
        }
        variants = alternate_experience_approach_candidates(base)
        self.assertEqual(len(variants), 5)
        self.assertTrue(all(not row["historical_approach_waypoints_object_xyz_quaternion_wxyz"] for row in variants))
        self.assertTrue(all(row["experience_object_relative_grasp_prior_used"] for row in variants))
        self.assertEqual(len({tuple(row["approach_direction_object"]) for row in variants}), 5)
        self.assertTrue(all(row["object_to_eef"] == base["object_to_eef"] for row in variants))

    def test_library_loader_rejects_official_demo_sourced_data(self):
        library = {
            "schema_version": "gs_opd_task_skill_experience_library.v1",
            "source_policy": {"official_safety_demonstration_data_used": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(library), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_policy"):
                load_experience_library(path)

    def test_verified_search_grasp_is_appended_and_prioritized_for_reuse(self):
        library = {
            "schema_version": "gs_opd_task_skill_experience_library.v1",
            "source_policy": {"official_safety_demonstration_data_used": False},
            "grasp_templates": [{"template_id": "old", "object_family": "moka_pot"}],
        }
        template = {
            "template_id": "moka_pot:stable_grasp:L2T2_seed0",
            "object_family": "moka_pot",
            "skill_success": True,
            "stable_bilateral_grasp_verified": True,
            "verified_lift_m": 0.04,
            "safety_certified": False,
            "official_safety_demonstrations_used": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(library), encoding="utf-8")
            self.assertTrue(append_verified_search_grasp_template(path, template))
            self.assertFalse(append_verified_search_grasp_template(path, template))
            updated = load_experience_library(path)
            self.assertEqual(len(updated["grasp_templates"]), 2)
            self.assertEqual(
                matching_grasp_template(updated, "moka_pot_2")["template_id"],
                template["template_id"],
            )

    def test_verified_run_skill_merges_to_master_without_replacing_history(self):
        original = {
            "schema_version": "gs_opd_task_skill_experience_library.v1",
            "source_policy": {"official_safety_demonstration_data_used": False},
            "grasp_templates": [{"template_id": "source_l0", "object_family": "moka_pot"}],
            "operation_skill_templates": [],
            "experience_update_history": [{"older_record": True}],
        }
        learned = {
            "template_id": "obstacle_avoidance:L1T2:verified_grasp",
            "object_family": "moka_pot",
            "object_to_eef": np.eye(4).tolist(),
            "approach_direction_object": [0.0, 0.0, 1.0],
            "approach_distance_m": 0.06,
            "stable_bilateral_grasp_verified": True,
            "verified_lift_m": 0.034,
            "full_native_task_success_in_source_episode": False,
            "safety_certified": False,
            "official_safety_demonstration_data_used": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "master.json"
            run_copy = root / "run.json"
            target.write_text(json.dumps(original), encoding="utf-8")
            run_copy.write_text(json.dumps({
                **original,
                "grasp_templates": [*original["grasp_templates"], learned],
            }), encoding="utf-8")

            merged = merge_verified_search_experiences(target, run_copy)
            self.assertEqual(merged["status"], "merged")
            self.assertEqual(merged["added_grasp_skill_templates"], [learned["template_id"]])
            updated = load_experience_library(target)
            self.assertEqual(updated["grasp_templates"][0]["template_id"], "source_l0")
            self.assertEqual(updated["experience_update_history"][0], {"older_record": True})
            self.assertFalse(updated["grasp_templates"][1]["safety_certified"])
            self.assertFalse(updated["grasp_templates"][1]["source_native_task_success"])
            self.assertEqual(updated["grasp_templates"][1]["cross_scene_physical_validation_count"], 1)

            repeated = merge_verified_search_experiences(target, run_copy)
            self.assertEqual(repeated["status"], "no_new_verified_skills")

    def test_master_merge_rejects_unverified_or_safety_labeled_run_skill(self):
        base = {
            "schema_version": "gs_opd_task_skill_experience_library.v1",
            "source_policy": {"official_safety_demonstration_data_used": False},
            "grasp_templates": [],
        }
        bad = {
            "template_id": "not_verified", "object_family": "moka_pot",
            "stable_bilateral_grasp_verified": True, "verified_lift_m": 0.01,
            "object_to_eef": np.eye(4).tolist(),
            "approach_direction_object": [0.0, 0.0, 1.0],
            "safety_certified": True, "official_safety_demonstration_data_used": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, run_copy = root / "master.json", root / "run.json"
            target.write_text(json.dumps(base), encoding="utf-8")
            run_copy.write_text(json.dumps({**base, "grasp_templates": [bad]}), encoding="utf-8")
            merged = merge_verified_search_experiences(target, run_copy)
            self.assertEqual(merged["status"], "no_new_verified_skills")
            self.assertEqual(merged["rejected_grasp_skill_templates"][0]["template_id"], "not_verified")
            self.assertEqual(load_experience_library(target)["grasp_templates"], [])

    def test_search_grasp_experience_rejects_unverified_or_safe_labeled_candidate(self):
        library = {
            "schema_version": "gs_opd_task_skill_experience_library.v1",
            "source_policy": {"official_safety_demonstration_data_used": False},
            "grasp_templates": [],
        }
        template = {
            "template_id": "candidate",
            "skill_success": False,
            "stable_bilateral_grasp_verified": False,
            "verified_lift_m": 0.0,
            "safety_certified": True,
            "official_safety_demonstrations_used": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            path.write_text(json.dumps(library), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "real_lift"):
                append_verified_search_grasp_template(path, template)


if __name__ == "__main__":
    unittest.main()
