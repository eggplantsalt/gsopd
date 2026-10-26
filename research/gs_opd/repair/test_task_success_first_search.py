# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for bounded task-success candidate scheduling."""

from __future__ import annotations

import unittest

import numpy as np

from research.gs_opd.repair.task_success_first_search import (
    SearchFailureMemory,
    blocked_candidate_ids_from_report,
    candidate_route_preflight_admitted,
    candidate_matches_failure_memory,
    classify_failure,
    diversify_candidates,
    high_approach_route,
    native_goal_counts,
    object_relative_grasp_targets,
    planner_seed,
    pending_object_goals,
    progress_label,
    route_control_action_budget,
    route_signature,
    robot_contact_escape_direction,
    unilateral_target_contact_sides,
)


class TaskSuccessFirstSearchTests(unittest.TestCase):
    def test_task_search_admits_ik_checked_osc_fallback_without_rrt_safe_path(self):
        self.assertTrue(candidate_route_preflight_admitted(
            "no_safe_path", True, direct_osc_fallback_added=True,
        ))
        self.assertTrue(candidate_route_preflight_admitted(
            "no_safe_path", True, live_contact_probe=True,
        ))
        self.assertFalse(candidate_route_preflight_admitted(
            "no_safe_path", True,
        ))
        self.assertFalse(candidate_route_preflight_admitted(
            "no_safe_path", False, direct_osc_fallback_added=True,
        ))

    def test_unilateral_live_target_contact_identifies_only_single_contact_side(self):
        self.assertEqual(unilateral_target_contact_sides({
            "finger_contacts": {"left": False, "right": True},
        }), ("right",))
        self.assertEqual(unilateral_target_contact_sides({
            "finger_contacts": {"left": True, "right": True},
        }), ())
        self.assertEqual(unilateral_target_contact_sides({
            "finger_contacts": {"left": False, "right": False},
        }), ())

    def test_experience_candidate_with_missing_geom_gets_nonnegative_deterministic_seed(self):
        seed = planner_seed(0, -1, 0)
        self.assertGreaterEqual(seed, 0)
        self.assertEqual(seed, planner_seed(0, -1, 0))

    def test_route_execution_budget_scales_with_path_length_and_remains_bounded(self):
        route = np.asarray([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.33, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])
        self.assertEqual(route_control_action_budget(route), 317)
        self.assertEqual(route_control_action_budget(route, maximum_actions=100), 100)

    def test_generic_high_approach_route_preserves_endpoints_and_adds_vertical_waypoint(self):
        start = np.asarray([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
        target = np.asarray([0.2, 0.1, 0.8, 1.0, 0.0, 0.0, 0.0])
        route = np.asarray(high_approach_route(start, target))
        np.testing.assert_array_equal(route[0], start)
        np.testing.assert_array_equal(route[-1], target)
        np.testing.assert_allclose(route[1, :3], [0.2, 0.1, 0.9])

    def test_migrated_pregrasp_direction_uses_object_frame_not_eef_frame(self):
        root = np.eye(4)
        root[:3, :3] = np.asarray([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        offset = np.eye(4)
        offset[:3, :3] = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        offset[:3, 3] = [0.1, 0.0, 0.0]
        grasp, pregrasp = object_relative_grasp_targets(
            root, offset, [0.0, 1.0, 0.0], 0.05
        )
        np.testing.assert_allclose(grasp[:3, 3], [0.0, 0.1, 0.0])
        np.testing.assert_allclose(pregrasp[:3, 3], [0.05, 0.1, 0.0])
        # The EEF itself is rotated differently; using its frame would move
        # the pregrasp point along world Z instead of world Y.
        self.assertAlmostEqual(float(pregrasp[2, 3]), 0.0, places=12)

    def test_native_goal_scheduler_preserves_completed_object_and_fixture_goals(self):
        status = {
            "status": "complete",
            "task_success": False,
            "native_goal_values": [
                {"predicate": ["In", "alphabet_soup_1", "basket_1_contain_region"], "value": True},
                {"predicate": ["In", "cream_cheese_1", "basket_1_contain_region"], "value": False},
                {"predicate": ["Close", "microwave_1"], "value": False},
            ],
        }
        self.assertEqual(
            pending_object_goals(status), [["In", "cream_cheese_1", "basket_1_contain_region"]]
        )
        self.assertEqual(native_goal_counts(status), (3, 1))

    def test_route_signature_ignores_vertex_density_but_distinguishes_route(self):
        direct = np.asarray([[0, 0, 0, 1, 0, 0, 0], [0.1, 0, 0, 1, 0, 0, 0]])
        dense = np.asarray([
            [0, 0, 0, 1, 0, 0, 0], [0.025, 0, 0, 1, 0, 0, 0],
            [0.05, 0, 0, 1, 0, 0, 0], [0.075, 0, 0, 1, 0, 0, 0],
            [0.1, 0, 0, 1, 0, 0, 0],
        ])
        side = np.asarray([[0, 0, 0, 1, 0, 0, 0], [0.0, 0.1, 0, 1, 0, 0, 0]])
        self.assertEqual(route_signature(direct), route_signature(dense))
        self.assertNotEqual(route_signature(direct), route_signature(side))

    def test_failure_memory_deduplicates_and_switches_after_contact_block(self):
        memory = SearchFailureMemory()
        self.assertTrue(memory.admit("In a region", "grasp_a", "route_0"))
        self.assertFalse(memory.admit("In a region", "grasp_a", "route_0"))
        event = memory.record(
            goal="In a region", candidate_id="grasp_a", route_id="route_0",
            reason="pregrasp_stalled", contact_pairs=["hand:beer"],
        )
        self.assertEqual(event["failure_class"], "contact_blocked")
        self.assertEqual(event["recommended_next_strategy"], "change_approach_side_or_height_and_route")
        summary = memory.summary()
        self.assertEqual(summary["repeated_contact_pairs"][0], {"pair": "hand:beer", "count": 1})

    def test_approach_budget_failure_is_route_stalled_and_imported_from_old_report(self):
        reason = "RuntimeError: migrated_object_relative_approach_action_budget_exhausted"
        self.assertEqual(classify_failure(reason), "route_stalled")
        blocked, imported = blocked_candidate_ids_from_report({
            "search_failure_memory": {"events": [{
                "goal_predicate": '["on","moka_pot_2","flat_stove_1_cook_region"]',
                "candidate_id": "moka_pot:stable_grasp:source",
                "route_id": "rrt:failed-route",
                "failure_reason": reason,
                "failure_class": "other_physical_failure",
            }]},
        }, goal_predicate='["on","moka_pot_2","flat_stove_1_cook_region"]')
        self.assertEqual(blocked, {"moka_pot:stable_grasp:source"})
        self.assertEqual(imported[0]["failure_class"], "route_stalled")
        self.assertEqual(imported[0]["reclassified_from"], "other_physical_failure")
        self.assertTrue(candidate_matches_failure_memory(
            "moka_pot:stable_grasp:source",
            {"experience__moka_pot:stable_grasp:source"},
        ))

    def test_candidate_order_prioritizes_different_approach_families(self):
        rows = [
            {"candidate_id": "a", "jaw_axis_index_in_geom": 0, "approach_axis_index_in_geom": 1,
             "approach_sign": 1, "jaw_sign": 1, "pregrasp_eef_distance_m": 0.01},
            {"candidate_id": "b", "jaw_axis_index_in_geom": 0, "approach_axis_index_in_geom": 1,
             "approach_sign": -1, "jaw_sign": 1, "pregrasp_eef_distance_m": 0.02},
            {"candidate_id": "c", "jaw_axis_index_in_geom": 2, "approach_axis_index_in_geom": 0,
             "approach_sign": 1, "jaw_sign": -1, "pregrasp_eef_distance_m": 0.03},
        ]
        ordered = diversify_candidates(rows)
        self.assertEqual(ordered[0]["candidate_id"], "a")
        self.assertEqual(ordered[1]["candidate_id"], "c")

    def test_alternate_experience_routes_are_preflighted_before_new_geometry(self):
        rows = [
            {"candidate_id": "geometry_near", "jaw_axis_index_in_geom": 0,
             "approach_axis_index_in_geom": 0, "approach_sign": 1, "jaw_sign": 1,
             "pregrasp_eef_distance_m": 0.01},
            {"candidate_id": "exp_side_a", "experience_object_relative_grasp_prior_used": True,
             "approach_route_family": "side_a", "pregrasp_eef_distance_m": 0.2},
            {"candidate_id": "exp_side_b", "experience_object_relative_grasp_prior_used": True,
             "approach_route_family": "side_b", "pregrasp_eef_distance_m": 0.3},
        ]
        ordered = diversify_candidates(rows)
        self.assertEqual([row["candidate_id"] for row in ordered[:2]], ["exp_side_a", "exp_side_b"])

    def test_multiple_ranked_history_templates_diversify_with_missing_timestep_metadata(self):
        rows = [
            {
                "candidate_id": "exp_missing_time",
                "historical_GS_OPD_object_relative_waypoint_prior_used": True,
                "experience_source_global_timestep": None,
                "approach_route_family": "template_a",
                "pregrasp_eef_distance_m": None,
            },
            {
                "candidate_id": "exp_later_time",
                "historical_GS_OPD_object_relative_waypoint_prior_used": True,
                "experience_source_global_timestep": 1053,
                "approach_route_family": "template_b",
                "pregrasp_eef_distance_m": 0.2,
            },
        ]
        ordered = diversify_candidates(rows)
        self.assertEqual(
            [row["candidate_id"] for row in ordered],
            ["exp_missing_time", "exp_later_time"],
        )

    def test_template_match_rank_precedes_route_diversity(self):
        rows = [
            {"candidate_id": "rank0_side_a", "experience_object_relative_grasp_prior_used": True,
             "experience_template_rank": 0, "approach_route_family": "side_a"},
            {"candidate_id": "rank0_side_b", "experience_object_relative_grasp_prior_used": True,
             "experience_template_rank": 0, "approach_route_family": "side_b"},
            {"candidate_id": "rank1", "experience_object_relative_grasp_prior_used": True,
             "experience_template_rank": 1, "approach_route_family": "side_c"},
            {"candidate_id": "geometry", "jaw_axis_index_in_geom": 0,
             "approach_axis_index_in_geom": 0},
        ]
        ordered = diversify_candidates(rows)
        self.assertEqual(
            [row["candidate_id"] for row in ordered],
            ["rank0_side_a", "rank0_side_b", "rank1", "geometry"],
        )

    def test_contact_escape_uses_robot_outward_normal_and_ignores_fixed_pedestal(self):
        contacts = [
            {"geom1": {"name": "white_place_box_1_g2"},
             "geom2": {"name": "gripper0_hand_collision"},
             "body1": {"name": "white_place_box_1_main"},
             "body2": {"name": "gripper0_right_gripper"},
             "normal_geom1_to_geom2_world": [0.0, 1.0, 0.0],
             "signed_contact_distance_m": -0.001,
             "force_and_torque": [0.0, 5.0, 0.0, 0, 0, 0]},
            {"geom1": {"name": "mount0_pedestal_col"},
             "geom2": {"name": "floor"},
             "body1": {"name": "world"}, "body2": {"name": "world"},
             "normal_geom1_to_geom2_world": [0.0, 0.0, 1.0]},
        ]
        result = robot_contact_escape_direction(contacts)
        self.assertEqual(result["blocker_geom_names"], ["white_place_box_1_g2"])
        np.testing.assert_allclose(result["direction_world"], [0, 1, 0])
        self.assertIsNone(robot_contact_escape_direction(
            [{
                "geom1": {"name": "moka_pot_2_g8"},
                "geom2": {"name": "gripper0_finger2_collision"},
                "body1": {"name": "moka_pot_2_main"},
                "body2": {"name": "gripper0_right_gripper"},
                "normal_geom1_to_geom2_world": [1, 0, 0],
            }],
            ignore_object_id="moka_pot_2",
        ))

    def test_partial_progress_never_upgrades_to_task_success(self):
        status = {"status": "complete", "task_success": False}
        self.assertEqual(
            progress_label(status, {"stable_grasp": True}), "stable_grasp_verified"
        )
        self.assertEqual(
            progress_label(status, {"partial_predicate": True, "stable_grasp": True}),
            "stable_grasp_verified",
        )
        self.assertEqual(
            progress_label({"status": "complete", "task_success": True}, {}),
            "native_full_task_success",
        )


if __name__ == "__main__":
    unittest.main()
