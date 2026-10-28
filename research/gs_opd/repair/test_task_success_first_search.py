# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for bounded task-success candidate scheduling."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import research.gs_opd.repair.task_success_first_grasp_search as grasp_search
from research.gs_opd.repair.task_success_first_grasp_search import (
    MAX_GEOMETRY_CANDIDATES,
    MAX_OFFLINE_GEOMETRY_GENERATION,
    MAX_OFFLINE_GEOMETRY_SCREEN_BATCH,
    _build_bddl_grasp_screen_policy,
    _candidate_generation_limit,
    _candidate_window,
    _combine_candidate_sources,
    _parse_args,
    _verify_cold_start_reset,
    generate_geometry_candidates,
    generate_grasp_roll_variants,
    generate_grasp_coupled_tilt_offset_variants,
    generate_grasp_coupled_pose_grid_variants,
    generate_grasp_surface_offset_variants,
    generate_grasp_tilt_variants,
    load_historical_grasp_pose_prior_candidates,
)

from research.gs_opd.repair.task_success_first_search import (
    SearchFailureMemory,
    blocked_candidate_ids_from_report,
    candidate_route_preflight_admitted,
    candidate_matches_failure_memory,
    classify_failure,
    diversify_candidates,
    generated_geometry_high_approach_admitted,
    high_approach_route,
    inspect_native_goal_coverage,
    native_goal_counts,
    object_relative_grasp_targets,
    planner_seed,
    pending_object_goals,
    progress_label,
    route_control_action_budget,
    route_signature,
    robot_contact_escape_direction,
    target_finger_contact_sides,
    unilateral_target_contact_sides,
)


class TaskSuccessFirstSearchTests(unittest.TestCase):
    def test_generated_fallback_never_changes_experience_baseline_path(self):
        with patch.object(grasp_search, "generated_geometry_high_approach_admitted", return_value=True) as admit:
            self.assertFalse(grasp_search._cold_start_geometry_fallback_allowed(False, {}, {}))
            admit.assert_not_called()
            self.assertTrue(grasp_search._cold_start_geometry_fallback_allowed(True, {}, {}))
            admit.assert_called_once()

    def test_cold_start_requires_exact_fresh_reset_state_and_zero_timestep(self):
        class Bridge:
            def __init__(self, state, timestep):
                self.state = state
                self.timestep = timestep

            def request(self, request):
                self_request = {"op": "observe", "include_snapshot_state": True}
                if request != self_request:
                    raise AssertionError(request)
                return {"timestep": self.timestep, "snapshot_state": {
                    "mujoco_integration_state": self.state,
                }}

        t0 = np.array([0.0, 1.0, -2.0])
        verified = _verify_cold_start_reset(Bridge(t0.copy(), 0), t0)
        self.assertTrue(verified["exact_integration_state_match"])
        with self.assertRaisesRegex(RuntimeError, "snapshot_differs_from_fresh_reset"):
            _verify_cold_start_reset(Bridge([0.0, 1.0, -1.0], 0), t0)
        with self.assertRaisesRegex(RuntimeError, "timestep_is_not_zero"):
            _verify_cold_start_reset(Bridge(t0.copy(), 1), t0)

    def test_cold_start_cli_rejects_historical_priors_and_pinned_candidates(self):
        with patch.object(sys, "argv", [
            "task_success_first_grasp_search.py", "--cold-start",
        ]):
            parsed = _parse_args()
        self.assertTrue(parsed.cold_start)
        self.assertIsNone(parsed.experience_library)
        self.assertEqual(parsed.previous_search_report, [])
        self.assertEqual(parsed.candidate_id, [])

        forbidden_inputs = [
            ["--experience-library", "experience.json"],
            ["--previous-search-report", "previous.json"],
            ["--historical-grasp-pose-priors-json", "successful-grasp.json"],
            ["--candidate-id", "geom113_jaw0_approach2_+1_flip+1"],
            ["--candidate-start-index", "2"],
        ]
        for extra in forbidden_inputs:
            with self.subTest(extra=extra), patch.object(
                sys, "argv", ["task_success_first_grasp_search.py", "--cold-start", *extra]
            ), self.assertRaises(SystemExit):
                _parse_args()

    def test_pinned_candidate_ids_reach_full_geometry_pool_with_experience_templates(self):
        from argparse import Namespace

        args = Namespace(
            candidate_id=["geom46", "geom47"],
            strict_grasp_screen_only=True,
            candidate_start_index=40,
            maximum_candidates=8,
        )
        self.assertEqual(_candidate_generation_limit(args), MAX_OFFLINE_GEOMETRY_GENERATION)

        geometry = [{"candidate_id": f"geom{index}"} for index in range(48)]
        experience = [{"candidate_id": f"experience{index}"} for index in range(9)]
        selected_geometry = _combine_candidate_sources(
            geometry,
            experience,
            generation_limit=48,
            requested_candidate_ids=args.candidate_id,
        )
        self.assertEqual(
            [candidate["candidate_id"] for candidate in selected_geometry],
            ["geom46", "geom47"],
        )
        selected_experience = _combine_candidate_sources(
            geometry,
            experience,
            generation_limit=48,
            requested_candidate_ids=["experience8"],
        )
        self.assertEqual(
            [candidate["candidate_id"] for candidate in selected_experience],
            ["experience8"],
        )

    def test_unpinned_candidate_pool_retains_experience_first_budget(self):
        geometry = [{"candidate_id": f"geom{index}"} for index in range(48)]
        experience = [{"candidate_id": f"experience{index}"} for index in range(9)]
        combined = _combine_candidate_sources(
            geometry,
            experience,
            generation_limit=48,
            requested_candidate_ids=[],
        )
        self.assertEqual(len(combined), 48)
        self.assertEqual(combined[:9], experience)
        self.assertEqual(combined[-1]["candidate_id"], "geom38")

    def test_jaw_axis_tilt_variants_keep_the_pad_center_anchor(self):
        base = {
            "candidate_id": "geom110_jaw0_approach2_+1_flip+1",
            "candidate_source": "compiled_geometry",
            "object_to_eef": np.eye(4).tolist(),
            "approach_direction_object": [0.0, 0.0, 1.0],
            "approach_distance_m": 0.06,
            "pad_center_eef_m_from_active_open_pose": [0.0, 0.0, 0.1],
        }
        variants = generate_grasp_tilt_variants(
            [base],
            target_pose_world=np.eye(4),
            eef_position_world=np.array([0.0, 0.0, 0.2]),
            tilt_degrees=[-20.0, -15.0, -10.0, 10.0, 15.0, 20.0],
        )
        self.assertEqual(len(variants), 6)
        original_pad_center = np.asarray(base["object_to_eef"])[:3, 3] + np.array([0.0, 0.0, 0.1])
        for variant in variants:
            transform = np.asarray(variant["object_to_eef"])
            rotated_pad_center = transform[:3, 3] + transform[:3, :3] @ np.array([0.0, 0.0, 0.1])
            np.testing.assert_allclose(rotated_pad_center, original_pad_center, atol=1e-12)
            np.testing.assert_allclose(transform[:3, 1], [0.0, 1.0, 0.0], atol=1e-12)
            self.assertNotEqual(variant["candidate_id"], base["candidate_id"])
            self.assertTrue(variant["orientation_perturbation"]["pad_center_anchor_preserved"])
        with self.assertRaisesRegex(ValueError, "grasp_tilt_angles"):
            generate_grasp_tilt_variants(
                [base], target_pose_world=np.eye(4), eef_position_world=np.zeros(3),
                tilt_degrees=[0.0],
            )

    def test_approach_axis_roll_variants_keep_approach_and_pad_center(self):
        base = {
            "candidate_id": "geom110_jaw0_approach2_+1_flip+1",
            "candidate_source": "compiled_geometry",
            "object_to_eef": np.eye(4).tolist(),
            "approach_direction_object": [0.0, 0.0, 1.0],
            "approach_distance_m": 0.06,
            "pad_center_eef_m_from_active_open_pose": [0.0, 0.0, 0.1],
        }
        variants = generate_grasp_roll_variants(
            [base],
            target_pose_world=np.eye(4),
            eef_position_world=np.array([0.0, 0.0, 0.2]),
            roll_degrees=[-30.0, -20.0, -10.0, 10.0, 20.0, 30.0],
        )
        self.assertEqual(len(variants), 6)
        for variant in variants:
            transform = np.asarray(variant["object_to_eef"])
            np.testing.assert_allclose(transform[:3, 2], [0.0, 0.0, 1.0], atol=1e-12)
            pad_center = transform[:3, 3] + transform[:3, :3] @ np.array([0.0, 0.0, 0.1])
            np.testing.assert_allclose(pad_center, [0.0, 0.0, 0.1], atol=1e-12)
            self.assertNotEqual(variant["candidate_id"], base["candidate_id"])

    def test_object_z_surface_offsets_move_contact_height_and_preserve_orientation(self):
        base_transform = np.eye(4)
        base_transform[:3, 3] = [0.01, -0.02, 0.03]
        base = {
            "candidate_id": "geom110_jaw0_approach2_+1_flip+1",
            "candidate_source": "compiled_geometry",
            "object_to_eef": base_transform.tolist(),
            "approach_direction_object": [0.0, 0.0, 1.0],
            "approach_distance_m": 0.06,
            "pad_center_eef_m_from_active_open_pose": [0.0, 0.0, 0.1],
        }
        variants = generate_grasp_surface_offset_variants(
            [base],
            target_pose_world=np.eye(4),
            eef_position_world=np.array([0.2, 0.0, 0.2]),
            offset_object_mm=[-15.0, -5.0, 5.0, 15.0],
            offset_object_axis="z",
        )
        self.assertEqual(len(variants), 4)
        for variant, offset in zip(variants, [-15.0, -5.0, 5.0, 15.0]):
            transform = np.asarray(variant["object_to_eef"])
            np.testing.assert_allclose(transform[:3, :3], base_transform[:3, :3], atol=1e-12)
            np.testing.assert_allclose(
                transform[:3, 3] - base_transform[:3, 3],
                [0.0, 0.0, offset / 1000.0],
                atol=1e-12,
            )
            self.assertEqual(variant["surface_contact_offset"]["offset_mm"], offset)
            self.assertEqual(variant["surface_contact_offset"]["object_axis"], "z")

    def test_object_y_contact_patch_offsets_move_pad_center_along_measured_face_axis(self):
        base_transform = np.eye(4)
        base_transform[:3, 3] = [0.01, -0.02, 0.03]
        base = {
            "candidate_id": "geom115_jaw0_approach1_+1_flip+1",
            "candidate_source": "compiled_geometry",
            "object_to_eef": base_transform.tolist(),
            "approach_direction_object": [0.0, 0.0, 1.0],
            "approach_distance_m": 0.06,
            "pad_center_eef_m_from_active_open_pose": [0.0, 0.0, 0.1],
        }
        variants = generate_grasp_surface_offset_variants(
            [base],
            target_pose_world=np.eye(4),
            eef_position_world=np.array([0.2, 0.0, 0.2]),
            offset_object_mm=[4.0, 8.0, 11.0],
            offset_object_axis="y",
        )
        self.assertEqual(len(variants), 3)
        for variant, offset in zip(variants, [4.0, 8.0, 11.0]):
            transform = np.asarray(variant["object_to_eef"])
            np.testing.assert_allclose(transform[:3, :3], base_transform[:3, :3], atol=1e-12)
            np.testing.assert_allclose(
                transform[:3, 3] - base_transform[:3, 3],
                [0.0, offset / 1000.0, 0.0],
                atol=1e-12,
            )
            self.assertEqual(variant["surface_contact_offset"]["object_axis"], "y")
            self.assertIn("__object_y_", variant["candidate_id"])

    def test_coupled_grasp_variants_apply_rotation_then_object_frame_offset(self):
        base = {
            "candidate_id": "geom110_jaw0_approach2_+1_flip+1",
            "candidate_source": "compiled_geometry",
            "object_to_eef": np.eye(4).tolist(),
            "approach_direction_object": [0.0, 0.0, 1.0],
            "approach_distance_m": 0.06,
            "pad_center_eef_m_from_active_open_pose": [0.0, 0.0, 0.1],
        }
        variants = generate_grasp_coupled_tilt_offset_variants(
            [base],
            target_pose_world=np.eye(4),
            eef_position_world=np.array([0.0, 0.0, 0.2]),
            tilt_degrees=[-10.0, 10.0],
            rotation_axis="jaw",
            offset_object_mm=[-4.0, 4.0],
            offset_object_axis="y",
        )
        self.assertEqual(len(variants), 4)
        self.assertEqual(len({item["candidate_id"] for item in variants}), 4)
        for variant in variants:
            transform = np.asarray(variant["object_to_eef"])
            angle = variant["orientation_perturbation"]["angle_degrees"]
            offset = variant["surface_contact_offset"]["offset_mm"]
            expected_rotation = np.asarray(base["object_to_eef"])[:3, :3] @ \
                grasp_search.Rotation.from_euler("y", angle, degrees=True).as_matrix()
            np.testing.assert_allclose(transform[:3, :3], expected_rotation, atol=1e-12)
            pad_center = np.asarray(base["pad_center_eef_m_from_active_open_pose"])
            expected_position = (
                np.asarray(base["object_to_eef"])[:3, 3]
                + np.asarray(base["object_to_eef"])[:3, :3] @ pad_center
                - expected_rotation @ pad_center
                + np.array([0.0, offset / 1000.0, 0.0])
            )
            np.testing.assert_allclose(transform[:3, 3], expected_position, atol=1e-12)
            self.assertEqual(
                variant["source_candidate_id"], base["candidate_id"]
            )
            self.assertEqual(
                variant["coupled_pose_perturbation"]["composition_order"],
                "rotate_about_gripper_axis_then_translate_in_object_frame_axes",
            )
            self.assertIn("__jaw_tilt_", variant["candidate_id"])
            self.assertIn("__object_y_", variant["candidate_id"])

    def test_coupled_pose_grid_composes_two_object_axes_and_records_provenance(self):
        base = {
            "candidate_id": "geom110_jaw0_approach2_+1_flip+1",
            "candidate_source": "compiled_geometry",
            "object_to_eef": np.eye(4).tolist(),
            "approach_direction_object": [0.0, 0.0, 1.0],
            "approach_distance_m": 0.06,
            "pad_center_eef_m_from_active_open_pose": [0.0, 0.0, 0.1],
        }
        variants = generate_grasp_coupled_pose_grid_variants(
            [base],
            target_pose_world=np.eye(4),
            eef_position_world=np.array([0.0, 0.0, 0.2]),
            tilt_degrees=[10.0],
            rotation_axis="approach",
            offsets_by_object_axis={"y": [-4.0, 4.0], "z": [6.0, 10.0]},
        )
        self.assertEqual(len(variants), 4)
        for variant in variants:
            offsets = variant["coupled_surface_contact_offsets"]["object_axes_mm"]
            self.assertEqual(set(offsets), {"y", "z"})
            delta = np.array([0.0, offsets["y"] / 1000.0, offsets["z"] / 1000.0])
            rotation = np.asarray(variant["object_to_eef"])[:3, :3]
            expected_position = np.array([0.0, 0.0, 0.1]) - rotation @ np.array([0.0, 0.0, 0.1]) + delta
            np.testing.assert_allclose(
                np.asarray(variant["object_to_eef"])[:3, 3],
                expected_position,
                atol=1e-12,
            )
            self.assertEqual(variant["source_candidate_id"], base["candidate_id"])
            self.assertIn("__object_y_", variant["candidate_id"])
            self.assertIn("__object_z_", variant["candidate_id"])

    def test_coupled_pose_cli_requires_explicit_screen_only_mode_and_bounds_product(self):
        common = [
            "task_success_first_grasp_search.py",
            "--maximum-candidates", "4",
            "--strict-grasp-screen-only",
            "--strict-screen-reference-report", "prior-report.json",
            "--candidate-id", "base-a",
        ]
        coupled = [
            *common,
            "--candidate-tilt-degrees", "-10",
            "--candidate-tilt-degrees", "10",
            "--candidate-offset-object-y-mm", "-4",
            "--candidate-offset-object-y-mm", "4",
            "--candidate-coupled-perturbations",
        ]
        with patch.object(sys, "argv", coupled):
            parsed = _parse_args()
        self.assertTrue(parsed.candidate_coupled_perturbations)
        with patch.object(sys, "argv", [
            *common,
            "--candidate-tilt-degrees", "-10",
            "--candidate-offset-object-y-mm", "-4",
        ]), self.assertRaises(SystemExit):
            _parse_args()
        over_budget = [
            *common,
            "--maximum-candidates", "24",
            "--candidate-coupled-perturbations",
        ]
        for angle in ("-10", "-5", "5", "10", "15"):
            over_budget.extend(["--candidate-tilt-degrees", angle])
        for offset in ("-8", "-4", "2", "4", "8"):
            over_budget.extend(["--candidate-offset-object-y-mm", offset])
        with patch.object(sys, "argv", over_budget):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_coupled_pose_cli_allows_multiple_offset_axes_with_product_budget(self):
        args = [
            "task_success_first_grasp_search.py",
            "--maximum-candidates", "8",
            "--strict-grasp-screen-only",
            "--strict-screen-reference-report", "prior-report.json",
            "--candidate-id", "base-a",
            "--candidate-roll-degrees", "5",
            "--candidate-offset-object-y-mm", "-4",
            "--candidate-offset-object-y-mm", "4",
            "--candidate-offset-object-z-mm", "6",
            "--candidate-offset-object-z-mm", "10",
            "--candidate-coupled-perturbations",
        ]
        with patch.object(sys, "argv", args):
            parsed = _parse_args()
        self.assertTrue(parsed.candidate_coupled_perturbations)
        multi_axis = [
            *args,
            "--candidate-offset-object-x-mm", "2",
            "--candidate-offset-object-x-mm", "4",
        ]
        with patch.object(sys, "argv", multi_axis):
            parsed = _parse_args()
        self.assertEqual(parsed.candidate_offset_object_x_mm, [2.0, 4.0])
        self.assertTrue(parsed.candidate_coupled_perturbations)
        over_budget = [
            *multi_axis,
            "--maximum-candidates", "24",
        ]
        for offset in ("-7", "-5", "-3", "3", "5", "7", "9"):
            over_budget.extend(["--candidate-offset-object-x-mm", offset])
        with patch.object(sys, "argv", over_budget), self.assertRaises(SystemExit):
            _parse_args()

    def test_surface_offset_screen_budget_accepts_four_bases_and_six_heights(self):
        args = [
            "task_success_first_grasp_search.py",
            "--maximum-candidates", "24",
            "--strict-grasp-screen-only",
            "--strict-screen-reference-report", "prior-report.json",
        ]
        for candidate_id in ("base-a", "base-b", "base-c", "base-d"):
            args.extend(["--candidate-id", candidate_id])
        for offset in ("-15", "-10", "-5", "5", "10", "15"):
            args.extend(["--candidate-offset-object-z-mm", offset])
        with patch.object(sys, "argv", args):
            parsed = _parse_args()
        self.assertEqual(len(parsed.candidate_id) * len(parsed.candidate_offset_object_z_mm), 24)

    def test_surface_contact_patch_budget_accepts_two_bases_and_three_y_offsets(self):
        args = [
            "task_success_first_grasp_search.py",
            "--maximum-candidates", "24",
            "--strict-grasp-screen-only",
            "--strict-screen-reference-report", "prior-report.json",
        ]
        for candidate_id in ("handle-a", "handle-b"):
            args.extend(["--candidate-id", candidate_id])
        for offset in ("4", "8", "11"):
            args.extend(["--candidate-offset-object-y-mm", offset])
        with patch.object(sys, "argv", args):
            parsed = _parse_args()
        self.assertEqual(
            len(parsed.candidate_id) * len(parsed.candidate_offset_object_y_mm), 6
        )

    def test_surface_contact_offset_axes_are_mutually_exclusive(self):
        args = [
            "task_success_first_grasp_search.py",
            "--strict-grasp-screen-only",
            "--strict-screen-reference-report", "prior-report.json",
            "--candidate-id", "handle-a",
            "--candidate-offset-object-y-mm", "8",
            "--candidate-offset-object-z-mm", "5",
        ]
        with patch.object(sys, "argv", args), self.assertRaises(SystemExit):
            _parse_args()

    def test_historical_pose_prior_loader_recomputes_task_matched_physical_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "source.npz"
            metadata = {
                "suite": "obstacle_avoidance", "level": 0, "task_id": 0,
                "initial_state_index": 0, "seed": 20261020, "env_seed": 0,
                "action_dim": 7, "steps": 5,
                "checkpoint_path": "/base", "lora_checkpoint_path": "/adapter",
            }
            np.savez(
                trajectory,
                metadata_json=np.asarray(json.dumps(metadata)),
                step_dones=np.asarray([False, False, False, False, True]),
            )
            result_path = root / "result.json"
            result_path.write_text(json.dumps({"success_once": True, "done": True, "steps": 5}))
            evaluation_path = root / "evaluation.json"
            evaluation_path.write_text(json.dumps({
                "task": {
                    "suite": "obstacle_avoidance", "level": 0,
                    "task_id": 0, "initial_state_index": 0,
                },
                "model": {
                    "base_checkpoint": "/base", "adapter_sha256": "abc",
                    "adapter_loaded_during_rollout": True,
                },
            }))
            contact_path = root / "contacts.json"
            contact_path.write_text(json.dumps({"status": "complete"}))
            pose = {
                "control_step_index": 2,
                "object_root_position_world_m": [0.0, 0.0, 0.03],
                "object_root_quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
                "eef_position_world_m": [0.03, 0.0, 0.02],
                "eef_quaternion_world_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
            start_pose = {
                **pose, "control_step_index": 0,
                "object_root_position_world_m": [0.0, 0.0, 0.0],
                "eef_position_world_m": [0.10, 0.0, 0.03],
            }
            end_pose = {
                **pose, "control_step_index": 1,
                "object_root_position_world_m": [0.0, 0.0, 0.0],
                "eef_position_world_m": [0.03, 0.0, 0.02],
            }
            prior_path = root / "priors.json"
            from research.gs_opd.repair.task_success_first_grasp_search import _sha256_file
            prior_path.write_text(json.dumps({
                "schema_version": "gs_opd_historical_grasp_pose_priors.v1",
                "task": {
                    "benchmark": "obstacle_avoidance", "level": 0, "task_id": 0,
                    "init_state": 0, "seed": 0, "target_object_id": "bowl",
                },
                "safety_qualification_inherited": False,
                "source_episodes": [{
                    "source_episode_id": "source-a",
                    "trajectory_npz": str(trajectory),
                    "trajectory_npz_sha256": _sha256_file(trajectory),
                    "result_json": str(result_path),
                    "result_json_sha256": _sha256_file(result_path),
                    "evaluation_report_json": str(evaluation_path),
                    "evaluation_report_json_sha256": _sha256_file(evaluation_path),
                    "contact_evidence_report": str(contact_path),
                    "contact_evidence_report_sha256": _sha256_file(contact_path),
                    "base_checkpoint_path": "/base",
                    "lora_adapter_path": "/adapter",
                    "adapter_sha256": "abc",
                    "steps": 5,
                    "native_task_success": True,
                    "strict_safety_qualified": False,
                    "safety_qualification_inherited": False,
                }],
                "candidates": [{
                    "candidate_id": "source-a-grasp",
                    "source_episode_id": "source-a",
                    "stable_grasp_window_control_steps": [2, 4],
                    "physical_pose_samples": [pose, {**pose, "control_step_index": 3}, {**pose, "control_step_index": 4}],
                    "bilateral_pad_contact_verified": True,
                    "bilateral_pad_contact_evidence": {
                        "left_pad_first_control_step": 2,
                        "right_pad_first_control_step": 2,
                    },
                    "object_lift_verified": True,
                    "object_lift_evidence": {
                        "threshold_m": 0.03, "initial_object_root_z_m": 0.0,
                    },
                    "approach_direction_evidence": {
                        "start_pose": start_pose, "end_pose": end_pose,
                    },
                    "first_gripper_target_contact_control_step": 2,
                    "approach_direction_object": [-0.9899494937, 0.0, -0.1414213562],
                    "approach_distance_m": 0.0606,
                }],
            }))
            candidates, summary = load_historical_grasp_pose_prior_candidates(
                prior_path,
                benchmark="obstacle_avoidance", level=0, task_id=0, init_state=0, seed=0,
                target_object_id="bowl", target_pose_world=np.eye(4),
                eef_position_world=np.zeros(3), active_pad_center_eef_m=np.zeros(3),
                maximum_candidates=2,
            )
            self.assertEqual(summary["candidate_count"], 1)
            self.assertTrue(summary["task_identity_verified"])
            self.assertFalse(summary["source_safety_qualification_inherited"])
            np.testing.assert_allclose(
                np.asarray(candidates[0]["object_to_eef"])[:3, 3], [0.03, 0.0, -0.01]
            )
            np.testing.assert_allclose(
                candidates[0]["approach_direction_object"],
                [-0.9899494937, 0.0, -0.1414213562], atol=1e-8,
            )
            with self.assertRaisesRegex(ValueError, "task_identity_mismatch"):
                load_historical_grasp_pose_prior_candidates(
                    prior_path,
                    benchmark="obstacle_avoidance", level=0, task_id=1,
                    init_state=0, seed=0, target_object_id="bowl",
                    target_pose_world=np.eye(4), eef_position_world=np.zeros(3),
                    active_pad_center_eef_m=np.zeros(3), maximum_candidates=2,
                )

    def test_offline_geometry_screen_can_extend_candidate_budget_without_expanding_physical_search(self):
        context = {
            "target_root_pose_world": {
                "position_m": [0.0, 0.0, 0.0],
                "rotation": np.eye(3).tolist(),
            },
            "symmetric_pad_center_eef_m": [0.0, 0.0, 0.0],
            "maximum_pad_separation_m": 0.08,
            "eef_pose_world": {"position_m": [0.4, 0.0, 0.4]},
            "target_collision_geometries": [{
                "geom_id": 7,
                "geom_name": "target_box",
                "geom_type_name": "box",
                "size_m": [0.02, 0.03, 0.04],
                "center_root_m": [0.0, 0.0, 0.0],
                "rotation_root": np.eye(3).tolist(),
            }],
            "target_object_id": "target",
            "native_goal_predicate": ["On", "target", "plate"],
            "gripper_pad_separation_m": 0.04,
        }
        context["target_collision_geometries"].append({
            "geom_id": 8,
            "geom_name": "target_box_farther",
            "geom_type_name": "box",
            "size_m": [0.02, 0.03, 0.04],
            "center_root_m": [0.1, 0.0, 0.0],
            "rotation_root": np.eye(3).tolist(),
        })
        physical_candidates, _ = generate_geometry_candidates(
            context, maximum_candidates=MAX_GEOMETRY_CANDIDATES
        )
        offline_candidates, metadata = generate_geometry_candidates(
            context, maximum_candidates=MAX_OFFLINE_GEOMETRY_GENERATION
        )
        self.assertEqual(len(physical_candidates), MAX_GEOMETRY_CANDIDATES)
        self.assertEqual(len(offline_candidates), MAX_OFFLINE_GEOMETRY_GENERATION)
        self.assertEqual(metadata["maximum_candidate_budget"], MAX_OFFLINE_GEOMETRY_GENERATION)
        with self.assertRaisesRegex(ValueError, "maximum_candidates_must_be_in_1..48"):
            generate_geometry_candidates(context, maximum_candidates=49)
        first_window = _candidate_window(offline_candidates, start=0, count=24)
        second_window = _candidate_window(offline_candidates, start=24, count=24)
        self.assertEqual(len(first_window), MAX_OFFLINE_GEOMETRY_SCREEN_BATCH)
        self.assertEqual(len(second_window), MAX_OFFLINE_GEOMETRY_SCREEN_BATCH)
        self.assertTrue(set(item["candidate_id"] for item in first_window).isdisjoint(
            item["candidate_id"] for item in second_window
        ))
        first_diverse = diversify_candidates(first_window)
        second_diverse = diversify_candidates(second_window)
        self.assertTrue(set(item["candidate_id"] for item in first_diverse).isdisjoint(
            item["candidate_id"] for item in second_diverse
        ))

    def test_candidate_budget_above_physical_limit_is_restricted_to_screen_only(self):
        common = ["task_success_first_grasp_search.py", "--maximum-candidates", "24"]
        with patch.object(sys, "argv", common):
            with self.assertRaises(SystemExit):
                _parse_args()
        with patch.object(sys, "argv", [
            *common,
            "--strict-grasp-screen-only",
            "--candidate-start-index", "24",
            "--strict-screen-reference-report", "prior-report.json",
        ]):
            args = _parse_args()
        self.assertEqual(args.maximum_candidates, 24)
        self.assertEqual(args.candidate_start_index, 24)
        self.assertTrue(args.strict_grasp_screen_only)
        with patch.object(sys, "argv", [
            "task_success_first_grasp_search.py",
            "--maximum-candidates", "24",
            "--strict-grasp-screen-only",
            "--strict-screen-reference-report", "prior-report.json",
            "--candidate-id", "geom115_jaw2_approach1_-1_flip-1",
            "--candidate-id", "geom115_jaw0_approach1_-1_flip+1",
        ]):
            args = _parse_args()
        self.assertEqual(len(args.candidate_id), 2)
        with patch.object(sys, "argv", [
            "task_success_first_grasp_search.py",
            "--maximum-candidates", "24",
            "--candidate-start-index", "25",
            "--strict-grasp-screen-only",
            "--strict-screen-reference-report", "prior-report.json",
        ]):
            with self.assertRaises(SystemExit):
                _parse_args()
        tilt_args = [
            "task_success_first_grasp_search.py",
            "--maximum-candidates", "24",
            "--strict-grasp-screen-only",
            "--strict-screen-reference-report", "prior-report.json",
        ]
        for candidate_id in ("base-a", "base-b", "base-c", "base-d"):
            tilt_args.extend(["--candidate-id", candidate_id])
        for angle in ("-20", "-15", "-10", "10", "15", "20"):
            tilt_args.extend(["--candidate-tilt-degrees", angle])
        with patch.object(sys, "argv", tilt_args):
            args = _parse_args()
        self.assertEqual(len(args.candidate_id) * len(args.candidate_tilt_degrees), 24)
        roll_args = [
            "task_success_first_grasp_search.py",
            "--maximum-candidates", "24",
            "--strict-grasp-screen-only",
            "--strict-screen-reference-report", "prior-report.json",
        ]
        for candidate_id in ("base-a", "base-b", "base-c", "base-d"):
            roll_args.extend(["--candidate-id", candidate_id])
        for angle in ("-30", "-20", "-10", "10", "20", "30"):
            roll_args.extend(["--candidate-roll-degrees", angle])
        with patch.object(sys, "argv", roll_args):
            args = _parse_args()
        self.assertEqual(len(args.candidate_id) * len(args.candidate_roll_degrees), 24)
        with patch.object(sys, "argv", [
            "task_success_first_grasp_search.py",
            "--maximum-candidates", "2",
            "--strict-grasp-screen-only",
            "--strict-screen-reference-report", "prior-report.json",
            "--historical-grasp-pose-priors-json", "historical-priors.json",
        ]):
            args = _parse_args()
        self.assertEqual(args.historical_grasp_pose_priors_json, Path("historical-priors.json"))

    def test_completed_screen_only_report_returns_successful_cli_exit(self):
        report = {
            "strict_grasp_screen_only": True,
            "strict_grasp_geometry_screen": "/tmp/strict-screen.json",
            "artifacts": {"candidate_count": 24},
            "status": "no_strict_pad_only_grasp_candidate_in_generated_set",
        }
        with patch.object(grasp_search, "_parse_args", return_value=object()), \
                patch.object(grasp_search, "run", return_value=report), \
                patch("builtins.print"):
            self.assertEqual(grasp_search.main(), 0)

    def test_bddl_screen_policy_uses_declared_forbidden_objects_and_does_not_add_support_exceptions(self):
        bddl = """(define (problem x)
(:fixtures flat_stove_1 - flat_stove white_place_box_1 - place_box main_table - table)
(:objects akita_black_bowl_1 - bowl bottle_of_perfume_1 - perfume plate_1 - plate)
(:obj_of_interest akita_black_bowl_1 plate_1)
(:init (On akita_black_bowl_1 flat_stove_1_cook_region))
(:goal (And (On akita_black_bowl_1 plate_1)))
(:constraints (And (CheckRobotContact bottle_of_perfume_1)
(CheckRobotContact white_place_box_1)
(CheckContact akita_black_bowl_1 bottle_of_perfume_1)
(CheckContact akita_black_bowl_1 white_place_box_1)))
)"""
        names = [
            "akita_black_bowl_1", "bottle_of_perfume_1", "white_place_box_1",
            "plate_1", "flat_stove_1",
        ]
        bindings = {}
        geometries = []
        bodies = []
        for index, name in enumerate(names, start=1):
            body_id, geom_id = index, 100 + index
            bindings[name] = {
                "collision_body_ids": [body_id],
                "collision_body_names": [f"{name}_main"],
                "geom_ids": [geom_id],
            }
            geometries.append({
                "body_id": body_id,
                "geom_id": geom_id,
                "category": "scene",
                "collision_enabled": True,
            })
            bodies.append({"body_id": body_id, "body_name": f"{name}_main"})
        policy, semantics = _build_bddl_grasp_screen_policy(
            {
                "target_object_id": "akita_black_bowl_1",
                "fingertip_pad_geometries": {
                    "left": [{"geom_id": 74}],
                    "right": [{"geom_id": 77}],
                },
            },
            {"bindings": bindings},
            {"geometries": geometries, "bodies": bodies},
            bddl,
        )
        self.assertEqual(
            {item["object_id"] for item in policy["forbidden_objects"]},
            {"bottle_of_perfume_1", "white_place_box_1"},
        )
        self.assertEqual(
            {item["object_id"] for item in policy["allowed_target_objects"]},
            {"akita_black_bowl_1"},
        )
        self.assertIn(
            "flat_stove_1",
            {item["object_id"] for item in policy["other_scene_objects"]},
        )
        self.assertEqual(len(policy["forbidden_object_pairs"]), 2)
        self.assertEqual(policy["allowed_carried_support_contact_geom_ids"], [])
        self.assertEqual(semantics["authorized_pad_target_penetration_m"], -0.002)
        self.assertFalse(semantics["support_contacts_added_as_exceptions"])

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

    def test_generated_geometry_high_approach_requires_cold_geometry_ik_and_unchanged_state(self):
        candidate = {
            "candidate_source": "active_BDDL_target_compiled_collision_geom_and_live_Panda_pad_centers",
        }
        response = {
            "status": "planning_time_budget_exhausted",
            "live_state_unchanged": True,
            "ik_candidates": [{"joint_positions": [0.0] * 7}],
        }
        self.assertTrue(generated_geometry_high_approach_admitted(candidate, response))

        for bad_candidate, bad_response in (
            (candidate, {**response, "status": "offline_feasible"}),
            ({"candidate_source": "historical_experience_library"}, response),
            (candidate, {**response, "ik_candidates": []}),
            (candidate, {**response, "live_state_unchanged": False}),
        ):
            with self.subTest(candidate=bad_candidate, response=bad_response):
                self.assertFalse(generated_geometry_high_approach_admitted(
                    bad_candidate, bad_response,
                ))

    def test_unilateral_live_target_contact_identifies_only_single_contact_side(self):
        self.assertEqual(unilateral_target_contact_sides({
            "finger_contacts": {"left": False, "right": True},
        }), ("right",))
        self.assertEqual(unilateral_target_contact_sides({
            "finger_contacts": {"left": True, "right": True},
        }), ())
        self.assertEqual(target_finger_contact_sides({
            "finger_contacts": {"left": True, "right": True},
        }), ("left", "right"))
        self.assertEqual(target_finger_contact_sides({
            "finger_contacts": {"left": False, "right": True},
        }), ("right",))
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

    def test_native_goal_coverage_exposes_switches_and_unknown_operators(self):
        status = {"status": "complete", "native_goal_values": [
            {"predicate": ["On", "pot", "region"], "value": False},
            {"predicate": ["Turnon", "stove"], "value": False},
            {"predicate": ["Unknown", "fixture"], "value": False},
            {"predicate": ["CheckGripperContactPart", "tool", [2, 3]], "value": False},
            {"predicate": ["Close", "door"], "value": True},
        ]}
        result = inspect_native_goal_coverage(status)
        self.assertEqual([row["unsupported_reason"] for row in result["pending_unsupported_goals"]],
                         ["unrecognized_native_goal_operator", "unrecognized_native_goal_operator"])
        self.assertTrue(result["goals"][1]["operation_supported"])
        self.assertEqual(result["pending_unsupported_goals"][1]["predicate"][2], [2, 3])
        self.assertEqual(result["goals"][0]["predicate"], ["On", "pot", "region"])
        with self.assertRaisesRegex(ValueError, "arity_mismatch"):
            inspect_native_goal_coverage({"status": "complete", "native_goal_values": [
                {"predicate": ["On", "pot"], "value": False},
            ]})
        with self.assertRaisesRegex(ValueError, "explicit_boolean"):
            inspect_native_goal_coverage({"status": "complete", "native_goal_values": [
                {"predicate": ["Close", "door"], "value": None},
            ]})

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
