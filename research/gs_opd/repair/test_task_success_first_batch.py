# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression tests for resumable bounded Task-Success-First batches."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.gs_opd.repair.task_success_first_batch import (
    _existing_stage_reports,
    _find_terminal_snapshot,
    _next_distinct_goal,
    _stage_action_count,
    _task_summary,
)


class TaskSuccessFirstBatchTests(unittest.TestCase):
    def test_multigoal_schedule_advances_only_after_native_goal_changes(self):
        first = ["on", "moka_pot_1", "stove"]
        second = ["on", "moka_pot_2", "stove"]
        unchanged = {
            "status": "complete",
            "native_goal_values": [
                {"predicate": first, "value": False},
                {"predicate": second, "value": False},
            ],
        }
        advanced = {
            "status": "complete",
            "native_goal_values": [
                {"predicate": first, "value": True},
                {"predicate": second, "value": False},
            ],
        }
        self.assertIsNone(_next_distinct_goal(unchanged, first))
        self.assertEqual(_next_distinct_goal(advanced, first), second)

    def test_null_expert_success_does_not_hide_student_terminal_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "student.pkl"
            snapshot.write_bytes(b"snapshot")
            report = {
                "segments": [
                    {"terminal_snapshot": str(snapshot)},
                    {"report": "/missing/expert.json"},
                ]
            }
            self.assertEqual(_find_terminal_snapshot(report), str(snapshot))

    def test_resume_loads_failed_stage_and_preserves_native_status(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            stage = task_dir / "stage_00" / "task_success_first_executor_run"
            stage.mkdir(parents=True)
            report_path = stage / "run_report.json"
            report_path.write_text(json.dumps({
                "status": "bounded_expert_search_exhausted",
                "failure_reason": ["no_route"],
                "segments": [{
                    "type": "student_first", "actions_executed": 4,
                    "native_task_status": {
                        "status": "complete", "task_success": False,
                        "native_goal_values": [],
                    },
                }],
            }), encoding="utf-8")
            loaded = _existing_stage_reports(task_dir)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["final_native_task_status"]["task_success"], False)
            self.assertTrue(loaded[0]["batch_stage"]["resumed_from_existing_archive"])
            self.assertEqual(_stage_action_count(loaded[0]), 4)

    def test_summary_counts_successful_grasp_branch_and_ignores_initial_predicates_as_progress(self):
        native_initial = {
            "status": "complete",
            "task_success": False,
            "native_goal_values": [
                {"predicate": ["on", "moka_pot_1", "stove"], "value": False},
                {"predicate": ["on", "moka_pot_2", "stove"], "value": False},
                {"predicate": ["turnon", "stove"], "value": True},
            ],
        }
        native_final = native_initial
        with tempfile.TemporaryDirectory() as directory:
            expert_path = Path(directory) / "grasp_report.json"
            expert_path.write_text(json.dumps({
                "status": "task_success_candidate_created",
                "preflight": {"native_task_status": native_initial},
                "success": {
                    "stable_bilateral_grasp": {"stable_grasp_observed": True},
                },
                "physical_branches": [{
                    "branch_control_actions": 205,
                    "target_lift_delta_m": 0.0345,
                }],
                "artifacts": {},
            }), encoding="utf-8")
            stage = {
                "final_native_task_status": native_final,
                "segments": [{"type": "bounded_expert_grasp_search", "report": str(expert_path)}],
                "batch_stage": {"elapsed_s": 400.0},
            }
            summary = _task_summary(
                "obstacle_avoidance_L2T2",
                {"benchmark": "obstacle_avoidance", "level": 2, "task_id": 2,
                 "instruction": "put both moka pots on the stove"},
                [stage],
            )
        self.assertEqual(summary["expert_actions"], 205)
        self.assertTrue(summary["stable_grasp_verified"])
        self.assertAlmostEqual(summary["maximum_verified_lift_m"], 0.0345)
        self.assertEqual(summary["maximum_progress"], "stable_grasp_verified")
        self.assertFalse(summary["native_task_success"])

    def test_summary_does_not_call_joint_recovery_actions_transport_or_native_progress(self):
        native = {
            "status": "complete",
            "task_success": False,
            "native_goal_values": [
                {"predicate": ["on", "moka_pot_1", "stove"], "value": False},
                {"predicate": ["on", "moka_pot_2", "stove"], "value": False},
                {"predicate": ["turnon", "stove"], "value": True},
            ],
        }
        stages = [{
            "status": "native_goal_route_preflight_failed",
            "final_native_task_status": native,
            "segments": [
                {
                    "type": "verified_live_grasp_snapshot_resume",
                    "native_task_status": native,
                    "grasp_stability": {"stable": True},
                },
                {
                    "type": "expert_native_goal_stage_control",
                    "recorded_actions": 8,
                    "transport_actions": 0,
                    "joint_limit_recovery_actions": 8,
                },
            ],
        }]
        summary = _task_summary(
            "obstacle_avoidance_human_L0T2",
            {"benchmark": "obstacle_avoidance_human", "level": 0, "task_id": 2,
             "instruction": "put both moka pots on the stove"},
            stages,
        )
        self.assertEqual(summary["physical_actions"], 8)
        self.assertTrue(summary["stable_grasp_verified"])
        self.assertFalse(summary["transport_phase_executed"])
        self.assertEqual(summary["maximum_progress"], "stable_grasp_verified")
        self.assertFalse(summary["native_task_success"])


if __name__ == "__main__":
    unittest.main()
