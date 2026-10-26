# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression tests for resumable bounded Task-Success-First batches."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from research.gs_opd.repair import task_success_first_batch
from research.gs_opd.repair.task_success_first_batch import (
    _artifacts_for_stage,
    _existing_stage_reports,
    _find_terminal_snapshot,
    _next_distinct_goal,
    _run_stage,
    _stage_action_count,
    _stage_physical_branch_count,
    _task_summary,
)


class TaskSuccessFirstBatchTests(unittest.TestCase):
    def test_stage_action_count_includes_each_attempted_placement_branch(self):
        report = {
            "segments": [{
                "recorded_actions": 78,
                "placement_search": {
                    "candidate_attempts": [
                        {"recorded_actions": 107},
                        {"recorded_actions": 78},
                    ],
                },
            }],
        }
        self.assertEqual(_stage_action_count(report), 185)

    def test_stage_artifact_index_includes_every_placement_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            branch = Path(directory) / "branch"
            branch.mkdir()
            record = branch / "physical_step_records.jsonl.gz"
            with gzip.open(record, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"contacts": [], "contact_monitoring": {}}) + "\n")
            video = branch / "dual_camera_agentview_wrist_20fps.mp4"
            video.write_bytes(b"video")
            artifacts = _artifacts_for_stage({
                "segments": [{"placement_search": {
                    "candidate_attempts": [{"branch_archive": str(branch)}],
                }}],
            })
        self.assertEqual(artifacts, [{"records": str(record), "video": str(video)}])

    def test_first_resumed_stage_passes_exact_runtime_snapshot_to_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "held.pkl"
            snapshot.write_bytes(b"runtime-snapshot")
            task_dir = root / "task"
            task_dir.mkdir()
            captured = {}

            def fake_run(command, **kwargs):
                captured["command"] = command
                stage_root = Path(command[command.index("--output-root") + 1])
                report_path = stage_root / "task_success_first_executor_test" / "run_report.json"
                report_path.parent.mkdir(parents=True)
                report_path.write_text(json.dumps({"final_native_task_status": {"task_success": False}}))
                return Namespace(returncode=0)

            args = Namespace(
                suite="obstacle_avoidance_human", level=0, init_state=0, seed=0,
                camera_resolution=64, base_port=39900, student_actions=1,
                student_handoff_actions=1, maximum_candidates=1, rrt_preflights=1,
                physical_branches=2, rrt_time_budget_s=8.0,
                demonstration_horizon=3600, checkpoint=root / "checkpoint",
                device="cuda:0", experience_library=root / "library.json",
                transport_strategy="native_region_progress_osc", keypose_experience=[],
                resume_from_runtime_snapshot=snapshot,
                recovery_fallback_runtime_snapshot=None,
            )
            with patch.object(task_success_first_batch.subprocess, "run", side_effect=fake_run):
                _run_stage(
                    args, task_id=2, stage_index=0, predicate=["on", "moka_pot_1", "stove"],
                    snapshot_root=root / "snapshot_root", task_dir=task_dir,
                )
            command = captured["command"]
            flag = command.index("--resume-from-runtime-snapshot")
            self.assertEqual(command[flag + 1], str(snapshot.resolve()))

    def test_task_branch_budget_counts_physical_region_branches_across_goals(self):
        report = {
            "segments": [
                {
                    "placement_search": {
                        "physical_branch_count": 1,
                        "candidate_attempts": [{"candidate_id": "region_layout_00"}],
                    },
                },
                {
                    "placement_search": {
                        "physical_branch_count": 1,
                        "candidate_attempts": [{"candidate_id": "region_layout_01"}],
                    },
                },
            ],
        }
        self.assertEqual(_stage_physical_branch_count(report), 2)

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
