from __future__ import annotations

import unittest

import numpy as np

from research.gs_opd.repair.verifier_guided_repair import (
    RepairConfig,
    build_repair_actions,
    derive_repair_directions,
    run_verifier_guided_repair,
)


def _controller_context():
    return {
        "status": "available",
        "world_up": [0.0, 0.0, 1.0],
        "robots": [
            {
                "status": "available",
                "joint_ranges": [[-2.9, 2.9]],
                "joint_limited": [True],
                "fields": {
                    "control_dim": 6,
                    "input_min": [-1.0] * 6,
                    "input_max": [1.0] * 6,
                    "output_min": [-2.0] * 6,
                    "output_max": [2.0] * 6,
                    "control_delta": True,
                    "use_delta": True,
                    "position_limits": None,
                },
            }
        ],
    }


def _branch(candidate_id="repair-A0", *, restored=True, stop=False, event=True):
    steps = []
    for index in range(5):
        x = 0.01 * (index + 1)
        steps.append(
            {
                "step_index": index,
                "reward": 0.0,
                "done": False,
                "state": {
                    "robot_states": [{"eef_pos": [x, 0.0, 0.0], "joint_qpos": [0.0]}],
                    "object_states": {},
                },
            }
        )
    if stop:
        for step in steps:
            step["state"]["robot_states"][0]["eef_pos"] = [0.0, 0.0, 0.0]
    events = []
    if event:
        events = [
            {
                "geom_pair": [{"id": 10, "name": "g"}, {"id": 20, "name": "o"}],
                "body_pair": [{"id": 1, "name": "robot"}, {"id": 2, "name": "obstacle"}],
                "first_control_step": 2,
                "first_physics_substep": 4,
                "contact_position_status": "available",
                "first_contact_position_world": [0.02, 0.0, 0.1],
                "contact_normal_status": "available",
                "first_contact_normal_geom1_to_geom2_world": [1.0, 0.0, 0.0],
                "contact_normal_geom_ids": [10, 20],
            }
        ]
    candidate = {
        "candidate_id": candidate_id,
        "status": "ok",
        "requested_steps": 5,
        "executed_steps": 5,
        "start_state": {
            "robot_states": [{"eef_pos": [0.0, 0.0, 0.0], "joint_qpos": [0.0]}],
            "object_states": {},
        },
        "steps": steps,
        "contact_monitoring": {
            "status": "complete",
            "sampling_point": "immediately_after_each_successful_MjSim.step",
            "robot_classification_status": "available",
            "events": events,
        },
    }
    return {"main_state_restored": restored, "candidates": [candidate]}


def _compiled(verdict="violation", candidate_id="repair-A0"):
    evidence = []
    if verdict == "violation":
        evidence = [
            {
                "geom_pair": [{"id": 10, "name": "g"}, {"id": 20, "name": "o"}],
                "body_pair": [{"id": 1, "name": "robot"}, {"id": 2, "name": "obstacle"}],
                "first_control_step": 2,
                "first_physics_substep": 4,
                "max_force_norm": 1.2,
            }
        ]
    return {
        "status": verdict,
        "gripper_geom_ids": [10],
        "candidates": [{"candidate_id": candidate_id, "verdict": verdict, "evidence": evidence}],
    }


def _safe_result(candidate_id, actions, *, stop=False, restored=True):
    branch = _branch(candidate_id, restored=restored, stop=stop, event=False)
    return {
        "candidate_id": candidate_id,
        "branch_result": branch,
        "compiled_result": _compiled("safe", candidate_id),
    }


class VerifierGuidedRepairTests(unittest.TestCase):
    def setUp(self):
        self.actions = np.zeros((5, 7), dtype=np.float64)
        self.actions[:, 0] = 0.1
        self.actions[:, 6] = -1.0
        self.context = _controller_context()

    def _run(self, baseline_verdict="violation", evaluator=None, *, baseline_restored=True):
        calls = []

        def evaluate(specifications):
            calls.append([item["metadata"]["amplitude_m_per_control_step"] for item in specifications])
            if evaluator is not None:
                return evaluator(specifications)
            return [_safe_result(item["candidate_id"], item["actions"]) for item in specifications]

        report = run_verifier_guided_repair(
            self.actions,
            _branch(restored=baseline_restored),
            _compiled(baseline_verdict),
            self.context,
            evaluate,
            config=RepairConfig(max_candidate_evaluations=96),
        )
        return report, calls

    def test_original_safe_skips_search(self):
        report, calls = self._run("safe")
        self.assertEqual(report["status"], "skipped_original_safe")
        self.assertEqual(calls, [])
        self.assertIsNone(report["repaired_action_chunk_A_plus"])

    def test_unknown_skips_search(self):
        report, calls = self._run("unknown")
        self.assertEqual(report["status"], "skipped_unknown")
        self.assertEqual(calls, [])

    def test_restore_failure_aborts_before_search(self):
        report, calls = self._run(baseline_restored=False)
        self.assertEqual(report["status"], "aborted_main_state_restore_failed")
        self.assertEqual(calls, [])

    def test_candidate_restore_failure_aborts_after_first_round(self):
        call_count = 0

        def evaluator(specifications):
            nonlocal call_count
            call_count += 1
            return [
                {
                    "candidate_id": item["candidate_id"],
                    "branch_result": _branch(item["candidate_id"], restored=False),
                    "compiled_result": _compiled("unknown", item["candidate_id"]),
                }
                for item in specifications
            ]

        report, _ = self._run(evaluator=evaluator)
        self.assertEqual(report["status"], "aborted_main_state_restore_failed")
        self.assertEqual(call_count, 1)
        self.assertIsNone(report["repaired_action_chunk_A_plus"])

    def test_contact_evidence_drives_direction_and_multi_step_window(self):
        branch = _branch()
        from research.gs_opd.repair.verifier_guided_repair import locate_first_violation

        violation, error = locate_first_violation(branch, _compiled())
        self.assertIsNone(error)
        self.assertEqual(violation["control_step"], 2)
        self.assertEqual(violation["physics_substep"], 4)
        self.assertEqual(violation["normal_away_from_obstacle"], [-1.0, -0.0, -0.0])
        directions = derive_repair_directions(violation, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        names = {item["name"] for item in directions}
        self.assertIn("contact_normal_away", names)
        self.assertIn("tangent_side_a", names)
        self.assertIn("tangent_side_b", names)
        self.assertIn("gravity_derived_lift", names)

        repaired, details = build_repair_actions(
            self.actions,
            [0.0, 1.0, 0.0],
            0.01,
            2,
            2,
            2,
            self.context,
        )
        self.assertTrue(details["valid"])
        self.assertEqual(details["window"], {"start_step": 0, "violation_step": 2, "end_step": 4})
        changed_steps = np.flatnonzero(np.linalg.norm(repaired[:, :3] - self.actions[:, :3], axis=1) > 0)
        self.assertGreaterEqual(len(changed_steps), 3)
        np.testing.assert_array_equal(repaired[:, 3:], self.actions[:, 3:])

    def test_expands_then_refines_amplitude_after_safe_candidate(self):
        def evaluator(specifications):
            results = []
            for item in specifications:
                safe = item["metadata"]["amplitude_m_per_control_step"] >= 0.017
                results.append(
                    {
                        "candidate_id": item["candidate_id"],
                        "branch_result": _branch(item["candidate_id"], event=not safe),
                        "compiled_result": _compiled("safe" if safe else "violation", item["candidate_id"]),
                    }
                )
            return results

        report, calls = self._run(evaluator=evaluator)
        self.assertEqual(report["status"], "repaired_candidate_found")
        self.assertIsNotNone(report["repaired_action_chunk_A_plus"])
        self.assertGreaterEqual(len(report["search"]["amplitude_rounds"]), 3)
        self.assertGreaterEqual(len(report["search"]["refinement_rounds"]), 1)
        selected = report["selected_candidate"]
        self.assertGreaterEqual(selected["amplitude_m_per_control_step"], 0.017)
        self.assertLess(selected["amplitude_m_per_control_step"], 0.021)
        self.assertTrue(calls)

    def test_no_safe_candidate_returns_no_a_plus(self):
        def evaluator(specifications):
            return [
                {
                    "candidate_id": item["candidate_id"],
                    "branch_result": _branch(item["candidate_id"], event=True),
                    "compiled_result": _compiled("violation", item["candidate_id"]),
                }
                for item in specifications
            ]

        report, _ = self._run(evaluator=evaluator)
        self.assertEqual(report["status"], "no_safe_progress_preserving_candidate")
        self.assertIsNone(report["repaired_action_chunk_A_plus"])

    def test_task_progress_failure_rejects_safe_but_stationary_candidate(self):
        def evaluator(specifications):
            return [
                _safe_result(item["candidate_id"], item["actions"], stop=True)
                for item in specifications
            ]

        report, _ = self._run(evaluator=evaluator)
        self.assertEqual(report["status"], "no_safe_progress_preserving_candidate")
        self.assertIsNone(report["repaired_action_chunk_A_plus"])
        self.assertTrue(
            any(
                item.get("reason") == "no_minimum_task_progress_evidence"
                for item in report["search_attempts"]
            )
        )

    def test_bound_task_object_state_can_supply_progress_evidence(self):
        def add_task_object(branch):
            candidate = branch["candidates"][0]
            candidate["start_state"]["object_states"] = {
                "objects_dict:task_target": {
                    "body_id": 77,
                    "root_pos": [0.0, 0.0, 0.0],
                    "joint_qpos": [],
                }
            }
            for index, step in enumerate(candidate["steps"]):
                step["state"]["object_states"] = {
                    "objects_dict:task_target": {
                        "body_id": 77,
                        "root_pos": [0.01 * (index + 1), 0.0, 0.0],
                        "joint_qpos": [],
                    }
                }
            for step in candidate["steps"]:
                step["state"]["robot_states"][0]["eef_pos"] = [0.0, 0.0, 0.0]
            return branch

        baseline = add_task_object(_branch())

        def evaluator(specifications):
            results = []
            for item in specifications:
                candidate_branch = add_task_object(_branch(item["candidate_id"], event=False))
                results.append({
                    "candidate_id": item["candidate_id"],
                    "branch_result": candidate_branch,
                    "compiled_result": _compiled("safe", item["candidate_id"]),
                })
            return results

        report = run_verifier_guided_repair(
            self.actions,
            baseline,
            _compiled(),
            self.context,
            evaluator,
            task_object_binding={
                "status": "matched_by_iou", "role": "target_object", "body_id": 77,
            },
        )
        self.assertEqual(report["status"], "repaired_candidate_found")
        progress = report["selected_candidate"]["task_progress"]
        self.assertEqual(progress["basis"], "task_object_state_proxy")
        self.assertEqual(progress["task_object_progress"]["status"], "pass")

    def test_missing_contact_normal_cannot_create_repair(self):
        branch = _branch()
        event = branch["candidates"][0]["contact_monitoring"]["events"][0]
        event["contact_normal_status"] = "unavailable"
        report = run_verifier_guided_repair(
            self.actions,
            branch,
            _compiled(),
            self.context,
            lambda _specs: self.fail("must not search without contact normal"),
        )
        self.assertEqual(report["status"], "skipped_unknown")
        self.assertIsNone(report["repaired_action_chunk_A_plus"])


if __name__ == "__main__":
    unittest.main()
