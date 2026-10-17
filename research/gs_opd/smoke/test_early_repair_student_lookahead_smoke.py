"""Lightweight unit tests for S70→S75′ Student lookahead selection and gates."""

from __future__ import annotations

import unittest

import numpy as np

from research.gs_opd.smoke import early_repair_student_lookahead_smoke as smoke


def _repair_candidate(candidate_id: str, gap: float, accepted: bool = True):
    return {
        "candidate_id": candidate_id,
        "accepted_for_probe": accepted,
        "endpoint_gap_m": gap,
        "actions": np.full((5, 7), 0.01, dtype=np.float32).tolist(),
    }


class CandidateSelectionTests(unittest.TestCase):
    def test_selects_only_accepted_ten_mm_candidates_and_caps_at_two(self):
        selected, report = smoke.select_repair_candidates({
            "candidate_results": [
                _repair_candidate("below-threshold", 0.009999),
                _repair_candidate("first", 0.010),
                _repair_candidate("second", 0.015),
                _repair_candidate("third", 0.030),
                _repair_candidate("not-accepted", 0.050, accepted=False),
            ]
        })

        self.assertEqual([item["candidate_id"] for item in selected], ["first", "second"])
        self.assertEqual(report["accepted_and_valid_candidate_count"], 3)
        self.assertEqual(report["selected_count"], 2)
        self.assertFalse(report["no_eligible_repair_candidates"])
        self.assertEqual(report["minimum_endpoint_gap_m"], 0.010)

    def test_no_eligible_candidate_is_reported_without_relaxing_threshold(self):
        selected, report = smoke.select_repair_candidates({
            "candidate_results": [
                _repair_candidate("too-close", 0.009),
                _repair_candidate("rejected", 0.040, accepted=False),
            ]
        })

        self.assertEqual(selected, [])
        self.assertTrue(report["no_eligible_repair_candidates"])
        self.assertEqual(report["minimum_endpoint_gap_m"], 0.010)
        self.assertEqual(report["selected_count"], 0)

    def test_malformed_or_non_finite_candidate_actions_are_rejected(self):
        bad_shape = _repair_candidate("bad-shape", 0.020)
        bad_shape["actions"] = np.zeros((4, 7)).tolist()
        non_finite = _repair_candidate("non-finite", 0.030)
        non_finite["actions"][0][0] = float("nan")
        selected, report = smoke.select_repair_candidates({
            "candidate_results": [bad_shape, non_finite]
        })

        self.assertEqual(selected, [])
        self.assertTrue(report["no_eligible_repair_candidates"])
        self.assertEqual(len(report["rejections"]), 2)


class LookaheadGateTests(unittest.TestCase):
    def test_current_chunk_requires_all_safety_progress_and_restore_gates(self):
        passed = smoke.current_chunk_gate(
            branch_restored=True,
            full_chunk=True,
            monitor_error=None,
            compiler_verdict="safe",
            task_progress_status="pass",
            action_bounds_status="pass",
        )
        self.assertTrue(passed["passed"])

        failed = smoke.current_chunk_gate(
            branch_restored=True,
            full_chunk=True,
            monitor_error=None,
            compiler_verdict="unknown",
            task_progress_status="fail",
            action_bounds_status="pass",
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["contact_compiler_safe"])
        self.assertFalse(failed["checks"]["task_progress_pass"])

    def test_student_prediction_must_be_exactly_five_finite_seven_dimensional_actions(self):
        actions = smoke.validate_student_chunk(np.zeros((5, 7), dtype=np.float32))
        self.assertEqual(actions.shape, (5, 7))
        with self.assertRaisesRegex(ValueError, r"shape \(5,7\)"):
            smoke.validate_student_chunk(np.zeros((6, 7), dtype=np.float32))
        invalid = np.zeros((5, 7), dtype=np.float32)
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            smoke.validate_student_chunk(invalid)

    def test_first_violation_includes_relative_and_episode_action_indices(self):
        compiled = {
            "verdict": "violation",
            "object_contact_checks": [{
                "verdict": "violation",
                "evidence": [
                    {"first_control_step": 3, "first_physics_substep": 7, "body_pair": [1, 2]},
                    {"first_control_step": 1, "first_physics_substep": 4, "body_pair": [3, 4]},
                ],
            }],
        }
        result = smoke.first_violation_summary(compiled, segment_start_step=75)

        self.assertEqual(result["status"], "found")
        self.assertEqual(result["segment_control_step_index"], 1)
        self.assertEqual(result["absolute_action_index"], 76)
        self.assertEqual(result["physics_substep_index"], 4)

    def test_violation_without_timed_evidence_is_unavailable_not_no_contact(self):
        result = smoke.first_violation_summary(
            {"verdict": "violation", "object_contact_checks": [{"verdict": "violation", "evidence": []}]},
            segment_start_step=75,
        )
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
