"""Offline tests for bounded S70 clearance probes; no Bridge or MuJoCo env."""

from __future__ import annotations

import unittest

import numpy as np

from research.gs_opd.repair.verifier_guided_repair import RepairConfig
from research.gs_opd.smoke import early_repair_s70_smoke as smoke
from research.gs_opd.smoke.contact_safety_compiler_smoke import SAMPLING_POINT


def _controller_context():
    return {
        "status": "available",
        "world_up": [0.0, 0.0, 1.0],
        "robots": [
            {
                "status": "available",
                "joint_ranges": [[-2.0, 2.0], [-2.0, 2.0]],
                "joint_limited": [True, True],
                "fields": {
                    "control_dim": 6,
                    "input_min": [-1.0] * 6,
                    "input_max": [1.0] * 6,
                    "output_min": [-0.05] * 3 + [-0.5] * 3,
                    "output_max": [0.05] * 3 + [0.5] * 3,
                    "control_delta": True,
                    "use_delta": True,
                    "position_limits": [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]],
                },
            }
        ],
    }


def _candidate(
    identifier: str,
    object_end_x: float,
    eef_end_x: float,
    steps=5,
    *,
    object_end_y: float = 0.0,
    eef_end_y: float = 0.0,
):
    start = {
        "robot_states": [{"eef_pos": [0.0, 0.0, 0.0], "joint_qpos": [0.0, 0.0]}],
        "object_states": {
            "bowl": {"body_id": 23, "root_pos": [0.0, 0.0, 0.0], "joint_qpos": []}
        },
    }
    records = []
    for index in range(steps):
        fraction = (index + 1) / steps
        records.append({
            "step_index": index,
            "reward": 0.0,
            "done": False,
            "state": {
                "robot_states": [{"eef_pos": [eef_end_x * fraction, eef_end_y * fraction, 0.0], "joint_qpos": [0.0, 0.0]}],
                "object_states": {
                    "bowl": {
                        "body_id": 23,
                        "root_pos": [object_end_x * fraction, object_end_y * fraction, 0.0],
                        "joint_qpos": [],
                    }
                },
            },
        })
    return {
        "candidate_id": identifier,
        "status": "ok",
        "requested_steps": 5,
        "executed_steps": steps,
        "start_state": start,
        "steps": records,
    }


def _assess_safe_candidate(baseline, candidate):
    context = _controller_context()
    actions = np.zeros((5, 7), dtype=np.float64)
    specs, _ = smoke.generate_clearance_candidates(
        actions,
        [{"name": "test_detour", "direction_world": [0, 1, 0]}],
        context,
        max_candidates=1,
    )
    spec = {**specs[0], "original_actions": actions}
    baseline = _complete_monitor(baseline)
    candidate = _complete_monitor(candidate)
    endpoint = {
        "status": "complete",
        "main_state_restored": True,
        "endpoint_distance": {"minimum_distance_m": 0.02, "distance_available": True},
    }
    return smoke.assess_candidate(
        spec,
        candidate,
        {"verdict": "safe", "object_contact_checks": []},
        baseline,
        context,
        {"body_id": 23, "object_id": "object_1", "status": "matched_by_iou"},
        smoke.PROBE_CONFIG,
        0.01,
        endpoint,
    )


def _complete_monitor(candidate):
    control_steps = []
    for index, step in enumerate(candidate["steps"]):
        control_steps.append({
            "control_step_index": index,
            "status": "complete",
            "expected_physics_substeps": 25,
            "captured_physics_substeps": 25,
            "unobserved_expected_substeps": 0,
            "contact_samples": {"contact": 0, "no_contact": 25, "unavailable": 0},
            "step_index": step["step_index"],
        })
    candidate["contact_monitoring"] = {
        "status": "complete",
        "sampling_point": SAMPLING_POINT,
        "robot_classification_status": "available",
        "expected_physics_substeps_total": 125,
        "captured_physics_substeps": 125,
        "unobserved_expected_substeps": 0,
        "unavailable_contact_substeps": 0,
        "contact_substeps": 0,
        "no_contact_substeps": 125,
        "control_steps": control_steps,
        "events": [],
    }
    return candidate


class EarlyRepairGeometryTests(unittest.TestCase):
    def test_away_and_lateral_directions_use_bound_nearest_pair_orientation(self):
        distance = {
            "query_complete": True,
            "distance_reliable": True,
            "minimum_distance_m": 0.03,
            "mesh_approximation": False,
            "minimum_pair": {
                "geom_a": {"id": 10, "body_id": 23, "name": "moving"},
                "geom_b": {"id": 20, "body_id": 27, "name": "obstacle"},
                # mj_geomDistance returns the first witness on geom_a, then geom_b.
                "closest_segment_world": [0.0, 0.0, 0.0, 0.03, 0.0, 0.0],
            },
        }
        directions, basis = smoke.derive_clearance_directions(distance, 23, 27, [0, 0, 1])
        self.assertEqual([item["name"] for item in directions], [
            "nearest_geom_away", "nearest_geom_lateral_positive", "nearest_geom_lateral_negative"
        ])
        np.testing.assert_allclose(directions[0]["direction_world"], [-1, 0, 0])
        for direction in directions[1:]:
            self.assertAlmostEqual(np.dot(directions[0]["direction_world"], direction["direction_world"]), 0.0)
        self.assertTrue(basis["collision_evidence_used"] is False)

    def test_candidate_grid_is_bounded_and_preserves_rotation_and_gripper(self):
        context = _controller_context()
        actions = np.zeros((5, 7), dtype=np.float64)
        actions[:, 3:6] = 0.1
        actions[:, 6] = -0.5
        directions = [
            {"name": f"d{index}", "direction_world": vector, "basis_source": "test"}
            for index, vector in enumerate(([1, 0, 0], [0, 1, 0], [0, -1, 0]))
        ]
        specs, rejected = smoke.generate_clearance_candidates(actions, directions, context)
        self.assertEqual(len(specs), 24)
        self.assertLessEqual(len(specs), smoke.MAX_CANDIDATES)
        self.assertEqual(rejected, [])
        self.assertEqual(smoke.AMPLITUDES_M, (0.008, 0.016, 0.024, 0.032))
        self.assertEqual(
            [spec["amplitude_m_per_control_step"] for spec in specs],
            list(smoke.AMPLITUDES_M) * 6,
        )
        self.assertEqual(smoke.PROBE_CONFIG.maximum_amplitude_m_per_step, 0.032)
        self.assertEqual(smoke.WINDOWS, ((1, 3), (2, 2)))
        self.assertEqual(smoke.PROBE_CONFIG.maximum_integrated_detour_m, 0.08)
        self.assertEqual(smoke.GAP_THRESHOLDS_M, (0.005, 0.010, 0.020))
        for spec in specs:
            np.testing.assert_array_equal(spec["actions"][:, 3:], actions[:, 3:])
            self.assertEqual(spec["action_diagnostics"]["window"]["window_anchor_step"], spec["window"]["window_anchor_step"])
            self.assertNotIn("violation_step", spec["action_diagnostics"]["window"])
            self.assertTrue(spec["input_bounds_check"]["status"] == "pass")

    def test_candidate_limit_and_out_of_range_a0_are_fail_closed(self):
        context = _controller_context()
        actions = np.zeros((5, 7), dtype=np.float64)
        directions = [{"name": "away", "direction_world": [1, 0, 0]}]
        specs, _ = smoke.generate_clearance_candidates(actions, directions, context, max_candidates=5)
        self.assertEqual(len(specs), 5)
        self.assertLessEqual(len(specs), 36)
        actions[0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, "active OSC input bounds"):
            smoke.generate_clearance_candidates(actions, directions, context)

    def test_generic_candidate_prefix_is_available_and_default_remains_compatible(self):
        actions = np.zeros((5, 7), dtype=np.float64)
        directions = [{"name": "away", "direction_world": [1, 0, 0]}]
        context = _controller_context()
        default_specs, _ = smoke.generate_clearance_candidates(
            actions, directions, context, max_candidates=1
        )
        generic_specs, _ = smoke.generate_clearance_candidates(
            actions, directions, context, max_candidates=1, candidate_prefix="early"
        )
        self.assertEqual(default_specs[0]["candidate_id"], "early-s70-001")
        self.assertEqual(generic_specs[0]["candidate_id"], "early-001")

    def test_reverse_geom_order_is_normalized_using_body_identity(self):
        distance = {
            "query_complete": True,
            "distance_reliable": True,
            "minimum_distance_m": 0.03,
            "minimum_pair": {
                "geom_a": {"id": 20, "body_id": 27, "name": "obstacle"},
                "geom_b": {"id": 10, "body_id": 23, "name": "moving"},
                "closest_segment_world": [0.03, 0, 0, 0, 0, 0],
            },
        }
        directions, basis = smoke.derive_clearance_directions(distance, 23, 27, [0, 0, 1])
        np.testing.assert_allclose(directions[0]["direction_world"], [-1, 0, 0])
        self.assertEqual(basis["nearest_pair"]["geom_a"]["body_id"], 23)
        self.assertEqual(basis["nearest_pair"]["geom_b"]["body_id"], 27)


class EarlyRepairProgressTests(unittest.TestCase):
    def test_progress_compares_full_five_action_horizon(self):
        actions = np.zeros((5, 7), dtype=np.float64)
        actions[:, 0] = 0.1
        baseline = _candidate("baseline", 0.01, 0.01)
        candidate = _candidate("candidate", 0.008, 0.008)
        result = smoke.evaluate_same_horizon_progress(
            baseline,
            candidate,
            actions,
            _controller_context(),
            {"body_id": 23},
            RepairConfig(minimum_object_motion_m=0.002, required_object_progress_fraction=0.7),
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["aligned_prefix_steps_compared"], 5)
        self.assertEqual(result["comparison_horizon_steps"], 5)
        self.assertEqual(result["basis"], "time_aligned_task_object_state_proxy")
        self.assertTrue(result["eef_displacement_is_not_task_success"])

    def test_incomplete_candidate_is_not_compared_using_empty_prefix(self):
        actions = np.zeros((5, 7), dtype=np.float64)
        result = smoke.evaluate_same_horizon_progress(
            _candidate("baseline", 0.01, 0.01),
            _candidate("short", 0.008, 0.008, steps=4),
            actions,
            _controller_context(),
            {"body_id": 23},
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["aligned_prefix_steps_compared"], None)
        self.assertEqual(result["comparison_horizon_steps"], 5)

    def test_progress_failure_when_target_object_motion_is_reversed(self):
        actions = np.zeros((5, 7), dtype=np.float64)
        actions[:, 0] = 0.1
        result = smoke.evaluate_same_horizon_progress(
            _candidate("baseline", 0.01, 0.01),
            _candidate("backwards", -0.01, 0.008),
            actions,
            _controller_context(),
            {"body_id": 23},
        )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["task_object_progress"]["status"], "fail")

    def test_safe_detour_off_a0_direction_passes_with_preserved_grasp(self):
        baseline = _candidate("baseline", 0.02, 0.02)
        # Only 10% of A0's +X object displacement is retained, while the
        # object and EEF move together around the obstacle in +Y.
        candidate = _candidate(
            "lateral_detour", 0.002, 0.002, object_end_y=0.012, eef_end_y=0.012
        )
        report = _assess_safe_candidate(baseline, candidate)

        self.assertTrue(report["accepted_for_probe"], report["rejection_reasons"])
        task = report["task_progress"]
        self.assertEqual(task["status"], "pass")
        self.assertEqual(task["grasp_evidence"]["status"], "reliable_heuristic")
        self.assertLess(
            task["original_a0_direction_70_percent_diagnostic"]["task_object_root_pos_projection"][
                "candidate_translation_progress_fraction"
            ],
            0.70,
        )
        self.assertFalse(task["original_a0_direction_70_percent_diagnostic"]["used_as_hard_gate"])
        self.assertFalse(task["high_quality_training_label_created"])
        self.assertFalse(report["training_label_valid"])

    def test_lost_task_object_fails_relative_grasp_check(self):
        baseline = _candidate("baseline", 0.02, 0.02)
        candidate = _candidate("object_lost", 0.002, 0.002)
        for index, step in enumerate(candidate["steps"]):
            fraction = (index + 1) / smoke.CHUNK_LENGTH
            step["state"]["object_states"]["bowl"]["root_pos"] = [
                0.002 * fraction,
                0.2 * fraction,
                0.0,
            ]
        report = _assess_safe_candidate(baseline, candidate)

        self.assertFalse(report["accepted_for_probe"])
        self.assertEqual(report["task_progress"]["status"], "fail")
        self.assertEqual(
            report["task_progress"]["reason"],
            "candidate_task_object_eef_relative_position_changed_abnormally",
        )
        self.assertIn(
            "candidate_task_object_eef_relative_position_changed_abnormally",
            report["rejection_reasons"],
        )

    def test_missing_task_object_state_fails_closed(self):
        baseline = _candidate("baseline", 0.02, 0.02)
        candidate = _candidate("missing_object_state", 0.01, 0.01)
        candidate["steps"][2]["state"]["object_states"] = {}
        report = _assess_safe_candidate(baseline, candidate)

        self.assertFalse(report["accepted_for_probe"])
        self.assertEqual(report["task_progress"]["status"], "unavailable")
        self.assertEqual(
            report["task_progress"]["reason"],
            "candidate_bound_task_object_state_3_missing_or_ambiguous",
        )
        self.assertIn(
            "candidate_bound_task_object_state_3_missing_or_ambiguous",
            report["rejection_reasons"],
        )

    def test_existing_positive_task_reward_must_be_preserved(self):
        baseline = _candidate("baseline", 0.02, 0.02)
        candidate = _candidate("candidate", 0.01, 0.01)
        baseline["steps"][0]["reward"] = 1.0
        report = _assess_safe_candidate(baseline, candidate)

        self.assertFalse(report["accepted_for_probe"])
        self.assertEqual(report["task_progress"]["status"], "fail")
        self.assertEqual(
            report["task_progress"]["reason"],
            "candidate_did_not_preserve_existing_positive_task_reward",
        )

    def test_early_termination_without_positive_reward_fails(self):
        baseline = _candidate("baseline", 0.02, 0.02)
        candidate = _candidate("candidate", 0.01, 0.01)
        candidate["steps"][2]["done"] = True
        report = _assess_safe_candidate(baseline, candidate)

        self.assertFalse(report["accepted_for_probe"])
        self.assertEqual(report["task_progress"]["status"], "fail")
        self.assertEqual(
            report["task_progress"]["reason"],
            "candidate_terminated_without_positive_success_reward",
        )

    def test_clearance_counts_are_separate_from_safe_progress_candidates(self):
        candidates = [
            {"endpoint_gap_m": 0.021, "accepted_for_probe": True},
            {"endpoint_gap_m": 0.012, "accepted_for_probe": False},
            {"endpoint_gap_m": 0.006, "accepted_for_probe": True},
            {"endpoint_gap_m": None, "accepted_for_probe": False},
        ]
        counts = smoke.clearance_margin_counts(candidates)
        self.assertEqual(counts["counts_by_margin"]["5mm"], {
            "all_measured_candidates": 3,
            "safe_progress_preserving_probe_candidates": 2,
        })
        self.assertEqual(counts["counts_by_margin"]["10mm"], {
            "all_measured_candidates": 2,
            "safe_progress_preserving_probe_candidates": 1,
        })
        self.assertEqual(counts["counts_by_margin"]["20mm"], {
            "all_measured_candidates": 1,
            "safe_progress_preserving_probe_candidates": 1,
        })
        self.assertTrue(counts["margin_is_exploratory_not_a_safety_proof"])

    def test_safety_unknown_or_incomplete_contact_monitoring_cannot_be_accepted(self):
        actions = np.zeros((5, 7), dtype=np.float64)
        spec = smoke.generate_clearance_candidates(
            actions,
            [{"name": "away", "direction_world": [1, 0, 0]}],
            _controller_context(),
            max_candidates=1,
        )[0][0]
        spec["original_actions"] = actions
        baseline = _candidate("baseline", 0.01, 0.01)
        candidate = _complete_monitor(_candidate(spec["candidate_id"], 0.008, 0.008))
        incomplete = {**candidate, "contact_monitoring": {"status": "incomplete"}}
        endpoint = {"status": "complete", "main_state_restored": True, "endpoint_distance": {"minimum_distance_m": 0.02, "distance_available": True}}
        report = smoke.assess_candidate(
            spec, incomplete, {"verdict": "unknown"}, baseline, _controller_context(),
            {"body_id": 23}, smoke.PROBE_CONFIG, 0.01, endpoint,
        )
        self.assertFalse(report["accepted_for_probe"])
        self.assertEqual(report["physical_safety_verdict"], "unknown")
        self.assertFalse(report["training_label_valid"])
        self.assertFalse(report["final_a_plus"])
        self.assertIn("contact_compiler_did_not_return_safe", report["rejection_reasons"])
        self.assertTrue(any("monitoring" in item for item in report["rejection_reasons"]))

    def test_all_probe_gates_can_pass_without_creating_an_a_plus_training_label(self):
        context = _controller_context()
        actions = np.zeros((5, 7), dtype=np.float64)
        specs, _ = smoke.generate_clearance_candidates(
            actions,
            [{"name": "away", "direction_world": [1, 0, 0]}],
            context,
            max_candidates=1,
        )
        spec = {**specs[0], "original_actions": actions}
        baseline = _complete_monitor(_candidate("baseline", 0.01, 0.01))
        candidate = _complete_monitor(_candidate(spec["candidate_id"], 0.008, 0.008))
        endpoint = {
            "status": "complete",
            "main_state_restored": True,
            "endpoint_distance": {"minimum_distance_m": 0.02, "distance_available": True},
        }
        report = smoke.assess_candidate(
            spec,
            candidate,
            {"verdict": "safe", "reason": "complete monitor found no forbidden contact", "object_contact_checks": []},
            baseline,
            context,
            {"body_id": 23},
            smoke.PROBE_CONFIG,
            0.01,
            endpoint,
        )
        self.assertTrue(report["accepted_for_probe"], report["rejection_reasons"])
        self.assertGreater(report["endpoint_gap_change_vs_a0_m"], smoke.MIN_GAP_IMPROVEMENT_M)
        self.assertEqual(report["task_progress"]["aligned_prefix_steps_compared"], 5)
        self.assertEqual(report["physical_safety_verdict"], "safe")
        self.assertFalse(report["training_label_valid"])
        self.assertFalse(report["final_a_plus"])

    def test_no_endpoint_gap_increase_is_not_a_successful_early_repair_probe(self):
        context = _controller_context()
        actions = np.zeros((5, 7), dtype=np.float64)
        specs, _ = smoke.generate_clearance_candidates(
            actions,
            [{"name": "away", "direction_world": [1, 0, 0]}],
            context,
            max_candidates=1,
        )
        spec = {**specs[0], "original_actions": actions}
        baseline = _complete_monitor(_candidate("baseline", 0.01, 0.01))
        candidate = _complete_monitor(_candidate(spec["candidate_id"], 0.008, 0.008))
        endpoint = {
            "status": "complete",
            "main_state_restored": True,
            "endpoint_distance": {"minimum_distance_m": 0.01, "distance_available": True},
        }
        report = smoke.assess_candidate(
            spec,
            candidate,
            {"verdict": "safe"},
            baseline,
            context,
            {"body_id": 23},
            smoke.PROBE_CONFIG,
            0.01,
            endpoint,
        )
        self.assertFalse(report["accepted_for_probe"])
        self.assertIn("candidate_did_not_increase_endpoint_gap_over_a0", report["rejection_reasons"])

    def test_endpoint_replay_always_attempts_restore_after_step_failure(self):
        class FakeConnection:
            def __init__(self):
                self.request = None
                self.requests = []

            def send(self, payload):
                self.request = payload
                self.requests.append(payload)

            def recv(self):
                if self.request["op"] == "restore":
                    return {"restored_snapshot_id": "s70", "snapshot_state": {"state": [1.0]}}
                if self.request["op"] == "step":
                    return {"error": "injected step error"}
                raise AssertionError(self.request)

        connection = FakeConnection()
        result = smoke.replay_and_measure_endpoint(
            connection,
            "s70",
            {"state": [1.0]},
            np.zeros((5, 7)),
            {"steps": []},
            [1],
            [2],
        )
        self.assertTrue(result["final_restore_attempted"])
        self.assertTrue(result["main_state_restored"])
        self.assertEqual(sum(request["op"] == "restore" for request in connection.requests), 2)
        self.assertIn("injected step error", result["reason"])

    @staticmethod
    def _endpoint_diagnostics(integration_state, eef_x, source):
        return {
            "mujoco_integration_state": integration_state,
            "robot_states": [{"eef_pos": [eef_x, 0.0, 0.0]}],
            "object_states": {"bowl": {"root_pos": [0.2, 0.1, 0.3]}},
            "kinematics_source": source,
        }

    def _run_endpoint_replay(self, branch_integration, sequential_integration, cache_delta):
        snapshot_state = self._endpoint_diagnostics([0.0, 1.0, 2.0], 0.1, "snapshot")
        branch_state = self._endpoint_diagnostics(
            branch_integration, 0.4, "live_mujoco_data_cache"
        )
        sequential_state = self._endpoint_diagnostics(
            sequential_integration,
            0.4 + cache_delta,
            "forwarded_copy_of_mujoco_integration_state",
        )

        class FakeConnection:
            def __init__(self):
                self.request = None
                self.requests = []

            def send(self, payload):
                self.request = payload
                self.requests.append(payload)

            def recv(self):
                operation = self.request["op"]
                if operation == "restore":
                    return {"restored_snapshot_id": "s70", "snapshot_state": snapshot_state}
                if operation == "step":
                    return {"done": False, "snapshot_state": sequential_state}
                if operation == "geom_set_min_distance":
                    return {
                        "status": "complete",
                        "query_complete": True,
                        "distance_reliable": True,
                        "minimum_distance_m": 0.012,
                    }
                raise AssertionError(self.request)

        connection = FakeConnection()
        candidate = {"steps": [{"state": branch_state} for _ in range(smoke.CHUNK_LENGTH)]}
        result = smoke.replay_and_measure_endpoint(
            connection,
            "s70",
            snapshot_state,
            np.zeros((smoke.CHUNK_LENGTH, 7)),
            candidate,
            [10],
            [20],
        )
        return result, connection

    def test_endpoint_distance_uses_integration_state_not_kinematics_cache(self):
        result, connection = self._run_endpoint_replay(
            [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 0.000331
        )

        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["branch_vs_sequential_integration_state"]["matched"])
        self.assertFalse(result["branch_vs_sequential_endpoint"]["matched"])
        self.assertFalse(result["branch_vs_sequential_endpoint"]["use_for_endpoint_distance_gate"])
        self.assertEqual(result["endpoint_distance"]["minimum_distance_m"], 0.012)
        self.assertEqual(
            sum(request["op"] == "geom_set_min_distance" for request in connection.requests), 1
        )
        self.assertTrue(result["main_state_restored"])

    def test_endpoint_distance_rejects_inconsistent_integration_state(self):
        result, connection = self._run_endpoint_replay(
            [0.0, 1.0, 2.0], [0.0, 1.000002, 2.0], 0.0
        )

        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["branch_vs_sequential_integration_state"]["matched"])
        self.assertIn("mujoco_integration_states_differ", result["reason"])
        self.assertIsNone(result["endpoint_distance"])
        self.assertEqual(
            sum(request["op"] == "geom_set_min_distance" for request in connection.requests), 0
        )
        self.assertTrue(result["main_state_restored"])


if __name__ == "__main__":
    unittest.main()
