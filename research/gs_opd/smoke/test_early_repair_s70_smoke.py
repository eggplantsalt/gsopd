"""Offline tests for bounded S70 clearance probes; no Bridge or MuJoCo env."""

from __future__ import annotations

from dataclasses import replace
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


def _with_task_object_offset(candidate, offset):
    states = [candidate["start_state"], *[step["state"] for step in candidate["steps"]]]
    for state in states:
        obj = state["object_states"]["bowl"]
        obj["root_pos"] = (np.asarray(obj["root_pos"]) + np.asarray(offset)).tolist()
    return candidate


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


class BrakeAndSteerTests(unittest.TestCase):
    def setUp(self):
        self.context = _controller_context()
        self.actions = np.zeros((5, 7), dtype=np.float64)
        self.actions[:, 0] = -0.1
        self.actions[:, 3:6] = 0.1
        self.actions[:, 6] = -0.5
        self.directions, _ = smoke.derive_clearance_directions(
            {
                "query_complete": True,
                "distance_reliable": True,
                "minimum_distance_m": 0.03,
                "minimum_pair": {
                    "geom_a": {"id": 10, "body_id": 23},
                    "geom_b": {"id": 20, "body_id": 27},
                    "closest_segment_world": [0.03, 0, 0, 0, 0, 0],
                },
            },
            23, 27, self.context["world_up"],
        )

    def _generate(self, *, config=smoke.PROBE_CONFIG, max_candidates=36):
        return smoke.generate_clearance_candidates(
            self.actions, self.directions, self.context, config,
            include_brake_and_steer=True, max_candidates=max_candidates,
        )

    @staticmethod
    def _new(items):
        return [item for item in items if item["action_diagnostics"].get("candidate_family") == "brake_and_steer"]

    def test_appends_twelve_candidates_without_changing_the_original_grid(self):
        original, old_rejected = smoke.generate_clearance_candidates(
            self.actions, self.directions, self.context
        )
        combined, rejected = self._generate()
        self.assertEqual(len(original), 24)
        self.assertEqual(len(combined), 36)
        self.assertEqual(old_rejected, [])
        self.assertEqual(rejected, [])
        for old, extended in zip(original, combined[:24]):
            np.testing.assert_array_equal(old["actions"], extended["actions"])
            self.assertEqual(
                {k: v for k, v in old.items() if k != "actions"},
                {k: v for k, v in extended.items() if k != "actions"},
            )
        extra = self._new(combined)
        self.assertEqual(len(extra), 12)
        self.assertEqual(len({item["candidate_id"] for item in combined}), 36)
        self.assertEqual([item["direction"]["name"] for item in extra].count("brake_only"), 2)
        for anchor in (1, 2):
            self.assertEqual(sum(item["window"]["window_anchor_step"] == anchor for item in extra), 6)

    def test_brake_cancels_approach_with_the_existing_smooth_envelope(self):
        specs, _ = self._generate()
        pure_brakes = [item for item in self._new(specs) if item["direction"]["name"] == "brake_only"]
        for item in pure_brakes:
            diagnostics = item["action_diagnostics"]
            envelope = np.asarray(diagnostics["envelope"])
            before = np.asarray(diagnostics["original_xyz_displacement_world_m"])
            correction = np.asarray(diagnostics["physical_xyz_bias_by_step_m"])
            after = before + correction
            np.testing.assert_allclose(after[:, 0], before[:, 0] * (1.0 - envelope), atol=1e-15)
            self.assertAlmostEqual(after[item["window"]["window_anchor_step"], 0], 0.0)
            np.testing.assert_array_equal(item["actions"][:, 3:], self.actions[:, 3:])
            self.assertGreater(len(diagnostics["active_action_steps"]), 1)

    def test_already_away_motion_is_not_cancelled(self):
        self.actions[:, 0] = 0.1
        specs, rejected = self._generate()
        for item in self._new(specs):
            np.testing.assert_array_equal(item["actions"][:, 0], self.actions[:, 0])
            np.testing.assert_array_equal(item["action_diagnostics"]["braking_xyz_bias_by_step_m"], np.zeros((5, 3)))
        brake_rejections = self._new(rejected)
        self.assertEqual(len(brake_rejections), 2)
        self.assertTrue(all(item["reason"] == "no_brake_or_steer_correction_in_window" for item in brake_rejections))

    def test_brake_lateral_and_up_diagnostics_pass_the_real_action_constraints(self):
        specs, _ = self._generate()
        controller, error = smoke._controller_fields(self.context)
        self.assertIsNone(error)
        candidate = _candidate("physical_state_fixture", 0.01, 0.01)
        for item in self._new(specs):
            diagnostics = item["action_diagnostics"]
            bias = np.asarray(diagnostics["physical_xyz_bias_by_step_m"])
            np.testing.assert_allclose(
                bias,
                (item["actions"][:, :3] - self.actions[:, :3]) * controller["gain_m_per_env_action"],
            )
            np.testing.assert_allclose(
                bias,
                np.asarray(diagnostics["braking_xyz_bias_by_step_m"])
                + np.asarray(diagnostics["steering_xyz_bias_by_step_m"]),
                atol=1e-15,
            )
            constraints = smoke._candidate_constraints(
                self.actions, item["actions"], diagnostics, candidate, controller, smoke.PROBE_CONFIG
            )
            self.assertEqual(constraints["status"], "pass", constraints)
            np.testing.assert_array_equal(item["actions"][:, 3:], self.actions[:, 3:])
            peak = item["window"]["window_anchor_step"]
            mode = item["direction"]["name"]
            if mode == "brake_lateral_positive":
                self.assertGreater(bias[peak, 1], 0.0)
            elif mode == "brake_lateral_negative":
                self.assertLess(bias[peak, 1], 0.0)
            elif mode == "brake_up":
                self.assertGreater(bias[peak, 2], 0.0)
            if mode != "brake_only":
                self.assertGreater(
                    diagnostics["amplitude_m_per_control_step"],
                    diagnostics["steering_amplitude_m_per_control_step"],
                )

    def test_out_of_bounds_candidates_are_rejected_without_clipping(self):
        self.actions[:, 1:3] = 0.95
        specs, rejected = self._generate()
        new_rejected = self._new(rejected)
        saturated = [item for item in new_rejected if item["reason"] == "candidate_exceeds_active_osc_input_bounds"]
        self.assertEqual(len(saturated), 6)  # two positive lateral amplitudes + up, for each window
        for item in saturated:
            diagnostics = item["action_diagnostics"]
            self.assertFalse(diagnostics["controller_input_clipping_applied"])
            self.assertGreater(np.asarray(diagnostics["candidate_xyz_actions"]).max(), 1.0)
            self.assertEqual(item["input_bounds_check"]["status"], "fail")
        self.assertTrue(all(item["input_bounds_check"]["status"] == "pass" for item in self._new(specs)))

    def test_total_braking_and_steering_respects_each_existing_correction_limit(self):
        cases = (
            (replace(smoke.PROBE_CONFIG, maximum_amplitude_m_per_step=0.004), "per_step_xyz_repair_exceeds_bound"),
            (replace(smoke.PROBE_CONFIG, maximum_integrated_detour_m=0.001), "integrated_xyz_repair_exceeds_bound"),
            (replace(smoke.PROBE_CONFIG, maximum_action_correction_slew_m=0.001), "xyz_repair_is_not_smooth"),
        )
        for config, reason in cases:
            with self.subTest(reason=reason):
                specs, rejected = self._generate(config=config)
                self.assertEqual(self._new(specs), [])
                new_rejected = self._new(rejected)
                self.assertEqual(len(new_rejected), 12)
                self.assertTrue(all(reason in item["action_diagnostics"]["constraint_violations"] for item in new_rejected))

    def test_large_braking_component_alone_can_exceed_the_unchanged_limit(self):
        self.actions[:, 0] = -0.9
        _, rejected = self._generate()
        brakes = [item for item in self._new(rejected) if item["direction"]["name"] == "brake_only"]
        self.assertEqual(len(brakes), 2)
        for item in brakes:
            self.assertGreater(item["amplitude_m_per_control_step"], smoke.PROBE_CONFIG.maximum_amplitude_m_per_step)
            self.assertIn("per_step_xyz_repair_exceeds_bound", item["action_diagnostics"]["constraint_violations"])

    def test_controller_affine_offset_is_used_to_identify_approaching_motion(self):
        fields = self.context["robots"][0]["fields"]
        fields["input_min"][0] = -2.0
        fields["input_max"][0] = 0.0
        fields["output_min"][0] = -0.04
        fields["output_max"][0] = 0.06
        self.actions[:, 0] = -1.4
        specs, _ = self._generate()
        brakes = [item for item in self._new(specs) if item["direction"]["name"] == "brake_only"]
        self.assertEqual(len(brakes), 2)
        for item in brakes:
            diagnostics = item["action_diagnostics"]
            np.testing.assert_allclose(np.asarray(diagnostics["original_xyz_displacement_world_m"])[:, 0], -0.01)
            peak = item["window"]["window_anchor_step"]
            corrected_world = (item["actions"][peak, 0] + 1.0) * 0.05 + 0.01
            self.assertAlmostEqual(corrected_world, 0.0)

    def test_budget_includes_rejected_attempts_and_never_exceeds_36(self):
        specs, rejected = self._generate(max_candidates=30)
        self.assertEqual(len(specs), 30)
        self.assertEqual(len(self._new(specs)), 6)
        self.assertEqual(rejected, [])
        self.actions[:, :3] = 0.0
        specs, rejected = self._generate()
        self.assertEqual(len(specs), 34)
        self.assertEqual(len(self._new(specs)), 10)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(len(specs) + len(rejected), 36)
        with self.assertRaisesRegex(ValueError, "max_candidates"):
            self._generate(max_candidates=37)


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
        self.assertTrue(task["evaluation_complete"])
        self.assertEqual(task["grasp_evidence"]["status"], "kinematic_consistency_only")
        self.assertFalse(task["physical_grasp_confirmed"])
        self.assertFalse(task["task_success_confirmed"])
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
        self.assertTrue(report["task_progress"]["evaluation_complete"])
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
        self.assertFalse(report["task_progress"]["evaluation_complete"])
        self.assertEqual(
            report["task_progress"]["reason"],
            "candidate_bound_task_object_state_3_missing_or_ambiguous",
        )
        self.assertIn(
            "candidate_bound_task_object_state_3_missing_or_ambiguous",
            report["rejection_reasons"],
        )

    def test_stationary_relative_state_can_pass_without_claiming_a_grasp(self):
        # The same positions could also describe an independently supported
        # object near a stationary hand; only kinematic continuity is tested.
        baseline = _with_task_object_offset(_candidate("baseline", 0.0, 0.0), [0.03, -0.02, -0.08])
        candidate = _with_task_object_offset(_candidate("stationary", 0.0, 0.0), [0.03, -0.02, -0.08])
        task = smoke.evaluate_early_repair_task_feasibility(baseline, candidate, {"body_id": 23})

        self.assertEqual(task["status"], "pass")
        self.assertTrue(task["evaluation_complete"])
        self.assertEqual(task["grasp_evidence"]["baseline_eef_path_length_m"], 0.0)
        self.assertEqual(task["grasp_evidence"]["candidate_max_relative_position_change_m"], 0.0)
        self.assertFalse(task["grasp_evidence"]["physical_grasp_confirmed"])
        self.assertFalse(task["physical_grasp_confirmed"])
        self.assertFalse(task["task_success_confirmed"])
        self.assertFalse(task["high_quality_training_label_created"])

    def test_low_speed_relative_state_has_no_minimum_motion_gate(self):
        for object_motion, eef_motion in ((0.0034822, 0.00469311), (0.0005, 0.0004)):
            with self.subTest(object_motion=object_motion, eef_motion=eef_motion):
                baseline = _with_task_object_offset(
                    _candidate("baseline", object_motion, eef_motion), [0.0, 0.0, -0.08]
                )
                candidate = _with_task_object_offset(
                    _candidate("low_speed", object_motion, eef_motion), [0.0, 0.0, -0.08]
                )
                task = smoke.evaluate_early_repair_task_feasibility(baseline, candidate, {"body_id": 23})
                self.assertEqual(task["status"], "pass")
                self.assertTrue(task["evaluation_complete"])
                self.assertLess(task["grasp_evidence"]["baseline_eef_path_length_m"], 0.005)
                self.assertFalse(task["physical_grasp_confirmed"])
                self.assertFalse(task["high_quality_training_label_created"])

    def test_unchanged_scalar_distance_cannot_hide_3d_relative_displacement(self):
        baseline = _with_task_object_offset(_candidate("baseline", 0.02, 0.02), [0.08, 0.0, 0.0])
        candidate = _candidate("relative_side_slip", 0.02, 0.02)
        states = [candidate["start_state"], *[step["state"] for step in candidate["steps"]]]
        for state, angle in zip(states, np.linspace(0.0, np.pi / 2.0, 6)):
            eef = np.asarray(state["robot_states"][0]["eef_pos"])
            state["object_states"]["bowl"]["root_pos"] = (
                eef + [0.08 * np.cos(angle), 0.08 * np.sin(angle), 0.0]
            ).tolist()
        task = smoke.evaluate_early_repair_task_feasibility(baseline, candidate, {"body_id": 23})

        self.assertEqual(task["status"], "fail")
        self.assertEqual(task["reason"], "candidate_task_object_eef_relative_position_changed_abnormally")
        self.assertLess(task["grasp_evidence"]["candidate_max_aligned_relative_distance_error_m"], 1e-12)
        self.assertGreater(task["grasp_evidence"]["candidate_max_relative_position_change_m"], 0.05)

    def test_inconsistent_baseline_relative_positions_are_completed_but_unavailable(self):
        baseline = _candidate("baseline", 0.1, 0.02)
        candidate = _candidate("candidate", 0.01, 0.01)
        report = _assess_safe_candidate(baseline, candidate)

        self.assertFalse(report["accepted_for_probe"])
        self.assertEqual(report["task_progress"]["status"], "unavailable")
        self.assertTrue(report["task_progress"]["evaluation_complete"])
        self.assertEqual(report["task_progress"]["reason"], "baseline_grasp_evidence_unreliable")
        self.assertFalse(report["training_label_valid"])

    def test_missing_reward_is_incomplete_even_with_consistent_relative_positions(self):
        baseline = _candidate("baseline", 0.003482, 0.004693)
        candidate = _candidate("candidate", 0.01, 0.01)
        del candidate["steps"][1]["reward"]
        report = _assess_safe_candidate(baseline, candidate)

        self.assertFalse(report["accepted_for_probe"])
        self.assertEqual(report["task_progress"]["status"], "unavailable")
        self.assertFalse(report["task_progress"]["evaluation_complete"])
        self.assertEqual(report["task_progress"]["reason"], "candidate_step_1_reward_missing_or_invalid")

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
