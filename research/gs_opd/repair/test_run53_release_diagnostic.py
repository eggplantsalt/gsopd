# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for run53 provenance and post-success diagnostic measurements."""

from __future__ import annotations

import unittest

from research.gs_opd.repair.run53_release_diagnostic import (
    contact_pair_names,
    has_target_hand_contact,
    normalize_grasp_binding,
    records_cover_range,
    summarize_settling,
)


class Run53ReleaseDiagnosticTests(unittest.TestCase):
    def test_segment_audit_requires_contiguous_timestep_action_and_state_rows(self):
        rows = [
            {"timestep": 11, "action_7d": [0.0] * 7, "snapshot_state": {}, "state": {}},
            {"timestep": 12, "action_7d": [0.0] * 7, "snapshot_state": {}, "state": {}},
        ]
        self.assertTrue(records_cover_range(rows, 11, 12))
        self.assertFalse(records_cover_range(rows, 11, 13))
        self.assertFalse(records_cover_range([rows[0], {**rows[1], "timestep": 13}], 11, 12))
        self.assertFalse(records_cover_range([{key: value for key, value in rows[0].items() if key != "state"}], 11, 11))

    def test_contact_pairs_are_canonical_across_geom_order(self):
        pairs = contact_pair_names({
            "contacts": [
                {"geom1": {"name": "pot2_body"}, "geom2": {"name": "finger_pad"}},
                {"geom1": {"name": "finger_pad"}, "geom2": {"name": "pot2_body"}},
            ]
        })
        self.assertEqual(pairs, {("finger_pad", "pot2_body")})

    def test_semantic_binding_normalization_and_contact_free_release_check(self):
        binding = normalize_grasp_binding({
            "object_id": "moka_pot_2",
            "root_body_id": 24,
            "root_body_name": "moka_pot_2_main",
            "geom_ids": [101],
            "geom_names": ["moka_pot_2_g1"],
        })
        self.assertEqual(binding["body_id"], 24)
        self.assertEqual(binding["body_name"], "moka_pot_2_main")
        self.assertTrue(has_target_hand_contact({
            "contacts": [{"geom1": {"name": "gripper0_finger1_collision"}, "geom2": {"name": "moka_pot_2_g1"}}]
        }, binding))
        self.assertFalse(has_target_hand_contact({
            "contacts": [{"geom1": {"name": "flat_stove_1_collision_burner"}, "geom2": {"name": "moka_pot_2_g1"}}]
        }, binding))

    def test_settling_span_and_velocity_are_measured_without_state_mutation(self):
        snapshots = [
            {"object_states": {"pot": {"body_id": 24, "root_pos": [0.0, 0.0, 0.1], "joint_qvel": [0.01, 0.0, 0.0, 0.0, 0.0, 0.02]}}},
            {"object_states": {"pot": {"body_id": 24, "root_pos": [0.001, 0.0, 0.1], "joint_qvel": [0.02, 0.0, 0.0, 0.0, 0.0, 0.03]}}},
        ]
        result = summarize_settling(snapshots, {"pot2": 24})
        measured = result["objects"]["pot2"]
        self.assertEqual(result["sample_count"], 2)
        self.assertAlmostEqual(measured["position_span_m"], 0.001)
        self.assertAlmostEqual(measured["maximum_linear_speed_m_s"], 0.02)
        self.assertAlmostEqual(measured["maximum_angular_speed_rad_s"], 0.03)
        self.assertEqual(snapshots[0]["object_states"]["pot"]["root_pos"], [0.0, 0.0, 0.1])


if __name__ == "__main__":
    unittest.main()
