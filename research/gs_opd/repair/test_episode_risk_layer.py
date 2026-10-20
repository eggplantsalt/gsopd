"""Tests for deterministic cross-Episode Near-Far risk-layer scheduling."""

import unittest

from research.gs_opd.repair.episode_risk_layer import (
    RISK_LAYER_THRESHOLD_M,
    classify_gap_risk_layer,
    layer_for_episode_index,
    new_episode_risk_layer_record,
    record_episode_risk_distance,
)
from research.gs_opd.repair.near_far import GAP_THRESHOLDS_M


class EpisodeRiskLayerTests(unittest.TestCase):
    def test_episode_indices_cycle_far_mid_near_reproducibly(self):
        self.assertEqual(
            [layer_for_episode_index(index) for index in range(7)],
            ["far", "mid", "near", "far", "mid", "near", "far"],
        )
        self.assertEqual(layer_for_episode_index(4), layer_for_episode_index(4))

    def test_layer_thresholds_reuse_existing_near_far_values(self):
        self.assertEqual(RISK_LAYER_THRESHOLD_M, {
            "far": GAP_THRESHOLDS_M[2],
            "mid": GAP_THRESHOLDS_M[1],
            "near": GAP_THRESHOLDS_M[0],
        })

    def test_existing_thresholds_define_disjoint_far_mid_near_bands(self):
        self.assertEqual(classify_gap_risk_layer(0.020), "far")
        self.assertEqual(classify_gap_risk_layer(0.010), "mid")
        self.assertEqual(classify_gap_risk_layer(0.005), "near")
        self.assertEqual(classify_gap_risk_layer(-0.001), "near")
        self.assertIsNone(classify_gap_risk_layer(0.020001))
        self.assertIsNone(classify_gap_risk_layer(float("nan")))

    def test_records_unknown_then_first_reached_distance_only_once(self):
        record = new_episode_risk_layer_record(1)
        self.assertEqual(record["episode_id"], "gs-opd-risk-layer-000001")
        self.assertIsNone(record_episode_risk_distance(
            record, step_index=5, distance_m=None, distance_available=False,
            query_status="unavailable", errors=["no witness"],
        ))
        self.assertIsNone(record_episode_risk_distance(
            record, step_index=6, distance_m=0.012, distance_available=True,
            query_status="complete",
        ))
        trigger = record_episode_risk_distance(
            record, step_index=7, distance_m=0.010, distance_available=True,
            query_status="complete", minimum_pair={"geom_a": 1, "geom_b": 2},
        )
        self.assertEqual(trigger["layer"], "mid")
        self.assertEqual(trigger["step_index"], 7)
        self.assertEqual(trigger["observed_distance_m"], 0.010)
        self.assertIsNone(record_episode_risk_distance(
            record, step_index=8, distance_m=0.001, distance_available=True,
            query_status="complete",
        ))
        self.assertEqual(record["trigger"]["step_index"], 7)
        self.assertEqual(len(record["observations"]), 4)

    def test_unavailable_or_nonfinite_gap_never_triggers(self):
        for distance, available in ((float("nan"), True), (-1.0, False)):
            record = new_episode_risk_layer_record(0)
            self.assertIsNone(record_episode_risk_distance(
                record, step_index=0, distance_m=distance,
                distance_available=available,
            ))
            self.assertIsNone(record["trigger"])

    def test_episode_does_not_backfill_a_layer_that_was_already_skipped(self):
        far_record = new_episode_risk_layer_record(0)
        self.assertIsNone(record_episode_risk_distance(
            far_record, step_index=4, distance_m=0.005, distance_available=True,
        ))
        self.assertIsNone(far_record["trigger"])
        mid_record = new_episode_risk_layer_record(1)
        self.assertIsNone(record_episode_risk_distance(
            mid_record, step_index=4, distance_m=0.004, distance_available=True,
        ))
        self.assertIsNone(mid_record["trigger"])

    def test_invalid_episode_index_is_rejected(self):
        for index in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                layer_for_episode_index(index)


if __name__ == "__main__":
    unittest.main()
