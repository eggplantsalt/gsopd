"""Unit tests for the reusable Snapshot-local Near-Far forecast."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from research.gs_opd.repair.near_far import forecast_near_far_risk


class FakeBridge:
    def __init__(self, gaps: list[float], unavailable_at: set[int] | None = None, fail_final_restore: bool = False):
        self.gaps = gaps
        self.unavailable_at = unavailable_at or set()
        self.fail_final_restore = fail_final_restore
        self.state_index = 0
        self.restore_count = 0
        self.pending: dict[str, Any] | None = None
        self.requests: list[dict[str, Any]] = []
        self.snapshot_state = {
            "mujoco_integration_state": [0.0],
            "robot_states": {"eef_pos": [0.1, 0.2, 0.3]},
        }

    def send(self, request: dict[str, Any]) -> None:
        self.pending = request
        self.requests.append(request)

    def recv(self) -> dict[str, Any]:
        assert self.pending is not None
        request = self.pending
        op = request["op"]
        if op == "restore":
            self.restore_count += 1
            self.state_index = 0
            state = self.snapshot_state
            if self.fail_final_restore and self.restore_count == 2:
                state = {
                    "mujoco_integration_state": [0.0],
                    "robot_states": {"eef_pos": [0.1, 0.2, 0.301]},
                }
            return {
                "restored_snapshot_id": request["snapshot_id"],
                "snapshot_state": state,
            }
        if op == "geom_set_min_distance":
            index = self.state_index
            complete = index not in self.unavailable_at
            return {
                "status": "complete" if complete else "unavailable",
                "query_complete": complete,
                "distance_reliable": complete,
                "minimum_distance_m": self.gaps[min(index, len(self.gaps) - 1)] if complete else None,
                "minimum_pair": {
                    "geom_a": {"id": request["geom_ids_a"][0], "name": "target_geom"},
                    "geom_b": {"id": request["geom_ids_b"][0], "name": "obstacle_geom"},
                    "distance_m": self.gaps[min(index, len(self.gaps) - 1)],
                } if complete else None,
                "pair_count_requested": 1,
                "pair_count_attempted": 1 if complete else 0,
                "pair_count_censored_at_distmax": 0,
                "pair_error_count": 0 if complete else 1,
                "errors": [] if complete else ["synthetic incomplete query"],
            }
        if op == "step":
            self.state_index += 1
            return {"reward": 0.0, "done": False}
        raise AssertionError(f"unexpected Bridge operation: {op}")


def _snapshot(bridge: FakeBridge) -> dict[str, Any]:
    return {"snapshot_id": "snapshot-1", "snapshot_state": bridge.snapshot_state}


class NearFarForecastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actions = np.zeros((5, 7), dtype=np.float32)

    def test_threshold_and_trend_warning_uses_arbitrary_decision_step(self):
        bridge = FakeBridge([0.03, 0.019, 0.018, 0.017, 0.016, 0.015])
        result = forecast_near_far_risk(
            bridge, _snapshot(bridge), self.actions, [10, 11], [20], decision_step=41,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["risk_verdict"], "warning")
        self.assertTrue(result["risk_warning"])
        self.assertEqual(result["first_trigger_step"], 42)
        self.assertEqual([item["step_index"] for item in result["distance_samples"]], list(range(41, 47)))
        self.assertEqual(result["endpoint_closest_geom_evidence"]["geom_a"]["id"], 10)
        self.assertTrue(result["sustained_approach"])
        self.assertEqual(result["snapshot_id"], "snapshot-1")
        self.assertEqual(len(result["action_chunk_sha256"]), 64)
        self.assertEqual(bridge.restore_count, 2)
        self.assertEqual(bridge.state_index, 0)

    def test_complete_non_triggering_chunk_reports_no_warning(self):
        bridge = FakeBridge([0.05] * 6)
        result = forecast_near_far_risk(
            bridge, _snapshot(bridge), self.actions, [10], [20], decision_step=9,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["risk_verdict"], "no_warning")
        self.assertFalse(result["risk_warning"])
        self.assertFalse(result["sustained_approach"])
        self.assertEqual(result["first_trigger_step"], None)

    def test_incomplete_distance_query_is_unknown_not_no_warning(self):
        bridge = FakeBridge([0.05] * 6, unavailable_at={3})
        result = forecast_near_far_risk(
            bridge, _snapshot(bridge), self.actions, [10], [20], decision_step=25,
        )
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["risk_verdict"], "unknown")
        self.assertIsNone(result["risk_warning"])
        self.assertTrue(result["unknown_reasons"])
        self.assertEqual(bridge.state_index, 0)

    def test_failed_final_snapshot_verification_is_unknown(self):
        bridge = FakeBridge([0.05] * 6, fail_final_restore=True)
        result = forecast_near_far_risk(
            bridge, _snapshot(bridge), self.actions, [10], [20], decision_step=103,
        )
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["risk_verdict"], "unknown")
        self.assertIsNone(result["risk_warning"])
        self.assertFalse(result["snapshot_restore"]["after"]["matched"])
        self.assertIn("snapshot_restore_after_forecast_failed", result["unknown_reasons"])


if __name__ == "__main__":
    unittest.main()
