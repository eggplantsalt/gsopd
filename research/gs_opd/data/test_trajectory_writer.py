"""Tests for optional per-decision safety labels in compressed traces."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.gs_opd.data.trajectory_writer import TrajectoryWriter


def _bridge_data(reward=0.0, done=False):
    main = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    wrist = (main + 1).astype(np.uint8)
    return {
        "instruction": "move the bowl to the target",
        "images": {
            "agentview_image": {"data": main.tobytes(), "dtype": "uint8", "shape": main.shape},
            "robot0_eye_in_hand_image": {"data": wrist.tobytes(), "dtype": "uint8", "shape": wrist.shape},
        },
        "state": {
            "robot0_eef_pos": [0.1, 0.2, 0.3],
            "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
            "robot0_gripper_qpos": [0.01, -0.01],
        },
        "reward": reward,
        "done": done,
    }


class TrajectoryWriterSafetyTests(unittest.TestCase):
    def test_safety_record_is_saved_as_non_object_json_and_episode_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "episode.npz"
            writer = TrajectoryWriter(path, {"suite": "obstacle_avoidance"})
            observation = _bridge_data()
            actions = np.zeros((2, 7), dtype=np.float32)
            safety = {
                "a0_verdict": "safe",
                "rules": [{"id": "rule_1", "decision": "forbid"}],
                "training_action_chunk": actions.tolist(),
                "training_label_valid": True,
            }
            writer.add_decision(0, 0, observation, actions, safety_record=safety)
            writer.add_step(0, 0, 0, actions[0], _bridge_data(0.0, False))
            summary = {"a0_verdict_counts": {"safe": 1, "violation": 0, "unknown": 0}}
            writer.metadata["safety_episode_summary"] = summary
            writer.save(steps=1, chunks=1)

            with np.load(path, allow_pickle=False) as saved:
                self.assertIn("decision_safety_json", saved.files)
                record = json.loads(saved["decision_safety_json"][0])
                self.assertEqual(record["a0_verdict"], "safe")
                self.assertTrue(record["training_label_valid"])
                metadata = json.loads(saved["metadata_json"].item())
                self.assertEqual(metadata["safety_decision_schema_version"], 1)
                self.assertEqual(metadata["safety_episode_summary"], summary)
                self.assertEqual(saved["decision_agentview_image"].shape, (1, 2, 2, 3))
                self.assertEqual(saved["executed_actions"].shape, (1, 7))

    def test_ordinary_trace_does_not_add_safety_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ordinary.npz"
            writer = TrajectoryWriter(path, {"suite": "obstacle_avoidance"})
            observation = _bridge_data()
            actions = np.zeros((1, 7), dtype=np.float32)
            writer.add_decision(0, 0, observation, actions)
            writer.save(steps=0, chunks=1)
            with np.load(path, allow_pickle=False) as saved:
                self.assertNotIn("decision_safety_json", saved.files)
                metadata = json.loads(saved["metadata_json"].item())
                self.assertNotIn("safety_decision_schema_version", metadata)


if __name__ == "__main__":
    unittest.main()
