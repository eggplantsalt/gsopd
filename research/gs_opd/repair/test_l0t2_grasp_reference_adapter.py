"""Focused tests for the provenance-bound L0T2 grasp reference adapter."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.gs_opd.repair.l0t2_grasp_reference_adapter import build_reference_adapter


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class L0T2GraspReferenceAdapterTests(unittest.TestCase):
    def _fixtures(self, root: Path):
        branch_dir = root / "branch"
        dataset_dir = root / "dataset"
        snapshot_dir = root / "snapshot"
        output_dir = root / "output"
        for path in (branch_dir, dataset_dir, snapshot_dir):
            path.mkdir(parents=True)
        trace = branch_dir / "observation_action_trace.npz"
        np.savez_compressed(
            trace,
            arm_joint_qpos=np.zeros((80, 7)),
            gripper_qpos_m=np.ones((80, 2)) * 0.01,
            pot2_root_position_world_m=np.tile([0.1, 0.2, 1.0], (80, 1)),
            pot2_root_quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (80, 1)),
        )
        start_snapshot = branch_dir / "start.pkl"
        start_snapshot.write_bytes(b"source snapshot")
        target_snapshot = snapshot_dir / "initial.pkl"
        target_snapshot.write_bytes(b"target snapshot")
        bddl = snapshot_dir / "task.bddl"
        bddl.write_text("(define (problem test))\n", encoding="utf-8")
        model_sha = "a" * 64
        branch = {
            "expected_source_model_geometry_sha256": model_sha,
            "native_reset_model_geometry_sha256": model_sha,
            "native_task_success": False,
            "physical_actions": 80,
            "terminal_timestep": 329,
            "source_snapshot": str(start_snapshot),
            "source_snapshot_sha256": _sha(start_snapshot),
            "data_paths": {"trace_npz": str(trace)},
            "live_grasp_state": {
                "status": "pass", "object_id": "moka_pot_2", "grasp_confirmed": True,
                "finger_contacts": {"left": True, "right": True},
            },
            "measured_pot2_vertical_rise_m": 0.039,
            "acceptance_verdicts": {"native_safety": True, "operational_safety": True, "strict_10mm": False},
        }
        branch_path = branch_dir / "branch_report.json"
        branch_path.write_text(json.dumps(branch), encoding="utf-8")
        dataset = {
            "task": "put both moka pots on the stove",
            "native_task_success": True,
            "operational_safety_pass": True,
            "strict_10mm_pass": False,
            "segments": [{
                "global_timestep_range_exclusive": [249, 329],
                "trace": str(trace),
                "trace_sha256": _sha(trace),
                "start_snapshot": {"sha256": _sha(start_snapshot)},
            }],
        }
        dataset_path = dataset_dir / "data_manifest.json"
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
        snapshot_manifest = {
            "scene_metadata": {
                "benchmark": "obstacle_avoidance", "level": 0, "task_id": 2,
                "task_name": "put_both_moka_pots_on_the_stove",
                "instruction": "put both moka pots on the stove",
                "init_state_index": 0, "environment_seed": 0,
                "standard_evaluation_horizon": 1000,
                "model_geometry_sha256": model_sha,
                "bddl_path": str(bddl), "bddl_sha256": _sha(bddl),
            },
            "snapshots": {"initial": {"payload_sha256": _sha(target_snapshot)}},
        }
        snapshot_manifest_path = snapshot_dir / "runtime_snapshot_manifest.json"
        snapshot_manifest_path.write_text(json.dumps(snapshot_manifest), encoding="utf-8")
        return {
            "branch_report_path": branch_path,
            "dataset_manifest_path": dataset_path,
            "snapshot_manifest_path": snapshot_manifest_path,
            "target_snapshot_path": target_snapshot,
            "bddl_path": bddl,
            "output_dir": output_dir,
        }

    def test_adapter_preserves_source_values_and_never_inherits_safety(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._fixtures(Path(temp))
            result = build_reference_adapter(**fixture)
            adapter = json.loads(Path(result["adapter_report"]).read_text())
            self.assertEqual(adapter["scope"]["task_id"], 2)
            self.assertFalse(adapter["source_provenance"]["strict_10mm_pass_for_source_episode"])
            self.assertFalse(adapter["source_provenance"]["safety_qualification_inherited"])
            self.assertFalse(adapter["source_provenance"]["training_label_valid"])
            self.assertEqual(
                json.loads((fixture["output_dir"] / "artifact_hashes.json").read_text())[
                    "l0t2_pot2_physical_grasp_pose_prior.npz"
                ]["sha256"],
                _sha(fixture["output_dir"] / "l0t2_pot2_physical_grasp_pose_prior.npz"),
            )
            with np.load(fixture["output_dir"] / "l0t2_pot2_physical_grasp_pose_prior.npz") as output:
                with np.load(fixture["branch_report_path"].parent / "observation_action_trace.npz") as source:
                    np.testing.assert_array_equal(output["arm_joint_qpos_rad"], source["arm_joint_qpos"])
                    np.testing.assert_array_equal(output["target_root_position_world_m"], source["pot2_root_position_world_m"])

    def test_rejects_changed_source_trace_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._fixtures(Path(temp))
            trace_path = fixture["branch_report_path"].parent / "observation_action_trace.npz"
            with np.load(trace_path) as source:
                changed = {key: source[key] for key in source.files}
            np.savez_compressed(trace_path, **changed, extra=np.array([1]))
            with self.assertRaisesRegex(ValueError, "trace_or_snapshot_hash_mismatch"):
                build_reference_adapter(**fixture)

    def test_rejects_strict_10mm_passing_source_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self._fixtures(Path(temp))
            dataset = json.loads(fixture["dataset_manifest_path"].read_text())
            dataset["strict_10mm_pass"] = True
            fixture["dataset_manifest_path"].write_text(json.dumps(dataset), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_acceptance_or_task_identity_mismatch"):
                build_reference_adapter(**fixture)


if __name__ == "__main__":
    unittest.main()
