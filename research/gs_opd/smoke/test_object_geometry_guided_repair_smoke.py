from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.gs_opd.smoke import object_geometry_guided_repair_smoke as smoke


class FakeConnection:
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        self.requests = []

    def send(self, request):
        self.requests.append(request)

    def recv(self):
        request = self.requests[-1]
        operation = request["op"]
        if operation in ("reset", "step"):
            return {"instruction": "move the carried object", "done": False}
        if operation == "geometry_inventory":
            return {"status": "complete", "robot_classification_status": "available"}
        if operation == "gripper_geometry_inventory":
            return {"status": "complete"}
        if operation == "repair_context":
            return {"status": "available", "world_up": [0.0, 0.0, 1.0], "robots": []}
        if operation == "snapshot":
            return {"snapshot_id": "snapshot-step-70", "snapshot_state": self.diagnostics}
        if operation == "restore":
            return {"restored_snapshot_id": request["snapshot_id"], "snapshot_state": self.diagnostics}
        if operation == "branch_rollout":
            candidates = []
            for spec in request["candidates"]:
                count = len(spec["actions"])
                candidates.append({
                    "candidate_id": spec["candidate_id"],
                    "status": "ok",
                    "requested_steps": count,
                    "executed_steps": count,
                    "steps": [{"step_index": i, "reward": 0.0, "done": False} for i in range(count)],
                    "contact_monitoring": {"status": "complete", "events": []},
                })
            return {"main_state_restored": True, "candidates": candidates}
        raise AssertionError(f"unexpected operation {operation}")


def _archive_episode():
    return {
        "archive": "/tmp/archive.npz",
        "decision_step": 70,
        "instruction": "move the carried object",
        "action_chunk": np.zeros((2, 7), dtype=np.float32),
        "replay_actions": np.zeros((1, 7), dtype=np.float32),
        "expected_observation": {},
    }


def _complete_coverage(_candidate, expected_steps):
    return {
        "full_chunk_executed": True,
        "contact_monitoring_complete": True,
        "requested_steps": expected_steps,
        "executed_steps": expected_steps,
    }


class ObjectGeometryGuidedRepairSmokeTests(unittest.TestCase):
    def test_search_uses_one_snapshot_and_keeps_repair_evidence(self):
        diagnostics = {"mujoco_integration_state": [0.1], "robot_states": [], "object_states": {}}
        connection = FakeConnection(diagnostics)
        baseline_violation = {
            "contact_type": "object_to_object",
            "moving_object_id": "carried",
            "obstacle_object_id": "obstacle",
            "control_step": 1,
            "physics_substep": 2,
            "normal_away_from_obstacle": [0.0, 1.0, 0.0],
        }
        selected_actions = np.full((2, 7), 0.01, dtype=np.float32).tolist()

        def fake_repair(*args, **kwargs):
            evaluate_candidates = args[4]
            specifications = [{
                "candidate_id": "repair-0001",
                "actions": selected_actions,
                "metadata": {"amplitude_m_per_control_step": 0.01},
            }]
            result = evaluate_candidates(specifications)
            self.assertEqual(result[0]["candidate_id"], "repair-0001")
            return {
                "status": "repaired_candidate_found",
                "failure_reason": None,
                "repaired_action_chunk_A_plus": selected_actions,
                "selected_candidate_id": "repair-0001",
                "search_attempts": [{
                    "candidate_id": "repair-0001",
                    "safety_verdict": "safe",
                    "task_progress": {"status": "pass"},
                    "action_constraints": {"status": "pass"},
                    "verdict": "accepted",
                }],
            }

        def fake_compile(_teacher, _object_id, _semantic, branch, *_args, **_kwargs):
            candidate_id = branch["candidates"][0]["candidate_id"]
            verdict = "violation" if candidate_id.startswith("repair-A0") else "safe"
            return {
                "status": verdict,
                "candidates": [{"candidate_id": candidate_id, "verdict": verdict, "evidence": []}],
            }

        with (
            patch.object(smoke, "compare_observation", return_value={
                "matched": True, "instruction_matched": True, "state_matched": True,
                "images_matched": True, "images_available": True,
            }),
            patch.object(smoke, "_validate_current_bindings", return_value={"valid": True}),
            patch.object(smoke, "_resolve_task_object_binding", return_value={
                "object_id": "carried", "status": "matched_by_iou", "role": "target_object", "body_id": 5,
            }),
            patch.object(smoke, "_candidate_coverage", side_effect=_complete_coverage),
            patch.object(smoke, "_compare_nested", return_value={"matched": True, "max_abs_delta": 0.0}),
            patch.object(smoke, "compile_forbid_contact", side_effect=fake_compile),
            patch.object(smoke, "locate_first_violation", return_value=(baseline_violation, None)),
            patch.object(smoke, "run_verifier_guided_repair", side_effect=fake_repair),
        ):
            report = smoke.run_experiment(
                connection,
                _archive_episode(),
                {"objects": [], "object_contact_rules": []},
                {},
                "carried",
                "carried",
            )

        branch_requests = [item for item in connection.requests if item["op"] == "branch_rollout"]
        self.assertEqual(len(branch_requests), 2)
        self.assertEqual({item["snapshot_id"] for item in branch_requests}, {"snapshot-step-70"})
        self.assertEqual(report["status"], "repaired_candidate_found")
        self.assertEqual(report["first_violation"], baseline_violation)
        self.assertEqual(report["repaired_action_chunk_A_plus"], selected_actions)
        self.assertTrue(report["all_branch_main_state_restores_passed"])
        self.assertEqual(report["candidate_counts"]["accepted_A_plus"], 1)

    def test_nonviolating_a0_stops_before_repair_search(self):
        connection = FakeConnection({"mujoco_integration_state": [0.1], "robot_states": [], "object_states": {}})
        with (
            patch.object(smoke, "compare_observation", return_value={
                "matched": True, "instruction_matched": True, "state_matched": True,
                "images_matched": True, "images_available": True,
            }),
            patch.object(smoke, "_validate_current_bindings", return_value={"valid": True}),
            patch.object(smoke, "_resolve_task_object_binding", return_value={
                "object_id": "carried", "status": "matched_by_iou", "role": "target_object", "body_id": 5,
            }),
            patch.object(smoke, "_candidate_coverage", side_effect=_complete_coverage),
            patch.object(smoke, "_compare_nested", return_value={"matched": True, "max_abs_delta": 0.0}),
            patch.object(smoke, "compile_forbid_contact", return_value={"status": "safe", "candidates": []}),
            patch.object(smoke, "run_verifier_guided_repair") as repair,
        ):
            report = smoke.run_experiment(
                connection,
                _archive_episode(),
                {"objects": [], "object_contact_rules": []},
                {},
                "carried",
                "carried",
            )
        self.assertEqual(report["status"], "baseline_not_explicitly_violating")
        repair.assert_not_called()


if __name__ == "__main__":
    unittest.main()
