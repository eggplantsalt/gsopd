"""Lightweight tests for the Pi0.5 repair probe Bridge request flow."""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ROLLOUT_PATH = Path(__file__).with_name("pi05_rollout.py")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_probe_function():
    tree = ast.parse(ROLLOUT_PATH.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "run_first_chunk_verifier_guided_repair"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"Path": Path, "np": np, "__file__": str(ROLLOUT_PATH)}
    exec(compile(module, str(ROLLOUT_PATH), "exec"), namespace)
    return namespace[node.name]


class FakeConnection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def send(self, payload):
        self.requests.append(payload)

    def recv(self):
        return self.responses.pop(0)


class Pi05RepairProbeTests(unittest.TestCase):
    def test_uses_snapshot_branch_and_never_steps_main_environment(self):
        probe = _load_probe_function()
        teacher = {
            "schema_version": "gs_opd.safety_teacher.v2",
            "objects": [
                {
                    "id": "object_4", "name": "bowl", "role": "avoid_object",
                    "visibility": "visible", "mobility": "likely_movable",
                    "visual_facts": [], "inferences": [], "uncertainties": [],
                }
            ],
            "rules": [
                {
                    "id": "rule_1", "subject_parts": ["gripper"],
                    "target_id": "object_4", "predicate": "contact",
                    "decision": "forbid", "phase": "any", "reason": "Avoid contact.",
                }
            ],
            "global_uncertainties": [],
        }
        branch = {"main_state_restored": True, "candidates": [{"candidate_id": "repair-A0"}]}
        candidate_branch = {
            "main_state_restored": True,
            "candidates": [{"candidate_id": "repair-0001"}],
        }
        conn = FakeConnection(
            [
                {"status": "complete"},
                {"status": "complete"},
                {"status": "available"},
                {"snapshot_id": "snapshot-1"},
                branch,
                candidate_branch,
            ]
        )
        chunk = np.asarray([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
        compiled = {"status": "safe", "candidates": [{"candidate_id": "repair-A0", "verdict": "safe"}]}
        candidate_compiled = {
            "status": "safe",
            "candidates": [{"candidate_id": "repair-0001", "verdict": "safe"}],
        }
        repair_result = {
            "status": "skipped_original_safe",
            "failure_reason": "A0 is safe",
            "repaired_action_chunk_A_plus": None,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            teacher_path = tmp_path / "teacher.json"
            teacher_path.write_text(json.dumps(teacher), encoding="utf-8")
            semantic_path = tmp_path / "semantic.json"
            semantic_path.write_text("{}", encoding="utf-8")
            with patch(
                "research.gs_opd.smoke.contact_safety_compiler_smoke.compile_forbid_contact",
                side_effect=[compiled, candidate_compiled],
            ) as compiler_mock, patch(
                "research.gs_opd.repair.verifier_guided_repair.run_verifier_guided_repair",
            ) as repair_mock:
                def invoke_core(*args, **kwargs):
                    evaluate_candidates = args[4]
                    evaluate_candidates(
                        [{"candidate_id": "repair-0001", "actions": chunk.tolist()}]
                    )
                    return repair_result

                repair_mock.side_effect = invoke_core
                report = probe(
                    conn,
                    chunk,
                    "move the bowl",
                    teacher_path,
                    "object_4",
                    semantic_path,
                    tmp_path / "repair_out",
                )

            self.assertEqual(
                [request["op"] for request in conn.requests],
                [
                    "geometry_inventory", "gripper_geometry_inventory", "repair_context",
                    "snapshot", "branch_rollout", "branch_rollout",
                ],
            )
            self.assertFalse(any(request["op"] == "step" for request in conn.requests))
            branch_requests = [request for request in conn.requests if request["op"] == "branch_rollout"]
            self.assertEqual([request["snapshot_id"] for request in branch_requests], ["snapshot-1", "snapshot-1"])
            self.assertEqual([request["candidates"][0]["candidate_id"] for request in branch_requests], ["repair-A0", "repair-0001"])
            self.assertEqual(report["status"], "skipped_original_safe")
            self.assertIsNone(report["repaired_action_chunk_A_plus"])
            self.assertTrue((tmp_path / "repair_out" / "verifier_guided_repair.json").is_file())
            self.assertEqual(compiler_mock.call_count, 2)
            repair_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
