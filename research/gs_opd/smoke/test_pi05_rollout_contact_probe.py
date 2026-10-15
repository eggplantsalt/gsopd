"""Lightweight tests for the rollout's first-chunk probe Bridge flow."""

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
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "run_first_chunk_contact_probe"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"Path": Path, "__file__": str(ROLLOUT_PATH)}
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


class FirstChunkContactProbeTests(unittest.TestCase):
    def test_probes_full_chunk_without_sending_real_step_and_saves_evidence(self):
        probe = _load_probe_function()
        teacher = {
            "schema_version": "gs_opd.safety_teacher.v2",
            "objects": [
                {
                    "id": "object_4",
                    "name": "bowl",
                    "role": "avoid_object",
                    "visibility": "visible",
                    "mobility": "likely_movable",
                    "visual_facts": [],
                    "inferences": [],
                    "uncertainties": [],
                }
            ],
            "rules": [
                {
                    "id": "rule_1",
                    "subject_parts": ["gripper"],
                    "target_id": "object_4",
                    "predicate": "contact",
                    "decision": "forbid",
                    "phase": "any",
                    "reason": "Avoid contact.",
                }
            ],
            "global_uncertainties": [],
        }
        geometry = {"status": "complete"}
        gripper = {"status": "complete"}
        branch = {
            "main_state_restored": True,
            "candidates": [{"candidate_id": "pi05-first-action-chunk"}],
        }
        conn = FakeConnection(
            [geometry, gripper, {"snapshot_id": "snapshot-1"}, branch]
        )
        chunk = np.asarray([[0.1, 0, 0, 0, 0, 0, 0], [0, 0.2, 0, 0, 0, 0, 0]])
        compile_result = {"status": "safe", "candidates": []}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            teacher_path = tmp_path / "teacher.json"
            teacher_path.write_text(json.dumps(teacher), encoding="utf-8")
            semantic_path = tmp_path / "semantic.json"
            semantic_path.write_text("{}", encoding="utf-8")
            out_dir = tmp_path / "output"
            with patch(
                "research.gs_opd.smoke.contact_safety_compiler_smoke.compile_forbid_contact",
                return_value=compile_result,
            ) as compile_mock:
                report = probe(
                    conn,
                    chunk,
                    "move the bowl",
                    teacher_path,
                    "object_4",
                    semantic_path,
                    out_dir,
                )

            self.assertEqual(
                [request["op"] for request in conn.requests],
                [
                    "geometry_inventory",
                    "gripper_geometry_inventory",
                    "snapshot",
                    "branch_rollout",
                ],
            )
            self.assertEqual(
                conn.requests[-1]["candidates"][0]["actions"], chunk.tolist()
            )
            self.assertEqual(report["predicted_action_chunk"], chunk.tolist())
            self.assertTrue(report["main_state_restored"])
            self.assertEqual(report["branch_rollout_result"], branch)
            self.assertTrue((out_dir / "first_chunk_contact_safety_probe.json").is_file())
            compile_mock.assert_called_once_with(
                unittest.mock.ANY,
                "object_4",
                {},
                branch,
                geometry,
                gripper,
            )


if __name__ == "__main__":
    unittest.main()
