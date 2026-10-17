"""Lightweight tests for continuous on-policy Safety Collection decisions."""

from __future__ import annotations

import argparse
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


def _load_helpers():
    names = {
        "_compact_repair_report",
        "parse_args",
        "_resolve_task_object_binding",
        "prepare_on_policy_safety_collection",
        "run_on_policy_safety_collection_decision",
        "_new_safety_episode_summary",
        "_update_safety_episode_summary",
        "_must_abort_safety_episode",
    }
    tree = ast.parse(ROLLOUT_PATH.read_text(encoding="utf-8"))
    body = [
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    module = ast.Module(body=body, type_ignores=[])
    namespace = {"Path": Path, "np": np, "argparse": argparse, "__file__": str(ROLLOUT_PATH)}
    exec(compile(module, str(ROLLOUT_PATH), "exec"), namespace)
    return namespace


class FakeConnection:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def send(self, payload):
        self.requests.append(payload)

    def recv(self):
        return self.responses.pop(0)


def _context():
    object_binding = {
        "semantic_decision": {"status": "resolved", "object_id": "object_4"},
        "teacher_object": {"object_id": "object_4"},
        "physical_association": {"status": "matched_by_iou", "body_id": 12},
    }
    return {
        "object_id": "object_4",
        "teacher": {
            "schema_version": "gs_opd.safety_teacher.v2",
            "objects": [{"id": "object_4", "name": "bowl", "role": "avoid_object"}],
            "rules": [{"id": "rule_1", "target_id": "object_4", "decision": "forbid"}],
            "global_uncertainties": [],
        },
        "semantic_result": object_binding,
        "semantic_results": {
            "object_4": object_binding,
            "object_9": {"semantic_decision": {"status": "resolved", "object_id": "object_9"}},
        },
        "safety_target_binding": {"object_id": "object_4", "body_id": 12},
        "geometry_inventory": {"status": "complete"},
        "gripper_inventory": {"status": "complete"},
        "repair_context": {"status": "available"},
        "task_object_binding": None,
    }


def _teacher_v2_object_only():
    def teacher_object(object_id, name, role):
        return {
            "id": object_id,
            "name": name,
            "role": role,
            "visibility": "visible",
            "mobility": "likely_movable" if role == "target_object" else "likely_fixed",
            "visual_facts": [],
            "inferences": [],
            "uncertainties": [],
        }

    return {
        "schema_version": "gs_opd.safety_teacher.v2",
        "objects": [
            teacher_object("object_1", "bowl", "target_object"),
            teacher_object("object_2", "blocker", "obstacle"),
        ],
        "rules": [],
        "object_contact_rules": [{
            "id": "object_rule_1",
            "subject_id": "object_1",
            "target_id": "object_2",
            "predicate": "contact",
            "decision": "forbid",
            "phase": "any",
            "reason": "The carried object should avoid the obstacle.",
        }],
        "global_uncertainties": [],
    }


def _semantic_binding(object_id, body_id, body_name, geom_id):
    return {
        "semantic_decision": {
            "status": "resolved",
            "object_id": object_id,
            "candidate_index": 0,
        },
        "teacher_object": {
            "object_id": object_id,
            "name": "bowl" if object_id == "object_1" else "blocker",
            "role": "target_object" if object_id == "object_1" else "obstacle",
        },
        "selected_candidate": {"candidate_index": 0, "body_id": body_id, "geom_ids": [geom_id]},
        "physical_association": {
            "status": "matched_by_iou",
            "physical_object_id": f"physical-{body_id}",
            "body_id": body_id,
            "body_name": body_name,
            "geom_ids": [geom_id],
        },
    }


def _branch(candidate_id, action_count, *, restored=True):
    candidate = {
        "candidate_id": candidate_id,
        "status": "ok",
        "requested_steps": action_count,
        "executed_steps": action_count,
        "steps": [{"reward": 0.0, "done": False} for _ in range(action_count)],
        "contact_monitoring": {"status": "complete", "events": []},
    }
    return {"main_state_restored": restored, "candidates": [candidate]}


class OnPolicySafetyCollectionTests(unittest.TestCase):
    def setUp(self):
        self.helpers = _load_helpers()
        self.chunk = np.zeros((3, 7), dtype=np.float32)
        self.chunk[:, 6] = -1.0
        self.call = self.helpers["run_on_policy_safety_collection_decision"]

    def test_safe_a0_is_a_valid_label_and_skips_repair(self):
        conn = FakeConnection([
            {"snapshot_id": "snapshot-1"},
            _branch("on-policy-A0-000000", len(self.chunk)),
        ])
        compiled = {
            "status": "safe",
            "candidates": [{"candidate_id": "on-policy-A0-000000", "verdict": "safe"}],
        }
        with patch(
            "research.gs_opd.smoke.contact_safety_compiler_smoke.compile_forbid_contact",
            return_value=compiled,
        ) as compiler, patch(
            "research.gs_opd.repair.verifier_guided_repair.run_verifier_guided_repair",
        ) as repair:
            result = self.call(conn, self.chunk, "move the bowl", 0, 0, _context())

        self.assertEqual(result["a0_verdict"], "safe")
        self.assertEqual(result["status"], "a0_safe")
        self.assertTrue(result["training_label_valid"])
        np.testing.assert_array_equal(result["training_action_chunk"], self.chunk.tolist())
        self.assertFalse(result["repair_attempted"])
        self.assertEqual(result["semantic_binding_results"], _context()["semantic_results"])
        self.assertEqual(compiler.call_args.kwargs["object_semantic_results"], _context()["semantic_results"])
        repair.assert_not_called()
        self.assertEqual([request["op"] for request in conn.requests], ["snapshot", "branch_rollout"])

    def test_unknown_a0_is_recorded_and_skips_repair(self):
        conn = FakeConnection([
            {"snapshot_id": "snapshot-2"},
            _branch("on-policy-A0-000000", len(self.chunk)),
        ])
        compiled = {
            "status": "unknown",
            "unknown_reasons": ["substep_contact_monitoring_not_complete"],
            "candidates": [{"candidate_id": "on-policy-A0-000000", "verdict": "unknown"}],
        }
        with patch(
            "research.gs_opd.smoke.contact_safety_compiler_smoke.compile_forbid_contact",
            return_value=compiled,
        ), patch(
            "research.gs_opd.repair.verifier_guided_repair.run_verifier_guided_repair",
        ) as repair:
            result = self.call(conn, self.chunk, "move the bowl", 0, 0, _context())

        self.assertEqual(result["a0_verdict"], "unknown")
        self.assertFalse(result["training_label_valid"])
        self.assertEqual(result["failure_reasons"], ["substep_contact_monitoring_not_complete"])
        repair.assert_not_called()

    def test_partial_a0_safe_branch_is_downgraded_to_unknown_label(self):
        partial = _branch("on-policy-A0-000000", len(self.chunk))
        candidate = partial["candidates"][0]
        candidate["executed_steps"] = 2
        candidate["steps"] = candidate["steps"][:2]
        conn = FakeConnection([{"snapshot_id": "snapshot-partial"}, partial])
        compiled = {
            "status": "safe",
            "candidates": [{"candidate_id": "on-policy-A0-000000", "verdict": "safe"}],
        }
        with patch(
            "research.gs_opd.smoke.contact_safety_compiler_smoke.compile_forbid_contact",
            return_value=compiled,
        ):
            result = self.call(conn, self.chunk, "move the bowl", 0, 0, _context())

        self.assertEqual(result["a0_compiler_verdict"], "safe")
        self.assertEqual(result["a0_verdict"], "unknown")
        self.assertFalse(result["training_label_valid"])
        self.assertEqual(
            result["failure_reasons"],
            ["A0_safe_verdict_did_not_cover_the_requested_full_chunk"],
        )

    def test_violation_runs_repair_from_same_snapshot_and_records_a_plus(self):
        conn = FakeConnection([
            {"snapshot_id": "snapshot-3"},
            _branch("on-policy-A0-000000", len(self.chunk)),
            _branch("repair-0001", len(self.chunk)),
        ])
        baseline_compiled = {
            "status": "violation",
            "candidates": [{"candidate_id": "on-policy-A0-000000", "verdict": "violation"}],
        }
        candidate_compiled = {
            "status": "safe",
            "candidates": [{"candidate_id": "repair-0001", "verdict": "safe"}],
        }
        repaired = (self.chunk + 0.01).tolist()
        repair_result = {
            "status": "repaired_candidate_found",
            "failure_reason": None,
            "repaired_action_chunk_A_plus": repaired,
            "supervision": {"training_label_valid": True},
            "search_attempts": [{"candidate_id": "repair-0001", "verdict": "accepted"}],
        }

        def invoke_core(*args, **kwargs):
            evaluate_candidates = args[4]
            evaluate_candidates([{"candidate_id": "repair-0001", "actions": repaired}])
            return repair_result

        with patch(
            "research.gs_opd.smoke.contact_safety_compiler_smoke.compile_forbid_contact",
            side_effect=[baseline_compiled, candidate_compiled],
        ) as compiler, patch(
            "research.gs_opd.repair.verifier_guided_repair.run_verifier_guided_repair",
            side_effect=invoke_core,
        ) as repair:
            result = self.call(conn, self.chunk, "move the bowl", 0, 0, _context())

        branch_requests = [request for request in conn.requests if request["op"] == "branch_rollout"]
        self.assertEqual([request["snapshot_id"] for request in branch_requests], ["snapshot-3", "snapshot-3"])
        self.assertEqual(result["a0_verdict"], "violation")
        self.assertEqual(result["status"], "violation_repaired")
        self.assertTrue(result["repair_attempted"])
        self.assertTrue(result["training_label_valid"])
        self.assertFalse(self.helpers["_must_abort_safety_episode"](result))
        self.assertEqual(result["label_source"], "compiler_verified_repaired_A_plus")
        self.assertEqual(result["training_action_chunk"], repaired)
        self.assertEqual(len(compiler.call_args_list), 2)
        for call in compiler.call_args_list:
            self.assertEqual(call.kwargs["object_semantic_results"], _context()["semantic_results"])
        self.assertNotIn("branch_result", result["repair_search"]["search_attempts"][0])
        self.assertFalse(any(request["op"] == "step" for request in conn.requests))
        repair.assert_called_once()

    def test_restore_failure_marks_episode_abort_before_main_action(self):
        conn = FakeConnection([
            {"snapshot_id": "snapshot-4"},
            _branch("on-policy-A0-000000", len(self.chunk), restored=False),
        ])
        compiled = {"status": "unknown", "unknown_reasons": ["branch_rollout_did_not_confirm_main_state_restore"]}
        with patch(
            "research.gs_opd.smoke.contact_safety_compiler_smoke.compile_forbid_contact",
            return_value=compiled,
        ):
            result = self.call(conn, self.chunk, "move the bowl", 0, 0, _context())

        self.assertFalse(result["main_state_restored"])
        self.assertTrue(result["must_abort_episode"])
        self.assertEqual(result["status"], "aborted_main_state_restore_failed")
        self.assertFalse(result["training_label_valid"])
        self.assertTrue(self.helpers["_must_abort_safety_episode"](result))

    def test_rollout_checks_restore_gate_before_any_main_step(self):
        tree = ast.parse(ROLLOUT_PATH.read_text(encoding="utf-8"))
        rollout_loop = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.While)
            and any(isinstance(child, ast.Name) and child.id == "steps" for child in ast.walk(node.test))
        )
        abort_gate = next(
            node for node in ast.walk(rollout_loop)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "_must_abort_safety_episode"
                for child in ast.walk(node.test)
            )
        )
        main_step_loop = next(
            node for node in rollout_loop.body
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "enumerate"
            and isinstance(node.iter.args[0], ast.Name)
            and node.iter.args[0].id == "chunk"
        )
        self.assertLess(abort_gate.lineno, main_step_loop.lineno)
        self.assertTrue(any(isinstance(node, ast.Break) for statement in abort_gate.body for node in ast.walk(statement)))

    def test_episode_summary_counts_verdicts_attempts_labels_and_failures(self):
        summary = self.helpers["_new_safety_episode_summary"]()
        self.helpers["_update_safety_episode_summary"](summary, {
            "a0_verdict": "violation",
            "repair_attempted": True,
            "repair_status": "no_safe_progress_preserving_candidate",
            "repaired_action_chunk_A_plus": None,
            "training_label_valid": False,
            "failure_reasons": ["no_safe_candidate"],
            "repair_search": {
                "search_attempts": [{"verdict": "rejected", "reason": "candidate_not_compiler_safe"}]
            },
        })
        self.assertEqual(summary["a0_verdict_counts"]["violation"], 1)
        self.assertEqual(summary["repair_attempts"], 1)
        self.assertEqual(summary["valid_a_plus_count"], 0)
        self.assertEqual(summary["failure_reason_counts"], {"no_safe_candidate": 1})
        self.assertEqual(summary["candidate_rejection_reason_counts"], {"candidate_not_compiler_safe": 1})

    def test_object_only_preflight_uses_full_mapping_and_target_progress_binding(self):
        teacher = _teacher_v2_object_only()
        semantic_results = {
            "object_1": _semantic_binding("object_1", 21, "bowl_body", 31),
            "object_2": _semantic_binding("object_2", 22, "blocker_body", 32),
        }
        geometry = {
            "status": "complete",
            "geometries": [
                {"geom_id": 31, "geom_name": "bowl_geom", "body_id": 21, "body_name": "bowl_body", "category": "scene"},
                {"geom_id": 32, "geom_name": "blocker_geom", "body_id": 22, "body_name": "blocker_body", "category": "scene"},
            ],
        }
        preflight = {
            "compiled_rule": None,
            "target_binding": {"object_id": "object_1", "body_id": 21, "geom_ids": [31]},
            "gripper_geom_ids": [3, 4],
            "unknown_reasons": ["branch_rollout_has_no_candidate_results"],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_path = root / "teacher.json"
            semantic_path = root / "semantic_results.json"
            teacher_path.write_text(json.dumps(teacher), encoding="utf-8")
            semantic_path.write_text(json.dumps(semantic_results), encoding="utf-8")
            conn = FakeConnection([
                geometry,
                {"status": "complete", "gripper_geom_ids": [3, 4]},
                {"status": "available"},
            ])
            with patch(
                "research.gs_opd.smoke.replay_archive_contact_smoke._validate_current_bindings",
                return_value={"valid": True, "objects": {"object_1": {}, "object_2": {}}, "errors": []},
            ) as validate_bindings, patch(
                "research.gs_opd.smoke.contact_safety_compiler_smoke.compile_forbid_contact",
                return_value=preflight,
            ) as compiler:
                context = self.helpers["prepare_on_policy_safety_collection"](
                    conn,
                    teacher_path,
                    "object_1",
                    semantic_results_path=semantic_path,
                )

        self.assertEqual(set(context["semantic_results"]), {"object_1", "object_2"})
        self.assertTrue(context["semantic_binding_validation"]["valid"])
        self.assertEqual(context["task_object_binding"]["object_id"], "object_1")
        self.assertEqual(context["task_object_id"], "object_1")
        self.assertEqual(context["task_object_binding"]["body_id"], 21)
        self.assertEqual(compiler.call_args.kwargs["object_semantic_results"], semantic_results)
        self.assertEqual(validate_bindings.call_args.args[1], semantic_results)

    def test_object_only_preflight_rejects_inconsistent_object_binding_map(self):
        teacher = _teacher_v2_object_only()
        semantic_results = {
            "object_1": _semantic_binding("object_1", 21, "bowl_body", 31),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_path = root / "teacher.json"
            semantic_path = root / "semantic_results.json"
            teacher_path.write_text(json.dumps(teacher), encoding="utf-8")
            semantic_path.write_text(json.dumps(semantic_results), encoding="utf-8")
            conn = FakeConnection([
                {"status": "complete"},
                {"status": "complete"},
                {"status": "available"},
            ])
            with patch(
                "research.gs_opd.smoke.replay_archive_contact_smoke._validate_current_bindings",
                return_value={"valid": False, "errors": ["semantic_binding_missing:object_2"]},
            ), patch(
                "research.gs_opd.smoke.contact_safety_compiler_smoke.compile_forbid_contact",
            ) as compiler:
                with self.assertRaisesRegex(ValueError, "semantic_binding_missing:object_2"):
                    self.helpers["prepare_on_policy_safety_collection"](
                        conn,
                        teacher_path,
                        "object_1",
                        semantic_results_path=semantic_path,
                    )
        compiler.assert_not_called()

    def test_legacy_gripper_rule_preflight_still_requires_and_accepts_compiled_rule(self):
        teacher = _teacher_v2_object_only()
        teacher["objects"] = [
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
        ]
        teacher["rules"] = [{
            "id": "rule_gripper_1",
            "subject_parts": ["gripper"],
            "target_id": "object_4",
            "predicate": "contact",
            "decision": "forbid",
            "phase": "any",
            "reason": "Avoid forbidden contact.",
        }]
        teacher.pop("object_contact_rules")
        semantic_result = {"semantic_decision": {"status": "resolved", "object_id": "object_4"}}
        preflight = {
            "compiled_rule": {"target_object_id": "object_4"},
            "target_binding": {"object_id": "object_4", "body_id": 9},
            "gripper_geom_ids": [3, 4],
            "unknown_reasons": ["branch_rollout_has_no_candidate_results"],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher_path = root / "teacher.json"
            semantic_path = root / "semantic.json"
            teacher_path.write_text(json.dumps(teacher), encoding="utf-8")
            semantic_path.write_text(json.dumps(semantic_result), encoding="utf-8")
            conn = FakeConnection([
                {"status": "complete"},
                {"status": "complete"},
                {"status": "available"},
            ])
            with patch(
                "research.gs_opd.smoke.contact_safety_compiler_smoke.compile_forbid_contact",
                return_value=preflight,
            ) as compiler:
                context = self.helpers["prepare_on_policy_safety_collection"](
                    conn,
                    teacher_path,
                    "object_4",
                    semantic_path,
                )

        self.assertEqual(context["semantic_results"], {"object_4": semantic_result})
        self.assertEqual(context["task_object_binding"], None)
        self.assertIsNotNone(context["preflight_result"]["compiled_rule"])
        self.assertEqual(compiler.call_args.kwargs["object_semantic_results"], {"object_4": semantic_result})

    def test_collection_cli_accepts_semantic_results_map_within_80_step_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher = root / "teacher.json"
            semantic_results = root / "semantic_results.json"
            output = root / "collection.npz"
            teacher.write_text("{}", encoding="utf-8")
            semantic_results.write_text("{}", encoding="utf-8")
            argv = [
                "pi05_rollout.py",
                "--on-policy-safety-collection",
                "--max-steps", "80",
                "--teacher-json", str(teacher),
                "--object-id", "object_1",
                "--semantic-results-json", str(semantic_results),
                "--on-policy-safety-out", str(output),
            ]
            with patch.object(sys, "argv", argv):
                args = self.helpers["parse_args"]()
        self.assertEqual(args.max_steps, 80)
        self.assertEqual(args.semantic_results_json, semantic_results)
        self.assertIsNone(args.semantic_result)


if __name__ == "__main__":
    unittest.main()
