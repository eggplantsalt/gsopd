"""Offline tests for generic Near-Far-triggered geometric Repair search."""

from __future__ import annotations

import unittest
from unittest.mock import patch
from typing import Any

import numpy as np

from research.gs_opd.repair.early_repair import _preflight, run_early_repair_search
from research.gs_opd.repair.near_far import action_chunk_fingerprint
from research.gs_opd.teacher.schema_v2 import SCHEMA_VERSION
from research.gs_opd.smoke.contact_safety_compiler_smoke import SAMPLING_POINT


SUBJECT_ID = "moving_item"
OBSTACLE_ID = "scene_obstacle"


def _object(object_id: str, name: str, role: str) -> dict[str, Any]:
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


def _teacher() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "objects": [
            _object(SUBJECT_ID, "moving item", "target_object"),
            _object(OBSTACLE_ID, "central obstacle", "obstacle"),
        ],
        "rules": [],
        "object_contact_rules": [{
            "id": "avoid-contact",
            "subject_id": SUBJECT_ID,
            "target_id": OBSTACLE_ID,
            "predicate": "contact",
            "decision": "forbid",
            "phase": "any",
            "reason": "Keep the carried item clear of the obstacle.",
        }],
        "global_uncertainties": [],
    }


def _bridge_candidate(candidate_id: str = "a0") -> dict[str, Any]:
    steps = []
    control_steps = []
    for index in range(5):
        steps.append({
            "step_index": index,
            "state": {
                "robot_states": [{"eef_pos": [0.01 * (index + 1), 0.0, 0.2]}],
                "object_states": {"moving": {"root_pos": [0.01 * (index + 1), 0.0, 0.2]}},
            },
        })
        control_steps.append({
            "control_step_index": index,
            "status": "complete",
            "expected_physics_substeps": 2,
            "captured_physics_substeps": 2,
            "unobserved_expected_substeps": 0,
            "contact_samples": {"contact": 0, "no_contact": 2, "unavailable": 0},
        })
    return {
        "candidate_id": candidate_id,
        "status": "ok",
        "requested_steps": 5,
        "executed_steps": 5,
        "start_state": {
            "robot_states": [{"eef_pos": [0.0, 0.0, 0.2]}],
            "object_states": {"moving": {"root_pos": [0.0, 0.0, 0.2]}},
        },
        "steps": steps,
        "contact_monitoring": {
            "status": "complete",
            "sampling_point": SAMPLING_POINT,
            "robot_classification_status": "available",
            "expected_physics_substeps_total": 10,
            "captured_physics_substeps": 10,
            "unobserved_expected_substeps": 0,
            "unavailable_contact_substeps": 0,
            "contact_substeps": 0,
            "no_contact_substeps": 10,
            "control_steps": control_steps,
            "events": [],
        },
    }


def _inputs(decision_step: int = 137) -> dict[str, Any]:
    actions = np.zeros((5, 7), dtype=np.float64)
    snapshot = {
        "snapshot_id": "snapshot-current",
        "snapshot_state": {"mujoco_integration_state": [0.0, 1.0]},
    }
    teacher = _teacher()
    bindings = {
        SUBJECT_ID: {
            "teacher_object": {"name": "moving item", "role": "target_object"},
        },
        OBSTACLE_ID: {
            "teacher_object": {"name": "central obstacle", "role": "obstacle"},
        },
    }
    binding_validation = {
        "valid": True,
        "objects": {
            SUBJECT_ID: {
                "status": "matches_current_geometry_inventory",
                "body_id": 10,
                "body_name": "moving-body",
                "geom_ids": [11],
                "teacher_name": "moving item",
            },
            OBSTACLE_ID: {
                "status": "matches_current_geometry_inventory",
                "body_id": 20,
                "body_name": "obstacle-body",
                "geom_ids": [21],
                "teacher_name": "central obstacle",
            },
        },
    }
    geometry = {
        "status": "complete",
        "robot_classification_status": "available",
        "geometries": [
            {"geom_id": 11, "body_id": 10, "category": "scene"},
            {"geom_id": 21, "body_id": 20, "category": "scene"},
        ],
    }
    near_far = {
        "status": "complete",
        "risk_verdict": "warning",
        "risk_warning": True,
        "simulation_complete": True,
        "requested_action_steps": 5,
        "executed_action_steps": 5,
        "decision_step": decision_step,
        "snapshot_id": snapshot["snapshot_id"],
        "action_chunk_sha256": action_chunk_fingerprint(actions),
        "target_geom_ids": [11],
        "obstacle_geom_ids": [21],
        "distance_samples": [
            {
                "step_index": decision_step + offset,
                "distance_available": True,
                "minimum_distance_m": 0.04 - offset * 0.002,
                "mesh_approximation": False,
            }
            for offset in range(6)
        ],
        "endpoint_closest_geom_evidence": {
            "geom_a": {"id": 11, "body_id": 10, "name": "moving-geom"},
            "geom_b": {"id": 21, "body_id": 20, "name": "obstacle-geom"},
            "closest_segment_world": [0.0, 0.0, 0.2, 0.02, 0.0, 0.2],
        },
        "snapshot_restore": {
            "before": {"matched": True},
            "after": {"matched": True},
        },
    }
    baseline = _bridge_candidate("a0-current")
    a0_validation = {
        "main_state_restored": True,
        "snapshot_id": snapshot["snapshot_id"],
        "action_chunk_sha256": action_chunk_fingerprint(actions),
        "branch_candidate": baseline,
        "compiler_candidate": {"candidate_id": "a0-current", "verdict": "safe"},
    }
    return {
        "actions": actions,
        "snapshot": snapshot,
        "near_far": near_far,
        "teacher": teacher,
        "bindings": bindings,
        "binding_validation": binding_validation,
        "geometry": geometry,
        "gripper": {"status": "complete", "robot_classification_status": "available"},
        "repair_context": {"status": "available", "world_up": [0.0, 0.0, 1.0]},
        "task_binding": {"body_id": 10},
        "a0_validation": a0_validation,
        "subject_object_id": SUBJECT_ID,
        "obstacle_object_id": OBSTACLE_ID,
    }


def _call_preflight(inputs: dict[str, Any]):
    return _preflight(
        snapshot=inputs["snapshot"],
        actions=inputs["actions"],
        near_far_result=inputs["near_far"],
        safety_ir=inputs["teacher"],
        semantic_bindings=inputs["bindings"],
        semantic_binding_validation=inputs["binding_validation"],
        geometry_inventory=inputs["geometry"],
        gripper_inventory=inputs["gripper"],
        repair_context=inputs["repair_context"],
        task_object_binding=inputs["task_binding"],
        a0_physical_validation=inputs["a0_validation"],
        subject_object_id=inputs["subject_object_id"],
        obstacle_object_id=inputs["obstacle_object_id"],
    )


class FakeConnection:
    def __init__(self, restore_ok: bool = True):
        self.restore_ok = restore_ok
        self.pending = None

    def send(self, request: dict[str, Any]) -> None:
        self.pending = request

    def recv(self) -> dict[str, Any]:
        request = self.pending
        if request["op"] == "branch_rollout":
            return {
                "main_state_restored": True,
                "candidates": [
                    {"candidate_id": item["candidate_id"], "status": "ok"}
                    for item in request["candidates"]
                ],
            }
        return {"error": f"unexpected operation {request['op']}"}


class EarlyRepairPreflightTests(unittest.TestCase):
    def test_arbitrary_non_s70_decision_index_passes_preflight(self):
        inputs = _inputs(decision_step=137)
        checks, resolved = _call_preflight(inputs)
        self.assertTrue(checks["passed"], checks["errors"])
        self.assertEqual(inputs["near_far"]["decision_step"], 137)
        self.assertIsNotNone(resolved)

    def test_near_far_unknown_fails_closed(self):
        inputs = _inputs(decision_step=29)
        inputs["near_far"].update({"status": "unknown", "risk_verdict": "unknown", "risk_warning": None})
        checks, resolved = _call_preflight(inputs)
        self.assertIsNone(resolved)
        self.assertIn("near_far_not_complete_with_positive_warning", checks["errors"])

    def test_unverified_a0_physical_result_fails_closed(self):
        inputs = _inputs(decision_step=29)
        inputs["a0_validation"]["main_state_restored"] = False
        checks, resolved = _call_preflight(inputs)
        self.assertIsNone(resolved)
        self.assertIn("a0_branch_snapshot_restore_not_confirmed", checks["errors"])


class EarlyRepairSearchTests(unittest.TestCase):
    def _run_search(
        self, *, accepted_count: int, restore_ok: bool = True, candidate_validation_complete: bool = True
    ):
        from research.gs_opd.smoke import early_repair_s70_smoke as probe

        inputs = _inputs(decision_step=137)
        specs = [
            {"candidate_id": f"early-{index:03d}", "actions": inputs["actions"].copy(),
             "direction": {"name": "away", "direction_world": [1.0, 0.0, 0.0]},
             "amplitude_m_per_control_step": 0.004, "window": {"window_anchor_step": 1},
             "action_diagnostics": {"window_semantics": "test"}, "input_bounds_check": {"status": "pass"}}
            for index in range(1, 3)
        ]
        baseline = inputs["a0_validation"]["branch_candidate"]
        resolved = {
            "baseline_candidate": baseline,
            "subject_binding": {"body_id": 10},
            "obstacle_binding": {"body_id": 20},
            "subject_geom_ids": [11],
            "obstacle_geom_ids": [21],
        }
        restore_calls = {"count": 0}

        def restore(*_args):
            restore_calls["count"] += 1
            return {"matched": restore_ok or restore_calls["count"] == 1}

        def assess(spec, *_args):
            accepted = int(spec["candidate_id"].split("-")[-1]) <= accepted_count
            return {
                "candidate_id": spec["candidate_id"],
                "actions": np.asarray(spec["actions"]).tolist(),
                "accepted_for_probe": accepted,
                "full_five_step_chunk": candidate_validation_complete,
                "contact_monitoring_complete": candidate_validation_complete,
                "physical_safety_verdict": "safe" if candidate_validation_complete else "unknown",
                "task_progress": {"status": "pass"} if candidate_validation_complete else {"status": "unavailable"},
                "action_constraints": {"status": "pass"} if candidate_validation_complete else {"status": "unavailable"},
                "endpoint_replay": {"status": "complete", "main_state_restored": True} if candidate_validation_complete else None,
                "rejection_reasons": [] if accepted else ["candidate_gate_rejected"],
            }

        with (
            patch("research.gs_opd.repair.early_repair._preflight", return_value=({"passed": True}, resolved)),
            patch.object(probe, "_restore_and_compare", side_effect=restore),
            patch.object(probe, "derive_clearance_directions", return_value=(
                [{"name": "away", "direction_world": [1.0, 0.0, 0.0]}], {"source": "endpoint geometry"}
            )),
            patch.object(probe, "generate_clearance_candidates", return_value=(specs, [])),
            patch.object(probe, "_branch_candidate", side_effect=lambda result, cid: {
                "candidate_id": cid, "status": "ok", "steps": [],
            }),
            patch.object(probe, "_compiler_candidate_summary", side_effect=lambda _result, cid: {
                "candidate_id": cid, "verdict": "safe",
            }),
            patch.object(probe, "replay_and_measure_endpoint", return_value={
                "status": "complete", "main_state_restored": True,
                "endpoint_distance": {"distance_available": True, "minimum_distance_m": 0.05},
            }),
            patch.object(probe, "assess_candidate", side_effect=assess),
            patch("research.gs_opd.smoke.contact_safety_compiler_smoke.compile_forbid_contact", return_value={}),
        ):
            report = run_early_repair_search(
                FakeConnection(), inputs["snapshot"], inputs["actions"], inputs["near_far"],
                inputs["teacher"], inputs["bindings"], inputs["binding_validation"],
                inputs["geometry"], inputs["gripper"], inputs["repair_context"],
                inputs["task_binding"], inputs["a0_validation"],
                subject_object_id=SUBJECT_ID, obstacle_object_id=OBSTACLE_ID,
            )
        return report

    def test_returns_all_qualified_candidates_without_selecting_one(self):
        report = self._run_search(accepted_count=2)
        self.assertEqual(report["status"], "qualified_candidates_available")
        self.assertTrue(report["executable"])
        self.assertEqual([item["candidate_id"] for item in report["qualified_candidates"]], ["early-001", "early-002"])
        self.assertEqual(len(report["candidate_results"]), 2)
        self.assertFalse(report["candidate_selected"])
        self.assertFalse(report["training_label_created"])
        self.assertTrue(report["snapshot_restore"]["after"]["matched"])

    def test_no_qualified_candidate_is_explicit_and_non_executable(self):
        report = self._run_search(accepted_count=0)
        self.assertEqual(report["status"], "no_qualified_candidates")
        self.assertFalse(report["executable"])
        self.assertEqual(report["qualified_candidates"], [])
        self.assertTrue(report["verification_complete"])

    def test_snapshot_restore_failure_makes_candidates_non_executable(self):
        report = self._run_search(accepted_count=2, restore_ok=False)
        self.assertEqual(report["status"], "not_executable")
        self.assertFalse(report["executable"])
        self.assertIn("snapshot_restore_unverified; candidates_are_not_executable", report["errors"])

    def test_incomplete_candidate_verification_makes_entire_search_non_executable(self):
        report = self._run_search(accepted_count=0, candidate_validation_complete=False)
        self.assertEqual(report["status"], "not_executable")
        self.assertFalse(report["executable"])
        self.assertFalse(report["verification_complete"])
        self.assertIn("candidate_validation_incomplete; no candidate is executable", report["errors"])


if __name__ == "__main__":
    unittest.main()
