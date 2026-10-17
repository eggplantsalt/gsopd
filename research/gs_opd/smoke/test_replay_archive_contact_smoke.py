"""Lightweight tests for guarded archive replay and branch verification."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.smoke.replay_archive_contact_smoke import (
    _branch_coverage,
    load_archive_episode,
    run_experiment,
)
from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


def _bridge_image(value: np.ndarray) -> dict[str, Any]:
    return {"shape": value.shape, "dtype": str(value.dtype), "data": value.tobytes()}


class FakeConnection:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []
        self.current = None

    def send(self, request):
        self.current = request
        self.requests.append(request)

    def recv(self):
        return self.handler(self.current)


class ReplayArchiveContactSmokeTest(unittest.TestCase):
    def _write_archive(self, root: Path) -> tuple[Path, dict[str, Any]]:
        instruction = "move the marked object to the tray"
        prefix = np.arange(21, dtype=np.float32).reshape(3, 7) / 100
        chunk = np.arange(14, dtype=np.float32).reshape(2, 7) / 50
        agentview = np.full((4, 5, 3), 7, dtype=np.uint8)
        wrist = np.full((4, 5, 3), 13, dtype=np.uint8)
        path = root / "trace.npz"
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps({"instruction": instruction, "env_seed": 0})),
            decision_step_indices=np.asarray([3], dtype=np.int64),
            decision_instructions=np.asarray([instruction]),
            predicted_actions=chunk,
            predicted_action_offsets=np.asarray([0, 2], dtype=np.int64),
            step_indices=np.arange(5, dtype=np.int64),
            executed_actions=np.concatenate([prefix, chunk]),
            step_dones=np.zeros(5, dtype=np.bool_),
            decision_agentview_image=agentview[None],
            decision_robot0_eye_in_hand_image=wrist[None],
            decision_robot0_eef_pos=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
            decision_robot0_eef_quat=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            decision_robot0_gripper_qpos=np.asarray([[0.01, -0.01]], dtype=np.float64),
        )
        expected = {
            "instruction": instruction,
            "state": {
                "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
                "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": np.asarray([0.01, -0.01]),
            },
            "images": {
                "agentview_image": agentview,
                "robot0_eye_in_hand_image": wrist,
            },
        }
        return path, {"instruction": instruction, "prefix": prefix, "chunk": chunk, "expected": expected}

    def _teacher_and_bindings(self):
        teacher = parse_teacher_output_v2(
            json.dumps(
                {
                    "schema_version": "gs_opd.safety_teacher.v2",
                    "objects": [
                        {
                            "id": "object_payload",
                            "name": "carried object",
                            "role": "target_object",
                            "visibility": "visible",
                            "mobility": "likely_movable",
                            "visual_facts": [],
                            "inferences": [],
                            "uncertainties": [],
                        },
                        {
                            "id": "object_obstacle",
                            "name": "route obstacle",
                            "role": "obstacle",
                            "visibility": "visible",
                            "mobility": "likely_fixed",
                            "visual_facts": [],
                            "inferences": [],
                            "uncertainties": [],
                        },
                    ],
                    "rules": [],
                    "object_contact_rules": [
                        {
                            "id": "payload-clearance",
                            "subject_id": "object_payload",
                            "target_id": "object_obstacle",
                            "predicate": "contact",
                            "decision": "forbid",
                            "phase": "any",
                            "reason": "Keep the carried object clear of the route obstacle.",
                        }
                    ],
                    "global_uncertainties": [],
                }
            )
        )

        def semantic(object_id, body_id, body_name, geom_id, geom_name):
            return {
                "status": "resolved",
                "semantic_decision": {
                    "object_id": object_id,
                    "candidate_index": 0,
                    "reason": "visual candidate association",
                    "status": "resolved",
                },
                "teacher_object": {"object_id": object_id, "name": body_name},
                "selected_candidate": {"candidate_index": 0},
                "physical_association": {
                    "status": "matched_by_iou",
                    "physical_object_id": f"physical-{body_id}",
                    "body_id": body_id,
                    "body_name": body_name,
                    "geom_ids": [geom_id],
                    "geom_names": [geom_name],
                },
            }

        bindings = {
            "object_payload": semantic("object_payload", 21, "payload_body", 1, "payload_geom"),
            "object_obstacle": semantic("object_obstacle", 22, "obstacle_body", 2, "obstacle_geom"),
        }
        return teacher, bindings

    def _inventories(self):
        geometry = {
            "status": "complete",
            "robot_classification_status": "available",
            "geom_count": 3,
            "geometries": [
                {"geom_id": 0, "geom_name": "robot_geom", "body_id": 10, "body_name": "robot_body", "category": "robot"},
                {"geom_id": 1, "geom_name": "payload_geom", "body_id": 21, "body_name": "payload_body", "category": "scene"},
                {"geom_id": 2, "geom_name": "obstacle_geom", "body_id": 22, "body_name": "obstacle_body", "category": "scene"},
            ],
        }
        gripper = {
            "status": "complete",
            "robot_classification_status": "available",
            "robot_body_ids": [10],
            "model_body_count": 30,
            "grippers": [
                {
                    "status": "complete",
                    "geometries": [
                        {"geom_id": 0, "geom_name": "robot_geom", "body_id": 10, "body_name": "robot_body"}
                    ],
                }
            ],
        }
        return geometry, gripper

    def _contact_branch(self):
        event = {
            "geom_pair": [
                {"id": 1, "name": "payload_geom"},
                {"id": 2, "name": "obstacle_geom"},
            ],
            "body_pair": [
                {"id": 21, "name": "payload_body"},
                {"id": 22, "name": "obstacle_body"},
            ],
            "classification": "scene_scene",
            "robot_involved": False,
            "first_control_step": 1,
            "first_physics_substep": 0,
            "occurrence_count": 1,
            "contact_point_count": 1,
            "force_sample_count": 1,
            "force_unavailable_count": 0,
            "max_force_norm": 0.8,
            "first_contact_position_world": [0.2, 0.3, 0.4],
            "first_contact_normal_geom1_to_geom2_world": [0.0, 1.0, 0.0],
        }
        control_steps = [
            {
                "control_step_index": index,
                "status": "complete",
                "expected_physics_substeps": 1,
                "captured_physics_substeps": 1,
                "unobserved_expected_substeps": 0,
                "contact_samples": {"contact": int(index == 1), "no_contact": int(index != 1), "unavailable": 0},
            }
            for index in range(2)
        ]
        return {
            "main_state_restored": True,
            "candidates": [
                {
                    "candidate_id": "archived-decision-3",
                    "status": "ok",
                    "requested_steps": 2,
                    "executed_steps": 2,
                    "steps": [{"step_index": index, "reward": 0.0, "done": False} for index in range(2)],
                    "contact_monitoring": {
                        "status": "complete",
                        "sampling_point": "immediately_after_each_successful_MjSim.step",
                        "robot_classification_status": "available",
                        "expected_physics_substeps_total": 2,
                        "captured_physics_substeps": 2,
                        "unobserved_expected_substeps": 0,
                        "contact_substeps": 1,
                        "no_contact_substeps": 1,
                        "unavailable_contact_substeps": 0,
                        "control_steps": control_steps,
                        "events": [event],
                    },
                }
            ],
        }

    def test_archive_loader_extracts_prefix_and_exact_archived_chunk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path, values = self._write_archive(Path(temp_dir))
            episode = load_archive_episode(archive_path, decision_step=3)
        np.testing.assert_array_equal(episode["replay_actions"], values["prefix"])
        np.testing.assert_array_equal(episode["action_chunk"], values["chunk"])
        self.assertEqual(episode["decision_index"], 0)

    def test_observation_mismatch_prevents_snapshot_and_branch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path, values = self._write_archive(Path(temp_dir))
            episode = load_archive_episode(archive_path, decision_step=3)
        teacher, bindings = self._teacher_and_bindings()
        call_counts = {"step": 0}

        def handler(request):
            if request["op"] == "reset":
                return self._packed_observation(values["expected"])
            if request["op"] == "step":
                call_counts["step"] += 1
                response = self._packed_observation(values["expected"])
                if call_counts["step"] == 3:
                    image = np.full((4, 5, 3), 99, dtype=np.uint8)
                    response["images"]["agentview_image"] = _bridge_image(image)
                return response
            self.fail(f"Unexpected Bridge request after state mismatch: {request}")

        connection = FakeConnection(handler)
        report = run_experiment(connection, episode, teacher, bindings, "object_payload")
        self.assertEqual(report["status"], "initial_state_mismatch")
        self.assertFalse(report["snapshot_created"])
        self.assertFalse(report["branch_called"])
        self.assertNotIn("snapshot", [request["op"] for request in connection.requests])
        self.assertNotIn("branch_rollout", [request["op"] for request in connection.requests])

    def test_matching_replay_runs_full_branch_compiler_and_restore_check(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path, values = self._write_archive(Path(temp_dir))
            episode = load_archive_episode(archive_path, decision_step=3)
        teacher, bindings = self._teacher_and_bindings()
        geometry, gripper = self._inventories()
        diagnostics = {
            "mujoco_integration_state": [0.1, 0.2],
            "robot_states": [{"joint_qpos": [0.3], "eef_pos": [0.1, 0.2, 0.3]}],
            "object_states": {},
        }
        snapshots = iter(("snapshot-before", "snapshot-after"))
        snapshot_count = {"value": 0}

        def handler(request):
            operation = request["op"]
            if operation == "reset" or operation == "step":
                return self._packed_observation(values["expected"])
            if operation == "geometry_inventory":
                return geometry
            if operation == "gripper_geometry_inventory":
                return gripper
            if operation == "snapshot":
                snapshot_count["value"] += 1
                return {"snapshot_id": next(snapshots), "snapshot_state": diagnostics}
            if operation == "branch_rollout":
                self.assertEqual(request["snapshot_id"], "snapshot-before")
                np.testing.assert_array_equal(request["candidates"][0]["actions"], values["chunk"])
                return self._contact_branch()
            self.fail(f"Unexpected Bridge operation: {operation}")

        connection = FakeConnection(handler)
        report = run_experiment(connection, episode, teacher, bindings, "object_payload")
        self.assertEqual(report["status"], "violation_reproduced", report)
        self.assertEqual(report["compiler_verdict"], "violation")
        self.assertTrue(report["branch_coverage"]["full_chunk_executed"])
        self.assertTrue(report["branch_coverage"]["contact_monitoring_complete"])
        self.assertTrue(report["main_state_restoration"]["snapshot_diagnostics_match"])
        self.assertEqual(report["contact_evidence"][0]["first_control_step"], 1)
        self.assertEqual(report["contact_evidence"][0]["first_physics_substep"], 0)
        self.assertEqual(snapshot_count["value"], 2)

    def test_short_chunk_is_not_reported_as_full(self):
        branch = {
            "candidates": [
                {
                    "status": "ok",
                    "requested_steps": 2,
                    "executed_steps": 1,
                    "steps": [{"step_index": 0}],
                    "contact_monitoring": {"status": "incomplete", "control_steps": []},
                }
            ]
        }
        coverage = _branch_coverage(branch, expected_steps=2)
        self.assertFalse(coverage["full_chunk_executed"])
        self.assertFalse(coverage["contact_monitoring_complete"])

    @staticmethod
    def _packed_observation(expected):
        return {
            "instruction": expected["instruction"],
            "state": {key: value.tolist() for key, value in expected["state"].items()},
            "images": {key: _bridge_image(value) for key, value in expected["images"].items()},
        }


if __name__ == "__main__":
    unittest.main()
