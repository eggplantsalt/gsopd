"""Lightweight archive, branch-prefix, and real-Compiler tests."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.smoke import cross_chunk_contact_smoke as smoke
from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


def _packed_image(image: np.ndarray) -> dict[str, Any]:
    return {"shape": image.shape, "dtype": str(image.dtype), "data": image.tobytes()}


class FakeConnection:
    def __init__(self, handler):
        self.handler = handler
        self.current = None
        self.requests = []

    def send(self, request):
        self.current = request
        self.requests.append(request)

    def recv(self):
        return self.handler(self.current)


class CrossChunkContactSmokeTest(unittest.TestCase):
    def _write_archive(self, directory: Path):
        instruction = "move the carried item to the marked destination"
        steps = np.arange(80, dtype=np.int64)
        decisions = np.arange(0, 80, 5, dtype=np.int64)
        actions = np.arange(80 * 7, dtype=np.float32).reshape(80, 7) / 1000.0
        offsets = np.arange(0, 81, 5, dtype=np.int64)
        images_a = np.stack(
            [np.full((3, 4, 3), index, dtype=np.uint8) for index in decisions]
        )
        images_b = np.stack(
            [np.full((3, 4, 3), index + 1, dtype=np.uint8) for index in decisions]
        )
        eef_pos = np.asarray(
            [[step * 0.01, 0.2, 0.3] for step in decisions], dtype=np.float64
        )
        path = directory / "on_policy.npz"
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps({"instruction": instruction})),
            decision_step_indices=decisions,
            decision_instructions=np.asarray([instruction] * len(decisions)),
            predicted_actions=actions,
            predicted_action_offsets=offsets,
            step_indices=steps,
            executed_actions=actions.copy(),
            step_dones=np.zeros(80, dtype=np.bool_),
            decision_agentview_image=images_a,
            decision_robot0_eye_in_hand_image=images_b,
            decision_robot0_eef_pos=eef_pos,
            decision_robot0_eef_quat=np.tile([0.0, 0.0, 0.0, 1.0], (len(decisions), 1)),
            decision_robot0_gripper_qpos=np.tile([0.01, -0.01], (len(decisions), 1)),
        )
        return path, {"instruction": instruction, "actions": actions, "images_a": images_a, "images_b": images_b}

    def _teacher_and_bindings(self):
        teacher = parse_teacher_output_v2(
            json.dumps(
                {
                    "schema_version": "gs_opd.safety_teacher.v2",
                    "objects": [
                        {
                            "id": "object_carried",
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
                    "rules": [
                        {
                            "id": "grasp-allowed",
                            "subject_parts": ["gripper"],
                            "target_id": "object_carried",
                            "predicate": "contact",
                            "decision": "allow",
                            "phase": "any",
                            "reason": "The object must be grasped for the task.",
                        }
                    ],
                    "object_contact_rules": [
                        {
                            "id": "carried-clearance",
                            "subject_id": "object_carried",
                            "target_id": "object_obstacle",
                            "predicate": "contact",
                            "decision": "forbid",
                            "phase": "any",
                            "reason": "Avoid collision while carrying the object.",
                        }
                    ],
                    "global_uncertainties": [],
                }
            )
        )

        def semantic(object_id, body_id, body_name, geom_id, geom_name, name, role):
            return {
                "status": "resolved",
                "semantic_decision": {
                    "object_id": object_id,
                    "candidate_index": 0,
                    "reason": "visual candidate association",
                    "status": "resolved",
                },
                "teacher_object": {"object_id": object_id, "name": name, "role": role},
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
            "object_carried": semantic(
                "object_carried", 21, "payload_body", 1, "payload_geom", "carried object", "target_object"
            ),
            "object_obstacle": semantic(
                "object_obstacle", 22, "obstacle_body", 2, "obstacle_geom", "route obstacle", "obstacle"
            ),
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
                {"status": "complete", "geometries": [{"geom_id": 0, "geom_name": "robot_geom", "body_id": 10, "body_name": "robot_body"}]}
            ],
        }
        return geometry, gripper

    @staticmethod
    def _snapshot_state(x=0.7):
        return {
            "mujoco_integration_state": [x],
            "robot_states": [
                {
                    "robot_index": 0,
                    "joint_qpos": [0.0],
                    "joint_qvel": [0.0],
                    "eef_pos": [x, 0.2, 0.3],
                    "gripper_qpos": [0.01, -0.01],
                }
            ],
            "object_states": {},
            "kinematics_source": "forwarded_copy_of_mujoco_integration_state",
        }

    def _branch_candidate(self, candidate_id, actions, contact_step=None):
        steps = []
        controls = []
        for index, action in enumerate(actions):
            contact_here = index == contact_step
            coverage = {
                "control_step_index": index,
                "status": "complete",
                "expected_physics_substeps": 1,
                "captured_physics_substeps": 1,
                "unobserved_expected_substeps": 0,
                "contact_samples": {"contact": int(contact_here), "no_contact": int(not contact_here), "unavailable": 0},
            }
            controls.append(coverage)
            state = self._snapshot_state(0.7 + 0.01 * (index + 1))
            contacts = {
                "status": "contact" if contact_here else "no_contact",
                "contacts": [],
            }
            if contact_here:
                contacts["contacts"].append(
                    {
                        "geom1": {"id": 1, "name": "payload_geom"},
                        "geom2": {"id": 2, "name": "obstacle_geom"},
                        "body1": {"id": 21, "name": "payload_body"},
                        "body2": {"id": 22, "name": "obstacle_body"},
                        "position": [0.1, 0.2, 0.3],
                    }
                )
            steps.append(
                {
                    "step_index": index,
                    "action": list(action),
                    "reward": 0.0,
                    "done": False,
                    "state": state,
                    "contacts": contacts,
                    "substep_contact_coverage": coverage,
                }
            )

        events = []
        if contact_step is not None:
            events.append(
                {
                    "geom_pair": [{"id": 1, "name": "payload_geom"}, {"id": 2, "name": "obstacle_geom"}],
                    "body_pair": [{"id": 21, "name": "payload_body"}, {"id": 22, "name": "obstacle_body"}],
                    "classification": "scene_scene",
                    "robot_involved": False,
                    "first_control_step": contact_step,
                    "first_physics_substep": 0,
                    "first_candidate_physics_substep": contact_step,
                    "occurrence_count": 1,
                    "contact_point_count": 1,
                    "force_sample_count": 1,
                    "force_unavailable_count": 0,
                    "max_force_norm": 0.8,
                    "first_contact_position_world": [0.1, 0.2, 0.3],
                    "contact_position_status": "available",
                    "first_contact_normal_geom1_to_geom2_world": [0.0, 1.0, 0.0],
                    "contact_normal_status": "available",
                    "contact_normal_geom_ids": [1, 2],
                    "first_contact_sample_recorded": True,
                }
            )
        contact_steps = int(contact_step is not None)
        monitor = {
            "status": "complete",
            "sampling_point": "immediately_after_each_successful_MjSim.step",
            "robot_classification_status": "available",
            "expected_physics_substeps_total": len(actions),
            "captured_physics_substeps": len(actions),
            "unobserved_expected_substeps": 0,
            "contact_substeps": contact_steps,
            "no_contact_substeps": len(actions) - contact_steps,
            "unavailable_contact_substeps": 0,
            "control_steps": controls,
            "events": events,
        }
        return {
            "candidate_id": candidate_id,
            "status": "ok",
            "requested_steps": len(actions),
            "executed_steps": len(actions),
            "start_state": self._snapshot_state(),
            "steps": steps,
            "contact_monitoring": monitor,
        }

    def _bridge(
        self,
        archive_pair,
        *,
        image_mismatch=False,
        state_mismatch=False,
        incomplete_monitor=False,
    ):
        geometry, gripper = self._inventories()
        snapshot = self._snapshot_state()
        observation = archive_pair["step_70"]["expected_observation"]
        images = {
            key: _packed_image(value)
            for key, value in observation["images"].items()
        }
        if image_mismatch:
            changed = np.full_like(observation["images"]["agentview_image"], 255)
            images["agentview_image"] = _packed_image(changed)
        state = {
            key: value.tolist() for key, value in observation["state"].items()
        }
        if state_mismatch:
            state["robot0_eef_pos"][0] += 1e-3
        packed = {"images": images, "state": state, "instruction": archive_pair["instruction"]}

        def handler(request):
            operation = request["op"]
            if operation in ("reset", "step"):
                return dict(packed)
            if operation == "geometry_inventory":
                return geometry
            if operation == "gripper_geometry_inventory":
                return gripper
            if operation == "snapshot":
                return {"snapshot_id": "snapshot-s70", "snapshot_state": snapshot}
            if operation == "restore":
                return {"restored_snapshot_id": request["snapshot_id"], "snapshot_state": snapshot}
            if operation == "branch_rollout":
                self.assertEqual(request["snapshot_id"], "snapshot-s70")
                self.assertEqual(len(request["candidates"]), 2)
                candidate_a, candidate_b = request["candidates"]
                np.testing.assert_array_equal(candidate_a["actions"], archive_pair["action_70"])
                np.testing.assert_array_equal(candidate_b["actions"], archive_pair["action_10"])
                candidate_a_result = self._branch_candidate(
                    candidate_a["candidate_id"], candidate_a["actions"]
                )
                candidate_b_result = self._branch_candidate(
                    candidate_b["candidate_id"], candidate_b["actions"], contact_step=6
                )
                if incomplete_monitor:
                    candidate_b_result["contact_monitoring"]["control_steps"][0]["status"] = "incomplete"
                return {
                    "snapshot_id": "snapshot-s70",
                    "main_state_restored": True,
                    "candidates": [candidate_a_result, candidate_b_result],
                }
            self.fail(f"Unexpected Bridge request: {request}")

        return FakeConnection(handler)

    def test_archive_extracts_s70_s75_chunks_by_offsets_and_checks_execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path, values = self._write_archive(Path(temp_dir))
            episode = smoke.load_cross_chunk_archive(path)
        np.testing.assert_array_equal(episode["action_70"], values["actions"][70:75])
        np.testing.assert_array_equal(episode["action_75"], values["actions"][75:80])
        np.testing.assert_array_equal(episode["action_10"], values["actions"][70:80])
        self.assertEqual(episode["archive_action_checks"]["step_70_decision_index"], 14)
        self.assertEqual(episode["archive_action_checks"]["step_75_decision_index"], 15)

    def test_archive_loader_rejects_action_chunk_that_disagrees_with_executed_actions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _values = self._write_archive(Path(temp_dir))
            with np.load(path, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            arrays["executed_actions"][75, 2] += 0.01
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(smoke.CrossChunkArchiveError, "differs from its archived executed actions"):
                smoke.load_cross_chunk_archive(path)

    def test_real_compiler_checks_both_branches_and_reports_computed_tail_violation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _values = self._write_archive(Path(temp_dir))
            archive_pair = smoke.load_cross_chunk_archive(path)
        teacher, bindings = self._teacher_and_bindings()
        connection = self._bridge(archive_pair)
        report = smoke.run_experiment(connection, archive_pair, teacher, bindings)
        self.assertEqual(report["status"], "diagnostic_complete", report.get("failure_reason"))
        self.assertTrue(report["main_state_restoration"]["snapshot_diagnostics_match"])
        self.assertTrue(report["snapshot_restore_roundtrip"]["matched"])
        self.assertTrue(report["shared_first_five_comparison"]["matched"])
        self.assertTrue(report["step_75_decision_observation_comparison"]["matched_available_fields"])
        self.assertFalse(report["step_75_decision_observation_comparison"]["complete"])
        self.assertIn("robot0_eef_quat", report["step_75_decision_observation_comparison"]["unavailable_fields"])
        candidate_a, candidate_b = report["candidates"]
        self.assertEqual(candidate_a["safety_verdict"], "safe")
        self.assertTrue(candidate_a["full_chunk_executed"])
        self.assertTrue(candidate_a["contact_monitoring_complete"])
        self.assertEqual(candidate_b["safety_verdict"], "violation")
        self.assertEqual(candidate_b["first_violation_control_step_index"], 6)
        self.assertTrue(candidate_b["first_violation_is_in_indices_5_to_9"])
        evidence = candidate_b["first_object_to_object_violation"]
        self.assertEqual(evidence["first_physics_substep"], 0)
        self.assertEqual(evidence["geom_pair"][0]["name"], "payload_geom")
        self.assertEqual(evidence["body_pair"][1]["name"], "obstacle_body")
        self.assertEqual(evidence["first_contact_position_world"], [0.1, 0.2, 0.3])
        self.assertEqual(evidence["first_contact_normal_geom1_to_geom2_world"], [0.0, 1.0, 0.0])
        self.assertEqual(evidence["max_force_norm"], 0.8)
        self.assertEqual(
            [request["op"] for request in connection.requests].count("branch_rollout"), 1
        )

    def test_allow_image_mismatch_does_not_relax_robot_state_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _values = self._write_archive(Path(temp_dir))
            archive_pair = smoke.load_cross_chunk_archive(path)
        teacher, bindings = self._teacher_and_bindings()
        connection = self._bridge(archive_pair, image_mismatch=True, state_mismatch=True)
        report = smoke.run_experiment(
            connection,
            archive_pair,
            teacher,
            bindings,
            allow_image_mismatch=True,
        )
        self.assertEqual(report["status"], "step_70_robot_state_mismatch")
        self.assertFalse(report["snapshot_created"])
        self.assertNotIn("branch_rollout", [request["op"] for request in connection.requests])

    def test_image_mismatch_is_diagnostic_only_when_explicitly_allowed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _values = self._write_archive(Path(temp_dir))
            archive_pair = smoke.load_cross_chunk_archive(path)
        teacher, bindings = self._teacher_and_bindings()
        connection = self._bridge(archive_pair, image_mismatch=True)
        report = smoke.run_experiment(
            connection,
            archive_pair,
            teacher,
            bindings,
            allow_image_mismatch=True,
        )
        self.assertEqual(report["status"], "diagnostic_complete", report.get("failure_reason"))
        self.assertFalse(report["step_70_historical_images_pixel_exact"])
        self.assertTrue(report["step_70_historical_robot_state_matches"])

    def test_object_binding_mismatch_stops_before_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _values = self._write_archive(Path(temp_dir))
            archive_pair = smoke.load_cross_chunk_archive(path)
        teacher, bindings = self._teacher_and_bindings()
        bad_bindings = copy.deepcopy(bindings)
        bad_bindings["object_obstacle"]["physical_association"]["geom_ids"] = [1]
        connection = self._bridge(archive_pair)
        report = smoke.run_experiment(connection, archive_pair, teacher, bad_bindings)
        self.assertEqual(report["status"], "binding_validation_failed")
        self.assertFalse(report["snapshot_created"])
        self.assertNotIn("branch_rollout", [request["op"] for request in connection.requests])

    def test_incomplete_substep_monitoring_cannot_produce_complete_verdict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path, _values = self._write_archive(Path(temp_dir))
            archive_pair = smoke.load_cross_chunk_archive(path)
        teacher, bindings = self._teacher_and_bindings()
        connection = self._bridge(archive_pair, incomplete_monitor=True)
        report = smoke.run_experiment(connection, archive_pair, teacher, bindings)
        self.assertEqual(report["status"], "diagnostic_integrity_failed")
        self.assertFalse(report["candidates"][1]["contact_monitoring_complete"])
        self.assertEqual(report["candidates"][1]["safety_verdict"], "unknown")
        self.assertIn(
            "one_or_both_candidates_have_incomplete_physics_substep_contact_monitoring",
            report["integrity_errors"],
        )


if __name__ == "__main__":
    unittest.main()
