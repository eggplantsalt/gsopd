"""Unit tests for the pure Contact Safety Compiler decision path."""

import json
import unittest

from research.gs_opd.smoke.contact_safety_compiler_smoke import compile_forbid_contact
from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


class ContactSafetyCompilerSmokeTest(unittest.TestCase):
    def _fixtures(self):
        teacher = parse_teacher_output_v2(
            json.dumps(
                {
                    "schema_version": "gs_opd.safety_teacher.v2",
                    "objects": [
                        {
                            "id": "object_4",
                            "name": "small metal bowl",
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
                            "id": "rule_avoid_bowl_contact",
                            "subject_parts": ["gripper"],
                            "target_id": "object_4",
                            "predicate": "contact",
                            "decision": "forbid",
                            "phase": "any",
                            "reason": "Avoid contact with this bowl.",
                        }
                    ],
                    "global_uncertainties": [],
                }
            )
        )
        semantic = {
            "status": "resolved",
            "semantic_decision": {
                "object_id": "object_4",
                "candidate_index": 1,
                "status": "resolved",
            },
            "teacher_object": {"object_id": "object_4"},
            "selected_candidate": {"candidate_index": 1},
            "physical_association": {
                "status": "matched_by_iou",
                "physical_object_id": "physical_26",
                "body_id": 20,
                "body_name": "target_body",
                "geom_ids": [2],
                "geom_names": ["target_geom"],
            },
        }
        geometry_inventory = {
            "status": "complete",
            "robot_classification_status": "available",
            "geom_count": 3,
            "geometries": [
                {"geom_id": 0, "geom_name": "finger_geom", "body_id": 10, "body_name": "gripper_body", "category": "robot"},
                {"geom_id": 1, "geom_name": "arm_geom", "body_id": 11, "body_name": "arm_body", "category": "robot"},
                {"geom_id": 2, "geom_name": "target_geom", "body_id": 20, "body_name": "target_body", "category": "scene"},
            ],
        }
        gripper_inventory = {
            "status": "complete",
            "robot_classification_status": "available",
            "robot_body_ids": [10, 11],
            "model_body_count": 32,
            "grippers": [
                {
                    "status": "complete",
                    "geometries": [
                        {"geom_id": 0, "geom_name": "finger_geom", "body_id": 10, "body_name": "gripper_body"}
                    ],
                }
            ],
        }
        branch_result = {
            "main_state_restored": True,
            "candidates": [
                {
                    "candidate_id": "try-contact",
                    "status": "ok",
                    "executed_steps": 1,
                    "steps": [{"step_index": 0}],
                    "contact_monitoring": {
                        "status": "complete",
                        "sampling_point": "immediately_after_each_successful_MjSim.step",
                        "robot_classification_status": "available",
                        "expected_physics_substeps_total": 4,
                        "captured_physics_substeps": 4,
                        "unobserved_expected_substeps": 0,
                        "contact_substeps": 1,
                        "no_contact_substeps": 3,
                        "unavailable_contact_substeps": 0,
                        "control_steps": [
                            {
                                "control_step_index": 0,
                                "status": "complete",
                                "expected_physics_substeps": 4,
                                "captured_physics_substeps": 4,
                                "unobserved_expected_substeps": 0,
                                "contact_samples": {"contact": 1, "no_contact": 3, "unavailable": 0},
                            }
                        ],
                        "events": [
                            {
                                "geom_pair": [
                                    {"id": 0, "name": "finger_geom"},
                                    {"id": 2, "name": "target_geom"},
                                ],
                                "body_pair": [
                                    {"id": 10, "name": "gripper_body"},
                                    {"id": 20, "name": "target_body"},
                                ],
                                "classification": "robot_scene",
                                "robot_involved": True,
                                "first_control_step": 0,
                                "first_physics_substep": 1,
                                "occurrence_count": 1,
                                "contact_point_count": 1,
                                "max_force_norm": 0.7,
                                "force_sample_count": 1,
                                "force_unavailable_count": 0,
                            }
                        ],
                    },
                }
            ],
        }
        return teacher, semantic, branch_result, geometry_inventory, gripper_inventory

    def test_exact_gripper_target_contact_is_a_violation_with_substep_evidence(self):
        teacher, semantic, branch, geometry, gripper = self._fixtures()
        result = compile_forbid_contact(
            teacher, "object_4", semantic, branch, geometry, gripper
        )
        self.assertEqual(result["status"], "violation")
        self.assertEqual(result["compiled_rule"]["target_body_id"], 20)
        self.assertEqual(result["compiled_rule"]["target_geom_ids"], [2])
        evidence = result["candidates"][0]["evidence"][0]
        self.assertEqual(evidence["geom_pair"][0]["id"], 0)
        self.assertEqual(evidence["geom_pair"][1]["id"], 2)
        self.assertEqual(evidence["first_physics_substep"], 1)
        self.assertEqual(evidence["max_force_norm"], 0.7)

    def test_other_robot_part_contact_is_not_misclassified_as_gripper_contact(self):
        teacher, semantic, branch, geometry, gripper = self._fixtures()
        event = branch["candidates"][0]["contact_monitoring"]["events"][0]
        event["geom_pair"] = [
            {"id": 1, "name": "arm_geom"},
            {"id": 2, "name": "target_geom"},
        ]
        event["body_pair"] = [
            {"id": 11, "name": "arm_body"},
            {"id": 20, "name": "target_body"},
        ]
        result = compile_forbid_contact(teacher, "object_4", semantic, branch, geometry, gripper)
        self.assertEqual(result["status"], "safe")
        self.assertEqual(
            result["candidates"][0]["evidence"]["other_robot_part_target_contacts"][0]["classification"],
            "other_robot_part_target_contact",
        )

    def test_incomplete_monitoring_or_missing_gripper_identity_is_unknown(self):
        teacher, semantic, branch, geometry, gripper = self._fixtures()
        branch["candidates"][0]["contact_monitoring"]["status"] = "incomplete"
        result = compile_forbid_contact(teacher, "object_4", semantic, branch, geometry, gripper)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["candidates"][0]["reason"], "substep_contact_monitoring_not_complete")

        branch["candidates"][0]["contact_monitoring"]["status"] = "complete"
        gripper["status"] = "partial"
        unknown = compile_forbid_contact(teacher, "object_4", semantic, branch, geometry, gripper)
        self.assertEqual(unknown["status"], "unknown")
        self.assertIsNone(unknown["gripper_geom_ids"])

    def test_no_contact_is_safe_only_with_complete_coverage_and_resolved_binding(self):
        teacher, semantic, branch, geometry, gripper = self._fixtures()
        monitor = branch["candidates"][0]["contact_monitoring"]
        monitor["contact_substeps"] = 0
        monitor["no_contact_substeps"] = 4
        monitor["control_steps"][0]["contact_samples"] = {
            "contact": 0,
            "no_contact": 4,
            "unavailable": 0,
        }
        monitor["events"] = []
        result = compile_forbid_contact(teacher, "object_4", semantic, branch, geometry, gripper)
        self.assertEqual(result["status"], "safe")
        monitor["unavailable_contact_substeps"] = 1
        unknown = compile_forbid_contact(teacher, "object_4", semantic, branch, geometry, gripper)
        self.assertEqual(unknown["status"], "unknown")

    def test_contact_count_without_events_is_unknown(self):
        teacher, semantic, branch, geometry, gripper = self._fixtures()
        monitor = branch["candidates"][0]["contact_monitoring"]
        monitor["events"] = []
        result = compile_forbid_contact(teacher, "object_4", semantic, branch, geometry, gripper)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(
            result["candidates"][0]["reason"],
            "contact_substep_count_and_event_list_disagree",
        )


if __name__ == "__main__":
    unittest.main()
