# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for live-evidence task phase progression."""

from __future__ import annotations

import unittest

from research.gs_opd.repair.task_phase_controller import (
    PhaseEvidence,
    TaskPhaseController,
)


def _config(object_ids=("object_a", "object_b")):
    return {
        "objects": [
            {
                "object_id": object_id,
                "predicate": ["on", object_id, "support_region"],
                "goal_source": "verified_task_prior",
            }
            for object_id in object_ids
        ],
        "stable_grasp_control_steps": 3,
        "lift_validation_height_m": 0.03,
        "settle_position_span_m": 0.001,
        "settle_linear_speed_m_s": 0.05,
        "settle_angular_speed_rad_s": 0.5,
    }


def _evidence(
    timestep,
    *,
    eef_goal_reached=False,
    held_object=None,
    grasp_confirmed=False,
    object_z=0.0,
    task_goal_satisfied=False,
    all_task_goals_satisfied=False,
    satisfied_object_ids=(),
    fixture_predicate_satisfied=True,
    task_success=False,
):
    return PhaseEvidence(
        timestep=timestep,
        eef_goal_reached=eef_goal_reached,
        held_object=held_object,
        grasp_confirmed=grasp_confirmed,
        object_position_world_m=(0.0, 0.0, object_z),
        object_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        object_qvel=(0.0,) * 6,
        eef_position_world_m=(0.0, 0.0, 0.0),
        task_goal_satisfied=task_goal_satisfied,
        all_task_goals_satisfied=all_task_goals_satisfied,
        satisfied_object_ids=tuple(satisfied_object_ids),
        fixture_predicate_satisfied=fixture_predicate_satisfied,
        trusted_fixture_activation_available=False,
        task_success=task_success,
    )


class TaskPhaseControllerTests(unittest.TestCase):
    def test_grasp_requires_stable_bilateral_contact_and_verified_lift(self):
        controller = TaskPhaseController(_config())
        controller.handoff_after_pregrasp()
        controller.update(_evidence(1, eef_goal_reached=True))
        self.assertEqual(controller.phase, "close_gripper")

        controller.update(_evidence(2, held_object="object_a", grasp_confirmed=True))
        controller.update(_evidence(3, held_object="object_a", grasp_confirmed=True))
        self.assertEqual(controller.phase, "stable_grasp")
        controller.update(_evidence(4, held_object="object_a", grasp_confirmed=True))
        self.assertEqual(controller.phase, "lift")

        controller.update(_evidence(5, held_object="object_a", grasp_confirmed=True, object_z=0.029))
        self.assertEqual(controller.phase, "lift")
        controller.update(_evidence(6, held_object="object_a", grasp_confirmed=True, object_z=0.031))
        self.assertEqual(controller.phase, "transport")

    def test_release_requires_native_on_then_settled_object_before_next_pick(self):
        controller = TaskPhaseController(_config())
        controller.adopt_held_object("object_a")
        controller.update(_evidence(1, eef_goal_reached=True, held_object="object_a", grasp_confirmed=True))
        self.assertEqual(controller.phase, "precise_placement")
        controller.update(_evidence(2, eef_goal_reached=True, held_object="object_a", grasp_confirmed=True))
        self.assertEqual(controller.phase, "release")

        controller.update(_evidence(3, task_goal_satisfied=False))
        self.assertEqual(controller.phase, "release")
        controller.update(_evidence(4, task_goal_satisfied=True))
        self.assertEqual(controller.phase, "withdrawal")
        controller.update(_evidence(5, eef_goal_reached=True, task_goal_satisfied=True))
        self.assertEqual(controller.phase, "settle")
        for timestep in range(6, 11):
            controller.update(_evidence(
                timestep,
                task_goal_satisfied=True,
                satisfied_object_ids=("object_a",),
            ))
        self.assertEqual(controller.phase, "second_object_recovery")
        self.assertEqual(controller.current_object["object_id"], "object_b")

    def test_unknown_native_success_and_unconfigured_fixture_fail_closed(self):
        controller = TaskPhaseController(_config(("object_a",)))
        with self.assertRaisesRegex(ValueError, "native_task_success_is_unknown"):
            controller.update(_evidence(1, task_success=None))

        controller.handoff_after_pregrasp()
        # The fixture predicate cannot be inferred from geometry or object
        # placement, and no trusted activation controller was configured.
        controller.phase = "settle"
        for timestep in range(2, 6):
            controller.update(
                _evidence(
                    timestep,
                    task_goal_satisfied=True,
                    all_task_goals_satisfied=False,
                    satisfied_object_ids=("object_a",),
                    fixture_predicate_satisfied=False,
                )
            )
        with self.assertRaisesRegex(RuntimeError, "native_fixture_predicate_false"):
            controller.update(
                _evidence(
                    6,
                    task_goal_satisfied=True,
                    all_task_goals_satisfied=False,
                    satisfied_object_ids=("object_a",),
                    fixture_predicate_satisfied=False,
                )
            )


if __name__ == "__main__":
    unittest.main()
