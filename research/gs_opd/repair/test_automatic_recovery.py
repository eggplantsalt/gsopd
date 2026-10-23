# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Contract tests for fail-closed automatic recovery orchestration."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from research.gs_opd.repair.automatic_recovery import (
    AutomaticSafetyRecoveryEngine,
    GoalSpec,
    JointSpacePlan,
    Observation,
    PrefixPreview,
    RecoveryConfig,
    RecoveryState,
    TaskContext,
)


def _observation(timestep: int, position, *, task_success=False) -> Observation:
    return Observation(
        timestep=timestep,
        qpos=np.zeros(7, dtype=np.float64),
        eef_position_world_m=np.asarray(position, dtype=np.float64),
        eef_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        minimum_forbidden_clearance_m=0.02,
        clearance_complete=True,
        contacts_complete=True,
        task_success=task_success,
    )


def _context() -> TaskContext:
    goal = GoalSpec(
        goal_id="object-pregrasp",
        phase="pregrasp",
        source="unit_test_fixture",
        source_artifacts=(),
        position_world_m=(0.1, 0.0, 0.0),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    return TaskContext(
        task_id="test_task",
        instruction="move the task object safely",
        forbidden_contact_objects=("obstacle",),
        allowed_target_contact_objects=("target",),
        held_object=None,
        phase="pregrasp",
        target_source="verified_task_context",
        safety_policy_source="verified_contact_constraints",
        goals=(goal,),
        semantic_bindings_verified=True,
        safety_constraints_verified=True,
        grasp_state_verified=True,
    )


class FakeBackend:
    def __init__(self, *, unknown_risk=False, reject_first_prefix=False):
        self.context = _context()
        self.state = _observation(0, [0.0, 0.0, 0.0])
        self.unknown_risk = unknown_risk
        self.reject_first_prefix = reject_first_prefix
        self.plan_calls = 0
        self.preview_calls = 0
        self.executed_actions = []
        self.student_action_calls = 0
        self.handed_off = False

    def current_context(self):
        return self.context

    def observe(self):
        return self.state

    def student_action(self, observation):
        self.student_action_calls += 1
        return np.zeros(7, dtype=np.float32)

    def predict_student_risk(self, observation, action, context):
        if self.unknown_risk:
            return {"status": "unavailable", "reason": "unknown"}
        return {"status": "complete", "minimum_predicted_forbidden_clearance_m": 0.02}

    def execute_action(self, action, context):
        self.executed_actions.append(np.asarray(action).copy())
        self.state = _observation(1, [0.1, 0.0, 0.0], task_success=True)
        return self.state

    def resolve_goal(self, observation, context):
        return context.goal_for_phase(context.phase)

    def plan_rrt_connect(self, observation, goal, context, config):
        self.plan_calls += 1
        path = np.stack([observation.qpos, np.ones(7, dtype=np.float64) * self.plan_calls])
        return JointSpacePlan(
            qpos_path=path,
            minimum_forbidden_clearance_m=0.02,
            planner_status="offline_feasible",
            planner_source="fake_rrt_connect",
            planning_seconds=0.0,
        )

    def preview_prefix(self, observation, plan, context, config):
        self.preview_calls += 1
        if self.reject_first_prefix and self.preview_calls == 1:
            return PrefixPreview(
                actions=np.zeros((1, 7), dtype=np.float32),
                observations=[_observation(1, [0.1, 0.0, 0.0])],
                complete=True,
                main_state_restored_verified=True,
                safe=False,
                failure_reason="known_geometry_risk",
                artifacts={"retryable_safety_rejection": True},
            )
        return PrefixPreview(
            actions=np.zeros((1, 7), dtype=np.float32),
            observations=[_observation(1, [0.1, 0.0, 0.0])],
            complete=True,
            main_state_restored_verified=True,
            safe=True,
        )

    def task_expert_action(self, observation, context):
        return None

    def handoff_to_student(self, observation, context):
        self.handed_off = True

    def is_recovery_goal_reached(self, observation, goal):
        return bool(observation.task_success)

    def save_snapshot(self, label):
        return None

    def record_event(self, event):
        return None


class SegmentedRouteBackend(FakeBackend):
    """Model normal OSC segments continuing along one still-valid route."""

    def __init__(self):
        super().__init__()
        self.state = _observation(0, [0.0, 0.0, 0.0])

    def plan_rrt_connect(self, observation, goal, context, config):
        self.plan_calls += 1
        path = np.stack([np.full(7, index, dtype=np.float64) for index in range(16)])
        return JointSpacePlan(
            qpos_path=path,
            minimum_forbidden_clearance_m=0.02,
            planner_status="offline_feasible",
            planner_source="fake_rrt_connect",
            planning_seconds=0.0,
        )

    def preview_prefix(self, observation, plan, context, config):
        self.preview_calls += 1
        next_step = self.state.timestep + 1
        position = [0.0, -next_step / 1000.0, 0.0]
        expected = _observation(next_step, position, task_success=next_step == 15)
        expected.qpos[:] = float(next_step)
        return PrefixPreview(
            actions=np.zeros((1, 7), dtype=np.float32),
            observations=[expected],
            complete=True,
            main_state_restored_verified=True,
            safe=True,
            artifacts={"last_path_index": next_step},
        )

    def execute_action(self, action, context):
        self.executed_actions.append(np.asarray(action).copy())
        next_step = self.state.timestep + 1
        position = [0.0, -next_step / 1000.0, 0.0]
        self.state = _observation(next_step, position, task_success=next_step == 15)
        self.state.qpos[:] = float(next_step)
        return self.state

    def is_recovery_goal_reached(self, observation, goal):
        return bool(observation.task_success)


class MissingPostGraspGoalBackend(FakeBackend):
    def __init__(self, *, start_at_goal=False):
        super().__init__()
        if start_at_goal:
            self.state = _observation(0, [0.1, 0.0, 0.0])

    def current_context(self):
        if self.handed_off:
            return replace(
                self.context,
                phase="grasp",
                unresolved_goal_phases=("grasp",),
            )
        return self.context

    def execute_action(self, action, context):
        self.executed_actions.append(np.asarray(action).copy())
        self.state = _observation(1, [0.1, 0.0, 0.0], task_success=False)
        return self.state

    def is_recovery_goal_reached(self, observation, goal):
        return float(np.linalg.norm(
            observation.eef_position_world_m - np.asarray(goal.position_world_m)
        )) <= 0.01


class AutomaticRecoveryTests(unittest.TestCase):
    def test_unknown_student_risk_fails_closed_before_execution(self):
        backend = FakeBackend(unknown_risk=True)
        result = AutomaticSafetyRecoveryEngine(
            backend,
            RecoveryConfig(segment_control_steps=2),
        ).run()

        self.assertEqual(result.state, RecoveryState.FAILED)
        self.assertIn("student_risk_prediction_unknown", result.reason)
        self.assertEqual(backend.executed_actions, [])
        self.assertFalse(result.local_recovery_success)

    def test_complete_recovery_hands_off_and_checks_native_task_success(self):
        backend = FakeBackend()
        result = AutomaticSafetyRecoveryEngine(
            backend,
            RecoveryConfig(segment_control_steps=2),
        ).run()

        self.assertEqual(result.state, RecoveryState.COMPLETE)
        self.assertTrue(result.local_recovery_success)
        self.assertTrue(result.complete_task_success)
        self.assertTrue(result.complete_safety_success)
        self.assertTrue(result.standard_horizon_success)
        self.assertTrue(backend.handed_off)
        self.assertEqual(result.steps_executed, 1)
        self.assertEqual(result.initial_episode_timestep, 0)
        self.assertEqual(result.final_episode_timestep, 1)

    def test_unsafe_isolated_prefix_replans_from_restored_main_state(self):
        backend = FakeBackend(reject_first_prefix=True)
        result = AutomaticSafetyRecoveryEngine(
            backend,
            RecoveryConfig(segment_control_steps=2, maximum_replans=2),
        ).run()

        self.assertEqual(result.state, RecoveryState.COMPLETE)
        self.assertEqual(backend.plan_calls, 2)
        self.assertEqual(result.replans, 1)
        self.assertTrue(result.complete_safety_success)

    def test_normal_segments_continue_on_the_same_validated_route(self):
        backend = SegmentedRouteBackend()
        result = AutomaticSafetyRecoveryEngine(
            backend,
            RecoveryConfig(segment_control_steps=1),
        ).run()

        self.assertEqual(result.state, RecoveryState.COMPLETE)
        self.assertEqual(backend.plan_calls, 1)
        self.assertEqual(backend.preview_calls, 15)
        self.assertEqual(result.replans, 0)
        self.assertEqual(result.steps_executed, 15)

    def test_handoff_without_a_verified_next_phase_fails_explicitly(self):
        backend = MissingPostGraspGoalBackend()
        result = AutomaticSafetyRecoveryEngine(
            backend,
            RecoveryConfig(segment_control_steps=2),
        ).run()

        self.assertEqual(result.state, RecoveryState.FAILED)
        self.assertTrue(result.local_recovery_success)
        self.assertFalse(result.complete_task_success)
        self.assertIn("task_goal_phase_unresolved:grasp", result.reason)
        self.assertTrue(backend.handed_off)

    def test_already_satisfied_goal_hands_off_without_replaying_policy_actions(self):
        backend = MissingPostGraspGoalBackend(start_at_goal=True)
        result = AutomaticSafetyRecoveryEngine(
            backend,
            RecoveryConfig(segment_control_steps=2),
        ).run()

        self.assertEqual(result.state, RecoveryState.FAILED)
        self.assertTrue(result.local_recovery_success)
        self.assertEqual(result.steps_executed, 0)
        self.assertEqual(backend.student_action_calls, 0)
        self.assertTrue(backend.handed_off)
        self.assertIn("task_goal_phase_unresolved:grasp", result.reason)

    def test_config_cannot_reduce_the_ten_millimetre_floor(self):
        with self.assertRaisesRegex(ValueError, "cannot be below 10 mm"):
            RecoveryConfig(hard_minimum_clearance_m=0.009).validate()


if __name__ == "__main__":
    unittest.main()
