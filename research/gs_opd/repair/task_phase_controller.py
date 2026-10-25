# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task-independent phase progression for verified manipulation programs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PhaseEvidence:
    """Current physical and native-task evidence used to advance one phase."""

    timestep: int
    eef_goal_reached: bool
    held_object: str | None
    grasp_confirmed: bool
    object_position_world_m: tuple[float, float, float]
    object_quaternion_wxyz: tuple[float, float, float, float]
    object_qvel: tuple[float, ...]
    eef_position_world_m: tuple[float, float, float]
    task_goal_satisfied: bool
    all_task_goals_satisfied: bool
    satisfied_object_ids: tuple[str, ...]
    fixture_predicate_satisfied: bool | None
    trusted_fixture_activation_available: bool
    task_success: bool | None
    count_phase_control: bool = True
    eef_position_error_m: float | None = None
    eef_orientation_error_rad: float | None = None


class TaskPhaseController:
    """Advance a configured pick, carry, place, release, and verify sequence.

    Object names, placement predicates, target frames, and provenance live in
    task configuration. The controller only consumes observed evidence and
    emits a phase name; it never writes simulator state or replays actions.
    """

    _PHASES = {
        "pregrasp",
        "approach",
        "close_gripper",
        "stable_grasp",
        "lift",
        "transport",
        "precise_placement",
        "release",
        "withdrawal",
        "settle",
        "second_object_recovery",
        "verify_task",
    }

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        default_goal_position_tolerance_m: float = 0.012,
        default_goal_orientation_tolerance_rad: float = 0.08,
    ):
        objects = config.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError("task_phase_program_requires_ordered_object_goals")
        self.objects = [dict(item) for item in objects]
        names = [item.get("object_id") for item in self.objects]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("task_phase_program_object_ids_must_be_explicit")
        if len(set(names)) != len(names):
            raise ValueError("task_phase_program_object_ids_must_be_unique")
        if any(not item.get("predicate") or not item.get("goal_source") for item in self.objects):
            raise ValueError("each_task_object_requires_a_goal_predicate_and_provenance")
        self.config = dict(config)
        self.default_goal_position_tolerance_m = float(default_goal_position_tolerance_m)
        self.default_goal_orientation_tolerance_rad = float(default_goal_orientation_tolerance_rad)
        if (
            not np.isfinite(self.default_goal_position_tolerance_m)
            or self.default_goal_position_tolerance_m <= 0.0
            or not np.isfinite(self.default_goal_orientation_tolerance_rad)
            or self.default_goal_orientation_tolerance_rad <= 0.0
        ):
            raise ValueError("default_pose_goal_tolerances_must_be_finite_and_positive")
        self.grasp_pose_candidates = [
            dict(item)
            for item in self.config.get(
                "grasp_pose_candidates",
                [{"candidate_id": "archived_source_pose"}],
            )
        ]
        if not 1 <= len(self.grasp_pose_candidates) <= 3:
            raise ValueError("task_phase_program_requires_one_to_three_grasp_pose_candidates")
        candidate_ids = [item.get("candidate_id") for item in self.grasp_pose_candidates]
        if any(not isinstance(item, str) or not item for item in candidate_ids):
            raise ValueError("grasp_pose_candidate_ids_must_be_explicit")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("grasp_pose_candidate_ids_must_be_unique")
        self.grasp_pose_candidate_index = 0
        self.phase = str(config.get("initial_phase", "pregrasp"))
        if self.phase not in self._PHASES:
            raise ValueError(f"task_phase_program_initial_phase_is_unknown:{self.phase}")
        self.object_index = 0
        self._expert_handoff = False
        self._last_timestep = -1
        self._phase_steps = 0
        self._grasp_streak = 0
        self._grasp_last_timestep = -1
        self._lift_start: np.ndarray | None = None
        self._lift_eef_start: np.ndarray | None = None
        self._settle_samples: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=5)
        self._settle_last_timestep = -1

    @property
    def current_object(self) -> dict[str, Any]:
        return self.objects[self.object_index]

    @property
    def current_grasp_pose_candidate(self) -> dict[str, Any]:
        return self.grasp_pose_candidates[self.grasp_pose_candidate_index]

    def advance_grasp_pose_candidate(self) -> dict[str, Any] | None:
        """Move to the next bounded grasp prior after a known unsafe target."""
        if self.grasp_pose_candidate_index + 1 >= len(self.grasp_pose_candidates):
            return None
        self.grasp_pose_candidate_index += 1
        return self.current_grasp_pose_candidate

    @property
    def complete(self) -> bool:
        return self.phase == "verify_task"

    def pose_goal_tolerances(self, phase: str) -> tuple[float, float]:
        """Return the one authoritative position/orientation threshold pair."""
        configured = self.config.get("phase_pose_tolerances", {}).get(phase, {})
        position = float(configured.get("position_m", self.default_goal_position_tolerance_m))
        orientation = float(
            configured.get("orientation_rad", self.default_goal_orientation_tolerance_rad)
        )
        if (
            not np.isfinite(position)
            or position <= 0.0
            or not np.isfinite(orientation)
            or orientation <= 0.0
        ):
            raise ValueError(f"phase_pose_goal_tolerances_invalid:{phase}")
        return position, orientation

    def is_pose_goal_reached(
        self,
        phase: str,
        position_error_m: float,
        orientation_error_rad: float,
    ) -> bool:
        """Apply the same phase threshold to recovery and phase progression."""
        position_tolerance, orientation_tolerance = self.pose_goal_tolerances(phase)
        return bool(
            float(position_error_m) <= position_tolerance
            and float(orientation_error_rad) <= orientation_tolerance
        )

    def handoff_after_pregrasp(self) -> None:
        """Start the configured phase program after recovery reaches pregrasp."""
        if self.phase != "pregrasp":
            raise RuntimeError(f"task_expert_handoff_requires_pregrasp:{self.phase}")
        self._expert_handoff = True
        self.phase = "approach"
        self._phase_steps = 0
        self._last_timestep = -1

    def adopt_held_object(self, object_id: str) -> None:
        """Resume a configured pick/place program from a live verified grasp."""
        indices = [i for i, item in enumerate(self.objects) if item["object_id"] == object_id]
        if len(indices) != 1:
            raise ValueError(f"held_object_is_not_in_task_program:{object_id}")
        self.object_index = indices[0]
        self.phase = "transport"
        self._expert_handoff = True
        self._phase_steps = 0
        self._last_timestep = -1

    def adopt_released_object_withdrawal(
        self,
        object_id: str,
        *,
        satisfied_object_ids: tuple[str, ...],
        held_object: str | None,
    ) -> None:
        """Reconstruct the live post-release clearance phase from observed state.

        This is for a fresh controller attached to a Runtime Snapshot where a
        native placement predicate is already true but the open gripper still
        has legal finger-only contact with that placed object. No historical
        phase counter or anchor is adopted; the caller supplies current native
        predicate and grasp evidence, then anchors withdrawal at the live EEF.
        """
        if held_object is not None:
            raise ValueError("released_object_withdrawal_requires_no_live_held_object")
        satisfied = set(str(name) for name in satisfied_object_ids)
        indices = [i for i, item in enumerate(self.objects) if item["object_id"] == object_id]
        if len(indices) != 1:
            raise ValueError(f"released_object_is_not_in_task_program:{object_id}")
        index = indices[0]
        preceding = {str(item["object_id"]) for item in self.objects[: index + 1]}
        if not preceding.issubset(satisfied):
            raise ValueError("released_object_withdrawal_requires_ordered_native_predicate_evidence")
        if self.phase != "pregrasp" or self._expert_handoff:
            raise RuntimeError(f"released_object_withdrawal_cannot_adopt_from:{self.phase}")
        self.object_index = index
        self.phase = "withdrawal"
        self._expert_handoff = True
        self._phase_steps = 0
        self._last_timestep = -1
        self._settle_samples.clear()
        self._settle_last_timestep = -1

    def _transition(self, phase: str) -> None:
        if phase not in self._PHASES:
            raise ValueError(f"unsupported_task_phase:{phase}")
        self.phase = phase
        self._phase_steps = 0
        self._last_timestep = -1

    def update(self, evidence: PhaseEvidence) -> str | None:
        """Advance one stage if its observed success condition is satisfied."""
        if evidence.task_success is None:
            raise ValueError("native_task_success_is_unknown")
        if evidence.count_phase_control and evidence.timestep != self._last_timestep:
            self._last_timestep = evidence.timestep
            self._phase_steps += 1
        limits = self.config.get("phase_control_step_limits", {})
        limit = limits.get(self.phase)
        if limit is not None and self._phase_steps > int(limit):
            raise RuntimeError(f"task_phase_control_budget_exhausted:{self.phase}")
        obj = self.current_object
        target = str(obj["object_id"])
        transitioned_from = self.phase

        if self.phase == "pregrasp" and evidence.held_object is None and evidence.task_goal_satisfied:
            self._expert_handoff = True
            next_index = next(
                (
                    index for index in range(self.object_index + 1, len(self.objects))
                    if self.objects[index]["object_id"] not in evidence.satisfied_object_ids
                ),
                None,
            )
            if next_index is None:
                if evidence.all_task_goals_satisfied and evidence.task_success is True:
                    self._transition("verify_task")
                elif evidence.fixture_predicate_satisfied is False and not evidence.trusted_fixture_activation_available:
                    raise RuntimeError("native_fixture_predicate_false_and_no_trusted_activation_controller")
                else:
                    raise RuntimeError("native_task_goal_set_is_incomplete_before_any_pending_object")
            else:
                self.object_index = next_index
                self._transition("second_object_recovery")
            return transitioned_from

        if not self._expert_handoff:
            return None
        eef_goal_reached = evidence.eef_goal_reached
        if (
            evidence.eef_position_error_m is not None
            and evidence.eef_orientation_error_rad is not None
        ):
            eef_goal_reached = self.is_pose_goal_reached(
                self.phase,
                evidence.eef_position_error_m,
                evidence.eef_orientation_error_rad,
            )

        if self.phase == "approach" and eef_goal_reached:
            self._transition("close_gripper")
        elif self.phase in {"close_gripper", "stable_grasp"}:
            self.phase = "stable_grasp"
            if evidence.timestep != self._grasp_last_timestep:
                self._grasp_last_timestep = evidence.timestep
                if evidence.held_object == target and evidence.grasp_confirmed:
                    self._grasp_streak += 1
                else:
                    self._grasp_streak = 0
            required = int(self.config.get("stable_grasp_control_steps", 3))
            if required < 2:
                raise ValueError("stable_grasp_control_steps_must_be_at_least_two")
            if self._grasp_streak >= required:
                self._lift_start = np.asarray(evidence.object_position_world_m, dtype=np.float64)
                self._lift_eef_start = np.asarray(evidence.eef_position_world_m, dtype=np.float64)
                self._transition("lift")
        elif self.phase == "lift":
            current = np.asarray(evidence.object_position_world_m, dtype=np.float64)
            distance = float(self.config.get("lift_validation_height_m", 0.03))
            if (
                evidence.held_object == target
                and evidence.grasp_confirmed
                and self._lift_start is not None
                and current[2] - self._lift_start[2] >= distance
            ):
                self._transition("transport")
        elif self.phase in {"transport", "second_object_recovery"} and eef_goal_reached:
            next_phase = "precise_placement" if self.phase == "transport" else "approach"
            self._transition(next_phase)
        elif self.phase == "precise_placement" and eef_goal_reached:
            if evidence.held_object != target or not evidence.grasp_confirmed:
                raise RuntimeError("held_object_lost_before_precise_placement")
            self._transition("release")
        elif self.phase == "release":
            if evidence.held_object is None and evidence.task_goal_satisfied:
                self._settle_samples.clear()
                self._settle_last_timestep = -1
                self._transition("withdrawal")
        elif self.phase == "withdrawal" and eef_goal_reached:
            self._transition("settle")
        elif self.phase == "settle":
            qvel = np.asarray(evidence.object_qvel, dtype=np.float64)
            position = np.asarray(evidence.object_position_world_m, dtype=np.float64)
            quaternion = np.asarray(evidence.object_quaternion_wxyz, dtype=np.float64)
            if evidence.timestep != self._settle_last_timestep:
                self._settle_last_timestep = evidence.timestep
                self._settle_samples.append((position, quaternion))
            settled = False
            if len(self._settle_samples) == self._settle_samples.maxlen and qvel.shape == (6,):
                positions = np.stack([sample[0] for sample in self._settle_samples])
                position_span = float(np.max(np.linalg.norm(positions - positions[-1], axis=1)))
                settled = bool(
                    evidence.held_object is None
                    and evidence.task_goal_satisfied
                    and position_span <= float(self.config.get("settle_position_span_m", 0.001))
                    and np.linalg.norm(qvel[:3]) <= float(self.config.get("settle_linear_speed_m_s", 0.05))
                    and np.linalg.norm(qvel[3:]) <= float(self.config.get("settle_angular_speed_rad_s", 0.5))
                )
            if settled:
                next_index = next(
                    (
                        index for index in range(self.object_index + 1, len(self.objects))
                        if self.objects[index]["object_id"] not in evidence.satisfied_object_ids
                    ),
                    None,
                )
                if next_index is not None:
                    self.object_index = next_index
                    self.grasp_pose_candidate_index = 0
                    self._transition("second_object_recovery")
                elif evidence.all_task_goals_satisfied and evidence.task_success is True:
                    self._transition("verify_task")
                elif evidence.fixture_predicate_satisfied is False and not evidence.trusted_fixture_activation_available:
                    raise RuntimeError("native_fixture_predicate_false_and_no_trusted_activation_controller")
                else:
                    raise RuntimeError("final_native_task_predicates_or_success_not_confirmed")
        if self.phase == "verify_task" and (
            not evidence.all_task_goals_satisfied or evidence.task_success is not True
        ):
            raise RuntimeError("native_task_success_verification_failed")
        return transitioned_from if self.phase != transitioned_from else None


__all__ = ["PhaseEvidence", "TaskPhaseController"]
