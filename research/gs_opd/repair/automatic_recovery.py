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

"""Fail-closed orchestration for automatic safety recovery."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class RecoveryState(str, Enum):
    STUDENT = "student_rollout"
    ASSESS = "risk_assessment"
    RESOLVE = "target_resolution"
    PLAN = "jointspace_planning"
    PREVIEW = "isolated_prefix_validation"
    EXECUTE = "native_osc_execution"
    HANDOFF = "task_handoff"
    VERIFY = "task_success_check"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class GoalSpec:
    """A task-stage goal with explicit provenance and contact semantics."""

    goal_id: str
    phase: str
    source: str
    source_artifacts: tuple[str, ...]
    position_world_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    allowed_target_contacts: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.goal_id or not self.phase or not self.source:
            raise ValueError("goal_id, phase, and source must be explicit")
        if self.source in {"vlm_teacher", "teacher"} and not self.source_artifacts:
            raise ValueError("teacher-derived goals require teacher evidence artifacts")
        position = np.asarray(self.position_world_m, dtype=np.float64)
        quaternion = np.asarray(self.quaternion_wxyz, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("goal position must be finite world XYZ")
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("goal quaternion must be finite WXYZ")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12 or abs(norm - 1.0) > 1e-4:
            raise ValueError("goal quaternion must be normalized WXYZ")


@dataclass(frozen=True)
class TaskContext:
    """Verified scene constraints and task-specific recovery-stage inputs."""

    task_id: str
    instruction: str
    forbidden_contact_objects: tuple[str, ...]
    allowed_target_contact_objects: tuple[str, ...]
    held_object: str | None
    phase: str
    target_source: str
    safety_policy_source: str
    goals: tuple[GoalSpec, ...]
    semantic_bindings_verified: bool
    safety_constraints_verified: bool
    grasp_state_verified: bool
    expert_handoff_available: bool = False
    handoff_policy: str = "student"
    post_recovery_policy: str | None = None
    unresolved_goal_phases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.task_id or not self.instruction or not self.phase:
            raise ValueError("task identity, instruction, and phase are required")
        if not self.semantic_bindings_verified or not self.safety_constraints_verified:
            raise ValueError("task semantic bindings and safety constraints must be verified")
        if not self.safety_policy_source or not self.target_source:
            raise ValueError("safety policy and goal sources must be explicit")
        if not self.grasp_state_verified:
            raise ValueError("held-object state is unknown; recovery cannot guess")
        if self.handoff_policy not in {"student", "expert"}:
            raise ValueError("handoff_policy must be student or expert")
        if self.handoff_policy == "expert" and not self.expert_handoff_available:
            raise ValueError("expert handoff was selected but no task expert is declared")
        if self.post_recovery_policy not in {None, "student", "expert"}:
            raise ValueError("post_recovery_policy must be student, expert, or unset")
        if self.post_recovery_policy == "expert" and not self.expert_handoff_available:
            raise ValueError("expert post-recovery policy was selected without a task expert")
        if any(not phase for phase in self.unresolved_goal_phases):
            raise ValueError("unresolved goal phase names must be explicit")
        if self.phase in self.unresolved_goal_phases:
            raise ValueError(f"task_goal_phase_unresolved:{self.phase}")
        if not self.goals:
            raise ValueError("task context has no verified phase goals")
        for goal in self.goals:
            goal.validate()
        self.goal_for_phase(self.phase)
        overlap = set(self.forbidden_contact_objects) & set(
            self.allowed_target_contact_objects
        )
        if overlap:
            raise ValueError(f"contact objects cannot be both forbidden and allowed: {sorted(overlap)}")

    def goal_for_phase(self, phase: str) -> GoalSpec:
        matches = [goal for goal in self.goals if goal.phase == phase]
        if len(matches) != 1:
            raise ValueError(f"phase {phase!r} has {len(matches)} verified goals")
        return matches[0]


@dataclass(frozen=True)
class RecoveryConfig:
    """Global bounds and hard safety limits for one autonomous run."""

    segment_control_steps: int = 8
    risk_trigger_distance_m: float = 0.05
    risk_rearm_distance_m: float = 0.06
    hard_minimum_clearance_m: float = 0.01
    target_object_clearance_m: float = 0.01
    object_clearance_m: dict[str, float] = field(default_factory=dict)
    maximum_replans: int = 8
    maximum_replans_per_phase: int = 12
    planning_time_budget_s: float = 360.0
    execution_step_budget: int = 1200
    total_wall_budget_s: float = 1800.0
    maximum_joint_path_deviation_rad: float = 0.18
    maximum_eef_tracking_error_m: float = 0.025
    maximum_eef_orientation_error_rad: float = 0.08
    stagnation_window_steps: int = 12
    maximum_same_state_handoffs: int = 3
    minimum_progress_m: float = 0.002
    horizon_steps: int = 1000
    standard_horizon_steps: int = 1000
    allow_horizon_extension: bool = False

    def validate(self) -> None:
        for name in (
            "segment_control_steps",
            "maximum_replans",
            "maximum_replans_per_phase",
            "execution_step_budget",
            "stagnation_window_steps",
            "maximum_same_state_handoffs",
            "horizon_steps",
            "standard_horizon_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "risk_trigger_distance_m",
            "risk_rearm_distance_m",
            "planning_time_budget_s",
            "total_wall_budget_s",
            "maximum_joint_path_deviation_rad",
            "maximum_eef_tracking_error_m",
            "maximum_eef_orientation_error_rad",
            "minimum_progress_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("hard_minimum_clearance_m", "target_object_clearance_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.object_clearance_m, dict):
            raise ValueError("object_clearance_m must be an object-id-to-margin mapping")
        for object_id, margin in self.object_clearance_m.items():
            if not isinstance(object_id, str) or not object_id:
                raise ValueError("object_clearance_m keys must be non-empty object ids")
            if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(float(margin)) or float(margin) < 0.0:
                raise ValueError(f"object_clearance_m[{object_id!r}] must be finite and non-negative")
        if self.risk_trigger_distance_m <= self.hard_minimum_clearance_m:
            raise ValueError("risk trigger must be greater than the hard clearance")
        if self.risk_rearm_distance_m <= self.risk_trigger_distance_m:
            raise ValueError("risk rearm distance must be greater than the entry distance")


@dataclass
class RiskBoundaryState:
    """Hysteretic entry/exit state for one continuously monitored risk band.

    The state persists across route replans. Entering the risk band can request
    one replan; another request is armed only after measured clearance exits
    beyond ``exit_distance_m``. Collision monitoring itself remains active on
    every physical substep regardless of this state.
    """

    entry_distance_m: float
    exit_distance_m: float
    armed: bool

    @classmethod
    def from_clearance(
        cls, clearance_m: float, entry_distance_m: float, exit_distance_m: float
    ) -> "RiskBoundaryState":
        return cls(
            entry_distance_m=float(entry_distance_m),
            exit_distance_m=float(exit_distance_m),
            armed=bool(float(clearance_m) > float(entry_distance_m)),
        )

    def update(self, clearance_m: float) -> str | None:
        """Report entry or exit from the risk band using live clearance."""
        clearance = float(clearance_m)
        if not math.isfinite(clearance):
            raise ValueError("risk_boundary_clearance_is_not_finite")
        if not self.armed:
            if clearance >= self.exit_distance_m:
                self.armed = True
                return "exited"
            return None
        if clearance <= self.entry_distance_m:
            self.armed = False
            return "entered"
        return None


@dataclass
class Observation:
    """One observed physical state; safety coverage must be complete."""

    timestep: int
    qpos: np.ndarray
    eef_position_world_m: np.ndarray
    eef_quaternion_wxyz: np.ndarray
    minimum_forbidden_clearance_m: float | None
    clearance_complete: bool
    contacts_complete: bool
    forbidden_contacts: tuple[dict[str, Any], ...] = ()
    self_contacts: tuple[dict[str, Any], ...] = ()
    physics_substeps: tuple[dict[str, Any], ...] = ()
    reward: float = 0.0
    done: bool = False
    task_success: bool | None = False
    raw: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float64)
        pos = np.asarray(self.eef_position_world_m, dtype=np.float64)
        quat = np.asarray(self.eef_quaternion_wxyz, dtype=np.float64)
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise ValueError("observed arm qpos must be finite 7D")
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError("observed EEF position must be finite world XYZ")
        if quat.shape != (4,) or not np.isfinite(quat).all():
            raise ValueError("observed EEF quaternion must be finite WXYZ")
        if not self.clearance_complete or not self.contacts_complete:
            raise ValueError("complete geometry and contact monitoring is required")
        if self.minimum_forbidden_clearance_m is None or not math.isfinite(
            float(self.minimum_forbidden_clearance_m)
        ):
            raise ValueError("forbidden-object clearance is unknown")
        if self.task_success is None:
            raise ValueError("task success state is unknown")


@dataclass
class JointSpacePlan:
    """Dense, verified 7D joint-space route from the current physical state."""

    qpos_path: np.ndarray
    minimum_forbidden_clearance_m: float
    planner_status: str
    planner_source: str
    planning_seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self, start_qpos: np.ndarray, goal: GoalSpec, config: RecoveryConfig) -> None:
        path = np.asarray(self.qpos_path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2:
            raise ValueError("RRT-Connect plan must contain at least two 7D configurations")
        if not np.isfinite(path).all():
            raise ValueError("RRT-Connect path contains non-finite joint values")
        if not np.allclose(path[0], start_qpos, rtol=0, atol=1e-6):
            raise ValueError("RRT-Connect path does not begin at current real qpos")
        if self.planner_status != "offline_feasible":
            raise ValueError(f"RRT-Connect path is not certified feasible: {self.planner_status}")
        if not self.planner_source:
            raise ValueError("planner provenance is missing")
        if not math.isfinite(float(self.planning_seconds)) or self.planning_seconds < 0:
            raise ValueError("planning time is invalid")
        if self.planning_seconds > config.planning_time_budget_s:
            raise ValueError("planner exceeded its configured time budget")
        if (
            not math.isfinite(float(self.minimum_forbidden_clearance_m))
            or self.minimum_forbidden_clearance_m < config.hard_minimum_clearance_m
        ):
            raise ValueError("planned path violates the hard clearance requirement")
        goal.validate()


@dataclass
class PrefixPreview:
    """Isolated Snapshot execution result for one proposed OSC prefix."""

    actions: np.ndarray
    observations: list[Observation]
    complete: bool
    main_state_restored_verified: bool
    safe: bool
    failure_reason: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


class RecoveryBackend(Protocol):
    """Simulator, planner, Student, task expert, and artifact integration boundary."""

    def current_context(self) -> TaskContext: ...

    def observe(self) -> Observation: ...

    def student_action(self, observation: Observation) -> np.ndarray: ...

    def predict_student_risk(
        self, observation: Observation, action: np.ndarray, context: TaskContext
    ) -> dict[str, Any]: ...

    def execute_action(self, action: np.ndarray, context: TaskContext) -> Observation: ...

    def resolve_goal(self, observation: Observation, context: TaskContext) -> GoalSpec: ...

    def plan_rrt_connect(
        self,
        observation: Observation,
        goal: GoalSpec,
        context: TaskContext,
        config: RecoveryConfig,
    ) -> JointSpacePlan: ...

    def preview_prefix(
        self,
        observation: Observation,
        plan: JointSpacePlan,
        context: TaskContext,
        config: RecoveryConfig,
        *,
        risk_boundary_armed: bool = True,
    ) -> PrefixPreview: ...

    def task_expert_action(
        self, observation: Observation, context: TaskContext
    ) -> np.ndarray | None: ...

    def handoff_to_student(self, observation: Observation, context: TaskContext) -> None: ...

    def handoff_to_task_expert(self, observation: Observation, context: TaskContext) -> None: ...

    def is_recovery_goal_reached(
        self, observation: Observation, goal: GoalSpec
    ) -> bool: ...

    def save_snapshot(self, label: str) -> str | None: ...

    def record_event(self, event: dict[str, Any]) -> None: ...


class RecoveryArtifactWriter:
    """Append machine-readable decisions and preserve supplied snapshots."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "recovery_events.jsonl"
        self.manifest_path = self.output_dir / "recovery_manifest.json"

    @staticmethod
    def _safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "value") and isinstance(value.value, str):
            return value.value
        if isinstance(value, dict):
            return {str(key): RecoveryArtifactWriter._safe(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [RecoveryArtifactWriter._safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self._safe(event), ensure_ascii=False, sort_keys=True) + "\n")

    def manifest(self, data: dict[str, Any]) -> None:
        self.manifest_path.write_text(
            json.dumps(self._safe(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass
class RecoveryResult:
    status: str
    state: RecoveryState
    local_recovery_success: bool
    complete_task_success: bool
    complete_safety_success: bool
    standard_horizon_success: bool
    extended_horizon: bool
    steps_executed: int
    initial_episode_timestep: int
    final_episode_timestep: int
    replans: int
    reason: str | None
    phase: str
    artifacts: dict[str, Any] = field(default_factory=dict)


class AutomaticSafetyRecoveryEngine:
    """Run Student, safety takeover, replanning, and verified task handoff."""

    def __init__(
        self,
        backend: RecoveryBackend,
        config: RecoveryConfig,
        *,
        artifacts: RecoveryArtifactWriter | None = None,
    ):
        config.validate()
        self.backend = backend
        self.config = config
        self.artifacts = artifacts
        self.state = RecoveryState.STUDENT
        self.steps = 0
        self.replans = 0
        self._events: list[dict[str, Any]] = []
        self._start_time = 0.0
        self._safe_transitions = 0
        self._physical_transitions = 0
        self._failure_reason: str | None = None
        self._local_recovery_success = False
        self._ever_local_recovery_success = False
        self._extended_horizon = False
        self._phase = "unknown"
        self._progress_distances: list[float] = []
        self._last_progress_eef_position: np.ndarray | None = None
        self._active_goal_position: np.ndarray | None = None
        self._initial_timestep = 0
        self._initial_history_safety_verified = False
        self._same_state_handoff_count = 0
        self._task_milestone: dict[str, Any] | None = None

    def _check_task_milestone(self, observation: Observation, action_source: str) -> bool:
        """Honor an optional backend-local stop milestone after a safe transition."""
        callback = getattr(self.backend, "task_milestone_reached", None)
        if not callable(callback):
            return False
        milestone = callback(observation)
        if not isinstance(milestone, dict) or not milestone:
            return False
        self._task_milestone = milestone
        self.state = RecoveryState.COMPLETE
        self._event(
            "task_milestone_reached",
            milestone=milestone,
            action_source=action_source,
            timestep=int(observation.timestep),
        )
        return True

    def _event(self, kind: str, **details: Any) -> None:
        event = {
            "event": kind,
            "state": self.state.value,
            "step": self.steps,
            "replan_count": self.replans,
            "phase": self._phase,
            "time_monotonic_s": time.monotonic() - self._start_time,
            **details,
        }
        self._events.append(event)
        if self.artifacts is not None:
            self.artifacts.event(event)
        self.backend.record_event(event)

    def _fail(self, reason: str, **details: Any) -> None:
        self._failure_reason = reason
        self.state = RecoveryState.FAILED
        self._event("fail_closed", reason=reason, **details)

    def _budget_exhausted(self) -> str | None:
        if self.steps >= self.config.execution_step_budget:
            return "execution_step_budget_exhausted"
        if self.steps >= self.config.horizon_steps and not self.config.allow_horizon_extension:
            return "standard_1000_step_horizon_exhausted"
        if time.monotonic() - self._start_time >= self.config.total_wall_budget_s:
            return "total_wall_budget_exhausted"
        return None

    @staticmethod
    def _safety_check(observation: Observation, config: RecoveryConfig) -> str | None:
        try:
            observation.validate()
        except Exception as exc:
            return f"unknown_physical_safety_state:{type(exc).__name__}:{exc}"
        if observation.forbidden_contacts:
            return "forbidden_contact_observed"
        if observation.self_contacts:
            return "robot_self_contact_observed"
        if observation.minimum_forbidden_clearance_m < config.hard_minimum_clearance_m:
            return "hard_clearance_margin_violated"
        if observation.physics_substeps:
            expected = observation.raw.get("expected_physics_substeps")
            if expected is None or len(observation.physics_substeps) != expected:
                return "physical_substep_coverage_incomplete"
            if any(item.get("status") != "complete" for item in observation.physics_substeps):
                return "physical_substep_safety_record_incomplete"
        return None

    @staticmethod
    def _quaternion_error(left: np.ndarray, right: np.ndarray) -> float:
        a = np.asarray(left, dtype=np.float64)
        b = np.asarray(right, dtype=np.float64)
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        if a.shape != (4,) or b.shape != (4,) or a_norm < 1e-12 or b_norm < 1e-12:
            raise ValueError("EEF orientation comparison received an invalid quaternion")
        dot = float(np.clip(abs(np.dot(a / a_norm, b / b_norm)), 0.0, 1.0))
        return 2.0 * math.acos(dot)

    def _record_transition(self, action: np.ndarray, observation: Observation, source: str) -> None:
        self.steps = max(self.steps + 1, int(observation.timestep))
        self._physical_transitions += 1
        self._extended_horizon = self.steps > self.config.standard_horizon_steps
        reason = self._safety_check(observation, self.config)
        if reason:
            self._fail(reason, action_source=source, observation=observation.raw)
            return
        self._safe_transitions += 1
        self._event(
            "physical_transition",
            action_source=source,
            action=np.asarray(action, dtype=np.float32),
            qpos=observation.qpos,
            eef_position_world_m=observation.eef_position_world_m,
            reward=observation.reward,
            done=observation.done,
            clearance_m=observation.minimum_forbidden_clearance_m,
            forbidden_contacts=observation.forbidden_contacts,
            self_contacts=observation.self_contacts,
            physics_substeps=observation.physics_substeps,
            raw_observation=observation.raw,
        )
        if self._active_goal_position is not None:
            current_position = np.asarray(observation.eef_position_world_m, dtype=np.float64)
            if self._last_progress_eef_position is not None:
                self._progress_distances.append(
                    float(np.linalg.norm(current_position - self._last_progress_eef_position))
                )
            self._last_progress_eef_position = current_position.copy()
            if len(self._progress_distances) > self.config.stagnation_window_steps:
                self._progress_distances.pop(0)

    def _run_policy_until_risk(self, context: TaskContext, policy: str) -> Observation:
        self.state = RecoveryState.STUDENT if policy == "student" else RecoveryState.HANDOFF
        while True:
            context = self.backend.current_context()
            self._phase = context.phase
            context.validate()
            exhausted = self._budget_exhausted()
            if exhausted:
                self._fail(exhausted)
                raise RuntimeError(exhausted)
            observation = self.backend.observe()
            failure = self._safety_check(observation, self.config)
            if failure:
                self._fail(failure, observation=observation.raw)
                raise RuntimeError(failure)
            if observation.task_success is True:
                self.state = RecoveryState.COMPLETE
                return observation
            if policy == "expert" and context.metadata.get("requires_rrt_recovery") is True:
                self._event(
                    "task_phase_rrt_route_requested",
                    phase=context.phase,
                    goal=asdict(context.goal_for_phase(context.phase)),
                    source=context.metadata.get("task_goal_source"),
                )
                return observation
            if policy == "student":
                raw_action = self.backend.student_action(observation)
            else:
                raw_action = self.backend.task_expert_action(observation, context)
                if raw_action is None:
                    self._fail("task_expert_has_no_current_state_action")
                    raise RuntimeError(self._failure_reason)
            action = np.asarray(raw_action, dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                self._fail("student_action_must_be_finite_7d")
                raise RuntimeError(self._failure_reason)
            self.state = RecoveryState.ASSESS
            risk = self.backend.predict_student_risk(observation, action, context)
            if not isinstance(risk, dict) or risk.get("status") != "complete":
                self._fail("student_risk_prediction_unknown", risk=risk)
                raise RuntimeError(self._failure_reason)
            distance = risk.get("minimum_predicted_forbidden_clearance_m")
            if distance is None or not math.isfinite(float(distance)):
                self._fail("student_risk_distance_unknown", risk=risk)
                raise RuntimeError(self._failure_reason)
            self._event(f"{policy}_risk_assessed", risk=risk)
            if policy == "expert":
                coverage = risk.get("substep_coverage")
                geometry = (
                    coverage.get("geometry_monitoring", {})
                    if isinstance(coverage, dict) else {}
                )
                expected_substeps = (
                    int(coverage.get("expected_physics_substeps_total", 0))
                    if isinstance(coverage, dict) else 0
                )
                all_contacts_observed = (
                    isinstance(coverage, dict)
                    and int(coverage.get("captured_physics_substeps", -1)) == expected_substeps
                    and int(coverage.get("contact_substeps", -1)) == expected_substeps
                )
                known_unsafe_prefix = bool(
                    all_contacts_observed
                    and expected_substeps > 0
                    and risk.get("main_state_restored_verified") is True
                    and risk.get("predicted_safety_status") == "unsafe"
                    and geometry.get("status") == "unsafe"
                    and len(geometry.get("substeps", [])) == expected_substeps
                )
                if known_unsafe_prefix:
                    self._event(
                        "automatic_takeover",
                        reason="task_expert_prefix_known_unsafe_replan_from_live_state",
                        risk=risk,
                    )
                    return observation
                if (
                    risk.get("main_state_restored_verified") is not True
                    or not isinstance(coverage, dict)
                    or coverage.get("status") != "complete"
                    or geometry.get("status") != "complete"
                    or not all_contacts_observed
                ):
                    self._fail(
                        "task_expert_isolated_prefix_validation_unknown",
                        risk=risk,
                    )
                    raise RuntimeError(self._failure_reason)
                if float(distance) < self.config.hard_minimum_clearance_m:
                    self._event(
                        "automatic_takeover",
                        reason="task_expert_prefix_requires_replanning",
                        risk=risk,
                    )
                    return observation
                if risk.get("predicted_safety_status") != "complete":
                    self._fail("task_expert_predicted_safety_status_unknown", risk=risk)
                    raise RuntimeError(self._failure_reason)
                if float(distance) <= self.config.risk_trigger_distance_m:
                    self._event(
                        "task_expert_safe_action_inside_risk_band",
                        clearance_m=float(distance),
                        entry_distance_m=self.config.risk_trigger_distance_m,
                        hard_minimum_clearance_m=self.config.hard_minimum_clearance_m,
                        phase=context.phase,
                    )
            elif float(distance) <= self.config.risk_trigger_distance_m:
                self._event("automatic_takeover", risk=risk)
                return observation
            next_observation = self.backend.execute_action(action, context)
            self._record_transition(action, next_observation, policy)
            if self.state is RecoveryState.FAILED:
                raise RuntimeError(self._failure_reason)
            if self._check_task_milestone(next_observation, policy):
                return next_observation

    def _execute_recovery(self, observation: Observation, context: TaskContext) -> Observation:
        self.state = RecoveryState.RESOLVE
        goal = self.backend.resolve_goal(observation, context)
        goal.validate()
        self._phase = goal.phase
        self._active_goal_position = np.asarray(goal.position_world_m, dtype=np.float64)
        self._progress_distances = []
        self._last_progress_eef_position = np.asarray(
            observation.eef_position_world_m, dtype=np.float64
        ).copy()
        self._event(
            "recovery_goal_resolved",
            goal=asdict(goal),
            target_source=context.target_source,
            safety_policy_source=context.safety_policy_source,
        )
        if self.backend.is_recovery_goal_reached(observation, goal):
            self._local_recovery_success = True
            self._ever_local_recovery_success = True
            return observation
        risk_boundary = RiskBoundaryState.from_clearance(
            observation.minimum_forbidden_clearance_m,
            self.config.risk_trigger_distance_m,
            self.config.risk_rearm_distance_m,
        )
        phase_replans: dict[str, int] = {}
        last_replan_reason: dict[str, str] = {}
        while not self.backend.is_recovery_goal_reached(observation, goal):
            exhausted = self._budget_exhausted()
            if exhausted:
                self._fail(exhausted)
                break
            if self.replans >= self.config.maximum_replans:
                self._fail("maximum_replanning_attempts_exhausted")
                break
            plan_start_observation = observation
            self.state = RecoveryState.PLAN
            plan_started = time.monotonic()
            try:
                plan = self.backend.plan_rrt_connect(observation, goal, context, self.config)
            except Exception as exc:
                advance_candidate = getattr(
                    self.backend, "advance_task_goal_candidate", None
                )
                if (
                    "goal_configuration_is_unsafe" in str(exc)
                    and callable(advance_candidate)
                    and advance_candidate(context, str(exc))
                ):
                    previous_goal = goal
                    context = self.backend.current_context()
                    context.validate()
                    goal = self.backend.resolve_goal(observation, context)
                    goal.validate()
                    self._phase = context.phase
                    self._active_goal_position = np.asarray(
                        goal.position_world_m, dtype=np.float64
                    )
                    self._progress_distances = []
                    self._last_progress_eef_position = np.asarray(
                        observation.eef_position_world_m, dtype=np.float64
                    ).copy()
                    self._event(
                        "task_goal_candidate_advanced_after_known_unsafe_goal",
                        previous_goal=asdict(previous_goal),
                        next_goal=asdict(goal),
                        failure_reason=str(exc),
                    )
                    continue
                self._fail(f"rrt_connect_failed:{type(exc).__name__}:{exc}")
                break
            planning_seconds = time.monotonic() - plan_started
            if planning_seconds > self.config.planning_time_budget_s:
                self._fail("planning_time_budget_exhausted", planning_seconds=planning_seconds)
                break
            try:
                plan.validate(observation.qpos, goal, self.config)
            except Exception as exc:
                self._fail(f"invalid_rrt_connect_plan:{type(exc).__name__}:{exc}")
                break
            self._event(
                "rrt_connect_plan_accepted",
                path=plan.qpos_path,
                planning_seconds=planning_seconds,
                planner_diagnostics=plan.diagnostics,
                minimum_forbidden_clearance_m=plan.minimum_forbidden_clearance_m,
            )
            # A new route establishes a fresh progress baseline. Movement
            # around an obstacle can temporarily increase straight-line goal
            # distance while still making valid path progress.
            self._progress_distances = []
            self._last_progress_eef_position = np.asarray(
                observation.eef_position_world_m, dtype=np.float64
            ).copy()
            while not self.backend.is_recovery_goal_reached(observation, goal):
                self.state = RecoveryState.PREVIEW
                preview = self.backend.preview_prefix(
                    observation,
                    plan,
                    context,
                    self.config,
                    risk_boundary_armed=risk_boundary.armed,
                )
                actions = np.asarray(preview.actions, dtype=np.float32)
                if (
                    not preview.complete
                    or not preview.main_state_restored_verified
                    or actions.ndim != 2
                    or actions.shape[1:] != (7,)
                    or len(actions) == 0
                    or len(actions) > self.config.segment_control_steps
                    or len(preview.observations) != len(actions)
                    or not np.isfinite(actions).all()
                ):
                    self._fail(
                        "isolated_prefix_validation_failed_or_unknown",
                        failure_reason=preview.failure_reason,
                        artifacts=preview.artifacts,
                    )
                    break
                if not preview.safe:
                    if preview.artifacts.get("retryable_safety_rejection") is True:
                        self._event(
                            "isolated_prefix_rejected_replan_from_restored_state",
                            failure_reason=preview.failure_reason,
                            artifacts=preview.artifacts,
                        )
                        self.replans += 1
                        break
                    self._fail(
                        "isolated_prefix_safety_verdict_is_unknown",
                        failure_reason=preview.failure_reason,
                        artifacts=preview.artifacts,
                    )
                    break
                preview_stop_reason = preview.artifacts.get("stop_reason")
                recognized_replan_stops = {
                    "narrow_channel_entry_replan",
                    "planned_joint_or_eef_reference_deviation",
                    "motion_stagnation",
                }
                if preview_stop_reason is not None and preview_stop_reason not in recognized_replan_stops:
                    self._fail(
                        "isolated_prefix_stop_reason_is_unknown",
                        stop_reason=preview_stop_reason,
                        artifacts=preview.artifacts,
                    )
                    break
                self._event(
                    "isolated_prefix_validated",
                    actions=actions,
                    preview_observations=[item.raw for item in preview.observations],
                    artifacts=preview.artifacts,
                )
                self.state = RecoveryState.EXECUTE
                replan_required = False
                replan_reason: str | None = None
                for index, action in enumerate(actions):
                    exhausted = self._budget_exhausted()
                    if exhausted:
                        self._fail(exhausted)
                        break
                    expected = preview.observations[index]
                    actual = self.backend.execute_action(action, context)
                    self._record_transition(action, actual, "safety_recovery_osc_pose")
                    if self.state is RecoveryState.FAILED:
                        break
                    if self._check_task_milestone(actual, "safety_recovery_osc_pose"):
                        observation = actual
                        break
                    risk_transition = risk_boundary.update(
                        actual.minimum_forbidden_clearance_m
                    )
                    if risk_transition == "exited":
                        self._event(
                            "risk_boundary_exited_and_rearmed",
                            clearance_m=actual.minimum_forbidden_clearance_m,
                            exit_distance_m=risk_boundary.exit_distance_m,
                        )
                    elif risk_transition == "entered":
                        self._event(
                            "risk_boundary_entered",
                            clearance_m=actual.minimum_forbidden_clearance_m,
                            entry_distance_m=risk_boundary.entry_distance_m,
                        )
                    commit_q_deviation = float(np.linalg.norm(actual.qpos - expected.qpos))
                    commit_eef_deviation = float(
                        np.linalg.norm(actual.eef_position_world_m - expected.eef_position_world_m)
                    )
                    target_qpos = expected.raw.get("reference_qpos")
                    target_eef = expected.raw.get("reference_eef_pose")
                    reference_q_deviation = (
                        float(np.linalg.norm(actual.qpos - np.asarray(target_qpos, dtype=np.float64)))
                        if target_qpos is not None
                        else 0.0
                    )
                    reference_eef_deviation = (
                        float(np.linalg.norm(actual.eef_position_world_m - np.asarray(target_eef, dtype=np.float64)[:3]))
                        if target_eef is not None
                        else 0.0
                    )
                    reference_eef_orientation_error = (
                        self._quaternion_error(
                            actual.eef_quaternion_wxyz,
                            np.asarray(target_eef, dtype=np.float64)[3:],
                        )
                        if target_eef is not None
                        else 0.0
                    )
                    self._event(
                        "preview_commit_comparison",
                        prefix_index=index,
                        commit_qpos_l2_deviation_rad=commit_q_deviation,
                        commit_eef_deviation_m=commit_eef_deviation,
                        reference_qpos_l2_deviation_rad=reference_q_deviation,
                        reference_eef_deviation_m=reference_eef_deviation,
                        reference_eef_orientation_error_rad=reference_eef_orientation_error,
                    )
                    if (
                        commit_q_deviation > 1e-3
                        or commit_eef_deviation > 1e-3
                        or reference_q_deviation > self.config.maximum_joint_path_deviation_rad
                        or reference_eef_deviation > self.config.maximum_eef_tracking_error_m
                        or reference_eef_orientation_error > self.config.maximum_eef_orientation_error_rad
                    ):
                        self._event("automatic_replan_triggered", reason="preview_tracking_deviation")
                        observation = actual
                        replan_required = True
                        replan_reason = "preview_tracking_deviation"
                        break
                    if risk_transition == "entered":
                        self._event("automatic_replan_triggered", reason="risk_margin_approaching")
                        observation = actual
                        replan_required = True
                        replan_reason = "risk_margin_approaching"
                        break
                    observation = actual
                    if self.backend.is_recovery_goal_reached(observation, goal):
                        self._local_recovery_success = True
                        self._ever_local_recovery_success = True
                        break
                    goal_distance = float(
                        np.linalg.norm(observation.eef_position_world_m - np.asarray(goal.position_world_m))
                    )
                    if (
                        len(self._progress_distances) == self.config.stagnation_window_steps
                        and sum(self._progress_distances) < self.config.minimum_progress_m
                    ):
                        self._event("automatic_replan_triggered", reason="motion_stagnation",
                                    eef_travel_window_m=self._progress_distances)
                        replan_required = True
                        replan_reason = "motion_stagnation"
                        break
                if self.state is RecoveryState.FAILED:
                    break
                if self._local_recovery_success:
                    break
                if (
                    not replan_required
                    and preview_stop_reason in recognized_replan_stops
                ):
                    self._event(
                        "automatic_replan_triggered",
                        reason=preview_stop_reason,
                        source="isolated_prefix_validation",
                    )
                    replan_required = True
                    replan_reason = str(preview_stop_reason)
                path_exhausted = (
                    preview.artifacts.get("last_path_index")
                    == len(plan.qpos_path) - 1
                    and not self.backend.is_recovery_goal_reached(observation, goal)
                )
                if path_exhausted and not replan_required:
                    self._event(
                        "automatic_replan_triggered",
                        reason="planned_route_exhausted_before_goal",
                    )
                    replan_required = True
                    replan_reason = "planned_route_exhausted_before_goal"
                if replan_required:
                    if replan_reason is None:
                        self._fail("replan_reason_unknown_fail_closed", phase=context.phase)
                        break
                    phase = context.phase
                    moved_eef = float(
                        np.linalg.norm(
                            observation.eef_position_world_m
                            - plan_start_observation.eef_position_world_m
                        )
                    )
                    moved_qpos = float(
                        np.linalg.norm(observation.qpos - plan_start_observation.qpos)
                    )
                    clearance_change = abs(
                        observation.minimum_forbidden_clearance_m
                        - plan_start_observation.minimum_forbidden_clearance_m
                    )
                    progressed = max(moved_eef, clearance_change) >= self.config.minimum_progress_m or moved_qpos >= 0.01
                    if not progressed:
                        self._fail(
                            "replan_without_physical_progress",
                            phase=phase,
                            replan_reason=replan_reason,
                            eef_displacement_m=moved_eef,
                            qpos_displacement_rad=moved_qpos,
                            clearance_change_m=clearance_change,
                        )
                        break
                    if last_replan_reason.get(phase) == replan_reason:
                        self._fail(
                            "repeated_replan_failure_reason",
                            phase=phase,
                            replan_reason=replan_reason,
                        )
                        break
                    used = phase_replans.get(phase, 0)
                    if used >= self.config.maximum_replans_per_phase:
                        self._fail(
                            "phase_replanning_budget_exhausted",
                            phase=phase,
                            maximum_replans_per_phase=self.config.maximum_replans_per_phase,
                        )
                        break
                    phase_replans[phase] = used + 1
                    last_replan_reason[phase] = replan_reason
                    self.replans += 1
                    self._event(
                        "productive_phase_replan_budget_consumed",
                        phase=phase,
                        reason=replan_reason,
                        phase_replans=phase_replans[phase],
                        maximum_replans_per_phase=self.config.maximum_replans_per_phase,
                        moved_eef_m=moved_eef,
                        moved_qpos_rad=moved_qpos,
                        clearance_change_m=clearance_change,
                        risk_boundary_armed=risk_boundary.armed,
                    )
                    break
                if not self.backend.is_recovery_goal_reached(observation, goal):
                    self._event(
                        "recovery_segment_continued",
                        controls_committed=len(actions),
                        plan_index=preview.artifacts.get("plan_index"),
                        last_path_index=preview.artifacts.get("last_path_index"),
                    )
            if self.state is RecoveryState.FAILED or self._local_recovery_success:
                break
        return observation

    def run(self) -> RecoveryResult:
        """Run autonomously until task success, a fail-closed condition, or budget."""
        self._start_time = time.monotonic()
        try:
            context = self.backend.current_context()
            context.validate()
            self._phase = context.phase
            self._event(
                "run_started",
                context=asdict(context),
                config=asdict(self.config),
                goal_sources={goal.phase: goal.source for goal in context.goals},
            )
            observation = self.backend.observe()
            self.steps = max(0, int(observation.timestep))
            self._initial_timestep = self.steps
            self._initial_history_safety_verified = bool(
                context.metadata.get("initial_history_safety_verified", self.steps == 0)
            )
            initial_safety_failure = self._safety_check(observation, self.config)
            if initial_safety_failure:
                self._fail(
                    initial_safety_failure,
                    initial_state_safety={
                        "minimum_forbidden_clearance_m": observation.minimum_forbidden_clearance_m,
                        "forbidden_contacts": observation.forbidden_contacts,
                        "self_contacts": observation.self_contacts,
                        "raw_safety_geometry": observation.raw.get("safety_geometry"),
                        "raw_contact_monitoring": observation.raw.get("contact_monitoring"),
                    },
                )
            elif observation.task_success is not True:
                initial_goal = context.goal_for_phase(context.phase)
                if self.backend.is_recovery_goal_reached(observation, initial_goal):
                    self._local_recovery_success = True
                    self._ever_local_recovery_success = True
                    self._phase = initial_goal.phase
                    self._event(
                        "recovery_goal_already_satisfied",
                        goal=asdict(initial_goal),
                        source="verified_current_physical_state",
                    )
            while self.state not in (RecoveryState.COMPLETE, RecoveryState.FAILED):
                if self.state is RecoveryState.FAILED:
                    break
                cycle_start_timestep = int(observation.timestep)
                cycle_start_phase = context.phase
                cycle_start_transitions = self._physical_transitions
                if not self._local_recovery_success:
                    observation = self._run_policy_until_risk(context, context.handoff_policy)
                    if self.state is RecoveryState.COMPLETE:
                        break
                    context = self.backend.current_context()
                    context.validate()
                    observation = self._execute_recovery(observation, context)
                if self.state is RecoveryState.FAILED:
                    break
                if self._local_recovery_success:
                    self.state = RecoveryState.HANDOFF
                    self._event("local_recovery_succeeded", goal_phase=self._phase)
                    next_policy = context.post_recovery_policy or context.handoff_policy
                    if next_policy == "student":
                        handoff_context = self.backend.current_context()
                        handoff_context.validate()
                        self.backend.handoff_to_student(observation, handoff_context)
                    elif next_policy == "expert":
                        handoff_context = self.backend.current_context()
                        handoff_context.validate()
                        handoff = getattr(self.backend, "handoff_to_task_expert", None)
                        if handoff is None:
                            self._fail("task_expert_handoff_is_not_implemented")
                            break
                        handoff(observation, handoff_context)
                    context = self.backend.current_context()
                    context.validate()
                    self.state = RecoveryState.VERIFY
                    self._event("task_success_check_started", handoff_policy=next_policy)
                    if observation.task_success is True:
                        self.state = RecoveryState.COMPLETE
                        break
                    self.state = RecoveryState.STUDENT
                    self._local_recovery_success = False
                    if observation.done and observation.task_success is not True:
                        self._fail("episode_terminated_without_task_success")
                        break
                    same_state_handoff = (
                        self._physical_transitions == cycle_start_transitions
                        and int(observation.timestep) == cycle_start_timestep
                        and context.phase == cycle_start_phase
                    )
                    if same_state_handoff:
                        self._same_state_handoff_count += 1
                        self._event(
                            "same_state_handoff_without_physical_progress",
                            consecutive_count=self._same_state_handoff_count,
                            maximum_count=self.config.maximum_same_state_handoffs,
                            timestep=int(observation.timestep),
                            phase=context.phase,
                        )
                        if self._same_state_handoff_count >= self.config.maximum_same_state_handoffs:
                            self._fail(
                                "same_state_handoff_without_physical_progress_limit_exceeded",
                                consecutive_count=self._same_state_handoff_count,
                                maximum_count=self.config.maximum_same_state_handoffs,
                                timestep=int(observation.timestep),
                                phase=context.phase,
                            )
                            break
                    else:
                        self._same_state_handoff_count = 0
                if self.config.horizon_steps <= self.steps and not self.config.allow_horizon_extension:
                    self._fail("standard_1000_step_horizon_exhausted")
                elif self.steps > self.config.standard_horizon_steps and self.config.allow_horizon_extension:
                    self._extended_horizon = True
            if self.state is RecoveryState.COMPLETE:
                self._event("complete_task_success", task_success=True)
        except Exception as exc:
            if self.state is not RecoveryState.FAILED:
                message = str(exc)
                if message.startswith("task_goal_phase_unresolved:"):
                    self._fail(message)
                else:
                    self._fail(f"unhandled_engine_error:{type(exc).__name__}:{exc}")
        final_task_success = False
        try:
            final_observation = self.backend.observe()
            final_task_success = final_observation.task_success is True
            if self._safety_check(final_observation, self.config) is None:
                final_safe = bool(
                    self._initial_history_safety_verified
                    and self._safe_transitions == self.steps - self._initial_timestep
                )
            else:
                final_safe = False
        except Exception:
            final_safe = False
        status = (
            "task_success" if final_task_success
            else "task_milestone" if self._task_milestone is not None
            else "failed"
        )
        result = RecoveryResult(
            status=status,
            state=self.state,
            local_recovery_success=self._ever_local_recovery_success,
            complete_task_success=final_task_success,
            complete_safety_success=bool(final_task_success and final_safe),
            standard_horizon_success=bool(
                final_task_success and self.steps <= self.config.standard_horizon_steps
            ),
            extended_horizon=self._extended_horizon or self.steps > self.config.standard_horizon_steps,
            steps_executed=self._physical_transitions,
            initial_episode_timestep=self._initial_timestep,
            final_episode_timestep=self.steps,
            replans=self.replans,
            reason=self._failure_reason,
            phase=self._phase,
            artifacts={
                "events": len(self._events),
                **({"task_milestone": self._task_milestone} if self._task_milestone else {}),
            },
        )
        if self.artifacts is not None:
            self.artifacts.manifest({
                "schema_version": "gs_opd_automatic_safety_recovery.v1",
                "result": asdict(result),
                "config": asdict(self.config),
                "events_path": str(self.artifacts.events_path),
            })
        return result
