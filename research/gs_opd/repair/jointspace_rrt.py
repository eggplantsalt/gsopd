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

"""Reusable seven degree of freedom joint-space RRT-Connect planning."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JointSpacePlannerConfig:
    """Finite search bounds for bidirectional RRT-Connect."""

    step_rad: float = 0.22
    connect_max_steps: int = 20
    goal_bias: float = 0.10
    max_iterations: int = 2000
    max_config_checks: int = 25000
    max_candidate_routes: int = 3
    search_time_budget_s: float = 360.0
    shortcut_attempts: int = 250
    edge_max_joint_delta_rad: float = 0.02
    edge_max_eef_translation_m: float = 0.005
    edge_max_eef_rotation_rad: float = 0.02
    edge_max_geom_center_step_m: float = 0.004
    edge_max_geom_midpoint_curvature_m: float = 0.0005
    minimum_clearance_m: float = 0.01
    start_joint_limit_tolerance_rad: float = 0.0
    require_safe_configurations: bool = True
    seed: int = 20261023

    def validate(self) -> None:
        for name in (
            "connect_max_steps",
            "max_iterations",
            "max_config_checks",
            "max_candidate_routes",
            "shortcut_attempts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "step_rad",
            "search_time_budget_s",
            "edge_max_joint_delta_rad",
            "edge_max_eef_translation_m",
            "edge_max_eef_rotation_rad",
            "edge_max_geom_center_step_m",
            "edge_max_geom_midpoint_curvature_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.goal_bias < 1.0:
            raise ValueError("goal_bias must lie in (0, 1)")
        if not math.isfinite(float(self.minimum_clearance_m)) or self.minimum_clearance_m < 0.0:
            raise ValueError("planner clearance must be finite and non-negative")
        if (
            not math.isfinite(float(self.start_joint_limit_tolerance_rad))
            or self.start_joint_limit_tolerance_rad < 0.0
        ):
            raise ValueError("start joint-limit tolerance must be finite and non-negative")
        if not isinstance(self.require_safe_configurations, bool):
            raise ValueError("require_safe_configurations must be a bool")


@dataclass
class ConfigurationCheck:
    """Complete geometric and contact assessment for one 7D configuration."""

    status: str
    minimum_forbidden_clearance_m: float | None
    clearances_by_object_m: dict[str, float]
    forbidden_contacts: list[dict[str, Any]] = field(default_factory=list)
    self_contacts: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class KinematicSample:
    """EEF pose and every active collision Geom center at a joint state."""

    eef_position_world_m: np.ndarray
    eef_quaternion_wxyz: np.ndarray
    robot_geom_centers_world_m: np.ndarray


@dataclass
class JointSpacePlanResult:
    """RRT-Connect path and auditable complete path checks."""

    status: str
    qpos_path: np.ndarray
    minimum_forbidden_clearance_m: float | None
    planner_source: str
    planning_seconds: float
    config_checks: int
    iterations: int
    selected_route_index: int | None
    candidates: list[dict[str, Any]]
    failure_reason: str | None = None


class _Tree:
    def __init__(self, root: np.ndarray, name: str):
        self.nodes = [np.asarray(root, dtype=np.float64).copy()]
        self.parents = [-1]
        self.name = name

    def add(self, qpos: np.ndarray, parent: int) -> int:
        self.nodes.append(np.asarray(qpos, dtype=np.float64).copy())
        self.parents.append(int(parent))
        return len(self.nodes) - 1

    def path(self, index: int) -> list[np.ndarray]:
        chain = []
        while index >= 0:
            chain.append(index)
            index = self.parents[index]
        return [self.nodes[item].copy() for item in reversed(chain)]


class JointSpaceRRTConnect:
    """Plan current 7D qpos to a task-resolved qpos under full-Geom gates.

    ``configuration_checker`` must return ``status='complete'`` only when all
    configured forbidden objects, allowed contacts, and robot self-collisions
    have been checked. ``kinematics`` must use a scratch copy of the current
    integration state and return centers for every active robot collision Geom.
    """

    def __init__(
        self,
        qpos_lower: np.ndarray,
        qpos_upper: np.ndarray,
        configuration_checker: Callable[[np.ndarray], ConfigurationCheck],
        kinematics: Callable[[np.ndarray], KinematicSample],
        *,
        config: JointSpacePlannerConfig | None = None,
    ):
        self.lower = self._vector(qpos_lower, "qpos_lower")
        self.upper = self._vector(qpos_upper, "qpos_upper")
        if np.any(self.lower >= self.upper):
            raise ValueError("joint-space planning bounds must be strictly ordered")
        self.configuration_checker = configuration_checker
        self.kinematics = kinematics
        self.config = config or JointSpacePlannerConfig()
        self.config.validate()
        self._configuration_cache: dict[tuple[float, ...], ConfigurationCheck] = {}
        self._kinematic_cache: dict[tuple[float, ...], KinematicSample] = {}
        self._check_count = 0
        self._started = 0.0

    @staticmethod
    def _vector(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (7,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite 7D vector")
        return array

    @staticmethod
    def _key(qpos: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(qpos, dtype=np.float64), 10).tolist())

    def _check(self, qpos: np.ndarray) -> ConfigurationCheck:
        key = self._key(qpos)
        if key in self._configuration_cache:
            return self._configuration_cache[key]
        if self._check_count >= self.config.max_config_checks:
            raise TimeoutError("configuration_check_budget_exhausted")
        if time.monotonic() - self._started >= self.config.search_time_budget_s:
            raise TimeoutError("planning_time_budget_exhausted")
        self._check_count += 1
        result = self.configuration_checker(np.asarray(qpos, dtype=np.float64).copy())
        if not isinstance(result, ConfigurationCheck):
            raise TypeError("configuration_checker must return ConfigurationCheck")
        if result.status != "complete":
            raise RuntimeError("configuration_safety_status_unknown")
        if result.minimum_forbidden_clearance_m is None or not math.isfinite(
            float(result.minimum_forbidden_clearance_m)
        ):
            raise RuntimeError("configuration_forbidden_clearance_unknown")
        safe = (
            result.minimum_forbidden_clearance_m >= self.config.minimum_clearance_m
            and not result.forbidden_contacts
            and not result.self_contacts
        )
        if not safe:
            result.diagnostics = {**result.diagnostics, "configuration_safe": False}
        else:
            result.diagnostics = {**result.diagnostics, "configuration_safe": True}
        self._configuration_cache[key] = result
        return result

    def _configuration_admissible(self, result: ConfigurationCheck) -> bool:
        return not self.config.require_safe_configurations or bool(
            result.diagnostics.get("configuration_safe", False)
        )

    def _kinematics(self, qpos: np.ndarray) -> KinematicSample:
        key = self._key(qpos)
        if key not in self._kinematic_cache:
            result = self.kinematics(np.asarray(qpos, dtype=np.float64).copy())
            centers = np.asarray(result.robot_geom_centers_world_m, dtype=np.float64)
            position = np.asarray(result.eef_position_world_m, dtype=np.float64)
            quaternion = np.asarray(result.eef_quaternion_wxyz, dtype=np.float64)
            if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
                raise ValueError("kinematics must cover every robot collision Geom center")
            if position.shape != (3,) or quaternion.shape != (4,) or not all(
                np.isfinite(item).all() for item in (centers, position, quaternion)
            ):
                raise ValueError("kinematic query returned malformed or non-finite data")
            self._kinematic_cache[key] = KinematicSample(position, quaternion, centers)
        return self._kinematic_cache[key]

    @staticmethod
    def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
        a = np.asarray(left, dtype=np.float64) / np.linalg.norm(left)
        b = np.asarray(right, dtype=np.float64) / np.linalg.norm(right)
        dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
        return 2.0 * math.acos(dot)

    def _dense_edge(self, start: np.ndarray, end: np.ndarray) -> list[np.ndarray] | None:
        a = np.asarray(start, dtype=np.float64)
        b = np.asarray(end, dtype=np.float64)
        fa, fb = self._kinematics(a), self._kinematics(b)
        cfg = self.config

        def subdivide(
            qa: np.ndarray, qb: np.ndarray, ka: KinematicSample, kb: KinematicSample, depth: int
        ) -> list[np.ndarray] | None:
            if depth >= 18:
                return None
            mid = (qa + qb) * 0.5
            km = self._kinematics(mid)
            geom_step = float(
                np.max(np.linalg.norm(kb.robot_geom_centers_world_m - ka.robot_geom_centers_world_m, axis=1))
            )
            geom_curve = float(
                np.max(
                    np.linalg.norm(
                        km.robot_geom_centers_world_m
                        - (ka.robot_geom_centers_world_m + kb.robot_geom_centers_world_m) * 0.5,
                        axis=1,
                    )
                )
            )
            eef_step = float(np.linalg.norm(kb.eef_position_world_m - ka.eef_position_world_m))
            eef_rotation = self._rotation_error(ka.eef_quaternion_wxyz, kb.eef_quaternion_wxyz)
            if (
                float(np.max(np.abs(qb - qa))) <= cfg.edge_max_joint_delta_rad
                and geom_step <= cfg.edge_max_geom_center_step_m
                and geom_curve <= cfg.edge_max_geom_midpoint_curvature_m
                and eef_step <= cfg.edge_max_eef_translation_m
                and eef_rotation <= cfg.edge_max_eef_rotation_rad
            ):
                return [qa.copy(), qb.copy()]
            left = subdivide(qa, mid, ka, km, depth + 1)
            if left is None:
                return None
            right = subdivide(mid, qb, km, kb, depth + 1)
            return None if right is None else left[:-1] + right

        return subdivide(a, b, fa, fb, 0)

    def _edge_valid(self, start: np.ndarray, end: np.ndarray) -> bool:
        samples = self._dense_edge(start, end)
        if samples is None:
            return False
        for qpos in samples:
            result = self._check(qpos)
            if not self._configuration_admissible(result):
                return False
        return True

    def _extend(self, tree: _Tree, target: np.ndarray) -> tuple[str, int]:
        distances = np.asarray([np.linalg.norm(q - target) for q in tree.nodes])
        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])
        if distance < 1e-9:
            return "reached", nearest
        ratio = min(1.0, self.config.step_rad / distance)
        new_q = tree.nodes[nearest] + (target - tree.nodes[nearest]) * ratio
        if not self._edge_valid(tree.nodes[nearest], new_q):
            return "trapped", nearest
        index = tree.add(new_q, nearest)
        return ("reached" if ratio >= 1.0 - 1e-10 else "advanced"), index

    def _connect(self, tree: _Tree, target: np.ndarray) -> tuple[str, int]:
        index = 0
        for _ in range(self.config.connect_max_steps):
            status, index = self._extend(tree, target)
            if status in ("trapped", "reached"):
                return status, index
        return "advanced", index

    @staticmethod
    def _length(path: list[np.ndarray]) -> float:
        return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))

    def _path_metrics(self, path: list[np.ndarray]) -> dict[str, Any]:
        min_clearance = math.inf
        by_object: dict[str, float] = {}
        samples = 0
        unsafe_samples = 0
        forbidden_contact_samples = 0
        self_contact_samples = 0
        for start, end in zip(path[:-1], path[1:]):
            dense = self._dense_edge(start, end)
            if dense is None:
                raise RuntimeError("final_path_adaptive_collision_sampling_incomplete")
            for qpos in dense:
                check = self._check(qpos)
                if not self._configuration_admissible(check):
                    raise RuntimeError("final_path_failed_full_geometry_recheck")
                samples += 1
                unsafe_samples += int(not check.diagnostics.get("configuration_safe", False))
                forbidden_contact_samples += int(bool(check.forbidden_contacts))
                self_contact_samples += int(bool(check.self_contacts))
                min_clearance = min(min_clearance, float(check.minimum_forbidden_clearance_m))
                for name, value in check.clearances_by_object_m.items():
                    by_object[name] = min(by_object.get(name, math.inf), float(value))
        return {
            "minimum_clearance_m": min_clearance,
            "clearances_by_object_m": by_object,
            "dense_samples": samples,
            "unsafe_configuration_samples": unsafe_samples,
            "forbidden_contact_samples": forbidden_contact_samples,
            "self_contact_samples": self_contact_samples,
            "joint_length_rad": self._length(path),
        }

    def plan(self, start_qpos: np.ndarray, goal_qpos: np.ndarray) -> JointSpacePlanResult:
        """Search multiple safe routes, prefer larger clearance, then shorter length."""
        start = self._vector(start_qpos, "start_qpos")
        goal = self._vector(goal_qpos, "goal_qpos")
        started = self._started = time.monotonic()
        candidates: list[dict[str, Any]] = []
        best_path: list[np.ndarray] = []
        stop_reason = "iteration_budget_exhausted"

        try:
            for endpoint_name, endpoint in (("start", start), ("goal", goal)):
                tolerance = (
                    float(self.config.start_joint_limit_tolerance_rad)
                    if endpoint_name == "start" else 0.0
                )
                if (
                    np.any(endpoint < self.lower - tolerance)
                    or np.any(endpoint > self.upper + tolerance)
                ):
                    raise ValueError(f"{endpoint_name}_outside_joint_limits")
                check = self._check(endpoint)
                if not self._configuration_admissible(check):
                    raise ValueError(
                        f"{endpoint_name}_configuration_is_unsafe:"
                        f"minimum_forbidden_clearance_m={check.minimum_forbidden_clearance_m};"
                        f"clearances_by_object_m={check.clearances_by_object_m};"
                        f"forbidden_contacts={check.forbidden_contacts};"
                        f"self_contacts={check.self_contacts}"
                    )
            rng = np.random.default_rng(self.config.seed)
            start_tree, goal_tree = _Tree(start, "start"), _Tree(goal, "goal")
            tree_a, tree_b = start_tree, goal_tree
            for iteration in range(self.config.max_iterations):
                elapsed = time.monotonic() - started
                if elapsed >= self.config.search_time_budget_s:
                    stop_reason = "planning_time_budget_exhausted"
                    break
                if self._check_count >= self.config.max_config_checks:
                    stop_reason = "configuration_check_budget_exhausted"
                    break
                if len(candidates) >= self.config.max_candidate_routes:
                    stop_reason = "candidate_route_budget_reached"
                    break
                if rng.random() < self.config.goal_bias:
                    sample = tree_b.nodes[0].copy()
                else:
                    sample = rng.uniform(self.lower, self.upper)
                status_a, index_a = self._extend(tree_a, sample)
                if status_a != "trapped":
                    status_b, index_b = self._connect(tree_b, tree_a.nodes[index_a])
                    if status_b == "reached":
                        if tree_a.name == "start":
                            route = tree_a.path(index_a) + tree_b.path(index_b)[::-1][1:]
                        else:
                            route = tree_b.path(index_b) + tree_a.path(index_a)[::-1][1:]
                        compact = [route[0]]
                        compact.extend(q for q in route[1:] if np.linalg.norm(q - compact[-1]) > 1e-9)
                        # Shortcut only edges that pass the same complete collision gate.
                        for _ in range(self.config.shortcut_attempts):
                            if len(compact) <= 2 or time.monotonic() - started >= self.config.search_time_budget_s:
                                break
                            i, j = sorted(rng.choice(len(compact), size=2, replace=False).tolist())
                            if j - i > 1 and self._edge_valid(compact[i], compact[j]):
                                if np.linalg.norm(compact[j] - compact[i]) + 1e-9 < self._length(compact[i : j + 1]):
                                    compact = compact[: i + 1] + compact[j:]
                        metrics = self._path_metrics(compact)
                        route_record = {
                            "route_index": len(candidates),
                            "iterations": iteration + 1,
                            "metrics": metrics,
                        }
                        candidates.append({"path": compact, **route_record})
                        candidates.sort(
                            key=lambda item: (
                                -item["metrics"]["minimum_clearance_m"],
                                item["metrics"]["joint_length_rad"],
                            )
                        )
                        best_path = candidates[0]["path"]
                        # Restart with fresh trees to search for a higher-margin path.
                        start_tree, goal_tree = _Tree(start, "start"), _Tree(goal, "goal")
                        tree_a, tree_b = start_tree, goal_tree
                        if len(candidates) >= self.config.max_candidate_routes:
                            stop_reason = "candidate_route_budget_reached"
                            break
                tree_a, tree_b = tree_b, tree_a
        except TimeoutError as exc:
            stop_reason = str(exc)
        except Exception as exc:
            return JointSpacePlanResult(
                status="planning_failed",
                qpos_path=np.empty((0, 7), dtype=np.float64),
                minimum_forbidden_clearance_m=None,
                planner_source="jointspace_rrt_connect.v1",
                planning_seconds=time.monotonic() - started,
                config_checks=self._check_count,
                iterations=iteration + 1 if "iteration" in locals() else 0,
                selected_route_index=None,
                candidates=candidates,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
        if not best_path:
            return JointSpacePlanResult(
                status="no_safe_path",
                qpos_path=np.empty((0, 7), dtype=np.float64),
                minimum_forbidden_clearance_m=None,
                planner_source="jointspace_rrt_connect.v1",
                planning_seconds=time.monotonic() - started,
                config_checks=self._check_count,
                iterations=iteration + 1 if "iteration" in locals() else 0,
                selected_route_index=None,
                candidates=candidates,
                failure_reason=stop_reason,
            )
        selected_metrics = candidates[0]["metrics"]
        dense_path = [best_path[0]]
        for a, b in zip(best_path[:-1], best_path[1:]):
            dense = self._dense_edge(a, b)
            if dense is None:
                return JointSpacePlanResult(
                    status="incomplete_final_validation",
                    qpos_path=np.empty((0, 7), dtype=np.float64),
                    minimum_forbidden_clearance_m=None,
                    planner_source="jointspace_rrt_connect.v1",
                    planning_seconds=time.monotonic() - started,
                    config_checks=self._check_count,
                    iterations=iteration + 1,
                    selected_route_index=0,
                    candidates=candidates,
                    failure_reason="final_path_adaptive_collision_sampling_incomplete",
                )
            dense_path.extend(dense[1:])
        # The final full-Geom pass also verifies the actual dense return path.
        final_metrics = self._path_metrics(dense_path)
        return JointSpacePlanResult(
            status="offline_feasible",
            qpos_path=np.stack(dense_path),
            minimum_forbidden_clearance_m=final_metrics["minimum_clearance_m"],
            planner_source="jointspace_rrt_connect.v1",
            planning_seconds=time.monotonic() - started,
            config_checks=self._check_count,
            iterations=iteration + 1 if "iteration" in locals() else 0,
            selected_route_index=0,
            candidates=[
                dict(item)
                for item in candidates
            ],
        )
