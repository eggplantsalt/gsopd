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
"""Standalone object-center B-spline proposals and conservative sphere screening.

Inputs describe externally verified, same-state world geometry. This module
checks those declarations, not their physical provenance. Its output is a set
of geometric proposals, never OSC commands or verified executable trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BundleConfig:
    """Bound the proposal count and the error of geometry-only curve screening."""

    candidate_count: int = 18
    clearance_m: float = 0.01
    minimum_curve_samples: int = 65
    maximum_curve_samples: int = 4097
    clearance_error_m: float = 0.001


def _vector(value: Any, field: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.shape != (3,)
        or raw.dtype.kind not in "iuf"
        or any(
            isinstance(item, (bool, np.bool_))
            for item in np.asarray(value, dtype=object).flat
        )
    ):
        raise ValueError(f"{field} must contain three real world coordinates")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{field} must be finite")
    return result


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(
            f"{field} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def sphere_from_aabb(lower_world_m: Any, upper_world_m: Any) -> dict[str, Any]:
    """Enclose an entire world AABB; the caller owns geometry/binding verification."""
    lower, upper = (
        _vector(lower_world_m, "AABB lower"),
        _vector(upper_world_m, "AABB upper"),
    )
    if np.any(lower > upper):
        raise ValueError("AABB lower bounds exceed upper bounds")
    center = lower + (upper - lower) / 2
    # Account for midpoint rounding as well as the diagonal's floating error.
    extent = np.maximum(np.abs(lower - center), np.abs(upper - center))
    radius = float(np.linalg.norm(extent)) * (1 + 8 * np.finfo(np.float64).eps)
    _number(radius, "AABB sphere radius", positive=True)
    return {
        "center_world_m": center.tolist(),
        "radius_m": float(np.nextafter(radius, np.inf)),
    }


def _scene_input(scene: Any, config: BundleConfig) -> dict[str, Any]:
    if (
        not isinstance(scene, dict)
        or scene.get("frame") != "world"
        or scene.get("units") != "m"
    ):
        raise ValueError("scene must explicitly use frame=world and units=m")
    state_id = _text(scene.get("state_id"), "state_id")

    def sphere(value, field):
        if not isinstance(value, dict):
            raise ValueError(f"{field} sphere is missing")
        if (
            value.get("binding_verified") is not True
            or value.get("enclosure_verified") is not True
        ):
            raise ValueError(
                f"{field} requires verified binding and full-geometry enclosure declarations"
            )
        if value.get("state_id") != state_id:
            raise ValueError(
                f"{field} geometry state_id differs from the current scene"
            )
        return {
            "object_id": _text(value.get("object_id"), f"{field}.object_id"),
            "center_world_m": _vector(value.get("center_world_m"), field).tolist(),
            "radius_m": _number(
                value.get("radius_m"), f"{field}.radius_m", positive=True
            ),
            "binding_source": _text(
                value.get("binding_source"), f"{field}.binding_source"
            ),
            "enclosure_source": _text(
                value.get("enclosure_source"), f"{field}.enclosure_source"
            ),
            "state_id": state_id,
            "binding_verified": True,
            "enclosure_verified": True,
        }

    carried = sphere(scene.get("carried"), "carried")
    values = scene.get("obstacles")
    if not isinstance(values, list) or not values:
        raise ValueError("obstacles must contain at least one verified sphere")
    obstacles = [
        sphere(item, f"obstacles[{index}]") for index, item in enumerate(values)
    ]
    ids = [item["object_id"] for item in [carried, *obstacles]]
    if len(set(ids)) != len(ids):
        raise ValueError("carried and obstacle object IDs must be unique")
    goal = scene.get("goal")
    if (
        not isinstance(goal, dict)
        or goal.get("verified") is not True
        or goal.get("state_id") != state_id
    ):
        raise ValueError("goal requires a verified position from the current state")
    if goal.get("position_semantics") != "carried_sphere_center":
        raise ValueError(
            "goal.position_semantics must be carried_sphere_center; a target Body origin is not inferred"
        )
    goal_type = goal.get("goal_type", "placement_goal")
    if goal_type not in ("placement_goal", "local_bypass_goal"):
        raise ValueError("goal.goal_type must be placement_goal or local_bypass_goal")
    position = _vector(goal.get("position_world_m"), "goal.position_world_m")
    normalized_goal = {
        "goal_type": goal_type,
        "position_world_m": position.tolist(),
        "position_semantics": "carried_sphere_center",
        "source": _text(goal.get("source"), "goal.source"),
        "state_id": state_id,
        "verified": True,
    }
    if goal_type == "local_bypass_goal":
        verification_scope = goal.get("verification_scope")
        if (
            verification_scope
            != "same_state_geometry_derived_waypoint_only_not_final_placement_or_physical_safety"
        ):
            raise ValueError("local_bypass_goal verification scope is not explicit")
        construction = goal.get("construction")
        if not isinstance(construction, dict):
            raise ValueError("local_bypass_goal requires construction evidence")
        obstacle_id = _text(
            construction.get("obstacle_object_id"),
            "goal.construction.obstacle_object_id",
        )
        obstacle = next(
            (item for item in obstacles if item["object_id"] == obstacle_id), None
        )
        if obstacle is None:
            raise ValueError("local_bypass_goal obstacle is not in the verified scene")
        obstacle_center = _vector(
            construction.get("obstacle_center_world_m"),
            "goal.construction.obstacle_center_world_m",
        )
        if not np.array_equal(
            obstacle_center, np.asarray(obstacle["center_world_m"], dtype=float)
        ):
            raise ValueError("local_bypass_goal obstacle center conflicts with scene")
        forward = _vector(
            construction.get("forward_world"), "goal.construction.forward_world"
        )
        forward_norm = float(np.linalg.norm(forward))
        if not math.isfinite(forward_norm) or abs(forward_norm - 1.0) > 1e-9:
            raise ValueError("local_bypass_goal forward_world must be a unit vector")
        target = construction.get("placement_target_direction_reference")
        if not isinstance(target, dict):
            raise ValueError(
                "local_bypass_goal requires a BDDL placement direction reference"
            )
        target_position = _vector(
            target.get("body_origin_world_m"),
            "goal.construction.placement_target_direction_reference.body_origin_world_m",
        )
        if target.get("state_id") != state_id:
            raise ValueError(
                "local_bypass_goal BDDL target reference does not match the current scene state"
            )
        target_body_id = target.get("body_id")
        if (
            isinstance(target_body_id, bool)
            or not isinstance(target_body_id, int)
            or target_body_id < 0
        ):
            raise ValueError("local_bypass_goal placement target Body ID is invalid")
        target_body_name = _text(
            target.get("body_name"),
            "goal.construction.placement_target_direction_reference.body_name",
        )
        if (
            construction.get("direction_source")
            != "same_state_carried_sphere_center_to_obstacle_sphere_center"
        ):
            raise ValueError(
                "local_bypass_goal direction must be derived from the carried and obstacle sphere centers"
            )
        if (
            construction.get("task_reference_source")
            != "active_bddl_on_target_body_origin_semantic_reference_only"
        ):
            raise ValueError(
                "local_bypass_goal must retain the active BDDL target as semantic context"
            )
        carried_task_object = _text(
            construction.get("carried_registered_task_object"),
            "goal.construction.carried_registered_task_object",
        )
        target_task_object = _text(
            target.get("registered_task_object"),
            "goal.construction.placement_target_direction_reference.registered_task_object",
        )
        native_target_key = _text(
            target.get("native_target_key"),
            "goal.construction.placement_target_direction_reference.native_target_key",
        )
        predicate = target.get("native_predicate")
        if (
            not isinstance(predicate, list)
            or len(predicate) != 3
            or str(predicate[0]).lower() != "on"
            or predicate[1] != carried_task_object
            or predicate[2] != native_target_key
            or target.get("used_as_final_goal") is not False
        ):
            raise ValueError(
                "local_bypass_goal target reference must match the active BDDL On relation"
            )
        start = np.asarray(carried["center_world_m"], dtype=float)
        carried_to_obstacle = obstacle_center - start
        obstacle_distance = float(np.linalg.norm(carried_to_obstacle))
        if not math.isfinite(obstacle_distance) or obstacle_distance <= 1e-9:
            raise ValueError(
                "local_bypass_goal carried and obstacle centers are coincident"
            )
        expected_forward = carried_to_obstacle / obstacle_distance
        if not np.allclose(expected_forward, forward, atol=1e-9, rtol=0):
            raise ValueError(
                "local_bypass_goal forward direction conflicts with same-state sphere centers"
            )
        safety_clearance = _number(
            construction.get("safety_clearance_m"),
            "goal.construction.safety_clearance_m",
        )
        if not math.isclose(
            safety_clearance, config.clearance_m, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "local_bypass_goal safety clearance must match the configured curve clearance"
            )
        exit_clearance = _number(
            construction.get("exit_clearance_m"),
            "goal.construction.exit_clearance_m",
        )
        required_separation = (
            carried["radius_m"]
            + obstacle["radius_m"]
            + config.clearance_m
            + exit_clearance
        )
        goal_offset = position - obstacle_center
        projected_offset = float(np.dot(goal_offset, forward))
        lateral_offset = goal_offset - projected_offset * forward
        numeric_margin = 1e-9 * max(1.0, required_separation)
        if (
            np.linalg.norm(lateral_offset) > numeric_margin
            or projected_offset < required_separation
        ):
            raise ValueError(
                "local_bypass_goal must lie on the far side of its obstacle with the configured conservative sphere clearance and exit margin"
            )
        normalized_goal["construction"] = {
            "obstacle_object_id": obstacle_id,
            "obstacle_center_world_m": obstacle_center.tolist(),
            "forward_world": forward.tolist(),
            "placement_target_direction_reference": {
                "body_id": target_body_id,
                "body_name": target_body_name,
                "body_origin_world_m": target_position.tolist(),
                "used_as_final_goal": False,
                "state_id": state_id,
            },
            "direction_source": construction["direction_source"],
            "task_reference_source": construction["task_reference_source"],
            "native_bddl_on_predicate": predicate,
            "carried_registered_task_object": carried_task_object,
            "placement_target_registered_task_object": target_task_object,
            "native_target_key": native_target_key,
            "safety_clearance_m": safety_clearance,
            "exit_clearance_m": exit_clearance,
            "required_sphere_center_separation_m": required_separation,
            "actual_sphere_center_separation_m": float(np.linalg.norm(goal_offset)),
            "actual_conservative_sphere_gap_m": float(
                np.linalg.norm(goal_offset) - carried["radius_m"] - obstacle["radius_m"]
            ),
        }
        normalized_goal["verification_scope"] = verification_scope
    normalized_goal["final_placement_goal_verified"] = (
        goal_type == "placement_goal"
        and goal.get("safe_center_region_verified") is True
    )
    goal = normalized_goal
    up = _vector(scene.get("world_up"), "world_up")
    norm = float(np.linalg.norm(up))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("world_up has no finite direction")
    return {
        "frame": "world",
        "units": "m",
        "state_id": state_id,
        "carried": carried,
        "obstacles": obstacles,
        "goal": goal,
        "world_up": (up / norm).tolist(),
    }


def _configuration(config: BundleConfig) -> None:
    for field, minimum, maximum in (
        ("candidate_count", 12, 24),
        ("minimum_curve_samples", 4, 4097),
        ("maximum_curve_samples", 4, 4097),
    ):
        value = getattr(config, field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    if config.maximum_curve_samples < config.minimum_curve_samples:
        raise ValueError("maximum_curve_samples is smaller than minimum_curve_samples")
    _number(config.clearance_m, "clearance_m")
    _number(config.clearance_error_m, "clearance_error_m", positive=True)


def _spline_data(control_points: Any) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(control_points)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 4
        or points.dtype.kind not in "iuf"
    ):
        raise ValueError(
            "cubic B-spline requires at least four three-dimensional control points"
        )
    points = points.astype(np.float64)
    if not np.isfinite(points).all():
        raise ValueError("B-spline control points must be finite")
    count = len(points)
    knots = np.concatenate([
        np.zeros(4),
        np.arange(1, count - 3) / (count - 3),
        np.ones(4),
    ])
    denominators = knots[4 : count + 3] - knots[1:count]
    derivative_controls = 3 * np.diff(points, axis=0) / denominators[:, None]
    bound = float(np.linalg.norm(derivative_controls, axis=1).max())
    if not math.isfinite(bound):
        raise ValueError("B-spline derivative bound is not finite")
    return points, knots, bound


def _evaluate(points, knots, parameters):
    # Vectorized de Boor evaluation of a clamped cubic B-spline.
    spans = np.clip(
        np.searchsorted(knots, parameters, side="right") - 1, 3, len(points) - 1
    )
    working = points[spans[:, None] - 3 + np.arange(4)].copy()
    for level in range(1, 4):
        for index in range(3, level - 1, -1):
            left = knots[spans - 3 + index]
            right = knots[spans + 1 + index - level]
            alpha = (parameters - left) / (right - left)
            working[:, index] = (1 - alpha[:, None]) * working[:, index - 1] + alpha[
                :, None
            ] * working[:, index]
    return working[:, 3]


def discretize_bspline(
    control_points_world_m: Any, sample_count: int = 129
) -> np.ndarray:
    """Evaluate a clamped cubic curve, including its exact start and goal centers."""
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or not 2 <= sample_count <= 100001
    ):
        raise ValueError("sample_count must be an integer in [2, 100001]")
    points, knots, _ = _spline_data(control_points_world_m)
    return _evaluate(points, knots, np.linspace(0, 1, sample_count))


def _screen_curve(control_points, carried_radius, obstacles, config):
    points, knots, speed = _spline_data(control_points)
    # A cubic derivative is a convex combination of its derivative controls.
    # Every parameter lies within dt/2 of a sample, so the entire curve has
    # distance >= minimum_sample_distance - max_speed*dt/2 to each sphere.
    requested_intervals = speed / (2 * config.clearance_error_m)
    if not math.isfinite(requested_intervals):
        raise ValueError("requested curve sampling resolution is not representable")
    required = max(config.minimum_curve_samples, math.ceil(requested_intervals) + 1)
    count = min(required, config.maximum_curve_samples)
    parameters = np.linspace(0, 1, count)
    curve = _evaluate(points, knots, parameters)
    slack = speed / (2 * (count - 1))
    numeric_margin = 1e-10 * max(1.0, float(np.abs(points).max()), speed)
    checks, rejections = [], []
    for obstacle in obstacles:
        center = np.asarray(obstacle["center_world_m"])
        radius = carried_radius + obstacle["radius_m"]
        distances = np.linalg.norm(curve - center, axis=1) - radius
        sampled_min = float(distances.min())
        lower = sampled_min - slack - numeric_margin
        if not np.isfinite(distances).all() or not math.isfinite(lower):
            raise ValueError("sphere clearance calculation is non-finite")
        object_id = obstacle["object_id"]
        if distances[0] < config.clearance_m:
            rejections.append(f"start_sphere_clearance_insufficient:{object_id}")
        if distances[-1] < config.clearance_m:
            rejections.append(f"goal_sphere_clearance_insufficient:{object_id}")
        if sampled_min < config.clearance_m:
            rejections.append(f"sphere_clearance_below_margin:{object_id}")
        elif lower < config.clearance_m:
            rejections.append(f"curve_clearance_bound_inconclusive:{object_id}")
        checks.append({
            "obstacle_id": object_id,
            "combined_sphere_radius_m": radius,
            "minimum_sampled_gap_m": sampled_min,
            "minimum_sample_index": int(np.argmin(distances)),
            "continuous_gap_lower_bound_m": lower,
            "passes_sphere_screen": lower >= config.clearance_m,
        })
    return (
        curve,
        {
            "degree": 3,
            "knots": knots.tolist(),
            "sample_count": count,
            "parameter_range": [0.0, 1.0],
            "maximum_derivative_norm_bound_m": speed,
            "between_samples_error_bound_m": slack,
            "numerical_margin_m": numeric_margin,
            "requested_sample_count": required,
            "resolution_target_met": count >= required,
            "checks": checks,
            "minimum_sampled_gap_m": min(
                item["minimum_sampled_gap_m"] for item in checks
            ),
            "continuous_gap_lower_bound_m": min(
                item["continuous_gap_lower_bound_m"] for item in checks
            ),
        },
        rejections,
    )


def generate_trajectory_bundle(
    scene: dict[str, Any],
    config: BundleConfig = BundleConfig(),
) -> dict[str, Any]:
    """Generate 12–24 object-center paths with explicit geometry-only screening.

    Args:
        scene: Same-state world/meter carried sphere, obstacle spheres, a verified
            placement or explicitly constructed local bypass carried-center goal,
            and world-up vector. Each sphere needs
            binding/enclosure verification declarations and their source strings.
        config: Deterministic bundle size, clearance and discretization settings.

    Returns:
        JSON-compatible candidates including rejected curves and their evidence.
        Spheres describe static obstacles and a translation-only object envelope;
        passing is conditional on those supplied bounds, not physical validation.

    Raises:
        ValueError: Geometry, provenance, coordinate frame or settings are invalid.
    """
    _configuration(config)
    scene = _scene_input(scene, config)
    carried, obstacles = scene["carried"], scene["obstacles"]
    start = np.asarray(carried["center_world_m"])
    goal = np.asarray(scene["goal"]["position_world_m"])
    up = np.asarray(scene["world_up"])
    length = float(np.linalg.norm(goal - start))
    if not math.isfinite(length) or length < 1e-9:
        raise ValueError("start and goal must define a finite nonzero path")
    forward = (goal - start) / length
    lateral = np.cross(up, forward)
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm < 1e-8:
        raise ValueError(
            "start-goal axis is parallel to world_up; left/right directions are undefined"
        )
    lateral /= lateral_norm
    above = np.cross(forward, lateral)
    above /= np.linalg.norm(above)
    straight = []
    for obstacle in obstacles:
        center = np.asarray(obstacle["center_world_m"])
        along = float(np.dot(center - start, forward))
        closest = start + np.clip(along, 0.0, length) * forward
        gap = float(
            np.linalg.norm(center - closest)
            - carried["radius_m"]
            - obstacle["radius_m"]
        )
        straight.append((gap, obstacle["object_id"], obstacle))
    # Geometric ties retain input order; semantic IDs never select a route.
    _, _, primary = min(straight, key=lambda value: value[0])
    center = np.asarray(primary["center_world_m"])
    anchor_distance = float(np.clip(np.dot(center - start, forward), 0, length))
    anchor = start + anchor_distance * forward
    inflated = carried["radius_m"] + primary["radius_m"] + config.clearance_m
    levels = np.linspace(1.2, 2.6, math.ceil(config.candidate_count / 3))
    candidates = []
    for index in range(config.candidate_count):
        direction_name, direction = (
            ("left", lateral),
            ("right", -lateral),
            ("above", above),
        )[index % 3]
        level = index // 3
        offset = max(
            inflated * levels[level],
            float(np.dot(center - anchor, direction)) + inflated * levels[level],
        )
        longitudinal = inflated * (1.0 if level % 2 == 0 else 1.6)
        entry = (
            start
            + max(0.0, anchor_distance - longitudinal) * forward
            + offset * direction
        )
        peak = anchor + offset * direction
        exit_point = (
            start
            + min(length, anchor_distance + longitudinal) * forward
            + offset * direction
        )
        controls = np.asarray([
            start,
            start + 0.35 * (entry - start),
            entry,
            peak,
            exit_point,
            goal + 0.35 * (exit_point - goal),
            goal,
        ])
        curve, screening, reasons = _screen_curve(
            controls, carried["radius_m"], obstacles, config
        )
        endpoint_error = float(np.linalg.norm(curve[-1] - goal))
        goal_axis = goal - center
        goal_axis_norm = float(np.linalg.norm(goal_axis))
        target_side = (
            float(np.dot(curve[-1] - center, goal_axis / goal_axis_norm))
            if goal_axis_norm > 1e-12
            else 0.0
        )
        if endpoint_error > 1e-10 or target_side <= 0:
            reasons.append("goal_side_region_not_reached")
        candidates.append({
            "candidate_id": f"trajectory-{index + 1:03d}",
            "direction": direction_name,
            "direction_world": direction.tolist(),
            "offset_scale": float(levels[level]),
            "lateral_or_above_offset_m": float(offset),
            "waypoints_world_m": np.asarray([
                start,
                entry,
                peak,
                exit_point,
                goal,
            ]).tolist(),
            "control_points_world_m": controls.tolist(),
            "points_world_m": curve.tolist(),
            "length_m": float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()),
            "length_method": "discretized_B_spline_chord_sum_not_timing",
            "goal_endpoint_error_m": endpoint_error,
            "goal_side_projection_m": target_side,
            "screening": screening,
            "passes_geometric_screen": not reasons,
            "rejection_reasons": reasons,
            "physical_safety_verified": False,
            "executable": False,
        })
    passed = sum(item["passes_geometric_screen"] for item in candidates)
    return {
        "schema_version": "gs_opd.trajectory_bundle.v1",
        "status": "complete",
        "scene": scene,
        "config": asdict(config),
        "geometry_verification": "external_input_declarations_checked_not_independently_verified",
        "path_reference": "carried_bounding_sphere_center_world_m",
        "primary_obstacle_id": primary["object_id"],
        "frame_axes": {
            "forward": forward.tolist(),
            "left": lateral.tolist(),
            "above": above.tolist(),
        },
        "goal_type": scene["goal"]["goal_type"],
        "straight_line_minimum_gap_m": min(item[0] for item in straight),
        "candidate_count": len(candidates),
        "screen_pass_count": passed,
        "screen_rejected_count": len(candidates) - passed,
        "rejection_reason_counts": dict(
            Counter(
                reason for item in candidates for reason in item["rejection_reasons"]
            )
        ),
        "candidates": candidates,
        "physical_safety_verified": False,
        "executable": False,
        "limitations": [
            "Caller must supply conservative enclosures of the entire carried object and every relevant obstacle in the same state.",
            "A local_bypass_goal is a geometry-derived waypoint behind an obstacle using the BDDL On target Body origin for direction only; it is not a final placement location or proof of task completion.",
            "B-spline control waypoints are not interpolation constraints; the sampled curve and its bound are screened.",
            "Static sphere geometry omits the robot/gripper, orientation, support contacts, dynamics and task feasibility.",
            "No control, time parameterization, Snapshot, contact verification or action execution is performed.",
        ],
    }


def _bundle_svg(bundle):
    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": "1140",
            "height": "435",
            "viewBox": "0 0 1140 435",
        },
    )

    def label(parent, x, y, text, **attributes):
        ET.SubElement(
            parent, "text", {"x": str(x), "y": str(y), "font-size": "12", **attributes}
        ).text = text

    label(
        root,
        18,
        20,
        (
            "Local bypass waypoint; not final placement or physical safety"
            if bundle.get("goal_type") == "local_bypass_goal"
            else "Geometry-only B-spline proposals; sphere screen is not physical safety"
        ),
    )
    scene = bundle["scene"]
    colors = {"left": "#1677b8", "right": "#8558ad", "above": "#c87415"}
    all_points = np.concatenate([
        np.asarray(item["control_points_world_m"]) for item in bundle["candidates"]
    ])
    start = np.asarray(scene["carried"]["center_world_m"])
    goal = np.asarray(scene["goal"]["position_world_m"])
    for panel, (a, b, name) in enumerate(((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))):
        group = ET.SubElement(root, "g", {"data-view": name})
        extents = [all_points[:, [a, b]], np.asarray([start[[a, b]], goal[[a, b]]])]
        for sphere in scene["obstacles"]:
            center = np.asarray(sphere["center_world_m"])[[a, b]]
            radius = (
                sphere["radius_m"]
                + scene["carried"]["radius_m"]
                + bundle["config"]["clearance_m"]
            )
            extents.append(np.asarray([center - radius, center + radius]))
        data = np.concatenate(extents)
        low, high = data.min(axis=0), data.max(axis=0)
        scale = min(
            310 / max(float(high[0] - low[0]), 1e-6),
            290 / max(float(high[1] - low[1]), 1e-6),
        )
        middle = (high + low) / 2

        def pixel(point):
            x, y = (np.asarray(point)[[a, b]] - middle) * scale
            return float(panel * 380 + 190 + x), float(195 - y)

        label(group, panel * 380 + 22, 43, f"{name}: world axes (m), equal scale")
        for sphere in scene["obstacles"]:
            x, y = pixel(sphere["center_world_m"])
            radius = (
                sphere["radius_m"]
                + scene["carried"]["radius_m"]
                + bundle["config"]["clearance_m"]
            )
            ET.SubElement(
                group,
                "circle",
                {
                    "cx": str(x),
                    "cy": str(y),
                    "r": str(radius * scale),
                    "fill": "#e9b9b9",
                    "fill-opacity": "0.3",
                    "stroke": "#b35b5b",
                },
            )
            label(group, x + 3, y - 3, sphere["object_id"])
        for candidate in bundle["candidates"]:
            values = candidate["points_world_m"]
            indices = list(range(0, len(values), max(1, len(values) // 240)))
            if indices[-1] != len(values) - 1:
                indices.append(len(values) - 1)
            passed = candidate["passes_geometric_screen"]
            line = ET.SubElement(
                group,
                "polyline",
                {
                    "points": " ".join(
                        f"{x:.3f},{y:.3f}"
                        for x, y in (pixel(values[index]) for index in indices)
                    ),
                    "fill": "none",
                    "stroke": colors[candidate["direction"]] if passed else "#888888",
                    "stroke-width": "1.5",
                    "opacity": "0.85" if passed else "0.3",
                    "stroke-dasharray": "none" if passed else "3 3",
                    "data-candidate-id": candidate["candidate_id"],
                },
            )
            ET.SubElement(
                line, "title"
            ).text = f"{candidate['candidate_id']} {candidate['direction']}: {candidate['rejection_reasons'] or 'passes geometric screen only'}"
        goal_label = (
            "local bypass"
            if bundle.get("goal_type") == "local_bypass_goal"
            else "goal"
        )
        for point, text, color in (
            (start, "start", "#111111"),
            (goal, goal_label, "#188237"),
        ):
            x, y = pixel(point)
            ET.SubElement(
                group, "circle", {"cx": str(x), "cy": str(y), "r": "4", "fill": color}
            )
            label(group, x + 6, y + 12, text)
    label(
        root,
        18,
        386,
        f"Candidates: {bundle['candidate_count']}; screen pass: {bundle['screen_pass_count']}; rejected: {bundle['screen_rejected_count']}",
    )
    label(
        root,
        18,
        405,
        "Blue=left, purple=right, orange=above; grey dashed=rejected; circles=obstacle + carried radius + clearance",
    )
    label(
        root,
        18,
        424,
        "SVG curves are subsampled for display; JSON retains full samples, control points, clearance bounds and rejection reasons.",
    )
    return ET.tostring(root, encoding="unicode")


def save_trajectory_bundle(
    bundle: dict[str, Any], out_dir: str | Path
) -> dict[str, str]:
    """Save full JSON diagnostics and XY/XZ/YZ SVG views, without raster dependencies."""
    rendered = json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False)
    svg = _bundle_svg(bundle)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, svg_path = (
        directory / "trajectory_bundle.json",
        directory / "trajectory_bundle.svg",
    )
    json_path.write_text(rendered + "\n", encoding="utf-8")
    svg_path.write_text(svg + "\n", encoding="utf-8")
    return {"json": str(json_path), "svg": str(svg_path)}


def main(argv: list[str] | None = None) -> int:
    """Run offline generation from an explicitly verified geometry JSON document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=18)
    parser.add_argument("--clearance-m", type=float, default=0.01)
    args = parser.parse_args(argv)
    try:

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def invalid_constant(value):
            raise ValueError(f"non-finite JSON constant: {value}")

        scene = json.loads(
            args.scene_json.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
        bundle = generate_trajectory_bundle(
            scene,
            BundleConfig(
                candidate_count=args.candidate_count, clearance_m=args.clearance_m
            ),
        )
        outputs = save_trajectory_bundle(bundle, args.out_dir)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        sys.stderr.write(f"trajectory bundle unavailable: {exc}\n")
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "candidate_count": bundle["candidate_count"],
                "screen_pass_count": bundle["screen_pass_count"],
                "screen_rejected_count": bundle["screen_rejected_count"],
                "physical_safety_verified": False,
                "outputs": outputs,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
