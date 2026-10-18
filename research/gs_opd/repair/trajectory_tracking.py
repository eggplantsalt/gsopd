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
"""Closed-loop OSC tracking and fail-closed verification for geometry bundles.

The path is a carried-object sphere-center reference, not an action sequence.
The Safety Bridge converts it into feedback OSC actions from the live Panda and
object state, runs isolated Snapshot branches, and returns substep contacts.
This module validates bindings and combines each branch with the existing
Contact Safety Compiler. No model or simulator is created here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from research.gs_opd.smoke.contact_safety_compiler_smoke import (
    _complete_monitoring,
    compile_forbid_contact,
)
from research.gs_opd.teacher.schema_v2 import validate_teacher_output_v2


@dataclass(frozen=True)
class TrackingConfig:
    """Bound path tracking work and require tight physical attachment checks."""

    max_control_steps_per_candidate: int = 240
    reference_speed_m_s: float = 0.04
    maximum_object_tracking_error_m: float = 0.025
    maximum_eef_tracking_error_m: float = 0.025
    maximum_grasp_relative_position_error_m: float = 0.012
    maximum_grasp_relative_rotation_error_rad: float = 0.20
    state_tolerance_m: float = 1e-6
    rotation_tolerance_rad: float = 1e-6

    def validate(self) -> None:
        if (
            isinstance(self.max_control_steps_per_candidate, bool)
            or not isinstance(self.max_control_steps_per_candidate, int)
            or not 1 <= self.max_control_steps_per_candidate <= 240
        ):
            raise ValueError("max_control_steps_per_candidate must be in 1..240")
        for name in (
            "reference_speed_m_s",
            "maximum_object_tracking_error_m",
            "maximum_eef_tracking_error_m",
            "maximum_grasp_relative_position_error_m",
            "maximum_grasp_relative_rotation_error_rad",
            "state_tolerance_m",
            "rotation_tolerance_rad",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def _unit_quaternion(value: Any, name: str) -> np.ndarray:
    quaternion = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError(f"{name} has no rotation")
    return quaternion / norm


def quaternion_conjugate(value: Any) -> np.ndarray:
    quaternion = _unit_quaternion(value, "quaternion")
    return np.asarray([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])


def quaternion_multiply(left: Any, right: Any) -> np.ndarray:
    """Multiply normalized MuJoCo wxyz quaternions."""
    a = _unit_quaternion(left, "left quaternion")
    b = _unit_quaternion(right, "right quaternion")
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    result = np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def quaternion_to_matrix(value: Any) -> np.ndarray:
    w, x, y, z = _unit_quaternion(value, "quaternion")
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_to_rotation_vector(value: Any) -> np.ndarray:
    quaternion = _unit_quaternion(value, "rotation delta quaternion")
    if quaternion[0] < 0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, max(0.0, float(quaternion[0])))
    return quaternion[1:] * (angle / vector_norm)


def resample_polyline_by_distance(
    points_world_m: Any,
    *,
    speed_m_s: float,
    control_timestep_s: float,
    maximum_steps: int,
) -> dict[str, Any]:
    """Resample a geometric path at bounded 20 Hz reference displacement."""
    points = np.asarray(points_world_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("trajectory points must have shape (N>=2, 3)")
    if not np.isfinite(points).all():
        raise ValueError("trajectory points contain non-finite coordinates")
    if (
        not math.isfinite(float(speed_m_s))
        or speed_m_s <= 0
        or not math.isfinite(float(control_timestep_s))
        or control_timestep_s <= 0
    ):
        raise ValueError("reference speed and control timestep must be positive")
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    total_length = float(cumulative[-1])
    if not math.isfinite(total_length) or total_length <= 1e-9:
        raise ValueError("trajectory has no finite nonzero path length")
    requested_steps = int(math.ceil(total_length / (speed_m_s * control_timestep_s)))
    if requested_steps > maximum_steps:
        raise ValueError(
            f"trajectory_requires_{requested_steps}_steps_exceeding_limit_{maximum_steps}"
        )
    query = np.linspace(0.0, total_length, requested_steps + 1)
    references = np.stack(
        [np.interp(query, cumulative, points[:, axis]) for axis in range(3)], axis=1
    )
    references[0] = points[0]
    references[-1] = points[-1]
    return {
        "reference_points_world_m": references,
        "path_length_m": total_length,
        "requested_steps": requested_steps,
        "maximum_reference_displacement_m": float(
            np.linalg.norm(np.diff(references, axis=0), axis=1).max()
        ),
        "reference_speed_limit_m_s": float(speed_m_s),
        "control_timestep_s": float(control_timestep_s),
    }


def inverse_osc_world_delta(
    position_delta_world_m: Any,
    rotation_delta_world_rad: Any,
    controller_fields: dict[str, Any],
    *,
    gripper_action: float = 1.0,
) -> np.ndarray:
    """Map a world-frame OSC delta to 7-D env action without clipping."""
    position = _finite_vector(position_delta_world_m, 3, "position delta")
    rotation = _finite_vector(rotation_delta_world_rad, 3, "rotation delta")
    input_min = _finite_vector(controller_fields.get("input_min"), 6, "OSC input_min")
    input_max = _finite_vector(controller_fields.get("input_max"), 6, "OSC input_max")
    output_min = _finite_vector(
        controller_fields.get("output_min"), 6, "OSC output_min"
    )
    output_max = _finite_vector(
        controller_fields.get("output_max"), 6, "OSC output_max"
    )
    input_range, output_range = input_max - input_min, output_max - output_min
    if np.any(input_range <= 0) or np.any(output_range <= 0):
        raise ValueError("OSC input/output bounds must have positive spans")
    if controller_fields.get("input_ref_frame") not in (None, "world"):
        raise ValueError("OSC controller does not expose world-frame deltas")
    desired_output = np.concatenate((position, rotation))
    action6 = input_min + (desired_output - output_min) * input_range / output_range
    if (
        isinstance(gripper_action, bool)
        or not isinstance(gripper_action, (int, float))
        or not math.isfinite(float(gripper_action))
    ):
        raise ValueError("gripper action must be finite")
    action = np.concatenate((action6, [float(gripper_action)]))
    if not np.isfinite(action).all():
        raise ValueError("computed OSC action is non-finite")
    if np.any(action6 < input_min - 1e-10) or np.any(action6 > input_max + 1e-10):
        raise ValueError("computed OSC command is outside normalized input bounds")
    if not -1.0 <= float(gripper_action) <= 1.0:
        raise ValueError("closed-gripper command is outside normalized gripper bounds")
    return action


def world_pose_tracking_action(
    current_eef_position_world_m: Any,
    current_eef_quaternion_wxyz: Any,
    target_eef_position_world_m: Any,
    target_eef_quaternion_wxyz: Any,
    controller_fields: dict[str, Any],
    *,
    gripper_action: float = 1.0,
) -> np.ndarray:
    """Build an OSC command from live pose feedback and a world pose target."""
    current_position = _finite_vector(current_eef_position_world_m, 3, "current EEF")
    target_position = _finite_vector(target_eef_position_world_m, 3, "target EEF")
    current_quat = _unit_quaternion(
        current_eef_quaternion_wxyz, "current EEF quaternion"
    )
    target_quat = _unit_quaternion(target_eef_quaternion_wxyz, "target EEF quaternion")
    # robosuite v1.4 OSC_POSE pre-multiplies the axis-angle delta in world frame.
    delta_quat = quaternion_multiply(target_quat, quaternion_conjugate(current_quat))
    rotation_delta = quaternion_to_rotation_vector(delta_quat)
    return inverse_osc_world_delta(
        target_position - current_position,
        rotation_delta,
        controller_fields,
        gripper_action=gripper_action,
    )


def check_workspace_target(
    position_world_m: Any,
    base_world_m: Any,
    maximum_reach_radius_m: float,
    controller_position_limits: Any = None,
) -> dict[str, Any]:
    """Check explicit controller Cartesian bounds and a model-derived reach bound."""
    target = np.asarray(position_world_m, dtype=np.float64)
    base = np.asarray(base_world_m, dtype=np.float64)
    if (
        target.shape != (3,)
        or base.shape != (3,)
        or not np.isfinite(target).all()
        or not np.isfinite(base).all()
    ):
        return {"status": "unavailable", "reason": "workspace_position_is_invalid"}
    if (
        isinstance(maximum_reach_radius_m, bool)
        or not isinstance(maximum_reach_radius_m, (int, float))
        or not math.isfinite(float(maximum_reach_radius_m))
        or float(maximum_reach_radius_m) <= 0
    ):
        return {"status": "unavailable", "reason": "workspace_reach_bound_is_invalid"}
    radius = float(np.linalg.norm(target - base))
    if controller_position_limits is not None:
        bounds = np.asarray(controller_position_limits, dtype=np.float64)
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or np.any(bounds[0] > bounds[1])
        ):
            return {
                "status": "unavailable",
                "reason": "controller_position_limits_are_invalid",
            }
        if np.any(target < bounds[0] - 1e-8) or np.any(target > bounds[1] + 1e-8):
            return {
                "status": "fail",
                "reason": "active_controller_workspace_limit_exceeded",
                "distance_from_base_m": radius,
            }
    if radius > float(maximum_reach_radius_m) + 1e-8:
        return {
            "status": "fail",
            "reason": "target_outside_conservative_Panda_reach_bound",
            "distance_from_base_m": radius,
            "maximum_reach_radius_m": float(maximum_reach_radius_m),
        }
    return {
        "status": "pass",
        "distance_from_base_m": radius,
        "maximum_reach_radius_m": float(maximum_reach_radius_m),
    }


def _valid_id_list(value: Any, name: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a non-empty unique list of non-negative IDs")
    return value


def _controller_fields(repair_context: dict[str, Any]) -> dict[str, np.ndarray]:
    robots = repair_context.get("robots")
    if not isinstance(robots, list) or len(robots) != 1:
        raise ValueError("active Panda OSC controller context is not unique")
    robot = robots[0]
    if (
        not isinstance(robot, dict)
        or robot.get("status") != "available"
        or robot.get("controller_name") != "OSC_POSE"
        or robot.get("action_dim") != 7
        or not isinstance(robot.get("fields"), dict)
    ):
        raise ValueError("active controller is not a complete 7-D Panda OSC_POSE")
    fields = robot["fields"]
    if fields.get("input_ref_frame") not in (None, "world"):
        raise ValueError("active OSC_POSE does not expose world-frame deltas")
    orientation_limits = fields.get("orientation_limits")
    if orientation_limits is not None:
        orientation_bounds = np.asarray(orientation_limits, dtype=np.float64)
        if (
            orientation_bounds.shape != (2, 3)
            or not np.isfinite(orientation_bounds).all()
            or np.any(orientation_bounds != 0)
        ):
            raise ValueError(
                "active OSC orientation clipping cannot be proven inactive"
            )
    result = {
        name: _finite_vector(fields.get(name), 6, name)
        for name in ("input_min", "input_max", "output_min", "output_max")
    }
    if np.any(result["input_max"] <= result["input_min"]) or np.any(
        result["output_max"] <= result["output_min"]
    ):
        raise ValueError("active OSC input/output bounds have nonpositive spans")
    return result


def _validate_grasp_rule(teacher: dict[str, Any], object_id: str) -> str:
    rules = [
        rule
        for rule in teacher.get("rules", [])
        if rule.get("target_id") == object_id
        and rule.get("predicate") == "contact"
        and "gripper" in rule.get("subject_parts", [])
    ]
    if len(rules) != 1:
        raise ValueError("gripper_contact_rule_missing_or_ambiguous")
    rule = rules[0]
    if rule.get("subject_parts") != ["gripper"] or rule.get("decision") != "allow":
        raise ValueError("initial_grasp_requires_an_explicit_gripper_allow_rule")
    return rule["id"]


def _validate_compiler_rule_coverage(teacher: dict[str, Any]) -> None:
    for rule in teacher.get("rules", []):
        if rule.get("decision") == "unknown":
            raise ValueError(f"teacher_contact_rule_is_unknown:{rule.get('id')}")
        if rule.get("subject_parts") != ["gripper"]:
            raise ValueError(
                f"teacher_robot_part_rule_is_not_supported_by_current_compiler:{rule.get('id')}"
            )


def _validate_carried_binding(
    bundle: dict[str, Any],
    binding: dict[str, Any],
    semantic_results: dict[str, Any],
    geometry_inventory: dict[str, Any],
) -> tuple[str, list[int]]:
    scene = bundle.get("scene")
    carried = scene.get("carried") if isinstance(scene, dict) else None
    if not isinstance(carried, dict):
        raise ValueError("trajectory bundle has no carried-object scene binding")
    object_id = carried.get("object_id")
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("trajectory bundle carried object_id is invalid")
    if not isinstance(binding, dict) or binding.get("object_id") != object_id:
        raise ValueError("carried object binding does not match the trajectory bundle")
    if binding.get("state_id") != scene.get("state_id"):
        raise ValueError("carried object binding and bundle state_id differ")
    body_id, body_name = binding.get("body_id"), binding.get("body_name")
    if (
        isinstance(body_id, bool)
        or not isinstance(body_id, int)
        or body_id <= 0
        or not isinstance(body_name, str)
        or not body_name
        or binding.get("binding_verified") is not True
        or binding.get("covers_all_associated_collision_geoms") is not True
    ):
        raise ValueError(
            "carried object binding or full collision coverage is unverified"
        )
    geom_ids = _valid_id_list(binding.get("geom_ids"), "carried collision geom_ids")
    body_ids = _valid_id_list(binding.get("body_ids"), "carried body_ids")
    semantic_geom_ids = _valid_id_list(
        binding.get("semantic_binding_geom_ids"), "semantic_binding_geom_ids"
    )
    if (
        body_id not in body_ids
        or not set(semantic_geom_ids).issubset(geom_ids)
        or any(value < 0 for value in body_ids + geom_ids + semantic_geom_ids)
    ):
        raise ValueError("carried Body/Geom binding coverage is inconsistent")
    if not isinstance(binding.get("geom_names"), list) or len(
        binding["geom_names"]
    ) != len(geom_ids):
        raise ValueError("carried collision Geom names are incomplete")

    semantic = (
        semantic_results.get(object_id) if isinstance(semantic_results, dict) else None
    )
    association = (
        semantic.get("physical_association") if isinstance(semantic, dict) else None
    )
    decision = semantic.get("semantic_decision") if isinstance(semantic, dict) else None
    if (
        not isinstance(association, dict)
        or association.get("status") != "matched_by_iou"
        or association.get("body_id") != body_id
        or decision.get("object_id") != object_id
        or decision.get("status") != "resolved"
        or association.get("geom_ids") != semantic_geom_ids
    ):
        raise ValueError(
            "carried semantic binding is unresolved or conflicts with geometry"
        )
    if association.get("body_name") not in (None, body_name):
        raise ValueError("carried semantic body name conflicts with the active binding")
    if (
        geometry_inventory.get("status") != "complete"
        or geometry_inventory.get("robot_classification_status") != "available"
    ):
        raise ValueError("current Geometry Inventory is incomplete")
    entries = geometry_inventory.get("geometries")
    geom_count = geometry_inventory.get("geom_count")
    if not isinstance(entries, list) or len(entries) != geom_count:
        raise ValueError("current Geometry Inventory is malformed")
    by_id = {item.get("geom_id"): item for item in entries if isinstance(item, dict)}
    if len(by_id) != geom_count:
        raise ValueError(
            "current Geometry Inventory has duplicate or malformed Geom IDs"
        )
    for geom_id, geom_name in zip(geom_ids, binding["geom_names"]):
        item = by_id.get(geom_id)
        # The current Compiler identifies objects by root Body ID. Until it can
        # compile articulated object subtrees, fail closed on child-body geoms.
        if (
            item is None
            or item.get("body_id") != body_id
            or item.get("body_name") != body_name
            or item.get("geom_name") != geom_name
            or item.get("category") != "scene"
        ):
            raise ValueError("carried collision Geom differs from the active root Body")
    for geom_id in semantic_geom_ids:
        if geom_id not in geom_ids:
            raise ValueError("semantic Geom binding is not covered by collision Geoms")
    if carried.get("state_id") != scene.get("state_id"):
        raise ValueError("carried sphere and trajectory bundle states differ")
    return object_id, geom_ids


def _candidate_contact_classification(
    candidate: dict[str, Any],
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    gripper_inventory: dict[str, Any],
) -> dict[str, Any]:
    monitor = candidate.get("contact_monitoring")
    events = monitor.get("events") if isinstance(monitor, dict) else None
    if not isinstance(events, list):
        return {"status": "unknown", "reason": "contact_events_missing", "events": []}
    gripper_ids = {
        item.get("geom_id")
        for gripper in gripper_inventory.get("grippers", [])
        if isinstance(gripper, dict)
        for item in gripper.get("geometries", [])
        if isinstance(item, dict)
    }
    body_objects: dict[int, list[str]] = {}
    for object_id, result in semantic_results.items():
        association = (
            result.get("physical_association") if isinstance(result, dict) else None
        )
        body_id = association.get("body_id") if isinstance(association, dict) else None
        if isinstance(body_id, int) and not isinstance(body_id, bool):
            body_objects.setdefault(body_id, []).append(object_id)
    classified = []
    overall = "pass"
    for event in events:
        try:
            geom_ids = [item["id"] for item in event["geom_pair"]]
            body_ids = [item["id"] for item in event["body_pair"]]
            if len(geom_ids) != 2 or len(body_ids) != 2:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            classified.append(
                {"status": "unknown", "reason": "contact_identity_malformed"}
            )
            overall = "unknown"
            continue
        gripper_side = next(
            (index for index, geom_id in enumerate(geom_ids) if geom_id in gripper_ids),
            None,
        )
        if gripper_side is not None:
            object_ids = body_objects.get(body_ids[1 - gripper_side], [])
            rules = [
                rule
                for rule in teacher.get("rules", [])
                if rule.get("target_id") in object_ids
                and rule.get("predicate") == "contact"
                and rule.get("subject_parts") == ["gripper"]
            ]
            if len(rules) == 1 and rules[0].get("decision") == "allow":
                status, reason = (
                    "allowed",
                    "teacher_allows_gripper_contact_for_grasped_object",
                )
                category = "allowed_grasp_contact"
            elif len(rules) == 1 and rules[0].get("decision") == "forbid":
                status, reason = (
                    "violation",
                    "teacher_forbids_gripper_contact_for_grasped_object",
                )
                category = "teacher_forbidden_contact"
                overall = "violation"
            else:
                status, reason = (
                    "unknown",
                    "gripper_contact_has_no_unambiguous_allow_rule",
                )
                category = "unclassified_contact"
                overall = "unknown"
        else:
            object_pair = [body_objects.get(body, []) for body in body_ids]
            matching = [
                rule
                for rule in teacher.get("object_contact_rules", [])
                if rule.get("predicate") == "contact"
                and rule.get("subject_id") in object_pair[0]
                and rule.get("target_id") in object_pair[1]
            ] + [
                rule
                for rule in teacher.get("object_contact_rules", [])
                if rule.get("predicate") == "contact"
                and rule.get("subject_id") in object_pair[1]
                and rule.get("target_id") in object_pair[0]
            ]
            if len(matching) == 1 and matching[0].get("decision") == "allow":
                status, reason = "allowed", "teacher_allows_bound_object_contact"
                category = "teacher_allowed_object_contact"
            elif len(matching) == 1 and matching[0].get("decision") == "forbid":
                status, reason = "violation", "teacher_forbids_bound_object_contact"
                category = "teacher_forbidden_contact"
                overall = "violation"
            else:
                status, reason = (
                    "unknown",
                    "contact_has_no_unambiguous_teacher_allow_rule",
                )
                category = "unclassified_or_other_contact"
                overall = "unknown"
        classified.append(
            {
                "status": status,
                "category": category,
                "reason": reason,
                "geom_pair": event.get("geom_pair"),
                "body_pair": event.get("body_pair"),
                "first_control_step": event.get("first_control_step"),
                "first_physics_substep": event.get("first_physics_substep"),
                "first_contact_position_world": event.get(
                    "first_contact_position_world"
                ),
                "contact_position_status": event.get("contact_position_status"),
                "first_contact_normal_geom1_to_geom2_world": event.get(
                    "first_contact_normal_geom1_to_geom2_world"
                ),
                "contact_normal_status": event.get("contact_normal_status"),
                "max_force_norm": event.get("max_force_norm"),
                "force_sample_count": event.get("force_sample_count"),
                "force_unavailable_count": event.get("force_unavailable_count"),
            }
        )
    return {"status": overall, "events": classified}


def _state_pose_trace(state: Any, carried_body_id: int) -> dict[str, np.ndarray]:
    if not isinstance(state, dict):
        raise ValueError("canonical_branch_state_missing")
    integration = np.asarray(state.get("mujoco_integration_state"), dtype=np.float64)
    robots = state.get("robot_states")
    object_states = state.get("object_states")
    if (
        integration.ndim != 1
        or not integration.size
        or not np.isfinite(integration).all()
        or not isinstance(robots, list)
        or len(robots) != 1
        or not isinstance(object_states, dict)
    ):
        raise ValueError("canonical_branch_state_is_incomplete")
    robot = robots[0]
    if not isinstance(robot, dict):
        raise ValueError("canonical_branch_robot_state_is_malformed")
    eef_position = _finite_vector(robot.get("eef_pos"), 3, "branch EEF position")
    eef_quaternion = _unit_quaternion(robot.get("eef_quat"), "branch EEF quaternion")
    matches = [
        value
        for value in object_states.values()
        if isinstance(value, dict) and value.get("body_id") == carried_body_id
    ]
    if len(matches) != 1:
        raise ValueError("canonical_branch_carried_object_state_missing_or_ambiguous")
    carried = matches[0]
    object_position = _finite_vector(
        carried.get("root_pos"), 3, "branch object position"
    )
    object_quaternion = _unit_quaternion(
        carried.get("root_quat"), "branch object quaternion"
    )
    return {
        "mujoco_integration_state": integration,
        "eef_position": eef_position,
        "eef_quaternion": eef_quaternion,
        "object_position": object_position,
        "object_quaternion": object_quaternion,
    }


def _compile_candidate_contacts(
    teacher: dict[str, Any],
    carried_object_id: str,
    semantic_results: dict[str, Any],
    branch_result: dict[str, Any],
    geometry_inventory: dict[str, Any],
    gripper_inventory: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    """Run the existing Compiler for the carried object and each gripper rule."""
    target_ids = [carried_object_id]
    for rule in teacher.get("rules", []):
        if (
            rule.get("predicate") == "contact"
            and rule.get("subject_parts") == ["gripper"]
            and rule.get("target_id") not in target_ids
        ):
            target_ids.append(rule["target_id"])
    compiled = []
    verdicts = []
    for target_id in target_ids:
        result = compile_forbid_contact(
            teacher,
            target_id,
            semantic_results.get(target_id),
            branch_result,
            geometry_inventory,
            gripper_inventory,
            object_semantic_results=semantic_results,
        )
        candidate_result = next(
            (
                item
                for item in result.get("candidates", [])
                if item.get("candidate_id") == candidate_id
            ),
            {"verdict": "unknown", "reason": "contact_compiler_candidate_missing"},
        )
        verdict = candidate_result.get("verdict")
        verdicts.append(
            verdict if verdict in ("safe", "violation", "unknown") else "unknown"
        )
        compiled.append({"target_object_id": target_id, **candidate_result})
    overall = (
        "violation"
        if "violation" in verdicts
        else "unknown"
        if not verdicts or "unknown" in verdicts
        else "safe"
    )
    return {"verdict": overall, "per_target_compilation": compiled}


def _candidate_assessment(
    candidate: dict[str, Any],
    compiler_result: dict[str, Any],
    *,
    restored: bool,
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    gripper_inventory: dict[str, Any],
    carried_object_id: str,
    carried_body_id: int,
    config: TrackingConfig,
    controller_bounds: dict[str, np.ndarray],
) -> dict[str, Any]:
    _, monitor_error = _complete_monitoring(candidate)
    tracking = candidate.get("closed_loop_tracking", {})
    grasp = candidate.get("grasp_evidence", {})
    task_feasibility = candidate.get("task_feasibility", {})
    classifications = _candidate_contact_classification(
        candidate, teacher, semantic_results, gripper_inventory
    )
    errors = []
    if not restored:
        errors.append("main_snapshot_restore_unverified")
    if candidate.get("snapshot_start_state_match") is not True:
        errors.append("candidate_did_not_start_from_requested_snapshot")
    try:
        start_pose = _state_pose_trace(candidate.get("start_state"), carried_body_id)
        if not isinstance(candidate.get("start_state_fingerprint"), str):
            raise ValueError("candidate_start_fingerprint_missing")
        planned_start = np.asarray(
            candidate.get("reference_path", [])[0], dtype=np.float64
        )
        if (
            planned_start.shape != (3,)
            or not np.isfinite(planned_start).all()
            or np.linalg.norm(start_pose["object_position"] - planned_start)
            > config.state_tolerance_m
        ):
            raise ValueError("candidate_start_object_pose_differs_from_reference")
    except Exception as exc:
        errors.append(f"candidate_start_state_invalid:{exc}")
    if candidate.get("execution_complete") is not True:
        errors.append(
            candidate.get("failure_reason") or "trajectory_execution_incomplete"
        )
    if (
        not isinstance(task_feasibility, dict)
        or task_feasibility.get("status") != "pass"
        or task_feasibility.get("final_task_success_confirmed") is not False
        or task_feasibility.get("complete_path_executed") is not True
        or task_feasibility.get("no_terminal_done") is not True
        or task_feasibility.get("finite_reward_trace") is not True
        or task_feasibility.get("opposing_finger_grasp_maintained") is not True
        or task_feasibility.get("endpoint_within_tracking_tolerance") is not True
    ):
        errors.append("local_task_feasibility_evidence_incomplete")
    steps = candidate.get("steps")
    requested_steps = candidate.get("requested_steps")
    executed_steps = candidate.get("executed_steps")
    if (
        not isinstance(steps, list)
        or isinstance(requested_steps, bool)
        or not isinstance(requested_steps, int)
        or requested_steps <= 0
        or isinstance(executed_steps, bool)
        or not isinstance(executed_steps, int)
        or executed_steps != requested_steps
        or len(steps) != requested_steps
    ):
        errors.append("complete_closed_loop_control_step_trace_missing")
    reference_path = np.asarray(candidate.get("reference_path"), dtype=np.float64)
    goal = np.asarray(candidate.get("goal_world_m"), dtype=np.float64)
    if (
        not isinstance(steps, list)
        or reference_path.shape != (len(steps) + 1, 3)
        or goal.shape != (3,)
        or not np.isfinite(reference_path).all()
        or not np.isfinite(goal).all()
        or np.linalg.norm(reference_path[-1] - goal) > config.state_tolerance_m
    ):
        errors.append("closed_loop_reference_path_or_goal_trace_invalid")
    if monitor_error is not None:
        errors.append(monitor_error)
    if tracking.get("status") != "pass":
        errors.append(tracking.get("reason") or "trajectory_tracking_not_passed")
    if (
        grasp.get("status") != "pass"
        or grasp.get("physical_grasp_confirmed_by_distance_only") is not False
    ):
        errors.append(grasp.get("reason") or "grasp_evidence_not_passed")
    initial_contacts = grasp.get("initial_finger_contacts", {})
    if (
        not isinstance(initial_contacts, dict)
        or initial_contacts.get("status") != "pass"
        or initial_contacts.get("left") is not True
        or initial_contacts.get("right") is not True
    ):
        errors.append("initial_opposing_finger_contact_evidence_missing")
    initial_relative_position = np.asarray(
        grasp.get("initial_eef_object_relative_position_local_m"), dtype=np.float64
    )
    initial_relative_quaternion = np.asarray(
        grasp.get("initial_eef_object_relative_quaternion_local_wxyz"),
        dtype=np.float64,
    )
    if (
        initial_relative_position.shape != (3,)
        or initial_relative_quaternion.shape != (4,)
        or not np.isfinite(initial_relative_position).all()
        or not np.isfinite(initial_relative_quaternion).all()
        or not math.isclose(
            float(np.linalg.norm(initial_relative_quaternion)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        errors.append("initial_gripper_object_relative_pose_missing_or_invalid")
    grasp_steps = grasp.get("per_step")
    if (
        not isinstance(grasp_steps, list)
        or not isinstance(steps, list)
        or len(grasp_steps) != len(steps)
    ):
        errors.append("per_step_grasp_evidence_length_mismatch")
        grasp_steps = []
    for index, step_grasp in enumerate(grasp_steps):
        if (
            not isinstance(step_grasp, dict)
            or step_grasp.get("status") != "pass"
            or step_grasp.get("left") is not True
            or step_grasp.get("right") is not True
        ):
            errors.append(f"step_{index}_opposing_finger_contact_missing")
        relative_position = np.asarray(
            step_grasp.get("eef_object_relative_position_local_m")
            if isinstance(step_grasp, dict)
            else None,
            dtype=np.float64,
        )
        relative_quaternion = np.asarray(
            step_grasp.get("eef_object_relative_quaternion_local_wxyz")
            if isinstance(step_grasp, dict)
            else None,
            dtype=np.float64,
        )
        if (
            relative_position.shape != (3,)
            or relative_quaternion.shape != (4,)
            or not np.isfinite(relative_position).all()
            or not np.isfinite(relative_quaternion).all()
            or not math.isclose(
                float(np.linalg.norm(relative_quaternion)),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            errors.append(
                f"step_{index}_gripper_object_relative_pose_missing_or_invalid"
            )
        for key, limit in (
            (
                "relative_position_error_m",
                config.maximum_grasp_relative_position_error_m,
            ),
            (
                "relative_rotation_error_rad",
                config.maximum_grasp_relative_rotation_error_rad,
            ),
        ):
            value = step_grasp.get(key) if isinstance(step_grasp, dict) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) > limit
            ):
                errors.append(f"step_{index}_{key}_invalid_or_over_limit")
        action = (
            step_grasp.get("gripper_current_action")
            if isinstance(step_grasp, dict)
            else None
        )
        if (
            not isinstance(action, list)
            or len(action) != 1
            or isinstance(action[0], bool)
            or not isinstance(action[0], (int, float))
            or not math.isfinite(float(action[0]))
            or float(action[0]) < 0.9
        ):
            errors.append(f"step_{index}_gripper_not_confirmed_closed")
    if isinstance(steps, list):
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"step_{index}_trace_is_malformed")
                continue
            for key, limit in (
                ("object_tracking_error_m", config.maximum_object_tracking_error_m),
                ("eef_tracking_error_m", config.maximum_eef_tracking_error_m),
            ):
                value = step.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) > limit
                ):
                    errors.append(f"step_{index}_{key}_invalid_or_over_limit")
            action = step.get("action")
            if (
                not isinstance(action, list)
                or len(action) != 7
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in action
                )
                or float(action[6]) < 0.9
            ):
                errors.append(
                    f"step_{index}_seven_dimensional_closed_gripper_action_invalid"
                )
            else:
                action_array = np.asarray(action, dtype=np.float64)
                input_min = controller_bounds["input_min"]
                input_max = controller_bounds["input_max"]
                output_min = controller_bounds["output_min"]
                output_max = controller_bounds["output_max"]
                if (
                    np.any(input_max <= input_min)
                    or np.any(output_max <= output_min)
                    or np.any(action_array[:6] < input_min - 1e-10)
                    or np.any(action_array[:6] > input_max + 1e-10)
                ):
                    errors.append(
                        f"step_{index}_action_outside_active_OSC_input_bounds"
                    )
                else:
                    scaled = output_min + (action_array[:6] - input_min) * (
                        output_max - output_min
                    ) / (input_max - input_min)
                    requested = np.asarray(
                        step.get("requested_osc_output_world_delta"),
                        dtype=np.float64,
                    )
                    actual = np.asarray(
                        step.get("controller_scaled_osc_output"), dtype=np.float64
                    )
                    if (
                        requested.shape != (6,)
                        or actual.shape != (6,)
                        or not np.isfinite(requested).all()
                        or not np.isfinite(actual).all()
                        or np.any(scaled < output_min - 1e-10)
                        or np.any(scaled > output_max + 1e-10)
                        or not np.allclose(scaled, actual, rtol=0.0, atol=1e-9)
                        or not np.allclose(actual, requested, rtol=0.0, atol=1e-9)
                    ):
                        errors.append(f"step_{index}_OSC_scale_or_world_delta_mismatch")
            if step.get("tracking_status") != "pass":
                errors.append(f"step_{index}_tracking_status_not_pass")
            if not isinstance(step.get("state"), dict):
                errors.append(f"step_{index}_canonical_state_missing")
            try:
                state_pose = _state_pose_trace(step.get("state"), carried_body_id)
                actual_object = _finite_vector(
                    step.get("actual_object_center_world_m"), 3, "actual object center"
                )
                actual_eef = _finite_vector(
                    step.get("actual_eef_position_world_m"), 3, "actual EEF position"
                )
                actual_eef_quaternion = _unit_quaternion(
                    step.get("actual_eef_quaternion_world_wxyz"),
                    "actual EEF quaternion",
                )
                actual_object_quaternion = _unit_quaternion(
                    step.get("actual_object_quaternion_world_wxyz"),
                    "actual object quaternion",
                )
                reference = _finite_vector(
                    step.get("reference_object_center_world_m"), 3, "object reference"
                )
                if (
                    np.linalg.norm(actual_object - state_pose["object_position"])
                    > config.state_tolerance_m
                    or np.linalg.norm(actual_eef - state_pose["eef_position"])
                    > config.state_tolerance_m
                    or np.linalg.norm(
                        actual_eef_quaternion - state_pose["eef_quaternion"]
                    )
                    > config.rotation_tolerance_rad
                    or np.linalg.norm(
                        actual_object_quaternion - state_pose["object_quaternion"]
                    )
                    > config.rotation_tolerance_rad
                    or np.linalg.norm(reference - reference_path[index + 1])
                    > config.state_tolerance_m
                ):
                    raise ValueError("reported_pose_differs_from_canonical_state")
                if (
                    np.linalg.norm(actual_object - reference)
                    > config.maximum_object_tracking_error_m
                ):
                    raise ValueError("object_path_tracking_error_exceeded")
            except Exception as exc:
                errors.append(f"step_{index}_state_pose_consistency_failed:{exc}")
            if not isinstance(step.get("contacts"), dict):
                errors.append(f"step_{index}_contact_diagnostic_missing")
            if (
                not isinstance(step.get("joint_limits"), dict)
                or step["joint_limits"].get("status") != "pass"
            ):
                errors.append(f"step_{index}_joint_limit_check_not_passed")
            workspace = step.get("workspace")
            if not isinstance(workspace, dict) or workspace.get("status") != "pass":
                errors.append(f"step_{index}_workspace_check_not_passed")
    if compiler_result.get("verdict") != "safe":
        errors.append(f"contact_compiler_{compiler_result.get('verdict', 'unknown')}")
    if classifications.get("status") != "pass":
        errors.append("contact_event_classification_incomplete_or_unsafe")
    if any(step.get("done") is True for step in candidate.get("steps", [])):
        errors.append("environment_terminated_during_local_tracking")
    if any(
        not isinstance(step.get("reward"), (int, float))
        or not math.isfinite(float(step["reward"]))
        for step in candidate.get("steps", [])
    ):
        errors.append("reward_trace_missing_or_nonfinite")
    passed = not errors
    return {
        "status": "verified_safe_tracking" if passed else "rejected",
        "executable_safe": passed,
        "task_success_confirmed": False,
        "task_feasibility_scope": "complete_local_carry_tracking_without_termination; final task completion is not tested",
        "contact_compiler": compiler_result,
        "contact_classification": classifications,
        "rejection_reasons": errors,
        "complete_substep_monitoring": monitor_error is None,
        "main_snapshot_restore_verified": restored,
    }


def track_trajectory_bundle(
    connection: Any,
    snapshot_id: str,
    bundle: dict[str, Any],
    carried_binding: dict[str, Any],
    repair_context: dict[str, Any],
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    geometry_inventory: dict[str, Any],
    gripper_inventory: dict[str, Any],
    config: TrackingConfig = TrackingConfig(),
) -> dict[str, Any]:
    """Track all geometry-screened paths in isolated Bridge Snapshot branches.

    Call only with a Snapshot captured at the same On-Policy decision state as
    the bundle and a current full-object planning-geometry binding. The helper
    never sends an action to the main environment. Each branch's feedback loop
    runs inside the Bridge and the Contact Compiler evaluates its full substep
    record after the Bridge restores the caller's state.
    """
    config.validate()
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("snapshot_id is required")
    teacher = validate_teacher_output_v2(teacher)
    _validate_compiler_rule_coverage(teacher)
    if not isinstance(semantic_results, dict):
        raise ValueError("semantic_results must map Teacher object IDs to bindings")
    object_id, geom_ids = _validate_carried_binding(
        bundle, carried_binding, semantic_results, geometry_inventory
    )
    allow_rule_id = _validate_grasp_rule(teacher, object_id)
    if (
        not isinstance(repair_context, dict)
        or repair_context.get("status") != "available"
    ):
        raise ValueError("active OSC repair_context is unavailable")
    controller_bounds = _controller_fields(repair_context)
    if (
        not isinstance(gripper_inventory, dict)
        or gripper_inventory.get("status") != "complete"
    ):
        raise ValueError("active Gripper Geometry Inventory is incomplete")
    scene = bundle.get("scene")
    if (
        not isinstance(bundle, dict)
        or bundle.get("status") != "complete"
        or not isinstance(scene, dict)
        or scene.get("frame") != "world"
        or scene.get("units") != "m"
        or not isinstance(scene.get("state_id"), str)
        or scene["state_id"] != carried_binding.get("state_id")
    ):
        raise ValueError(
            "trajectory bundle is incomplete or not from the binding state"
        )
    candidates = bundle.get("candidates")
    if not isinstance(candidates, list) or not candidates or len(candidates) > 24:
        raise ValueError("trajectory bundle must contain 1 to 24 candidates")
    geometric_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("passes_geometric_screen") is True
    ]
    if not geometric_candidates:
        return {
            "status": "no_geometric_candidates",
            "snapshot_id": snapshot_id,
            "candidate_count": len(candidates),
            "tracked_candidate_count": 0,
            "candidates": [],
            "executable_safe_candidate_ids": [],
        }
    request_candidates = []
    for candidate in geometric_candidates:
        candidate_id = candidate.get("candidate_id")
        points = np.asarray(candidate.get("points_world_m"), dtype=np.float64)
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate.get("executable") is not False
            or candidate.get("physical_safety_verified") is not False
            or points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 2
            or not np.isfinite(points).all()
            or np.linalg.norm(
                points[0] - np.asarray(scene["carried"]["center_world_m"])
            )
            > config.state_tolerance_m
            or np.linalg.norm(
                points[-1] - np.asarray(candidate.get("goal_world_m"), dtype=np.float64)
            )
            > config.state_tolerance_m
        ):
            raise ValueError(
                f"geometric candidate {candidate_id!r} has an invalid path endpoint"
            )
        request_candidates.append(
            {
                "candidate_id": candidate_id,
                "direction": candidate.get("direction"),
                "goal_world_m": candidate["goal_world_m"],
                "control_points_world_m": candidate.get("control_points_world_m"),
                "points_world_m": points.tolist(),
            }
        )

    connection.send(
        {
            "op": "track_trajectory_bundle",
            "snapshot_id": snapshot_id,
            "trajectory_state_id": scene["state_id"],
            "carried_binding": {
                "object_id": object_id,
                "body_id": carried_binding["body_id"],
                "body_name": carried_binding["body_name"],
                "body_ids": carried_binding["body_ids"],
                "geom_ids": geom_ids,
                "geom_names": carried_binding["geom_names"],
                "semantic_binding_geom_ids": carried_binding[
                    "semantic_binding_geom_ids"
                ],
                "center_world_m": scene["carried"]["center_world_m"],
                "radius_m": scene["carried"]["radius_m"],
                "state_id": scene["state_id"],
                "binding_verified": True,
                "covers_all_associated_collision_geoms": True,
            },
            "repair_context": repair_context,
            "options": asdict(config),
            "candidates": request_candidates,
        }
    )
    response = connection.recv()
    if not isinstance(response, dict):
        return {
            "status": "unknown",
            "snapshot_id": snapshot_id,
            "error": "Safety Bridge returned a non-object trajectory tracking response",
            "main_state_restored": False,
            "candidates": [],
            "executable_safe_candidate_ids": [],
        }
    results = response.get("candidates")
    if not isinstance(results, list):
        results = []
    expected_ids = {item["candidate_id"] for item in request_candidates}
    by_id = {
        item.get("candidate_id"): item
        for item in results
        if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
    }
    restored = response.get("main_state_restored") is True
    assessed = []
    for request_candidate in request_candidates:
        candidate_id = request_candidate["candidate_id"]
        branch_candidate = by_id.get(candidate_id)
        if branch_candidate is None:
            assessed.append(
                {
                    "candidate_id": candidate_id,
                    "status": "rejected",
                    "executable_safe": False,
                    "task_success_confirmed": False,
                    "rejection_reasons": ["bridge_candidate_result_missing"],
                }
            )
            continue
        branch_result = {
            "main_state_restored": restored,
            "candidates": [branch_candidate],
        }
        compiler_candidate = _compile_candidate_contacts(
            teacher,
            object_id,
            semantic_results,
            branch_result,
            geometry_inventory,
            gripper_inventory,
            candidate_id,
        )
        assessment = _candidate_assessment(
            branch_candidate,
            compiler_candidate,
            restored=restored,
            teacher=teacher,
            semantic_results=semantic_results,
            gripper_inventory=gripper_inventory,
            carried_object_id=object_id,
            carried_body_id=carried_binding["body_id"],
            config=config,
            controller_bounds=controller_bounds,
        )
        assessed.append(
            {
                **branch_candidate,
                **assessment,
                "candidate_id": candidate_id,
                "geometric_candidate": next(
                    item
                    for item in geometric_candidates
                    if item["candidate_id"] == candidate_id
                ),
            }
        )
    if len(by_id) != len(results) or set(by_id) != expected_ids:
        response_errors = ["bridge_candidate_identity_or_count_mismatch"]
    else:
        response_errors = []
    fingerprints = [
        item.get("start_state_fingerprint")
        for item in results
        if isinstance(item, dict)
    ]
    if (
        len(fingerprints) != len(request_candidates)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in fingerprints
        )
        or len(set(fingerprints)) != 1
    ):
        response_errors.append(
            "candidate_snapshot_start_fingerprints_missing_or_mismatched"
        )
    if response.get("error"):
        response_errors.append(str(response["error"]))
    if not restored:
        response_errors.append("main_snapshot_restore_unverified")
    if response_errors:
        for candidate in assessed:
            candidate["executable_safe"] = False
            candidate["status"] = "rejected"
            candidate.setdefault("rejection_reasons", []).extend(response_errors)
    eligible_ids = [
        item["candidate_id"] for item in assessed if item.get("executable_safe")
    ]
    status = (
        "complete"
        if restored and not response_errors and len(results) == len(request_candidates)
        else "unknown"
    )
    return {
        "status": status,
        "snapshot_id": snapshot_id,
        "trajectory_state_id": scene["state_id"],
        "carried_object_id": object_id,
        "allowed_grasp_rule_id": allow_rule_id,
        "requested_candidate_count": len(request_candidates),
        "tracked_candidate_count": len(results),
        "main_state_restored": restored,
        "response_errors": response_errors,
        "candidates": assessed,
        "executable_safe_candidate_ids": eligible_ids if restored else [],
        "final_task_success_confirmed": False,
        "training_label_created": False,
    }
