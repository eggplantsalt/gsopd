# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Execute the live left B-Spline sphere-center route with native OSC_POSE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from research.gs_opd.repair.trajectory_tracking import (
    TrackingConfig,
    _controller_fields,
    body_local_offset_from_world_point,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_matrix,
    quaternion_to_rotation_vector,
    resample_polyline_by_distance,
    world_pose_tracking_action,
)
from research.gs_opd.smoke.trajectory_bundle_scene_smoke import run_scene_smoke


CONTROL_TIMESTEP_S = 0.05
REFERENCE_SPEED_M_S = 0.020
SOFT_SPHERE_CENTER_ERROR_M = 0.012
MAX_CATCHUP_STEPS_PER_WAYPOINT = 32
MAX_CATCHUP_STEPS_TOTAL = 240
MAX_TOTAL_EXECUTION_STEPS = 1200


def _request(connection: Any, payload: dict[str, Any]) -> dict[str, Any]:
    connection.send(payload)
    result = connection.recv()
    if not isinstance(result, dict):
        raise RuntimeError(f"Safety Bridge returned a non-object for {payload.get('op')}")
    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    return result


def _named_object_state(snapshot_state: dict[str, Any], body_id: int) -> dict[str, Any]:
    objects = snapshot_state.get("object_states")
    matches = [
        value
        for value in objects.values()
        if isinstance(value, dict) and value.get("body_id") == body_id
    ] if isinstance(objects, dict) else []
    if len(matches) != 1:
        raise ValueError("live_snapshot_does_not_have_one_carried_object_state")
    return matches[0]


def _unit_quaternion(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError(f"{label}_must_be_finite_wxyz")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError(f"{label}_has_zero_norm")
    return result / norm


def _pose(snapshot_state: dict[str, Any], body_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    robots = snapshot_state.get("robot_states")
    if not isinstance(robots, list) or len(robots) != 1:
        raise ValueError("live_snapshot_does_not_have_one_Panda_state")
    robot = robots[0]
    obj = _named_object_state(snapshot_state, body_id)
    eef_pos = np.asarray(robot.get("eef_pos"), dtype=np.float64)
    root_pos = np.asarray(obj.get("root_pos"), dtype=np.float64)
    if eef_pos.shape != (3,) or root_pos.shape != (3,) or not np.isfinite(eef_pos).all() or not np.isfinite(root_pos).all():
        raise ValueError("live_snapshot_contains_invalid_EEF_or_object_position")
    return (
        eef_pos,
        _unit_quaternion(robot.get("eef_quat"), "EEF_quaternion"),
        root_pos,
        _unit_quaternion(obj.get("root_quat"), "carried_object_quaternion"),
    )


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _unexpected_contact_events(
    monitoring: dict[str, Any],
    start_contacts: dict[str, Any],
    finger_geom_ids: set[int],
    carried_geom_ids: set[int],
    carried_body_ids: set[int],
) -> list[dict[str, Any]]:
    baseline_pairs = set()
    for contact in start_contacts.get("contacts", []):
        try:
            ids = sorted((int(contact["geom1"]["id"]), int(contact["geom2"]["id"])))
            baseline_pairs.add(tuple(ids))
        except (KeyError, TypeError, ValueError):
            continue
    unexpected = []
    for event in monitoring.get("events", []):
        pair = event.get("geom_pair") or []
        try:
            pair_ids_list = [int(pair[0]["id"]), int(pair[1]["id"])]
            pair_ids = set(pair_ids_list)
            body_ids = {int(item["id"]) for item in event.get("body_pair", [])}
        except (IndexError, KeyError, TypeError, ValueError):
            unexpected.append(event)
            continue
        if tuple(sorted(pair_ids_list)) in baseline_pairs:
            continue
        allowed_grasp_contact = (
            len(pair_ids) == 2
            and bool(pair_ids & finger_geom_ids)
            and bool(pair_ids & carried_geom_ids)
        )
        relevant_new_contact = bool(event.get("robot_involved")) or bool(
            body_ids & carried_body_ids
        )
        if relevant_new_contact and not allowed_grasp_contact:
            unexpected.append(event)
    return unexpected


def _write_execution_svg(path: Path, planned: np.ndarray, actual: list[list[float]]) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    traces = [("B-Spline sphere reference", planned, "#2166ac")]
    if len(actual_array):
        traces.append(("MuJoCo sphere center", actual_array, "#b2182b"))
    points = np.concatenate([trace for _, trace, _ in traces], axis=0)
    low, high = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    span = np.maximum(high - low, 1e-6)
    width, height, margin = 900, 600, 50

    def xy(point: np.ndarray) -> tuple[float, float]:
        return (
            margin + (point[0] - low[0]) / span[0] * (width - 2 * margin),
            height - margin - (point[1] - low[1]) / span[1] * (height - 2 * margin),
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50" y="25" font-family="sans-serif" font-size="16">Direct B-Spline → OSC physical execution (world XY)</text>',
    ]
    for label, trace, color in traces:
        coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(xy, trace))
        lines.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="55" y="{35 + 20 * len(lines)}" font-family="sans-serif" font-size="13" fill="{color}">{label}</text>')
    lines.extend(("</svg>", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _sphere_center_in_exit_region(
    sphere_center: np.ndarray,
    region: dict[str, Any] | None,
    normal_tolerance_m: float,
) -> tuple[bool, dict[str, float] | None]:
    if not isinstance(region, dict):
        return False, None
    delta = np.asarray(sphere_center, dtype=np.float64) - np.asarray(
        region["base_center_world_m"], dtype=np.float64
    )
    coordinates = {
        axis: float(np.dot(delta, np.asarray(region[f"{axis}_world"], dtype=np.float64)))
        for axis in ("normal", "lateral", "up")
    }
    extent = float(region["half_extent_m"])
    inside = (
        abs(coordinates["normal"]) <= normal_tolerance_m
        and abs(coordinates["lateral"]) <= extent
        and abs(coordinates["up"]) <= extent
    )
    return inside, coordinates


def run_live_direct_bspline_osc_smoke(
    connection: Any,
    context: dict[str, Any],
    trigger_grasp_state: dict[str, Any],
    *,
    decision_step: int,
    grasp_stability: dict[str, Any],
    output_json: str | Path,
    trace_writer: Any,
    chunk_index: int,
    step_index: int,
    initial_bridge_data: dict[str, Any],
    video_frames: list[Any] | None,
    video_frame_encoder: Callable[[dict[str, Any]], np.ndarray] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move the actual grasped object along the current left route on the main env."""
    output_path = Path(output_json)
    out_dir = output_path.parent
    report: dict[str, Any] = {
        "operation": "live_direct_bspline_left_native_osc_execution",
        "status": "blocked",
        "decision_step": int(decision_step),
        "grasp_stability": grasp_stability,
        "controller": "native seven-dimensional OSC_POSE through Safety Bridge",
        "execution_environment": "main_MuJoCo_environment",
        "reference_source": "current-state geometrically screened left B-Spline carried sphere-center route",
        "reference_speed_m_s": REFERENCE_SPEED_M_S,
        "control_timestep_s": CONTROL_TIMESTEP_S,
        "max_control_steps_per_candidate": 1200,
        "reference_progression": {
            "type": "adaptive_actual_sphere_center_error",
            "soft_error_threshold_m": SOFT_SPHERE_CENTER_ERROR_M,
            "max_catchup_steps_per_waypoint": MAX_CATCHUP_STEPS_PER_WAYPOINT,
            "max_catchup_steps_total": MAX_CATCHUP_STEPS_TOTAL,
            "max_total_execution_steps": MAX_TOTAL_EXECUTION_STEPS,
        },
        "training_label_created": False,
        "task_success_confirmed": False,
        "physical_collision_observed": False,
        "failure_reason": None,
        "real_execution_steps": 0,
        "route_completed": False,
        "snapshot_baseline": {"captured": False, "restore": "not_applicable_main_environment_execution"},
        "steps": [],
        "artifacts": {},
    }
    latest_bridge_data = initial_bridge_data
    try:
        if (
            grasp_stability.get("stable_grasp_observed") is not True
            or trigger_grasp_state.get("counted_as_stable_sample") is not True
            or trigger_grasp_state.get("grasp_confirmed") is not True
        ):
            raise ValueError("Student_did_not_reach_three_distinct_stable_dual_finger_grasp_states")
        teacher = context.get("teacher")
        semantic_results = context.get("semantic_results")
        if not isinstance(teacher, dict) or not isinstance(semantic_results, dict):
            raise ValueError("live_Teacher_or_semantic_bindings_are_unavailable")

        scene_dir = out_dir / "current_state_scene"
        scene_report = run_scene_smoke(
            connection,
            teacher,
            semantic_results,
            scene_dir,
            subject_id=context.get("subject_id"),
            expected_instruction=context.get("instruction"),
        )
        report["scene_smoke"] = scene_report
        if scene_report.get("status") != "complete" or scene_report.get("geometry_query_succeeded") is not True:
            raise ValueError(
                "current_state_B_Spline_geometry_unavailable:"
                + str(scene_report.get("error") or scene_report.get("reason"))
            )
        query = scene_report["planning_geometry"]
        payload = scene_report["query_payload"]
        carried = payload["carried_binding"]
        bundle_path = scene_dir / "trajectory_bundle.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        candidates = [
            item for item in bundle.get("candidates", [])
            if item.get("direction") == "left" and item.get("passes_geometric_screen") is True
        ]
        if not candidates:
            raise ValueError("current_grasp_state_has_no_geometrically_screened_left_B_Spline_route")
        selected = sorted(candidates, key=lambda item: item["candidate_id"])[0]
        path_points = np.asarray(selected["points_world_m"], dtype=np.float64)
        path_plan = resample_polyline_by_distance(
            path_points,
            speed_m_s=REFERENCE_SPEED_M_S,
            control_timestep_s=CONTROL_TIMESTEP_S,
            maximum_steps=1200,
        )
        reference = path_plan["reference_points_world_m"]
        if not np.isfinite(reference).all() or reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError("left_B_Spline_reference_is_not_a_finite_world_xyz_path")

        snapshot = _request(connection, {"op": "snapshot"})
        snapshot_id = snapshot.get("snapshot_id")
        snapshot_state = snapshot.get("snapshot_state")
        if (
            not isinstance(snapshot_id, str)
            or not snapshot_id
            or not isinstance(snapshot_state, dict)
            or not isinstance(snapshot_state.get("mujoco_integration_state"), list)
            or not snapshot_state.get("robot_states")
            or not snapshot_state.get("object_states")
        ):
            raise ValueError("Bridge_did_not_capture_complete_pre_execution_Snapshot")
        body_id = int(carried["body_id"])
        eef_pos, eef_quat, root_pos, object_quat = _pose(snapshot_state, body_id)
        initial_geometry = query.get("objects", {}).get(carried["object_id"])
        if not isinstance(initial_geometry, dict):
            raise ValueError("carried_semantic_object_missing_from_current_B_Spline_geometry")
        sphere_center = np.asarray(initial_geometry.get("center_world_m"), dtype=np.float64)
        if sphere_center.shape != (3,) or not np.isfinite(sphere_center).all():
            raise ValueError("carried_sphere_center_is_invalid")
        sphere_local = body_local_offset_from_world_point(sphere_center, root_pos, object_quat)
        initial_rotation = quaternion_to_matrix(object_quat)
        initial_grasp_position = initial_rotation.T @ (eef_pos - sphere_center)
        initial_grasp_quaternion = quaternion_multiply(quaternion_conjugate(object_quat), eef_quat)
        if not np.array_equal(
            np.asarray(snapshot_state["mujoco_integration_state"], dtype=np.float64),
            np.asarray(query.get("mujoco_integration_state"), dtype=np.float64),
        ):
            raise ValueError("B_Spline_geometry_state_changed_before_direct_execution")

        live_grasp = _request(
            connection,
            {"op": "grasp_state", "carried_binding": carried, "expected_instruction": context["instruction"]},
        )
        if (
            live_grasp.get("status") != "pass"
            or live_grasp.get("grasp_confirmed") is not True
            or live_grasp.get("state_id") != trigger_grasp_state.get("state_id")
        ):
            raise ValueError("stable_dual_finger_grasp_changed_before_direct_execution")
        controller = _request(connection, {"op": "repair_context"})
        controller_fields = _controller_fields(controller)
        robots = controller.get("robots") or []
        if len(robots) != 1 or robots[0].get("controller_name") != "OSC_POSE" or robots[0].get("action_dim") != 7:
            raise ValueError("active_controller_is_not_native_seven_dimensional_Panda_OSC_POSE")
        gripper_inventory = _request(connection, {"op": "gripper_geometry_inventory"})
        if gripper_inventory.get("status") != "complete":
            raise ValueError("active_Panda_gripper_geometry_inventory_is_incomplete")
        finger_geom_ids = {
            int(geometry["geom_id"])
            for gripper in gripper_inventory.get("grippers", [])
            for geometry in gripper.get("geometries", [])
        }
        carried_geom_ids = {int(item) for item in initial_geometry["geom_ids"]}
        carried_body_ids = {int(item) for item in initial_geometry["body_ids"]}
        if not finger_geom_ids or not carried_geom_ids:
            raise ValueError("finger_or_carried_object_contact_geometries_are_unavailable")

        config = TrackingConfig()
        report.update({
            "selected_route": {
                "candidate_id": selected["candidate_id"],
                "direction": "left",
                "path_length_m": path_plan["path_length_m"],
                "control_steps_requested": path_plan["requested_steps"],
                "reference_points": int(len(reference)),
                "maximum_reference_displacement_m": path_plan["maximum_reference_displacement_m"],
                "passes_geometric_screen": True,
                "physical_safety_verified_before_execution": False,
            },
            "active_controller": controller,
            "gripper_geometry_inventory": gripper_inventory,
            "snapshot_baseline": {
                "captured": True,
                "snapshot_id": snapshot_id,
                "integration_state_length": len(snapshot_state["mujoco_integration_state"]),
                "robot_state_count": len(snapshot_state["robot_states"]),
                "object_state_count": len(snapshot_state["object_states"]),
                "comparison_with_B_Spline_geometry_state": "exact_numeric_match",
                "restore": "not_applicable_main_environment_execution",
            },
            "initial_grasp": live_grasp,
            "limits": {
                "sphere_center_tracking_error_m": config.maximum_sphere_center_tracking_error_m,
                "eef_tracking_error_m": config.maximum_eef_tracking_error_m,
                "grasp_relative_position_error_m": config.maximum_grasp_relative_position_error_m,
                "grasp_relative_rotation_error_rad": config.maximum_grasp_relative_rotation_error_rad,
            "sphere_center_soft_progression_error_m": SOFT_SPHERE_CENTER_ERROR_M,
            "maximum_catchup_steps_per_waypoint": MAX_CATCHUP_STEPS_PER_WAYPOINT,
            "maximum_catchup_steps_total": MAX_CATCHUP_STEPS_TOTAL,
            "maximum_total_execution_steps": min(
                MAX_TOTAL_EXECUTION_STEPS,
                int(path_plan["requested_steps"]) + MAX_CATCHUP_STEPS_TOTAL,
            ),
            "exit_region_normal_tolerance_m": config.maximum_sphere_center_tracking_error_m,
            },
            "training_data_metadata": {
                "source": "direct_b_spline_closed_loop_native_osc_pose",
                "pi05_action_dimension": 7,
                "normalization_modified": False,
                "predicted_student_action_chunk": False,
                "training_label_valid": False,
                "high_quality_task_demonstration": False,
            },
        })
        if trace_writer is not None:
            direct_trace_chunk_index = 1_000_000 + chunk_index
            trace_writer.metadata["direct_bspline_osc_execution"] = report["training_data_metadata"] | {
                "route_candidate_id": selected["candidate_id"],
                "route_direction": "left",
                "reference_speed_m_s": REFERENCE_SPEED_M_S,
                "control_timestep_s": CONTROL_TIMESTEP_S,
            }
            trace_writer.add_decision(
                direct_trace_chunk_index,
                step_index,
                initial_bridge_data,
                np.empty((0, 7), dtype=np.float32),
                safety_record={
                    "source": "direct_b_spline_closed_loop_native_osc_pose",
                    "training_label_valid": False,
                    "high_quality_task_demonstration": False,
                    "selected_route_candidate_id": selected["candidate_id"],
                },
            )

        current_snapshot = snapshot_state
        current_eef_pos, current_eef_quat = eef_pos, eef_quat
        object_state = _named_object_state(current_snapshot, body_id)
        current_sphere = np.asarray(object_state["root_pos"], dtype=np.float64) + quaternion_to_matrix(
            _unit_quaternion(object_state["root_quat"], "carried_object_quaternion")
        ) @ sphere_local
        initial_eef_target_quat = eef_quat.copy()
        actual_sphere_trace: list[list[float]] = [current_sphere.tolist()]
        exit_region = bundle.get("local_exit_region")
        if bundle.get("goal_type") != "local_bypass_goal" or not isinstance(exit_region, dict):
            raise ValueError("left_route_has_no_geometry_defined_local_safe_exit_region")
        execution_step_budget = min(
            MAX_TOTAL_EXECUTION_STEPS,
            int(path_plan["requested_steps"]) + MAX_CATCHUP_STEPS_TOTAL,
        )
        report["execution_budget_steps"] = execution_step_budget
        reference_index = 1
        catchup_steps_at_waypoint = 0
        total_catchup_steps = 0
        waypoint_advances = 0

        while (
            reference_index < len(reference)
            and report["real_execution_steps"] < execution_step_budget
        ):
            target_sphere = reference[reference_index]
            object_state = _named_object_state(current_snapshot, body_id)
            object_pos = np.asarray(object_state["root_pos"], dtype=np.float64)
            object_quat_now = _unit_quaternion(object_state["root_quat"], "carried_object_quaternion")
            object_rotation = quaternion_to_matrix(object_quat_now)
            current_sphere = object_pos + object_rotation @ sphere_local
            live_grasp_position = object_rotation.T @ (current_eef_pos - current_sphere)
            target_eef = target_sphere + initial_rotation @ live_grasp_position
            action = world_pose_tracking_action(
                current_eef_pos,
                current_eef_quat,
                target_eef,
                initial_eef_target_quat,
                controller_fields,
                gripper_action=1.0,
            )
            response = _request(
                connection,
                {
                    "op": "step_monitored",
                    "action": action.tolist(),
                    "action_index": report["real_execution_steps"],
                },
            )
            report["real_execution_steps"] += 1
            latest_bridge_data = response
            current_snapshot = response.get("snapshot_state")
            if not isinstance(current_snapshot, dict):
                raise RuntimeError("OSC_transition_did_not_return_live_MuJoCo_state")
            current_eef_pos, current_eef_quat, root_pos_now, object_quat_now = _pose(current_snapshot, body_id)
            object_rotation_now = quaternion_to_matrix(object_quat_now)
            actual_sphere = root_pos_now + object_rotation_now @ sphere_local
            actual_sphere_trace.append(actual_sphere.tolist())
            grasp_position_now = object_rotation_now.T @ (current_eef_pos - actual_sphere)
            grasp_relative_position_error = float(np.linalg.norm(grasp_position_now - initial_grasp_position))
            relative_quaternion_now = quaternion_multiply(quaternion_conjugate(object_quat_now), current_eef_quat)
            relative_delta = quaternion_multiply(relative_quaternion_now, quaternion_conjugate(initial_grasp_quaternion))
            grasp_relative_rotation_error = float(np.linalg.norm(quaternion_to_rotation_vector(relative_delta)))
            eef_position_error = float(np.linalg.norm(current_eef_pos - target_eef))
            eef_orientation_error = float(np.linalg.norm(quaternion_to_rotation_vector(
                quaternion_multiply(current_eef_quat, quaternion_conjugate(initial_eef_target_quat))
            )))
            sphere_center_error = float(np.linalg.norm(actual_sphere - target_sphere))
            inside_exit_region, exit_coordinates = _sphere_center_in_exit_region(
                actual_sphere,
                exit_region,
                config.maximum_sphere_center_tracking_error_m,
            )
            bowl_orientation_change = float(np.linalg.norm(quaternion_to_rotation_vector(
                quaternion_multiply(object_quat_now, quaternion_conjugate(object_quat))
            )))
            grasp_state = _request(
                connection,
                {"op": "grasp_state", "carried_binding": carried, "expected_instruction": context["instruction"]},
            )
            monitoring = response.get("contact_monitoring", {})
            start_contacts = response.get("start_contacts", {})
            unexpected_contacts = _unexpected_contact_events(
                monitoring,
                start_contacts,
                finger_geom_ids,
                carried_geom_ids,
                carried_body_ids,
            )
            if report["real_execution_steps"] == 1:
                start_contact_list = start_contacts.get("contacts", [])
                report["starting_contact_baseline"] = {
                    "status": start_contacts.get("status", "unavailable"),
                    "relevant_robot_or_carried_pairs": [
                        item
                        for item in start_contact_list
                        if (
                            any(int(body.get("id", -1)) in carried_body_ids for body in (item.get("body1"), item.get("body2")))
                            or any(int(body.get("id", -1)) in set(monitoring.get("robot_body_ids") or []) for body in (item.get("body1"), item.get("body2")))
                        )
                    ],
                }
            record = {
                "step_index": step_index + report["real_execution_steps"] - 1,
                "route_reference_index": reference_index,
                "waypoint_advanced": False,
                "catchup_steps_at_waypoint": catchup_steps_at_waypoint,
                "total_catchup_steps": total_catchup_steps,
                "osc_action_7d": np.asarray(action, dtype=np.float32).tolist(),
                "target_sphere_center_world_m": np.asarray(target_sphere).tolist(),
                "actual_sphere_center_world_m": actual_sphere.tolist(),
                "target_eef_position_world_m": target_eef.tolist(),
                "actual_eef_position_world_m": current_eef_pos.tolist(),
                "actual_eef_quaternion_world_wxyz": current_eef_quat.tolist(),
                "sphere_center_tracking_error_m": sphere_center_error,
                "sphere_center_soft_progression_threshold_m": SOFT_SPHERE_CENTER_ERROR_M,
                "inside_geometry_defined_exit_region": inside_exit_region,
                "exit_region_coordinates_m": exit_coordinates,
                "eef_position_tracking_error_m": eef_position_error,
                "eef_orientation_tracking_error_rad": eef_orientation_error,
                "grasp_relative_position_error_m": grasp_relative_position_error,
                "grasp_relative_rotation_error_rad": grasp_relative_rotation_error,
                "bowl_orientation_change_from_start_rad": bowl_orientation_change,
                "grasp_state": grasp_state,
                "contact_monitoring_status": monitoring.get("status"),
                "contact_monitoring_control_step": (monitoring.get("control_steps") or [None])[0],
                "unexpected_contact_events": unexpected_contacts,
                "reward": float(response.get("reward", 0.0)),
                "done": bool(response.get("done", False)),
            }
            report["steps"].append(record)
            if trace_writer is not None:
                trace_writer.add_step(
                    step_index=record["step_index"],
                    chunk_index=direct_trace_chunk_index,
                    action_index=report["real_execution_steps"] - 1,
                    action=action,
                    bridge_data=response,
                )
            if video_frames is not None and video_frame_encoder is not None:
                video_frames.append(video_frame_encoder(response))
            if response.get("reward", 0.0) > 0:
                report["task_success_confirmed"] = True
            report["physical_collision_observed"] = bool(unexpected_contacts)
            if monitoring.get("status") != "complete":
                report["failure_reason"] = "MuJoCo_substep_contact_monitoring_incomplete"
            elif unexpected_contacts:
                report["failure_reason"] = "unexpected_physical_contact_observed"
            elif grasp_state.get("status") != "pass" or grasp_state.get("grasp_confirmed") is not True:
                report["failure_reason"] = "bilateral_grasp_or_gripper_closure_lost"
            elif sphere_center_error > config.maximum_sphere_center_tracking_error_m:
                report["failure_reason"] = "sphere_center_tracking_error_exceeded_existing_limit"
            elif eef_position_error > config.maximum_eef_tracking_error_m:
                report["failure_reason"] = "EEF_tracking_error_exceeded_existing_limit"
            elif grasp_relative_position_error > config.maximum_grasp_relative_position_error_m:
                report["failure_reason"] = "grasp_relative_position_error_exceeded_existing_limit"
            elif grasp_relative_rotation_error > config.maximum_grasp_relative_rotation_error_rad:
                report["failure_reason"] = "grasp_relative_rotation_error_exceeded_existing_0.20_rad_limit"
            elif response.get("done"):
                report["failure_reason"] = "MuJoCo_environment_terminated_before_left_route_completion"
            if report["failure_reason"]:
                break
            if inside_exit_region and grasp_state.get("status") == "pass" and not unexpected_contacts:
                report["route_completed"] = True
                report["status"] = "left_bypass_route_physically_completed"
                report["safe_exit_region_reached"] = True
                report["safe_exit_region_actual_sphere_center_world_m"] = actual_sphere.tolist()
                break
            if sphere_center_error <= SOFT_SPHERE_CENTER_ERROR_M:
                if reference_index < len(reference) - 1:
                    record["waypoint_advanced"] = True
                    reference_index += 1
                    waypoint_advances += 1
                    catchup_steps_at_waypoint = 0
                else:
                    catchup_steps_at_waypoint += 1
                    total_catchup_steps += 1
            else:
                catchup_steps_at_waypoint += 1
                total_catchup_steps += 1
            record["catchup_steps_at_waypoint"] = catchup_steps_at_waypoint
            record["total_catchup_steps"] = total_catchup_steps
            report["catchup_steps_total"] = total_catchup_steps
            report["waypoint_advances"] = waypoint_advances
            if catchup_steps_at_waypoint >= MAX_CATCHUP_STEPS_PER_WAYPOINT:
                report["failure_reason"] = "waypoint_catchup_budget_exhausted"
                break
            if total_catchup_steps >= MAX_CATCHUP_STEPS_TOTAL:
                report["failure_reason"] = "total_catchup_budget_exhausted"
                break
            if report["real_execution_steps"] % 50 == 0:
                print(
                    "DIRECT B-SPLINE→OSC:", report["real_execution_steps"], "/", execution_step_budget,
                    "sphere_err_m=", round(sphere_center_error, 4),
                    "reference=", reference_index,
                    "catchup=", total_catchup_steps,
                    "grasp_rot_err_rad=", round(grasp_relative_rotation_error, 4),
                    "grasp=", grasp_state.get("status"),
                    flush=True,
                )

        if report["failure_reason"] is None and not report["route_completed"]:
            if report["real_execution_steps"] >= execution_step_budget:
                report["failure_reason"] = "total_execution_budget_exhausted_before_safe_exit_region"
            elif reference_index >= len(reference):
                report["failure_reason"] = "left_route_reference_ended_outside_safe_exit_region"
            else:
                report["failure_reason"] = "left_route_stopped_before_safe_exit_region"
            report["status"] = "incomplete"
        elif report["failure_reason"] is not None:
            report["status"] = "stopped_on_first_physical_or_tracking_failure"

        final_error = float(np.linalg.norm(actual_sphere_trace[-1] - np.asarray(reference[-1])))
        report["final_sphere_center_error_to_route_endpoint_m"] = final_error
        report["catchup_steps_total"] = total_catchup_steps
        report["waypoint_advances"] = waypoint_advances

        report["execution_summary"] = {
            "actual_osc_steps": report["real_execution_steps"],
            "first_failure_reason": report["failure_reason"],
            "maximum_sphere_center_tracking_error_m": max((item["sphere_center_tracking_error_m"] for item in report["steps"]), default=None),
            "maximum_eef_position_tracking_error_m": max((item["eef_position_tracking_error_m"] for item in report["steps"]), default=None),
            "maximum_eef_orientation_tracking_error_rad": max((item["eef_orientation_tracking_error_rad"] for item in report["steps"]), default=None),
            "maximum_grasp_relative_position_error_m": max((item["grasp_relative_position_error_m"] for item in report["steps"]), default=None),
            "maximum_grasp_relative_rotation_error_rad": max((item["grasp_relative_rotation_error_rad"] for item in report["steps"]), default=None),
            "final_dual_finger_grasp_status": report["steps"][-1]["grasp_state"].get("status") if report["steps"] else live_grasp.get("status"),
        }
        execution_svg = out_dir / "direct_bspline_osc_execution.svg"
        _write_execution_svg(execution_svg, reference, actual_sphere_trace)
        report["artifacts"].update({
            "trajectory_bundle": str(bundle_path),
            "planned_route_svg": str(scene_dir / "trajectory_bundle.svg"),
            "executed_route_svg": str(execution_svg),
            "trajectory_npz": str(trace_writer.output_path) if trace_writer is not None else None,
        })
        _json_write(output_path, report)
        return report, latest_bridge_data
    except Exception as exc:
        report["status"] = "blocked_before_or_during_direct_execution"
        report["failure_reason"] = f"{type(exc).__name__}:{exc}"
        report["execution_summary"] = {
            "actual_osc_steps": report["real_execution_steps"],
            "first_failure_reason": report["failure_reason"],
        }
        try:
            _json_write(output_path, report)
        except Exception:
            pass
        return report, latest_bridge_data
