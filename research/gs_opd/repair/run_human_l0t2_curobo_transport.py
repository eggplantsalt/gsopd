# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0.

"""Execute bounded Human L0T2 placement with CuRobo and real MuJoCo feedback."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import sys
import traceback
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gs_opd.repair.task_success_first_grasp_search import (  # noqa: E402
    BRIDGE_PYTHON,
    BRIDGE_SCRIPT,
    BridgeSession,
    _PythonObjectState,
    _snapshot_state_vector,
)

SNAPSHOT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "gs_opd_human_l0t2_collision_aware_rrt_20261026_run1/"
    "obstacle_avoidance_human_L0T2/stage_00/"
    "task_success_first_executor_obstacle_avoidance_human_L0T2_"
    "20261026_224008_781508768/expert_native_goal_stage_control/"
    "placement_branch_00_region_layout_04/"
    "native_goal_transport_start_runtime_snapshot.pkl"
)
CUROBO_PYTHON = Path("/workspace/gsopd_curobo_eval/.venv/bin/python")
CUROBO_HELPER = Path(__file__).with_name("curobo_human_l0t2_plan_transport.py")
ARCHIVE_ROOT = Path("/opt/data/private/gs_opd/rollouts")
EXPECTED_MODEL_SHA = "633b8b0956f2f38e7f1c62067608f42743d0bccdc6bfeaca11976b5c7f301606"
EXPECTED_BDDL_SHA = "2c4010e44700b0d2069dcd47f489a7411ec364f31cfd69141801b49db3398af7"
EXPECTED_SNAPSHOT_SHA = "b348a065ccb7259a69fd0dc042b42264d257e9f3aba4adb271ff23956729801e"
RECOVERY_CONFIG = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "gs_opd_human_l0t2_t215_bridge_recovery_20261026_run3/launch_config.json"
)
INSTRUCTION = "put both moka pots on the stove"
MAX_DEMO_ACTIONS = 3600
MUJOCO_SUBSTEPS = 25


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pose_matrix(position: Any, quaternion_wxyz: Any) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("invalid_quaternion")
    q = q / np.linalg.norm(q)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def _matrix_pose(matrix: np.ndarray) -> tuple[list[float], list[float]]:
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return matrix[:3, 3].tolist(), [float(q[3]), *q[:3].tolist()]


def _binding(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "object_id": str(row["object_id"]),
        "body_id": int(row["root_body_id"]),
        "body_name": str(row["root_body_name"]),
        "geom_ids": [int(value) for value in row["root_geom_ids"]],
        "geom_names": [str(value) for value in row["root_geom_names"]],
    }


def _query_scene(
    bridge: BridgeSession,
    bindings: dict[str, dict[str, Any]],
    carried_name: str,
    excluded_obstacles: set[str] | None = None,
) -> dict[str, Any]:
    excluded_obstacles = excluded_obstacles or set()
    if carried_name not in bindings:
        raise KeyError(f"missing_semantic_binding:{carried_name}")
    carried_bodies = set(int(value) for value in bindings[carried_name].get("body_ids", []))
    obstacles = []
    for name, row in bindings.items():
        if name == carried_name or name in excluded_obstacles:
            continue
        if not row.get("root_geom_ids"):
            continue
        if carried_bodies & set(int(value) for value in row.get("body_ids", [])):
            continue
        obstacles.append(_binding(row))
        break
    if not obstacles:
        raise ValueError("planning_scene_has_no_verified_obstacle_bindings")
    result = bridge.request({
        "op": "planning_geometry",
        "carried_binding": _binding(bindings[carried_name]),
        "obstacle_bindings": obstacles,
        "expected_instruction": INSTRUCTION,
    })
    if (
        result.get("status") != "complete"
        or result.get("query_complete") is not True
        or result.get("physical_state_unchanged") is not True
        or result.get("model_geometry_sha256") != EXPECTED_MODEL_SHA
    ):
        raise RuntimeError(
            "MuJoCo_collision_world_export_failed:"
            + str(result.get("error") or result.get("status"))
        )
    return result


def _obstacle_spheres(geometry: dict[str, Any], excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    rows: list[dict[str, Any]] = []
    for object_id, obj in geometry.get("objects", {}).items():
        if object_id in excluded:
            continue
        for component in obj.get("components", []):
            rows.append({
                "name": f"{object_id}_geom_{int(component['geom_id'])}",
                "object_id": object_id,
                "geom_id": int(component["geom_id"]),
                "geom_name": component.get("geom_name"),
                "center_world_m": component["center_world_m"],
                "radius_m": component["radius_m"],
                "source": component.get("source"),
            })
    if not rows:
        raise ValueError("MuJoCo_scene_contains_no_collision_sphere_bounds")
    return rows


def _collision_obstacles(
    export: dict[str, Any],
    bindings: dict[str, dict[str, Any]],
    *,
    excluded_objects: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert all active scene collision geoms to CuRobo finite primitives."""
    excluded_objects = excluded_objects or set()
    excluded_ids = {
        int(geom_id)
        for name in excluded_objects
        for geom_id in bindings.get(name, {}).get("geom_ids", [])
    }
    rows = []
    for source in export.get("scene_collision_geometries", []):
        geom_ids = [int(value) for value in source.get("geom_ids", [source.get("geom_id")]) if value is not None]
        if geom_ids and set(geom_ids).issubset(excluded_ids):
            continue
        shape = source["shape"]
        row = {
            "name": source.get("name") or f"mujoco_geom_{geom_ids[0] if geom_ids else 'unknown'}_{source.get('geom_name') or 'unnamed'}",
            "geom_id": geom_ids[0] if geom_ids else None,
            "geom_ids": geom_ids,
            "geom_name": source.get("geom_name"),
            "body_id": int(source["body_id"]),
            "body_name": source.get("body_name"),
            "shape": shape,
            "source": source.get("source"),
        }
        if shape == "sphere":
            row["center_world_m"] = source["center_world_m"]
            row["radius_m"] = float(source["radius_m"])
        elif shape == "cuboid":
            row["center_world_m"] = source["center_world_m"]
            row["rotation_world"] = source["rotation_world"]
            row["half_extents_world_axes_m"] = source["half_extents_world_axes_m"]
        elif shape == "mesh":
            row["vertices_world_m"] = source["vertices_world_m"]
            row["faces"] = source["faces"]
        else:
            raise ValueError(f"unknown_collision_scene_shape:{shape}")
        rows.append(row)
    if not rows:
        raise ValueError("full_MuJoCo_collision_scene_has_no_planner_obstacles")
    return rows


def _object_sphere(geometry: dict[str, Any], object_name: str, root_pose: np.ndarray) -> dict[str, Any]:
    obj = geometry.get("objects", {}).get(object_name)
    if not isinstance(obj, dict) or not obj.get("components"):
        raise ValueError(f"complete_collision_geometry_missing_for_{object_name}")
    inverse_root = np.linalg.inv(root_pose)
    radius = max(
        float(np.linalg.norm((inverse_root @ np.r_[component["center_world_m"], 1.0])[:3]))
        + float(component["radius_m"])
        for component in obj["components"]
    )
    collision_components = [
        {
            "center_object_root_m": (
                inverse_root @ np.r_[component["center_world_m"], 1.0]
            )[:3].tolist(),
            "radius_m": float(component["radius_m"]),
            "source_geom_id": int(component["geom_id"]),
        }
        for component in obj["components"]
    ]
    return {
        "object_id": object_name,
        "root_position_world_m": root_pose[:3, 3].tolist(),
        "root_quaternion_world_wxyz": _matrix_pose(root_pose)[1],
        "conservative_root_sphere_radius_m": float(radius),
        "source_collision_geom_count": len(obj["components"]),
        "covers_all_collision_geoms": bool(obj.get("covers_all_associated_collision_geoms")),
        "collision_components": collision_components,
    }


def _candidate_placements(
    region: dict[str, Any],
    object_pose: np.ndarray,
    object_geometry: dict[str, Any],
    object_name: str,
    occupied_world_positions: list[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    site_position = np.asarray(region["site_position_world_m"], dtype=np.float64)
    site_rotation = np.asarray(region["site_rotation_world"], dtype=np.float64).reshape(3, 3)
    size = np.asarray(region["site_size_m"], dtype=np.float64)
    object_components = object_geometry["objects"][object_name]["components"]
    root_rotation = object_pose[:3, :3]
    local_rotation = site_rotation.T @ root_rotation
    local_components = []
    for component in object_components:
        local_center = root_rotation.T @ (
            np.asarray(component["center_world_m"], dtype=np.float64) - object_pose[:3, 3]
        )
        local_components.append((local_center, float(component["radius_m"])))
    site_up = site_rotation[:, 2]
    support_center = site_position + site_up * float(size[2])
    # Native BDDL On checks the subject root against the Region. Sample a small
    # region-relative grid and four task-equivalent yaw orientations; physical
    # collision/settling remains delegated to CuRobo and MuJoCo execution.
    span_x, span_y = 0.72 * float(size[0]), 0.72 * float(size[1])
    offsets = [
        (0.0, 0.0), (-span_x, -span_y), (-span_x, span_y),
        (span_x, -span_y), (span_x, span_y),
        (-span_x, 0.0), (span_x, 0.0), (0.0, -span_y), (0.0, span_y),
    ]
    yaw_angles_deg = (0.0, 90.0, 180.0, 270.0)
    candidates = []
    for yaw_deg in yaw_angles_deg:
        yaw_local = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
        target_rotation = site_rotation @ yaw_local @ local_rotation
        local_low_z = min(
            float((target_rotation @ local_center)[2]) - radius
            for local_center, radius in local_components
        )
        footprint_radius = max(
            float(np.linalg.norm(local_center[:2])) + radius
            for local_center, radius in local_components
        )
        for offset_index, (dx, dy) in enumerate(offsets):
            target_position = (
                support_center
                + site_rotation[:, 0] * dx
                + site_rotation[:, 1] * dy
                - site_up * local_low_z
                + site_up * 0.003
            )
            root_region_position = site_rotation.T @ (target_position - site_position)
            if not (
                float(size[2]) - 0.005 < root_region_position[2] < float(size[2]) + 0.10
                and abs(root_region_position[0]) < float(size[0])
                and abs(root_region_position[1]) < float(size[1])
            ):
                continue
            if occupied_world_positions:
                target_centers = [
                    target_position + target_rotation @ local_center
                    for local_center, _ in local_components
                ]
                overlap = False
                for prior in occupied_world_positions:
                    if isinstance(prior, dict) and prior.get("components_world"):
                        for target_center, (_, target_radius) in zip(target_centers, local_components):
                            for prior_component in prior["components_world"]:
                                prior_center = np.asarray(prior_component["center_world_m"], dtype=np.float64)
                                prior_radius = float(prior_component["radius_m"])
                                if np.linalg.norm(target_center - prior_center) < target_radius + prior_radius:
                                    overlap = True
                                    break
                            if overlap:
                                break
                    else:
                        prior_position = np.asarray(
                            prior.get("position_world_m") if isinstance(prior, dict) else prior,
                            dtype=np.float64,
                        )
                        prior_radius = float(prior.get("footprint_radius_m", 0.05)) if isinstance(prior, dict) else 0.05
                        delta_local = site_rotation.T @ (target_position - prior_position)
                        if np.linalg.norm(delta_local[:2]) < footprint_radius + prior_radius:
                            overlap = True
                    if overlap:
                        break
                if overlap:
                    continue
            root_goal = np.eye(4, dtype=np.float64)
            root_goal[:3, :3] = target_rotation
            root_goal[:3, 3] = target_position
            candidates.append({
                "candidate_id": f"{object_name}_region_offset_{offset_index:02d}_yaw_{int(yaw_deg):03d}",
                "object_root_pose_world": root_goal.tolist(),
                "site_offset_local_xy_m": [dx, dy],
                "yaw_reorientation_deg": yaw_deg,
                "native_region_root_position": root_region_position.tolist(),
                "predicted_bottom_to_site_top_m": 0.003,
                "conservative_footprint_radius_m": footprint_radius,
            })
            if len(candidates) >= 24:
                return candidates
    if not candidates:
        raise ValueError(f"no_task_equivalent_On_goals_within_native_region:{object_name}")
    return candidates


def _eef_goal_for_object(root_pose: np.ndarray, object_to_eef: np.ndarray) -> tuple[list[float], list[float]]:
    return _matrix_pose(root_pose @ object_to_eef)


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _plan(
    bridge: BridgeSession,
    output_dir: Path,
    stage: str,
    current_q: np.ndarray,
    eef_pose: np.ndarray,
    obstacle_spheres: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    attached_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    input_path = output_dir / f"{stage}_curobo_input.json"
    output_path = output_dir / f"{stage}_curobo_plan.json"
    eef_position, eef_quaternion = _matrix_pose(eef_pose)
    frame_audit = bridge.request({
        "op": "scratch_robot_fk_audit",
        "arm_joint_positions_rad": [current_q.tolist()],
    })
    if (
        frame_audit.get("status") != "complete"
        or frame_audit.get("physical_state_unchanged") is not True
        or len(frame_audit.get("poses", [])) != 1
    ):
        raise RuntimeError(
            "MuJoCo_base_and_tool_frame_audit_failed:"
            + str(frame_audit.get("error") or frame_audit.get("status"))
        )
    pose_audit = frame_audit["poses"][0]
    root_body_name = str(frame_audit["root_body_name"])
    root_pose = pose_audit["bodies"][root_body_name]
    hand_pose = pose_audit["bodies"].get("robot0_right_hand")
    if not isinstance(hand_pose, dict):
        raise RuntimeError("CuRobo panda_hand to MuJoCo robot0_right_hand binding unavailable")
    base_world = _pose_matrix(
        root_pose["position_world_m"], root_pose["quaternion_world_wxyz"]
    )
    live_hand_world = _pose_matrix(
        hand_pose["position_world_m"], hand_pose["quaternion_world_wxyz"]
    )
    live_eef_world = _pose_matrix(
        pose_audit["eef_position_world_m"], pose_audit["eef_quaternion_world_wxyz"]
    )
    eef_to_tool = np.linalg.inv(live_eef_world) @ live_hand_world
    input_data = {
        "task": "obstacle_avoidance_human/L0T2",
        "current_arm_qpos_rad": current_q.tolist(),
        "live_eef_position_world_m": eef_position,
        "live_eef_quaternion_world_wxyz": eef_quaternion,
        "robot_base_pose_world_m": root_pose["position_world_m"],
        "robot_base_quaternion_world_wxyz": root_pose["quaternion_world_wxyz"],
        "live_planner_tool_pose_world_m": hand_pose["position_world_m"],
        "live_planner_tool_quaternion_world_wxyz": hand_pose["quaternion_world_wxyz"],
        "eef_to_planner_tool_transform": eef_to_tool.tolist(),
        "mujoco_fk_audit": frame_audit,
        "obstacle_geometries_world": obstacle_spheres,
        "goal_poses_world": goals,
        "attached_payload": attached_payload,
        "random_seed": 20261026,
        "interpolation_dt_s": 0.05,
        "interpolation_buffer_size": 1800,
    }
    _save_json(input_path, input_data)
    if not CUROBO_PYTHON.is_file() or not CUROBO_HELPER.is_file():
        raise FileNotFoundError("existing_isolated_CuRobo_environment_or_helper_missing")
    completed = subprocess.run(
        [str(CUROBO_PYTHON), str(CUROBO_HELPER), "--input", str(input_path), "--output", str(output_path)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )
    if not output_path.is_file():
        raise RuntimeError(
            "CuRobo planner produced no report: " + completed.stderr[-4000:] + completed.stdout[-2000:]
        )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["process"] = {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
        "input": str(input_path),
        "report": str(output_path),
    }
    _save_json(output_path, result)
    return result


def _decode_rgb(block: dict[str, Any]) -> np.ndarray:
    shape = tuple(block["shape"])
    return np.frombuffer(block["data"], dtype=np.dtype(block["dtype"])).reshape(shape).copy()


def _eef_from_snapshot(snapshot_state: dict[str, Any]) -> np.ndarray:
    robot = snapshot_state["robot_states"][0]
    return _pose_matrix(robot["eef_pos"], robot["eef_quat"])


def _arm_q_from_snapshot(snapshot_state: dict[str, Any]) -> np.ndarray:
    row = np.asarray(snapshot_state["robot_states"][0]["joint_qpos"], dtype=np.float64)
    if row.ndim != 1 or row.size < 7 or not np.isfinite(row[:7]).all():
        raise ValueError("snapshot_robot_arm_joint_state_unavailable")
    return row[:7].copy()


def _object_pose_from_goal_query(result: dict[str, Any]) -> np.ndarray:
    subject = result["subject"]
    return _pose_matrix(subject["root_position_world_m"], subject["root_quaternion_wxyz"])


def _object_state_by_binding(snapshot_state: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row for row in snapshot_state.get("object_states", {}).values()
        if int(row.get("body_id", -1)) == int(binding["root_body_id"])
    ]
    if len(matches) != 1:
        raise ValueError(f"runtime_body_{binding['root_body_id']}_resolved_to_{len(matches)}_objects")
    return matches[0]


def _settling_evidence(
    before_state: dict[str, Any], after_state: dict[str, Any], binding: dict[str, Any]
) -> dict[str, Any]:
    before = _object_state_by_binding(before_state, binding)
    after = _object_state_by_binding(after_state, binding)
    p0 = np.asarray(before.get("root_pos"), dtype=np.float64)
    p1 = np.asarray(after.get("root_pos"), dtype=np.float64)
    qvel = np.asarray(after.get("joint_qvel"), dtype=np.float64)
    if p0.shape != (3,) or p1.shape != (3,) or qvel.ndim != 1 or not np.isfinite(p0).all() or not np.isfinite(p1).all() or not np.isfinite(qvel).all():
        return {"status": "unknown", "reason": "runtime_pose_or_velocity_missing"}
    linear_speed = float(np.linalg.norm(qvel[:3])) if qvel.size >= 3 else None
    angular_speed = float(np.linalg.norm(qvel[3:6])) if qvel.size >= 6 else None
    displacement = float(np.linalg.norm(p1 - p0))
    return {
        "status": "measured",
        "root_displacement_during_settling_m": displacement,
        "final_root_linear_speed_m_s": linear_speed,
        "final_root_angular_speed_rad_s": angular_speed,
        "final_joint_qvel": qvel.tolist(),
        "stable_under_report_policy": bool(
            displacement <= 0.005
            and linear_speed is not None and linear_speed <= 0.02
            and angular_speed is not None and angular_speed <= 0.20
        ),
        "report_policy": {
            "settling_displacement_max_m": 0.005,
            "linear_speed_max_m_s": 0.02,
            "angular_speed_max_rad_s": 0.20,
            "strict_safety_certification": False,
        },
    }


class EpisodeRecorder:
    def __init__(self, output_dir: Path, bridge: BridgeSession):
        self.output_dir = output_dir
        self.bridge = bridge
        self.frames_dir = output_dir / "dual_camera_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.agent_frames_dir = self.frames_dir / "agentview"
        self.wrist_frames_dir = self.frames_dir / "wrist"
        self.agent_frames_dir.mkdir(parents=True, exist_ok=True)
        self.wrist_frames_dir.mkdir(parents=True, exist_ok=True)
        self.records = gzip.open(output_dir / "physical_step_records.jsonl.gz", "at", encoding="utf-8")
        self.action_count = 0
        self.snapshot_paths: list[str] = []
        self.monitoring_incomplete_actions = 0
        self.control_mode = "joint_space_mass_matrix_PD_torque"

    def snapshot(self, label: str) -> str | None:
        result = self.bridge.request({"op": "export_runtime_snapshot"})
        if result.get("status") != "complete" or not isinstance(result.get("payload"), bytes):
            return None
        path = self.output_dir / f"{label}_runtime_snapshot.pkl"
        path.write_bytes(result["payload"])
        self.snapshot_paths.append(str(path))
        return str(path)

    def record(
        self,
        reply: dict[str, Any],
        target_q: np.ndarray | None,
        phase: str,
        gripper_action: float | None,
    ) -> None:
        if reply.get("status") == "failed":
            raise RuntimeError(reply.get("error", "joint_trajectory_step_failed"))
        images = reply.get("images", {})
        if not {"agentview_image", "robot0_eye_in_hand_image"}.issubset(images):
            raise RuntimeError("dual_camera_observation_missing_during_physical_episode")
        agent = _decode_rgb(images["agentview_image"])
        wrist = _decode_rgb(images["robot0_eye_in_hand_image"])
        combined = np.concatenate((agent, wrist), axis=1)
        frame_index = self.action_count
        frame_path = self.frames_dir / f"frame_{frame_index:06d}.jpg"
        agent_path = self.agent_frames_dir / f"frame_{frame_index:06d}.jpg"
        wrist_path = self.wrist_frames_dir / f"frame_{frame_index:06d}.jpg"
        if not all((
            cv2.imwrite(str(frame_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)),
            cv2.imwrite(str(agent_path), cv2.cvtColor(agent, cv2.COLOR_RGB2BGR)),
            cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)),
        )):
            raise OSError("could_not_write_dual_camera_episode_frame")
        monitor = reply.get("contact_monitoring", {})
        if (
            monitor.get("expected_physics_substeps") != MUJOCO_SUBSTEPS
            or monitor.get("captured_physics_substeps") != MUJOCO_SUBSTEPS
            or monitor.get("status") != "complete"
        ):
            self.monitoring_incomplete_actions += 1
        snapshot_state = reply.get("snapshot_state", {})
        q_actual = _arm_q_from_snapshot(snapshot_state)
        record = {
            "schema": "gs_opd_curobo_human_l0t2_episode_step.v1",
            "step_index": self.action_count,
            "phase": phase,
            "control_mode": self.control_mode,
            "control_action_space": "7D_joint_target_qpos_rad",
            "pi05_native_action_compatible": False,
            "control_action": {
                "joint_target_qpos_rad": target_q.tolist() if target_q is not None else None,
                "gripper_action_normalized": gripper_action,
                "applied_joint_torques_nm_per_substep": reply.get("applied_joint_torques_nm_per_substep"),
                "source": "CuRobo joint waypoint executed by MuJoCo mass-matrix PD torque controller",
            },
            "target_joint_qpos_rad": target_q.tolist() if target_q is not None else None,
            "actual_joint_qpos_rad": q_actual.tolist(),
            "observation": {
                "scene_timestep": reply.get("timestep"),
                "robot_state": reply.get("state"),
                "object_states": snapshot_state.get("object_states"),
                "images": {
                    "agentview_image": str(agent_path),
                    "robot0_eye_in_hand_image": str(wrist_path),
                    "composite_dual_camera_frame": str(frame_path),
                },
                "snapshot_state_sha256": hashlib.sha256(
                    np.ascontiguousarray(snapshot_state.get("mujoco_integration_state", [])).tobytes()
                ).hexdigest(),
            },
            "reward": reply.get("reward"),
            "done": reply.get("done"),
            "task_success": reply.get("task_success"),
            "contact_monitoring": monitor,
            "substep_contacts": reply.get("substep_contacts"),
            "snapshot_state": snapshot_state,
        }
        self.records.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        self.records.flush()
        self.action_count += 1

    def close(self) -> None:
        self.records.close()
        images = sorted(self.frames_dir.glob("frame_*.jpg"))
        if not images:
            return
        first = cv2.imread(str(images[0]))
        height, width = first.shape[:2]
        video_path = self.output_dir / "dual_camera_agentview_wrist_20fps.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height)
        )
        if not writer.isOpened():
            raise OSError("OpenCV_video_writer_failed")
        for path in images:
            frame = cv2.imread(str(path))
            if frame is None or frame.shape[:2] != (height, width):
                writer.release()
                raise ValueError(f"invalid_saved_episode_frame:{path.name}")
            writer.write(frame)
        writer.release()


def _joint_step(
    bridge: BridgeSession,
    recorder: EpisodeRecorder,
    q_target: np.ndarray,
    phase: str,
    *,
    gripper_action: float | None = 0.0,
) -> dict[str, Any]:
    if recorder.action_count >= MAX_DEMO_ACTIONS:
        raise RuntimeError("demo_only_horizon_budget_exhausted")
    reply = bridge.request({
        "op": "joint_trajectory_step",
        "action_index": recorder.action_count,
        "joint_target_qpos_rad": np.asarray(q_target, dtype=np.float64).tolist(),
        "gripper_action": gripper_action,
        "joint_position_kp": 50.0,
    })
    recorder.record(reply, q_target, phase, gripper_action)
    if (recorder.action_count % 50) == 0:
        recorder.snapshot(f"step_{recorder.action_count:05d}")
    return reply


def _execute_joint_path(
    bridge: BridgeSession,
    recorder: EpisodeRecorder,
    plan: dict[str, Any],
    phase: str,
    hold_gripper_action: float | None = 0.0,
) -> dict[str, Any]:
    if plan.get("status") != "planned" or not plan.get("joint_path_rad"):
        raise RuntimeError(f"CuRobo route unavailable for {phase}:{plan.get('status')}:{plan.get('error')}")
    path = np.asarray(plan["joint_path_rad"], dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 7 or not np.isfinite(path).all():
        raise ValueError(f"invalid_CuRobo_7D_joint_path:{phase}")
    start = recorder.action_count
    for row in path[1:]:
        reply = _joint_step(bridge, recorder, row, phase, gripper_action=hold_gripper_action)
        if reply.get("task_success") is True:
            break
    recorder.snapshot(f"{phase}_end")
    return {
        "requested_path_points": int(len(path)),
        "executed_control_actions": recorder.action_count - start,
        "final_native_task_success": bool(reply.get("task_success")) if len(path) > 1 else None,
    }


def _make_goals(
    candidates: list[dict[str, Any]],
    object_to_eef: np.ndarray,
    *,
    preplace_up_m: float = 0.14,
) -> list[dict[str, Any]]:
    goals = []
    for candidate in candidates:
        root = np.asarray(candidate["object_root_pose_world"], dtype=np.float64)
        preplace = root.copy()
        preplace[:3, 3] += preplace[:3, 2] * preplace_up_m
        position, quaternion = _eef_goal_for_object(preplace, object_to_eef)
        goals.append({
            "candidate_id": candidate["candidate_id"],
            "eef_position_world_m": position,
            "eef_quaternion_world_wxyz": quaternion,
            "object_root_pose_world": root.tolist(),
            "phase": "preplace",
        })
    return goals


def _make_final_goals(candidates: list[dict[str, Any]], object_to_eef: np.ndarray) -> list[dict[str, Any]]:
    goals = []
    for candidate in candidates:
        root = np.asarray(candidate["object_root_pose_world"], dtype=np.float64)
        position, quaternion = _eef_goal_for_object(root, object_to_eef)
        goals.append({
            "candidate_id": candidate["candidate_id"],
            "eef_position_world_m": position,
            "eef_quaternion_world_wxyz": quaternion,
            "object_root_pose_world": root.tolist(),
            "phase": "place",
        })
    return goals


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    _save_json(output_dir / "run_report.json", report)
    rows = [
        "# Human L0T2 CuRobo transport episode",
        "",
        f"- Result: `{report.get('result')}`",
        f"- Pot1 stable placement: `{report.get('pot1', {}).get('stable_release_verified')}`",
        f"- Pot2 stable placement: `{report.get('pot2', {}).get('stable_release_verified')}`",
        f"- Native task success: `{report.get('final_native_task_status', {}).get('task_success')}`",
        f"- Control mode: `{report.get('control_mode')}`",
        f"- π₀.₅ native action-label compatible: `{report.get('pi05_native_action_compatible')}`",
        f"- Actions: `{report.get('recorded_actions')}`; MuJoCo substeps: `{report.get('recorded_physics_substeps')}`",
        f"- CuRobo plans: `{report.get('curobo_plan_count')}`; planning seconds: `{report.get('planning_seconds')}`",
        f"- Video: `{report.get('artifacts', {}).get('video')}`",
        f"- Step archive: `{report.get('artifacts', {}).get('records')}`",
        f"- Latest Snapshot: `{report.get('artifacts', {}).get('latest_snapshot')}`",
        "",
        "Dangerous contacts are retained as task-success search evidence; this experiment does not certify safety.",
    ]
    (output_dir / "run_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    recovery_config = json.loads(RECOVERY_CONFIG.read_text(encoding="utf-8"))
    payload = SNAPSHOT.read_bytes()
    envelope = pickle.loads(payload)
    source_state = np.asarray(envelope["snapshot"]["mujoco_state"], dtype=np.float64)
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != EXPECTED_MODEL_SHA
        or envelope.get("scene_metadata", {}).get("bddl_sha256") != EXPECTED_BDDL_SHA
        or int(envelope["snapshot"].get("task_bookkeeping", {}).get("timestep", -1)) != 215
        or _sha(payload) != EXPECTED_SNAPSHOT_SHA
        or source_state.shape != (331,)
        or recovery_config.get("source_snapshot", {}).get("payload_sha256") != EXPECTED_SNAPSHOT_SHA
        or recovery_config.get("source_snapshot", {}).get("model_geometry_sha256") != EXPECTED_MODEL_SHA
        or recovery_config.get("task", {}).get("bddl_sha256") != EXPECTED_BDDL_SHA
    ):
        raise ValueError("source_t215_Runtime_Snapshot_or_recovery_config_identity_mismatch")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ARCHIVE_ROOT / f"gs_opd_human_l0t2_curobo_transport_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source_runtime_snapshot.pkl").write_bytes(payload)
    config = {
        "recovery_config": str(RECOVERY_CONFIG),
        "recovery_config_sha256": _sha(RECOVERY_CONFIG.read_bytes()),
        "bridge_python": str(BRIDGE_PYTHON),
        "curobo_python": str(CUROBO_PYTHON),
        "bridge_class": "BridgeSession",
        "task": {
            "benchmark": "obstacle_avoidance_human",
            "level": 0,
            "task_id": 2,
            "seed": 0,
            "init_state": 0,
            "camera_resolution": 256,
            "demonstration_only_horizon": MAX_DEMO_ACTIONS,
            "bddl_sha256": EXPECTED_BDDL_SHA,
        },
        "snapshot": {
            "path": str(SNAPSHOT),
            "payload_sha256": _sha(payload),
            "model_geometry_sha256": EXPECTED_MODEL_SHA,
            "timestep": 215,
        },
        "startup_sequence": [
            "BridgeSession.start() -> seed/reset/set_init_state",
            "one additional Bridge reset from historical executor",
            "require compiled_model_fingerprint_details == source SHA",
            "restore exact t=215 Snapshot",
            "no reset after Snapshot restore",
        ],
        "planner": {
            "existing_environment": str(CUROBO_PYTHON),
            "planner": "CuRobo MotionPlanner goalset with packaged Franka self-collision and complete active MuJoCo scene collision export",
            "payload": "one attached sphere per bound Pot1 MuJoCo collision-Geom enclosure, preserving all 15 source centers/radii",
            "obstacles": "all collision-enabled non-robot MuJoCo Geoms except the carried object and native target support during final docking; cuboids/meshes enclosed by oriented boxes, other finite shapes by exact spheres or verified broadphase spheres",
            "floor": "bounded 5 m workspace cuboid representing the active infinite MuJoCo floor plane",
            "robot": "CuRobo packaged Franka collision model with self_collision_check=true; current MuJoCo robot Geom inventory saved for identity audit",
        },
        "controller": {
            "mode": "joint_space_mass_matrix_PD_torque",
            "training_label_compatible_with_pi05_native_action": False,
            "physical_state_setter_used": False,
        },
    }
    _save_json(output_dir / "launch_config.json", config)
    report: dict[str, Any] = {
        "schema": "gs_opd_curobo_human_l0t2_episode.v1",
        "status": "in_progress",
        "result": "blocked_before_planning",
        "task": "obstacle_avoidance_human/L0T2",
        "source_snapshot_sha256": _sha(payload),
        "output_dir": str(output_dir),
        "source_model_sha256": EXPECTED_MODEL_SHA,
        "bddl_sha256": EXPECTED_BDDL_SHA,
        "control_mode": "joint_space_mass_matrix_PD_torque",
        "pi05_native_action_compatible": False,
        "safety_certified": False,
        "curobo_plan_count": 0,
        "planning_seconds": 0.0,
        "recorded_actions": 0,
        "recorded_physics_substeps": 0,
        "monitoring_incomplete_actions": 0,
        "pot1": {"stable_release_verified": False},
        "pot2": {"attempted": False, "stable_release_verified": False},
        "artifacts": {},
    }
    bridge = BridgeSession(
        ("127.0.0.1", args.port), output_dir / "safety_bridge.log",
        argparse.Namespace(
            benchmark="obstacle_avoidance_human",
            level=0,
            task_id=2,
            init_state=0,
            seed=0,
            camera_resolution=256,
            demonstration_horizon=MAX_DEMO_ACTIONS,
        ),
    )
    recorder = None
    try:
        bridge.start()
        bridge.request({"op": "reset"})
        fingerprints = bridge.request({"op": "compiled_model_fingerprint_details"})
        report["fingerprint_after_required_reset"] = fingerprints
        if fingerprints.get("model_geometry_sha256") != EXPECTED_MODEL_SHA:
            report["result"] = "model_sha_mismatch_before_restore"
            report["status"] = "failed"
            report["blocking_reason"] = "required_historical_reset_did_not_reproduce_source_model_sha"
            return report
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
        restored_state = _snapshot_state_vector(restored.get("snapshot_state", {}))
        if restored.get("status") != "restored" or not np.array_equal(restored_state, source_state):
            report["result"] = "snapshot_restore_not_exact"
            report["status"] = "failed"
            report["blocking_reason"] = str(restored.get("error") or restored.get("status"))
            return report
        report["restore"] = {
            "status": restored.get("status"),
            "model_geometry_sha256": restored.get("model_geometry_sha256"),
            "integration_state_shape": list(restored_state.shape),
            "integration_state_exact_match": True,
            "integration_state_sha256": hashlib.sha256(restored_state.tobytes()).hexdigest(),
        }
        recorder = EpisodeRecorder(output_dir, bridge)
        recorder.snapshot("initial_t215")
        initial_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
        initial_snapshot_state = initial_observation["snapshot_state"]
        initial_eef = _eef_from_snapshot(initial_snapshot_state)
        initial_q = _arm_q_from_snapshot(initial_snapshot_state)
        bindings_response = bridge.request({"op": "task_semantic_geometry_bindings"})
        if bindings_response.get("status") != "complete":
            raise RuntimeError("active_BDDL_semantic_geometry_bindings_unavailable")
        bindings = bindings_response.get("bindings", {})
        if not {"moka_pot_1", "moka_pot_2"}.issubset(bindings):
            raise RuntimeError("active_BDDL_Pot1_or_Pot2_binding_missing")
        grasp_start = bridge.request({
            "op": "grasp_state",
            "carried_binding": _binding(bindings["moka_pot_1"]),
            "expected_instruction": INSTRUCTION,
        })
        if grasp_start.get("grasp_confirmed") is not True or grasp_start.get("physical_state_unchanged") is not True:
            raise RuntimeError("source_Pot1_grasp_no_longer_verified_after_restore")
        region = bridge.request({
            "op": "native_goal_query",
            "predicate": ["on", "moka_pot_1", "flat_stove_1_cook_region"],
        })
        if region.get("status") != "complete" or region.get("predicate_is_native_goal") is not True:
            raise RuntimeError("native_Pot1_BDDL_region_query_failed")
        pot1_pose = _object_pose_from_goal_query(region)
        initial_scene = _query_scene(bridge, bindings, "moka_pot_1")
        if not initial_scene.get("objects", {}).get("moka_pot_1", {}).get("covers_all_associated_collision_geoms"):
            raise RuntimeError("Pot1_collision_geometry_enclosure_is_incomplete")
        collision_export = bridge.request({"op": "collision_scene_export"})
        if (
            collision_export.get("status") != "complete"
            or collision_export.get("query_complete") is not True
            or collision_export.get("physical_state_unchanged") is not True
            or collision_export.get("model_geometry_sha256") != EXPECTED_MODEL_SHA
            or not np.array_equal(
                _snapshot_state_vector({"mujoco_integration_state": collision_export.get("mujoco_integration_state", [])}),
                _snapshot_state_vector({"mujoco_integration_state": initial_snapshot_state.get("mujoco_integration_state", [])}),
            )
        ):
            raise RuntimeError(
                "complete_current_state_CuRobo_collision_scene_unavailable:"
                + str(collision_export.get("error") or collision_export.get("status"))
            )
        initial_object_to_eef = np.linalg.inv(pot1_pose) @ initial_eef
        candidates = _candidate_placements(region["region"], pot1_pose, initial_scene, "moka_pot_1")
        preplace_goals = _make_goals(candidates, initial_object_to_eef)
        obstacle_spheres = _collision_obstacles(
            collision_export, bindings, excluded_objects={"moka_pot_1"}
        )
        _save_json(output_dir / "mujoco_collision_world_t215.json", {
            "model_geometry_sha256": EXPECTED_MODEL_SHA,
            "state_timestep": 215,
            "integration_state_sha256": hashlib.sha256(
                _snapshot_state_vector({"mujoco_integration_state": initial_snapshot_state["mujoco_integration_state"]}).tobytes()
            ).hexdigest(),
            "geometry_source": "read-only Safety Bridge full compiled-model collision export; boxes and meshes use conservative oriented enclosures, spheres are exact, other finite geoms use verified compiled bounds",
            "registered_object_ids": list(initial_scene.get("objects", {})),
            "obstacle_geometry_count": len(obstacle_spheres),
            "obstacle_shape_counts": {
                shape: sum(row["shape"] == shape for row in obstacle_spheres)
                for shape in ("sphere", "cuboid")
            },
            "obstacle_geometries": obstacle_spheres,
            "active_robot_collision_geom_count": len(collision_export.get("robot_collision_geometries", [])),
            "planner_robot_model": "CuRobo packaged Franka model with self_collision_check=True; complete active MuJoCo robot Geom identity inventory retained in collision export",
            "collision_scene_export": collision_export,
            "target_region": region,
            "Pot1_geometry": initial_scene["objects"]["moka_pot_1"],
        })
        payload_sphere = _object_sphere(initial_scene, "moka_pot_1", pot1_pose)
        report.update({
            "result": "planning",
            "start_state": {
                "timestep": initial_observation.get("timestep"),
                "eef_pose_world": _matrix_pose(initial_eef),
                "arm_qpos_rad": initial_q.tolist(),
                "native_task_status": bridge.request({"op": "native_task_status"}),
                "pot1_grasp_state": grasp_start,
            },
            "world": {
                "registered_objects": list(initial_scene["objects"]),
                "obstacle_geometries": len(obstacle_spheres),
                "obstacle_shape_counts": {
                    shape: sum(row["shape"] == shape for row in obstacle_spheres)
                    for shape in ("sphere", "cuboid")
                },
                "active_robot_collision_geom_count": len(collision_export.get("robot_collision_geometries", [])),
                "Pot1_source_collision_geoms": payload_sphere["source_collision_geom_count"],
                "Pot1_conservative_payload_radius_m": payload_sphere["conservative_root_sphere_radius_m"],
            },
            "placement_candidates": candidates,
        })

        t0 = time.monotonic()
        preplan = _plan(bridge, output_dir, "pot1_preplace", initial_q, initial_eef, obstacle_spheres, preplace_goals, payload_sphere)
        report["curobo_plan_count"] += 1
        report["planning_seconds"] += time.monotonic() - t0
        report["pot1"]["preplace_plan"] = preplan
        if preplan.get("status") != "planned":
            report["result"] = "curobo_preplace_route_unavailable"
            report["status"] = "complete"
            report["blocking_reason"] = preplan.get("error") or preplan.get("planner_status")
            return report
        recorder.snapshot("before_pot1_transport")
        report["pot1"]["preplace_execution"] = _execute_joint_path(
            bridge, recorder, preplan, "pot1_curobo_transport", hold_gripper_action=0.0
        )
        live = bridge.request({"op": "observe", "include_snapshot_state": True})
        live_state = live["snapshot_state"]
        live_eef = _eef_from_snapshot(live_state)
        live_q = _arm_q_from_snapshot(live_state)
        live_grasp = bridge.request({
            "op": "grasp_state",
            "carried_binding": _binding(bindings["moka_pot_1"]),
            "expected_instruction": INSTRUCTION,
        })
        if live_grasp.get("grasp_confirmed") is not True:
            raise RuntimeError("Pot1_grasp_lost_during_CuRobo_joint_path_execution")
        live_region = bridge.request({
            "op": "native_goal_query",
            "predicate": ["on", "moka_pot_1", "flat_stove_1_cook_region"],
        })
        live_pot1_pose = _object_pose_from_goal_query(live_region)
        live_object_to_eef = np.linalg.inv(live_pot1_pose) @ live_eef
        live_scene = _query_scene(bridge, bindings, "moka_pot_1")
        live_collision_export = bridge.request({"op": "collision_scene_export"})
        if (
            live_collision_export.get("status") != "complete"
            or live_collision_export.get("physical_state_unchanged") is not True
            or live_collision_export.get("model_geometry_sha256") != EXPECTED_MODEL_SHA
        ):
            raise RuntimeError("live_CuRobo_collision_world_export_failed_after_transport")
        live_candidates = _candidate_placements(
            region["region"], live_pot1_pose, live_scene, "moka_pot_1"
        )
        final_goals = _make_final_goals(live_candidates, live_object_to_eef)
        final_obstacles = _collision_obstacles(
            live_collision_export,
            bindings,
            excluded_objects={"flat_stove_1", "moka_pot_1"},
        )
        t1 = time.monotonic()
        final_plan = _plan(
            bridge,
            output_dir,
            "pot1_place",
            live_q,
            live_eef,
            final_obstacles,
            final_goals,
            _object_sphere(live_scene, "moka_pot_1", live_pot1_pose),
        )
        report["curobo_plan_count"] += 1
        report["planning_seconds"] += time.monotonic() - t1
        report["pot1"]["placement_plan"] = final_plan
        if final_plan.get("status") != "planned":
            report["result"] = "curobo_final_placement_route_unavailable"
            report["status"] = "complete"
            report["blocking_reason"] = final_plan.get("error") or final_plan.get("planner_status")
            return report
        report["pot1"]["selected_candidate_id"] = final_plan["goal_candidates"][final_plan["selected_goal_index"]]["candidate_id"]
        report["pot1"]["placement_execution"] = _execute_joint_path(
            bridge, recorder, final_plan, "pot1_curobo_place", hold_gripper_action=0.0
        )

        release_state = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
        release_q = _arm_q_from_snapshot(release_state)
        release_confirmed = False
        release_actions = 0
        while release_actions < 210 and recorder.action_count < MAX_DEMO_ACTIONS:
            reply = _joint_step(
                bridge, recorder, release_q, "pot1_gripper_release", gripper_action=-1.0
            )
            release_actions += 1
            if release_actions % 10 == 0:
                state = bridge.request({"op": "grasp_state", "carried_binding": _binding(bindings["moka_pot_1"]), "expected_instruction": INSTRUCTION})
                closure = state.get("gripper_closure", {})
                if state.get("grasp_confirmed") is False and closure.get("closed") is False:
                    release_confirmed = True
                    break
        report["pot1"]["release_actions"] = release_actions
        report["pot1"]["release_contact_separated"] = release_confirmed
        recorder.snapshot("pot1_released")
        if not release_confirmed:
            raise RuntimeError("Pot1_gripper_did_not_fully_release_within_210_actions")

        after_release = bridge.request({"op": "observe", "include_snapshot_state": True})
        pot1_after = bridge.request({
            "op": "native_goal_query",
            "predicate": ["on", "moka_pot_1", "flat_stove_1_cook_region"],
        })
        pot1_placed_pose = _object_pose_from_goal_query(pot1_after)
        post_release_export = bridge.request({"op": "collision_scene_export"})
        if (
            post_release_export.get("status") != "complete"
            or post_release_export.get("physical_state_unchanged") is not True
            or post_release_export.get("model_geometry_sha256") != EXPECTED_MODEL_SHA
        ):
            raise RuntimeError("released_scene_collision_export_incomplete")
        retreat_obstacles = _collision_obstacles(
            post_release_export, bindings, excluded_objects={"flat_stove_1", "moka_pot_2"}
        )
        after_release_state = after_release["snapshot_state"]
        retreat_q = _arm_q_from_snapshot(after_release_state)
        retreat_eef = _eef_from_snapshot(after_release_state)
        retreat_goal_world = retreat_eef.copy()
        site_rot = np.asarray(region["region"]["site_rotation_world"], dtype=np.float64).reshape(3, 3)
        retreat_goal_world[:3, 3] += site_rot[:, 2] * 0.12
        retreat_position, retreat_quaternion = _matrix_pose(retreat_goal_world)
        t2 = time.monotonic()
        retreat_plan = _plan(
            bridge,
            output_dir,
            "pot1_retreat",
            retreat_q,
            retreat_eef,
            retreat_obstacles,
            [{"candidate_id": "pot1_release_retreat", "eef_position_world_m": retreat_position,
              "eef_quaternion_world_wxyz": retreat_quaternion}],
            None,
        )
        report["curobo_plan_count"] += 1
        report["planning_seconds"] += time.monotonic() - t2
        report["pot1"]["retreat_plan"] = retreat_plan
        if retreat_plan.get("status") == "planned":
            report["pot1"]["retreat_execution"] = _execute_joint_path(
                bridge, recorder, retreat_plan, "pot1_curobo_retreat", hold_gripper_action=0.0
            )
        else:
            report["pot1"]["retreat_execution"] = {"status": "not_executed", "reason": retreat_plan.get("error") or retreat_plan.get("planner_status")}
            raise RuntimeError("CuRobo_could_not_plan_post_release_retreat")

        # Physical settling uses the same low-level torque hold and open-gripper control.
        settle_q = _arm_q_from_snapshot(bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"])
        for _ in range(40):
            if recorder.action_count >= MAX_DEMO_ACTIONS:
                break
            _joint_step(bridge, recorder, settle_q, "pot1_settling", gripper_action=0.0)
        pot1_settled_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
        final_native = bridge.request({"op": "native_task_status"})
        final_pot1 = bridge.request({"op": "native_goal_query", "predicate": ["on", "moka_pot_1", "flat_stove_1_cook_region"]})
        pot1_stability = _settling_evidence(
            after_release["snapshot_state"],
            pot1_settled_observation["snapshot_state"],
            bindings["moka_pot_1"],
        )
        report["pot1"].update({
            "native_on_after_release_retreat_settling": final_pot1.get("value"),
            "settling_actions": 40,
            "settling_evidence": pot1_stability,
            "stable_release_verified": bool(
                release_confirmed
                and final_pot1.get("value") is True
                and pot1_stability.get("stable_under_report_policy") is True
            ),
            "post_release_native_task_status": final_native,
            "post_release_pose": final_pot1.get("subject"),
        })
        report["final_native_task_status"] = final_native
        report["result"] = "pot1_stable_partial_task_progress"

        # Continue within the demo-only action budget: reuse Pot1's proven object-relative grasp.
        remaining = MAX_DEMO_ACTIONS - recorder.action_count
        report["pot2"]["attempted"] = (
            remaining > 500 and report["pot1"].get("stable_release_verified") is True
        )
        if report["pot1"].get("stable_release_verified") is not True:
            report["pot2"]["status"] = "not_started_Pot1_stable_release_not_verified"
        elif remaining > 500:
            pot2_goal = bridge.request({"op": "native_goal_query", "predicate": ["on", "moka_pot_2", "flat_stove_1_cook_region"]})
            pot2_pose = _object_pose_from_goal_query(pot2_goal)
            pot2_scene = _query_scene(bridge, bindings, "moka_pot_2")
            pot2_export = bridge.request({"op": "collision_scene_export"})
            if pot2_export.get("status") != "complete" or pot2_export.get("physical_state_unchanged") is not True:
                raise RuntimeError("Pot2_pregrasp_collision_export_incomplete")
            pot2_obstacles = _collision_obstacles(
                pot2_export, bindings, excluded_objects={"moka_pot_2"}
            )
            # Reuse the t=215 Pot1-to-hand transform because the paired Pots share the same compiled model family.
            pot2_grasp_eef = pot2_pose @ initial_object_to_eef
            pot2_pregrasp = pot2_grasp_eef.copy()
            pot2_pregrasp[:3, 3] += np.asarray(region["region"]["site_rotation_world"])[:, 2] * 0.12
            p2p, p2q = _matrix_pose(pot2_pregrasp)
            p2_start = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
            pot2_preplan = _plan(
                bridge,
                output_dir, "pot2_pregrasp", _arm_q_from_snapshot(p2_start), _eef_from_snapshot(p2_start),
                pot2_obstacles,
                [{"candidate_id": "pot2_transferred_pot1_grasp", "eef_position_world_m": p2p, "eef_quaternion_world_wxyz": p2q}],
                None,
            )
            report["curobo_plan_count"] += 1
            report["pot2"]["pregrasp_plan"] = pot2_preplan
            if pot2_preplan.get("status") == "planned":
                report["pot2"]["pregrasp_execution"] = _execute_joint_path(
                    bridge, recorder, pot2_preplan, "pot2_curobo_pregrasp", hold_gripper_action=-1.0
                )
                # A bounded final approach uses CuRobo with the target Pot2 omitted from its own world.
                approach_export = bridge.request({"op": "collision_scene_export"})
                if approach_export.get("status") != "complete" or approach_export.get("physical_state_unchanged") is not True:
                    raise RuntimeError("Pot2_grasp_approach_collision_export_incomplete")
                approach_obstacles = _collision_obstacles(
                    approach_export, bindings, excluded_objects={"moka_pot_2"}
                )
                p2_live = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
                approach_plan = _plan(
                    bridge,
                    output_dir, "pot2_grasp", _arm_q_from_snapshot(p2_live), _eef_from_snapshot(p2_live),
                    approach_obstacles,
                    [{"candidate_id": "pot2_transferred_pot1_grasp", "eef_position_world_m": _matrix_pose(pot2_grasp_eef)[0],
                      "eef_quaternion_world_wxyz": _matrix_pose(pot2_grasp_eef)[1]}],
                    None,
                )
                report["curobo_plan_count"] += 1
                report["pot2"]["grasp_plan"] = approach_plan
                if approach_plan.get("status") == "planned":
                    report["pot2"]["grasp_execution"] = _execute_joint_path(
                        bridge, recorder, approach_plan, "pot2_curobo_grasp_approach", hold_gripper_action=-1.0
                    )
                    hold_q = _arm_q_from_snapshot(bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"])
                    pot2_grasp_confirmed = False
                    close_budget = min(210, max(0, MAX_DEMO_ACTIONS - recorder.action_count))
                    for _ in range(close_budget):
                        _joint_step(bridge, recorder, hold_q, "pot2_gripper_close", gripper_action=1.0)
                        if (recorder.action_count % 5) == 0:
                            pot2_grasp = bridge.request({
                                "op": "grasp_state", "carried_binding": _binding(bindings["moka_pot_2"]),
                                "expected_instruction": INSTRUCTION,
                            })
                            if pot2_grasp.get("grasp_confirmed") is True:
                                report["pot2"]["grasp_state"] = pot2_grasp
                                pot2_grasp_confirmed = True
                                break
                    if not pot2_grasp_confirmed:
                        report["pot2"]["grasp_state"] = {"grasp_confirmed": False, "reason": "gripper_close_budget_exhausted"}
                else:
                    report["pot2"]["grasp_plan_failure"] = approach_plan.get("error") or approach_plan.get("planner_status")
            else:
                report["pot2"]["pregrasp_plan_failure"] = pot2_preplan.get("error") or pot2_preplan.get("planner_status")
            if report["pot2"].get("grasp_state", {}).get("grasp_confirmed") is True:
                recorder.snapshot("pot2_grasped_prefix")
                pot2_state = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
                pot2_eef = _eef_from_snapshot(pot2_state)
                pot2_q = _arm_q_from_snapshot(pot2_state)
                pot2_live_goal = bridge.request({"op": "native_goal_query", "predicate": ["on", "moka_pot_2", "flat_stove_1_cook_region"]})
                pot2_live_pose = _object_pose_from_goal_query(pot2_live_goal)
                pot2_live_scene = _query_scene(bridge, bindings, "moka_pot_2")
                if not pot2_live_scene.get("objects", {}).get("moka_pot_2", {}).get("covers_all_associated_collision_geoms"):
                    raise RuntimeError("Pot2_payload_collision_geometry_enclosure_incomplete")
                pot2_object_to_eef = np.linalg.inv(pot2_live_pose) @ pot2_eef
                pot1_live_goal = bridge.request({"op": "native_goal_query", "predicate": ["on", "moka_pot_1", "flat_stove_1_cook_region"]})
                pot1_live_pose = _object_pose_from_goal_query(pot1_live_goal)
                pot1_payload = _object_sphere(pot2_live_scene, "moka_pot_1", pot1_live_pose)
                occupied = [{
                    "position_world_m": pot1_live_pose[:3, 3].tolist(),
                    "footprint_radius_m": pot1_payload["conservative_root_sphere_radius_m"],
                    "components_world": pot2_live_scene["objects"]["moka_pot_1"]["components"],
                }]
                pot2_candidates = _candidate_placements(
                    region["region"], pot2_live_pose, pot2_live_scene, "moka_pot_2",
                    occupied_world_positions=occupied,
                )
                pot2_preplace_goals = _make_goals(pot2_candidates, pot2_object_to_eef)
                current_export = bridge.request({"op": "collision_scene_export"})
                pot2_transport_obstacles = _collision_obstacles(
                    current_export, bindings, excluded_objects={"moka_pot_2"}
                )
                t3 = time.monotonic()
                pot2_transport_plan = _plan(
                    bridge,
                    output_dir, "pot2_preplace", pot2_q, pot2_eef,
                    pot2_transport_obstacles, pot2_preplace_goals,
                    _object_sphere(pot2_live_scene, "moka_pot_2", pot2_live_pose),
                )
                report["curobo_plan_count"] += 1
                report["planning_seconds"] += time.monotonic() - t3
                report["pot2"]["preplace_plan"] = pot2_transport_plan
                if pot2_transport_plan.get("status") != "planned":
                    report["pot2"]["status"] = "transport_route_unavailable"
                    report["pot2"]["blocking_reason"] = pot2_transport_plan.get("error") or pot2_transport_plan.get("planner_status")
                else:
                    report["pot2"]["preplace_execution"] = _execute_joint_path(
                        bridge, recorder, pot2_transport_plan, "pot2_curobo_transport", hold_gripper_action=0.0
                    )
                    p2_after_transport = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
                    p2_after_eef, p2_after_q = _eef_from_snapshot(p2_after_transport), _arm_q_from_snapshot(p2_after_transport)
                    p2_grasp_check = bridge.request({"op": "grasp_state", "carried_binding": _binding(bindings["moka_pot_2"]), "expected_instruction": INSTRUCTION})
                    if p2_grasp_check.get("grasp_confirmed") is not True:
                        raise RuntimeError("Pot2_grasp_lost_during_CuRobo_transport")
                    p2_goal_live = bridge.request({"op": "native_goal_query", "predicate": ["on", "moka_pot_2", "flat_stove_1_cook_region"]})
                    p2_pose_live = _object_pose_from_goal_query(p2_goal_live)
                    p2_scene_live = _query_scene(bridge, bindings, "moka_pot_2")
                    p2_object_to_eef = np.linalg.inv(p2_pose_live) @ p2_after_eef
                    p2_pot1_live = _object_pose_from_goal_query(bridge.request({"op": "native_goal_query", "predicate": ["on", "moka_pot_1", "flat_stove_1_cook_region"]}))
                    p2_pot1_radius = _object_sphere(p2_scene_live, "moka_pot_1", p2_pot1_live)["conservative_root_sphere_radius_m"]
                    p2_candidates = _candidate_placements(
                        region["region"], p2_pose_live, p2_scene_live, "moka_pot_2",
                        occupied_world_positions=[{
                            "position_world_m": p2_pot1_live[:3, 3].tolist(),
                            "footprint_radius_m": p2_pot1_radius,
                            "components_world": p2_scene_live["objects"]["moka_pot_1"]["components"],
                        }],
                    )
                    p2_final_goals = _make_final_goals(p2_candidates, p2_object_to_eef)
                    p2_final_export = bridge.request({"op": "collision_scene_export"})
                    p2_final_obstacles = _collision_obstacles(
                        p2_final_export, bindings,
                        excluded_objects={"flat_stove_1", "moka_pot_2"},
                    )
                    t4 = time.monotonic()
                    p2_final_plan = _plan(
                        bridge,
                        output_dir, "pot2_place", p2_after_q, p2_after_eef,
                        p2_final_obstacles, p2_final_goals,
                        _object_sphere(p2_scene_live, "moka_pot_2", p2_pose_live),
                    )
                    report["curobo_plan_count"] += 1
                    report["planning_seconds"] += time.monotonic() - t4
                    report["pot2"]["placement_plan"] = p2_final_plan
                    if p2_final_plan.get("status") != "planned":
                        report["pot2"]["status"] = "placement_route_unavailable"
                        report["pot2"]["blocking_reason"] = p2_final_plan.get("error") or p2_final_plan.get("planner_status")
                    else:
                        report["pot2"]["selected_candidate_id"] = p2_final_plan["goal_candidates"][p2_final_plan["selected_goal_index"]]["candidate_id"]
                        report["pot2"]["placement_execution"] = _execute_joint_path(
                            bridge, recorder, p2_final_plan, "pot2_curobo_place", hold_gripper_action=0.0
                        )
                        p2_release_state = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
                        p2_release_q = _arm_q_from_snapshot(p2_release_state)
                        p2_released = False
                        p2_release_actions = 0
                        while p2_release_actions < 210 and recorder.action_count < MAX_DEMO_ACTIONS:
                            _joint_step(bridge, recorder, p2_release_q, "pot2_gripper_release", gripper_action=-1.0)
                            p2_release_actions += 1
                            if p2_release_actions % 10 == 0:
                                release_check = bridge.request({"op": "grasp_state", "carried_binding": _binding(bindings["moka_pot_2"]), "expected_instruction": INSTRUCTION})
                                if release_check.get("grasp_confirmed") is False and release_check.get("gripper_closure", {}).get("closed") is False:
                                    p2_released = True
                                    break
                        report["pot2"]["release_actions"] = p2_release_actions
                        report["pot2"]["release_contact_separated"] = p2_released
                        recorder.snapshot("pot2_released")
                        if p2_released:
                            p2_after_release = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
                            p2_retreat_q = _arm_q_from_snapshot(p2_after_release)
                            p2_retreat_eef = _eef_from_snapshot(p2_after_release)
                            p2_retreat_goal = p2_retreat_eef.copy()
                            p2_retreat_goal[:3, 3] += np.asarray(region["region"]["site_rotation_world"], dtype=np.float64).reshape(3, 3)[:, 2] * 0.12
                            p2r, p2rq = _matrix_pose(p2_retreat_goal)
                            p2_retreat_export = bridge.request({"op": "collision_scene_export"})
                            p2_retreat_obstacles = _collision_obstacles(
                                p2_retreat_export, bindings,
                                excluded_objects={"flat_stove_1", "moka_pot_2"},
                            )
                            t5 = time.monotonic()
                            p2_retreat_plan = _plan(
                                bridge,
                                output_dir, "pot2_retreat", p2_retreat_q, p2_retreat_eef,
                                p2_retreat_obstacles,
                                [{"candidate_id": "pot2_release_retreat", "eef_position_world_m": p2r, "eef_quaternion_world_wxyz": p2rq}],
                                None,
                            )
                            report["curobo_plan_count"] += 1
                            report["planning_seconds"] += time.monotonic() - t5
                            report["pot2"]["retreat_plan"] = p2_retreat_plan
                            if p2_retreat_plan.get("status") == "planned":
                                report["pot2"]["retreat_execution"] = _execute_joint_path(
                                    bridge, recorder, p2_retreat_plan, "pot2_curobo_retreat", hold_gripper_action=0.0
                                )
                                p2_settle_q = _arm_q_from_snapshot(bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"])
                                p2_settling_actions = 0
                                for _ in range(40):
                                    if recorder.action_count >= MAX_DEMO_ACTIONS:
                                        break
                                    _joint_step(bridge, recorder, p2_settle_q, "pot2_settling", gripper_action=0.0)
                                    p2_settling_actions += 1
                                p2_settled_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
                                p2_native = bridge.request({"op": "native_task_status"})
                                p2_on = bridge.request({"op": "native_goal_query", "predicate": ["on", "moka_pot_2", "flat_stove_1_cook_region"]})
                                p1_on = bridge.request({"op": "native_goal_query", "predicate": ["on", "moka_pot_1", "flat_stove_1_cook_region"]})
                                p2_stability = _settling_evidence(
                                    p2_after_release["snapshot_state"],
                                    p2_settled_observation["snapshot_state"],
                                    bindings["moka_pot_2"],
                                )
                                report["pot2"].update({
                                    "settling_actions": p2_settling_actions,
                                    "native_on_after_release_retreat_settling": p2_on.get("value"),
                                    "settling_evidence": p2_stability,
                                    "stable_release_verified": bool(
                                        p2_released
                                        and p2_on.get("value") is True
                                        and p1_on.get("value") is True
                                        and p2_stability.get("stable_under_report_policy") is True
                                    ),
                                    "post_release_native_task_status": p2_native,
                                    "post_release_pose": p2_on.get("subject"),
                                    "pot1_native_on_preserved": p1_on.get("value"),
                                })
                                report["final_native_task_status"] = p2_native
                            else:
                                report["pot2"]["status"] = "retreat_route_unavailable_after_release"
                                report["pot2"]["blocking_reason"] = p2_retreat_plan.get("error") or p2_retreat_plan.get("planner_status")
                        else:
                            report["pot2"]["status"] = "gripper_release_not_confirmed"
        report["final_native_task_status"] = bridge.request({"op": "native_task_status"})
        if (
            report["final_native_task_status"].get("task_success") is True
            and report["pot1"].get("stable_release_verified") is True
            and report["pot2"].get("stable_release_verified") is True
        ):
            report["result"] = "native_task_success"
        elif report["final_native_task_status"].get("task_success") is True:
            report["result"] = "native_success_but_stability_or_release_unverified"
        elif report["pot1"].get("stable_release_verified"):
            report["result"] = "pot1_stable_partial_task_progress"
        else:
            report["result"] = "physical_episode_incomplete"
        report["status"] = "complete"
        report["end_timestep"] = report["final_native_task_status"].get("timestep")
        report["terminal_runtime_snapshot"] = recorder.snapshot("terminal")
    except Exception as exc:
        report["status"] = "failed"
        report["result"] = "physical_execution_failed"
        report["blocking_reason"] = f"{type(exc).__name__}: {exc}"
        report["exception_traceback"] = traceback.format_exc()
        try:
            report["failure_native_task_status"] = bridge.request({"op": "native_task_status"})
            report["terminal_runtime_snapshot"] = recorder.snapshot("failure_terminal") if recorder else None
        except Exception as save_exc:
            report["terminal_snapshot_error"] = f"{type(save_exc).__name__}: {save_exc}"
    finally:
        if recorder is not None:
            if report.get("final_native_task_status") is None:
                try:
                    report["final_native_task_status"] = bridge.request({"op": "native_task_status"})
                except Exception as status_exc:
                    report["final_native_task_status_error"] = f"{type(status_exc).__name__}: {status_exc}"
            if report.get("terminal_runtime_snapshot") is None:
                report["terminal_runtime_snapshot"] = recorder.snapshot("terminal")
            recorder.close()
            report["recorded_actions"] = recorder.action_count
            report["recorded_physics_substeps"] = recorder.action_count * MUJOCO_SUBSTEPS
            report["monitoring_incomplete_actions"] = recorder.monitoring_incomplete_actions
            report["snapshot_paths"] = recorder.snapshot_paths
            report["artifacts"] = {
                "records": str(output_dir / "physical_step_records.jsonl.gz"),
                "frames": str(output_dir / "dual_camera_frames"),
                "video": (
                    str(output_dir / "dual_camera_agentview_wrist_20fps.mp4")
                    if (output_dir / "dual_camera_agentview_wrist_20fps.mp4").is_file()
                    else None
                ),
                "video_status": (
                    "complete"
                    if (output_dir / "dual_camera_agentview_wrist_20fps.mp4").is_file()
                    else "not_created_zero_physical_actions"
                ),
                "latest_snapshot": report.get("terminal_runtime_snapshot"),
                "collision_world": str(output_dir / "mujoco_collision_world_t215.json"),
            }
            if report.get("final_native_task_status") is not None:
                report["end_timestep"] = report["final_native_task_status"].get("timestep")
        report["output_dir"] = str(output_dir)
        bridge.close()
        _write_report(output_dir, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--port", type=int, default=38747)
    args = parser.parse_args()
    result = _run(args)
    print(json.dumps({
        "status": result.get("status"),
        "result": result.get("result"),
        "actions": result.get("recorded_actions"),
        "task_success": result.get("final_native_task_status", {}).get("task_success"),
        "blocking_reason": result.get("blocking_reason"),
        "output_dir": result.get("output_dir"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
