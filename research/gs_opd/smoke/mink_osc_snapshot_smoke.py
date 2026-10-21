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
"""Live-grasp Mink planning and native OSC Snapshot branch Smoke.

Student runs through the persistent Safety Bridge until it reports three
distinct dual-finger grasp states. The same Bridge Snapshot then seeds a new
left B-Spline and Mink plan; its MuJoCo-FK EEF poses are tracked in an isolated
Snapshot branch using the native OSC_POSE controller. No archived action
sequence or archived physical state is loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEACHER = (
    REPO_ROOT / "logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt"
)
DEFAULT_SEMANTICS = REPO_ROOT / "logs/gs_opd/semantic_bindings_l0t0.json"
DEFAULT_MINK_PYTHON = Path("/workspace/zy_workspace/envs/gsopd_mink/bin/python")
DEFAULT_LIBERO_PYTHON = Path("/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python")
DEFAULT_ROLLOUT_PYTHON = Path("/workspace/zy_workspace/RLinf/.venv/bin/python")
DEFAULT_LIBERO_PYTHONPATH = ":".join(
    (
        "/workspace/zy_workspace/LIBERO-Safety",
        "/workspace/zy_workspace/LIBERO-Safety/third_party/robosuite-1.4",
        str(REPO_ROOT),
    )
)
CONTROL_DT_S = 0.05
REFERENCE_SPEED_M_S = 0.02
STATE_ATOL = 1e-6


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}_must_be_finite_with_shape_{shape}")
    return array


def _quat_normalize(value: Any) -> np.ndarray:
    quat = _finite_array(value, (4,), "quaternion")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("quaternion_has_zero_norm")
    return quat / norm


def quaternion_angle_rad(left: Any, right: Any) -> float:
    a, b = _quat_normalize(left), _quat_normalize(right)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0)))


def _validated_mink_trace(plan: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return qpos, attached sphere centers, and complete FK EEF poses (N x 7)."""
    if not isinstance(plan, dict):
        raise ValueError("Mink_plan_must_be_an_object")
    joints = np.asarray(plan.get("joint_trajectory_rad"), dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 7 or len(joints) < 2 or not np.isfinite(joints).all():
        raise ValueError("Mink_joint_trajectory_must_be_finite_N_by_7")
    positions = _finite_array(
        plan.get("eef_trajectory_world_m"), (len(joints), 3), "Mink EEF positions"
    )
    quaternions = np.asarray(plan.get("eef_quaternion_world_wxyz"), dtype=np.float64)
    if quaternions.shape != (len(joints), 4) or not np.isfinite(quaternions).all():
        raise ValueError("Mink_FK_EEF_quaternions_must_be_finite_N_by_4")
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-12) or np.any(np.abs(norms - 1.0) > 1e-5):
        raise ValueError("Mink_FK_EEF_quaternions_must_be_unit_wxyz")
    quaternions = quaternions / norms[:, None]
    spheres = _finite_array(
        plan.get("sphere_trajectory_world_m"),
        (len(joints), 3),
        "Mink attached-sphere centers",
    )
    return joints, spheres, np.concatenate((positions, quaternions), axis=1)


def _named_object_state(snapshot_state: Any, body_id: int) -> dict[str, Any] | None:
    objects = (
        snapshot_state.get("object_states")
        if isinstance(snapshot_state, dict)
        else None
    )
    if not isinstance(objects, dict):
        return None
    matches = [
        value
        for value in objects.values()
        if isinstance(value, dict) and value.get("body_id") == body_id
    ]
    return matches[0] if len(matches) == 1 else None


def _bridge_request(connection: Any, request: dict[str, Any]) -> dict[str, Any]:
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError(f"Bridge returned non-object for {request.get('op')}")
    if response.get("error"):
        raise RuntimeError(str(response["error"]))
    return response


def _snapshot_comparison(expected: Any, actual: Any) -> dict[str, Any]:
    mismatches: list[str] = []
    maximum_error = 0.0

    def compare(left: Any, right: Any, path: str) -> None:
        nonlocal maximum_error
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                mismatches.append(path + ":keys")
                return
            for key in left:
                compare(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatches.append(path + ":length")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{path}[{index}]")
        elif isinstance(left, bool) or isinstance(right, bool):
            if left is not right:
                mismatches.append(path)
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not math.isfinite(float(left)) or not math.isfinite(float(right)):
                mismatches.append(path + ":nonfinite")
            else:
                error = abs(float(left) - float(right))
                maximum_error = max(maximum_error, error)
                if error > STATE_ATOL:
                    mismatches.append(path)
        elif left != right:
            mismatches.append(path)

    compare(expected, actual, "snapshot_state")
    return {
        "matched": not mismatches,
        "atol": STATE_ATOL,
        "maximum_numeric_error": maximum_error,
        "mismatches": mismatches[:64],
        "mismatch_count": len(mismatches),
    }


def _write_svg(
    path: Path, planned: np.ndarray, actual_eef: np.ndarray, actual_sphere: np.ndarray
) -> None:
    width, height, margin, gap = 1200, 420, 55, 40
    views = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    all_points = np.concatenate((planned, actual_eef, actual_sphere), axis=0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    panel_w = (width - 2 * margin - 2 * gap) / 3
    colors = {
        "Mink FK sphere": "#2563eb",
        "OSC EEF": "#dc2626",
        "OSC sphere": "#16a34a",
    }
    for panel, (xaxis, yaxis, title) in enumerate(views):
        x0 = margin + panel * (panel_w + gap)
        values = all_points[:, [xaxis, yaxis]]
        lo, hi = values.min(axis=0), values.max(axis=0)
        span = np.maximum(hi - lo, 1e-4)
        lo, hi = lo - 0.05 * span, hi + 0.05 * span

        def convert(point: np.ndarray) -> tuple[float, float]:
            px = x0 + 10 + (point[xaxis] - lo[0]) / (hi[0] - lo[0]) * (panel_w - 20)
            py = 60 + 320 - (point[yaxis] - lo[1]) / (hi[1] - lo[1]) * 300
            return px, py

        parts.append(
            f'<text x="{x0 + 8:.1f}" y="30" font-family="sans-serif" font-size="20">{title}</text>'
        )
        for label, trace in (
            ("Mink FK sphere", planned),
            ("OSC EEF", actual_eef),
            ("OSC sphere", actual_sphere),
        ):
            points = " ".join(
                f"{x:.2f},{y:.2f}" for x, y in (convert(item) for item in trace)
            )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{colors[label]}" stroke-width="2" opacity="0.85"/>'
            )
    for index, (label, color) in enumerate(colors.items()):
        x = margin + index * 225
        parts.append(
            f'<line x1="{x}" y1="405" x2="{x + 28}" y2="405" stroke="{color}" stroke-width="3"/><text x="{x + 36}" y="410" font-family="sans-serif" font-size="13">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def run_live_snapshot_mink_osc_smoke(
    connection: Any,
    context: dict[str, Any],
    trigger_grasp_state: dict[str, Any],
    *,
    decision_step: int,
    grasp_stability: dict[str, Any],
    output_json: str | Path,
    mink_python: str | Path,
    model_binary: str | Path,
    model_metadata: str | Path,
) -> dict[str, Any]:
    """Plan and execute one current-state left route on the persistent Bridge."""
    import subprocess

    from research.gs_opd.repair.trajectory_bundle_acceptance import (
        _diagnostic_equal,
    )
    from research.gs_opd.repair.trajectory_tracking import (
        TrackingConfig,
        body_local_offset_from_world_point,
        quaternion_conjugate,
        quaternion_multiply,
        quaternion_to_matrix,
        track_trajectory_bundle,
    )
    from research.gs_opd.smoke.trajectory_bundle_scene_smoke import run_scene_smoke

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_path.parent
    teacher = context.get("teacher")
    semantic_results = context.get("semantic_results")
    instruction = context.get("instruction")
    subject_id = context.get("subject_id")
    report: dict[str, Any] = {
        "operation": "live_snapshot_mink_left_native_osc_smoke",
        "status": "blocked",
        "decision_step": int(decision_step),
        "grasp_stability": grasp_stability,
        "reference_source": "current-state Scene Smoke B-Spline -> Mink joint reference -> current-model MuJoCo FK EEF poses",
        "controller": "native LIBERO-Safety OSC_POSE via persistent Safety Bridge env.step",
        "main_environment_candidate_actions": 0,
        "training_label_created": False,
        "final_task_success_confirmed": False,
        "failure_reasons": [],
        "snapshot_capture": {"captured": False},
        "initial_snapshot_restore": {"attempted": False, "verified": False},
        "branch": None,
        "final_snapshot_restore": {"attempted": False, "verified": False},
    }
    snapshot_id = None
    snapshot_state = None
    query = None
    controller_before = None
    try:
        if (
            grasp_stability.get("stable_grasp_observed") is not True
            or trigger_grasp_state.get("counted_as_stable_sample") is not True
            or trigger_grasp_state.get("grasp_confirmed") is not True
        ):
            raise ValueError("Student_did_not_reach_three_distinct_stable_dual_finger_grasp_states")
        if not isinstance(teacher, dict) or not isinstance(semantic_results, dict):
            raise ValueError("live_Teacher_or_semantic_binding_context_is_missing")
        capabilities = _bridge_request(
            connection, {"op": "trajectory_tracking_capabilities"}
        )
        if (
            capabilities.get("explicit_eef_reference_poses_world") is not True
            or capabilities.get("substep_contact_monitoring") is not True
            or capabilities.get("full_snapshot_restore_verification") is not True
        ):
            raise RuntimeError("persistent_Bridge_lacks_required_Mink_pose_tracking_protocol")
        report["bridge_capabilities"] = capabilities
        snapshot = _bridge_request(connection, {"op": "snapshot"})
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
            raise ValueError("Bridge_did_not_capture_complete_physical_Snapshot_diagnostics")
        report["snapshot_capture"] = {
            "snapshot_id": snapshot_id,
            "captured": True,
            "integration_state_length": len(snapshot_state["mujoco_integration_state"]),
            "robot_state_count": len(snapshot_state["robot_states"]),
            "object_state_count": len(snapshot_state["object_states"]),
            "bridge_snapshot_scope": "internal snapshot also captures OSC controller fields, robot buffers, gripper action, task caches, RNG and MuJoCo integration state",
            "snapshot_state_diagnostics": snapshot_state,
        }
        controller_before = _bridge_request(connection, {"op": "repair_context"})
        restored = _bridge_request(
            connection, {"op": "restore", "snapshot_id": snapshot_id}
        )
        restore_comparison = _snapshot_comparison(
            snapshot_state, restored.get("snapshot_state")
        )
        report["initial_snapshot_restore"] = {
            "attempted": True,
            "restored_snapshot_id": restored.get("restored_snapshot_id"),
            "comparison": restore_comparison,
            "verified": (
                restored.get("restored_snapshot_id") == snapshot_id
                and restore_comparison.get("matched") is True
            ),
        }
        if report["initial_snapshot_restore"]["verified"] is not True:
            raise RuntimeError("full_Bridge_Snapshot_restore_preflight_failed")
        controller_after = _bridge_request(connection, {"op": "repair_context"})
        controller_restore_comparison = _snapshot_comparison(
            controller_before, controller_after
        )
        report["initial_snapshot_restore"]["controller_context_comparison"] = (
            controller_restore_comparison
        )
        if controller_restore_comparison.get("matched") is not True:
            raise RuntimeError("OSC_controller_context_changed_after_Snapshot_restore")

        from research.gs_opd.smoke.trajectory_bundle_scene_smoke import (
            _selected_bindings,
        )

        inventory = _bridge_request(connection, {"op": "geometry_inventory"})
        query_payload, binding_validation = _selected_bindings(
            teacher, semantic_results, inventory, subject_id, instruction
        )
        report["binding_validation"] = binding_validation
        if binding_validation.get("valid") is not True:
            raise ValueError("live_semantic_Body_Geom_binding_validation_failed")
        live_grasp = _bridge_request(
            connection,
            {
                "op": "grasp_state",
                "carried_binding": query_payload["carried_binding"],
                "expected_instruction": instruction,
            },
        )
        report["live_grasp_at_snapshot"] = live_grasp
        if (
            live_grasp.get("status") != "pass"
            or live_grasp.get("grasp_confirmed") is not True
            or live_grasp.get("state_id") != trigger_grasp_state.get("state_id")
        ):
            raise ValueError("Snapshot_state_no_longer_matches_Student_stable_grasp")

        scene_report = run_scene_smoke(
            connection,
            teacher,
            semantic_results,
            out_dir / "current_state_scene",
            subject_id=subject_id,
            expected_instruction=instruction,
        )
        report["scene_smoke"] = scene_report
        if (
            scene_report.get("status") != "complete"
            or scene_report.get("geometry_query_succeeded") is not True
        ):
            raise ValueError(
                "current_state_B_Spline_geometry_unavailable:"
                + str(scene_report.get("error") or scene_report.get("reason"))
            )
        query = scene_report.get("planning_geometry")
        query_after = scene_report.get("planning_geometry_after")
        if (
            not isinstance(query, dict)
            or query.get("state_id") != live_grasp.get("state_id")
            or not _diagnostic_equal(
                snapshot_state.get("mujoco_integration_state"),
                query.get("mujoco_integration_state"),
            ).get("matched")
            or not isinstance(query_after, dict)
            or query_after.get("state_id") != query.get("state_id")
        ):
            raise ValueError("geometry_B_Spline_query_does_not_match_captured_Snapshot")
        bundle_path = out_dir / "current_state_scene" / "trajectory_bundle.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        left_candidates = [
            item
            for item in bundle.get("candidates", [])
            if item.get("direction") == "left"
            and item.get("passes_geometric_screen") is True
        ]
        if not left_candidates:
            raise ValueError("current_state_has_no_geometrically_screened_left_B_Spline_route")
        selected = sorted(left_candidates, key=lambda item: item["candidate_id"])[0]
        report["selected_geometry_route"] = {
            "candidate_id": selected["candidate_id"],
            "direction": selected["direction"],
            "geometric_screened_left_candidate_count": len(left_candidates),
            "path_length_m": selected.get("length_m"),
            "points_world_m": selected["points_world_m"],
            "goal_world_m": selected["goal_world_m"],
            "control_points_world_m": selected.get("control_points_world_m"),
            "physical_safety_verified": False,
            "executable": False,
        }

        controller_context = controller_after
        gripper_inventory = _bridge_request(
            connection, {"op": "gripper_geometry_inventory"}
        )
        if (
            controller_context.get("status") != "available"
            or gripper_inventory.get("status") != "complete"
        ):
            raise ValueError("active_OSC_POSE_or_gripper_geometry_inventory_unavailable")
        robots = controller_context.get("robots") or []
        if (
            len(robots) != 1
            or robots[0].get("controller_name") != "OSC_POSE"
            or robots[0].get("action_dim") != 7
        ):
            raise ValueError("active_controller_is_not_native_seven_dimensional_Panda_OSC_POSE")
        report["active_controller"] = controller_context
        report["gripper_geometry_inventory"] = gripper_inventory
        model_meta_path = Path(model_metadata)
        metadata = json.loads(model_meta_path.read_text(encoding="utf-8"))
        report["model_fingerprint"] = {
            "bridge_geometry_sha256": query.get("model_geometry_sha256"),
            "mink_export_signature": metadata.get("model_signature"),
            "model_geometry_difference_is_diagnostic_only": True,
            "geometry_inventory_identity_is_still_validated": True,
        }

        carried_id = query_payload["carried_binding"]["object_id"]
        carried_geometry = query.get("objects", {}).get(carried_id)
        if not isinstance(carried_geometry, dict):
            raise ValueError("bound_carried_object_missing_from_current_planning_geometry")
        carried_body_id = int(carried_geometry["body_id"])
        object_state = _named_object_state(snapshot_state, carried_body_id)
        robot_states = snapshot_state.get("robot_states") or []
        if object_state is None or len(robot_states) != 1:
            raise ValueError("snapshot_lacks_unambiguous_carried_object_or_Panda_state")
        eef_position = _finite_array(robot_states[0].get("eef_pos"), (3,), "Snapshot EEF position")
        eef_quaternion = _quat_normalize(robot_states[0].get("eef_quat"))
        root_position = _finite_array(object_state.get("root_pos"), (3,), "carried Body root position")
        object_quaternion = _quat_normalize(object_state.get("root_quat"))
        sphere_center = _finite_array(carried_geometry.get("center_world_m"), (3,), "carried sphere center")
        rotation = quaternion_to_matrix(object_quaternion)
        grasp_offset = rotation.T @ (eef_position - sphere_center)
        grasp_relative_quaternion = quaternion_multiply(
            quaternion_conjugate(object_quaternion), eef_quaternion
        )
        sphere_local = body_local_offset_from_world_point(
            sphere_center, root_position, object_quaternion
        )
        route = {
            "candidate_id": selected["candidate_id"],
            "direction": selected["direction"],
            "reference_path": selected["points_world_m"],
            "geometric_candidate": selected,
            "sphere_center_body_local_offset_m": sphere_local.tolist(),
            "grasp_evidence": {
                "initial_eef_sphere_center_relative_position_body_local_m": grasp_offset.tolist(),
                "initial_eef_body_relative_quaternion_local_wxyz": grasp_relative_quaternion.tolist(),
            },
        }
        primary_obstacle_id = bundle["primary_obstacle_id"]
        live_scene = dict(bundle["scene"])
        live_scene["carried"] = {
            **live_scene["carried"],
            **carried_geometry,
        }
        live_scene["obstacles"] = [
            {
                **obstacle,
                **query["objects"][obstacle["object_id"]],
            }
            for obstacle in live_scene["obstacles"]
        ]
        planning_archive = {
            "active_osc_configuration": controller_context,
            "geometry_inventory": query["geometry_inventory"],
            "gripper_geometry_inventory": gripper_inventory,
            "scene": live_scene,
            "teacher_source": {"safety_ir_v2": teacher},
            "trajectory_bundle": {"primary_obstacle_id": primary_obstacle_id},
        }
        live_plan_input = out_dir / "live_mink_plan_input.json"
        live_plan_output = out_dir / "live_mink_plan.json"
        live_plan_input.write_text(
            json.dumps(
                {
                    "snapshot_state_id": query["state_id"],
                    "mujoco_integration_state": snapshot_state["mujoco_integration_state"],
                    "planning_archive": planning_archive,
                    "route": route,
                    "model_fingerprint_comparison": report["model_fingerprint"],
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        worker_command = [
            str(mink_python),
            str(REPO_ROOT / "research/gs_opd/smoke/mink_panda_physical_smoke.py"),
            "--live-plan-input",
            str(live_plan_input),
            "--live-plan-output",
            str(live_plan_output),
            "--model-binary",
            str(model_binary),
            "--model-metadata",
            str(model_meta_path),
            "--speed",
            "0.02",
            "--max-steps",
            "1200",
        ]
        worker_env = os.environ.copy()
        worker_env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT), worker_env.get("PYTHONPATH", "")]
        )
        worker = subprocess.run(
            worker_command,
            cwd=REPO_ROOT,
            env=worker_env,
            text=True,
            capture_output=True,
            check=False,
        )
        report["mink_worker"] = {
            "command": worker_command,
            "return_code": worker.returncode,
            "stdout_tail": worker.stdout[-3000:],
            "stderr_tail": worker.stderr[-3000:],
            "input": str(live_plan_input),
            "output": str(live_plan_output),
        }
        if worker.returncode != 0 or not live_plan_output.is_file():
            raise RuntimeError("current_state_Mink_planning_worker_failed")
        plan_report = json.loads(live_plan_output.read_text(encoding="utf-8"))
        report["mink_plan_report"] = plan_report
        plan = plan_report.get("plan")
        if plan_report.get("status") != "complete" or not isinstance(plan, dict):
            raise ValueError(
                "Mink_did_not_produce_a_complete_current_state_left_route:"
                + str(plan_report.get("failure_reason") or (plan or {}).get("failure_reason"))
            )
        joint_trajectory, spheres, eef_poses = _validated_mink_trace(plan)
        eef_positions, eef_quaternions = eef_poses[:, :3], eef_poses[:, 3:]
        recorded_joints = _finite_array(
            robot_states[0].get("joint_qpos"), (7,), "Snapshot Panda joints"
        )
        start_checks = {
            "joint_qpos_max_abs_error_rad": float(np.max(np.abs(joint_trajectory[0] - recorded_joints))),
            "eef_position_error_m": float(np.linalg.norm(eef_positions[0] - eef_position)),
            "eef_orientation_error_rad": quaternion_angle_rad(eef_quaternions[0], eef_quaternion),
            "sphere_center_error_m": float(np.linalg.norm(spheres[0] - sphere_center)),
        }
        start_checks["matched"] = (
            start_checks["joint_qpos_max_abs_error_rad"] <= STATE_ATOL
            and start_checks["eef_position_error_m"] <= 1e-6
            and start_checks["eef_orientation_error_rad"] <= 1e-6
            and start_checks["sphere_center_error_m"] <= 1e-6
        )
        report["mink_start_state_consistency"] = start_checks
        if start_checks["matched"] is not True:
            raise ValueError("Mink_FK_or_joint_start_does_not_match_current_Bridge_Snapshot")
        requested_steps = len(joint_trajectory) - 1
        if requested_steps < 1 or requested_steps > 1200:
            raise ValueError(f"Mink_reference_step_count_outside_Bridge_budget:{requested_steps}")
        live_candidate = {
            **selected,
            "candidate_id": "live-mink-left",
            "direction": "left",
            "points_world_m": spheres.tolist(),
            "goal_world_m": spheres[-1].tolist(),
            "goal_sphere_center_world_m": spheres[-1].tolist(),
            "eef_reference_poses_world": eef_poses.tolist(),
            "length_m": float(np.linalg.norm(np.diff(spheres, axis=0), axis=1).sum()),
            "passes_geometric_screen": True,
            "executable": False,
            "physical_safety_verified": False,
            "mink_joint_trajectory_rad": joint_trajectory.tolist(),
            "source_mink_plan": str(live_plan_output.resolve()),
        }
        branch_bundle = {
            **bundle,
            "candidate_count": 1,
            "screen_pass_count": 1,
            "candidates": [live_candidate],
            "physical_safety_verified": False,
            "executable": False,
        }
        tracking = track_trajectory_bundle(
            connection=connection,
            snapshot_id=snapshot_id,
            bundle=branch_bundle,
            carried_binding=carried_geometry,
            repair_context=controller_context,
            teacher=teacher,
            semantic_results=semantic_results,
            geometry_inventory=query["geometry_inventory"],
            gripper_inventory=gripper_inventory,
            config=TrackingConfig(
                max_control_steps_per_candidate=requested_steps,
                max_total_control_steps=requested_steps,
                reference_speed_m_s=REFERENCE_SPEED_M_S,
            ),
        )
        report["tracking_result"] = tracking
        branch = next(
            (
                item
                for item in tracking.get("candidates", [])
                if item.get("candidate_id") == "live-mink-left"
            ),
            None,
        )
        if not isinstance(branch, dict):
            raise RuntimeError("Bridge_returned_no_live_mink_left_branch_result")
        report["branch"] = branch
        if branch.get("eef_reference_pose_source") != "explicit_Mink_FK_world_pose_trajectory":
            raise RuntimeError("Bridge_did_not_execute_the_explicit_Mink_FK_EEF_pose_reference")
        if (
            tracking.get("main_state_restored") is not True
            or tracking.get("status") != "complete"
        ):
            raise RuntimeError("Snapshot_branch_did_not_restore_full_main_state")
        report["status"] = (
            "osc_physical_segment_passed"
            if branch.get("executable_safe") is True
            and branch.get("execution_complete") is True
            else "physical_branch_rejected"
        )
        if report["status"] != "osc_physical_segment_passed":
            report["failure_reasons"].extend(
                branch.get("rejection_reasons", [])
                or [branch.get("failure_reason") or "branch_failed_closed"]
            )
        steps = branch.get("steps", [])
        if steps:
            actual_eef = np.asarray(
                [item.get("actual_eef_position_world_m") for item in steps],
                dtype=np.float64,
            )
            actual_spheres = np.asarray(
                [item.get("actual_object_sphere_center_world_m") for item in steps],
                dtype=np.float64,
            )
            _write_svg(
                output_path.with_suffix(".svg"), spheres, actual_eef, actual_spheres
            )
            report["visualization"] = str(output_path.with_suffix(".svg").resolve())
    except Exception as exc:
        report["status"] = "blocked"
        report["failure_reasons"].append(f"{type(exc).__name__}:{exc}")
    finally:
        if snapshot_id is not None and isinstance(snapshot_state, dict):
            report["final_snapshot_restore"]["attempted"] = True
            try:
                restored = _bridge_request(
                    connection, {"op": "restore", "snapshot_id": snapshot_id}
                )
                comparison = _snapshot_comparison(
                    snapshot_state, restored.get("snapshot_state")
                )
                report["final_snapshot_restore"].update(
                    {
                        "restored_snapshot_id": restored.get("restored_snapshot_id"),
                        "comparison": comparison,
                        "verified": restored.get("restored_snapshot_id") == snapshot_id
                        and comparison.get("matched") is True,
                    }
                )
                if report["final_snapshot_restore"]["verified"]:
                    binding = (report.get("scene_smoke") or {}).get("query_payload", {}).get("carried_binding")
                    if binding:
                        final_grasp = _bridge_request(
                            connection,
                            {
                                "op": "grasp_state",
                                "carried_binding": binding,
                                "expected_instruction": instruction,
                            },
                        )
                        report["final_snapshot_restore"]["grasp_state"] = final_grasp
                        report["final_snapshot_restore"]["verified"] = (
                            final_grasp.get("status") == "pass"
                            and final_grasp.get("grasp_confirmed") is True
                        )
            except Exception as exc:
                report["final_snapshot_restore"].update(
                    {"verified": False, "error": f"{type(exc).__name__}:{exc}"}
                )
            if report["final_snapshot_restore"].get("verified") is not True:
                report["status"] = "snapshot_restore_failed"
                report["failure_reasons"].append("final_full_Snapshot_restore_or_grasp_verification_failed")
    report["snapshot_restore_verified"] = (
        report["final_snapshot_restore"].get("verified") is True
    )
    report["main_environment_candidate_actions"] = 0
    report["training_label_created"] = False
    report["final_task_success_confirmed"] = False
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _export_current_model(
    args: argparse.Namespace, model_path: Path, metadata_path: Path
) -> None:
    command = [
        str(args.libero_python),
        str(REPO_ROOT / "research/gs_opd/smoke/mink_panda_model.py"),
        "--_export-real-model",
        str(model_path),
        "--_export-metadata",
        str(metadata_path),
        "--suite",
        "obstacle_avoidance",
        "--level",
        "0",
        "--task-id",
        "0",
    ]
    env = os.environ.copy()
    env["LIBERO_CONFIG_PATH"] = "/workspace/zy_workspace/.libero_safety_config"
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["PYTHONPATH"] = DEFAULT_LIBERO_PYTHONPATH
    completed = subprocess.run(
        command, env=env, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            "LIBERO environment failed to export current model: "
            + completed.stderr[-3000:]
            + completed.stdout[-1000:]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-json", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--semantic-results-json", type=Path, default=DEFAULT_SEMANTICS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/opt/data/private/gs_opd/rollouts/live_left_osc_smoke"),
    )
    parser.add_argument("--mink-python", type=Path, default=DEFAULT_MINK_PYTHON)
    parser.add_argument("--libero-python", type=Path, default=DEFAULT_LIBERO_PYTHON)
    parser.add_argument("--rollout-python", type=Path, default=DEFAULT_ROLLOUT_PYTHON)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--direct-bspline-osc-left",
        action="store_true",
        help="Execute the existing left B-Spline route directly through native OSC_POSE; do not export or launch Mink.",
    )
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=38567)
    args = parser.parse_args()
    if args.max_steps is None:
        args.max_steps = 1600 if args.direct_bspline_osc_left else 240
    for required in (args.teacher_json, args.semantic_results_json):
        if not required.is_file():
            parser.error(f"required input file does not exist: {required}")
    if (
        (not args.direct_bspline_osc_left and not args.mink_python.is_file())
        or not args.libero_python.is_file()
        or not args.rollout_python.is_file()
    ):
        parser.error(
            "the required RLinf and LIBERO-Safety Python environments must be available"
        )
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Check that the already-running persistent Bridge is reachable before
    # spending time exporting the model or loading the VLA.
    from multiprocessing.connection import Client

    try:
        conn = Client((args.bridge_host, args.bridge_port), authkey=b"gs-opd-local")
        conn.send({"op": "geometry_inventory"})
        inventory = conn.recv()
        conn.send({"op": "trajectory_tracking_capabilities"})
        tracker_capabilities = conn.recv()
        conn.close()
    except Exception as exc:
        parser.error(
            f"persistent Safety Bridge is unavailable: {type(exc).__name__}: {exc}"
        )
    if not isinstance(inventory, dict) or inventory.get("status") != "complete":
        parser.error(
            "persistent Safety Bridge returned an incomplete Geometry Inventory"
        )
    if not (
        isinstance(tracker_capabilities, dict)
        and tracker_capabilities.get("status") == "available"
        and tracker_capabilities.get("explicit_eef_reference_poses_world") is True
        and tracker_capabilities.get("substep_contact_monitoring") is True
        and tracker_capabilities.get("full_snapshot_restore_verification") is True
    ):
        parser.error(
            "persistent Safety Bridge lacks the current Mink EEF-pose tracking protocol; restart it from this working tree"
        )
    model_path = args.out_dir / "current_libero_safety_model.mjb"
    metadata_path = args.out_dir / "current_libero_safety_model.json"
    report_path = args.out_dir / (
        "live_left_bspline_osc_report.json"
        if args.direct_bspline_osc_left
        else "live_left_osc_snapshot_report.json"
    )
    if report_path.exists():
        parser.error(
            f"Choose a fresh --out-dir; report already exists: {report_path}"
        )
    if not args.direct_bspline_osc_left:
        _export_current_model(args, model_path, metadata_path)
    video_path = args.out_dir / (
        "student_and_direct_osc.mp4"
        if args.direct_bspline_osc_left
        else "student_grasp_search.mp4"
    )
    command = [
        str(args.rollout_python),
        str(REPO_ROOT / "research/gs_opd/smoke/pi05_rollout.py"),
        "--max-steps",
        str(args.max_steps),
        "--trajectory-bundle-physical-validation",
        "--teacher-json",
        str(args.teacher_json.resolve()),
        "--semantic-results-json",
        str(args.semantic_results_json.resolve()),
        "--trajectory-validation-out",
        str(report_path.resolve()),
        "--result-out",
        str((args.out_dir / "result.json").resolve()),
        "--video-out",
        str(video_path.resolve()),
    ]
    if args.direct_bspline_osc_left:
        command.extend([
            "--direct-bspline-osc-left",
            "--trace-out",
            str((args.out_dir / "direct_bspline_osc_trajectory.npz").resolve()),
        ])
    else:
        command.extend([
            "--mink-osc-left",
            "--mink-python",
            str(args.mink_python),
            "--mink-model-binary",
            str(model_path.resolve()),
            "--mink-model-metadata",
            str(metadata_path.resolve()),
        ])
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(REPO_ROOT),
            "/workspace/zy_workspace/LIBERO-Safety",
            env.get("PYTHONPATH", ""),
        )
    )
    launch_record = {
        "command": command,
        "environment": {"PYTHONPATH": env["PYTHONPATH"]},
        "libero_export_python": str(args.libero_python.resolve()),
        "persistent_bridge": {
            "host": args.bridge_host,
            "port": args.bridge_port,
            "geometry_count": inventory.get("geom_count"),
            "trajectory_tracking_capabilities": tracker_capabilities,
        },
        "execution_mode": "direct_B_Spline_to_native_OSC_POSE" if args.direct_bspline_osc_left else "Mink_snapshot_native_OSC_POSE",
        "exported_model": None if args.direct_bspline_osc_left else str(model_path.resolve()),
        "model_metadata": None if args.direct_bspline_osc_left else str(metadata_path.resolve()),
        "output_report": str(report_path.resolve()),
        "video_path": str(video_path.resolve()),
        "main_environment_candidate_actions": 0,
        "training_label_created": False,
    }
    (args.out_dir / "launch.json").write_text(
        json.dumps(launch_record, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Starting Student rollout; direct B-Spline/OSC execution begins after three stable dual-finger grasp states." if args.direct_bspline_osc_left else "Starting Student rollout; current-state B-Spline/Mink/OSC branch runs only after three stable dual-finger grasp states.",
        flush=True,
    )
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode:
        print(
            f"Student rollout process exited with status {completed.returncode}",
            file=sys.stderr,
        )
        return int(completed.returncode)
    if not report_path.is_file():
        print(
            "Student rollout ended without reaching the stable-grasp branch trigger.",
            file=sys.stderr,
        )
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
            "status": report.get("status"),
            "decision_step": report.get("decision_step"),
                "executed_steps": (
                    report.get("real_execution_steps")
                    if args.direct_bspline_osc_left
                    else (report.get("branch") or {}).get("executed_steps")
                ),
                "failure_reasons": report.get("failure_reasons"),
                "compiler_verdict": (
                (report.get("branch") or {}).get("contact_compiler") or {}
                ).get("verdict"),
                "grasp_status": (
                (report.get("branch") or {}).get("grasp_evidence") or {}
                ).get("status"),
            "restore_verified": (report.get("final_snapshot_restore") or {}).get(
                    "verified"
                ),
                "report": str(report_path.resolve()),
                "visualization": report.get("visualization"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
