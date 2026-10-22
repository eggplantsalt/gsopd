"""Snapshot-only MuJoCo exploration of inconclusive L0T1 B-Spline routes.

This runner restores attempt6's physical grasp snapshot, regenerates the
same-state geometry bundle without stepping the environment, verifies it
against the archived 18 candidates, and sends at most one inconclusive route
from each Left/Right/Above direction to isolated OSC Snapshot branches.
Exploratory execution never promotes a geometrically inconclusive route to a
safe trajectory.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/workspace/zy_workspace/RLinf-gs-opd-overfit-recovery")
LIBERO = Path("/workspace/zy_workspace/LIBERO-Safety")
ROLLOUTS = Path("/opt/data/private/gs_opd/rollouts")
OUT = ROLLOUTS / "obstacle_avoidance_L0T1_snapshot_exploratory_bspline_physics_20261022_attempt2"
BRIDGE = REPO / "research/gs_opd/smoke/safety_bridge.py"
SCENE_SMOKE = REPO / "research/gs_opd/smoke/trajectory_bundle_scene_smoke.py"
TRACKING = REPO / "research/gs_opd/repair/trajectory_tracking.py"
SCENE_HELPER = REPO / "research/gs_opd/repair/trajectory_bundle.py"
SNAPSHOT = ROLLOUTS / "obstacle_avoidance_L0T1_osc_assisted_book_grasp_20261022_attempt6/recovered_grasp_runtime_snapshot.pkl"
ATTEMPT6_REPORT = ROLLOUTS / "obstacle_avoidance_L0T1_osc_assisted_book_grasp_20261022_attempt6/experiment_report.json"
EXISTING_BUNDLE_DIR = ROLLOUTS / "obstacle_avoidance_L0T1_snapshot_restored_trajectory_bundle_20261022_attempt1"
EXISTING_BUNDLE = EXISTING_BUNDLE_DIR / "trajectory_bundle.json"
GOAL_QUERY = EXISTING_BUNDLE_DIR / "current_goal_predicates.json"
INPUT_DIR = ROLLOUTS / "obstacle_avoidance_L0T1_reviewed_semantic_compiler_input_20261022_attempt1"
TEACHER = INPUT_DIR / "source_evidence/teacher_ir_v2.validated.json"
SEMANTIC = INPUT_DIR / "compiler_inputs/semantic_results_by_object.json"
COMPATIBILITY = INPUT_DIR / "compiler_inputs/geometry_inventory_compiler_compatible.json"
INSTRUCTION = "pick up the book and place it in the back compartment of the caddy"
ADDRESS = ("127.0.0.1", 38567)
AUTH = b"gs-opd-local"
SELECTED_DIRECTIONS = ("left", "right", "above")
MAX_STEPS_PER_CANDIDATE = 1200
VIDEO_STRIDE = 5
BOOK_BODY_ID = 23
TARGET_CADDY_BODY_ID = 26


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def bridge_request(connection: Any, request: dict[str, Any]) -> dict[str, Any]:
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError(f"Bridge returned a non-object response for {request.get('op')}")
    if response.get("error"):
        raise RuntimeError(str(response["error"]))
    return response


def _state_vector(snapshot_state: dict[str, Any]) -> np.ndarray:
    vector = np.asarray(
        snapshot_state.get("mujoco_integration_state"), dtype=np.float64
    )
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("snapshot_integration_state_is_missing_or_invalid")
    return vector


def _body_state(state: dict[str, Any], body_id: int) -> dict[str, Any] | None:
    matches = [
        item
        for item in state.get("object_states", {}).values()
        if isinstance(item, dict) and item.get("body_id") == body_id
    ]
    return matches[0] if len(matches) == 1 else None


def _compare_candidate_bundles(existing: dict[str, Any], regenerated: dict[str, Any]) -> dict[str, Any]:
    existing_items = existing.get("candidates")
    regenerated_items = regenerated.get("candidates")
    if not isinstance(existing_items, list) or not isinstance(regenerated_items, list):
        raise ValueError("candidate_bundle_candidate_lists_are_missing")
    existing_by_id = {item.get("candidate_id"): item for item in existing_items}
    regenerated_by_id = {item.get("candidate_id"): item for item in regenerated_items}
    if set(existing_by_id) != set(regenerated_by_id) or len(existing_by_id) != 18:
        raise ValueError("regenerated_candidate_identity_differs_from_archived_18")
    comparisons = []
    for candidate_id in sorted(existing_by_id):
        before, after = existing_by_id[candidate_id], regenerated_by_id[candidate_id]
        checks = {
            "direction_equal": before.get("direction") == after.get("direction"),
            "path_points_exactly_equal": np.array_equal(
                np.asarray(before.get("points_world_m"), dtype=np.float64),
                np.asarray(after.get("points_world_m"), dtype=np.float64),
            ),
            "control_points_exactly_equal": np.array_equal(
                np.asarray(before.get("control_points_world_m"), dtype=np.float64),
                np.asarray(after.get("control_points_world_m"), dtype=np.float64),
            ),
            "screen_result_equal": before.get("passes_geometric_screen")
            == after.get("passes_geometric_screen"),
            "rejection_reasons_equal": before.get("rejection_reasons")
            == after.get("rejection_reasons"),
        }
        comparisons.append({"candidate_id": candidate_id, **checks})
    matched = all(
        all(value for key, value in item.items() if key != "candidate_id")
        for item in comparisons
    )
    return {"matched": matched, "candidate_count": len(comparisons), "candidates": comparisons}


def _select_exploration_candidates(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    selected = {}
    for candidate in bundle.get("candidates", []):
        direction = candidate.get("direction")
        if direction not in SELECTED_DIRECTIONS or direction in selected:
            continue
        reasons = candidate.get("rejection_reasons")
        if (
            candidate.get("passes_geometric_screen") is not False
            or not isinstance(reasons, list)
            or not reasons
            or not all(
                isinstance(reason, str)
                and reason.startswith("curve_clearance_bound_inconclusive:")
                for reason in reasons
            )
        ):
            continue
        selected[direction] = candidate
    if tuple(selected) != SELECTED_DIRECTIONS:
        raise ValueError(
            "one_inconclusive_candidate_per_left_right_above_direction_unavailable"
        )
    return [selected[direction] for direction in SELECTED_DIRECTIONS]


def _robot_scene_collisions(
    candidate: dict[str, Any], robot_body_ids: set[int]
) -> list[dict[str, Any]]:
    monitor = candidate.get("contact_monitoring", {})
    events = monitor.get("events", []) if isinstance(monitor, dict) else []
    collisions = []
    for event in events:
        if not isinstance(event, dict):
            continue
        body_pair = event.get("body_pair", [])
        body_ids = {
            body.get("id")
            for body in body_pair
            if isinstance(body, dict) and isinstance(body.get("id"), int)
        }
        non_robot_ids = body_ids - robot_body_ids
        if (
            event.get("robot_involved") is True
            and any(body_id not in (BOOK_BODY_ID, TARGET_CADDY_BODY_ID) for body_id in non_robot_ids)
        ) or (
            BOOK_BODY_ID in body_ids
            and any(
                body_id not in (BOOK_BODY_ID, TARGET_CADDY_BODY_ID)
                for body_id in non_robot_ids
            )
        ):
            collisions.append(event)
    return collisions


def _write_candidate_artifacts(
    candidate: dict[str, Any], output_dir: Path, robot_body_ids: set[int]
) -> dict[str, Any]:
    import cv2

    candidate_id = candidate["candidate_id"]
    frames = candidate.pop("video_frames", [])
    video_name = f"{candidate_id}.mp4"
    video_path = output_dir / "videos" / video_name
    if frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frames[0]["height"], frames[0]["width"]
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"video_writer_open_failed:{video_path}")
        try:
            for frame in frames:
                rgb = np.frombuffer(frame["rgb_bytes"], dtype=np.uint8).reshape(
                    frame["height"], frame["width"], 3
                )
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    else:
        video_path = None

    candidate_dir = output_dir / "candidates" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    step_records = []
    integration_states = []
    book_positions = []
    book_quaternions = []
    sphere_centers = []
    eef_positions = []
    eef_quaternions = []
    actions = []
    references = []
    tracking_errors = []
    for step in candidate.get("steps", []):
        state = step.get("state", {})
        book_state = _body_state(state, BOOK_BODY_ID)
        if not isinstance(book_state, dict):
            book_state = {
                "body_id": BOOK_BODY_ID,
                "root_pos": step.get("actual_object_root_pos_world_m"),
                "root_quat": step.get("actual_object_root_quat_world_wxyz"),
            }
        integration = state.get("mujoco_integration_state")
        if isinstance(integration, list):
            integration_states.append(integration)
        book_positions.append(book_state.get("root_pos"))
        book_quaternions.append(book_state.get("root_quat"))
        sphere_centers.append(step.get("actual_object_sphere_center_world_m"))
        eef_positions.append(step.get("actual_eef_position_world_m"))
        eef_quaternions.append(step.get("actual_eef_quaternion_world_wxyz"))
        actions.append(step.get("action"))
        references.append(step.get("reference_sphere_center_world_m"))
        tracking_errors.append(
            [
                step.get("sphere_center_tracking_error_m"),
                step.get("eef_tracking_error_m"),
                step.get("grasp_relative_position_error_m"),
                step.get("grasp_relative_rotation_error_rad"),
            ]
        )
        step_records.append(
            {
                **{key: value for key, value in step.items() if key != "state"},
                "book_state": book_state,
                "state_integration_vector_saved_in_npz": isinstance(integration, list),
            }
        )
    with (candidate_dir / "action_book_contact_trace.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in step_records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(candidate_dir / "substep_contact_monitoring.json", candidate.get("contact_monitoring", {}))

    def matrix(values: list[Any], columns: int) -> np.ndarray:
        if not values:
            return np.empty((0, columns), dtype=np.float64)
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (len(values), columns) or not np.isfinite(result).all():
            raise ValueError(f"candidate_trace_matrix_invalid:{candidate_id}")
        return result

    arrays: dict[str, Any] = {
        "step_index": np.asarray(
            [step.get("step_index", index) for index, step in enumerate(candidate.get("steps", []))],
            dtype=np.int64,
        ),
        "actions_7d": matrix(actions, 7),
        "book_root_position_world_m": matrix(book_positions, 3),
        "book_root_quaternion_world_wxyz": matrix(book_quaternions, 4),
        "book_sphere_center_world_m": matrix(sphere_centers, 3),
        "eef_position_world_m": matrix(eef_positions, 3),
        "eef_quaternion_world_wxyz": matrix(eef_quaternions, 4),
        "reference_sphere_center_world_m": matrix(references, 3),
        "tracking_and_grasp_errors": matrix(tracking_errors, 4),
    }
    if integration_states:
        integration_array = np.asarray(integration_states, dtype=np.float64)
        if integration_array.ndim != 2 or not np.isfinite(integration_array).all():
            raise ValueError(f"candidate_integration_trace_invalid:{candidate_id}")
        arrays["mujoco_integration_state"] = integration_array
    trace_path = candidate_dir / "physical_trace.npz"
    np.savez_compressed(trace_path, **arrays)

    start_book = _body_state(candidate.get("start_state", {}), BOOK_BODY_ID)
    end_book = _body_state(
        candidate.get("steps", [])[-1].get("state", {}) if candidate.get("steps") else {},
        BOOK_BODY_ID,
    )
    start_position = np.asarray(
        (start_book or {}).get("root_pos", candidate.get("start_object_root_pos_world_m")),
        dtype=np.float64,
    )
    end_position = np.asarray(
        (end_book or {}).get("root_pos", candidate.get("start_object_root_pos_world_m")),
        dtype=np.float64,
    )
    displacement = (
        end_position - start_position
        if start_position.shape == end_position.shape == (3,)
        else None
    )
    contact_events = candidate.get("contact_monitoring", {}).get("events", [])
    non_target_collisions = _robot_scene_collisions(candidate, robot_body_ids)
    endpoint = candidate.get("task_feasibility", {}).get(
        "endpoint_within_tracking_tolerance"
    ) is True
    reached_endpoint = bool(candidate.get("execution_complete") is True and endpoint)
    failure_reason = candidate.get("failure_reason") or ""
    if candidate.get("non_target_collision_detected") or non_target_collisions:
        disposition = "rejected_actual_non_target_collision"
    elif candidate.get("unresolved_target_contact_detected") is True:
        disposition = "rejected_unresolved_target_contact"
    elif any(
        token in failure_reason
        for token in (
            "Panda_opposing_finger_grasp_contact_lost",
            "initial_Panda_gripper",
            "gripper_object_relative_position_changed",
            "gripper_object_relative_rotation_changed",
        )
    ):
        disposition = "rejected_grasp_loss_or_drift"
    elif any(token in failure_reason for token in ("joint_limit", "workspace")):
        disposition = "rejected_control_or_workspace_limit"
    elif reached_endpoint:
        disposition = "exploratory_endpoint_reached_geometry_still_inconclusive"
    else:
        disposition = "incomplete_or_rejected_without_endpoint"
    write_json(
        candidate_dir / "candidate_result.json",
        {
            key: value
            for key, value in candidate.items()
            if key not in ("steps", "contact_monitoring")
        },
    )
    return {
        "candidate_id": candidate_id,
        "direction": candidate.get("direction"),
        "geometric_screen_status": candidate.get("geometric_screen_status"),
        "geometric_rejection_reasons": candidate.get("geometric_rejection_reasons"),
        "physical_disposition": disposition,
        "requested_steps": candidate.get("requested_steps"),
        "executed_steps": candidate.get("executed_steps", 0),
        "execution_complete": candidate.get("execution_complete") is True,
        "failure_reason": candidate.get("failure_reason"),
        "reached_bypass_endpoint": reached_endpoint,
        "endpoint_sphere_center_tracking_error_m": candidate.get("task_feasibility", {}).get(
            "endpoint_sphere_center_tracking_error_m"
        ),
        "book_root_start_world_m": start_position.tolist() if start_position.shape == (3,) else None,
        "book_root_end_world_m": end_position.tolist() if end_position.shape == (3,) else None,
        "book_root_displacement_world_m": displacement.tolist() if displacement is not None else None,
        "book_root_displacement_norm_m": float(np.linalg.norm(displacement)) if displacement is not None else None,
        "max_eef_tracking_error_m": candidate.get("closed_loop_tracking", {}).get(
            "maximum_eef_tracking_error_m"
        ),
        "max_grasp_relative_rotation_error_rad": candidate.get("grasp_evidence", {}).get(
            "maximum_relative_rotation_error_rad"
        ),
        "grasp_status": candidate.get("grasp_evidence", {}).get("status"),
        "contact_monitoring_status": candidate.get("contact_monitoring", {}).get("status"),
        "expected_physics_substeps": candidate.get("contact_monitoring", {}).get(
            "expected_physics_substeps_total"
        ),
        "captured_physics_substeps": candidate.get("contact_monitoring", {}).get(
            "captured_physics_substeps"
        ),
        "unobserved_expected_substeps": candidate.get("contact_monitoring", {}).get(
            "unobserved_expected_substeps"
        ),
        "contact_event_count": len(contact_events) if isinstance(contact_events, list) else None,
        "contact_compiler_verdict": candidate.get("contact_compiler", {}).get("verdict"),
        "non_target_collision_detected": bool(
            candidate.get("non_target_collision_detected") or non_target_collisions
        ),
        "non_target_collision_events": non_target_collisions
        or candidate.get("non_target_collision_events", []),
        "training_label_created": candidate.get("training_label_created") is True,
        "video": video_name if video_path is not None else None,
        "video_frame_count": len(frames),
        "trace_jsonl": str((candidate_dir / "action_book_contact_trace.jsonl").relative_to(output_dir)),
        "trace_npz": str(trace_path.relative_to(output_dir)),
        "contact_evidence": str((candidate_dir / "substep_contact_monitoring.json").relative_to(output_dir)),
        "unresolved_target_contact_detected": candidate.get(
            "unresolved_target_contact_detected"
        ) is True,
        "unresolved_target_contact_events": candidate.get(
            "unresolved_target_contact_events", []
        ),
    }


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"Independent output directory already exists: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "inputs").mkdir()
    (OUT / "source").mkdir()
    for path in (
        SNAPSHOT,
        ATTEMPT6_REPORT,
        EXISTING_BUNDLE,
        GOAL_QUERY,
        TEACHER,
        SEMANTIC,
        COMPATIBILITY,
    ):
        shutil.copy2(path, OUT / "inputs" / path.name)
    for path in (BRIDGE, SCENE_SMOKE, SCENE_HELPER, TRACKING, Path(__file__)):
        shutil.copy2(path, OUT / "source" / path.name)

    report: dict[str, Any] = {
        "schema_version": "gs_opd_l0t1_snapshot_exploratory_bspline_physics.v1",
        "status": "incomplete",
        "scope": {
            "benchmark": "obstacle_avoidance",
            "level": 0,
            "task_id": 1,
            "init_state_index": 0,
            "environment_seed": 0,
            "stage": "snapshot_local_inconclusive_candidate_physics_exploration_only",
        },
        "start_source": str(SNAPSHOT),
        "candidate_bundle_source": str(EXISTING_BUNDLE),
        "candidate_count_reused": None,
        "exploratory_candidate_ids": [],
        "tracking_config": {
            "max_control_steps_per_candidate": MAX_STEPS_PER_CANDIDATE,
            "max_total_control_steps": 3 * MAX_STEPS_PER_CANDIDATE,
            "maximum_eef_tracking_error_m": 0.04,
            "maximum_grasp_relative_rotation_error_rad": 0.25,
            "maximum_grasp_relative_position_error_m": 0.012,
            "maximum_sphere_center_tracking_error_m": 0.025,
            "reference_speed_m_s": 0.04,
            "video_capture_stride_control_steps": VIDEO_STRIDE,
        },
        "trajectory_bundle_algorithm_modified": False,
        "safety_compiler_modified": False,
        "default_control_configuration_modified": False,
        "training_executed": False,
        "vla_actions_executed": False,
        "main_environment_transport_or_placement_executed": False,
        "physical_safety_verified": False,
        "safe_success_declared": False,
        "final_task_success_confirmed": False,
        "candidates": [],
        "failure_reasons": [],
    }
    bridge_log = (OUT / "snapshot_physics_bridge.log").open("ab", buffering=0)
    connection = None
    process = None
    try:
        sys.path.insert(0, str(REPO))
        from research.gs_opd.repair.trajectory_bundle import BundleConfig
        from research.gs_opd.repair.trajectory_tracking import (
            TrackingConfig,
            track_trajectory_bundle,
        )
        from research.gs_opd.smoke.trajectory_bundle_scene_smoke import (
            _geometry_inventory_validation_copy,
            _query_with_inventory_validation_copy,
            run_scene_smoke,
        )
        from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2

        attempt6_report = json.loads(ATTEMPT6_REPORT.read_text(encoding="utf-8"))
        expected_snapshot = attempt6_report["recovered_grasp_snapshot"]
        snapshot_bytes = SNAPSHOT.read_bytes()
        snapshot_hash = sha256(SNAPSHOT)
        if snapshot_hash != expected_snapshot.get("sha256"):
            raise ValueError("attempt6_recovered_snapshot_hash_mismatch")
        report["snapshot_source_verification"] = {
            "sha256": snapshot_hash,
            "expected_sha256": expected_snapshot.get("sha256"),
            "matched": True,
        }
        write_json(OUT / "snapshot_restore_result.json", {"status": "pending"})

        with socket.socket() as sock:
            if sock.connect_ex(ADDRESS) == 0:
                raise RuntimeError("safety_bridge_port_already_in_use")
        environment = os.environ.copy()
        environment["MUJOCO_GL"] = "egl"
        environment["PYOPENGL_PLATFORM"] = "egl"
        environment["LIBERO_CONFIG_PATH"] = "/workspace/zy_workspace/.libero_safety_config"
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(LIBERO), str(LIBERO / "third_party/robosuite-1.4"), str(REPO)]
        )
        command = [
            str(LIBERO / ".venv/bin/python"),
            str(BRIDGE),
            "--benchmark",
            "obstacle_avoidance",
            "--level",
            "0",
            "--task-id",
            "1",
            "--init-state",
            "0",
            "--seed",
            "0",
            "--camera-resolution",
            "256",
        ]
        report["bridge_command"] = command
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=bridge_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"safety_bridge_exited_{process.returncode}")
            try:
                connection = Client(ADDRESS, authkey=AUTH)
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)
        if connection is None:
            raise TimeoutError("safety_bridge_startup_timeout")

        restored = bridge_request(
            connection,
            {"op": "restore_runtime_snapshot", "payload": snapshot_bytes},
        )
        restored_vector = _state_vector(restored.get("snapshot_state", {}))
        expected_vector = _state_vector(expected_snapshot.get("snapshot_state", {}))
        restore_exact = bool(
            restored_vector.shape == expected_vector.shape
            and np.array_equal(restored_vector, expected_vector)
        )
        restore_result = {
            "status": "matched" if restored.get("status") == "restored" and restore_exact else "mismatch",
            "source_snapshot_sha256": snapshot_hash,
            "model_geometry_sha256": restored.get("model_geometry_sha256"),
            "integration_state_length": int(restored_vector.size),
            "integration_state_exact_match": restore_exact,
            "maximum_absolute_integration_state_error": float(
                np.max(np.abs(restored_vector - expected_vector))
            )
            if restored_vector.shape == expected_vector.shape
            else None,
        }
        write_json(OUT / "snapshot_restore_result.json", restore_result)
        report["snapshot_restore"] = restore_result
        if restore_result["status"] != "matched":
            raise RuntimeError("attempt6_runtime_snapshot_restore_did_not_match_exactly")

        snapshot = bridge_request(connection, {"op": "snapshot"})
        snapshot_id = snapshot.get("snapshot_id")
        snapshot_state = snapshot.get("snapshot_state")
        snapshot_vector = _state_vector(snapshot_state or {})
        if (
            not isinstance(snapshot_id, str)
            or not snapshot_id
            or not np.array_equal(snapshot_vector, expected_vector)
        ):
            raise RuntimeError("restored_grasp_state_snapshot_capture_mismatch")
        report["branch_snapshot"] = {
            "snapshot_id": snapshot_id,
            "integration_state_exact_match_to_attempt6": True,
        }

        teacher = parse_teacher_output_v2(TEACHER.read_text(encoding="utf-8"))
        semantic_results = json.loads(SEMANTIC.read_text(encoding="utf-8"))
        compatibility_inventory = json.loads(COMPATIBILITY.read_text(encoding="utf-8"))
        scene_report = run_scene_smoke(
            connection,
            teacher,
            semantic_results,
            OUT,
            subject_id="object_1",
            expected_instruction=INSTRUCTION,
            geometry_inventory_compatibility=compatibility_inventory,
        )
        write_json(OUT / "scene_adaptation_report.json", scene_report)
        if (
            scene_report.get("status") != "complete"
            or scene_report.get("geometry_query_succeeded") is not True
            or scene_report.get("state_consistency", {}).get("matched") is not True
            or scene_report.get("automatic_environment_actions_executed") != 0
        ):
            raise RuntimeError(
                f"L0T1_In_scene_adaptation_preflight_failed:{scene_report.get('error', scene_report.get('reason'))}"
            )
        planning = scene_report["planning_geometry"]
        native_goals = planning.get("goal", {}).get("native_goal_predicates")
        if native_goals != [["in", "black_book_1", "desk_caddy_1_back_contain_region"]]:
            raise RuntimeError("active_bridge_native_In_goal_changed_or_unexpected")
        query_copy, query_placeholders = _query_with_inventory_validation_copy(
            planning, compatibility_inventory
        )
        inventory_raw = bridge_request(connection, {"op": "geometry_inventory"})
        inventory_copy, inventory_placeholders = _geometry_inventory_validation_copy(
            inventory_raw, compatibility_inventory
        )
        if query_placeholders != inventory_placeholders:
            raise RuntimeError("reviewed_geom_name_placeholder_map_changed_between_queries")
        if not np.array_equal(_state_vector(query_copy), expected_vector):
            raise RuntimeError("scene_geometry_query_does_not_match_attempt6_snapshot")

        regenerated_bundle = json.loads((OUT / "trajectory_bundle.json").read_text(encoding="utf-8"))
        existing_bundle = json.loads(EXISTING_BUNDLE.read_text(encoding="utf-8"))
        if regenerated_bundle.get("screen_pass_count") != 0:
            raise RuntimeError("regenerated_bundle_unexpectedly_claims_geometric_safe_candidates")
        bundle_match = _compare_candidate_bundles(existing_bundle, regenerated_bundle)
        write_json(OUT / "existing_candidate_bundle_equivalence.json", bundle_match)
        if bundle_match.get("matched") is not True:
            raise RuntimeError("current_state_candidate_geometry_does_not_match_archived_18")
        if existing_bundle.get("scene", {}).get("state_id") != planning.get("state_id"):
            raise RuntimeError("archived_candidate_bundle_state_id_does_not_match_restored_snapshot")
        tracking_bundle = copy.deepcopy(existing_bundle)
        selected = _select_exploration_candidates(tracking_bundle)
        tracking_bundle["candidates"] = selected
        selected_ids = [candidate["candidate_id"] for candidate in selected]
        if len(set(candidate["direction"] for candidate in selected)) != 3:
            raise RuntimeError("selected_exploration_directions_are_not_distinct")
        report["candidate_count_reused"] = len(existing_bundle["candidates"])
        report["archived_18_candidate_bundle_hash"] = sha256(EXISTING_BUNDLE)
        report["candidate_bundle_equivalence"] = bundle_match
        report["all_18_geometric_screen_results"] = {
            "screen_pass_count": existing_bundle.get("screen_pass_count"),
            "screen_rejected_count": existing_bundle.get("screen_rejected_count"),
            "rejection_reason_counts": existing_bundle.get("rejection_reason_counts"),
            "candidate_count_by_direction": dict(
                sorted(Counter(item.get("direction") for item in existing_bundle["candidates"]).items())
            ),
        }
        report["exploratory_candidate_ids"] = selected_ids
        report["exploratory_candidates"] = [
            {
                "candidate_id": candidate["candidate_id"],
                "direction": candidate["direction"],
                "path_length_m": candidate["length_m"],
                "passes_geometric_screen": candidate["passes_geometric_screen"],
                "rejection_reasons": candidate["rejection_reasons"],
            }
            for candidate in selected
        ]
        goal_reference = existing_bundle["scene"]["goal"]["construction"][
            "placement_target_direction_reference"
        ]
        report["placement_reference_semantics"] = {
            "native_goal_predicates": native_goals,
            "target_body_id": goal_reference["body_id"],
            "target_body_name": goal_reference["body_name"],
            "target_body_origin_world_m_diagnostic_only": goal_reference["body_origin_world_m"],
            "region_center_world_m_reference_only": goal_reference["native_target_region"]["center_world_m"],
            "region_half_size_site_frame_m": goal_reference["native_target_region"]["half_size_site_frame_m"],
            "region_safe_center_verified": goal_reference["native_target_region"]["safe_center_region_verified"],
            "used_as_final_goal": goal_reference["used_as_final_goal"],
            "local_bypass_endpoint_world_m": existing_bundle["scene"]["goal"]["position_world_m"],
            "local_bypass_endpoint_derived_from": existing_bundle["scene"]["goal"]["construction"]["direction_source"],
        }

        carried_binding = query_copy["objects"]["object_1"]
        live_grasp = bridge_request(
            connection,
            {
                "op": "grasp_state",
                "carried_binding": scene_report["query_payload"]["carried_binding"],
                "expected_instruction": INSTRUCTION,
            },
        )
        report["restored_grasp_contact_preflight"] = live_grasp
        if live_grasp.get("status") != "pass" or live_grasp.get("grasp_confirmed") is not True:
            raise RuntimeError("restored_attempt6_book_grasp_is_not_physically_confirmed")
        repair_context = bridge_request(connection, {"op": "repair_context"})
        gripper_inventory = bridge_request(connection, {"op": "gripper_geometry_inventory"})
        if repair_context.get("status") != "available":
            raise RuntimeError("active_OSC_repair_context_unavailable")
        if gripper_inventory.get("status") != "complete":
            raise RuntimeError("active_gripper_geometry_inventory_incomplete")

        unresolved_rules = [
            rule for rule in teacher.get("rules", [])
            if rule.get("decision") == "unknown"
        ]
        expected_unknown_rule = {
            "id": "rule_2",
            "subject_parts": ["gripper"],
            "target_id": "object_2",
            "predicate": "contact",
            "decision": "unknown",
        }
        if len(unresolved_rules) != 1 or any(
            unresolved_rules[0].get(key) != value
            for key, value in expected_unknown_rule.items()
        ):
            raise ValueError("unexpected_or_ambiguous_unknown_teacher_contact_rule")
        if goal_reference.get("body_id") != TARGET_CADDY_BODY_ID:
            raise ValueError("unknown_contact_rule_target_is_not_the_bound_caddy_body")
        teacher_for_tracking = copy.deepcopy(teacher)
        teacher_for_tracking["rules"] = [
            rule for rule in teacher_for_tracking.get("rules", [])
            if rule.get("decision") != "unknown"
        ]
        unresolved_target_contact_body_ids = {TARGET_CADDY_BODY_ID}
        report["teacher_contact_policy_projection"] = {
            "full_teacher_ir_sha256": hashlib.sha256(
                TEACHER.read_bytes()
            ).hexdigest(),
            "full_teacher_retained_in_inputs": True,
            "unknown_rule": unresolved_rules[0],
            "tracking_compiler_input_excludes_unknown_rule": "rule_2",
            "unknown_rule_conservative_runtime_handling": {
                "body_id": TARGET_CADDY_BODY_ID,
                "action": "stop_and_reject_if_robot_or_carried_book_contacts_caddy_during_local_segment",
                "monitor": "Bridge substep contact monitor",
            },
            "original_teacher_file_modified": False,
            "safety_compiler_modified": False,
            "safe_status_promotion_enabled": False,
        }

        tracking_config = TrackingConfig(
            max_control_steps_per_candidate=MAX_STEPS_PER_CANDIDATE,
            max_total_control_steps=3 * MAX_STEPS_PER_CANDIDATE,
            maximum_eef_tracking_error_m=0.04,
            maximum_grasp_relative_rotation_error_rad=0.25,
        )
        report["tracking_config"] = {
            **report["tracking_config"],
            **tracking_config.__dict__,
        }
        report["tracking_started_local_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        write_json(OUT / "experiment_report.json", report)
        tracking = track_trajectory_bundle(
            connection=connection,
            snapshot_id=snapshot_id,
            bundle=tracking_bundle,
            carried_binding=carried_binding,
            repair_context=repair_context,
            teacher=teacher_for_tracking,
            semantic_results=semantic_results,
            geometry_inventory=inventory_copy,
            gripper_inventory=gripper_inventory,
            config=tracking_config,
            exploratory_candidate_ids=set(selected_ids),
            video_capture_stride=VIDEO_STRIDE,
            unresolved_contact_body_ids=unresolved_target_contact_body_ids,
        )
        report["snapshot_tracking"] = {
            key: value
            for key, value in tracking.items()
            if key != "candidates"
        }
        robot_body_ids = {
            int(item["body_id"])
            for item in inventory_copy.get("geometries", [])
            if isinstance(item, dict)
            and item.get("category") == "robot"
            and isinstance(item.get("body_id"), int)
            and not isinstance(item.get("body_id"), bool)
        }
        candidate_summaries = []
        for candidate in tracking.get("candidates", []):
            candidate_summaries.append(
                _write_candidate_artifacts(candidate, OUT, robot_body_ids)
            )
        report["candidates"] = candidate_summaries
        report["failure_reasons"] = [
            {
                "candidate_id": item["candidate_id"],
                "direction": item["direction"],
                "physical_disposition": item["physical_disposition"],
                "executed_steps": item["executed_steps"],
                "requested_steps": item["requested_steps"],
                "reason": item["failure_reason"],
            }
            for item in candidate_summaries
            if item.get("failure_reason")
        ]
        report["actual_executed_branch_steps"] = sum(
            int(item.get("executed_steps", 0) or 0) for item in candidate_summaries
        )
        report["snapshot_restore_integrity"] = tracking.get("main_state_restored") is True
        report["status"] = (
            "complete_exploratory_physics_validation"
            if tracking.get("status") == "complete"
            and tracking.get("main_state_restored") is True
            and len(candidate_summaries) == len(selected_ids)
            else "incomplete_exploratory_physics_validation"
        )
        report["safe_success_declared"] = False
        report["physical_safety_verified"] = False
        report["interpretation"] = (
            "Snapshot OSC execution outcomes are physical observations only. Every selected candidate remains geometrically inconclusive; endpoint arrival does not certify a safe trajectory or final In placement."
        )

        restored_after = bridge_request(
            connection, {"op": "restore", "snapshot_id": snapshot_id}
        )
        final_vector = _state_vector(restored_after.get("snapshot_state", {}))
        final_restore = {
            "restored_snapshot_id": restored_after.get("restored_snapshot_id"),
            "expected_snapshot_id": snapshot_id,
            "integration_state_exact_match": bool(np.array_equal(final_vector, snapshot_vector)),
            "maximum_absolute_integration_state_error": float(np.max(np.abs(final_vector - snapshot_vector)))
            if final_vector.shape == snapshot_vector.shape
            else None,
        }
        final_restore["verified"] = bool(
            final_restore["restored_snapshot_id"] == snapshot_id
            and final_restore["integration_state_exact_match"]
        )
        report["final_snapshot_restore"] = final_restore
        if final_restore["verified"] is not True:
            report["status"] = "incomplete_snapshot_restore_unverified"
            report["failure_reasons"].append("final_snapshot_restore_did_not_match_branch_start")
        write_json(OUT / "experiment_report.json", report)

        with (OUT / "candidate_physics_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = [
                "candidate_id",
                "direction",
                "physical_disposition",
                "requested_steps",
                "executed_steps",
                "execution_complete",
                "reached_bypass_endpoint",
                "failure_reason",
                "book_root_displacement_norm_m",
                "contact_monitoring_status",
                "non_target_collision_detected",
                "contact_compiler_verdict",
                "video",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for summary in candidate_summaries:
                writer.writerow({key: summary.get(key) for key in fields})
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "snapshot_restore_integrity": report["snapshot_restore_integrity"],
                    "selected_candidates": report["exploratory_candidates"],
                    "actual_executed_branch_steps": report["actual_executed_branch_steps"],
                    "candidates": candidate_summaries,
                    "safe_success_declared": False,
                    "report": str(OUT / "experiment_report.json"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0 if report["status"] == "complete_exploratory_physics_validation" else 2
    except Exception as exc:
        report["status"] = "blocked_or_incomplete"
        report["failure_reasons"].append(f"{type(exc).__name__}:{exc}")
        write_json(OUT / "experiment_report.json", report)
        raise
    finally:
        if connection is not None:
            connection.close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        bridge_log.close()
        manifest = {
            "schema_version": "gs_opd_l0t1_snapshot_exploratory_bspline_physics_manifest.v1",
            "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "attempt6_snapshot_source": str(SNAPSHOT),
            "attempt6_snapshot_sha256": sha256(SNAPSHOT),
            "source_sha256": {},
            "files_sha256": {},
        }
        for source in sorted((OUT / "source").iterdir()):
            if source.is_file():
                manifest["source_sha256"][source.name] = sha256(source)
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.name != "run_manifest.json":
                manifest["files_sha256"][str(path.relative_to(OUT))] = sha256(path)
        write_json(OUT / "run_manifest.json", manifest)


if __name__ == "__main__":
    raise SystemExit(main())
