# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Student-first, bounded task-success execution for active LIBERO BDDL goals.

The executor owns orchestration only. It reuses the original Pi0.5 inference
adapter, Safety Bridge, Runtime Snapshots, geometry driven grasp search,
record-only RRT, native OSC_POSE tracking, and TaskPhaseController. Safety
Reconstruction is not called and every physical action remains a real MuJoCo
step. Search artifacts are candidates, never safe training data.
"""

from __future__ import annotations

import argparse
import cv2
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.repair.task_phase_controller import PhaseEvidence, TaskPhaseController
from research.gs_opd.repair.task_success_first_search import pending_object_goals


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_ROOT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "obstacle_avoidance_L0T1_raw_student_seed0_init0_20261025_234627"
)
DEFAULT_OUTPUT_ROOT = Path("/opt/data/private/gs_opd/rollouts")
DEFAULT_CHECKPOINT = Path(
    "/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft"
)
LIBERO_PYTHON = Path("/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python")
BRIDGE_AUTHKEY = b"gs-opd-local"
MAX_NATIVE_TRANSPORT_ACTIONS = 440
MAX_NATIVE_PLACEMENT_ACTIONS = 192
MAX_NATIVE_RELEASE_ACTIONS = 32
MAX_NATIVE_WITHDRAWAL_ACTIONS = 64
MAX_NATIVE_SETTLE_ACTIONS = 144


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _integration_state(snapshot_state: dict[str, Any]) -> np.ndarray:
    value = np.asarray(snapshot_state.get("mujoco_integration_state", []), dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError("runtime_snapshot_integration_state_is_invalid")
    return value


def _state_digest(value: Any) -> str:
    return _sha(np.ascontiguousarray(value, dtype=np.float64).tobytes())


def _object_state(snapshot: dict[str, Any], body_id: int) -> dict[str, Any]:
    matches = [
        row for row in snapshot.get("object_states", {}).values()
        if int(row.get("body_id", -1)) == int(body_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"active_target_body_state_not_unique:{body_id}:{len(matches)}")
    return matches[0]


def _finger_contacts_clear(finger_contacts: dict[str, Any]) -> bool:
    """Require explicit empty, model-verified target contact on both pads.

    Safety Bridge reports ``status=fail`` when a bilateral grasp is absent,
    including the valid released state with zero matched contacts.  The
    previous ``status=pass`` check therefore made the release transition
    unreachable after a physically successful release.
    """
    if not isinstance(finger_contacts, dict):
        return False
    matched = finger_contacts.get("matched_contacts")
    return bool(
        finger_contacts.get("status") in {"pass", "fail"}
        and finger_contacts.get("left") is False
        and finger_contacts.get("right") is False
        and finger_contacts.get("contact_count") == 0
        and isinstance(matched, dict)
        and matched.get("left") == []
        and matched.get("right") == []
    )


def _object_contact_escape_direction(
    contacts: dict[str, Any], target_binding: dict[str, Any],
    target_root_position_world_m: Any,
) -> dict[str, Any] | None:
    """Infer an object-outward direction from live target/scene contacts."""
    if not isinstance(contacts, dict) or contacts.get("status") not in {"contact", "no_contact"}:
        return None
    target_names = {str(name) for name in target_binding.get("geom_names", [])}
    root = np.asarray(target_root_position_world_m, dtype=np.float64)
    if not target_names or root.shape != (3,) or not np.isfinite(root).all():
        return None
    weighted: list[tuple[float, np.ndarray, np.ndarray, str]] = []
    for row in contacts.get("contacts", []):
        if not isinstance(row, dict):
            continue
        geom1 = row.get("geom1", {})
        geom2 = row.get("geom2", {})
        name1, name2 = str(geom1.get("name", "")), str(geom2.get("name", ""))
        target_is_1, target_is_2 = name1 in target_names, name2 in target_names
        if target_is_1 == target_is_2:
            continue
        blocker_name = name2 if target_is_1 else name1
        if blocker_name.startswith(("robot0_", "gripper0_")):
            continue
        normal = np.asarray(row.get("normal_geom1_to_geom2"), dtype=np.float64)
        position = np.asarray(row.get("position"), dtype=np.float64)
        if (
            normal.shape != (3,) or position.shape != (3,)
            or not np.isfinite(normal).all() or not np.isfinite(position).all()
        ):
            continue
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        # MuJoCo's contact normal points geom1 -> geom2. Move the target
        # along the opposite direction when it is geom1, and along the
        # contact normal when it is geom2.
        outward = (-normal if target_is_1 else normal) / norm
        force = np.asarray(row.get("force_and_torque", [])[:3], dtype=np.float64)
        force_norm = float(np.linalg.norm(force)) if force.shape == (3,) and np.isfinite(force).all() else 0.0
        weighted.append((max(1.0, force_norm), outward, position, blocker_name))
    if not weighted:
        return None
    resultant = sum((weight * direction for weight, direction, _, _ in weighted), np.zeros(3))
    norm = float(np.linalg.norm(resultant))
    if norm <= 1e-9:
        _, direction, _, _ = max(weighted, key=lambda row: row[0])
    else:
        direction = resultant / norm
    support_radius = max(
        0.0,
        max(float(np.dot(root - position, direction)) for _, _, position, _ in weighted),
    )
    return {
        "direction_world": direction.tolist(),
        "blocker_geom_names": sorted({row[3] for row in weighted}),
        "contact_count": len(weighted),
        "contact_positions_world_m": [row[2].tolist() for row in weighted],
        "contact_escape_source": "live_MuJoCo_target_object_to_scene_contact_normals",
        "support_radius_estimate_m": support_radius,
    }


def _task_values(native: dict[str, Any]) -> tuple[dict[tuple[str, ...], bool], bool]:
    values: dict[tuple[str, ...], bool] = {}
    for row in native.get("native_goal_values", []):
        predicate, value = row.get("predicate"), row.get("value")
        if isinstance(predicate, list) and isinstance(value, bool):
            values[tuple(str(item) for item in predicate)] = value
    if native.get("status") != "complete" or native.get("task_success") is None or not values:
        raise ValueError("native_BDDL_goal_status_is_incomplete")
    return values, bool(native["task_success"])


def _select_active_target_predicate(
    native_status: dict[str, Any], requested: list[str] | None
) -> tuple[list[str] | None, str, list[list[str]]]:
    """Select an explicit goal or the first pending native object goal."""
    pending = pending_object_goals(native_status)
    if requested is not None:
        return [str(token) for token in requested], "explicit_cli", pending
    if pending:
        return pending[0], "first_pending_native_object_goal_bddl_order", pending
    return None, "no_pending_object_goal", pending


def _verified_grasp_resume_descriptor(
    report: dict[str, Any],
    *,
    benchmark: str,
    level: int,
    task_id: int,
    seed: int,
    init_state: int,
    model_sha: str,
    bddl_sha: str,
    target_id: str,
    predicate: list[str],
) -> dict[str, Any]:
    scope = report.get("scope", {})
    success = report.get("success") or {}
    snapshot = success.get("snapshot") or {}
    expected_scope = {
        "benchmark": benchmark,
        "level": level,
        "task_id": task_id,
        "environment_seed": seed,
        "init_state": init_state,
        "model_geometry_sha256": model_sha,
        "bddl_sha256": bddl_sha,
    }
    if report.get("status") != "task_success_candidate_created":
        raise ValueError("resume_report_is_not_a_successful_grasp_skill_report")
    if success.get("task_success_candidate") is not True or success.get("safety_certified") is not False:
        raise ValueError("resume_report_candidate_provenance_is_invalid")
    if success.get("target_object_id") != target_id:
        raise ValueError("resume_report_target_object_differs_from_active_BDDL")
    if report.get("active_target", {}).get("native_goal_predicate") != predicate:
        raise ValueError("resume_report_native_predicate_differs_from_active_BDDL")
    for key, value in expected_scope.items():
        if scope.get(key) != value:
            raise ValueError(f"resume_report_{key}_mismatch")
    if report.get("route_constraints", {}).get("official_safety_demonstrations_used") is not False:
        raise ValueError("resume_report_official_demo_provenance_is_not_allowed")
    snapshot_path = Path(str(snapshot.get("path") or ""))
    snapshot_state = snapshot.get("snapshot_state")
    if not snapshot_path.is_file() or not isinstance(snapshot_state, dict):
        raise ValueError("resume_report_success_snapshot_is_missing_or_incomplete")
    return {"path": snapshot_path, "snapshot_state": snapshot_state, "success": success}


def _verified_placement_resume_descriptor(
    report: dict[str, Any],
    *,
    benchmark: str,
    level: int,
    task_id: int,
    seed: int,
    init_state: int,
    model_sha: str,
    bddl_sha: str,
    target_id: str,
    predicate: list[str],
) -> dict[str, Any]:
    """Validate a physical placement prefix whose native goal is already true."""
    scope = report.get("scope", {})
    task = report.get("task", {})
    segment = next((
        row for row in reversed(report.get("segments", []))
        if row.get("failure_phase") == "precise_placement"
    ), None)
    expected_scope = {
        "benchmark": benchmark,
        "level": level,
        "task_id": task_id,
        "seed": seed,
        "init_state": init_state,
        "model_geometry_sha256": model_sha,
    }
    if report.get("status") != "native_goal_precise_placement_failed" or segment is None:
        raise ValueError("placement_resume_report_is_not_an_archived_precise_placement_prefix")
    for key, value in expected_scope.items():
        if scope.get(key) != value:
            raise ValueError(f"placement_resume_{key}_mismatch")
    if task.get("BDDL_sha256") != bddl_sha:
        raise ValueError("placement_resume_BDDL_fingerprint_mismatch")
    if task.get("target_object_from_active_BDDL") != target_id:
        raise ValueError("placement_resume_target_object_differs_from_active_BDDL")
    if task.get("native_goal_predicate") != predicate or segment.get("native_goal_predicate") != predicate:
        raise ValueError("placement_resume_native_predicate_differs_from_active_BDDL")
    policy = report.get("policy", {})
    if (
        policy.get("official_safety_demonstration_data_used") is not False
        or policy.get("object_or_robot_state_written_directly") is not False
        or policy.get("all_actions_are_real_MuJoCo_steps") is not True
        or report.get("safety_certified") is True
    ):
        raise ValueError("placement_resume_provenance_is_not_allowed")
    native = segment.get("final_native_task_status")
    if not isinstance(native, dict) or native.get("task_success") is True:
        raise ValueError("placement_resume_requires_an_incomplete_native_task")
    values = {
        tuple(row.get("predicate", [])): row.get("value")
        for row in native.get("native_goal_values", [])
        if isinstance(row, dict)
    }
    if values.get(tuple(predicate)) is not True:
        raise ValueError("placement_resume_requires_current_native_goal_to_be_true")
    snapshot_path = Path(str(segment.get("terminal_snapshot") or ""))
    if not snapshot_path.is_file():
        raise ValueError("placement_resume_terminal_snapshot_is_missing")
    from research.gs_opd.repair.task_success_first_grasp_search import _PythonObjectState

    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    payload = snapshot_path.read_bytes()
    envelope = pickle.loads(payload)
    scene = envelope.get("scene_metadata", {})
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != model_sha
        or scene.get("benchmark") != benchmark
        or scene.get("level") != level
        or scene.get("task_id") != task_id
        or scene.get("environment_seed") != seed
        or scene.get("init_state_index") != init_state
        or scene.get("bddl_sha256") != bddl_sha
    ):
        raise ValueError("placement_resume_snapshot_scene_or_model_identity_mismatch")
    mujoco_state = envelope.get("snapshot", {}).get("mujoco_state")
    if mujoco_state is None:
        raise ValueError("placement_resume_snapshot_has_no_MuJoCo_integration_state")
    return {
        "path": snapshot_path,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot_state": {"mujoco_integration_state": np.asarray(mujoco_state, dtype=np.float64)},
        "native_task_status": native,
        "segment": segment,
    }


def _register_verified_operation_skill_from_report(
    library_path: Path | None,
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any] | None:
    """Add an executed grasp/placement cycle after native placement and release."""
    if library_path is None or not library_path.is_file():
        return None
    task = report.get("task", {})
    target_id = str(task.get("target_object_from_active_BDDL") or "")
    predicate = task.get("native_goal_predicate")
    final_native = report.get("final_native_task_status")
    if not target_id or not isinstance(predicate, list) or not isinstance(final_native, dict):
        return None
    native_value = next((
        row.get("value") for row in final_native.get("native_goal_values", [])
        if row.get("predicate") == predicate
    ), None)
    if native_value is not True:
        return None
    placement = next((
        segment for segment in reversed(report.get("segments", []))
        if segment.get("type") in {
            "expert_native_goal_stage_control",
            "resumed_expert_native_goal_stage_control",
            "resumed_expert_release_withdrawal",
        }
    ), None)
    if not isinstance(placement, dict) or int(placement.get("recorded_actions", 0) or 0) <= 0:
        return None
    released_and_stable = bool(
        placement.get("partial_goal_settled") is True
        or (final_native.get("task_success") is True
            and placement.get("task_phase_controller_phase") in {"settle", "verify_task"})
    )
    if not released_and_stable:
        return None

    from research.gs_opd.repair.task_skill_experience import (
        _object_family,
        append_verified_search_operation_skill,
        load_experience_library,
        matching_operation_skill,
    )

    library = load_experience_library(library_path)
    source_operation = matching_operation_skill(library, target_id)
    phases = list((source_operation or {}).get("phase_sequence", []))
    if not phases:
        phases = [
            "approach", "stable_grasp", "lift", "transport", "precise_placement",
            "release", "withdrawal", "settle",
        ]

    documents: list[tuple[Path, dict[str, Any]]] = [(report_path, report)]
    seen = {str(report_path.resolve())}
    cursor = 0
    while cursor < len(documents):
        current_path, current = documents[cursor]
        cursor += 1
        pending_paths: list[Path] = []
        for key in ("resumed_from_grasp_skill", "resumed_from_native_goal_contact_snapshot"):
            source = current.get(key)
            if isinstance(source, dict) and source.get("source_report"):
                pending_paths.append(Path(source["source_report"]))
        for segment in current.get("segments", []):
            if segment.get("report"):
                pending_paths.append(Path(segment["report"]))
        while pending_paths:
            candidate = pending_paths.pop()
            resolved = str(candidate.resolve())
            if resolved in seen or not candidate.is_file():
                continue
            seen.add(resolved)
            try:
                child = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            documents.append((candidate, child))
    stable_grasp = False
    physical_actions = 0
    source_reports = []
    for source_path, document in documents:
        source_reports.append(str(source_path))
        if document.get("policy", {}).get("official_safety_demonstration_data_used") is True:
            raise ValueError("official_safety_demonstration_provenance_found_in_operation_skill_sources")
        if document.get("route_constraints", {}).get("official_safety_demonstrations_used") is True:
            raise ValueError("official_safety_demonstration_provenance_found_in_operation_skill_sources")
        for container in [document.get("success", {})] + [
            segment.get("success", {}) for segment in document.get("segments", [])
        ]:
            stable = container.get("stable_bilateral_grasp", {}) if isinstance(container, dict) else {}
            stable_grasp |= bool(stable.get("stable") is True or stable.get("stable_grasp_observed") is True)
        for segment in document.get("segments", []):
            physical_actions += int(segment.get("actions_executed", 0) or 0)
            physical_actions += int(segment.get("recorded_actions", 0) or 0)
            if not segment.get("report"):
                for branch in segment.get("physical_branches", []):
                    physical_actions += int(
                        branch.get("attempted_control_actions_archived", branch.get("branch_control_actions", 0))
                        or 0
                    )
        for branch in document.get("physical_branches", []):
            physical_actions += int(
                branch.get("attempted_control_actions_archived", branch.get("branch_control_actions", 0))
                or 0
            )
            grasp = branch.get("grasp_before_lift", {})
            stable_grasp |= bool(
                branch.get("stable_bilateral_grasp_observed") is True
                or grasp.get("stable") is True or grasp.get("stable_grasp_observed") is True
            )
    if not stable_grasp or physical_actions <= 0:
        return None

    scope = report.get("scope", {})
    scene = f"{scope.get('benchmark')}/L{scope.get('level')}T{scope.get('task_id')}"
    signature = hashlib.sha256(json.dumps(
        [scene, int(scope.get("seed", 0)), int(scope.get("init_state", 0)), target_id, predicate],
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()[:10]
    template = {
        "skill_id": f"object_transfer_cycle:{_object_family(target_id)}:{signature}",
        "object_family": _object_family(target_id),
        "operation": "pick_transport_place_release",
        "phase_sequence": phases,
        "source_task": task.get("instruction"),
        "source_scene": scene,
        "source_object_id": target_id,
        "native_goal_predicate": predicate,
        "source_native_task_success": final_native.get("task_success") is True,
        "skill_success": True,
        "native_object_goal_satisfied": True,
        "released_and_stable": True,
        "stable_bilateral_grasp_verified": True,
        "physical_actions": physical_actions,
        "source_execution_reports": list(dict.fromkeys(source_reports)),
        "historical_actions_replayed": False,
        "source_actions_used_as_direct_cross_scene_labels": False,
        "official_safety_demonstrations_used": False,
        "safety_certified": False,
    }
    added = append_verified_search_operation_skill(library_path, template)
    return {
        "status": "skill_added" if added else "duplicate_skill_already_present",
        "skill_id": template["skill_id"],
        "physical_actions_in_source_prefix": physical_actions,
        "source_execution_reports": template["source_execution_reports"],
        "cross_scene_physical_validation_count": 1 if added else None,
        "source_native_task_success": template["source_native_task_success"],
        "safety_certified": False,
        "official_safety_demonstrations_used": False,
    }


def _controller_config(target_id: str, predicate: list[str], grasp_candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "objects": [{
            "object_id": target_id,
            "predicate": predicate,
            "goal_source": "active_native_BDDL_goal_and_native_goal_region_query",
        }],
        "grasp_pose_candidates": [{"candidate_id": str(grasp_candidate.get("candidate_id", "live_verified_grasp"))}],
        "phase_control_step_limits": {
            "pregrasp": 64, "approach": 128, "close_gripper": 32,
            "stable_grasp": 32, "lift": 32, "transport": 160,
            "precise_placement": 96, "release": 32, "withdrawal": 64,
            "settle": 32,
        },
        "stable_grasp_control_steps": 3,
        "lift_validation_height_m": 0.03,
        "settle_position_span_m": 0.002,
        "settle_linear_speed_m_s": 0.08,
        "settle_angular_speed_rad_s": 0.8,
    }


def _phase_evidence(
    bridge,
    controller: TaskPhaseController,
    target_id: str,
    target_body_id: int,
    predicate: list[str],
    *,
    target_binding: dict[str, Any] | None = None,
    eef_goal_reached: bool = False,
) -> tuple[PhaseEvidence, dict[str, Any]]:
    observed = bridge.request({"op": "observe", "include_snapshot_state": True})
    snapshot = observed["snapshot_state"]
    native = bridge.request({"op": "native_task_status"})
    values, task_success = _task_values(native)
    obj = _object_state(snapshot, target_body_id)
    robot_rows = snapshot.get("robot_states", [])
    if len(robot_rows) != 1:
        raise ValueError("task_success_first_executor_requires_one_Panda_state")
    robot = robot_rows[0]
    binding = dict(target_binding) if target_binding is not None else {
        "object_id": target_id,
        "body_id": target_body_id,
        "body_name": obj.get("body_name"),
        "geom_ids": obj.get("geom_ids", []),
        "geom_names": obj.get("geom_names", []),
    }
    if (
        binding.get("object_id") != target_id
        or int(binding.get("body_id", -1)) != int(target_body_id)
        or not isinstance(binding.get("geom_ids"), list)
        or not isinstance(binding.get("geom_names"), list)
        or not binding["geom_ids"]
        or len(binding["geom_ids"]) != len(binding["geom_names"])
    ):
        raise ValueError("phase_evidence_requires_complete_model_verified_target_binding")
    grasp = bridge.request({
        "op": "grasp_state", "carried_binding": binding,
        "expected_instruction": observed.get("instruction"),
    })
    grasped = grasp.get("grasp_confirmed") is True
    finger_contacts = grasp.get("finger_contacts", {})
    finger_contacts_clear = _finger_contacts_clear(finger_contacts)
    all_satisfied = bool(values) and all(values.values())
    state = PhaseEvidence(
        timestep=int(observed.get("timestep", -1)),
        eef_goal_reached=eef_goal_reached,
        held_object=target_id if grasped else None,
        grasp_confirmed=grasped,
        object_position_world_m=tuple(float(x) for x in obj["root_pos"]),
        object_quaternion_wxyz=tuple(float(x) for x in obj["root_quat"]),
        object_qvel=tuple(float(x) for x in obj.get("joint_qvel", [0.0] * 6)),
        eef_position_world_m=tuple(float(x) for x in robot["eef_pos"]),
        task_goal_satisfied=bool(values.get(tuple(predicate), False)),
        all_task_goals_satisfied=all_satisfied,
        satisfied_object_ids=tuple(
            target_id for key, value in values.items() if value and len(key) == 3 and key[1] == target_id
        ),
        fixture_predicate_satisfied=None,
        trusted_fixture_activation_available=False,
        task_success=task_success,
        finger_contacts_clear=finger_contacts_clear,
        eef_position_error_m=0.0 if eef_goal_reached else None,
        eef_orientation_error_rad=0.0 if eef_goal_reached else None,
    )
    return state, {"observation": observed, "native_task_status": native, "grasp_state": grasp}


def _release_retreat_target(
    eef_position_world_m: Any,
    object_position_world_m: Any,
    *,
    retreat_distance_m: float = 0.003,
) -> np.ndarray | None:
    """Return a short horizontal escape waypoint away from a released object."""
    eef = np.asarray(eef_position_world_m, dtype=np.float64)
    obj = np.asarray(object_position_world_m, dtype=np.float64)
    if (
        eef.shape != (3,) or obj.shape != (3,)
        or not np.isfinite(eef).all() or not np.isfinite(obj).all()
        or not np.isfinite(retreat_distance_m) or retreat_distance_m <= 0.0
    ):
        return None
    away = eef - obj
    away[2] = 0.0
    norm = float(np.linalg.norm(away))
    if norm <= 1e-6:
        return None
    return eef + away * (float(retreat_distance_m) / norm)


def _student_segment(
    bridge,
    student,
    *,
    max_actions: int,
    target_binding: dict[str, Any],
    instruction: str,
    policy: dict[str, Any],
    output_dir: Path,
    segment: str,
) -> dict[str, Any]:
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _record_step,
        _write_recording,
    )
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor

    records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    monitor = GraspStabilityMonitor(required_samples=3)
    data = bridge.request({"op": "observe", "include_snapshot_state": True})
    initial_eef = np.asarray(data["snapshot_state"]["robot_states"][0]["eef_pos"], dtype=np.float64)
    initial_object = np.asarray(
        _object_state(data["snapshot_state"], int(target_binding["body_id"]))["root_pos"],
        dtype=np.float64,
    )
    native = bridge.request({"op": "native_task_status"})
    _, success = _task_values(native)
    prediction: np.ndarray | None = None
    cursor = 0
    environment_done = False
    failure_trigger = None
    target_distance_history: list[float] = []
    object_displacement_history: list[float] = []
    for action_index in range(max_actions):
        if success:
            break
        if prediction is None or cursor >= len(prediction):
            prediction = student.predict(data)
            cursor = 0
        action = prediction[cursor]
        cursor += 1
        try:
            _record_step(
                bridge, action, f"student_{segment}", f"student_{segment}", action_index,
                target_binding, instruction, policy, monitor, records, frames,
            )
        except RuntimeError as exc:
            # _record_step archives the executed transition before reporting
            # done. A non-success terminal state is a Student skill failure;
            # the caller may restore its branch Snapshot and hand off.
            if "environment_done_during_grasp_only_branch" not in str(exc):
                raise
            environment_done = True
        data = bridge.request({"op": "observe", "include_snapshot_state": True})
        snapshot_state = data["snapshot_state"]
        target_position = np.asarray(
            _object_state(snapshot_state, int(target_binding["body_id"]))["root_pos"],
            dtype=np.float64,
        )
        eef_position = np.asarray(snapshot_state["robot_states"][0]["eef_pos"], dtype=np.float64)
        target_distance_history.append(float(np.linalg.norm(eef_position - target_position)))
        object_displacement_history.append(float(np.linalg.norm(target_position - initial_object)))
        native = bridge.request({"op": "native_task_status"})
        _, success = _task_values(native)
        if success or environment_done:
            break
        if len(target_distance_history) >= 12 and not monitor.stable:
            start_distance = target_distance_history[-12]
            best_distance = min(target_distance_history[-12:])
            object_progress = max(object_displacement_history[-12:]) - min(object_displacement_history[-12:])
            if start_distance - best_distance < 0.006 and object_progress < 0.002:
                failure_trigger = "student_goal_approach_and_object_progress_stagnated_for_12_actions"
                break
    _write_recording(output_dir, records, frames)
    final_snapshot = bridge.request({"op": "export_runtime_snapshot"})
    if final_snapshot.get("status") != "complete" or not isinstance(final_snapshot.get("payload"), bytes):
        raise RuntimeError(f"student_{segment}_terminal_snapshot_export_failed")
    path = output_dir / f"student_{segment}_terminal_runtime_snapshot.pkl"
    path.write_bytes(final_snapshot["payload"])
    final = final_snapshot["snapshot_state"]
    final_eef = np.asarray(final["robot_states"][0]["eef_pos"], dtype=np.float64)
    final_obj = np.asarray(
        _object_state(final, int(target_binding["body_id"]))["root_pos"], dtype=np.float64
    )
    return {
        "segment": segment,
        "policy_source": "original_pi05_rlinf_libero_sft_base_only_no_LoRA",
        "actions_executed": len(records),
        "task_success": bool(success),
        "native_task_status": native,
        "environment_done_without_native_success": bool(environment_done and not success),
        "failure_trigger": failure_trigger,
        "stagnation_window_actions": 12,
        "target_displacement_m": float(np.linalg.norm(final_obj - initial_object)),
        "eef_displacement_m": float(np.linalg.norm(final_eef - initial_eef)),
        "grasp_stability": monitor.diagnostics(),
        "terminal_snapshot": str(path),
        "terminal_snapshot_sha256": _sha(path.read_bytes()),
        "artifacts": {
            "records": str(output_dir / "physical_step_records.jsonl.gz"),
            "npz": str(output_dir / "task_success_first_actions_states.npz"),
            "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
        },
    }


def _run_articulated_fixture_goal(
    bridge,
    *,
    predicate: list[str],
    target_binding: dict[str, Any],
    instruction: str,
    policy: dict[str, Any],
    articulation_context: dict[str, Any],
    controller_fields: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Physically push a native articulated fixture goal through OSC_POSE."""
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _advance_pose,
        _write_recording,
    )
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor

    if articulation_context.get("status") != "complete" or not articulation_context.get("read_only"):
        raise RuntimeError(f"articulated_fixture_context_unavailable:{articulation_context.get('error')}")
    path = np.asarray(articulation_context.get("contact_point_path_world_m"), dtype=np.float64)
    outward = np.asarray(articulation_context.get("outward_direction_world"), dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or outward.shape != (3,):
        raise ValueError("articulation_context_does_not_contain_a_finite_physical_contact_path")
    records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    monitor = GraspStabilityMonitor(required_samples=3)
    observed = bridge.request({"op": "observe", "include_snapshot_state": True})
    current = observed["snapshot_state"]["robot_states"][0]
    quaternion = np.asarray(current["eef_quat"], dtype=np.float64)
    _advance_pose.controller_fields = controller_fields
    counter = [0]
    precontact = path[0] + outward * 0.075
    contact_offset = outward * 0.035
    waypoints = [precontact, path[0] + contact_offset]
    waypoints.extend(point + contact_offset for point in path[1:])
    for waypoint in waypoints:
        _advance_pose(
            bridge, waypoint, quaternion, "articulated_fixture_close",
            f"native_{predicate[0].lower()}_{predicate[1]}", counter,
            max(1, 128 - counter[0]), target_binding, instruction, policy,
            monitor, records, frames,
        )
        native = bridge.request({"op": "native_task_status"})
        if any(row.get("predicate") == predicate and row.get("value") is True
               for row in native.get("native_goal_values", [])):
            break
        if counter[0] >= 128:
            break
    final_native = bridge.request({"op": "native_task_status"})
    articulation_after = bridge.request({
        "op": "task_success_first_articulation_context",
        "expected_instruction": instruction,
        "predicate": predicate,
    })
    _write_recording(output_dir, records, frames)
    final_snapshot = bridge.request({"op": "export_runtime_snapshot"})
    snapshot_path = output_dir / "articulated_fixture_terminal_runtime_snapshot.pkl"
    if final_snapshot.get("status") == "complete" and isinstance(final_snapshot.get("payload"), bytes):
        snapshot_path.write_bytes(final_snapshot["payload"])
    predicate_value = next(
        (row.get("value") for row in final_native.get("native_goal_values", [])
         if row.get("predicate") == predicate), None
    )
    return {
        "status": "complete_task_success" if final_native.get("task_success") is True
        else "partial_predicate_completed" if predicate_value is True
        else "articulated_fixture_goal_incomplete",
        "native_goal_predicate": predicate,
        "native_goal_value": predicate_value,
        "native_task_success": final_native.get("task_success"),
        "final_native_task_status": final_native,
        "articulation_before": articulation_context,
        "articulation_after": articulation_after,
        "recorded_actions": len(records),
        "recorded_physics_substeps": int(sum(
            item.get("contact_monitoring", {}).get("captured_physics_substeps", 0)
            for item in records
        )),
        "terminal_snapshot": str(snapshot_path) if snapshot_path.is_file() else None,
        "safety_certified": False,
        "artifacts": {
            "records": str(output_dir / "physical_step_records.jsonl.gz"),
            "npz": str(output_dir / "task_success_first_actions_states.npz"),
            "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
        },
    }


def _native_place(
    bridge,
    *,
    target_id: str,
    target_binding: dict[str, Any],
    predicate: list[str],
    instruction: str,
    model_sha: str,
    safety_policy: dict[str, Any],
    controller_fields: dict[str, Any],
    controller: TaskPhaseController,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    from scipy.spatial.transform import Rotation
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _advance_pose,
        _compress_eef_route,
        _record_step,
        _snapshot_state_vector,
        _state_sha,
        _wxyz,
        _write_recording,
    )
    from research.gs_opd.repair.task_success_first_search import (
        high_approach_route,
        route_control_action_budget,
    )
    from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor

    native_goal = bridge.request({"op": "native_goal_query", "predicate": predicate})
    if native_goal.get("status") != "complete" or native_goal.get("predicate_is_native_goal") is not True:
        raise RuntimeError(f"active_BDDL_goal_region_unavailable:{native_goal.get('error')}")
    region = native_goal["region"]
    observed = bridge.request({"op": "observe", "include_snapshot_state": True})
    state = observed["snapshot_state"]
    obj = _object_state(state, int(target_binding["body_id"]))
    robot = state["robot_states"][0]
    object_world = np.eye(4)
    object_world[:3, :3] = Rotation.from_quat(
        np.asarray(obj["root_quat"], dtype=np.float64)[[1, 2, 3, 0]]
    ).as_matrix()
    object_world[:3, 3] = obj["root_pos"]
    eef_world = np.eye(4)
    eef_world[:3, :3] = Rotation.from_quat(
        np.asarray(robot["eef_quat"], dtype=np.float64)[[1, 2, 3, 0]]
    ).as_matrix()
    eef_world[:3, 3] = robot["eef_pos"]
    object_to_eef = np.linalg.inv(object_world) @ eef_world
    goal_object = np.eye(4)
    goal_object[:3, :3] = np.asarray(region["site_rotation_world"], dtype=np.float64)
    goal_object[:3, 3] = np.asarray(region["site_position_world_m"], dtype=np.float64)
    goal_eef = goal_object @ object_to_eef
    high_eef = goal_eef.copy()
    high_eef[:3, 3] += np.asarray([0.0, 0.0, 0.14])
    high_quat = _wxyz(high_eef[:3, :3])
    state_sha = _state_sha(_snapshot_state_vector(state))
    resume_at_release = controller.phase == "release"
    start_snapshot = bridge.request({"op": "export_runtime_snapshot"})
    if start_snapshot.get("status") != "complete" or not isinstance(start_snapshot.get("payload"), bytes):
        raise RuntimeError("native_goal_transport_start_snapshot_export_failed")
    start_snapshot_path = output_dir / "native_goal_transport_start_runtime_snapshot.pkl"
    start_snapshot_path.write_bytes(start_snapshot["payload"])
    planned = (
        {
            "status": "not_required_native_goal_satisfied_resume_at_release",
            "live_state_unchanged": True,
            "safety_certified": False,
        }
        if resume_at_release else bridge.request({
            "op": "task_success_first_rrt",
            "expected_instruction": instruction,
            "model_geometry_sha256": model_sha,
            "start_state_sha256": state_sha,
            "goal_pose_world": {"position_world_m": high_eef[:3, 3].tolist(), "quaternion_wxyz": high_quat.tolist()},
            "safety_geometry_policy": safety_policy,
            "planner_config": {
                "step_rad": 0.22, "connect_max_steps": 20, "goal_bias": 0.15,
                "max_iterations": 80, "max_config_checks": 800,
                "max_candidate_routes": 1, "search_time_budget_s": 2.0,
                "shortcut_attempts": 40, "minimum_clearance_m": 0.0,
                "require_safe_configurations": False, "seed": 74103,
            },
        })
    )
    _advance_pose.controller_fields = controller_fields
    records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    monitor = GraspStabilityMonitor(required_samples=3)
    counter = [0]
    durable_records_path = output_dir / "durable_physical_step_records.jsonl"
    durable_frames_dir = output_dir / "durable_dual_camera_frames"
    durable_frames_dir.mkdir(parents=True, exist_ok=True)
    contact_escape: dict[str, Any] | None = None

    def persist_step(record: dict[str, Any], frame: dict[str, Any]) -> None:
        # Preserve each executed action before the phase returns. The final
        # bundle still writes compressed records, NPZ and MP4; these files make
        # the physical prefix recoverable if a long RRT/OSC stage is interrupted.
        with durable_records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                record, ensure_ascii=False, separators=(",", ":"),
                default=lambda item: item.tolist() if isinstance(item, np.ndarray) else str(item),
            ))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        combined = np.concatenate((frame["agentview_rgb"], frame["wrist_rgb"]), axis=1)
        frame_path = durable_frames_dir / f"frame_{int(frame['step_index']):06d}.jpg"
        if not cv2.imwrite(str(frame_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)):
            raise RuntimeError("durable_dual_camera_frame_write_failed")

    def persist_latest_step() -> None:
        if records and frames:
            persist_step(records[-1], frames[-1])

    def archive_failure(failure_phase: str, exc: Exception, route_source: str | None = None):
        """Persist every executed physical prefix before reporting a failed phase."""
        _write_recording(output_dir, records, frames)
        failure_snapshot_path = output_dir / f"native_goal_{failure_phase}_failed_runtime_snapshot.pkl"
        snapshot_error = None
        try:
            final_snapshot = bridge.request({"op": "export_runtime_snapshot"})
            if final_snapshot.get("status") == "complete" and isinstance(final_snapshot.get("payload"), bytes):
                failure_snapshot_path.write_bytes(final_snapshot["payload"])
        except Exception as snapshot_exc:
            snapshot_error = f"{type(snapshot_exc).__name__}: {snapshot_exc}"
        try:
            final_native = bridge.request({"op": "native_task_status"})
        except Exception as native_exc:
            final_native = {
                "status": "unavailable_bridge_terminated",
                "error": f"{type(native_exc).__name__}: {native_exc}",
                "last_archived_task_success": records[-1].get("task_success") if records else None,
            }
        return {
            "status": f"native_goal_{failure_phase}_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "native_goal_predicate": predicate,
            "native_task_success": final_native.get("task_success"),
            "final_native_task_status": final_native,
            "transport_route_source": route_source,
            "target_object_contact_escape": contact_escape,
            "transport_route_RRT": {
                "status": planned.get("status"),
                "failure_reason": planned.get("failure_reason"),
                "time_budget_s": 2.0,
                "maximum_configuration_checks": 800,
                "safety_certified": False,
            },
            "recorded_actions": len(records),
            "recorded_physics_substeps": int(sum(
                item.get("contact_monitoring", {}).get("captured_physics_substeps", 0)
                for item in records
            )),
            "failure_phase": failure_phase,
            "start_snapshot": str(start_snapshot_path),
            "terminal_snapshot": str(failure_snapshot_path) if failure_snapshot_path.is_file() else None,
            "terminal_snapshot_export_error": snapshot_error,
            "terminal_snapshot_sha256": (
                hashlib.sha256(failure_snapshot_path.read_bytes()).hexdigest()
                if failure_snapshot_path.is_file() else None
            ),
            "artifacts": {
                "records": str(output_dir / "physical_step_records.jsonl.gz"),
                "npz": str(output_dir / "task_success_first_actions_states.npz"),
                "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                "durable_records": str(durable_records_path),
                "durable_frames": str(durable_frames_dir),
            },
            "safety_certified": False,
        }

    def native_goal_satisfied() -> bool:
        status = bridge.request({"op": "native_task_status"})
        if status.get("status") != "complete":
            raise RuntimeError("native_goal_status_unavailable_during_precise_placement")
        return any(
            row.get("predicate") == predicate and row.get("value") is True
            for row in status.get("native_goal_values", [])
        )
    if resume_at_release:
        route = [np.r_[robot["eef_pos"], _wxyz(eef_world[:3, :3])]]
        transport_route_source = "verified_native_goal_snapshot_resume_at_release"
    elif planned.get("status") == "offline_feasible" and planned.get("live_state_unchanged") is True:
        route = _compress_eef_route(planned.get("eef_reference_poses_world"))
        transport_route_source = "jointspace_RRT"
    else:
        if planned.get("live_state_unchanged") is not True:
            raise RuntimeError("native_goal_transport_RRT_modified_live_state")
        start_pose = np.r_[robot["eef_pos"], _wxyz(eef_world[:3, :3])]
        route = _compress_eef_route(high_approach_route(start_pose, np.r_[high_eef[:3, 3], high_quat]))
        transport_route_source = "high_OSC_waypoints_after_RRT_failure"
    if not resume_at_release:
        live_contacts = bridge.request({"op": "inspect_contacts"})
        contact_escape = _object_contact_escape_direction(
            live_contacts, target_binding, obj["root_pos"],
        )
        if contact_escape is not None:
            outward = np.asarray(contact_escape["direction_world"], dtype=np.float64)
            root_position = np.asarray(obj["root_pos"], dtype=np.float64)
            support_radius = float(contact_escape["support_radius_estimate_m"])
            escape_distance = min(0.25, max(0.06, support_radius + 0.04))
            escaped_object = object_world.copy()
            escaped_object[:3, 3] = root_position + outward * escape_distance
            escaped_eef = escaped_object @ object_to_eef
            start_pose = np.r_[robot["eef_pos"], _wxyz(eef_world[:3, :3])]
            escape_pose = np.r_[escaped_eef[:3, 3], _wxyz(escaped_eef[:3, :3])]
            raised_escape = escape_pose.copy()
            raised_escape[2] += max(0.10, support_radius + 0.05)
            raised_goal = np.r_[high_eef[:3, 3], high_quat]
            raised_goal[2] += max(0.08, support_radius + 0.04)
            route = _compress_eef_route([
                start_pose, escape_pose, raised_escape, raised_goal,
                np.r_[high_eef[:3, 3], high_quat],
            ])
            contact_escape.update({
                "status": "contact_normal_waypoint_scheduled",
                "escape_displacement_m": escape_distance,
                "object_support_radius_estimate_m": support_radius,
                "elevated_route_extra_height_m": max(0.10, support_radius + 0.05),
                "route_uses_real_OSC_MuJoCo_actions": True,
                "object_or_robot_state_written_directly": False,
            })
            transport_route_source = "target_object_contact_normal_escape_then_elevated_OSC_waypoints"
    transport_action_budget = 0 if resume_at_release else route_control_action_budget(
        route, minimum_actions=74, maximum_actions=MAX_NATIVE_TRANSPORT_ACTIONS
    )
    try:
        for waypoint in route[1:]:
            _advance_pose(
                bridge, waypoint[:3], waypoint[3:], "transport", "native_goal_transport", counter,
                max(1, transport_action_budget - counter[0]), target_binding,
                instruction, safety_policy, monitor, records, frames,
                step_callback=persist_step,
            )
            if counter[0] > transport_action_budget:
                raise RuntimeError("native_goal_transport_action_budget_exhausted")
    except Exception as exc:
        _write_recording(output_dir, records, frames)
        failure_snapshot_path = output_dir / "native_goal_transport_failed_runtime_snapshot.pkl"
        snapshot_error = None
        try:
            final_snapshot = bridge.request({"op": "export_runtime_snapshot"})
            if final_snapshot.get("status") == "complete" and isinstance(final_snapshot.get("payload"), bytes):
                failure_snapshot_path.write_bytes(final_snapshot["payload"])
        except Exception as snapshot_exc:
            snapshot_error = f"{type(snapshot_exc).__name__}: {snapshot_exc}"
        try:
            final_native = bridge.request({"op": "native_task_status"})
        except Exception as native_exc:
            final_native = {
                "status": "unavailable_bridge_terminated",
                "error": f"{type(native_exc).__name__}: {native_exc}",
                "last_archived_task_success": records[-1].get("task_success") if records else None,
            }
        return {
            "status": "native_goal_transport_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "native_goal_predicate": predicate,
            "native_task_success": final_native.get("task_success"),
            "final_native_task_status": final_native,
            "transport_route_source": transport_route_source,
            "target_object_contact_escape": contact_escape,
            "transport_route_RRT": {
                "status": planned.get("status"),
                "failure_reason": planned.get("failure_reason"),
                "time_budget_s": 2.0,
                "maximum_configuration_checks": 800,
                "safety_certified": False,
            },
            "transport_actions": len(records),
            "physics_substeps": int(sum(
                item.get("contact_monitoring", {}).get("captured_physics_substeps", 0)
                for item in records
            )),
            "start_snapshot": str(start_snapshot_path),
            "terminal_snapshot": str(failure_snapshot_path) if failure_snapshot_path.is_file() else None,
            "terminal_snapshot_export_error": snapshot_error,
            "artifacts": {
                "records": str(output_dir / "physical_step_records.jsonl.gz"),
                "npz": str(output_dir / "task_success_first_actions_states.npz"),
                "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                "durable_records": str(durable_records_path),
                "durable_frames": str(durable_frames_dir),
            },
            "safety_certified": False,
        }
    evidence, stage = _phase_evidence(
        bridge, controller, target_id, int(target_binding["body_id"]), predicate,
        target_binding=target_binding,
        eef_goal_reached=True,
    )
    controller.update(evidence)
    phase_events = [{"timestep": evidence.timestep, "phase": controller.phase, "native": stage["native_task_status"]}]
    precise_pose_reached = True
    if not resume_at_release:
        try:
            precise_pose_reached = _advance_pose(
                bridge, goal_eef[:3, 3], _wxyz(goal_eef[:3, :3]), "precise_placement",
                "native_goal_place", counter, MAX_NATIVE_PLACEMENT_ACTIONS, target_binding, instruction,
                safety_policy, monitor, records, frames,
                stop_condition=native_goal_satisfied,
                step_callback=persist_step,
            )
        except Exception as exc:
            return archive_failure("precise_placement", exc, transport_route_source)
        if counter[0] > transport_action_budget + MAX_NATIVE_PLACEMENT_ACTIONS:
            return archive_failure(
                "precise_placement", RuntimeError("native_goal_place_total_action_budget_exhausted"),
                transport_route_source,
            )
    evidence, stage = _phase_evidence(
        bridge, controller, target_id, int(target_binding["body_id"]), predicate,
        target_binding=target_binding,
        eef_goal_reached=precise_pose_reached,
    )
    controller.update(evidence)
    phase_events.append({"timestep": evidence.timestep, "phase": controller.phase, "native": stage["native_task_status"]})
    release_limit = (
        MAX_NATIVE_RELEASE_ACTIONS if resume_at_release
        else transport_action_budget + MAX_NATIVE_PLACEMENT_ACTIONS + MAX_NATIVE_RELEASE_ACTIONS
    )
    release_contact_escape_actions = 0
    while counter[0] < release_limit and controller.phase == "release":
        release_action = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        grasp_state = stage.get("grasp_state", {})
        finger_contacts = grasp_state.get("finger_contacts", {})
        closure = grasp_state.get("gripper_closure", {})
        close_fraction = closure.get("close_command_fraction")
        if (
            closure.get("status") == "pass"
            and isinstance(close_fraction, (int, float))
            and float(close_fraction) <= 1e-6
            and (finger_contacts.get("left") is True or finger_contacts.get("right") is True)
        ):
            snapshot = stage.get("observation", {}).get("snapshot_state", {})
            robot = snapshot.get("robot_states", [{}])[0]
            obj = _object_state(snapshot, int(target_binding["body_id"]))
            current_eef = np.asarray(robot.get("eef_pos", []), dtype=np.float64)
            current_quat = np.asarray(robot.get("eef_quat", []), dtype=np.float64)
            escape_target = _release_retreat_target(current_eef, obj.get("root_pos", []))
            if escape_target is not None and current_quat.shape == (4,) and np.isfinite(current_quat).all():
                release_action = world_pose_tracking_action(
                    current_eef, current_quat, escape_target, current_quat,
                    controller_fields, gripper_action=-1.0,
                ).astype(np.float32)
                release_contact_escape_actions += 1
        _record_step(
            bridge, release_action,
            "release", "native_goal_place", counter[0], target_binding, instruction,
            safety_policy, monitor, records, frames,
        )
        counter[0] += 1
        persist_latest_step()
        evidence, stage = _phase_evidence(
            bridge, controller, target_id, int(target_binding["body_id"]), predicate,
            target_binding=target_binding,
        )
        before_phase = controller.phase
        try:
            controller.update(evidence)
        except Exception as exc:
            return archive_failure("release", exc, transport_route_source)
        if controller.phase != before_phase:
            phase_events.append({"timestep": evidence.timestep, "phase": controller.phase, "native": stage["native_task_status"]})
        if stage["native_task_status"].get("task_success") is True:
            break
    if controller.phase == "withdrawal":
        current = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]
        withdraw = np.asarray(current["eef_pos"], dtype=np.float64) + np.asarray([0.0, 0.0, 0.10])
        try:
            _advance_pose(
                bridge, withdraw, np.asarray(current["eef_quat"], dtype=np.float64),
                "withdrawal", "native_goal_place", counter, MAX_NATIVE_WITHDRAWAL_ACTIONS, target_binding,
                instruction, safety_policy, monitor, records, frames,
                step_callback=persist_step,
            )
        except Exception as exc:
            return archive_failure("withdrawal", exc, transport_route_source)
        evidence, stage = _phase_evidence(
            bridge, controller, target_id, int(target_binding["body_id"]), predicate,
            target_binding=target_binding,
            eef_goal_reached=True,
        )
        controller.update(evidence)
        phase_events.append({"timestep": evidence.timestep, "phase": controller.phase, "native": stage["native_task_status"]})
    partial_goal_settled = False
    settle_history: list[tuple[np.ndarray, np.ndarray]] = []
    stage_action_limit = (
        (MAX_NATIVE_RELEASE_ACTIONS if resume_at_release else
         transport_action_budget + MAX_NATIVE_PLACEMENT_ACTIONS + MAX_NATIVE_RELEASE_ACTIONS)
        + MAX_NATIVE_WITHDRAWAL_ACTIONS + MAX_NATIVE_SETTLE_ACTIONS
    )
    while controller.phase == "settle" and counter[0] < stage_action_limit:
        _record_step(
            bridge, np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
            "settle", "native_goal_place", counter[0], target_binding, instruction,
            safety_policy, monitor, records, frames,
        )
        counter[0] += 1
        persist_latest_step()
        evidence, stage = _phase_evidence(
            bridge, controller, target_id, int(target_binding["body_id"]), predicate,
            target_binding=target_binding,
        )
        if evidence.task_goal_satisfied:
            settle_history.append((
                np.asarray(evidence.object_position_world_m, dtype=np.float64),
                np.asarray(evidence.object_qvel, dtype=np.float64),
            ))
            settle_history = settle_history[-5:]
        else:
            settle_history.clear()
        pending = [
            row.get("predicate") for row in stage["native_task_status"].get("native_goal_values", [])
            if row.get("value") is False and row.get("predicate") != predicate
        ]
        if pending and len(settle_history) == 5:
            positions = np.stack([item[0] for item in settle_history])
            velocities = settle_history[-1][1]
            stable_partial = bool(
                velocities.shape == (6,)
                and np.max(np.linalg.norm(positions - positions[-1], axis=1)) <= 0.004
                and np.linalg.norm(velocities[:3]) <= 0.10
                and np.linalg.norm(velocities[3:]) <= 1.0
            )
            if stable_partial:
                partial_goal_settled = True
                phase_events.append({
                    "timestep": evidence.timestep,
                    "phase": "partial_predicate_settled",
                    "native": stage["native_task_status"],
                    "pending_native_goals": pending,
                })
                break
        before_phase = controller.phase
        try:
            controller.update(evidence)
        except Exception as exc:
            return archive_failure("settle", exc, transport_route_source)
        if controller.phase != before_phase:
            phase_events.append({"timestep": evidence.timestep, "phase": controller.phase, "native": stage["native_task_status"]})
    final_native = bridge.request({"op": "native_task_status"})
    _write_recording(output_dir, records, frames)
    final_snapshot = bridge.request({"op": "export_runtime_snapshot"})
    path = output_dir / "native_goal_placement_terminal_runtime_snapshot.pkl"
    if final_snapshot.get("status") == "complete" and isinstance(final_snapshot.get("payload"), bytes):
        path.write_bytes(final_snapshot["payload"])
    return {
        "status": "complete_task_success" if final_native.get("task_success") is True
        else "partial_predicate_completed" if partial_goal_settled and any(
            row.get("predicate") == predicate and row.get("value") is True
            for row in final_native.get("native_goal_values", [])
        ) else "native_goal_placement_incomplete",
        "native_goal_predicate": predicate,
        "native_goal_value": next((row.get("value") for row in final_native.get("native_goal_values", []) if row.get("predicate") == predicate), None),
        "native_task_success": final_native.get("task_success"),
        "final_native_task_status": final_native,
        "task_phase_controller_phase": controller.phase,
        "partial_goal_settled": partial_goal_settled,
        "next_pending_native_goals": [
            row.get("predicate") for row in final_native.get("native_goal_values", [])
            if row.get("value") is False
        ],
        "phase_events": phase_events,
        "release_contact_escape_actions": release_contact_escape_actions,
        "transport_route_source": transport_route_source,
        "target_object_contact_escape": contact_escape,
        "transport_route_RRT": {
            "status": planned.get("status"),
            "failure_reason": planned.get("failure_reason"),
            "time_budget_s": 2.0,
            "maximum_configuration_checks": 800,
            "safety_certified": False,
        },
        "transport_action_budget": transport_action_budget,
        "stage_action_limit": stage_action_limit,
        "recorded_actions": len(records),
        "recorded_physics_substeps": int(sum(
            item.get("contact_monitoring", {}).get("captured_physics_substeps", 0) for item in records
        )),
        "transport_rrt": {
            "status": planned.get("status"),
            "safety_certified": False,
            "diagnostic_minimum_clearance_m": planned.get("minimum_forbidden_clearance_m"),
        },
        "terminal_snapshot": str(path) if path.is_file() else None,
        "artifacts": {
            "records": str(output_dir / "physical_step_records.jsonl.gz"),
            "npz": str(output_dir / "task_success_first_actions_states.npz"),
            "video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
            "durable_records": str(durable_records_path),
            "durable_frames": str(durable_frames_dir),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from research.gs_opd.repair.automatic_recovery_runtime import Pi05StudentPolicy
    from research.gs_opd.repair.task_success_first_grasp_search import (
        BridgeSession, _PythonObjectState, _build_safety_policy, _restore_initial,
        _snapshot_state_vector,
    )

    for dependency_path in (
        Path("/workspace/zy_workspace/LIBERO-Safety"),
        Path("/workspace/zy_workspace/LIBERO-Safety/third_party/robosuite-1.4"),
        Path("/workspace/zy_workspace/RLinf"),
    ):
        if dependency_path.is_dir() and str(dependency_path) not in sys.path:
            sys.path.insert(0, str(dependency_path))
    snapshot_root = args.snapshot_root.resolve()
    snapshot_path = snapshot_root / "snapshots/initial_runtime_snapshot.pkl"
    manifest_path = snapshot_root / "snapshots/runtime_snapshot_manifest.json"
    if not snapshot_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("source_t0_Snapshot_or_manifest_missing")
    payload = snapshot_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["snapshots"]["initial"]
    if _sha(payload) != source["payload_sha256"]:
        raise ValueError("source_t0_Snapshot_hash_mismatch")
    model_sha = str(source["model_geometry_sha256"])
    # Archived Runtime Snapshots pickle a small LIBERO state adapter under
    # __main__; expose the existing compatibility wrapper before loading it.
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    envelope = pickle.loads(payload)
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != model_sha
        or envelope.get("scene_metadata", {}).get("instruction") != manifest.get("instruction")
    ):
        raise ValueError("source_t0_runtime_snapshot_envelope_identity_mismatch")
    initial_state = _snapshot_state_vector({
        "mujoco_integration_state": envelope["snapshot"]["mujoco_state"]
    })
    if _state_digest(initial_state) != source.get("mujoco_integration_state_sha256"):
        raise ValueError("source_t0_integration_state_hash_mismatch")
    output_dir = args.output_root.resolve() / (
        f"task_success_first_executor_{args.benchmark}_L{args.level}T{args.task_id}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    bridge_args = argparse.Namespace(
        benchmark=args.benchmark, level=args.level, task_id=args.task_id,
        init_state=args.init_state, seed=args.seed,
        camera_resolution=args.camera_resolution, port=args.port,
        demonstration_horizon=args.demonstration_horizon,
    )
    bridge = BridgeSession(("127.0.0.1", args.port), output_dir / "safety_bridge.log", bridge_args)
    student: Pi05StudentPolicy | None = None
    report: dict[str, Any] = {
        "schema_version": "gs_opd_task_success_first_execution.v1",
        "status": "in_progress",
        "scope": {
            "benchmark": args.benchmark, "level": args.level, "task_id": args.task_id,
            "init_state": args.init_state, "seed": args.seed,
            "source_snapshot": str(snapshot_path), "source_snapshot_sha256": _sha(payload),
            "model_geometry_sha256": model_sha,
            "cross_scene_experience_library": str(args.experience_library) if args.experience_library else None,
        },
        "policy": {
            "student_first": True,
            "student_checkpoint": str(args.checkpoint.resolve()),
            "student_weights_only_no_LoRA": True,
            "official_safety_demonstration_data_used": False,
            "historical_student_actions_replayed": False,
            "strict_safety_reconstruction_modified": False,
            "search_traces_are_safe_training_data": False,
            "object_or_robot_state_written_directly": False,
            "all_actions_are_real_MuJoCo_steps": True,
            "collision_safety_policy": "record-only diagnostics during task-success search",
            "complete_success_predicate": "active native BDDL task_success only",
        },
        "budgets": {
            "student_actions_before_expert_fallback": args.student_action_budget,
            "student_actions_after_verified_expert_stage": args.student_handoff_action_budget,
            "grasp_geometry_candidates": args.maximum_candidates,
            "RRT_preflights": args.maximum_rrt_preflights,
            "physical_grasp_branches": args.maximum_physical_branches,
            "expected_MuJoCo_substeps_per_action": 25,
            "demonstration_horizon": args.demonstration_horizon,
            "standard_benchmark_horizon": 1000,
            "horizon_mode": "standard_benchmark" if args.demonstration_horizon == 1000 else "demonstration_only_extension",
        },
        "segments": [],
    }
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        if reset.get("scene_metadata", {}).get("model_geometry_sha256") != model_sha:
            raise RuntimeError("active_scene_model_fingerprint_differs_from_source_snapshot")
        initial = _restore_initial(bridge, payload, initial_state, model_sha)
        observed = bridge.request({"op": "observe", "include_snapshot_state": True})
        if observed.get("instruction") != manifest.get("instruction"):
            raise RuntimeError("source_Snapshot_instruction_mismatch")
        native = bridge.request({"op": "native_task_status"})
        values, success = _task_values(native)
        if success:
            raise RuntimeError("initial_t0_native_task_success_unexpectedly_true")
        # Match the batch scheduler so direct Snapshot resumes remain task
        # agnostic for multi-object scenes.
        active_target_predicate, target_selection_source, pending_goals = (
            _select_active_target_predicate(native, args.target_predicate)
        )
        report["target_goal_selection"] = {
            "predicate": active_target_predicate,
            "source": target_selection_source,
            "pending_native_object_goals": pending_goals,
        }
        geometry = bridge.request({"op": "geometry_inventory"})
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        gripper = bridge.request({"op": "gripper_geometry_inventory"})
        controller_inventory = bridge.request({"op": "repair_context"})
        context = bridge.request({
            "op": "task_success_first_context",
            "expected_instruction": observed["instruction"],
            "target_predicate": active_target_predicate,
        })
        if context.get("status") != "complete" or geometry.get("status") != "complete" or semantic.get("status") != "complete":
            raise RuntimeError("active_BDDL_target_or_geometry_binding_unavailable")
        target_id = str(context["target_object_id"])
        predicate = list(context["native_goal_predicate"])
        object_goal = len(predicate) == 3 and predicate[0].lower() in {"in", "on"}
        experience_template = None
        if object_goal and args.experience_library and args.experience_library.is_file():
            from research.gs_opd.repair.task_skill_experience import (
                load_experience_library,
                matching_grasp_template,
                matching_operation_skill,
            )
            experience_library = load_experience_library(args.experience_library)
            experience_template = matching_grasp_template(experience_library, target_id)
            experience_operation_skill = matching_operation_skill(experience_library, target_id)
        else:
            experience_operation_skill = None
        experience_first = experience_template is not None
        report["experience_transfer_decision"] = {
            "library_path": str(args.experience_library) if args.experience_library else None,
            "matching_template_id": experience_template.get("template_id") if experience_template else None,
            "matching_object_family": experience_template.get("object_family") if experience_template else None,
            "matching_operation_skill_id": (
                experience_operation_skill.get("skill_id") if experience_operation_skill else None
            ),
            "historical_operation_phase_sequence": (
                experience_operation_skill.get("phase_sequence") if experience_operation_skill else []
            ),
            "historical_operation_structure_scheduled": experience_operation_skill is not None,
            "experience_attempt_before_Student": experience_first,
            "reuse_is_not_safety_certification": True,
            "official_safety_demonstration_data_used": False,
        }
        binding = semantic["bindings"][target_id]
        target_binding = {
            "object_id": target_id, "body_id": int(binding["root_body_id"]),
            "body_name": str(binding["root_body_name"]),
            "geom_ids": sorted(int(x) for x in binding["root_geom_ids"]),
            "geom_names": list(binding["root_geom_names"]),
        }
        live_grasp_state = bridge.request({
            "op": "grasp_state", "carried_binding": target_binding,
            "expected_instruction": observed["instruction"],
        })
        live_target_grasp_verified = bool(
            object_goal
            and live_grasp_state.get("grasp_confirmed") is True
            and live_grasp_state.get("finger_contacts", {}).get("left") is True
            and live_grasp_state.get("finger_contacts", {}).get("right") is True
            and not values.get(tuple(predicate), False)
        )
        report["initial_live_target_grasp"] = {
            "verified_bilateral_grasp": live_target_grasp_verified,
            "grasp_state_status": live_grasp_state.get("status"),
            "target_finger_contacts": live_grasp_state.get("finger_contacts"),
            "resume_source": "model_verified_initial_Runtime_Snapshot" if live_target_grasp_verified else None,
        }
        policy = _build_safety_policy(context, semantic, geometry, gripper)
        initial_export = bridge.request({"op": "export_runtime_snapshot"})
        if initial_export.get("status") != "complete" or not isinstance(initial_export.get("payload"), bytes):
            raise RuntimeError("task_success_first_initial_snapshot_export_failed")
        start_payload = initial_export["payload"]
        start_state = _snapshot_state_vector(initial_export["snapshot_state"])
        if not np.array_equal(start_state, initial_state):
            raise RuntimeError("restored_t0_state_does_not_match_source_integration_state")
        controller = (
            TaskPhaseController(_controller_config(target_id, predicate, {"candidate_id": "not_yet_generated"}))
            if object_goal else None
        )
        report["task"] = {
            "instruction": observed["instruction"],
            "target_object_from_active_BDDL": target_id,
            "native_goal_predicate": predicate,
            "initial_native_task_status": native,
            "semantic_binding": binding,
            "model_geometry_sha256": model_sha,
            "BDDL_sha256": observed.get("scene_metadata", {}).get("bddl_sha256"),
            "task_phase_controller_initial_phase": controller.phase if controller is not None else None,
            "historical_operation_phase_sequence": (
                experience_operation_skill.get("phase_sequence") if experience_operation_skill else []
            ),
        }
        if args.resume_from_grasp_report is not None:
            source_report = json.loads(
                args.resume_from_grasp_report.read_text(encoding="utf-8")
            )
            resume = _verified_grasp_resume_descriptor(
                source_report,
                benchmark=args.benchmark,
                level=args.level,
                task_id=args.task_id,
                seed=args.seed,
                init_state=args.init_state,
                model_sha=model_sha,
                bddl_sha=str(observed.get("scene_metadata", {}).get("bddl_sha256")),
                target_id=target_id,
                predicate=predicate,
            )
            restored = bridge.request({
                "op": "restore_runtime_snapshot",
                "payload": resume["path"].read_bytes(),
            })
            actual_state = _integration_state(restored.get("snapshot_state", {}))
            expected_state = _integration_state(resume["snapshot_state"])
            if (
                restored.get("status") != "restored"
                or restored.get("model_geometry_sha256") != model_sha
                or actual_state.shape != expected_state.shape
                or not np.array_equal(actual_state, expected_state)
            ):
                raise RuntimeError("resume_grasp_snapshot_exact_restore_verification_failed")
            report["resumed_from_grasp_skill"] = {
                "source_report": str(args.resume_from_grasp_report),
                "snapshot": str(resume["path"]),
                "status": "verified_exact",
                "integration_state_sha256": _state_digest(actual_state),
                "model_geometry_sha256": model_sha,
                "source_task_success": False,
                "safety_certified": False,
            }
            resumed_native = bridge.request({"op": "native_task_status"})
            _, resumed_task_success = _task_values(resumed_native)
            if resumed_task_success:
                report["status"] = "complete_task_success"
                report["final_native_task_status"] = resumed_native
                report["archive"] = str(output_dir)
                _json(output_dir / "run_report.json", report)
                return report
            resumed_grasp = bridge.request({
                "op": "grasp_state",
                "carried_binding": target_binding,
                "expected_instruction": observed["instruction"],
            })
            if (
                resumed_grasp.get("grasp_confirmed") is not True
                or resumed_grasp.get("finger_contacts", {}).get("left") is not True
                or resumed_grasp.get("finger_contacts", {}).get("right") is not True
            ):
                raise RuntimeError("resume_grasp_snapshot_did_not_retain_stable_bilateral_contact")
            controller = TaskPhaseController(_controller_config(target_id, predicate, resume["success"]))
            controller.adopt_held_object(target_id)
            place_result = _native_place(
                bridge, target_id=target_id, target_binding=target_binding,
                predicate=predicate, instruction=observed["instruction"], model_sha=model_sha,
                safety_policy=policy, controller_fields=controller_inventory["robots"][0]["fields"],
                controller=controller, output_dir=output_dir / "expert_native_placement",
            )
            report["segments"].append({"type": "resumed_expert_native_goal_stage_control", **place_result})
            report["final_native_task_status"] = place_result["final_native_task_status"]
            report["task_success"] = report["final_native_task_status"].get("task_success") is True
            report["status"] = (
                "complete_task_success" if report["task_success"] else place_result["status"]
            )
            learned_operation = _register_verified_operation_skill_from_report(
                args.experience_library, report, output_dir / "run_report.json"
            )
            if learned_operation is not None:
                report["operation_experience_learning"] = learned_operation
            report["archive"] = str(output_dir)
            _json(output_dir / "run_report.json", report)
            return report
        if args.resume_from_placement_report is not None:
            source_report = json.loads(
                args.resume_from_placement_report.read_text(encoding="utf-8")
            )
            resume = _verified_placement_resume_descriptor(
                source_report,
                benchmark=args.benchmark,
                level=args.level,
                task_id=args.task_id,
                seed=args.seed,
                init_state=args.init_state,
                model_sha=model_sha,
                bddl_sha=str(observed.get("scene_metadata", {}).get("bddl_sha256")),
                target_id=target_id,
                predicate=predicate,
            )
            restored = bridge.request({
                "op": "restore_runtime_snapshot",
                "payload": resume["path"].read_bytes(),
            })
            actual_state = _integration_state(restored.get("snapshot_state", {}))
            expected_state = _integration_state(resume["snapshot_state"])
            if (
                restored.get("status") != "restored"
                or restored.get("model_geometry_sha256") != model_sha
                or actual_state.shape != expected_state.shape
                or not np.array_equal(actual_state, expected_state)
            ):
                raise RuntimeError("placement_failure_snapshot_exact_restore_verification_failed")
            resumed_native = bridge.request({"op": "native_task_status"})
            resumed_values, resumed_task_success = _task_values(resumed_native)
            if resumed_task_success or resumed_values.get(tuple(predicate)) is not True:
                raise RuntimeError("placement_failure_snapshot_native_goal_evidence_changed")
            resumed_grasp = bridge.request({
                "op": "grasp_state",
                "carried_binding": target_binding,
                "expected_instruction": observed["instruction"],
            })
            if (
                resumed_grasp.get("grasp_confirmed") is not True
                or resumed_grasp.get("finger_contacts", {}).get("left") is not True
                or resumed_grasp.get("finger_contacts", {}).get("right") is not True
            ):
                raise RuntimeError("placement_failure_snapshot_no_longer_has_bilateral_grasp")
            satisfied_objects = tuple(
                str(row["predicate"][1])
                for row in resumed_native.get("native_goal_values", [])
                if isinstance(row.get("predicate"), list)
                and len(row["predicate"]) == 3
                and str(row["predicate"][0]).lower() in {"in", "on"}
                and row.get("value") is True
            )
            controller = TaskPhaseController(
                _controller_config(target_id, predicate, {"candidate_id": "verified_native_goal_release_resume"})
            )
            controller.adopt_native_goal_contact_for_release(
                target_id,
                native_goal_satisfied=True,
                satisfied_object_ids=satisfied_objects,
                held_object=target_id,
                grasp_confirmed=True,
            )
            report["resumed_from_native_goal_contact_snapshot"] = {
                "source_report": str(args.resume_from_placement_report),
                "snapshot": str(resume["path"]),
                "snapshot_sha256": resume["payload_sha256"],
                "status": "verified_exact",
                "integration_state_sha256": _state_digest(actual_state),
                "model_geometry_sha256": model_sha,
                "native_goal_already_satisfied": True,
                "bilateral_grasp_retained": True,
                "next_phase": controller.phase,
                "safety_certified": False,
            }
            place_result = _native_place(
                bridge, target_id=target_id, target_binding=target_binding,
                predicate=predicate, instruction=observed["instruction"], model_sha=model_sha,
                safety_policy=policy, controller_fields=controller_inventory["robots"][0]["fields"],
                controller=controller, output_dir=output_dir / "expert_native_release_resume",
            )
            report["segments"].append({"type": "resumed_expert_release_withdrawal", **place_result})
            report["final_native_task_status"] = place_result["final_native_task_status"]
            report["task_success"] = report["final_native_task_status"].get("task_success") is True
            report["status"] = (
                "complete_task_success" if report["task_success"] else place_result["status"]
            )
            learned_operation = _register_verified_operation_skill_from_report(
                args.experience_library, report, output_dir / "run_report.json"
            )
            if learned_operation is not None:
                report["operation_experience_learning"] = learned_operation
            report["archive"] = str(output_dir)
            _json(output_dir / "run_report.json", report)
            return report
        if live_target_grasp_verified:
            student_first = {
                "task_success": False,
                "actions_executed": 0,
                "target_displacement_m": None,
                "grasp_stability": {"stable": True, "source": "initial_Runtime_Snapshot"},
                "native_task_status": native,
                "skipped_reason": "verified_live_target_grasp_resumed_without_regrasp_or_Student_actions",
            }
            report["segments"].append({"type": "verified_live_grasp_snapshot_resume", **student_first})
        elif experience_first:
            student_first = {
                "task_success": False,
                "actions_executed": 0,
                "target_displacement_m": None,
                "grasp_stability": {},
                "native_task_status": native,
                "skipped_reason": "matching_GS_OPD_object_relative_success_skill_selected_before_Student",
                "experience_template_id": experience_template["template_id"],
            }
            report["segments"].append({"type": "student_first_skipped_transferable_experience", **student_first})
        else:
            student = Pi05StudentPolicy(args.checkpoint, device=args.device)
            student_first = _student_segment(
                bridge, student, max_actions=args.student_action_budget,
                target_binding=target_binding, instruction=observed["instruction"], policy=policy,
                output_dir=output_dir / "student_first", segment="first",
            )
            report["segments"].append({"type": "student_first", **student_first})
        if not object_goal:
            if student_first["task_success"]:
                report["status"] = "complete_task_success"
                report["final_native_task_status"] = student_first["native_task_status"]
            else:
                _restore_initial(bridge, start_payload, start_state, model_sha)
                articulation_context = bridge.request({
                    "op": "task_success_first_articulation_context",
                    "expected_instruction": observed["instruction"],
                    "predicate": predicate,
                })
                articulated = _run_articulated_fixture_goal(
                    bridge, predicate=predicate, target_binding=target_binding,
                    instruction=observed["instruction"], policy=policy,
                    articulation_context=articulation_context,
                    controller_fields=controller_inventory["robots"][0]["fields"],
                    output_dir=output_dir / "expert_articulated_fixture_goal",
                )
                report["segments"].append({"type": "expert_articulated_fixture_goal", **articulated})
                report["status"] = articulated["status"]
                report["final_native_task_status"] = articulated["final_native_task_status"]
                report["fixture_goal_failure_reason"] = (
                    None if articulated["native_goal_value"] is True
                    else "bounded_physical_OSC_fixture_push_did_not_satisfy_native_goal"
                )
            report["archive"] = str(output_dir)
            _json(output_dir / "run_report.json", report)
            return report
        if live_target_grasp_verified and not student_first["task_success"]:
            controller = TaskPhaseController(
                _controller_config(target_id, predicate, {"candidate_id": "verified_live_grasp_snapshot"})
            )
            controller.adopt_held_object(target_id)
            place_result = _native_place(
                bridge, target_id=target_id, target_binding=target_binding,
                predicate=predicate, instruction=observed["instruction"], model_sha=model_sha,
                safety_policy=policy, controller_fields=controller_inventory["robots"][0]["fields"],
                controller=controller, output_dir=output_dir / "expert_native_goal_stage_control",
            )
            report["segments"].append({"type": "expert_native_goal_stage_control", **place_result})
            report["final_native_task_status"] = place_result["final_native_task_status"]
            report["task_success"] = place_result["final_native_task_status"].get("task_success") is True
            report["status"] = "complete_task_success" if report["task_success"] else place_result["status"]
            learned_operation = _register_verified_operation_skill_from_report(
                args.experience_library, report, output_dir / "run_report.json"
            )
            if learned_operation is not None:
                report["operation_experience_learning"] = learned_operation
            report["archive"] = str(output_dir)
            _json(output_dir / "run_report.json", report)
            return report
        if student_first["task_success"]:
            report["status"] = "complete_task_success"
            report["final_native_task_status"] = student_first["native_task_status"]
        else:
            _restore_initial(bridge, start_payload, start_state, model_sha)
            report["student_failure_trigger"] = {
                "triggered": True,
                "reason": (
                    "matching_successful_object_relative_skill_requires_physical_transfer_validation"
                    if experience_first else "native_BDDL_task_success_not_reached_within_student_action_budget"
                ),
                "target_displacement_m": student_first["target_displacement_m"],
                "grasp_stable": student_first["grasp_stability"].get("stable"),
                "exact_t0_snapshot_restored": True,
            }
            controller.handoff_after_pregrasp()
            report["task_phase_controller_student_to_expert_handoff"] = controller.phase
            bridge.close()
            expert_root = output_dir / "expert_grasp_search"
            command = [
                str(LIBERO_PYTHON), "-m", "research.gs_opd.repair.task_success_first_grasp_search",
                "--snapshot-root", str(snapshot_root), "--output-root", str(expert_root),
                "--benchmark", args.benchmark, "--level", str(args.level),
                "--task-id", str(args.task_id), "--init-state", str(args.init_state),
                "--seed", str(args.seed), "--camera-resolution", str(args.camera_resolution),
                "--port", str(args.expert_port),
                "--maximum-candidates", str(args.maximum_candidates),
                "--maximum-rrt-preflights", str(args.maximum_rrt_preflights),
                "--maximum-physical-branches", str(args.maximum_physical_branches),
                "--rrt-time-budget-s", str(args.rrt_time_budget_s),
                "--demonstration-horizon", str(args.demonstration_horizon),
                "--target-predicate", *predicate,
            ]
            if args.experience_library is not None:
                command.extend(["--experience-library", str(args.experience_library)])
            previous_search_report = getattr(args, "previous_search_report", None)
            if previous_search_report is not None:
                previous_reports = (
                    previous_search_report
                    if isinstance(previous_search_report, (list, tuple))
                    else [previous_search_report]
                )
                for previous_report in previous_reports:
                    command.extend(["--previous-search-report", str(Path(previous_report).resolve())])
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
            with (output_dir / "expert_grasp_search.log").open("wb") as log:
                process = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            expert_runs = sorted(expert_root.glob("task_success_first_grasp_*"))
            if not expert_runs:
                raise RuntimeError(f"expert_grasp_search_produced_no_archive:exit={process.returncode}")
            expert_report_path = expert_runs[-1] / "run_report.json"
            expert_report = json.loads(expert_report_path.read_text(encoding="utf-8"))
            expert_failure_reasons = [
                str(value) for value in (
                    [expert_report.get("error")]
                    + [item.get("rrt_failure_reason") for item in expert_report.get("candidates", [])]
                    + [item.get("failure_reason") for item in expert_report.get("physical_branches", [])]
                ) if value
            ]
            report["segments"].append({
                "type": "bounded_expert_grasp_search",
                "status": expert_report.get("status"),
                "archive": str(expert_runs[-1]),
                "report": str(expert_report_path),
                "process_exit_code": process.returncode,
                "physical_branches": expert_report.get("physical_branches", []),
                "candidate_count": len(expert_report.get("candidates", [])),
                "failure_reasons": expert_failure_reasons,
                "success": expert_report.get("success"),
                "safety_certified": False,
            })
            if expert_report.get("success") is None:
                report["status"] = "bounded_expert_search_exhausted"
                report["failure_reason"] = (
                    expert_report.get("failure_classification")
                    or (expert_failure_reasons[0] if expert_failure_reasons else None)
                )
            else:
                success_snapshot = expert_report["success"].get("snapshot", {}).get("path")
                if not success_snapshot or not Path(success_snapshot).is_file():
                    raise RuntimeError("expert_stage_success_has_no_recoverable_runtime_snapshot")
                bridge = BridgeSession(("127.0.0.1", args.expert_port + 1), output_dir / "student_handoff_bridge.log", bridge_args)
                bridge.start()
                reset = bridge.request({"op": "reset"})
                if reset.get("scene_metadata", {}).get("model_geometry_sha256") != model_sha:
                    raise RuntimeError("student_handoff_scene_fingerprint_mismatch")
                restored = bridge.request({
                    "op": "restore_runtime_snapshot",
                    "payload": Path(success_snapshot).read_bytes(),
                })
                expected = expert_report["success"]["snapshot"].get("snapshot_state", {})
                actual_state = _integration_state(restored.get("snapshot_state", {}))
                expected_state = _integration_state(expected)
                if (
                    restored.get("status") != "restored"
                    or restored.get("model_geometry_sha256") != model_sha
                    or actual_state.shape != expected_state.shape
                    or not np.array_equal(actual_state, expected_state)
                ):
                    raise RuntimeError("expert_stage_runtime_snapshot_exact_restore_verification_failed")
                report["student_handoff_snapshot_restore"] = {
                    "status": "verified_exact",
                    "path": success_snapshot,
                    "mujoco_integration_state_sha256": _state_digest(actual_state),
                    "model_geometry_sha256": model_sha,
                }
                bridge.request({"op": "observe", "include_snapshot_state": True})
                native = bridge.request({"op": "native_task_status"})
                _, success = _task_values(native)
                if not success and not experience_first:
                    handoff = _student_segment(
                        bridge, student, max_actions=args.student_handoff_action_budget,
                        target_binding=target_binding, instruction=observed["instruction"], policy=policy,
                        output_dir=output_dir / "student_handoff", segment="after_expert_grasp_lift",
                    )
                    report["segments"].append({"type": "student_handoff_after_stage_goal", **handoff})
                    native = bridge.request({"op": "native_task_status"})
                    _, success = _task_values(native)
                elif not success:
                    report["student_handoff_skipped"] = (
                        "matched_successful_skill_transfer_keeps_expert_OSC_in_control_until_native_goal"
                    )
                if success:
                    report["status"] = "complete_task_success"
                    report["final_native_task_status"] = native
                else:
                    live_grasp = bridge.request({
                        "op": "grasp_state",
                        "carried_binding": target_binding,
                        "expected_instruction": observed["instruction"],
                    })
                    if live_grasp.get("grasp_confirmed") is not True:
                        restored = bridge.request({
                            "op": "restore_runtime_snapshot",
                            "payload": Path(success_snapshot).read_bytes(),
                        })
                        restored_state = _integration_state(restored.get("snapshot_state", {}))
                        expected_state = _integration_state(
                            expert_report["success"]["snapshot"].get("snapshot_state", {})
                        )
                        if (
                            restored.get("status") != "restored"
                            or restored.get("model_geometry_sha256") != model_sha
                            or restored_state.shape != expected_state.shape
                            or not np.array_equal(restored_state, expected_state)
                        ):
                            raise RuntimeError("failed_student_handoff_did_not_restore_verified_grasp_snapshot")
                        report["student_handoff_recovery"] = {
                            "status": "exact_expert_grasp_snapshot_restored_after_Student_lost_grasp",
                            "integration_state_sha256": _state_digest(restored_state),
                        }
                        live_grasp = bridge.request({
                            "op": "grasp_state",
                            "carried_binding": target_binding,
                            "expected_instruction": observed["instruction"],
                        })
                        if live_grasp.get("grasp_confirmed") is not True:
                            raise RuntimeError("restored_expert_snapshot_no_longer_has_verified_grasp")
                    controller = TaskPhaseController(_controller_config(target_id, predicate, expert_report["success"]))
                    controller.adopt_held_object(target_id)
                    report["student_handoff_failure_trigger"] = "native_BDDL_goal_not_reached_after_verified_grasp_lift_stage"
                    place_result = _native_place(
                        bridge, target_id=target_id, target_binding=target_binding,
                        predicate=predicate, instruction=observed["instruction"], model_sha=model_sha,
                        safety_policy=policy, controller_fields=controller_inventory["robots"][0]["fields"],
                        controller=controller, output_dir=output_dir / "expert_native_placement",
                    )
                    report["segments"].append({"type": "expert_native_goal_stage_control", **place_result})
                    report["final_native_task_status"] = place_result["final_native_task_status"]
                    report["task_success"] = report["final_native_task_status"].get("task_success") is True
                    report["status"] = (
                        "complete_task_success" if report["task_success"] else place_result["status"]
                    )
                    learned_operation = _register_verified_operation_skill_from_report(
                        args.experience_library, report, output_dir / "run_report.json"
                    )
                    if learned_operation is not None:
                        report["operation_experience_learning"] = learned_operation
        report["archive"] = str(output_dir)
        _json(output_dir / "run_report.json", report)
        return report
    except Exception as exc:
        report["status"] = "executor_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["archive"] = str(output_dir)
        _json(output_dir / "run_report.json", report)
        raise
    finally:
        bridge.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark", default="obstacle_avoidance")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--target-predicate", nargs="+", default=None)
    parser.add_argument("--resume-from-grasp-report", type=Path, default=None)
    parser.add_argument("--resume-from-placement-report", type=Path, default=None)
    parser.add_argument("--previous-search-report", type=Path, action="append", default=[])
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--demonstration-horizon", type=int, default=1000)
    parser.add_argument("--port", type=int, default=38569)
    parser.add_argument("--expert-port", type=int, default=38570)
    parser.add_argument("--student-action-budget", type=int, default=32)
    parser.add_argument("--student-handoff-action-budget", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=16)
    parser.add_argument("--maximum-rrt-preflights", type=int, default=8)
    parser.add_argument("--maximum-physical-branches", type=int, default=3)
    parser.add_argument("--rrt-time-budget-s", type=float, default=20.0)
    parser.add_argument("--experience-library", type=Path, default=None)
    args = parser.parse_args()
    if args.resume_from_grasp_report is not None and args.resume_from_placement_report is not None:
        parser.error("choose only one Runtime Snapshot resume source")
    for name in ("student_action_budget", "student_handoff_action_budget"):
        if not 1 <= getattr(args, name) <= 256:
            parser.error(f"--{name.replace('_', '-')} must be in 1..256")
    if not 1 <= args.maximum_candidates <= 16:
        parser.error("--maximum-candidates must be in 1..16")
    if not 1 <= args.maximum_rrt_preflights <= 8:
        parser.error("--maximum-rrt-preflights must be in 1..8")
    if not 1 <= args.maximum_physical_branches <= 3:
        parser.error("--maximum-physical-branches must be in 1..3")
    if not 0.5 <= args.rrt_time_budget_s <= 60:
        parser.error("--rrt-time-budget-s must be in 0.5..60")
    if not 1000 <= args.demonstration_horizon <= 1800:
        parser.error("--demonstration-horizon must be in 1000..1800")
    if args.target_predicate is not None and len(args.target_predicate) not in (2, 3):
        parser.error("--target-predicate must contain two or three BDDL tokens")
    return args


def main() -> int:
    result = run(_parse_args())
    print(json.dumps({
        "status": result.get("status"),
        "archive": result.get("archive"),
        "task": result.get("task"),
        "segments": [
            {key: row.get(key) for key in ("type", "status", "actions_executed", "task_success", "archive", "recorded_actions") if key in row}
            for row in result.get("segments", [])
        ],
        "failure_reason": result.get("failure_reason") or result.get("error"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("status") == "complete_task_success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
