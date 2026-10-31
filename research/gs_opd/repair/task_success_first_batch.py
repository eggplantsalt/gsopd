# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run bounded Task-Success-First stages from a local task manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.repair.task_success_first_search import (
    native_goal_counts,
    pending_object_goals,
    progress_label,
)
from research.gs_opd.repair.task_skill_experience import DEFAULT_OUTPUT as DEFAULT_EXPERIENCE_LIBRARY


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "docs/GS_OPD_LIBERO_SAFETY_TASK_MANIFEST.json"
DEFAULT_OUTPUT = Path("/opt/data/private/gs_opd/rollouts/task_success_first_l0_batch")
DEFAULT_CHECKPOINT = Path("/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft")
L0T2_HISTORICAL_POT2_SNAPSHOT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_execution_aware_pot2_release_continue_t642_20261025/"
    "snapshots/main_timestep_000649.pkl"
)
BRIDGE_AUTHKEY = b"gs-opd-local"
TASK_ORDER = (0, 1, 3, 4, 2)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _snapshot_root_from_payload(
    payload: bytes,
    root: Path,
    *,
    task_bddl: Path | None = None,
    source: str = "current_seed_init_state_or_verified_prior_task_stage_snapshot",
) -> Path:
    from research.gs_opd.repair.task_success_first_grasp_search import _PythonObjectState

    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    envelope = pickle.loads(payload)
    if envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
        raise ValueError("Runtime Snapshot schema is unsupported")
    scene = envelope.get("scene_metadata", {})
    snapshot = envelope.get("snapshot", {})
    integration = snapshot.get("mujoco_state")
    if integration is None:
        raise ValueError("Runtime Snapshot has no MuJoCo integration state")
    state_bytes = __import__("numpy").ascontiguousarray(integration, dtype="float64").tobytes()
    root.mkdir(parents=True, exist_ok=False)
    snapshots = root / "snapshots"
    snapshots.mkdir()
    payload_path = snapshots / "initial_runtime_snapshot.pkl"
    payload_path.write_bytes(payload)
    source_bddl = task_bddl or Path(scene["bddl_path"])
    if not source_bddl.is_file():
        raise FileNotFoundError(f"active_BDDL_file_is_missing:{source_bddl}")
    (root / "task.bddl").write_bytes(source_bddl.read_bytes())
    manifest = {
        "schema_version": "gs_opd_runtime_snapshot_manifest.v1",
        "instruction": scene.get("instruction"),
        "scene_metadata": scene,
        "snapshots": {
            "initial": {
                "payload_sha256": _sha(payload),
                "model_geometry_sha256": envelope.get("model_geometry_sha256"),
                "mujoco_integration_state_sha256": hashlib.sha256(state_bytes).hexdigest(),
            }
        },
        "source": source,
    }
    _write_json(snapshots / "runtime_snapshot_manifest.json", manifest)
    _write_json(root / "snapshot_source.json", {
        "scene_metadata": scene,
        "payload_sha256": _sha(payload),
        "model_geometry_sha256": envelope.get("model_geometry_sha256"),
        "integration_state_sha256": hashlib.sha256(state_bytes).hexdigest(),
    })
    return root


def _verify_existing_t0_snapshot(root: Path, current_payload: bytes) -> None:
    from research.gs_opd.repair.task_success_first_grasp_search import _PythonObjectState

    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    previous = pickle.loads((root / "snapshots/initial_runtime_snapshot.pkl").read_bytes())
    current = pickle.loads(current_payload)
    previous_state = np.asarray(previous["snapshot"]["mujoco_state"], dtype=np.float64)
    current_state = np.asarray(current["snapshot"]["mujoco_state"], dtype=np.float64)
    if (
        previous.get("model_geometry_sha256") != current.get("model_geometry_sha256")
        or previous_state.shape != current_state.shape
        or not np.array_equal(previous_state, current_state)
    ):
        raise RuntimeError("existing_t0_snapshot_does_not_match_fresh_seed_init_state")


def _capture_task_initial(args: argparse.Namespace, task_dir: Path, task_id: int) -> tuple[Path, dict[str, Any]]:
    from research.gs_opd.repair.task_success_first_grasp_search import BridgeSession

    if getattr(args, "no_prior_experience", False) and args.resume_from_runtime_snapshot is not None:
        raise ValueError("no_prior_experience_forbids_partial_task_snapshot_resume")
    source_root = task_dir / "stage_snapshots" / "t0"
    bridge_args = argparse.Namespace(
        benchmark=args.suite, level=args.level, task_id=task_id,
        init_state=args.init_state, seed=args.seed,
        camera_resolution=args.camera_resolution,
        demonstration_horizon=args.demonstration_horizon,
    )
    port = args.base_port + task_id * 8
    bridge = BridgeSession(("127.0.0.1", port), task_dir / "preflight_bridge.log", bridge_args)
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        reset_metadata = reset.get("scene_metadata", {})
        resume_source = None
        initial_snapshot_payload = None
        if args.resume_from_runtime_snapshot is not None:
            from research.gs_opd.repair.task_success_first_grasp_search import _PythonObjectState

            setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
            resume_source = Path(args.resume_from_runtime_snapshot).resolve()
            if not resume_source.is_file():
                raise FileNotFoundError(f"resume_Runtime_Snapshot_missing:{resume_source}")
            resume_payload = resume_source.read_bytes()
            envelope = pickle.loads(resume_payload)
            scene = envelope.get("scene_metadata", {})
            expected_scope = {
                "benchmark": args.suite,
                "level": args.level,
                "task_id": task_id,
                "environment_seed": args.seed,
                "init_state_index": args.init_state,
            }
            if envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
                raise ValueError("resume_Runtime_Snapshot_schema_mismatch")
            for key, value in expected_scope.items():
                if scene.get(key) != value:
                    raise ValueError(f"resume_Runtime_Snapshot_{key}_mismatch")
            expected_bddl = args.expected_bddl_hashes.get(str(task_id))
            if expected_bddl and scene.get("bddl_sha256") != expected_bddl:
                raise ValueError("resume_Runtime_Snapshot_BDDL_fingerprint_mismatch")
            if scene.get("instruction") != reset_metadata.get("instruction"):
                raise ValueError("resume_Runtime_Snapshot_instruction_mismatch")
            if scene.get("bddl_sha256") != reset_metadata.get("bddl_sha256"):
                raise ValueError("resume_Runtime_Snapshot_live_BDDL_mismatch")
            baseline = bridge.request({"op": "export_runtime_snapshot"})
            if (
                baseline.get("status") != "complete"
                or baseline.get("model_geometry_sha256") != envelope.get("model_geometry_sha256")
            ):
                raise ValueError("resume_Runtime_Snapshot_model_fingerprint_mismatch")
            integration = np.asarray(
                envelope.get("snapshot", {}).get("mujoco_state"), dtype=np.float64,
            )
            if integration.ndim != 1 or not len(integration) or not np.isfinite(integration).all():
                raise ValueError("resume_Runtime_Snapshot_integration_state_invalid")
            restored = bridge.request({"op": "restore_runtime_snapshot", "payload": resume_payload})
            restored_state = np.asarray(
                restored.get("snapshot_state", {}).get("mujoco_integration_state", []),
                dtype=np.float64,
            )
            if (
                restored.get("status") != "restored"
                or restored.get("model_geometry_sha256") != envelope.get("model_geometry_sha256")
                or restored_state.shape != integration.shape
                or not np.array_equal(restored_state, integration)
            ):
                raise RuntimeError("resume_Runtime_Snapshot_exact_restore_verification_failed")
            initial_snapshot_payload = resume_payload
        observed = bridge.request({"op": "observe", "include_snapshot_state": True})
        native = bridge.request({"op": "native_task_status"})
        object_goals = pending_object_goals(native)
        object_goal_context = None
        if object_goals:
            object_goal_context = bridge.request({
                "op": "task_success_first_context",
                "expected_instruction": observed.get("instruction"),
                "target_predicate": object_goals[0],
            })
        articulation_contexts = []
        for row in native.get("native_goal_values", []):
            predicate = row.get("predicate")
            if (
                row.get("value") is False
                and isinstance(predicate, list)
                and len(predicate) == 2
                and str(predicate[0]).lower() in {"close", "open", "turnon", "turnoff"}
            ):
                articulation_contexts.append(bridge.request({
                    "op": "task_success_first_articulation_context",
                    "expected_instruction": observed.get("instruction"),
                    "predicate": predicate,
                }))
        exported = bridge.request({"op": "export_runtime_snapshot"})
        if reset.get("scene_metadata", {}).get("init_state_index") != args.init_state:
            raise RuntimeError("preflight_init_state_was_not_the_requested_snapshot")
        if exported.get("status") != "complete" or not isinstance(exported.get("payload"), bytes):
            raise RuntimeError("preflight_Runtime_Snapshot_export_failed")
        if initial_snapshot_payload is None:
            initial_snapshot_payload = exported["payload"]
        else:
            restored_state = np.asarray(
                exported.get("snapshot_state", {}).get("mujoco_integration_state", []),
                dtype=np.float64,
            )
            source_envelope = pickle.loads(initial_snapshot_payload)
            source_state = np.asarray(source_envelope["snapshot"]["mujoco_state"], dtype=np.float64)
            if restored_state.shape != source_state.shape or not np.array_equal(restored_state, source_state):
                raise RuntimeError("resume_Runtime_Snapshot_changed_during_read_only_preflight")
        if source_root.exists():
            _verify_existing_t0_snapshot(source_root, initial_snapshot_payload)
        else:
            _snapshot_root_from_payload(
                initial_snapshot_payload, source_root,
                source=("fresh_reset_no_prior_experience" if getattr(args, "no_prior_experience", False) else None)
                or ("verified_failed_stage_Runtime_Snapshot_resume" if resume_source else None)
                or "current_seed_init_state_or_verified_prior_task_stage_snapshot",
            )
        expected = args.expected_bddl_hashes.get(str(task_id))
        actual = reset.get("scene_metadata", {}).get("bddl_sha256")
        if expected and actual != expected:
            raise RuntimeError(f"manifest_BDDL_hash_mismatch:{expected}:{actual}")
        if observed.get("instruction") != reset.get("scene_metadata", {}).get("instruction"):
            raise RuntimeError("reset_instruction_and_live_observation_do_not_match")
        historical_regression = None
        # The legacy task-2 diagnostic restores a historical partial-task
        # state. Explicit no-prior callers must never even read that archive.
        if (not getattr(args, "no_prior_experience", False)
                and args.suite == "obstacle_avoidance" and args.level == 0 and task_id == 2):
            historical_regression = {
                "source_snapshot": str(L0T2_HISTORICAL_POT2_SNAPSHOT),
                "source": "GS-OPD-owned physical Pot-2 placement subgoal snapshot",
                "action_trajectory_replayed": False,
                "full_task_success_trajectory_replayed": False,
                "official_safety_demonstration_used": False,
            }
            if L0T2_HISTORICAL_POT2_SNAPSHOT.is_file():
                try:
                    historical_payload = L0T2_HISTORICAL_POT2_SNAPSHOT.read_bytes()
                    historical_restore = bridge.request({
                        "op": "restore_runtime_snapshot", "payload": historical_payload,
                    })
                    historical_regression["restore_status"] = historical_restore.get("status")
                    if historical_restore.get("status") == "restored":
                        historical_status = bridge.request({"op": "native_task_status"})
                        historical_regression["native_task_status"] = historical_status
                        history_values = {
                            tuple(row.get("predicate", [])): row.get("value")
                            for row in historical_status.get("native_goal_values", [])
                        }
                        historical_regression["pot2_on_regression_pass"] = any(
                            value is True and len(predicate) == 3
                            and str(predicate[0]).lower() == "on"
                            and str(predicate[1]).lower() == "moka_pot_2"
                            for predicate, value in history_values.items()
                        )
                        historical_regression["full_task_success"] = historical_status.get("task_success")
                except Exception as exc:
                    historical_regression["regression_error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    initial_restore = bridge.request({
                        "op": "restore_runtime_snapshot",
                        "payload": (source_root / "snapshots/initial_runtime_snapshot.pkl").read_bytes(),
                    })
                    current_state = np.asarray(
                        initial_restore.get("snapshot_state", {}).get("mujoco_integration_state", []),
                        dtype=np.float64,
                    )
                    expected_state = np.asarray(
                        exported.get("snapshot_state", {}).get("mujoco_integration_state", []),
                        dtype=np.float64,
                    )
                    historical_regression["initial_t0_restored_exactly_after_regression"] = bool(
                        initial_restore.get("status") == "restored"
                        and current_state.shape == expected_state.shape
                        and np.array_equal(current_state, expected_state)
                    )
                    if not historical_regression["initial_t0_restored_exactly_after_regression"]:
                        raise RuntimeError("historical_L0T2_snapshot_regression_did_not_restore_exact_t0")
            else:
                historical_regression["restore_status"] = "historical_snapshot_not_found"
        preflight = {
            "scene_metadata": reset.get("scene_metadata"),
            "instruction": observed.get("instruction"),
            "initial_native_task_status": native,
            "initial_goal_counts": native_goal_counts(native),
            "first_pending_object_goal": object_goals[0] if object_goals else None,
            "object_goal_context_preflight": object_goal_context,
            "initial_articulation_contexts": articulation_contexts,
            "historical_GS_OPD_regression": historical_regression,
            "model_geometry_sha256": exported.get("model_geometry_sha256"),
            "runtime_snapshot": str(source_root / "snapshots/initial_runtime_snapshot.pkl"),
            "layout_observation_timestep": observed.get("timestep"),
            "actual_scene_layout_verified_from_runtime": True,
            "actual_object_layout": {
                str(name): {
                    "body_id": int(value.get("body_id", -1)),
                    "body_name": value.get("body_name"),
                    "root_position_world_m": value.get("root_pos"),
                    "root_quaternion_wxyz": value.get("root_quat"),
                    "joint_qvel": value.get("joint_qvel"),
                }
                for name, value in observed.get("snapshot_state", {}).get("object_states", {}).items()
            },
        }
        if resume_source is not None:
            preflight["resumed_from_runtime_snapshot"] = {
                "path": str(resume_source),
                "payload_sha256": _sha(initial_snapshot_payload),
                "integration_state_sha256": hashlib.sha256(
                    np.ascontiguousarray(source_state, dtype=np.float64).tobytes()
                ).hexdigest(),
                "model_geometry_sha256": exported.get("model_geometry_sha256"),
                "bddl_sha256": reset_metadata.get("bddl_sha256"),
                "exact_restore_verified": True,
                "no_robot_or_object_state_write": True,
                "safety_certified": False,
            }
        return source_root, preflight
    finally:
        bridge.close()


def _next_goal(native: dict[str, Any]) -> list[str] | None:
    object_goals = pending_object_goals(native)
    if object_goals:
        return object_goals[0]
    values = native.get("native_goal_values", [])
    return next((list(row["predicate"]) for row in values
                 if row.get("value") is False
                 and isinstance(row.get("predicate"), list)
                 and len(row["predicate"]) == 2
                 and str(row["predicate"][0]).lower() in {"close", "open", "turnon", "turnoff"}), None)


def _next_distinct_goal(native: dict[str, Any], completed_goal: list[str]) -> list[str] | None:
    """Advance a multi-goal task only when its current BDDL goal changed."""
    next_goal = _next_goal(native)
    return next_goal if next_goal is not None and next_goal != completed_goal else None


def _stage_action_count(report: dict[str, Any]) -> int:
    total = 0
    for segment in report.get("segments", []):
        total += int(segment.get("actions_executed", 0) or 0)
        total += _segment_recorded_action_count(segment)
        expert_report_path = segment.get("report")
        if expert_report_path and Path(expert_report_path).is_file():
            expert = json.loads(Path(expert_report_path).read_text(encoding="utf-8"))
            for branch in expert.get("physical_branches", []):
                total += int(branch.get(
                    "attempted_control_actions_archived", branch.get("branch_control_actions", 0)
                ) or 0)
        else:
            for branch in segment.get("physical_branches", []):
                total += int(branch.get(
                    "attempted_control_actions_archived", branch.get("branch_control_actions", 0)
                ) or 0)
    return total


def _segment_recorded_action_count(segment: dict[str, Any]) -> int:
    placement = segment.get("placement_search")
    attempts = placement.get("candidate_attempts") if isinstance(placement, dict) else None
    if isinstance(attempts, list) and attempts:
        return sum(int(row.get("recorded_actions", 0) or 0) for row in attempts)
    return int(segment.get("recorded_actions", 0) or 0)


def _stage_physical_branch_count(report: dict[str, Any]) -> int:
    """Count real placement Snapshot branches in one native task stage."""
    count = 0
    for segment in report.get("segments", []):
        placement = segment.get("placement_search")
        if isinstance(placement, dict):
            count += int(placement.get("physical_branch_count", 0) or 0)
        nested_path = segment.get("report")
        if nested_path and Path(nested_path).is_file():
            nested = json.loads(Path(nested_path).read_text(encoding="utf-8"))
            for nested_segment in nested.get("segments", []):
                nested_placement = nested_segment.get("placement_search")
                if isinstance(nested_placement, dict):
                    count += int(nested_placement.get("physical_branch_count", 0) or 0)
    return count


def _find_terminal_snapshot(report: dict[str, Any]) -> str | None:
    for segment in reversed(report.get("segments", [])):
        path = segment.get("terminal_snapshot")
        if path and Path(path).is_file():
            return str(path)
        expert_path = segment.get("report")
        if expert_path and Path(expert_path).is_file():
            expert = json.loads(Path(expert_path).read_text(encoding="utf-8"))
            success = expert.get("success")
            snapshot = success.get("snapshot") if isinstance(success, dict) else None
            candidate = snapshot.get("path") if isinstance(snapshot, dict) else None
            if candidate and Path(candidate).is_file():
                return str(candidate)
    return None


def _register_stage_grasp_experience(stage_report: dict[str, Any], library_path: Path) -> list[str]:
    if not library_path.is_file():
        return []
    library = json.loads(library_path.read_text(encoding="utf-8"))
    added = []
    for segment in stage_report.get("segments", []):
        expert_path = segment.get("report")
        if not expert_path or not Path(expert_path).is_file():
            continue
        expert = json.loads(Path(expert_path).read_text(encoding="utf-8"))
        success = expert.get("success")
        if not isinstance(success, dict):
            continue
        learning = expert.get("experience_learning") or {}
        if learning.get("status") == "skill_added":
            template_id = learning.get("template_id")
            if template_id:
                added.append(str(template_id))
            continue
        if learning.get("status") == "skill_already_present":
            continue
        stable = success.get("stable_bilateral_grasp") or {}
        if stable.get("stable", stable.get("stable_grasp_observed")) is not True:
            continue
        geometry_id = None
        for branch in expert.get("physical_branches", []):
            if branch.get("candidate_id") == success.get("candidate_id"):
                geometry_id = branch.get("geometry_candidate_id")
                break
        candidate = next((row.get("candidate_geometry") for row in expert.get("candidates", [])
            if row.get("candidate_id") == geometry_id), None)
        if not isinstance(candidate, dict) or not candidate.get("object_to_eef"):
            continue
        source_id = (
            f"{expert.get('scope', {}).get('benchmark')}:L"
            f"{expert.get('scope', {}).get('level')}T{expert.get('scope', {}).get('task_id')}"
            f":{candidate.get('candidate_id')}"
        )
        if any(row.get("template_id") == source_id for row in library.get("grasp_templates", [])):
            continue
        target_id = str(success.get("target_object_id", ""))
        family = target_id.rsplit("_", 1)[0] if target_id.rsplit("_", 1)[-1].isdigit() else target_id
        library.setdefault("grasp_templates", []).append({
            "template_id": source_id,
            "object_id_in_source": target_id,
            "object_family": family.lower(),
            "operation": "bilateral_grasp_with_object_relative_approach",
            "source_task": expert.get("active_target", {}).get("instruction"),
            "source_episode_archive": expert.get("archive"),
            "source_global_timestep": None,
            "object_to_eef": candidate["object_to_eef"],
            "approach_direction_object": candidate["approach_direction_object"],
            "approach_distance_m": float(candidate.get("approach_distance_m", 0.06)),
            "approach_waypoints_object_xyz_quaternion_wxyz": candidate.get(
                "historical_approach_waypoints_object_xyz_quaternion_wxyz", []
            ),
            "stable_bilateral_grasp_verified": True,
            "skill_success": True,
            "verified_lift_m": success.get("target_lift_delta_m"),
            "full_native_task_success_in_source_episode": False,
            "source_native_task_success": False,
            "historical_actions_replayed": False,
            "source_actions_used_as_direct_cross_scene_labels": False,
            "runtime_physics_revalidation_required": True,
            "official_safety_demonstration_data_used": False,
            "official_safety_demonstrations_used": False,
            "safety_certified": False,
        })
        added.append(source_id)
    if added:
        library.setdefault("experience_update_history", []).append({
            "source_stage_report": str(stage_report.get("archive")),
            "added_grasp_skill_templates": added,
            "official_safety_demonstration_data_used": False,
            "safety_certified": False,
        })
        library_path.write_text(json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return added


def _existing_stage_reports(task_dir: Path) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for stage_root in sorted(task_dir.glob("stage_[0-9][0-9]")):
        reports = sorted(stage_root.glob("task_success_first_executor_*/run_report.json"))
        if not reports:
            continue
        report = json.loads(reports[-1].read_text(encoding="utf-8"))
        final_native = report.get("final_native_task_status")
        if not isinstance(final_native, dict):
            final_native = next((segment.get("native_task_status")
                for segment in reversed(report.get("segments", []))
                if isinstance(segment.get("native_task_status"), dict)), None)
        if isinstance(final_native, dict):
            report["final_native_task_status"] = final_native
        report["batch_stage"] = {
            "stage_index": int(stage_root.name.rsplit("_", 1)[-1]),
            "elapsed_s": None,
            "executor_log": str(stage_root / "executor.log"),
            "resumed_from_existing_archive": True,
        }
        stages.append(report)
    return stages


def _artifacts_for_stage(stage: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = []
    for segment in stage.get("segments", []):
        if isinstance(segment.get("artifacts"), dict):
            artifacts.append(segment["artifacts"])
        placement = segment.get("placement_search")
        attempts = placement.get("candidate_attempts", []) if isinstance(placement, dict) else []
        for attempt in attempts:
            branch_archive = attempt.get("branch_archive")
            if branch_archive:
                branch_root = Path(branch_archive)
                artifacts.append({
                    "records": str(branch_root / "physical_step_records.jsonl.gz"),
                    "video": str(branch_root / "dual_camera_agentview_wrist_20fps.mp4"),
                })
        recovery = segment.get("joint_limit_recovery")
        if isinstance(recovery, dict) and isinstance(recovery.get("artifacts"), dict):
            artifacts.append(recovery["artifacts"])
        nested = segment.get("report")
        if nested and Path(nested).is_file():
            expert = json.loads(Path(nested).read_text(encoding="utf-8"))
            if isinstance(expert.get("artifacts"), dict):
                artifacts.append(expert["artifacts"])
    return artifacts


def _episode_manifest(task_dir: Path, stages: list[dict[str, Any]], final_native: dict[str, Any]) -> Path:
    segments = []
    for stage_index, stage in enumerate(stages):
        for segment_index, artifact in enumerate(_artifacts_for_stage(stage)):
            row = {"stage_index": stage_index, "segment_index": segment_index}
            for key in ("records", "per_action_records_gzip", "npz", "video", "dual_camera_video"):
                path = artifact.get(key)
                if path and Path(path).is_file():
                    data = Path(path).read_bytes()
                    row[key] = {
                        "path": str(path), "size_bytes": len(data), "sha256": _sha(data),
                    }
            segments.append(row)
    payload = {
        "schema_version": "gs_opd_task_success_episode_manifest.v1",
        "task_success": True,
        "native_task_status": final_native,
        "safety_certified": False,
        "official_safety_demonstration_data_used": False,
        "training_data_eligible": False,
        "segments": segments,
        "best_snapshot": next((path for stage in reversed(stages)
            for path in [_find_terminal_snapshot(stage)] if path), None),
        "physical_execution_source": "Student_and_expert_OSC_actions_in_real_MuJoCo",
    }
    path = task_dir / "task_success_episode_manifest.json"
    _write_json(path, payload)
    return path


def _run_stage(
    args: argparse.Namespace,
    *,
    task_id: int,
    stage_index: int,
    predicate: list[str],
    snapshot_root: Path,
    task_dir: Path,
) -> tuple[dict[str, Any], int, Path]:
    stage_root = task_dir / f"stage_{stage_index:02d}"
    stage_root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), "/workspace/zy_workspace/LIBERO-Safety",
         "/workspace/zy_workspace/LIBERO-Safety/third_party/robosuite-1.4",
         env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    command = [
        sys.executable, "-m", "research.gs_opd.repair.task_success_first_executor",
        "--snapshot-root", str(snapshot_root), "--output-root", str(stage_root),
        "--checkpoint", str(args.checkpoint), "--device", args.device,
        "--benchmark", args.suite, "--level", str(args.level), "--task-id", str(task_id),
        "--init-state", str(args.init_state), "--seed", str(args.seed),
        "--camera-resolution", str(args.camera_resolution),
        "--port", str(args.base_port + task_id * 8 + 1),
        "--expert-port", str(args.base_port + task_id * 8 + 2),
        "--student-action-budget", str(args.student_actions),
        "--student-handoff-action-budget", str(args.student_handoff_actions),
        "--maximum-candidates", str(args.maximum_candidates),
        "--maximum-rrt-preflights", str(args.rrt_preflights),
        "--maximum-physical-branches", str(args.physical_branches),
        "--rrt-time-budget-s", str(args.rrt_time_budget_s),
        "--demonstration-horizon", str(args.demonstration_horizon),
        "--target-predicate", *[str(token) for token in predicate],
        "--experience-library", str(args.experience_library),
        "--transport-strategy", args.transport_strategy,
    ]
    if args.skip_student_first:
        command.append("--skip-student-first")
    if args.native_region_route_variant is not None:
        command.extend(["--native-region-route-variant", args.native_region_route_variant])
    command.extend([
        "--native-region-candidate-offset", str(args.native_region_candidate_offset),
        "--native-region-transit-clearance-mode", args.native_region_transit_clearance_mode,
        "--native-region-entry-mode", args.native_region_entry_mode,
        "--native-region-aperture-tilt-degrees", str(args.native_region_aperture_tilt_degrees),
        "--native-region-progress-max-step-m", str(args.native_region_progress_max_step_m),
    ])
    if args.native_region_yaw_delta_degrees is not None:
        command.extend([
            "--native-region-yaw-delta-degrees", str(args.native_region_yaw_delta_degrees),
        ])
    for keypose_path in args.keypose_experience:
        command.extend(["--keypose-experience", str(Path(keypose_path).resolve())])
    if stage_index == 0 and args.resume_from_runtime_snapshot is not None:
        command.extend([
            "--resume-from-runtime-snapshot",
            str(args.resume_from_runtime_snapshot.resolve()),
        ])
    if stage_index == 0 and args.recovery_fallback_runtime_snapshot is not None:
        command.extend([
            "--recovery-fallback-runtime-snapshot",
            str(args.recovery_fallback_runtime_snapshot.resolve()),
        ])
    log_path = stage_root / "executor.log"
    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    reports = sorted(stage_root.glob("task_success_first_executor_*/run_report.json"))
    if not reports:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        return {
            "status": "executor_report_missing", "process_exit_code": process.returncode,
            "log": str(log_path), "log_tail": tail,
            "elapsed_s": time.monotonic() - started,
        }, process.returncode, stage_root
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    if not isinstance(report.get("final_native_task_status"), dict):
        for segment in reversed(report.get("segments", [])):
            final_status = segment.get("final_native_task_status") or segment.get("native_task_status")
            nested = segment.get("report")
            if not isinstance(final_status, dict) and nested and Path(nested).is_file():
                expert = json.loads(Path(nested).read_text(encoding="utf-8"))
                final_status = expert.get("final_native_task_status") or expert.get("preflight", {}).get("native_task_status")
            if isinstance(final_status, dict):
                report["final_native_task_status"] = final_status
                break
    report["batch_stage"] = {
        "stage_index": stage_index,
        "requested_goal_predicate": predicate,
        "process_exit_code": process.returncode,
        "elapsed_s": time.monotonic() - started,
        "executor_log": str(log_path),
    }
    return report, process.returncode, stage_root


def _task_summary(task_id: str, scene: dict[str, Any], stages: list[dict[str, Any]]) -> dict[str, Any]:
    final = stages[-1].get("final_native_task_status") if stages else None
    if final is None and stages:
        final = stages[-1].get("segments", [{}])[-1].get("native_task_status")
    total_goals = satisfied = 0
    if isinstance(final, dict) and final.get("status") == "complete":
        total_goals, satisfied = native_goal_counts(final)
    student_actions = sum(
        int(segment.get("actions_executed", 0) or 0)
        for stage in stages for segment in stage.get("segments", [])
    )
    expert_actions = sum(
        _segment_recorded_action_count(segment)
        for stage in stages for segment in stage.get("segments", [])
    )
    expert_actions += sum(
        int(branch.get("attempted_control_actions_archived", branch.get("branch_control_actions", 0)) or 0)
        for stage in stages for segment in stage.get("segments", [])
        for branch in segment.get("physical_branches", []) if not segment.get("report")
    )
    expert_actions += sum(
        int(branch.get("attempted_control_actions_archived", branch.get("branch_control_actions", 0)) or 0)
        for stage in stages for segment in stage.get("segments", [])
        if segment.get("report") and Path(segment["report"]).is_file()
        for branch in json.loads(Path(segment["report"]).read_text(encoding="utf-8")).get("physical_branches", [])
    )
    branches = sum(
        len(segment.get("physical_branches", []))
        for stage in stages for segment in stage.get("segments", []) if not segment.get("report")
    )
    branches += sum(
        len(json.loads(Path(segment["report"]).read_text(encoding="utf-8")).get("physical_branches", []))
        for stage in stages for segment in stage.get("segments", [])
        if segment.get("report") and Path(segment["report"]).is_file()
    )
    rrt_attempts = 0
    for stage in stages:
        for segment in stage.get("segments", []):
            placement = segment.get("placement_search", {})
            if not isinstance(placement, dict):
                placement = {}
            rrt_attempts += int(segment.get("candidate_count", 0) or 0)
            rrt_attempts += int(placement.get("RRT_preflight_count", 0) or 0)
            attempts = placement.get("candidate_attempts", [])
            candidate_route_attempts = sum(
                1 for attempt in attempts
                if isinstance(attempt, dict)
                and (
                    attempt.get("failure_phase") == "route_preflight"
                    or (
                        isinstance(attempt.get("transport_route_RRT"), dict)
                        and attempt["transport_route_RRT"].get("status") not in {None, "not_run"}
                    )
                )
            )
            segment_route = segment.get("transport_route_RRT")
            segment_route_attempted = int(
                isinstance(segment_route, dict)
                and segment_route.get("status") not in {None, "not_run"}
            )
            # Candidate-level evidence is authoritative when present; the
            # summary field reflects only the final physical branch.
            rrt_attempts += max(candidate_route_attempts, segment_route_attempted)
    candidate_reports = []
    experience_templates = []
    experience_skills = []
    skill_success_times = []
    for stage in stages:
        for segment in stage.get("segments", []):
            expert_path = segment.get("report")
            if expert_path and Path(expert_path).is_file():
                expert_report = json.loads(Path(expert_path).read_text(encoding="utf-8"))
                candidate_reports.append(expert_report)
                skill_time = expert_report.get("time_to_first_verified_grasp_skill_success_s")
                if isinstance(skill_time, (int, float)) and skill_time >= 0:
                    skill_success_times.append(float(skill_time))
                transfer = expert_report.get("experience_transfer") or {}
                if transfer.get("migrated_grasp_prior_inserted"):
                    experience_templates.append(transfer.get("template_id"))
                if transfer.get("operation_skill_id"):
                    experience_skills.append(transfer.get("operation_skill_id"))
    reachable = any(
        candidate.get("rrt_status") == "offline_feasible"
        for report in candidate_reports for candidate in report.get("candidates", [])
    )
    stable_grasp = any(
        segment.get("grasp_stability", {}).get("stable") is True
        for stage in stages for segment in stage.get("segments", [])
    ) or any(
        (report.get("success") or {}).get("stable_bilateral_grasp", {}).get(
            "stable", (report.get("success") or {}).get("stable_bilateral_grasp", {}).get(
                "stable_grasp_observed"
            )
        ) is True
        or any(
            branch.get("stable_bilateral_grasp_observed") is True
            or (branch.get("grasp_before_lift") or {}).get(
                "stable", (branch.get("grasp_before_lift") or {}).get("stable_grasp_observed")
            ) is True
               for branch in report.get("physical_branches", []))
        for report in candidate_reports
    )
    transported = any(
        segment.get("type") == "expert_native_goal_stage_control"
        and int(segment.get("transport_actions", 0) or 0) > 0
        for stage in stages for segment in stage.get("segments", [])
    )
    initial_satisfied = 0
    for stage in stages:
        found_initial = False
        for segment in stage.get("segments", []):
            initial_status = segment.get("native_task_status")
            if isinstance(initial_status, dict) and initial_status.get("status") == "complete":
                _, initial_satisfied = native_goal_counts(initial_status)
                found_initial = True
                break
        if found_initial:
            break
    for report in candidate_reports:
        if initial_satisfied:
            break
        initial_status = (report.get("preflight") or {}).get("native_task_status")
        if isinstance(initial_status, dict) and initial_status.get("status") == "complete":
            _, initial_satisfied = native_goal_counts(initial_status)
            break
    partial_predicate = satisfied > initial_satisfied
    maximum_lift = max((
        float(
            branch.get("target_lift_delta_m")
            if branch.get("target_lift_delta_m") is not None
            else branch.get("maximum_bilateral_grasp_lift_m", 0.0)
        )
        for report in candidate_reports for branch in report.get("physical_branches", [])
    ), default=0.0)
    progress = progress_label(final or {}, {
        "partial_predicate": partial_predicate,
        "transported": transported,
        "stable_grasp": stable_grasp,
        "reachable": reachable,
    })
    collision_events = 0
    incomplete_monitor_actions = 0
    monitored_substeps = 0
    record_files: list[Path] = []
    videos: list[str] = []
    for stage in stages:
        for artifacts in _artifacts_for_stage(stage):
            record_path = artifacts.get("records") or artifacts.get("per_action_records_gzip")
            if record_path and Path(record_path).is_file():
                record_files.append(Path(record_path))
            video_path = artifacts.get("video") or artifacts.get("dual_camera_video")
            if video_path and Path(video_path).is_file():
                videos.append(str(video_path))
    for record_path in dict.fromkeys(record_files):
        try:
            with gzip.open(record_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    contacts = item.get("contacts", [])
                    if isinstance(contacts, dict):
                        contacts = contacts.get("contacts", [])
                    collision_events += len(contacts) if isinstance(contacts, list) else 0
                    monitor = item.get("contact_monitoring", {})
                    monitored_substeps += int(monitor.get("captured_physics_substeps", 0) or 0)
                    incomplete_monitor_actions += int(not item.get("substep_monitoring_complete", False))
        except (OSError, json.JSONDecodeError):
            incomplete_monitor_actions += 1
    search_seconds = sum(float(stage.get("batch_stage", {}).get("elapsed_s", 0.0) or 0.0) for stage in stages)
    return {
        "task_id": task_id,
        "scene": f"{scene.get('benchmark')}/L{scene.get('level')}T{scene.get('task_id')}",
        "instruction": scene.get("instruction"),
        "native_task_success": final.get("task_success") if isinstance(final, dict) else None,
        "native_predicates_satisfied": satisfied,
        "native_predicates_total": total_goals,
        "stage_count": len(stages),
        "student_actions": student_actions,
        "expert_actions": expert_actions,
        "physical_actions": student_actions + expert_actions,
        "snapshot_branches": branches,
        "placement_route_branches": sum(
            _stage_physical_branch_count(stage) for stage in stages
        ),
        "maximum_progress": progress,
        "stable_grasp_verified": stable_grasp,
        "maximum_verified_lift_m": maximum_lift,
        "transport_phase_executed": transported,
        "rrt_attempts": rrt_attempts,
        "historical_experience_templates_used": list(dict.fromkeys(experience_templates)),
        "historical_experience_reuse_count": len(experience_templates),
        "cross_scene_grasp_transfer_physically_verified": bool(stable_grasp and experience_templates),
        "cross_scene_grasp_transfer_verified_count": 1 if stable_grasp and experience_templates else 0,
        "historical_operation_skills_used": list(dict.fromkeys(experience_skills)),
        "historical_operation_skill_reuse_count": len(experience_skills),
        "task_specific_full_script_count": 0,
        "task_specific_code_changes_count": 0,
        "manual_task_adjustments_count": 0,
        "search_time_s": search_seconds,
        "time_to_first_verified_grasp_skill_success_s": min(skill_success_times)
        if skill_success_times else None,
        "collision_events_recorded": collision_events,
        "physics_substeps_recorded": monitored_substeps,
        "incomplete_monitor_actions": incomplete_monitor_actions,
        "monitoring_complete": incomplete_monitor_actions == 0,
        "failure_reasons": [
            reason
            for stage in stages
            for reason in (
                [stage.get("failure_reason") or stage.get("error")]
                + [segment.get("failure_reason") for segment in stage.get("segments", [])]
            )
            if reason
        ],
        "best_snapshot": next((path for stage in reversed(stages)
            for path in [_find_terminal_snapshot(stage)] if path), None),
        "video_paths": list(dict.fromkeys(videos)),
        "safety_certified": False,
        "final_status": stages[-1].get("status") if stages else "not_run",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    original_library = args.experience_library.resolve() if args.experience_library else None
    working_library = args.output_dir / "task_skill_experience_library.json"
    if working_library.is_file():
        args.experience_library = working_library
    elif original_library and original_library.is_file():
        shutil.copyfile(original_library, working_library)
        args.experience_library = working_library
    task_rows = {}
    for row in manifest.get("tasks", []):
        suffix = str(row.get("task_id", "")).rsplit("T", 1)[-1]
        level_text = str(row.get("level", "L-1"))
        level_index = level_text.removeprefix("L")
        if suffix.isdigit() and level_index.isdigit():
            task_rows[(row["suite"], int(level_index), int(suffix))] = row
    # Benchmark level task ids match the manifest's within-level order.
    chosen = args.task_ids or list(TASK_ORDER)
    if args.resume_from_runtime_snapshot is not None and len(chosen) != 1:
        raise ValueError("resume_from_runtime_snapshot_requires_exactly_one_selected_task")
    results: list[dict[str, Any]] = []
    global_actions = 0
    for task_id in chosen:
        task_row = task_rows.get((args.suite, args.level, task_id))
        if task_row is None:
            raise KeyError(f"task_missing_from_manifest:{args.suite}:L{args.level}T{task_id}")
        key = f"{args.suite}_L{args.level}T{task_id}"
        task_dir = args.output_dir / key
        task_dir_preexisted = task_dir.exists()
        if task_dir_preexisted and not args.resume:
            raise FileExistsError(f"existing task archive requires --resume: {task_dir}")
        task_dir.mkdir(parents=True, exist_ok=True)
        existing_task_report = task_dir / "task_report.json"
        if existing_task_report.is_file():
            existing = json.loads(existing_task_report.read_text(encoding="utf-8"))
            summary = existing.setdefault("summary", {})
            summary.setdefault("time_to_first_native_success_s", None)
            summary.setdefault("actions_to_first_native_success", None)
            summary.setdefault("total_task_wall_time_s", summary.get("search_time_s"))
            summary.setdefault("wall_time_excludes_prior_archived_run", True)
            summary.setdefault("historical_experience_templates_used", [])
            summary.setdefault("historical_experience_reuse_count", 0)
            summary.setdefault("task_specific_full_script_count", 0)
            _write_json(existing_task_report, existing)
            results.append(existing)
            global_actions += int(existing.get("summary", {}).get("physical_actions", 0) or 0)
            continue
        args.expected_bddl_hashes = {str(task_id): task_row.get("bddl_sha256")}
        task_started = time.monotonic()
        time_to_first_success_s = None
        actions_before_first_success = None
        initial_root, preflight = _capture_task_initial(args, task_dir, task_id)
        task_stages: list[dict[str, Any]] = _existing_stage_reports(task_dir) if args.resume else []
        if args.resume and task_dir_preexisted and not task_stages:
            task_stages.append({
                "status": "stage_interrupted_before_executor_report",
                "failure_reason": "priority_reordered_before_any_stage_report_was_written; no stage was replayed",
                "segments": [],
                "batch_stage": {"resumed_from_existing_archive": True},
            })
        task_actions = 0
        for prior_stage in task_stages:
            prior_actions = _stage_action_count(prior_stage)
            task_actions += prior_actions
            global_actions += prior_actions
        current_root = initial_root
        current_native = preflight["initial_native_task_status"]
        stage_index = len(task_stages)
        while stage_index < args.maximum_stages_per_task:
            predicate = _next_goal(current_native)
            if predicate is None:
                break
            remaining_branches = max(
                0, args.physical_branches - sum(
                    _stage_physical_branch_count(row) for row in task_stages
                ),
            )
            estimated_stage_actions = (
                args.student_actions + args.student_handoff_actions
                + remaining_branches * 192
            )
            if remaining_branches == 0:
                task_stages.append({
                    "status": "task_physical_branch_budget_exhausted_before_next_goal",
                    "requested_goal_predicate": predicate,
                    "physical_branches_used": args.physical_branches,
                    "physical_branch_budget": args.physical_branches,
                    "final_native_task_status": current_native,
                    "segments": [],
                })
                break
            if (
                global_actions + estimated_stage_actions > args.global_action_budget
                or task_actions + estimated_stage_actions > args.per_task_action_budget
            ):
                task_stages.append({
                    "status": (
                        "global_action_budget_exhausted_before_stage"
                        if global_actions + estimated_stage_actions > args.global_action_budget
                        else "per_task_action_budget_exhausted_before_stage"
                    ),
                    "requested_goal_predicate": predicate,
                    "estimated_actions": estimated_stage_actions,
                    "global_actions_used": global_actions,
                    "task_actions_used": task_actions,
                })
                break
            stage_args = argparse.Namespace(**vars(args))
            stage_args.physical_branches = min(args.physical_branches, remaining_branches)
            stage_report, _, stage_root = _run_stage(
                stage_args, task_id=task_id, stage_index=stage_index,
                predicate=predicate, snapshot_root=current_root, task_dir=task_dir,
            )
            task_stages.append(stage_report)
            stage_actions = _stage_action_count(stage_report)
            global_actions += stage_actions
            task_actions += stage_actions
            if args.experience_library is not None:
                stage_report["experience_skills_added"] = _register_stage_grasp_experience(
                    stage_report, args.experience_library
                )
            final_native = stage_report.get("final_native_task_status")
            if not isinstance(final_native, dict):
                final_native = current_native
            current_native = final_native
            if current_native.get("task_success") is True and time_to_first_success_s is None:
                time_to_first_success_s = time.monotonic() - task_started
                actions_before_first_success = task_actions
            if current_native.get("task_success") is True:
                break
            # A stage can terminate its Safety Bridge after a hard simulator
            # error (for example, restoring a Snapshot at the prior episode
            # horizon). Preserve that stage's physical audit and stop goal
            # scheduling cleanly; pending_object_goals requires a complete
            # native predicate list and must not turn the bridge failure into
            # an unrelated batch exception.
            if not isinstance(current_native.get("native_goal_values"), list):
                break
            next_predicate = _next_distinct_goal(current_native, predicate)
            # Continue multi-object tasks only after the current native goal
            # actually changed. A partial grasp/transport candidate must not
            # trigger another search for the same predicate from an ambiguous
            # terminal state.
            if next_predicate is None:
                break
            terminal = _find_terminal_snapshot(stage_report)
            if terminal is None:
                break
            next_root = task_dir / "stage_snapshots" / f"after_{stage_index:02d}"
            current_root = _snapshot_root_from_payload(Path(terminal).read_bytes(), next_root)
            stage_index += 1
        summary = _task_summary(key, preflight["scene_metadata"], task_stages)
        summary["time_to_first_native_success_s"] = time_to_first_success_s
        summary["actions_to_first_native_success"] = actions_before_first_success
        summary["total_task_wall_time_s"] = time.monotonic() - task_started
        summary["wall_time_excludes_prior_archived_run"] = bool(task_dir_preexisted and task_stages)
        task_result = {
            "schema_version": "gs_opd_task_success_first_task_report.v1",
            "manifest_task_id": task_row["task_id"],
            "task_id": task_id,
            "preflight": preflight,
            "stages": task_stages,
            "summary": summary,
            "historical_GS_OPD_data_policy": {
                "historical_data_used_as_search_prior": False,
                "historical_replay_count": 0,
                "autonomous_search_stages": len(task_stages),
                "official_safety_demonstrations_used": False,
            },
        }
        final_native = task_stages[-1].get("final_native_task_status") if task_stages else None
        if isinstance(final_native, dict) and final_native.get("task_success") is True:
            episode_path = _episode_manifest(task_dir, task_stages, final_native)
            task_result["task_success_episode_manifest"] = str(episode_path)
        _write_json(task_dir / "task_report.json", task_result)
        summary["report"] = str(task_dir / "task_report.json")
        task_result["summary"] = summary
        _write_json(task_dir / "task_report.json", task_result)
        results.append(task_result)
        if global_actions >= args.global_action_budget:
            break
    report = {
        "schema_version": "gs_opd_task_success_first_batch_report.v1",
        "scope": {
            "suite": args.suite, "level": args.level,
            "task_ids_requested": chosen, "seed": args.seed,
            "init_state": args.init_state,
            "manifest": str(args.manifest),
            "official_safety_demonstration_data_used": False,
            "training_started": False,
            "strict_safety_reconstruction_modified": False,
            "experience_library": str(args.experience_library) if args.experience_library else None,
        },
        "budgets": {
            "global_physical_action_budget": args.global_action_budget,
            "physical_actions_archived": global_actions,
            "per_task_physical_action_budget": args.per_task_action_budget,
            "per_task_student_actions": args.student_actions,
            "student_handoff_actions": args.student_handoff_actions,
            "geometry_candidates": args.maximum_candidates,
            "RRT_preflights_per_goal": args.rrt_preflights,
            "physical_snapshot_branches_per_goal": args.physical_branches,
            "physical_snapshot_branch_budget_per_task": args.physical_branches,
            "physical_snapshot_branches_used": sum(
                _stage_physical_branch_count(stage)
                for task in results for stage in task.get("stages", [])
            ),
            "RRT_time_budget_s": args.rrt_time_budget_s,
            "demonstration_horizon": args.demonstration_horizon,
            "standard_benchmark_horizon": 1000,
            "horizon_mode": "standard_benchmark" if args.demonstration_horizon == 1000 else "demonstration_only_extension",
            "transport_strategy": args.transport_strategy,
            "keypose_experience_references": [str(path.resolve()) for path in args.keypose_experience],
            "maximum_stages_per_task": args.maximum_stages_per_task,
        },
        "tasks": [item["summary"] for item in results],
        "task_reports": [item["summary"]["report"] for item in results],
    }
    if (
        original_library is not None
        and original_library.is_file()
        and args.experience_library is not None
        and args.experience_library.resolve() != original_library
        and args.experience_library.is_file()
    ):
        from research.gs_opd.repair.task_skill_experience import merge_verified_search_experiences

        try:
            report["experience_library_merge"] = merge_verified_search_experiences(
                original_library, args.experience_library
            )
        except Exception as exc:
            report["experience_library_merge"] = {
                "status": "failed",
                "target_library": str(original_library),
                "run_library": str(args.experience_library),
                "error": f"{type(exc).__name__}: {exc}",
            }
    _write_json(args.output_dir / "batch_report.json", report)
    _write_markdown(args.output_dir / "batch_report.md", report)
    _write_csv(args.output_dir / "cross_task_summary.csv", report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Task-Success-First L0 batch",
        "",
        f"Suite `{report['scope']['suite']}`, L{report['scope']['level']}, seed="
        f"{report['scope']['seed']}, init-state={report['scope']['init_state']}",
        "",
        "| Task | Native success | Predicates | Student actions | Expert actions | Branches | RRT | Search s | First skill s | Contacts | Monitor gaps | Status | Best Snapshot |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["tasks"]:
        lines.append(
            f"| {row['scene']} | {row['native_task_success']} | "
            f"{row['native_predicates_satisfied']}/{row['native_predicates_total']} | "
            f"{row['student_actions']} | {row['expert_actions']} | {row['snapshot_branches']} | "
            f"{row['rrt_attempts']} | {row['search_time_s']:.1f} | "
            f"{row.get('time_to_first_verified_grasp_skill_success_s')} | "
            f"{row['collision_events_recorded']} | {row['incomplete_monitor_actions']} | "
            f"{row['final_status']} | {row['best_snapshot'] or ''} |"
        )
    lines.extend(["", f"Physical actions archived: {report['budgets']['physical_actions_archived']} / "
                  f"{report['budgets']['global_physical_action_budget']}",
                  "", "Search trajectories are not safety-certified training data."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, report: dict[str, Any]) -> None:
    import csv

    fields = [
        "task_id", "scene", "native_task_success", "native_predicates_satisfied",
        "native_predicates_total", "student_actions", "expert_actions",
        "snapshot_branches", "rrt_attempts", "search_time_s", "collision_events_recorded",
        "time_to_first_verified_grasp_skill_success_s", "stable_grasp_verified",
        "cross_scene_grasp_transfer_physically_verified", "task_specific_code_changes_count",
        "manual_task_adjustments_count",
        "physics_substeps_recorded", "incomplete_monitor_actions", "monitoring_complete",
        "final_status", "best_snapshot", "report",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in report["tasks"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suite", default="obstacle_avoidance")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--task-ids", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true",
                        help="resume an existing batch archive without rerunning archived task stages")
    parser.add_argument("--resume-from-runtime-snapshot", type=Path, default=None,
                        help="start one task from an identity-checked prior physical Runtime Snapshot")
    parser.add_argument("--recovery-fallback-runtime-snapshot", type=Path, default=None,
                        help="legal identity-matched hold Snapshot used only after failed OSC limit recovery")
    parser.add_argument("--base-port", type=int, default=39100)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--student-actions", type=int, default=16)
    parser.add_argument("--skip-student-first", action="store_true",
                        help="skip the policy branch when starting from a verified physical Snapshot")
    parser.add_argument("--student-handoff-actions", type=int, default=8)
    parser.add_argument("--maximum-candidates", type=int, default=16)
    parser.add_argument("--rrt-preflights", type=int, default=6)
    parser.add_argument("--physical-branches", type=int, default=2)
    parser.add_argument("--rrt-time-budget-s", type=float, default=10.0)
    parser.add_argument("--demonstration-horizon", type=int, default=1000)
    parser.add_argument("--maximum-stages-per-task", type=int, default=3)
    parser.add_argument("--global-action-budget", type=int, default=5000)
    parser.add_argument("--per-task-action-budget", type=int, default=1800)
    parser.add_argument("--experience-library", type=Path, default=DEFAULT_EXPERIENCE_LIBRARY)
    parser.add_argument("--keypose-experience", type=Path, action="append", default=[])
    parser.add_argument(
        "--transport-strategy",
        choices=("native_region_progress_osc", "legacy_rrt"),
        default="native_region_progress_osc",
    )
    parser.add_argument(
        "--native-region-route-variant",
        choices=(
            "direct_region_progress", "keypose_side_region_progress",
            "keypose_side_region_progress_reverse",
        ),
        default=None,
    )
    parser.add_argument("--native-region-candidate-offset", type=int, default=0)
    parser.add_argument(
        "--native-region-transit-clearance-mode",
        choices=("compiled_obstacle", "reachability_biased", "surface_slide", "low_clearance"),
        default="compiled_obstacle",
    )
    parser.add_argument(
        "--native-region-entry-mode",
        choices=("over_rim", "horizontal_aperture"),
        default="over_rim",
    )
    parser.add_argument("--native-region-aperture-tilt-degrees", type=float, default=0.0)
    parser.add_argument("--native-region-yaw-delta-degrees", type=float, default=None)
    parser.add_argument("--native-region-progress-max-step-m", type=float, default=0.008)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and not args.resume:
        parser.error(f"output directory already exists: {args.output_dir}")
    if not args.checkpoint.exists():
        parser.error(f"Student checkpoint does not exist: {args.checkpoint}")
    if not 1 <= args.student_actions <= 256 or not 1 <= args.student_handoff_actions <= 256:
        parser.error("Student action budgets must be in 1..256")
    if not 1 <= args.maximum_candidates <= 16 or not 1 <= args.rrt_preflights <= 8:
        parser.error("candidate and RRT budgets exceed the implemented bounded search")
    if not 1 <= args.physical_branches <= 3 or not 1 <= args.maximum_stages_per_task <= 8:
        parser.error("physical branch or stage budget is outside its bounded range")
    if args.global_action_budget < 1 or args.per_task_action_budget < 1 or not 0.5 <= args.rrt_time_budget_s <= 60:
        parser.error("global action or RRT time budget is invalid")
    if not 1000 <= args.demonstration_horizon <= 3600:
        parser.error("--demonstration-horizon must be in 1000..3600")
    if args.resume_from_runtime_snapshot is not None:
        if args.task_ids is None or len(args.task_ids) != 1:
            parser.error("--resume-from-runtime-snapshot requires exactly one explicit --task-ids value")
        if not args.resume_from_runtime_snapshot.is_file():
            parser.error("--resume-from-runtime-snapshot must reference an existing Snapshot")
    if any(not path.is_file() for path in args.keypose_experience):
        parser.error("--keypose-experience must reference existing GS-OPD experience JSON files")
    if (
        args.recovery_fallback_runtime_snapshot is not None
        and not args.recovery_fallback_runtime_snapshot.is_file()
    ):
        parser.error("--recovery-fallback-runtime-snapshot must reference an existing Snapshot")
    if args.task_ids is not None and any(item < 0 or item > 4 for item in args.task_ids):
        parser.error("L0 task ids must be in 0..4")
    args.manifest = args.manifest.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    return args


def main() -> int:
    args = _parse_args()
    report = run(args)
    print(json.dumps({"batch_report": str(args.output_dir / "batch_report.json"), "tasks": report["tasks"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
