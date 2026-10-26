# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Build and query a provenance-checked, object-relative skill experience library."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_DATASET = Path(
    "/opt/data/private/gs_opd/datasets/"
    "lerobot_l0t2_round0_operational_success_v21_20261025"
)
DEFAULT_OUTPUT = Path(
    "/opt/data/private/gs_opd/experience/"
    "gs_opd_task_skill_experience_library_20261026_v2.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_family(object_id: str) -> str:
    return re.sub(r"_\d+$", "", str(object_id)).lower()


def _pose(position: Any, quaternion_wxyz: Any) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("experience_pose_shape_is_invalid")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("experience_pose_contains_non_finite_values")
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = position
    return matrix


def _pose_row(matrix: np.ndarray) -> list[float]:
    xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    quaternion = np.r_[xyzw[3], xyzw[:3]]
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return np.r_[matrix[:3, 3], quaternion].tolist()


def _verified_source(dataset_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest_path = dataset_root / "data_manifest.json"
    acceptance_path = dataset_root / "round0_lerobot_acceptance.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    trajectory = acceptance.get("source_trajectory", {})
    safety = acceptance.get("safety_acceptance", {})
    if (
        acceptance.get("status") != "verified"
        or trajectory.get("native_task_success") is not True
        or trajectory.get("done") is not True
        or manifest.get("native_task_success") is not True
        or acceptance.get("data_provenance", {}).get("official_safety_demonstrations_used") is True
    ):
        raise ValueError("experience_source_is_not_a_verified_GS_OPD_native_success_episode")
    if manifest.get("episode_count") != 1 or manifest.get("total_action_samples") != trajectory.get("physical_actions"):
        raise ValueError("experience_source_episode_provenance_is_inconsistent")
    if safety.get("operational_safety") != "pass":
        raise ValueError("experience_source_did_not_pass_its_recorded_operational_acceptance")
    provenance_path = dataset_root / "per_action_provenance.jsonl"
    expected_rows = int(acceptance.get("data_provenance", {}).get("per_action_provenance_rows", -1))
    return manifest, acceptance, provenance_path


def _load_phase_rows(path: Path, expected_count: int) -> tuple[dict[int, str], list[dict[str, Any]]]:
    phase_at_pre_timestep: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            timestep = int(row["global_pre_action_timestep"])
            if timestep in phase_at_pre_timestep:
                raise ValueError(f"experience_provenance_duplicate_timestep:{timestep}")
            phase_at_pre_timestep[timestep] = str(row.get("task_phase") or "unknown")
            rows.append(row)
    if len(rows) != expected_count:
        raise ValueError(f"experience_provenance_row_count_mismatch:{len(rows)}!={expected_count}")
    rows.sort(key=lambda row: int(row["global_pre_action_timestep"]))
    return phase_at_pre_timestep, rows


def _trace_sha(segment: dict[str, Any], trace_path: Path) -> str:
    expected = segment.get("trace_sha256")
    actual = _sha256(trace_path)
    if expected and actual != expected:
        raise ValueError(f"experience_trace_hash_mismatch:{trace_path}")
    return actual


def _physical_rows(
    trace_path: Path,
    *,
    global_start: int,
    action_count: int,
    phase_at_pre_timestep: dict[int, str],
    segment_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with np.load(trace_path, allow_pickle=False) as trace:
        keys = set(trace.files)
        if "post_physical_state_json" in keys:
            states = trace["post_physical_state_json"]
            timesteps = trace.get("step_timesteps")
            timestamps_are_post_action = timesteps is not None
            if timesteps is None:
                timesteps = trace.get("timesteps")
            if timesteps is None:
                timesteps = np.arange(global_start, global_start + len(states))
            for index, (time_value, state_value) in enumerate(zip(timesteps, states)):
                recorded_timestep = int(time_value)
                timestep = recorded_timestep - 1 if timestamps_are_post_action else recorded_timestep
                state = json.loads(str(state_value))
                robot_states = state.get("robot_states", [])
                if not robot_states:
                    continue
                robot = robot_states[0]
                eef = _pose(robot.get("eef_pos"), robot.get("eef_quat"))
                objects = state.get("object_states", {})
                object_poses = {}
                for key, item in objects.items():
                    object_id = str(key).split(":", 1)[-1]
                    if not isinstance(item, dict) or "root_pos" not in item or "root_quat" not in item:
                        continue
                    object_poses[object_id] = _pose(item["root_pos"], item["root_quat"])
                rows.append({
                    "timestep": timestep,
                    "phase": phase_at_pre_timestep.get(timestep, "unknown"),
                    "eef_world": eef,
                    "object_world": object_poses,
                    "gripper_qpos": robot.get("gripper_qpos"),
                    "segment_index": segment_index,
                    "trace_row": index,
                })
        elif all(key in keys for key in (
            "eef_position_world_m", "eef_quaternion_wxyz",
            "pot2_root_position_world_m", "pot2_root_quaternion_wxyz",
        )):
            timesteps = trace.get("timesteps", np.arange(global_start, global_start + action_count))
            for index in range(min(len(timesteps), action_count)):
                rows.append({
                    "timestep": int(timesteps[index]),
                    "phase": phase_at_pre_timestep.get(int(timesteps[index]), "unknown"),
                    "eef_world": _pose(trace["eef_position_world_m"][index], trace["eef_quaternion_wxyz"][index]),
                    "object_world": {
                        "moka_pot_2": _pose(
                            trace["pot2_root_position_world_m"][index],
                            trace["pot2_root_quaternion_wxyz"][index],
                        )
                    },
                    "gripper_qpos": trace["gripper_qpos_m"][index].tolist()
                    if "gripper_qpos_m" in keys else None,
                    "segment_index": segment_index,
                    "trace_row": index,
                })
    return rows


def _select_trace_for_segment(
    segment: dict[str, Any], manifest: dict[str, Any], dataset_root: Path,
) -> tuple[Path | None, str | None]:
    direct = Path(str(segment.get("trace", "")))
    if direct.is_file():
        _trace_sha(segment, direct)
        with np.load(direct, allow_pickle=False) as archive:
            if "post_physical_state_json" in archive.files or "eef_position_world_m" in archive.files:
                return direct, _sha256(direct)
    start, end = (int(value) for value in segment["global_timestep_range_exclusive"])
    expected_actions = end - start
    candidates = []
    source_files = manifest.get("source_files", {})
    for raw_path, metadata in source_files.items():
        path = Path(raw_path)
        if path.suffix != ".npz" or not path.is_file():
            continue
        try:
            expected_hash = metadata.get("sha256")
            actual_hash = _sha256(path)
            if expected_hash and actual_hash != expected_hash:
                continue
            with np.load(path, allow_pickle=False) as archive:
                if "post_physical_state_json" not in archive.files:
                    continue
                states = archive["post_physical_state_json"]
                timesteps = archive.get("step_timesteps")
                if timesteps is None or len(states) != expected_actions:
                    continue
                t = np.asarray(timesteps, dtype=np.int64)
                if len(t) and start <= int(t[0]) <= start + 1 and end - 2 <= int(t[-1]) <= end:
                    candidates.append((path, actual_hash))
        except (OSError, ValueError, KeyError):
            continue
    if candidates:
        return sorted(candidates, key=lambda item: ("safety_reconstruction" not in str(item[0]), str(item[0])))[0]
    return None, None


def _contiguous_spans(rows: list[dict[str, Any]], phase: str) -> list[list[dict[str, Any]]]:
    selected = [row for row in rows if row["phase"] == phase]
    spans: list[list[dict[str, Any]]] = []
    for row in selected:
        if not spans or row["timestep"] != spans[-1][-1]["timestep"] + 1:
            spans.append([row])
        else:
            spans[-1].append(row)
    return spans


def build_experience_library(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    manifest, acceptance, provenance_path = _verified_source(dataset_root)
    expected_rows = int(manifest["total_action_samples"])
    phase_map, provenance = _load_phase_rows(provenance_path, expected_rows)
    physical: list[dict[str, Any]] = []
    trace_sources = []
    for segment in manifest.get("segments", []):
        trace, digest = _select_trace_for_segment(segment, manifest, dataset_root)
        if trace is None:
            continue
        start, end = (int(value) for value in segment["global_timestep_range_exclusive"])
        trace_rows = _physical_rows(
            trace,
            global_start=start,
            action_count=end - start,
            phase_at_pre_timestep=phase_map,
            segment_index=int(segment["segment_index"]),
        )
        physical.extend(trace_rows)
        trace_sources.append({
            "segment_index": int(segment["segment_index"]),
            "global_timestep_range_exclusive": [start, end],
            "path": str(trace),
            "sha256": digest,
            "physical_state_rows": len(trace_rows),
        })
    physical.sort(key=lambda row: (row["timestep"], row["segment_index"], row["trace_row"]))
    deduplicated = {}
    for row in physical:
        deduplicated[row["timestep"]] = row
    physical = [deduplicated[timestep] for timestep in sorted(deduplicated)]

    grasp_templates = []
    approach_spans = _contiguous_spans(physical, "approach")
    stable_spans = _contiguous_spans(physical, "stable_grasp")
    for stable in stable_spans:
        stable_objects = []
        for object_id in stable[-1]["object_world"]:
            eef = stable[-1]["eef_world"]
            obj = stable[-1]["object_world"][object_id]
            distance = float(np.linalg.norm(eef[:3, 3] - obj[:3, 3]))
            if distance <= 0.25:
                stable_objects.append((distance, object_id, obj))
        if not stable_objects:
            continue
        _, object_id, object_pose = min(stable_objects, key=lambda item: item[0])
        contact_eef = stable[-1]["eef_world"]
        object_to_eef = np.linalg.inv(object_pose) @ contact_eef
        prior_approaches = [span for span in approach_spans
            if stable[0]["timestep"] - 80 <= span[-1]["timestep"] < stable[0]["timestep"]]
        approach_path = []
        if prior_approaches:
            approach = max(prior_approaches, key=lambda span: span[-1]["timestep"])
            matching_rows = [row for row in approach if object_id in row["object_world"]]
            matching_rows = matching_rows[-12:]
            for row in matching_rows:
                relative = np.linalg.inv(row["object_world"][object_id]) @ row["eef_world"]
                approach_path.append(_pose_row(relative))
        if len(approach_path) > 6:
            indices = np.linspace(0, len(approach_path) - 1, 6).round().astype(int)
            approach_path = [approach_path[index] for index in sorted(set(indices.tolist()))]
        direction = object_to_eef[:3, 3].copy()
        if len(approach_path) >= 2:
            first = np.asarray(approach_path[0][:3], dtype=np.float64)
            last = np.asarray(approach_path[-1][:3], dtype=np.float64)
            direction = last - first
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            direction = object_to_eef[:3, 2]
            norm = float(np.linalg.norm(direction))
        direction /= max(norm, 1e-12)
        grasp_templates.append({
            "template_id": f"{_object_family(object_id)}:stable_grasp:t{stable[-1]['timestep']}",
            "object_id_in_source": object_id,
            "object_family": _object_family(object_id),
            "operation": "bilateral_grasp_with_object_relative_approach",
            "source_task": manifest.get("task"),
            "source_global_timestep": stable[-1]["timestep"],
            "source_segment_index": stable[-1]["segment_index"],
            "object_to_eef": object_to_eef.tolist(),
            "approach_direction_object": direction.tolist(),
            "approach_distance_m": 0.06,
            "approach_waypoints_object_xyz_quaternion_wxyz": approach_path,
            "approach_waypoint_count": len(approach_path),
            "source_grasp_span": [stable[0]["timestep"], stable[-1]["timestep"]],
            "runtime_physics_revalidation_required": True,
        })

    phase_spans = []
    phase_sequence = []
    for row in provenance:
        phase = str(row.get("task_phase") or "unknown")
        if not phase_sequence or phase_sequence[-1] != phase:
            phase_sequence.append(phase)
        if not phase_spans or phase_spans[-1]["phase"] != phase:
            phase_spans.append({"phase": phase, "start": int(row["global_pre_action_timestep"]),
                                "end": int(row["global_post_action_timestep"])})
        else:
            phase_spans[-1]["end"] = int(row["global_post_action_timestep"])

    operation_skill_templates = []
    stable_starts = sorted(
        int(item["source_grasp_span"][0]) for item in grasp_templates
    )
    for template in grasp_templates:
        stable_start, stable_end = (int(value) for value in template["source_grasp_span"])
        later_grasps = [value for value in stable_starts if value > stable_start]
        cycle_end = min(later_grasps) if later_grasps else int(provenance[-1]["global_post_action_timestep"])
        preceding_approaches = [
            span for span in phase_spans
            if span["phase"] == "approach" and int(span["end"]) <= stable_start
        ]
        cycle_spans = []
        if preceding_approaches:
            cycle_spans.append(max(preceding_approaches, key=lambda span: int(span["end"])))
        for span in phase_spans:
            span_start = int(span["start"])
            if stable_start <= span_start < cycle_end:
                cycle_spans.append(span)
                if span["phase"] == "settle":
                    # A completed place/release cycle is a reusable operation
                    # boundary. Do not absorb the next object's recovery.
                    break
        ordered_spans = []
        for span in cycle_spans:
            if not ordered_spans or ordered_spans[-1]["phase"] != span["phase"]:
                ordered_spans.append(span)
        phase_budgets = {
            span["phase"]: int(span["end"]) - int(span["start"])
            for span in ordered_spans
        }
        operation_skill_templates.append({
            "skill_id": f"object_transfer_cycle:{template['object_family']}:t{stable_start}",
            "operation": "pick_transport_place_release",
            "object_family": template["object_family"],
            "source_object_id": template["object_id_in_source"],
            "source_task": manifest.get("task"),
            "source_native_task_success": True,
            "grasp_template_id": template["template_id"],
            "phase_sequence": [span["phase"] for span in ordered_spans],
            "source_phase_action_budgets": phase_budgets,
            "destination_resolution": "active_native_BDDL_goal_and_live_region_geometry",
            "relocalization_frame": "active_target_object_root_for_grasp_then_live_goal_region_for_placement",
            "historical_actions_replayed": False,
            "source_actions_used_as_direct_cross_scene_labels": False,
            "runtime_physics_revalidation_required": True,
            "safety_certified_for_new_scene": False,
        })

    return {
        "schema_version": "gs_opd_task_skill_experience_library.v1",
        "library_id": "l0t2_round0_operational_success_20261025",
        "source_policy": {
            "source_is_GS_OPD_owned": True,
            "official_safety_demonstration_data_used": False,
            "recorded_executed_actions_used_as_direct_cross_scene_labels": False,
            "actions_are_migrated_as_object_relative_OSC_waypoints": True,
            "new_scene_requires_real_MuJoCo_revalidation": True,
            "safety_certified_for_new_scene": False,
        },
        "source_dataset": str(dataset_root),
        "source_episode": {
            "task": manifest.get("task"),
            "episode_count": manifest.get("episode_count"),
            "physical_actions": manifest.get("total_action_samples"),
            "native_task_success": manifest.get("native_task_success"),
            "reward": manifest.get("reward"),
            "done": manifest.get("done"),
            "operational_safety_pass": acceptance.get("safety_acceptance", {}).get("operational_safety"),
            "strict_10mm": acceptance.get("safety_acceptance", {}).get("strict_10mm"),
            "acceptance_report": str(dataset_root / "round0_lerobot_acceptance.json"),
            "acceptance_report_sha256": _sha256(dataset_root / "round0_lerobot_acceptance.json"),
            "data_manifest_sha256": _sha256(dataset_root / "data_manifest.json"),
            "per_action_provenance_sha256": _sha256(provenance_path),
        },
        "source_trace_coverage": trace_sources,
        "phase_sequence": phase_sequence,
        "phase_spans": phase_spans,
        "grasp_templates": grasp_templates,
        "operation_skill_templates": operation_skill_templates,
        "query_policy": {
            "object_match": "runtime_BDDL_target_id_with_numeric_instance_suffix_removed",
            "placement_relation": "runtime_native_BDDL_predicate",
            "path_frame": "active_target_object_root_frame",
            "unmatched_object_family": "fall_back_to_geometry_generated_search",
        },
    }


def load_experience_library(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "gs_opd_task_skill_experience_library.v1":
        raise ValueError("unsupported_task_skill_experience_library_schema")
    if payload.get("source_policy", {}).get("official_safety_demonstration_data_used") is not False:
        raise ValueError("experience_library_source_policy_is_not_allowed")
    return payload


def matching_grasp_templates(library: dict[str, Any], object_id: str) -> list[dict[str, Any]]:
    family = _object_family(object_id)
    matches = [item for item in library.get("grasp_templates", [])
               if item.get("object_family") == family]
    operation_by_grasp = {
        str(item.get("grasp_template_id")): item
        for item in library.get("operation_skill_templates", [])
        if item.get("grasp_template_id")
    }

    def rank(item: dict[str, Any]) -> tuple[int, int, int, int]:
        operation = operation_by_grasp.get(str(item.get("template_id")), {})
        complete_native_source = bool(
            item.get("source_native_task_success") is True
            or operation.get("source_native_task_success") is True
        )
        verified_object_goal_source = bool(
            item.get("native_object_goal_satisfied") is True
            or operation.get("native_object_goal_satisfied") is True
        )
        return (
            int(complete_native_source),
            int(verified_object_goal_source),
            int(item.get("cross_scene_physical_validation_count", 0) or 0),
            -int(item.get("source_global_timestep", 0) or 0),
        )

    return sorted(matches, key=rank, reverse=True)


def matching_grasp_template(library: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    """Return the highest ranked historical prior for backwards compatibility."""
    matches = matching_grasp_templates(library, object_id)
    return matches[0] if matches else None


def append_verified_search_grasp_template(
    path: Path, template: dict[str, Any]
) -> bool:
    """Add a physically verified skill to a GS-OPD-owned library atomically.

    A stable bilateral grasp with a real MuJoCo lift is a reusable skill even
    when the surrounding task has not reached its native BDDL goal. The record
    therefore remains explicitly task-incomplete and uncertified for safety.
    """
    if (
        template.get("skill_success") is not True
        or template.get("stable_bilateral_grasp_verified") is not True
        or float(template.get("verified_lift_m", 0.0) or 0.0) < 0.03
        or template.get("safety_certified") is not False
        or template.get("official_safety_demonstrations_used") is not False
    ):
        raise ValueError("search_grasp_experience_requires_real_lift_and_explicit_provenance")
    library = load_experience_library(path)
    rows = library.setdefault("grasp_templates", [])
    template_id = str(template.get("template_id") or "")
    if not template_id:
        raise ValueError("search_grasp_experience_template_id_is_required")
    if any(str(item.get("template_id")) == template_id for item in rows):
        return False
    row = dict(template)
    row["cross_scene_physical_validation_count"] = 1
    row["runtime_physics_revalidation_required"] = True
    row["historical_actions_replayed"] = False
    row["source_actions_used_as_direct_cross_scene_labels"] = False
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return True


def matching_operation_skill(library: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    """Select a reusable object-transfer phase structure by live object family."""
    family = _object_family(object_id)
    matches = [
        item for item in library.get("operation_skill_templates", [])
        if item.get("object_family") == family
        and item.get("operation") == "pick_transport_place_release"
        and item.get("historical_actions_replayed") is False
    ]
    return max(
        matches,
        key=lambda item: (
            int(item.get("source_native_task_success") is True),
            int(item.get("native_object_goal_satisfied") is True),
            len(item.get("phase_sequence", [])),
            int(item.get("cross_scene_physical_validation_count", 0) or 0),
        ),
        default=None,
    )


def append_verified_search_operation_skill(path: Path, template: dict[str, Any]) -> bool:
    """Append a physically completed native object-goal cycle as reusable structure."""
    required_phases = {
        "approach", "stable_grasp", "lift", "transport", "precise_placement",
        "release", "withdrawal", "settle",
    }
    phases = template.get("phase_sequence")
    failures = []
    checks = {
        "operation": template.get("operation") == "pick_transport_place_release",
        "skill_success": template.get("skill_success") is True,
        "native_goal": template.get("native_object_goal_satisfied") is True,
        "released_stable": template.get("released_and_stable") is True,
        "bilateral_grasp": template.get("stable_bilateral_grasp_verified") is True,
        "safety_false": template.get("safety_certified") is False,
        "official_false": template.get("official_safety_demonstrations_used") is False,
        "actions_not_replayed": template.get("historical_actions_replayed") is False,
        "phases_present": isinstance(phases, list) and required_phases.issubset(set(phases)),
        "physical_actions": int(template.get("physical_actions", 0) or 0) > 0,
        "source_reports": bool(template.get("source_execution_reports")),
        "predicate": isinstance(template.get("native_goal_predicate"), list),
    }
    failures = [key for key, passed in checks.items() if not passed]
    if failures:
        raise ValueError(
            "operation_skill_experience_requires_physical_native_goal_cycle_provenance:"
            + ",".join(failures)
        )
    library = load_experience_library(path)
    template_id = str(template.get("skill_id") or "")
    if not template_id:
        raise ValueError("operation_skill_experience_id_is_required")
    rows = library.setdefault("operation_skill_templates", [])
    if any(str(item.get("skill_id")) == template_id for item in rows):
        return False
    row = dict(template)
    row["cross_scene_physical_validation_count"] = 1
    row["runtime_physics_revalidation_required"] = True
    row["source_actions_used_as_direct_cross_scene_labels"] = False
    rows.append(row)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return True


def merge_verified_search_experiences(
    target_path: Path, run_library_path: Path
) -> dict[str, Any]:
    """Persist verified skills from an isolated run copy without replacing history."""
    target_path = Path(target_path).resolve()
    run_library_path = Path(run_library_path).resolve()
    if target_path == run_library_path:
        raise ValueError("experience_library_merge_requires_an_isolated_run_copy")
    source = load_experience_library(run_library_path)
    target = load_experience_library(target_path)
    grasp_ids = {
        str(row.get("template_id")) for row in target.get("grasp_templates", [])
    }
    operation_ids = {
        str(row.get("skill_id")) for row in target.get("operation_skill_templates", [])
    }
    added_grasps: list[str] = []
    rejected_grasps: list[dict[str, str]] = []
    added_operations: list[str] = []
    rejected_operations: list[dict[str, str]] = []

    for source_row in source.get("grasp_templates", []):
        template_id = str(source_row.get("template_id") or "")
        if not template_id or template_id in grasp_ids:
            continue
        transform = np.asarray(source_row.get("object_to_eef", []), dtype=np.float64)
        approach = np.asarray(source_row.get("approach_direction_object", []), dtype=np.float64)
        lift_m = float(source_row.get("verified_lift_m", 0.0) or 0.0)
        valid = bool(
            source_row.get("stable_bilateral_grasp_verified") is True
            and lift_m >= 0.03
            and transform.shape == (4, 4)
            and np.isfinite(transform).all()
            and approach.shape == (3,)
            and np.isfinite(approach).all()
            and np.linalg.norm(approach) > 1e-8
            and source_row.get("safety_certified") is False
            and (
                source_row.get("official_safety_demonstration_data_used") is False
                or source_row.get("official_safety_demonstrations_used") is False
            )
        )
        if not valid:
            rejected_grasps.append({
                "template_id": template_id,
                "reason": "missing_verified_bilateral_grasp_lift_geometry_or_provenance",
            })
            continue
        row = dict(source_row)
        row.update({
            "skill_success": True,
            "verified_lift_m": lift_m,
            "source_native_task_success": bool(
                source_row.get("source_native_task_success") is True
                or source_row.get("full_native_task_success_in_source_episode") is True
            ),
            "historical_actions_replayed": False,
            "source_actions_used_as_direct_cross_scene_labels": False,
            "official_safety_demonstrations_used": False,
        })
        try:
            if append_verified_search_grasp_template(target_path, row):
                added_grasps.append(template_id)
                grasp_ids.add(template_id)
        except (TypeError, ValueError) as exc:
            rejected_grasps.append({"template_id": template_id, "reason": str(exc)})

    for source_row in source.get("operation_skill_templates", []):
        skill_id = str(source_row.get("skill_id") or "")
        if not skill_id or skill_id in operation_ids:
            continue
        try:
            if append_verified_search_operation_skill(target_path, dict(source_row)):
                added_operations.append(skill_id)
                operation_ids.add(skill_id)
        except (TypeError, ValueError) as exc:
            rejected_operations.append({"skill_id": skill_id, "reason": str(exc)})

    if added_grasps or added_operations:
        target = load_experience_library(target_path)
        target.setdefault("experience_update_history", []).append({
            "run_library": str(run_library_path),
            "added_grasp_skill_templates": added_grasps,
            "added_operation_skill_templates": added_operations,
            "rejected_grasp_skill_templates": rejected_grasps,
            "rejected_operation_skill_templates": rejected_operations,
            "official_safety_demonstration_data_used": False,
            "safety_certified": False,
        })
        temporary = target_path.with_name(target_path.name + ".tmp")
        temporary.write_text(
            json.dumps(target, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target_path)
    return {
        "status": "merged" if added_grasps or added_operations else "no_new_verified_skills",
        "target_library": str(target_path),
        "run_library": str(run_library_path),
        "added_grasp_skill_templates": added_grasps,
        "added_operation_skill_templates": added_operations,
        "rejected_grasp_skill_templates": rejected_grasps,
        "rejected_operation_skill_templates": rejected_operations,
        "safety_certified": False,
    }


def instantiate_grasp_candidate(
    template: dict[str, Any], context: dict[str, Any], *, source_library: str
) -> dict[str, Any]:
    """Relocalize a successful grasp and its local approach path to a live BDDL target."""
    root = _pose(
        context["target_root_pose_world"]["position_m"],
        Rotation.from_matrix(np.asarray(
            context["target_root_pose_world"]["rotation"], dtype=np.float64
        )).as_quat()[[3, 0, 1, 2]],
    )
    relative = np.asarray(template["object_to_eef"], dtype=np.float64)
    if relative.shape != (4, 4) or not np.isfinite(relative).all():
        raise ValueError("experience_grasp_transform_is_invalid")
    approach = np.asarray(template["approach_direction_object"], dtype=np.float64)
    if approach.shape != (3,) or not np.isfinite(approach).all():
        raise ValueError("experience_approach_direction_is_invalid")
    approach /= max(float(np.linalg.norm(approach)), 1e-12)
    grasp = root @ relative
    pregrasp = grasp[:3, 3] - root[:3, :3] @ approach * float(template["approach_distance_m"])
    live_eef = np.asarray(context["eef_pose_world"]["position_m"], dtype=np.float64)
    return {
        "candidate_id": f"experience__{template['template_id']}",
        "target_geom_id": -1,
        "target_geom_name": "source_skill_object_relative_frame",
        "geom_type": "experience_template",
        "object_to_eef": relative.tolist(),
        "approach_direction_object": approach.tolist(),
        "approach_distance_m": float(template["approach_distance_m"]),
        "jaw_axis_index_in_geom": -1,
        "approach_axis_index_in_geom": -1,
        "approach_sign": 0,
        "jaw_sign": 0,
        "estimated_clamp_width_m": None,
        "maximum_opening_m": float(context["maximum_pad_separation_m"]),
        "opening_overrun_m_record_only": 0.0,
        "pregrasp_eef_distance_m": float(np.linalg.norm(pregrasp - live_eef)),
        "pad_center_eef_m_from_active_open_pose": context["symmetric_pad_center_eef_m"],
        "candidate_source": "GS_OPD_success_episode_object_relative_skill_experience",
        "experience_template_id": template["template_id"],
        "experience_library": str(source_library),
        "experience_source_task": template.get("source_task"),
        "experience_source_global_timestep": template.get("source_global_timestep"),
        "historical_approach_waypoints_object_xyz_quaternion_wxyz": template.get(
            "approach_waypoints_object_xyz_quaternion_wxyz", []
        ),
        "official_demo_or_historical_action_prior_used": False,
        "historical_GS_OPD_object_relative_waypoint_prior_used": True,
        "runtime_physics_revalidation_required": True,
    }


def alternate_experience_approach_candidates(
    candidate: dict[str, Any], *, approach_distance_m: float = 0.08
) -> list[dict[str, Any]]:
    """Keep a verified object-relative grasp transform and vary only its approach.

    This preserves the useful grasp geometry while changing the route family
    after an exact historical approach has stalled. It contains no replayed
    actions or task-specific scene coordinates.
    """
    base = np.asarray(candidate["approach_direction_object"], dtype=np.float64)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError("experience_approach_direction_is_invalid")
    base /= max(float(np.linalg.norm(base)), 1e-12)
    axes = np.eye(3, dtype=np.float64)
    seed_axis = axes[int(np.argmin(np.abs(axes @ base)))]
    tangent_u = seed_axis - np.dot(seed_axis, base) * base
    tangent_u /= max(float(np.linalg.norm(tangent_u)), 1e-12)
    tangent_v = np.cross(base, tangent_u)
    tangent_v /= max(float(np.linalg.norm(tangent_v)), 1e-12)
    directions = [
        ("opposite", -base),
        ("side_u_pos", tangent_u),
        ("side_u_neg", -tangent_u),
        ("side_v_pos", tangent_v),
        ("side_v_neg", -tangent_v),
    ]
    variants = []
    for name, direction in directions:
        row = dict(candidate)
        row["candidate_id"] = f"{candidate['candidate_id']}__alt_{name}"
        row["approach_direction_object"] = direction.tolist()
        row["approach_distance_m"] = float(approach_distance_m)
        row["historical_approach_waypoints_object_xyz_quaternion_wxyz"] = []
        row["historical_GS_OPD_object_relative_waypoint_prior_used"] = False
        row["experience_object_relative_grasp_prior_used"] = True
        row["approach_route_family"] = name
        row["candidate_source"] = "GS_OPD_object_relative_grasp_transform_with_alternate_approach_geometry"
        variants.append(row)
    return variants


def migrate_object_relative_pose(object_pose_world: Any, pose_object: Any) -> np.ndarray:
    object_pose_world = np.asarray(object_pose_world, dtype=np.float64)
    pose_object = np.asarray(pose_object, dtype=np.float64)
    if object_pose_world.shape != (4, 4) or pose_object.shape != (4, 4):
        raise ValueError("object_relative_pose_migration_requires_4x4_transforms")
    return object_pose_world @ pose_object


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    library = build_experience_library(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "experience_library": str(args.output),
        "source_task_success": library["source_episode"]["native_task_success"],
        "phase_span_count": len(library["phase_spans"]),
        "grasp_template_count": len(library["grasp_templates"]),
        "physical_trace_segment_count": len(library["source_trace_coverage"]),
        "official_safety_demonstration_data_used": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
