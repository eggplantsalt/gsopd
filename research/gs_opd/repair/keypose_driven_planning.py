# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Extract and retarget physical keyposes from verified GS-OPD successes.

This module transfers geometry and phase structure only.  It never returns or
replays source actions, and it deliberately keeps task success separate from
safety certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "gs_opd_keypose_experience.v1"
_PHASE_TO_KEYPOSE = (
    ("lift", "lift_to_clearance", "object"),
    ("high_west_clearance", "high_west_clearance", "object"),
    ("high_south_corridor", "high_south_corridor", "object"),
    ("yaw_alignment", "high_yaw_alignment", "object"),
    ("preplace", "southwest_target_overhead", "region"),
    ("place", "slow_descend", "region"),
    ("release", "open_release", "region"),
    ("retreat", "post_release_grasp_axis_withdrawal", "region"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_matrix(position: Any, quaternion_wxyz: Any) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("keypose_pose_shape_is_invalid")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("keypose_pose_is_not_finite")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("keypose_quaternion_has_zero_norm")
    quaternion /= norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = position
    return matrix


def pose_record(matrix: Any) -> dict[str, list[float]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("keypose_transform_must_be_finite_4x4")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("keypose_transform_has_invalid_homogeneous_row")
    quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    quaternion_wxyz = np.r_[quaternion_xyzw[3], quaternion_xyzw[:3]]
    return {
        "position_m": matrix[:3, 3].tolist(),
        "quaternion_wxyz": (quaternion_wxyz / np.linalg.norm(quaternion_wxyz)).tolist(),
        "matrix": matrix.tolist(),
    }


def update_pose_record(
    pose: dict[str, Any],
    *,
    position_m: Any | None = None,
    quaternion_wxyz: Any | None = None,
) -> dict[str, list[float]]:
    """Update pose fields as one rigid transform, never leaving a stale matrix."""
    if not isinstance(pose, dict):
        raise ValueError("keypose_pose_record_must_be_a_mapping")
    position = pose.get("position_m") if position_m is None else position_m
    quaternion = pose.get("quaternion_wxyz") if quaternion_wxyz is None else quaternion_wxyz
    synchronized = pose_record(pose_matrix(position, quaternion))
    pose.clear()
    pose.update(synchronized)
    return pose


def _load_row_state(row: dict[str, Any], prefix: str) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = row.get("physical_state_ref") or {}
    archive = reference.get("archive") or reference.get("npz")
    if not archive:
        raise ValueError(f"physical_state_reference_missing_for_timestep:{row.get('episode_timestep_before')}")
    path = Path(str(archive))
    if not path.is_file():
        raise FileNotFoundError(f"physical_state_archive_missing:{path}")
    index = int(reference.get("row", -1))
    if index < 0:
        raise ValueError("physical_state_reference_row_is_invalid")
    with np.load(path, allow_pickle=False) as data:
        robot_key = f"{prefix}_robot_state_json"
        objects_key = f"{prefix}_all_object_states_json"
        if robot_key in data.files and objects_key in data.files:
            robot = json.loads(str(data[robot_key][index]))
            objects = json.loads(str(data[objects_key][index]))
        elif f"{prefix}_physical_state_json" in data.files:
            state = json.loads(str(data[f"{prefix}_physical_state_json"][index]))
            robot_rows = state.get("robot_states", [])
            if not robot_rows:
                raise ValueError("physical_state_has_no_robot_state")
            robot = robot_rows[0]
            objects = state.get("object_states", {})
        else:
            raise ValueError(f"physical_state_archive_lacks_{prefix}_robot_and_object_states:{path}")
    return robot, objects


def _object_row(objects: dict[str, Any], object_id: str) -> dict[str, Any]:
    matches = [
        value for key, value in objects.items()
        if str(key).split(":", 1)[-1] == object_id
    ]
    if len(matches) != 1:
        raise ValueError(f"source_object_physical_state_not_unique:{object_id}:{len(matches)}")
    row = matches[0]
    if "root_pos" not in row or "root_quat" not in row:
        raise ValueError(f"source_object_root_pose_missing:{object_id}")
    return row


def _actual_pose(row: dict[str, Any], object_id: str, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    robot, objects = _load_row_state(row, prefix)
    eef = pose_matrix(robot.get("eef_pos"), robot.get("eef_quat"))
    obj_row = _object_row(objects, object_id)
    obj = pose_matrix(obj_row.get("root_pos"), obj_row.get("root_quat"))
    return eef, obj


def _source_acceptance(
    trace_path: Path,
    acceptance_path: Path,
    *,
    object_id: str,
    predicate: list[str],
) -> dict[str, Any]:
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    task = str(acceptance.get("task", ""))
    criteria = acceptance.get("success_criteria", {})
    goal_values = acceptance.get("final_native_goal_values", [])
    native = next((
        row for row in goal_values
        if row.get("predicate") == predicate and row.get("value") is True
        and row.get("is_native_goal") is True
    ), None)
    if (
        acceptance.get("schema_version") != "gs_opd_l0t2_complete_task_success_report.v1"
        or "obstacle_avoidance/L0T2" not in task
        or acceptance.get("task_success") is not True
        or criteria.get("native_predicates_all_true") is not True
        or acceptance.get("bridge_last_reward") != 1.0
        or acceptance.get("bridge_last_done") is not True
        or native is None
        or acceptance.get("data_provenance", {}).get("official_safety_demonstrations_used") is True
    ):
        raise ValueError("keypose_source_is_not_the_verified_GS_OPD_L0T2_native_success")
    with trace_path.open(encoding="utf-8") as handle:
        trace_rows = sum(1 for line in handle if line.strip())
    expected = int(acceptance.get("trajectory", {}).get("action_count", -1))
    if trace_rows != expected or trace_rows != int(acceptance.get("episode_timestep", -2)):
        raise ValueError(f"keypose_source_trace_length_mismatch:{trace_rows}!={expected}")
    return acceptance


def extract_keypose_experience(
    trace_path: Path,
    acceptance_path: Path,
    *,
    object_id: str,
    predicate: list[str],
    source_region_world: Any,
) -> dict[str, Any]:
    """Derive actual object/EEF keyposes from one accepted own-source trace."""
    trace_path = trace_path.resolve()
    acceptance_path = acceptance_path.resolve()
    acceptance = _source_acceptance(
        trace_path, acceptance_path, object_id=object_id, predicate=predicate,
    )
    region = np.asarray(source_region_world, dtype=np.float64)
    if region.shape != (4, 4) or not np.isfinite(region).all():
        raise ValueError("source_native_region_transform_must_be_finite_4x4")
    if not np.allclose(region[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("source_native_region_transform_is_not_rigid")
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["episode_timestep_before"]))
    expected_t = list(range(len(rows)))
    actual_t = [int(row["episode_timestep_before"]) for row in rows]
    if actual_t != expected_t:
        raise ValueError("keypose_source_timestep_sequence_is_not_contiguous")
    phase_end_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("target_object") != object_id:
            continue
        phase_end_rows[str(row.get("phase", ""))] = row
    missing = [phase for _, phase, _ in _PHASE_TO_KEYPOSE if phase not in phase_end_rows]
    if missing:
        raise ValueError(f"keypose_source_missing_required_physical_phases:{missing}")
    # The carry-frame origin is the first actual state of the final placement
    # segment (t=854 in this accepted source), before the lift keypose.
    lift_rows = [
        row for row in rows
        if row.get("target_object") == object_id and row.get("phase") == "lift_to_clearance"
    ]
    if not lift_rows:
        raise ValueError("keypose_source_has_no_lift_segment")
    start_row = lift_rows[0]
    start_eef, start_object = _actual_pose(start_row, object_id, "pre")
    source_region_inverse = np.linalg.inv(region)
    keyposes = []
    for name, phase, frame in _PHASE_TO_KEYPOSE:
        row = phase_end_rows[phase]
        eef, obj = _actual_pose(row, object_id, "post")
        timestep = int(row["episode_timestep_after"])
        keyposes.append({
            "name": name,
            "phase": phase,
            "source_timestep_after": timestep,
            "frame": frame,
            "object_delta_from_carry_start": pose_record(np.linalg.inv(start_object) @ obj),
            "object_to_eef": pose_record(np.linalg.inv(obj) @ eef),
            "region_to_object": pose_record(source_region_inverse @ obj),
            "region_to_eef": pose_record(source_region_inverse @ eef),
            "object_world_source": pose_record(obj),
            "eef_world_source": pose_record(eef),
            "source_phase_action": str(row.get("action_source", "unknown")),
        })
    keyposes.insert(0, {
        "name": "grasp",
        "phase": "verified_held_grasp_start",
        "source_timestep_after": int(start_row["episode_timestep_before"]),
        "frame": "object",
        "object_delta_from_carry_start": pose_record(np.eye(4, dtype=np.float64)),
        "object_to_eef": pose_record(np.linalg.inv(start_object) @ start_eef),
        "region_to_object": pose_record(source_region_inverse @ start_object),
        "region_to_eef": pose_record(source_region_inverse @ start_eef),
        "object_world_source": pose_record(start_object),
        "eef_world_source": pose_record(start_eef),
        "source_phase_action": "verified_source_physical_grasp_prefix",
    })
    return {
        "schema_version": SCHEMA,
        "experience_id": f"{object_id.split('_')[0]}_native_success_keypose_v1",
        "source": {
            "task": acceptance["task"],
            "instruction": acceptance["instruction"],
            "trace_path": str(trace_path),
            "trace_sha256": _sha256(trace_path),
            "acceptance_report_path": str(acceptance_path),
            "acceptance_report_sha256": _sha256(acceptance_path),
            "native_task_success": True,
            "native_predicate": predicate,
            "episode_timestep": int(acceptance["episode_timestep"]),
            "safety_success": acceptance.get("safety_label", {}).get("safety_success"),
            "source_safety_reason": acceptance.get("safety_label", {}).get("reason"),
            "official_safety_demonstrations_used": False,
            "source_actions_replayed": False,
        },
        "object_family": "_".join(object_id.split("_")[:-1]) or object_id,
        "target_relation": str(predicate[0]).lower(),
        "target_region_name": str(predicate[-1]),
        "source_carry_start_timestep": int(start_row["episode_timestep_before"]),
        "source_carry_start_object_world": pose_record(start_object),
        "source_carry_start_eef_world": pose_record(start_eef),
        "source_target_region_world": pose_record(region),
        "keypose_sequence": keyposes,
        "route_semantics": {
            "object_frame_phases": [row["name"] for row in keyposes if row["frame"] == "object"],
            "target_region_frame_phases": [row["name"] for row in keyposes if row["frame"] == "region"],
            "phase_order_is_source_physical_execution_order": True,
            "actual_7d_actions_copied": False,
            "safety_certified": False,
        },
    }


def retarget_keypose_sequence(
    experience: dict[str, Any],
    *,
    held_object_world: Any,
    current_eef_world: Any,
    destination_region_world: Any,
    destination_object_world: Any,
) -> list[dict[str, Any]]:
    """Relocate source carry and placement keyposes to a live held-object goal."""
    if experience.get("schema_version") != SCHEMA:
        raise ValueError("keypose_experience_schema_is_unsupported")
    source = experience.get("source", {})
    if source.get("native_task_success") is not True or source.get("official_safety_demonstrations_used") is not False:
        raise ValueError("keypose_experience_source_provenance_is_not_accepted")
    held = np.asarray(held_object_world, dtype=np.float64)
    eef = np.asarray(current_eef_world, dtype=np.float64)
    destination_region = np.asarray(destination_region_world, dtype=np.float64)
    destination_object = np.asarray(destination_object_world, dtype=np.float64)
    for name, matrix in (("held_object", held), ("eef", eef), ("destination_region", destination_region), ("destination_object", destination_object)):
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError(f"retarget_{name}_transform_is_invalid")
    source_region = np.asarray(experience["source_target_region_world"]["matrix"], dtype=np.float64)
    source_keyposes = experience.get("keypose_sequence")
    if not isinstance(source_keyposes, list) or len(source_keyposes) < 3:
        raise ValueError("keypose_experience_sequence_is_incomplete")
    source_place = next(row for row in source_keyposes if row["name"] == "release")
    source_place_local = np.asarray(source_place["region_to_object"]["matrix"], dtype=np.float64)
    destination_place_local = np.linalg.inv(destination_region) @ destination_object
    region_retarget = destination_place_local @ np.linalg.inv(source_place_local)
    current_object_to_eef = np.linalg.inv(held) @ eef
    result = []
    for row in source_keyposes:
        if row["frame"] == "object":
            object_target = held @ np.asarray(row["object_delta_from_carry_start"]["matrix"], dtype=np.float64)
        else:
            object_local = region_retarget @ np.asarray(row["region_to_object"]["matrix"], dtype=np.float64)
            object_target = destination_region @ object_local
        if row["name"] == "retreat":
            eef_local = region_retarget @ np.asarray(row["region_to_eef"]["matrix"], dtype=np.float64)
            eef_target = destination_region @ eef_local
        else:
            eef_target = object_target @ current_object_to_eef
        result.append({
            "name": row["name"],
            "phase": row["phase"],
            "frame": row["frame"],
            "source_timestep_after": row["source_timestep_after"],
            "object_pose_world": pose_record(object_target),
            "eef_pose_world": pose_record(eef_target),
            "route_source": "GS_OPD_own_success_actual_physical_keypose",
            "source_actions_replayed": False,
        })
    return result


def select_keypose_suffix(
    sequence: list[dict[str, Any]], start_after: str | None = None,
) -> list[dict[str, Any]]:
    """Resume after an exactly named verified keypose without replaying its prefix."""
    if not isinstance(sequence, list) or not sequence:
        raise ValueError("keypose_sequence_is_empty")
    if start_after is None:
        return list(sequence)
    names = [str(row.get("name", "")) for row in sequence]
    if start_after not in names:
        raise ValueError(f"keypose_resume_phase_is_not_in_relocated_sequence:{start_after}")
    result = sequence[names.index(start_after) + 1:]
    if not result:
        raise ValueError("keypose_resume_phase_has_no_remaining_physical_keyposes")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--object-id", default="moka_pot_1")
    parser.add_argument("--predicate", nargs=3, default=["on", "moka_pot_1", "flat_stove_1_cook_region"])
    parser.add_argument("--source-region-position", nargs=3, type=float, required=True)
    parser.add_argument("--source-region-quaternion-wxyz", nargs=4, type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    region = pose_matrix(args.source_region_position, args.source_region_quaternion_wxyz)
    result = extract_keypose_experience(
        args.trace, args.acceptance_report, object_id=args.object_id,
        predicate=args.predicate, source_region_world=region,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "extracted",
        "output": str(args.output),
        "keypose_count": len(result["keypose_sequence"]),
        "source_trace_sha256": result["source"]["trace_sha256"],
        "native_task_success": result["source"]["native_task_success"],
        "safety_success": result["source"]["safety_success"],
        "source_actions_replayed": result["source"]["source_actions_replayed"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
