# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Audit RRT qpos/OSC tracking and run an isolated 8-action t463 preview.

The preview is a branch from the strictly restored t463 runtime Snapshot.
It never appends actions to a formal Episode or changes its frozen 0.18 rad
reference threshold. The threshold is reported diagnostically while native
OSC_POSE tracks the existing route's EEF references; MuJoCo certifies every
substep against the original Native/Operational safety policy.
"""

from __future__ import annotations

import __main__
import hashlib
import json
import math
import os
import pickle
import signal
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from research.gs_opd.repair.automatic_recovery import RecoveryArtifactWriter
from research.gs_opd.repair.automatic_recovery_runtime import (
    AutomaticRecoveryBridgeBackend,
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
    sha256_file,
)
from research.gs_opd.repair.run_t329_native_task_suffix import (
    ARCHIVED_ROUTE_DIR,
    CONFIG_PATH,
    EXPECTED_SUBSTEPS,
    MODEL_SHA256,
    OBJECT_ID,
    _all_robot_contacts_finger_target,
    _canonical_sha256,
    _load_archived_route,
    _write_json,
)
from research.gs_opd.repair.resume_t359_pot2_placement import (
    _PythonObjectState,
    _normalized_state_hash,
)


ROOT = Path(__file__).resolve().parents[3]
T329_PHYSICAL_RUN = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_round0_t329_native_task_suffix_20261024_205500"
)
T359_TO_T398_RUN = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_round0_t359_pot2_placement_20261024_213308"
)
T398_TO_T463_RUN = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_round0_t398_pot2_single_replan_20261024_214225"
)
T463_TIMESTEP = 463
T463_CURSOR = 65
MAX_PREVIEW_ACTIONS = 8
JOINT_REFERENCE_LIMIT_RAD = 0.18  # frozen; diagnostic_only does not change it
EEF_REFERENCE_LIMIT_M = 0.025
EEF_ORIENTATION_LIMIT_RAD = 0.08
RISK_TRIGGER_M = 0.05
RISK_REARM_M = 0.06
POT2_NONFINGER_CLEARANCE_M = 0.002
FORBIDDEN_CLEARANCE_M = 0.01


def _sha(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = np.ascontiguousarray(np.asarray(value)).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _read_json_npz(archive: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    value = archive[key]
    if value.shape != ():
        raise ValueError(f"archive_field_{key}_is_not_scalar")
    decoded: Any = value.item()
    if isinstance(decoded, bytes):
        decoded = decoded.decode("utf-8")
    if isinstance(decoded, str):
        decoded = json.loads(decoded)
    if isinstance(decoded, str):
        decoded = json.loads(decoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"archive_field_{key}_did_not_decode_to_object")
    return decoded


def _read_journal(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "physical": _read_json_npz(archive, "post_physical_state_json"),
            "native": _read_json_npz(archive, "post_native_task_status_json"),
            "monitor": _read_json_npz(archive, "post_contact_monitoring_json"),
            "contacts": _read_json_npz(archive, "post_contacts_json"),
            "agentview": np.asarray(archive["post_agentview_image"]).copy(),
            "wrist": np.asarray(archive["post_robot0_eye_in_hand_image"]).copy(),
        }


def _indexed_journals(run_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        run_dir / "durable_action_journal" / "index.jsonl",
        run_dir / "index.jsonl",
    ]
    index_path = next((item for item in candidates if item.is_file()), None)
    if index_path is None:
        raise FileNotFoundError(f"action_journal_index_missing:{run_dir}")
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["journal_path"] = str(row.get("journal_path", ""))
        if not Path(row["journal_path"]).is_file():
            raise FileNotFoundError(row["journal_path"])
    return rows


def _segment_rows(run_dir: Path, route_qpos: np.ndarray, route_poses: np.ndarray,
                  *, route_t0: int, phase: str) -> dict[str, Any]:
    rows = _indexed_journals(run_dir)
    records = []
    for row in rows:
        timestep = int(row["timestep"])
        journal_path = Path(row["journal_path"])
        if row.get("journal_sha256") and sha256_file(journal_path) != row["journal_sha256"]:
            raise ValueError(f"journal_hash_mismatch:{journal_path}")
        data = _read_journal(journal_path)
        physical = data["physical"]
        robot = physical["robot_states"][0]
        actual_q = np.asarray(robot["joint_qpos"], dtype=np.float64)
        actual_eef_pos = np.asarray(robot["eef_pos"], dtype=np.float64)
        actual_eef_quat = np.asarray(robot["eef_quat"], dtype=np.float64)
        expected_index = timestep - route_t0
        if not 0 <= expected_index < len(route_qpos):
            raise ValueError(f"reference_path_index_out_of_range:{phase}:t={timestep}")
        delta = actual_q - route_qpos[expected_index]
        nearest_index = int(np.argmin(np.linalg.norm(route_qpos - actual_q[None, :], axis=1)))
        nearest_q_error = float(np.linalg.norm(actual_q - route_qpos[nearest_index]))
        expected_q_error = float(np.linalg.norm(delta))
        expected_pose = route_poses[expected_index]
        nearest_pose = route_poses[nearest_index]
        eef_position_expected = float(np.linalg.norm(actual_eef_pos - expected_pose[:3]))
        eef_position_nearest = float(np.linalg.norm(actual_eef_pos - nearest_pose[:3]))
        # Orientation uses the SO(3) geodesic angle from relative quaternion.
        def orientation_error(q_ref: np.ndarray) -> float:
            q_a = actual_eef_quat / np.linalg.norm(actual_eef_quat)
            q_b = q_ref / np.linalg.norm(q_ref)
            dot = float(np.clip(abs(np.dot(q_a, q_b)), 0.0, 1.0))
            return float(2.0 * math.acos(dot))

        records.append({
            "timestep": timestep,
            "route_index": expected_index,
            "nearest_qpos_route_index": nearest_index,
            "nearest_minus_cursor_route_points": nearest_index - expected_index,
            "actual_qpos_rad": actual_q.tolist(),
            "reference_qpos_rad": route_qpos[expected_index].tolist(),
            "per_joint_error_actual_minus_reference_rad": delta.tolist(),
            "per_joint_abs_error_rad": np.abs(delta).tolist(),
            "cursor_qpos_error_l2_rad": expected_q_error,
            "nearest_qpos_error_l2_rad": nearest_q_error,
            "nearest_index_qpos_error_reduction_rad": expected_q_error - nearest_q_error,
            "actual_eef_position_world_m": actual_eef_pos.tolist(),
            "actual_eef_quaternion_wxyz": actual_eef_quat.tolist(),
            "reference_eef_pose_at_cursor": expected_pose.tolist(),
            "cursor_eef_position_error_m": eef_position_expected,
            "cursor_eef_orientation_error_rad": orientation_error(expected_pose[3:]),
            "nearest_qpos_index_eef_position_error_m": eef_position_nearest,
            "nearest_qpos_index_eef_orientation_error_rad": orientation_error(nearest_pose[3:]),
            "source_journal": str(journal_path),
        })
    if not records:
        raise ValueError(f"tracking_segment_has_no_journal_records:{phase}")
    q_matrix = np.asarray([item["per_joint_error_actual_minus_reference_rad"] for item in records])
    lag = np.asarray([item["route_index"] - item["nearest_qpos_route_index"] for item in records])
    summary = {
        "phase": phase,
        "source_run": str(run_dir),
        "samples": len(records),
        "timestep_range": [records[0]["timestep"], records[-1]["timestep"]],
        "route_index_range": [records[0]["route_index"], records[-1]["route_index"]],
        "mean_abs_error_per_joint_rad": np.mean(np.abs(q_matrix), axis=0).tolist(),
        "mean_cursor_qpos_error_l2_rad": float(np.mean([
            item["cursor_qpos_error_l2_rad"] for item in records
        ])),
        "mean_nearest_qpos_error_l2_rad": float(np.mean([
            item["nearest_qpos_error_l2_rad"] for item in records
        ])),
        "mean_error_reduction_by_nearest_index_rad": float(np.mean([
            item["nearest_index_qpos_error_reduction_rad"] for item in records
        ])),
        "nearest_index_lag_route_points_min_median_max": [
            int(np.min(lag)), float(np.median(lag)), int(np.max(lag))
        ],
        "nearest_cursor_index_equal_within_one_fraction": float(np.mean(np.abs(lag) <= 1)),
        "samples_over_frozen_0_18_rad": int(sum(
            item["cursor_qpos_error_l2_rad"] > JOINT_REFERENCE_LIMIT_RAD for item in records
        )),
        "mean_cursor_eef_position_error_mm": 1000.0 * float(np.mean([
            item["cursor_eef_position_error_m"] for item in records
        ])),
        "mean_cursor_eef_orientation_error_rad": float(np.mean([
            item["cursor_eef_orientation_error_rad"] for item in records
        ])),
        "records": records,
    }
    return summary


def _audit_tracking() -> dict[str, Any]:
    old_route = _load_archived_route(ARCHIVED_ROUTE_DIR)
    old = old_route["qpos_path"], old_route["eef_reference_poses_world"]
    old_329_359 = _segment_rows(
        T329_PHYSICAL_RUN, *old, route_t0=329, phase="t329_to_t359_cached_route_execution"
    )
    old_359_398 = _segment_rows(
        T359_TO_T398_RUN, *old, route_t0=329, phase="t359_to_t398_cached_route_execution"
    )
    # The diagnostic requested the complete t329→398 behavior, so report both
    # physically archived spans as one 69-control reference segment as well.
    old_combined = {
        "phase": "t329_to_t398_cached_route_execution",
        "source_runs": [str(T329_PHYSICAL_RUN), str(T359_TO_T398_RUN)],
        "records": old_329_359["records"] + old_359_398["records"],
    }
    combined_errors = np.asarray([
        row["per_joint_error_actual_minus_reference_rad"]
        for row in old_combined["records"]
    ], dtype=np.float64)
    combined_lags = np.asarray([
        row["route_index"] - row["nearest_qpos_route_index"]
        for row in old_combined["records"]
    ], dtype=np.int64)
    combined = {
        "phase": old_combined["phase"],
        "source_runs": old_combined["source_runs"],
        "samples": len(old_combined["records"]),
        "timestep_range": [old_combined["records"][0]["timestep"], old_combined["records"][-1]["timestep"]],
        "route_index_range": [old_combined["records"][0]["route_index"], old_combined["records"][-1]["route_index"]],
        "mean_abs_error_per_joint_rad": np.mean(np.abs(combined_errors), axis=0).tolist(),
        "mean_cursor_qpos_error_l2_rad": float(np.mean([r["cursor_qpos_error_l2_rad"] for r in old_combined["records"]])),
        "mean_nearest_qpos_error_l2_rad": float(np.mean([r["nearest_qpos_error_l2_rad"] for r in old_combined["records"]])),
        "mean_error_reduction_by_nearest_index_rad": float(np.mean([r["nearest_index_qpos_error_reduction_rad"] for r in old_combined["records"]])),
        "nearest_index_lag_route_points_min_median_max": [int(np.min(combined_lags)), float(np.median(combined_lags)), int(np.max(combined_lags))],
        "nearest_cursor_index_equal_within_one_fraction": float(np.mean(np.abs(combined_lags) <= 1)),
        "samples_over_frozen_0_18_rad": int(sum(r["cursor_qpos_error_l2_rad"] > JOINT_REFERENCE_LIMIT_RAD for r in old_combined["records"])),
        "mean_cursor_eef_position_error_mm": 1000.0 * float(np.mean([
            r["cursor_eef_position_error_m"] for r in old_combined["records"]
        ])),
        "mean_cursor_eef_orientation_error_rad": float(np.mean([
            r["cursor_eef_orientation_error_rad"] for r in old_combined["records"]
        ])),
        "records": old_combined["records"],
    }

    route_npz = T398_TO_T463_RUN / "replanned_route_t398.npz"
    with np.load(route_npz, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        new_qpos = np.asarray(archive["qpos_path"], dtype=np.float64)
        new_poses = np.asarray(archive["eef_reference_poses_world"], dtype=np.float64)
    if metadata.get("model_geometry_sha256") != MODEL_SHA256:
        raise ValueError("t398_route_model_fingerprint_mismatch")
    new = _segment_rows(
        T398_TO_T463_RUN, new_qpos, new_poses,
        route_t0=398, phase="t398_to_t463_replanned_route_execution",
    )
    return {
        "old_route": {
            "planner_source": old_route["planner_result"].get("planner_source"),
            "route_points": int(len(old[0])),
            "minimum_forbidden_clearance_m": old_route["planner_result"].get("minimum_forbidden_clearance_m"),
            "full_physical_segment_t329_to_t398": combined,
            "t329_to_t359": old_329_359,
            "t359_to_t398": old_359_398,
        },
        "new_route": {
            "source": str(route_npz),
            "metadata": metadata,
            "tracking": new,
        },
        "interpretation": {
            "minor_progress_lag_present": True,
            "progress_lag_is_primary_cause": False,
            "dominant_pattern": "actual_qpos_remains_far_from_nearest_route_configuration_even_when_that_configuration_has_a_near-identical_EEF_pose",
            "mechanism_evidence": "robosuite_OSC_POSE_adds_nullspace_torque_toward_controller_initial_joint; recovery increments path_cursor_per_action_without_nearest_qpos_association",
            "causal_status": "strong_mechanism_consistent_evidence_not_an_intervention_proof",
        },
    }


def _source_463() -> dict[str, Any]:
    source_result = json.loads((T398_TO_T463_RUN / "suffix_result.json").read_text())
    rows = _indexed_journals(T398_TO_T463_RUN)
    if len(rows) != 65 or int(rows[-1]["timestep"]) != T463_TIMESTEP:
        raise ValueError("t463_source_journal_not_65_actions_ending_at_t463")
    last = rows[-1]
    last_journal = Path(last["journal_path"])
    if sha256_file(last_journal) != last.get("journal_sha256"):
        raise ValueError("t463_terminal_journal_sha256_mismatch")
    snapshot_path = Path(last["runtime_snapshot_path"])
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    snapshot_sha = sha256_file(snapshot_path)
    with snapshot_path.open("rb") as stream:
        envelope = pickle.load(stream)
    if envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
        raise ValueError("t463_snapshot_schema_mismatch")
    if envelope.get("model_geometry_sha256") != MODEL_SHA256:
        raise ValueError("t463_snapshot_model_geometry_fingerprint_mismatch")
    snapshot = envelope.get("snapshot")
    if not isinstance(snapshot, dict) or int(snapshot.get("task_bookkeeping", {}).get("timestep", -1)) != T463_TIMESTEP:
        raise ValueError("t463_snapshot_task_timestep_mismatch")
    archived_step = _read_journal(last_journal)
    physical = archived_step["physical"]
    if archived_step["native"].get("task_success") is not False:
        raise ValueError("t463_native_task_should_be_pending")
    if archived_step["monitor"].get("status") != "complete":
        raise ValueError("t463_terminal_action_substep_monitor_incomplete")
    geometry = archived_step["monitor"].get("geometry_monitoring", {}).get("substeps", [])
    if len(geometry) != EXPECTED_SUBSTEPS or any(
        item.get("status") != "complete"
        or any(item.get(key) for key in (
            "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
            "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
        )) for item in geometry
    ):
        raise ValueError("t463_terminal_action_did_not_pass_full_substep_monitor")
    integration = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    physical_integration = np.asarray(physical.get("mujoco_integration_state"), dtype=np.float64)
    if _sha(integration) != _sha(physical_integration):
        raise ValueError("t463_snapshot_and_terminal_action_integration_state_mismatch")
    robots = physical.get("robot_states", [])
    if len(robots) != 1:
        raise ValueError("t463_source_robot_inventory_mismatch")
    robot = robots[0]
    object_key = f"objects_dict:{OBJECT_ID}"
    object_state = physical.get("object_states", {}).get(object_key)
    if not isinstance(object_state, dict):
        raise ValueError("t463_source_pot2_state_missing")
    runtime_robot = snapshot.get("robots", [{}])[0]
    if not runtime_robot.get("has_gripper_current_action"):
        raise ValueError("t463_snapshot_gripper_controller_state_missing")
    controller_initial_joint = np.asarray(
        runtime_robot.get("controller", {}).get("initial_joint"), dtype=np.float64
    )
    if controller_initial_joint.shape != (7,):
        raise ValueError("t463_snapshot_OSC_controller_initial_joint_missing")
    integration_sha = _sha(integration)
    attestation = {
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(snapshot_path),
        "source_snapshot_sha256": snapshot_sha,
        "source_action_record": str(last_journal),
        "source_action_record_sha256": last["journal_sha256"],
        "timestep": T463_TIMESTEP,
        "model_geometry_sha256": MODEL_SHA256,
        "object_id": OBJECT_ID,
        "arm_joint_qpos": robot["joint_qpos"],
        "eef_position_world_m": robot["eef_pos"],
        "eef_quaternion_wxyz": robot["eef_quat"],
        "gripper_qpos_m": robot["gripper_qpos"],
        "object_position_world_m": object_state["root_pos"],
        "object_quaternion_wxyz": object_state["root_quat"],
        "restored_gripper_current_action": np.asarray(
            runtime_robot["gripper_current_action"], dtype=np.float64
        ).tolist(),
        "integration_state_sha256": integration_sha,
        "runtime_snapshot_normalized_sha256": _normalized_state_hash(snapshot),
        "controller_initial_joint_rad": controller_initial_joint.tolist(),
    }
    return {
        "source_run": str(T398_TO_T463_RUN),
        "result": source_result,
        "rows": rows,
        "snapshot_path": snapshot_path,
        "snapshot": snapshot,
        "attestation": attestation,
        "physical": physical,
        "native": archived_step["native"],
        "monitor": archived_step["monitor"],
        "contacts": archived_step["contacts"],
        "controller_initial_joint_rad": controller_initial_joint.tolist(),
        "integration_sha256": integration_sha,
    }


def _camera_array(image_record: dict[str, Any]) -> np.ndarray:
    array = np.frombuffer(image_record["data"], dtype=np.dtype(image_record["dtype"]))
    return array.reshape(tuple(image_record["shape"])).copy()


def _policy_clearance_summary(geometry: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in ("moka_pot_2", "bottle_of_ginger_beer_1", "white_place_box_1"):
        row = geometry.get("clearances_by_object", {}).get(key, {})
        result[key] = {
            "minimum_distance_m": row.get("minimum_distance_m"),
            "status": row.get("status"),
        }
    carried = geometry.get("carried_clearances_by_object", {})
    for key in ("bottle_of_ginger_beer_1", "white_place_box_1"):
        row = carried.get(key, {})
        result[f"carried_moka_pot_2_to_{key}"] = {
            "minimum_distance_m": row.get("minimum_distance_m"),
            "status": row.get("status"),
        }
    return result


def _verify_geometry_substeps(substeps: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    if len(substeps) != EXPECTED_SUBSTEPS:
        return False, {"reason": "expected_25_geometry_samples", "observed": len(substeps)}
    per_object_mins: dict[str, float | None] = {
        "moka_pot_2": None,
        "bottle_of_ginger_beer_1": None,
        "white_place_box_1": None,
        "carried_moka_pot_2_to_bottle_of_ginger_beer_1": None,
        "carried_moka_pot_2_to_white_place_box_1": None,
    }
    failures = []
    for substep_index, item in enumerate(substeps):
        if item.get("status") != "complete":
            failures.append({"substep": substep_index, "reason": "geometry_status_not_complete", "status": item.get("status")})
        for key in ("forbidden_contacts", "self_contacts", "forbidden_object_contacts", "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations"):
            if item.get(key):
                failures.append({"substep": substep_index, "reason": key, "count": len(item[key])})
        clearances = item.get("clearances_by_object", {})
        for object_id, threshold in (
            ("moka_pot_2", POT2_NONFINGER_CLEARANCE_M),
            ("bottle_of_ginger_beer_1", FORBIDDEN_CLEARANCE_M),
            ("white_place_box_1", FORBIDDEN_CLEARANCE_M),
        ):
            clearance = clearances.get(object_id, {}).get("minimum_distance_m")
            if clearance is None or clearance < threshold:
                failures.append({"substep": substep_index, "reason": "robot_object_clearance_below_frozen_policy", "object_id": object_id, "distance_m": clearance, "required_m": threshold})
            elif per_object_mins[object_id] is None or clearance < per_object_mins[object_id]:
                per_object_mins[object_id] = float(clearance)
        carried_clearances = item.get("carried_clearances_by_object", {})
        for object_id in ("bottle_of_ginger_beer_1", "white_place_box_1"):
            clearance = carried_clearances.get(object_id, {}).get("minimum_distance_m")
            key = f"carried_moka_pot_2_to_{object_id}"
            if clearance is None or clearance < FORBIDDEN_CLEARANCE_M:
                failures.append({"substep": substep_index, "reason": "carried_object_forbidden_clearance_below_frozen_policy", "object_id": object_id, "distance_m": clearance, "required_m": FORBIDDEN_CLEARANCE_M})
            elif per_object_mins[key] is None or clearance < per_object_mins[key]:
                per_object_mins[key] = float(clearance)
    return not failures, {"minimum_clearance_by_object_m": per_object_mins, "failures": failures}


def _save_video(frames: list[np.ndarray], path: Path) -> None:
    import cv2

    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could_not_open_preview_video:{path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def run_preview(*, output_dir: Path | None = None, python: Path = DEFAULT_ENV_PYTHON,
                bridge_port: int = 38764) -> dict[str, Any]:
    __main__._PythonObjectState = _PythonObjectState
    audit = _audit_tracking()
    source = _source_463()
    route_npz = T398_TO_T463_RUN / "replanned_route_t398.npz"
    with np.load(route_npz, allow_pickle=False) as archive:
        path_qpos = np.asarray(archive["qpos_path"], dtype=np.float64)
        path_poses = np.asarray(archive["eef_reference_poses_world"], dtype=np.float64)
        route_metadata = json.loads(str(archive["metadata_json"].item()))
    if route_metadata.get("safety_policy_sha256") is None:
        raise ValueError("t398_route_has_no_safety_policy_fingerprint")
    policy_path = T398_TO_T463_RUN / "replanned_route_safety_policy.json"
    policy = json.loads(policy_path.read_text())
    if _canonical_sha256(policy) != route_metadata["safety_policy_sha256"]:
        raise ValueError("t398_route_archived_safety_policy_fingerprint_mismatch")
    if len(path_qpos) != 193 or len(path_poses) != 193 or T463_CURSOR >= len(path_qpos) - 1:
        raise ValueError("t463_route_cursor_or_path_shape_is_invalid")

    if output_dir is None:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"/opt/data/private/gs_opd/rollouts/l0t2_round0_t463_osc_tracking_diagnostic_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "route_tracking_analysis.json", audit)
    _write_json(output_dir / "source_t463_attestation.json", {
        "attestation": source["attestation"],
        "source_suffix_native_safety": source["result"].get("suffix_native_safety"),
        "source_suffix_operational_safety": source["result"].get("suffix_operational_safety"),
        "source_suffix_strict_10mm": source["result"].get("suffix_strict_10mm"),
        "source_reason": source["result"].get("reason"),
        "path_cursor": T463_CURSOR,
        "path_cursor_index_expected_timestep": T463_TIMESTEP,
        "controller_initial_joint_rad": source["controller_initial_joint_rad"],
    })

    config_data, recovery_config = load_runtime_config(CONFIG_PATH)
    config_data = json.loads(json.dumps(config_data))
    config_data["candidate_search"]["enabled"] = False
    recovery_config = replace(
        recovery_config,
        maximum_joint_path_deviation_rad=JOINT_REFERENCE_LIMIT_RAD,
        maximum_eef_tracking_error_m=EEF_REFERENCE_LIMIT_M,
        maximum_eef_orientation_error_rad=EEF_ORIENTATION_LIMIT_RAD,
        hard_minimum_clearance_m=float(policy["minimum_clearance_m"]),
        target_object_clearance_m=min(float(row["minimum_clearance_m"]) for row in policy["allowed_target_objects"]),
        object_clearance_m={
            str(row["object_id"]): float(row["minimum_clearance_m"])
            for row in policy["allowed_target_objects"]
        },
        execution_step_budget=1800,
        horizon_steps=1800,
        allow_horizon_extension=True,
    )
    recovery_config.validate()
    config_data["recovery"] = {
        **vars(recovery_config),
        "object_clearance_m": dict(recovery_config.object_clearance_m),
    }

    artifacts = RecoveryArtifactWriter(output_dir)
    bridge = BridgeProcess(output_dir, python=python, port=bridge_port)
    backend: AutomaticRecoveryBridgeBackend | None = None
    result: dict[str, Any] = {
        "status": "not_started",
        "formal_episode_physical_actions_committed": 0,
        "isolated_branch_action_limit": MAX_PREVIEW_ACTIONS,
        "isolated_branch_actions_executed": 0,
        "qpos_reference_mode": "diagnostic_only_in_this_isolated_entrypoint",
        "formal_joint_threshold_rad": JOINT_REFERENCE_LIMIT_RAD,
        "safety_thresholds_changed": False,
        "route_source": str(route_npz),
        "source_snapshot": str(source["snapshot_path"]),
        "actions": [],
        "main_state_restored_to_t463": False,
    }
    original_handlers = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"signal_{signum}_received_at_isolated_action_boundary")

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.signal(signum, interrupt)
    try:
        bridge.start(
            benchmark=config_data["benchmark"],
            level=int(config_data["level"]),
            task_id=int(config_data["task_id"]),
            init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=1800,
        )
        backend = AutomaticRecoveryBridgeBackend(
            bridge,
            config_data,
            recovery_config,
            output_dir,
            artifacts,
            mode="task-runtime-snapshot",
            student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=source["snapshot_path"],
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
            runtime_snapshot_attestation=source["attestation"],
        )
        restored = backend.restore_start_state()
        if int(restored.get("timestep", -1)) != T463_TIMESTEP:
            raise RuntimeError("strict_t463_restore_timestep_mismatch")
        if restored.get("scene_metadata", {}).get("model_geometry_sha256") != MODEL_SHA256:
            raise RuntimeError("strict_t463_restore_model_fingerprint_mismatch")
        live_integration = np.asarray(
            restored["snapshot_state"]["mujoco_integration_state"], dtype=np.float64
        )
        if _sha(live_integration) != source["integration_sha256"]:
            raise RuntimeError("strict_t463_restore_integration_state_mismatch")
        exported = bridge.request({"op": "export_runtime_snapshot"})
        if exported.get("status") != "complete":
            raise RuntimeError("strict_t463_runtime_controller_export_unavailable")
        restored_envelope = pickle.loads(exported["payload"])
        restored_internal_hash = _normalized_state_hash(restored_envelope.get("snapshot", {}))
        source_internal_hash = _normalized_state_hash(source["snapshot"])
        if restored_internal_hash != source_internal_hash:
            raise RuntimeError("strict_t463_controller_or_gripper_internal_state_mismatch")
        native = bridge.request({"op": "native_task_status"})
        if native.get("status") != "complete" or native.get("task_success") is not False:
            raise RuntimeError("t463_restored_native_task_predicate_state_unknown_or_terminal")
        preflight = bridge.request({"op": "observe", "include_snapshot_state": True})
        backend._load_context_inputs(preflight)
        backend.latest_data = preflight
        backend.safety_policy["joint_limit_tolerance_policy"] = policy["joint_limit_tolerance_policy"]
        backend.safety_policy["source"] = policy["source"]
        backend.latest_data["native_task_status"] = native
        contacts = bridge.request({"op": "inspect_contacts"})
        backend.latest_data["contacts"] = contacts
        context = backend.current_context()
        backend.safety_policy["source"] = policy["source"]
        # The only mutation above derives held-object geometry from current
        # finger contact; it must reproduce the frozen archived policy exactly.
        if _canonical_sha256(backend.safety_policy) != _canonical_sha256(policy):
            differing = sorted(
                key for key in set(backend.safety_policy) | set(policy)
                if _canonical_sha256({key: backend.safety_policy.get(key)})
                != _canonical_sha256({key: policy.get(key)})
            )
            raise RuntimeError(f"t463_live_policy_fingerprint_mismatch:{differing}")
        grasp = context.metadata.get("held_object_state_evidence", {}).get(OBJECT_ID, {})
        if (
            context.held_object != OBJECT_ID
            or grasp.get("grasp_confirmed") is not True
            or grasp.get("finger_contacts", {}).get("left") is not True
            or grasp.get("finger_contacts", {}).get("right") is not True
            or not _all_robot_contacts_finger_target(contacts, backend)
        ):
            raise RuntimeError("t463_live_finger_only_bilateral_grasp_preflight_failed")
        # Make the exact carried-object policy active for all full-Geom samples.
        backend.safety_policy = json.loads(json.dumps(policy))
        geometry_response = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        initial_geometry = geometry_response.get("safety_geometry")
        if not isinstance(initial_geometry, dict) or initial_geometry.get("status") != "complete":
            raise RuntimeError("t463_initial_full_geometry_preflight_incomplete")
        if initial_geometry.get("forbidden_contacts") or initial_geometry.get("self_contacts") or initial_geometry.get("forbidden_object_contacts") or initial_geometry.get("unallowed_scene_contacts") or initial_geometry.get("unresolved_contacts") or initial_geometry.get("joint_limit_violations"):
            raise RuntimeError("t463_initial_native_operational_safety_preflight_failed")
        initial_clearances = _policy_clearance_summary(initial_geometry)
        required_initial = {
            "moka_pot_2": POT2_NONFINGER_CLEARANCE_M,
            "bottle_of_ginger_beer_1": FORBIDDEN_CLEARANCE_M,
            "white_place_box_1": FORBIDDEN_CLEARANCE_M,
        }
        for object_id, minimum in required_initial.items():
            measured = initial_clearances[object_id]["minimum_distance_m"]
            if measured is None or float(measured) < minimum:
                raise RuntimeError(f"t463_initial_clearance_failed:{object_id}:{measured}:{minimum}")
        for key in ("carried_moka_pot_2_to_bottle_of_ginger_beer_1", "carried_moka_pot_2_to_white_place_box_1"):
            measured = initial_clearances[key]["minimum_distance_m"]
            if measured is None or float(measured) < FORBIDDEN_CLEARANCE_M:
                raise RuntimeError(f"t463_initial_carried_clearance_failed:{key}:{measured}")

        qpos_before = np.asarray(restored["snapshot_state"]["robot_states"][0]["joint_qpos"], dtype=np.float64)
        eef_before = np.asarray(restored["snapshot_state"]["robot_states"][0]["eef_pos"], dtype=np.float64)
        q_cursor_error = float(np.linalg.norm(qpos_before - path_qpos[T463_CURSOR]))
        eef_cursor_position_error = float(np.linalg.norm(eef_before - path_poses[T463_CURSOR, :3]))
        if abs(q_cursor_error - 0.18204520604490024) > 1e-5:
            raise RuntimeError(f"t463_route_cursor_joint_error_reproduction_mismatch:{q_cursor_error}")
        risk_armed = float(initial_geometry.get("minimum_clearance_m") or 0.0) >= RISK_REARM_M
        result["strict_restore_and_preflight"] = {
            "status": "passed",
            "restored_timestep": T463_TIMESTEP,
            "model_geometry_sha256": MODEL_SHA256,
            "integration_state_sha256_source": source["integration_sha256"],
            "integration_state_sha256_restored": _sha(live_integration),
            "runtime_internal_state_hash_expected": source_internal_hash,
            "runtime_internal_state_hash_restored": restored_internal_hash,
            "controller_initial_joint_rad": source["controller_initial_joint_rad"],
            "arm_qpos_rad": qpos_before.tolist(),
            "path_cursor": T463_CURSOR,
            "cursor_qpos_error_rad_frozen_0_18": q_cursor_error,
            "cursor_eef_position_error_m": eef_cursor_position_error,
            "eef_reference_limits_m_rad": [EEF_REFERENCE_LIMIT_M, EEF_ORIENTATION_LIMIT_RAD],
            "grasp": grasp,
            "robot_live_contact_count": len(contacts.get("contacts", [])),
            "all_robot_contacts_finger_target": True,
            "operational_safety_geometry": initial_geometry,
            "initial_clearances_by_object": initial_clearances,
            "risk_hysteresis_state_initially_armed": risk_armed,
            "native_task_success": native.get("task_success"),
            "phase_observed_from_live_context": context.phase,
            "formal_episode_actions_committed": 0,
        }
        _write_json(output_dir / "t463_restore_preflight.json", result["strict_restore_and_preflight"])
        _write_json(output_dir / "safety_policy.json", backend.safety_policy)

        branch_start_payload = Path(source["snapshot_path"]).read_bytes()
        Path(output_dir / "snapshots").mkdir(exist_ok=True)
        (output_dir / "snapshots" / "source_t463.pkl").write_bytes(branch_start_payload)
        images_agentview: list[np.ndarray] = []
        images_wrist: list[np.ndarray] = []
        branch_results = []
        branch_baseline_timestep = T463_TIMESTEP
        total_movement = 0.0
        previous_eef = eef_before.copy()
        previous_margin = float(initial_geometry.get("minimum_clearance_m") or 0.0)
        movement_history: list[float] = []
        if not risk_armed and previous_margin >= RISK_REARM_M:
            risk_armed = True

        for offset in range(MAX_PREVIEW_ACTIONS):
            current = bridge.request({"op": "observe", "include_snapshot_state": True})
            expected_timestep = branch_baseline_timestep + offset
            if int(current.get("timestep", -1)) != expected_timestep:
                raise RuntimeError(f"isolated_branch_timestep_discontinuity:{current.get('timestep')}:{expected_timestep}")
            integration_now = np.asarray(current["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
            snap = bridge.request({"op": "snapshot"})
            if not isinstance(snap.get("snapshot_id"), str):
                raise RuntimeError("isolated_branch_snapshot_id_unavailable")
            probe = bridge.request({
                "op": "osc_prefix_probe",
                "snapshot_id": snap["snapshot_id"],
                "current_state_sha256": _sha(integration_now),
                "jointspace_path_qpos": path_qpos.tolist(),
                "eef_reference_poses_world": path_poses.tolist(),
                "path_start_index": T463_CURSOR + offset,
                "max_control_steps": 1,
                "gripper_action": 0.0,
                "risk_trigger_distance_m": RISK_TRIGGER_M,
                "risk_rearm_distance_m": RISK_REARM_M,
                "risk_crossing_armed": risk_armed,
                "maximum_joint_path_deviation_rad": JOINT_REFERENCE_LIMIT_RAD,
                "maximum_eef_tracking_error_m": EEF_REFERENCE_LIMIT_M,
                "maximum_eef_orientation_error_rad": EEF_ORIENTATION_LIMIT_RAD,
                "qpos_reference_mode": "diagnostic_only",
                "current_minimum_forbidden_clearance_m": previous_margin,
                "safety_geometry_policy": backend.safety_policy,
            })
            if not probe.get("main_state_restored_verified"):
                raise RuntimeError(f"probe_caller_state_restore_failed_at_offset_{offset}")
            if probe.get("executed_steps") != 1 or len(probe.get("steps", [])) != 1:
                branch_results.append({"offset": offset, "probe": probe})
                result["stop_reason"] = probe.get("stop_reason") or probe.get("failure_reason") or "probe_did_not_execute_one_action"
                break
            step = probe["steps"][0]
            if int(step["observation"].get("timestep", -1)) != expected_timestep + 1:
                raise RuntimeError(f"isolated_branch_observation_timestep_mismatch_at_offset_{offset}")
            geometry_ok, geometry_audit = _verify_geometry_substeps(step.get("safety_geometry_substeps", []))
            if probe.get("contact_monitoring", {}).get("status") != "complete":
                result["stop_reason"] = "contact_monitoring_coverage_incomplete"
            if probe.get("contact_monitoring", {}).get("geometry_monitoring", {}).get("status") != "complete":
                result["stop_reason"] = "contact_monitoring_geometry_not_complete"
            if probe.get("safe") is not True:
                result["stop_reason"] = result.get("stop_reason") or "bridge_probe_native_operational_verdict_not_safe"
            eef_errors_ok = (
                float(step["eef_tracking_error_m"]) <= EEF_REFERENCE_LIMIT_M
                and float(step["eef_orientation_error_rad"]) <= EEF_ORIENTATION_LIMIT_RAD
            )
            if not geometry_ok:
                result["stop_reason"] = "full_substep_native_operational_safety_failure"
            elif not eef_errors_ok:
                result["stop_reason"] = "frozen_eef_reference_tracking_limit"
            elif step.get("done"):
                result["stop_reason"] = "environment_done_in_isolated_preview"
            elif probe.get("stop_reason"):
                result["stop_reason"] = probe["stop_reason"]

            # Promote the branch Snapshot only after full substep geometry and
            # EEF tracking checks pass. Each accepted branch step is restored
            # via the Bridge API; no qpos write or historical action replay.
            terminal_payload = step.get("terminal_runtime_snapshot")
            if not isinstance(terminal_payload, bytes) or not terminal_payload:
                raise RuntimeError(f"isolated_branch_terminal_snapshot_missing_at_offset_{offset}")
            if result.get("stop_reason") is None:
                restore_branch = bridge.request({"op": "restore_runtime_snapshot", "payload": terminal_payload})
                if restore_branch.get("status") != "restored":
                    raise RuntimeError(f"isolated_branch_snapshot_restore_failed_at_offset_{offset}:{restore_branch.get('error')}")
                branch_live = bridge.request({"op": "observe", "include_snapshot_state": True})
                if int(branch_live.get("timestep", -1)) != expected_timestep + 1:
                    raise RuntimeError(f"isolated_branch_restore_timestep_mismatch_at_offset_{offset}")
                if _sha(np.asarray(branch_live["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)) != _sha(np.asarray(step["observation"]["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)):
                    raise RuntimeError(f"isolated_branch_restore_integration_state_mismatch_at_offset_{offset}")
                branch_internal = bridge.request({"op": "export_runtime_snapshot"})
                if branch_internal.get("status") != "complete":
                    raise RuntimeError(f"isolated_branch_runtime_internal_state_unavailable_at_offset_{offset}")
                branch_internal_hash = _normalized_state_hash(
                    pickle.loads(branch_internal["payload"]).get("snapshot", {})
                )
                expected_branch_internal_hash = _normalized_state_hash(
                    pickle.loads(terminal_payload).get("snapshot", {})
                )
                if branch_internal_hash != expected_branch_internal_hash:
                    raise RuntimeError(f"isolated_branch_controller_or_gripper_state_mismatch_at_offset_{offset}")
                backend.latest_data = branch_live
                branch_native = bridge.request({"op": "native_task_status"})
                backend.latest_data["native_task_status"] = branch_native
                if (
                    branch_native.get("status") != "complete"
                    or branch_native.get("done") is not False
                    or branch_native.get("task_success") is not False
                ):
                    result["stop_reason"] = "native_task_state_not_pending_after_branch_action"
                branch_contacts = bridge.request({"op": "inspect_contacts"})
                backend.latest_data["contacts"] = branch_contacts
                branch_context = backend.current_context()
                branch_grasp = branch_context.metadata.get("held_object_state_evidence", {}).get(OBJECT_ID, {})
                if (
                    branch_context.held_object != OBJECT_ID
                    or branch_grasp.get("grasp_confirmed") is not True
                    or branch_grasp.get("finger_contacts", {}).get("left") is not True
                    or branch_grasp.get("finger_contacts", {}).get("right") is not True
                    or not _all_robot_contacts_finger_target(branch_contacts, backend)
                ):
                    result["stop_reason"] = "bilateral_finger_only_grasp_not_maintained"
                branch_geometry_response = bridge.request({
                    "op": "observe", "include_snapshot_state": True,
                    "safety_geometry_policy": backend.safety_policy,
                })
                branch_geometry = branch_geometry_response.get("safety_geometry")
                if not isinstance(branch_geometry, dict) or branch_geometry.get("status") != "complete":
                    result["stop_reason"] = "restored_branch_current_geometry_incomplete"
                elif branch_geometry.get("forbidden_contacts") or branch_geometry.get("self_contacts") or branch_geometry.get("forbidden_object_contacts") or branch_geometry.get("unallowed_scene_contacts") or branch_geometry.get("unresolved_contacts") or branch_geometry.get("joint_limit_violations"):
                    result["stop_reason"] = "restored_branch_current_safety_violation"
                elif any(
                    branch_geometry.get("clearances_by_object", {}).get(object_id, {}).get("minimum_distance_m") is None
                    or branch_geometry["clearances_by_object"][object_id]["minimum_distance_m"] < minimum
                    for object_id, minimum in required_initial.items()
                ):
                    result["stop_reason"] = "restored_branch_current_object_clearance_below_frozen_policy"
                current_margin = float(branch_geometry.get("minimum_clearance_m") or 0.0)
                if not risk_armed and current_margin >= RISK_REARM_M:
                    risk_armed = True
                elif risk_armed and current_margin <= RISK_TRIGGER_M:
                    result["stop_reason"] = "risk_boundary_hysteresis_triggered"
                    risk_armed = False
                new_eef = np.asarray(step["actual_eef_position_world_m"], dtype=np.float64)
                movement = float(np.linalg.norm(new_eef - previous_eef))
                total_movement += movement
                movement_history.append(movement)
                previous_eef = new_eef
                previous_margin = current_margin
            else:
                branch_live = None

            observation = step["observation"]
            image_records = observation.get("images", {})
            agentview = _camera_array(image_records["agentview_image"])
            wrist = _camera_array(image_records["robot0_eye_in_hand_image"])
            images_agentview.append(agentview)
            images_wrist.append(wrist)
            contacts_json = step.get("contacts", {})
            action_result = {
                "offset": offset,
                "timestep_before": expected_timestep,
                "timestep_after": int(observation["timestep"]),
                "route_index": int(step["path_index"]),
                "action_7d": step["action"],
                "gripper_action": float(step["action"][6]),
                "reference_qpos_rad": step["reference_qpos"],
                "actual_qpos_rad": step["actual_qpos"],
                "per_joint_error_rad": (np.asarray(step["actual_qpos"]) - np.asarray(step["reference_qpos"])).tolist(),
                "qpos_path_deviation_rad_diagnostic_only": step["joint_path_deviation_rad"],
                "would_exceed_frozen_0_18_rad_threshold": step["joint_path_deviation_exceeds_frozen_reference_limit"],
                "reference_eef_pose": step["reference_eef_pose"],
                "actual_eef_position_world_m": step["actual_eef_position_world_m"],
                "actual_eef_quaternion_wxyz": step["actual_eef_quaternion_wxyz"],
                "eef_position_error_m": step["eef_tracking_error_m"],
                "eef_orientation_error_rad": step["eef_orientation_error_rad"],
                "eef_movement_since_previous_action_m": movement_history[-1] if movement_history else None,
                "reward": step["reward"],
                "done": step["done"],
                "native_task_success": observation.get("task_success"),
                "native_task_timestep": observation.get("timestep"),
                "live_grasp_after_restore": branch_grasp if branch_live is not None else None,
                "contact_count": len(contacts_json.get("contacts", [])),
                "contact_identities": contacts_json.get("contacts", []),
                "substep_count": len(step.get("safety_geometry_substeps", [])),
                "substep_geometry_audit": geometry_audit,
                "safety_geometry_substeps": step.get("safety_geometry_substeps", []),
                "risk_margin_after_m": previous_margin,
                "risk_hysteresis_armed_after": risk_armed,
                "reward_done_native_predicate_observation": {
                    "reward": step["reward"],
                    "done": step["done"],
                    "native_task_success": observation.get("task_success"),
                    "timestep": observation.get("timestep"),
                },
                "native_task_status_after_branch_restore": branch_native if branch_live is not None else None,
                "camera_npz_frame_index": len(images_agentview) - 1,
            }
            branch_results.append(action_result)
            result["actions"].append(action_result)
            result["isolated_branch_actions_executed"] = len(result["actions"])
            Path(output_dir / "snapshots" / f"branch_t{expected_timestep + 1:06d}.pkl").write_bytes(terminal_payload)
            if (
                len(movement_history) >= MAX_PREVIEW_ACTIONS
                and sum(movement_history[-MAX_PREVIEW_ACTIONS:]) < 0.002
            ):
                result["stop_reason"] = "motion_stagnation_conservative_8_action_window"
            if result.get("stop_reason") is not None:
                break

        result["status"] = "safe_progress" if result["actions"] and result.get("stop_reason") is None else "stopped_or_failed"
        result["all_executed_actions_native_operational_safe"] = bool(
            len(result["actions"]) > 0
            and all(len(action.get("safety_geometry_substeps", [])) == EXPECTED_SUBSTEPS
                    and not action.get("substep_geometry_audit", {}).get("failures")
                    and not action.get("done") for action in result["actions"])
        )
        result["all_executed_actions_bilateral_grasp_maintained"] = bool(
            result["actions"] and all(
                action.get("live_grasp_after_restore", {}).get("grasp_confirmed") is True
                and action.get("live_grasp_after_restore", {}).get("finger_contacts", {}).get("left") is True
                and action.get("live_grasp_after_restore", {}).get("finger_contacts", {}).get("right") is True
                for action in result["actions"]
            )
        )
        result["total_eef_translation_m"] = total_movement
        result["motion_stagnation_check"] = {
            "window_actions": MAX_PREVIEW_ACTIONS,
            "conservative_minimum_progress_m": 0.002,
            "observed_progress_m": sum(movement_history[-MAX_PREVIEW_ACTIONS:]),
            "would_stop_for_stagnation": bool(
                len(movement_history) >= MAX_PREVIEW_ACTIONS
                and sum(movement_history[-MAX_PREVIEW_ACTIONS:]) < 0.002
            ),
            "eef_movement_per_action_m": movement_history,
        }
        result["risk_hysteresis"] = {"trigger_m": RISK_TRIGGER_M, "rearm_m": RISK_REARM_M, "final_armed": risk_armed}
        result["native_status_after_preview"] = bridge.request({"op": "native_task_status"})
        result["physical_branch_snapshot_chain"] = [
            str(output_dir / "snapshots" / f"branch_t{T463_TIMESTEP + index + 1:06d}.pkl")
            for index in range(len(result["actions"]))
        ]
        np.savez_compressed(
            output_dir / "isolated_observation_action.npz",
            actions=np.asarray([item["action_7d"] for item in result["actions"]], dtype=np.float32),
            actual_qpos=np.asarray([item["actual_qpos_rad"] for item in result["actions"]], dtype=np.float64),
            reference_qpos=np.asarray([item["reference_qpos_rad"] for item in result["actions"]], dtype=np.float64),
            actual_eef_position=np.asarray([item["actual_eef_position_world_m"] for item in result["actions"]], dtype=np.float64),
            reference_eef_pose=np.asarray([item["reference_eef_pose"] for item in result["actions"]], dtype=np.float64),
            agentview=np.asarray(images_agentview, dtype=np.uint8),
            wrist=np.asarray(images_wrist, dtype=np.uint8),
        )
        _save_video(images_agentview, output_dir / "agentview_preview.mp4")
        _save_video(images_wrist, output_dir / "wrist_preview.mp4")
        result["observation_action_archive"] = str(output_dir / "isolated_observation_action.npz")
        result["videos"] = [str(output_dir / "agentview_preview.mp4"), str(output_dir / "wrist_preview.mp4")]

        # Restore the actual t463 source state exactly. This bridge is dedicated
        # to the isolated preview and is torn down afterward; the original
        # archived Episode and Snapshot are read-only inputs.
        restore_source = bridge.request({"op": "restore_runtime_snapshot", "payload": branch_start_payload})
        if restore_source.get("status") != "restored":
            raise RuntimeError(f"original_t463_source_restore_failed_at_end:{restore_source.get('error')}")
        restored_final = bridge.request({"op": "observe", "include_snapshot_state": True})
        final_export = bridge.request({"op": "export_runtime_snapshot"})
        final_envelope = pickle.loads(final_export["payload"])
        final_internal_hash = _normalized_state_hash(final_envelope.get("snapshot", {}))
        result["main_state_restored_to_t463"] = bool(
            int(restored_final.get("timestep", -1)) == T463_TIMESTEP
            and _sha(np.asarray(restored_final["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)) == source["integration_sha256"]
            and final_internal_hash == source_internal_hash
        )
        result["final_restore_evidence"] = {
            "timestep": restored_final.get("timestep"),
            "integration_state_sha256": _sha(np.asarray(restored_final["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)),
            "expected_integration_state_sha256": source["integration_sha256"],
            "runtime_internal_state_sha256": final_internal_hash,
            "expected_runtime_internal_state_sha256": source_internal_hash,
        }
        if not result["main_state_restored_to_t463"]:
            result["status"] = "failed_restore_integrity"
            result["all_executed_actions_native_operational_safe"] = False
            raise RuntimeError("isolated_preview_did_not_restore_original_t463_state")
    except BaseException as exc:
        result["status"] = "failed_or_interrupted"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        result["formal_episode_physical_actions_committed"] = 0
        try:
            # If the source was restored at least once, fail closed by restoring
            # it again before bridge teardown; the exact state is separately
            # verified in the successful path.
            if bridge.connection is not None and source.get("snapshot_path"):
                restore = bridge.request({"op": "restore_runtime_snapshot", "payload": Path(source["snapshot_path"]).read_bytes()})
                result["best_effort_source_restore_status"] = restore.get("status")
        except Exception as restore_exc:
            result["best_effort_source_restore_error"] = f"{type(restore_exc).__name__}: {restore_exc}"
    finally:
        bridge.close()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
    result["diagnostic_archive"] = str(output_dir)
    _write_json(output_dir / "preview_result.json", result)
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38764)
    args = parser.parse_args()
    try:
        result = run_preview(output_dir=args.output_dir, python=args.python, bridge_port=args.bridge_port)
    except Exception as exc:
        result = {"status": "failed_before_archive", "failure_reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps({
        "status": result.get("status"),
        "failure_reason": result.get("failure_reason"),
        "stop_reason": result.get("stop_reason"),
        "diagnostic_archive": result.get("diagnostic_archive"),
        "strict_restore_status": result.get("strict_restore_and_preflight", {}).get("status"),
        "isolated_actions": result.get("isolated_branch_actions_executed", 0),
        "formal_episode_actions_committed": result.get("formal_episode_physical_actions_committed", 0),
        "native_operational_safe": result.get("all_executed_actions_native_operational_safe"),
        "bilateral_grasp_maintained": result.get("all_executed_actions_bilateral_grasp_maintained"),
        "main_state_restored_to_t463": result.get("main_state_restored_to_t463"),
        "actions": [{
            "timestep_after": item.get("timestep_after"),
            "route_index": item.get("route_index"),
            "qpos_deviation_rad_diagnostic_only": item.get("qpos_path_deviation_rad_diagnostic_only"),
            "eef_position_error_m": item.get("eef_position_error_m"),
            "eef_orientation_error_rad": item.get("eef_orientation_error_rad"),
            "substeps": item.get("substep_count"),
        } for item in result.get("actions", [])],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "safe_progress" and result.get("main_state_restored_to_t463") else 2


if __name__ == "__main__":
    raise SystemExit(main())
