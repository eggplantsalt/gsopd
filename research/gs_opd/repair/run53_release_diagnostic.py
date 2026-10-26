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

"""Replay run53's own successful suffix, then diagnose physical release."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from research.gs_opd.repair.task_success_first_grasp_search import (
    BRIDGE_AUTHKEY,
    BridgeSession,
    _image,
    _verified_resume_runtime_snapshot,
    _write_json,
)
from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action


ROLLOUT_ROOT = Path("/opt/data/private/gs_opd/rollouts")
RUN19 = ROLLOUT_ROOT / "task_success_first_cross_level_L1T2_20261026_run19_L0_success_migration/obstacle_avoidance_L1T2/stage_00/task_success_first_executor_obstacle_avoidance_L1T2_20261026_065056_331307395"
RUN46 = ROLLOUT_ROOT / "task_success_first_cross_level_L1T2_20261026_run46_shared_placement_transfer/task_success_first_executor_obstacle_avoidance_L1T2_20261026_133025_720979512"
RUN47 = ROLLOUT_ROOT / "task_success_first_cross_level_L1T2_20261026_run47_release_partial_goal/task_success_first_executor_obstacle_avoidance_L1T2_20261026_134434_039862217"
RUN48 = ROLLOUT_ROOT / "task_success_first_cross_level_L1T2_20261026_run48_second_object_from_partial_prefix/task_success_first_executor_obstacle_avoidance_L1T2_20261026_134731_258088424"
RUN52 = ROLLOUT_ROOT / "keypose_driven_L1T2_20261026_run52/task_success_first_executor_obstacle_avoidance_L1T2_20261026_142348_459675599"
RUN53 = ROLLOUT_ROOT / "keypose_driven_L1T2_20261026_run53_suffix_replan/task_success_first_executor_obstacle_avoidance_L1T2_20261026_143005_978311542"

INITIAL_SNAPSHOT = RUN19 / "expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L1T2_20261026_065110_007137754/source_initial_runtime_snapshot.pkl"
POT1_LIFTED_SNAPSHOT = RUN19 / "expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L1T2_20261026_065110_007137754/task_success_candidate_lifted_runtime_snapshot.pkl"
RUN46_BRANCH = RUN46 / "expert_native_goal_stage_control/placement_branch_03_region_layout_19"
RUN47_SEGMENT = RUN47 / "expert_native_release_resume"
RUN48_GRASP = RUN48 / "expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L1T2_20261026_134743_113432329"
RUN48_LIFTED_SNAPSHOT = RUN48_GRASP / "task_success_candidate_lifted_runtime_snapshot.pkl"
RUN52_BRANCH = RUN52 / "expert_native_goal_stage_control/placement_branch_00_region_layout_00"
RUN53_BRANCH = RUN53 / "expert_native_goal_stage_control/placement_branch_00_region_layout_00"
PRE_SUCCESS_SNAPSHOT = RUN52_BRANCH / "native_goal_transport_failed_runtime_snapshot.pkl"
RUN53_RECORDS = RUN53_BRANCH / "physical_step_records.jsonl.gz"

EXPECTED_MODEL_SHA = "6996aaa940b011861510ebd1911ae19f7db66c54e189ca4c107d372b06033edf"
EXPECTED_BDDL_SHA = "88c64be300fa5934a096c6ef545514e1f541cc67c7c1f09bed1bc4f7e6583bc6"
EXPECTED_INSTRUCTION = "put both moka pots on the stove"
NATIVE_GOALS = {
    "pot1_on": ["on", "moka_pot_1", "flat_stove_1_cook_region"],
    "pot2_on": ["on", "moka_pot_2", "flat_stove_1_cook_region"],
    "stove_on": ["turnon", "flat_stove_1"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integration_state_sha(snapshot_state: dict[str, Any]) -> str:
    state = np.asarray(snapshot_state.get("mujoco_integration_state", []), dtype=np.float64)
    if state.ndim != 1 or not state.size or not np.isfinite(state).all():
        raise ValueError("integration_state_is_missing_or_invalid")
    return hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_settling(
    snapshot_rows: list[dict[str, Any]], body_ids: dict[str, int]
) -> dict[str, Any]:
    """Measure object motion over the settling window without changing thresholds."""
    output: dict[str, Any] = {"sample_count": len(snapshot_rows), "objects": {}}
    for label, body_id in body_ids.items():
        poses = []
        linear_speeds = []
        angular_speeds = []
        for snapshot in snapshot_rows:
            matches = [
                state for state in snapshot.get("object_states", {}).values()
                if int(state.get("body_id", -1)) == int(body_id)
            ]
            if len(matches) != 1:
                raise ValueError(f"settle_body_{body_id}_resolved_to_{len(matches)}_states")
            state = matches[0]
            poses.append(np.asarray(state["root_pos"], dtype=np.float64))
            qvel = np.asarray(state.get("joint_qvel", []), dtype=np.float64)
            if qvel.shape == (6,) and np.isfinite(qvel).all():
                linear_speeds.append(float(np.linalg.norm(qvel[:3])))
                angular_speeds.append(float(np.linalg.norm(qvel[3:])))
        pose_array = np.asarray(poses, dtype=np.float64)
        if pose_array.ndim != 2 or pose_array.shape[1:] != (3,) or not np.isfinite(pose_array).all():
            raise ValueError(f"settle_body_{body_id}_poses_are_invalid")
        output["objects"][label] = {
            "body_id": int(body_id),
            "position_span_m": float(np.max(np.linalg.norm(pose_array - pose_array[0], axis=1))),
            "first_position_world_m": pose_array[0].tolist(),
            "last_position_world_m": pose_array[-1].tolist(),
            "maximum_linear_speed_m_s": max(linear_speeds, default=None),
            "maximum_angular_speed_rad_s": max(angular_speeds, default=None),
            "velocity_samples_available": len(linear_speeds) == len(snapshot_rows),
        }
    return output


def contact_pair_names(contact_result: dict[str, Any]) -> set[tuple[str, str]]:
    pairs = set()
    for contact in contact_result.get("contacts", []):
        first = contact.get("geom1", {}).get("name")
        second = contact.get("geom2", {}).get("name")
        if isinstance(first, str) and isinstance(second, str):
            pairs.add(tuple(sorted((first, second))))
    return pairs


def normalize_grasp_binding(binding: dict[str, Any]) -> dict[str, Any]:
    """Adapt a semantic root binding to the grasp-query body's explicit fields."""
    normalized = dict(binding)
    body_id = normalized.get("body_id", normalized.get("root_body_id"))
    body_name = normalized.get("body_name", normalized.get("root_body_name"))
    if body_id is None or not isinstance(normalized.get("geom_ids"), list) or not normalized["geom_ids"]:
        raise ValueError("semantic_binding_cannot_be_normalized_for_grasp_query")
    normalized["body_id"] = int(body_id)
    normalized["body_name"] = body_name
    return normalized


def has_target_hand_contact(contact_result: dict[str, Any], binding: dict[str, Any]) -> bool:
    """Return whether a compiled gripper geom contacts the bound target geom set."""
    target_names = set(binding.get("geom_names", []))
    for first, second in contact_pair_names(contact_result):
        if first in target_names and "gripper" in second.lower():
            return True
        if second in target_names and "gripper" in first.lower():
            return True
    return False


def records_cover_range(records: list[dict[str, Any]], start_t: int, end_t: int) -> bool:
    timesteps = [int(row["timestep"]) for row in records]
    return (
        len(records) == end_t - start_t + 1
        and timesteps == list(range(start_t, end_t + 1))
        and all("action_7d" in row and "snapshot_state" in row and "state" in row for row in records)
    )


def _segment_entry(name: str, records_path: Path, start_t: int, end_t: int) -> dict[str, Any]:
    records = read_gzip_jsonl(records_path)
    timesteps = [int(row["timestep"]) for row in records]
    return {
        "name": name,
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "record_count": len(records),
        "expected_timestep_range_inclusive": [start_t, end_t],
        "recorded_timestep_range_inclusive": [min(timesteps), max(timesteps)] if timesteps else None,
        "action_observation_state_rows_contiguous": records_cover_range(records, start_t, end_t),
        "full_raw_policy_observation_saved": all("observation" in row for row in records),
        "records": records,
    }


def audit_source_chain() -> dict[str, Any]:
    """Audit the selected L1T2 branch lineage from reset through run53 success."""
    segments = [
        _segment_entry("reset_to_pot1_lift_run19", RUN19 / "expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L1T2_20261026_065110_007137754/physical_step_records.jsonl.gz", 1, 182),
        _segment_entry("pot1_transport_placement_run46_branch03", RUN46_BRANCH / "physical_step_records.jsonl.gz", 183, 542),
        _segment_entry("pot1_release_resume_run47", RUN47_SEGMENT / "physical_step_records.jsonl.gz", 543, 598),
        _segment_entry("pot2_grasp_run48", RUN48_GRASP / "physical_step_records.jsonl.gz", 599, 857),
        _segment_entry("pot2_transport_run52", RUN52_BRANCH / "physical_step_records.jsonl.gz", 858, 1217),
        _segment_entry("native_success_suffix_run53", RUN53_RECORDS, 1218, 1335),
    ]
    segment_summaries = []
    for segment in segments:
        row = {key: value for key, value in segment.items() if key != "records"}
        row["endpoint_task_success"] = segment["records"][-1].get("task_success") if segment["records"] else None
        row["endpoint_done"] = segment["records"][-1].get("done") if segment["records"] else None
        segment_summaries.append(row)

    snapshot_files = [
        INITIAL_SNAPSHOT,
        POT1_LIFTED_SNAPSHOT,
        RUN46_BRANCH / "native_goal_transport_failed_runtime_snapshot.pkl",
        RUN47_SEGMENT / "native_goal_placement_terminal_runtime_snapshot.pkl",
        RUN48_LIFTED_SNAPSHOT,
        PRE_SUCCESS_SNAPSHOT,
    ]
    snapshot_links = [
        {"from": "run19 initial reset", "to": "run19 action trace", "sha256": sha256_file(INITIAL_SNAPSHOT), "link": "source_initial_runtime_snapshot; timestep 0"},
        {"from": "run19 Pot1 lift", "to": "run46 branch03", "sha256": sha256_file(POT1_LIFTED_SNAPSHOT), "link": "run46 report verifies this exact payload; timestep 182"},
        {"from": "run46 branch03", "to": "run47", "sha256": sha256_file(RUN46_BRANCH / "native_goal_transport_failed_runtime_snapshot.pkl"), "link": "run47 report verifies this exact payload; timestep 542"},
        {"from": "run47", "to": "run48", "sha256": sha256_file(RUN47_SEGMENT / "native_goal_placement_terminal_runtime_snapshot.pkl"), "link": "run48 interruption recovery names this payload; timestep 598"},
        {"from": "run48 Pot2 lift", "to": "run52", "sha256": sha256_file(RUN48_LIFTED_SNAPSHOT), "link": "run52 report verifies this exact payload; timestep 857"},
        {"from": "run52", "to": "run53", "sha256": sha256_file(PRE_SUCCESS_SNAPSHOT), "link": "run53 report verifies this exact payload; timestep 1217"},
    ]
    report_paths = {
        "run19": RUN19 / "run_report.json",
        "run46": RUN46 / "run_report.json",
        "run47": RUN47 / "run_report.json",
        "run48": RUN48 / "run_report.json",
        "run48_interruption_recovery": RUN48 / "interruption_recovery.json",
        "run52": RUN52 / "run_report.json",
        "run53": RUN53 / "run_report.json",
    }
    report_data = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in report_paths.items()
        if path.is_file()
    }
    link_checks = [
        json.loads((RUN19 / "expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L1T2_20261026_065110_007137754/source_snapshot_manifest.json").read_text(encoding="utf-8"))["snapshots"]["initial"]["payload_sha256"] == sha256_file(INITIAL_SNAPSHOT),
        report_data["run46"]["resumed_from_runtime_snapshot"]["payload_sha256"] == sha256_file(POT1_LIFTED_SNAPSHOT),
        report_data["run47"]["resumed_from_native_goal_contact_snapshot"]["snapshot_sha256"] == sha256_file(RUN46_BRANCH / "native_goal_transport_failed_runtime_snapshot.pkl"),
        report_data["run48_interruption_recovery"]["resumed_from_runtime_snapshot"]["payload_sha256"] == sha256_file(RUN47_SEGMENT / "native_goal_placement_terminal_runtime_snapshot.pkl"),
        report_data["run52"]["resumed_from_runtime_snapshot"]["payload_sha256"] == sha256_file(RUN48_LIFTED_SNAPSHOT),
        report_data["run53"]["resumed_from_runtime_snapshot"]["payload_sha256"] == sha256_file(PRE_SUCCESS_SNAPSHOT),
    ]
    expected_boundary_state_shas = [
        report_data["run46"]["resumed_from_runtime_snapshot"]["integration_state_sha256"],
        report_data["run47"]["resumed_from_native_goal_contact_snapshot"]["integration_state_sha256"],
        report_data["run48_interruption_recovery"]["resumed_from_runtime_snapshot"]["integration_state_sha256"],
        report_data["run52"]["resumed_from_runtime_snapshot"]["integration_state_sha256"],
        report_data["run53"]["resumed_from_runtime_snapshot"]["integration_state_sha256"],
    ]
    actual_boundary_state_shas = [
        integration_state_sha(segments[index]["records"][-1]["snapshot_state"])
        for index in range(5)
    ]
    boundary_state_checks = [
        actual == expected
        for actual, expected in zip(actual_boundary_state_shas, expected_boundary_state_shas, strict=True)
    ]
    return {
        "task": "obstacle_avoidance/L1T2 seed=0 init_state=0",
        "model_geometry_sha256": EXPECTED_MODEL_SHA,
        "bddl_sha256": EXPECTED_BDDL_SHA,
        "selected_physical_lineage_timestep_range": [0, 1335],
        "segments": segment_summaries,
        "segment_timestep_coverage_contiguous": all(row["action_observation_state_rows_contiguous"] for row in segment_summaries),
        "snapshot_links": snapshot_links,
        "all_snapshot_payloads_present": all(path.is_file() for path in snapshot_files),
        "snapshot_link_payload_hashes_match_resuming_reports": all(link_checks),
        "boundary_integration_state_hashes_match_resuming_reports": all(boundary_state_checks),
        "boundary_integration_state_hashes": [
            {"segment_terminal_timestep": [182, 542, 598, 857, 1217][index], "recorded_sha256": actual_boundary_state_shas[index], "resuming_report_sha256": expected_boundary_state_shas[index], "match": boundary_state_checks[index]}
            for index in range(5)
        ],
        "reports": {name: {"path": str(path), "exists": path.is_file()} for name, path in report_paths.items()},
        "run48_root_report_missing_but_durable_interruption_index_present": not (RUN48 / "run_report.json").exists() and (RUN48 / "interruption_recovery.json").is_file() and (RUN48 / "run48_archive_index.json").is_file(),
        "observations": {
            "per_step_robot_object_physical_state_saved": all(row["action_observation_state_rows_contiguous"] for row in segment_summaries),
            "dual_camera_video_available_in_all_selected_segments": all(
                (Path(row["records_path"]).parent / "dual_camera_agentview_wrist_20fps.mp4").is_file()
                for row in segment_summaries
            ),
            "full_raw_policy_observation_saved": all(row["full_raw_policy_observation_saved"] for row in segment_summaries),
            "interpretation": "actions and physical robot/object state rows are contiguous; camera videos exist, but full raw policy Observation tensors are not present in the per-step records",
        },
        "safety_certified": False,
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _native_values(status: dict[str, Any]) -> dict[str, bool | None]:
    values = {
        tuple(str(token).lower() for token in row.get("predicate", [])): row.get("value")
        for row in status.get("native_goal_values", [])
    }
    return {
        name: values.get(tuple(predicate))
        for name, predicate in NATIVE_GOALS.items()
    }


def _object_state(snapshot: dict[str, Any], body_id: int) -> dict[str, Any]:
    matches = [
        row for row in snapshot.get("object_states", {}).values()
        if int(row.get("body_id", -1)) == int(body_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"body_{body_id}_resolved_to_{len(matches)}_object_states")
    return matches[0]


def _export_snapshot(bridge: BridgeSession, path: Path) -> dict[str, Any]:
    result = bridge.request({"op": "export_runtime_snapshot"})
    if result.get("status") != "complete" or not isinstance(result.get("payload"), bytes):
        raise RuntimeError(f"runtime_snapshot_export_failed:{result.get('error') or result.get('status')}")
    path.write_bytes(result["payload"])
    return {"path": str(path), "sha256": sha256_file(path), "timestep": result.get("snapshot_state", {}).get("task_bookkeeping", {}).get("timestep")}


def run_diagnostic(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    records_path = output_dir / "diagnostic_physical_step_records.jsonl"
    frames_dir = output_dir / "dual_camera_frames"
    frames_dir.mkdir()
    chain = audit_source_chain()
    _write_json(output_dir / "source_chain_audit.json", chain)

    source_report = json.loads((RUN53 / "run_report.json").read_text(encoding="utf-8"))
    source_descriptor = _verified_resume_runtime_snapshot(
        PRE_SUCCESS_SNAPSHOT,
        benchmark="obstacle_avoidance", level=1, task_id=2, seed=0, init_state=0,
        model_sha=EXPECTED_MODEL_SHA, bddl_sha=EXPECTED_BDDL_SHA,
        instruction=EXPECTED_INSTRUCTION,
    )
    expected_source_sha = source_report["resumed_from_runtime_snapshot"]["payload_sha256"]
    if source_descriptor["payload_sha256"] != expected_source_sha:
        raise RuntimeError("run53_pre_success_snapshot_sha_does_not_match_original_report")
    source_actions = read_gzip_jsonl(RUN53_RECORDS)
    if len(source_actions) != 118 or source_actions[-1].get("task_success") is not True:
        raise RuntimeError("run53_source_suffix_is_not_the_expected_118_step_native_success_trace")
    (output_dir / "pre_success_source_runtime_snapshot.pkl").write_bytes(source_descriptor["payload"])

    diagnostic_horizon = int(source_report["budgets"]["demonstration_horizon"])
    if diagnostic_horizon != int(source_descriptor["scene_metadata"].get("bridge_horizon", -1)):
        raise RuntimeError("run53_snapshot_and_report_horizons_do_not_match")
    args = SimpleNamespace(
        benchmark="obstacle_avoidance", level=1, task_id=2, init_state=0, seed=0,
        camera_resolution=256, demonstration_horizon=diagnostic_horizon,
    )
    bridge = BridgeSession(("127.0.0.1", _free_port()), output_dir / "safety_bridge.log", args)
    video = None
    frame_index = []
    action_index = 0
    source_matches = []
    phase_contacts: dict[str, list[dict[str, Any]]] = {"native_success": [], "release": [], "retreat": [], "settle": []}
    native_history: dict[str, list[dict[str, bool | None]]] = {"release": [], "retreat": [], "settle": []}
    phase_status_path = output_dir / "diagnostic_phase_status.jsonl"
    settle_rows = []
    pot_bindings: dict[str, dict[str, Any]] = {}
    goal_status_before_release: dict[str, Any] | None = None
    release_grasp_states = []
    retreat_action_count = 0
    post_success_physics_steps = 0

    def append_step(phase: str, action: np.ndarray, source_row: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal action_index, video
        reply = bridge.request({
            "op": "step",
            "action": np.asarray(action, dtype=np.float32).tolist(),
            "include_snapshot_state": True,
            "include_contacts": True,
        })
        action_index += 1
        snapshot = reply.get("snapshot_state", {})
        expected_state = None if source_row is None else np.asarray(
            source_row.get("snapshot_state", {}).get("mujoco_integration_state", []), dtype=np.float64
        )
        actual_state = np.asarray(snapshot.get("mujoco_integration_state", []), dtype=np.float64)
        if source_row is not None:
            max_delta = (
                float(np.max(np.abs(actual_state - expected_state)))
                if actual_state.shape == expected_state.shape and actual_state.size else None
            )
            source_matches.append({
                "source_timestep": source_row.get("timestep"),
                "replayed_timestep": reply.get("timestep"),
                "timestep_matches": int(reply.get("timestep", -1)) == int(source_row.get("timestep", -2)),
                "integration_state_shape_matches": actual_state.shape == expected_state.shape,
                "integration_state_max_abs_delta": max_delta,
                "integration_state_exact": bool(
                    actual_state.shape == expected_state.shape and np.array_equal(actual_state, expected_state)
                ),
            })
        agent = _image(reply, "agentview_image")
        wrist = _image(reply, "robot0_eye_in_hand_image")
        if video is None:
            video = cv2.VideoWriter(
                str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
                (agent.shape[1] + wrist.shape[1], agent.shape[0]),
            )
            if not video.isOpened():
                raise RuntimeError("diagnostic_dual_camera_video_writer_failed")
        combined = np.concatenate((agent, wrist), axis=1)
        video.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        frame_index.append({"frame": len(frame_index), "action_index": action_index - 1, "phase": phase, "timestep": reply.get("timestep")})
        cv2.imwrite(str(frames_dir / f"frame_{len(frame_index)-1:06d}_agentview.jpg"), cv2.cvtColor(agent, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frames_dir / f"frame_{len(frame_index)-1:06d}_wrist.jpg"), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
        record = {
            "action_index": action_index - 1,
            "phase": phase,
            "action_7d": np.asarray(action, dtype=np.float32).tolist(),
            "state": reply.get("state", {}),
            "snapshot_state": snapshot,
            "contacts": reply.get("contacts", {}),
            "substep_contact_capture_complete": False,
            "contact_sampling": "end_of_control_step_only",
            "safety_certified": False,
            "reward": reply.get("reward", 0.0),
            "done": reply.get("done", False),
            "task_success": reply.get("task_success"),
            "timestep": reply.get("timestep"),
            "source_timestep": source_row.get("timestep") if source_row else None,
            "replay_integration_state_exact": source_matches[-1]["integration_state_exact"] if source_row else None,
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        phase_contacts.setdefault(phase, []).append(reply.get("contacts", {}))
        return {"reply": reply, "record": record}

    try:
        bridge.start()
        initial = bridge.request({"op": "observe", "include_snapshot_state": True})
        scene = initial.get("scene_metadata", {})
        if scene.get("instruction") != EXPECTED_INSTRUCTION or scene.get("level") != 1 or scene.get("task_id") != 2:
            raise RuntimeError(f"diagnostic_bridge_scene_mismatch:{scene}")
        reset_reply = bridge.request({"op": "reset"})
        reset_scene = reset_reply.get("scene_metadata", {})
        if reset_scene.get("model_geometry_sha256") != EXPECTED_MODEL_SHA:
            raise RuntimeError(
                "post_reset_model_fingerprint_mismatch:"
                f"{reset_scene.get('model_geometry_sha256')}"
            )
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": source_descriptor["payload"]})
        if restored.get("status") != "restored" or restored.get("model_geometry_sha256") != EXPECTED_MODEL_SHA:
            raise RuntimeError(f"pre_success_snapshot_restore_failed:{restored.get('error') or restored.get('status')}")
        restored_state = np.asarray(restored.get("snapshot_state", {}).get("mujoco_integration_state", []), dtype=np.float64)
        if not np.array_equal(restored_state, source_descriptor["snapshot_state"]["mujoco_integration_state"]):
            raise RuntimeError("pre_success_snapshot_restore_not_bitwise_exact")
        _export_snapshot(bridge, output_dir / "replay_start_runtime_snapshot.pkl")

        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        bindings = semantic.get("bindings", {})
        for object_id in ("moka_pot_1", "moka_pot_2"):
            binding = bindings.get(object_id)
            if not isinstance(binding, dict) or not binding.get("geom_ids"):
                raise RuntimeError(f"active_semantic_binding_missing:{object_id}")
            pot_bindings[object_id] = normalize_grasp_binding(binding)

        for index, row in enumerate(source_actions):
            append_step("replay_run53_success_suffix", np.asarray(row["action_7d"], dtype=np.float32), row)
            if index % 16 == 15:
                _write_json(output_dir / "replay_progress.json", {
                    "replayed_source_actions": index + 1,
                    "expected_source_actions": len(source_actions),
                    "current_timestep": row.get("timestep"),
                    "all_replay_states_exact_so_far": all(item["integration_state_exact"] for item in source_matches),
                })
        success_status = bridge.request({"op": "native_task_status"})
        phase_contacts["native_success"].append(bridge.request({"op": "inspect_contacts"}))
        values_at_success = _native_values(success_status)
        if success_status.get("task_success") is not True or not all(values_at_success.get(key) is True for key in ("pot1_on", "pot2_on", "stove_on")):
            raise RuntimeError(f"replayed_native_success_not_reproduced:{success_status}")
        goal_status_before_release = success_status
        release_start = bridge.request({"op": "observe", "include_snapshot_state": True})
        release_start_t = int(release_start.get("timestep", -1))
        _export_snapshot(bridge, output_dir / "native_success_before_release_runtime_snapshot.pkl")
        pre_release_grasp = bridge.request({
            "op": "grasp_state", "carried_binding": pot_bindings["moka_pot_2"],
            "expected_instruction": EXPECTED_INSTRUCTION,
        })
        source_terminal = source_actions[-1]
        source_terminal_state = np.asarray(source_terminal["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
        replay_terminal_state = np.asarray(release_start["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
        if not np.array_equal(source_terminal_state, replay_terminal_state):
            raise RuntimeError("run53_success_terminal_did_not_replay_exactly")

        # Continue only this independent Bridge after native done; the BDDL checker remains unchanged.
        released = False
        for _ in range(16):
            item = append_step("release", np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            post_success_physics_steps += 1
            native = bridge.request({"op": "native_task_status"})
            native_history["release"].append(_native_values(native))
            grasp = bridge.request({
                "op": "grasp_state", "carried_binding": pot_bindings["moka_pot_2"],
                "expected_instruction": EXPECTED_INSTRUCTION,
            })
            contacts = bridge.request({"op": "inspect_contacts"})
            release_grasp_states.append(grasp)
            with phase_status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"timestep": item["record"]["timestep"], "phase": "release", "native_task_status": native, "grasp_state": grasp, "contacts": contacts}, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if grasp.get("status") != "unavailable" and grasp.get("grasp_confirmed") is False and not has_target_hand_contact(contacts, pot_bindings["moka_pot_2"]):
                released = True
                break
        _export_snapshot(bridge, output_dir / "post_release_runtime_snapshot.pkl")

        post_release = bridge.request({"op": "observe", "include_snapshot_state": True})
        robot = post_release["snapshot_state"]["robot_states"][0]
        current_eef = np.asarray(robot["eef_pos"], dtype=np.float64)
        current_quat = np.asarray(robot["eef_quat"], dtype=np.float64)
        retreat_target = current_eef + np.asarray([0.0, 0.0, 0.10])
        controller = bridge.request({"op": "repair_context"})
        controller_fields = controller.get("robots", [{}])[0].get("fields")
        if not isinstance(controller_fields, dict):
            raise RuntimeError("active_OSC_controller_fields_unavailable_for_diagnostic_retreat")
        for _ in range(32):
            observed = bridge.request({"op": "observe", "include_snapshot_state": True})
            robot = observed["snapshot_state"]["robot_states"][0]
            position = np.asarray(robot["eef_pos"], dtype=np.float64)
            quaternion = np.asarray(robot["eef_quat"], dtype=np.float64)
            if float(np.linalg.norm(retreat_target - position)) <= 0.006:
                break
            action = world_pose_tracking_action(
                position, quaternion, retreat_target, current_quat,
                controller_fields, gripper_action=-1.0,
            )
            append_step("retreat", action)
            retreat_action_count += 1
            post_success_physics_steps += 1
            native = bridge.request({"op": "native_task_status"})
            native_history["retreat"].append(_native_values(native))
            with phase_status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"timestep": native.get("timestep"), "phase": "retreat", "native_task_status": native}, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        _export_snapshot(bridge, output_dir / "post_retreat_runtime_snapshot.pkl")

        for _ in range(20):
            item = append_step("settle", np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            post_success_physics_steps += 1
            native = bridge.request({"op": "native_task_status"})
            native_history["settle"].append(_native_values(native))
            with phase_status_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"timestep": native.get("timestep"), "phase": "settle", "native_task_status": native}, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            settle_rows.append(item["reply"].get("snapshot_state", {}))
        _export_snapshot(bridge, output_dir / "settled_runtime_snapshot.pkl")
        final_native = bridge.request({"op": "native_task_status"})
        final_observation = bridge.request({"op": "observe", "include_snapshot_state": True})

        body_ids = {
            "pot1": int(pot_bindings["moka_pot_1"]["body_id"]),
            "pot2": int(pot_bindings["moka_pot_2"]["body_id"]),
        }
        stability = summarize_settling(settle_rows, body_ids)
        for obj in stability["objects"].values():
            obj["passes_2mm_position_span"] = obj["position_span_m"] <= 0.002
            obj["passes_0.08m_s_linear_speed"] = (
                obj["maximum_linear_speed_m_s"] is not None
                and obj["maximum_linear_speed_m_s"] <= 0.08
            )
            obj["passes_0.8rad_s_angular_speed"] = (
                obj["maximum_angular_speed_rad_s"] is not None
                and obj["maximum_angular_speed_rad_s"] <= 0.8
            )
        pair_sets = {phase: set().union(*(contact_pair_names(row) for row in rows)) if rows else set() for phase, rows in phase_contacts.items()}
        contact_changes = {
            "native_success_to_release_added": sorted(pair_sets["release"] - pair_sets["native_success"]),
            "native_success_to_release_removed": sorted(pair_sets["native_success"] - pair_sets["release"]),
            "release_to_retreat_added": sorted(pair_sets["retreat"] - pair_sets["release"]),
            "release_to_retreat_removed": sorted(pair_sets["release"] - pair_sets["retreat"]),
            "retreat_to_settle_added": sorted(pair_sets["settle"] - pair_sets["retreat"]),
            "retreat_to_settle_removed": sorted(pair_sets["retreat"] - pair_sets["settle"]),
            "phase_contact_pair_counts": {phase: len(values) for phase, values in pair_sets.items()},
        }
        all_settle_goals_true = bool(native_history["settle"]) and all(
            all(value is True for value in sample.values()) for sample in native_history["settle"]
        )
        monitoring_rows = read_step_rows(records_path)
        report = {
            "schema_version": "gs_opd_run53_release_diagnostic.v1",
            "task": "obstacle_avoidance/L1T2 seed=0 init_state=0",
            "branch_type": "independent_physical_diagnostic_branch",
            "source_run53_report_sha256": sha256_file(RUN53 / "run_report.json"),
            "source_snapshot_sha256": source_descriptor["payload_sha256"],
            "source_snapshot_model_sha256": EXPECTED_MODEL_SHA,
            "source_snapshot_integration_state_sha256": source_descriptor["integration_state_sha256"],
            "run53_source_actions": len(source_actions),
            "replayed_source_action_states_exact": sum(row["integration_state_exact"] for row in source_matches),
            "replayed_source_action_states_total": len(source_matches),
            "replay_exact": len(source_matches) == len(source_actions) and all(row["integration_state_exact"] and row["timestep_matches"] for row in source_matches),
            "replay_max_integration_state_abs_delta": max((row["integration_state_max_abs_delta"] or 0.0 for row in source_matches), default=None),
            "native_task_success": {
                "original_run53_report": source_report.get("task_success") is True,
                "reproduced_before_release": success_status.get("task_success") is True,
                "native_predicates_before_release": values_at_success,
                "diagnostic_final_task_success": final_native.get("task_success"),
                "diagnostic_final_predicates": _native_values(final_native),
            },
            "post_release_stability": {
                "released_from_pot2": released,
                "release_grasp_samples": len(release_grasp_states),
                "release_last_grasp_state": release_grasp_states[-1] if release_grasp_states else pre_release_grasp,
                "settling_actions": len(settle_rows),
                "all_native_goals_true_at_each_settle_sample": all_settle_goals_true,
                "settling_motion": stability,
                "contact_pair_changes": contact_changes,
            },
            "trace_integrity": {
                "source_chain_audit_path": str(output_dir / "source_chain_audit.json"),
                "selected_source_action_physical_state_chain_contiguous": chain["segment_timestep_coverage_contiguous"],
            "snapshot_links_verified_from_reports_and_archived_payloads": chain["snapshot_link_payload_hashes_match_resuming_reports"] and chain["all_snapshot_payloads_present"],
                "run48_root_report_missing": not (RUN48 / "run_report.json").exists(),
                "run48_durable_records_and_recovery_metadata_present": (RUN48 / "interruption_recovery.json").is_file() and (RUN48 / "run48_archive_index.json").is_file(),
                "per_step_robot_object_physical_state_present": chain["observations"]["per_step_robot_object_physical_state_saved"],
                "full_raw_policy_observation_complete": chain["observations"]["full_raw_policy_observation_saved"],
                "dual_camera_source_videos_available": chain["observations"]["dual_camera_video_available_in_all_selected_segments"],
                "post_success_diagnostic_actions": post_success_physics_steps,
            },
            "safety_certification": False,
            "contact_monitoring": {
                "control_actions": len(monitoring_rows),
                "end_of_control_step_contact_samples": len(monitoring_rows),
                "physics_substep_capture": "not_collected_in_retry_branch_after_step_monitored_bridge_disconnect",
                "monitoring_complete_for_safety_certification": False,
            },
            "release_retreat": {
                "native_success_timestep": release_start_t,
                "release_actions": len(release_grasp_states),
                "retreat_actions": retreat_action_count,
                "post_success_physics_steps": post_success_physics_steps,
                "done_flag_remained_native": bool(final_native.get("done")),
                "task_success_definition_changed": False,
                "object_state_directly_modified": False,
                "diagnostic_terminal_snapshot": str(output_dir / "settled_runtime_snapshot.pkl"),
            },
            "artifacts": {
                "records": str(records_path),
                "dual_camera_video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                "frame_index": str(output_dir / "video_frame_index.json"),
                "pre_success_source_snapshot_copy": str(output_dir / "pre_success_source_runtime_snapshot.pkl"),
                "native_success_snapshot": str(output_dir / "native_success_before_release_runtime_snapshot.pkl"),
                "post_release_snapshot": str(output_dir / "post_release_runtime_snapshot.pkl"),
                "post_retreat_snapshot": str(output_dir / "post_retreat_runtime_snapshot.pkl"),
                "settled_snapshot": str(output_dir / "settled_runtime_snapshot.pkl"),
            },
            "original_run53_report_modified": False,
        }
        _write_json(output_dir / "run53_release_diagnostic_report.json", report)
        _write_json(output_dir / "video_frame_index.json", frame_index)
        return report
    except Exception as exc:
        _write_json(output_dir / "diagnostic_failure.json", {
            "status": "failed_before_diagnostic_completion",
            "error": f"{type(exc).__name__}: {exc}",
            "physical_actions_recorded": action_index,
            "original_run53_report_modified": False,
        })
        raise
    finally:
        if video is not None:
            video.release()
        bridge.close()


def read_step_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROLLOUT_ROOT / f"run53_release_stability_diagnostic_{timestamp}"
    report = run_diagnostic(output)
    print(json.dumps({"archive": str(output), "native_task_success": report["native_task_success"], "post_release_stability": report["post_release_stability"], "replay_exact": report["replay_exact"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
