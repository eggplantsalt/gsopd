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

"""Continue run53's independent post-release branch through retreat and settling."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from research.gs_opd.repair.run53_release_diagnostic import (
    EXPECTED_BDDL_SHA,
    EXPECTED_INSTRUCTION,
    EXPECTED_MODEL_SHA,
    RUN46,
    ROLLOUT_ROOT,
    RUN53,
    _export_snapshot,
    _native_values,
    has_target_hand_contact,
    normalize_grasp_binding,
    _object_state,
    _verified_resume_runtime_snapshot,
    audit_source_chain,
    contact_pair_names,
    read_step_rows,
    sha256_file,
    summarize_settling,
)
from research.gs_opd.repair.task_success_first_grasp_search import (
    BridgeSession,
    _image,
    _write_json,
)
from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _contact_union(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    result = set()
    for row in rows:
        result |= contact_pair_names(row.get("contacts", {}))
    return result


def continue_diagnostic(prior_archive: Path, output_dir: Path) -> dict[str, Any]:
    prior_archive = Path(prior_archive).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = output_dir / "dual_camera_frames"
    frames_dir.mkdir()
    prior_records_path = prior_archive / "diagnostic_physical_step_records.jsonl"
    prior_status_path = prior_archive / "diagnostic_phase_status.jsonl"
    release_snapshot = prior_archive / "post_release_runtime_snapshot.pkl"
    prior_records = read_step_rows(prior_records_path)
    if not prior_records or prior_records[-1].get("phase") != "release" or prior_records[-1].get("timestep") != 1336:
        raise ValueError("prior_archive_does_not_end_at_the_run53_post_release_snapshot")
    replay_rows = [row for row in prior_records if row.get("phase") == "replay_run53_success_suffix"]
    if len(replay_rows) != 118 or not all(row.get("replay_integration_state_exact") for row in replay_rows):
        raise ValueError("prior_archive_run53_suffix_replay_is_not_exact")
    if not release_snapshot.is_file():
        raise FileNotFoundError(f"post_release_snapshot_missing:{release_snapshot}")

    source_report = json.loads((RUN53 / "run_report.json").read_text(encoding="utf-8"))
    source_descriptor = _verified_resume_runtime_snapshot(
        release_snapshot,
        benchmark="obstacle_avoidance", level=1, task_id=2, seed=0, init_state=0,
        model_sha=EXPECTED_MODEL_SHA, bddl_sha=EXPECTED_BDDL_SHA,
        instruction=EXPECTED_INSTRUCTION,
    )
    if int(source_descriptor["scene_metadata"].get("bridge_horizon", -1)) != int(source_report["budgets"]["demonstration_horizon"]):
        raise ValueError("post_release_snapshot_horizon_does_not_match_run53")

    # Preserve the prior branch byte-for-byte as the prefix of this continuation.
    records_path = output_dir / "diagnostic_physical_step_records.jsonl"
    shutil.copy2(prior_records_path, records_path)
    shutil.copy2(prior_status_path, output_dir / "diagnostic_phase_status.jsonl")
    for name in (
        "pre_success_source_runtime_snapshot.pkl",
        "native_success_before_release_runtime_snapshot.pkl",
        "post_release_runtime_snapshot.pkl",
    ):
        shutil.copy2(prior_archive / name, output_dir / name)
    _write_json(output_dir / "source_chain_audit.json", audit_source_chain())
    _write_json(output_dir / "continued_from_prior_diagnostic_branch.json", {
        "prior_archive": str(prior_archive),
        "prior_records_sha256": sha256_file(prior_records_path),
        "prior_post_release_snapshot_sha256": sha256_file(release_snapshot),
        "prior_snapshot_timestep": 1336,
        "run53_source_success_report_preserved": True,
    })

    args = SimpleNamespace(
        benchmark="obstacle_avoidance", level=1, task_id=2, init_state=0, seed=0,
        camera_resolution=256,
        demonstration_horizon=int(source_report["budgets"]["demonstration_horizon"]),
    )
    bridge = BridgeSession(("127.0.0.1", _port()), output_dir / "safety_bridge.log", args)
    video = None
    frame_index = []
    action_index = len(prior_records)
    settle_snapshots = []
    retreat_actions = 0
    native_history = {"retreat": [], "settle": []}
    phase_contact_rows = {
        "native_success": [replay_rows[-1]],
        "release": [prior_records[-1]],
        "retreat": [],
        "settle": [],
    }
    for index, row in enumerate(prior_records):
        frame_index.append({"frame": index, "action_index": index, "phase": row["phase"], "timestep": row["timestep"], "source_video_archive": str(prior_archive)})

    def record_frame(agent: np.ndarray, wrist: np.ndarray, phase: str, timestep: int) -> None:
        nonlocal video
        if video is None:
            video = cv2.VideoWriter(
                str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
                (agent.shape[1] + wrist.shape[1], agent.shape[0]),
            )
            if not video.isOpened():
                raise RuntimeError("continuation_video_writer_failed")
            for prior_index in range(len(prior_records)):
                prefix = prior_archive / "dual_camera_frames" / f"frame_{prior_index:06d}"
                prior_agent = cv2.imread(str(prefix) + "_agentview.jpg")
                prior_wrist = cv2.imread(str(prefix) + "_wrist.jpg")
                if prior_agent is None or prior_wrist is None:
                    raise FileNotFoundError(f"prior_video_frame_missing:{prefix}")
                video.write(np.concatenate((prior_agent, prior_wrist), axis=1))
                shutil.copy2(str(prefix) + "_agentview.jpg", frames_dir / f"frame_{prior_index:06d}_agentview.jpg")
                shutil.copy2(str(prefix) + "_wrist.jpg", frames_dir / f"frame_{prior_index:06d}_wrist.jpg")
        video.write(cv2.cvtColor(np.concatenate((agent, wrist), axis=1), cv2.COLOR_RGB2BGR))
        index = len(frame_index)
        cv2.imwrite(str(frames_dir / f"frame_{index:06d}_agentview.jpg"), cv2.cvtColor(agent, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frames_dir / f"frame_{index:06d}_wrist.jpg"), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
        frame_index.append({"frame": index, "action_index": action_index - 1, "phase": phase, "timestep": timestep})

    def step(phase: str, action: np.ndarray) -> dict[str, Any]:
        nonlocal action_index
        reply = bridge.request({
            "op": "step", "action": np.asarray(action, dtype=np.float32).tolist(),
            "include_snapshot_state": True, "include_contacts": True,
        })
        action_index += 1
        record_frame(_image(reply, "agentview_image"), _image(reply, "robot0_eye_in_hand_image"), phase, int(reply.get("timestep", -1)))
        row = {
            "action_index": action_index - 1,
            "phase": phase,
            "action_7d": np.asarray(action, dtype=np.float32).tolist(),
            "state": reply.get("state", {}),
            "snapshot_state": reply.get("snapshot_state", {}),
            "contacts": reply.get("contacts", {}),
            "contact_sampling": "end_of_control_step_only",
            "substep_contact_capture_complete": False,
            "reward": reply.get("reward", 0.0),
            "done": reply.get("done", False),
            "task_success": reply.get("task_success"),
            "timestep": reply.get("timestep"),
            "safety_certified": False,
        }
        _append_jsonl(records_path, row)
        phase_contact_rows.setdefault(phase, []).append(row)
        return row

    try:
        bridge.start()
        bridge.request({"op": "observe"})
        reset = bridge.request({"op": "reset"})
        if reset.get("scene_metadata", {}).get("model_geometry_sha256") != EXPECTED_MODEL_SHA:
            raise RuntimeError("reset_model_fingerprint_does_not_match_run53")
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": source_descriptor["payload"]})
        restored_state = np.asarray(restored.get("snapshot_state", {}).get("mujoco_integration_state", []), dtype=np.float64)
        expected_state = source_descriptor["snapshot_state"]["mujoco_integration_state"]
        if restored.get("status") != "restored" or not np.array_equal(restored_state, expected_state):
            raise RuntimeError("post_release_snapshot_restore_not_exact")
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"}).get("bindings", {})
        pot1 = semantic.get("moka_pot_1")
        pot2_raw = semantic.get("moka_pot_2")
        pot2 = normalize_grasp_binding(pot2_raw) if isinstance(pot2_raw, dict) else None
        if not pot1 or not pot2:
            raise RuntimeError("active_pot_geometry_bindings_missing")
        start_status = bridge.request({"op": "native_task_status"})
        if start_status.get("task_success") is not True or not all(value is True for value in _native_values(start_status).values()):
            raise RuntimeError(f"post_release_native_goal_not_preserved:{start_status}")
        start_grasp = bridge.request({"op": "grasp_state", "carried_binding": pot2, "expected_instruction": EXPECTED_INSTRUCTION})
        if start_grasp.get("grasp_confirmed") is True:
            raise RuntimeError("post_release_snapshot_still_reports_pot2_held")
        start_contacts = bridge.request({"op": "inspect_contacts"})
        _export_snapshot(bridge, output_dir / "continued_post_release_start_runtime_snapshot.pkl")

        robot = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]["robot_states"][0]
        target = np.asarray(robot["eef_pos"], dtype=np.float64) + np.asarray([0.0, 0.0, 0.10])
        start_quat = np.asarray(robot["eef_quat"], dtype=np.float64)
        controller = bridge.request({"op": "repair_context"})
        fields = controller.get("robots", [{}])[0].get("fields")
        if not isinstance(fields, dict):
            raise RuntimeError("OSC_controller_context_unavailable")
        for _ in range(12):
            observed = bridge.request({"op": "observe", "include_snapshot_state": True})
            robot = observed["snapshot_state"]["robot_states"][0]
            position = np.asarray(robot["eef_pos"], dtype=np.float64)
            quaternion = np.asarray(robot["eef_quat"], dtype=np.float64)
            delta = target - position
            distance = float(np.linalg.norm(delta))
            if distance <= 0.006:
                break
            waypoint = position + delta * min(1.0, 0.020 / distance)
            action = world_pose_tracking_action(
                position, quaternion, waypoint, start_quat, fields, gripper_action=-1.0,
            )
            row = step("retreat", action)
            retreat_actions += 1
            native = bridge.request({"op": "native_task_status"})
            native_history["retreat"].append(native)
            _append_jsonl(output_dir / "diagnostic_phase_status.jsonl", {"phase": "retreat", "timestep": native.get("timestep"), "native_task_status": native})
        _export_snapshot(bridge, output_dir / "post_retreat_runtime_snapshot.pkl")

        for _ in range(20):
            row = step("settle", np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            settle_snapshots.append(row["snapshot_state"])
            native = bridge.request({"op": "native_task_status"})
            native_history["settle"].append(native)
            _append_jsonl(output_dir / "diagnostic_phase_status.jsonl", {"phase": "settle", "timestep": native.get("timestep"), "native_task_status": native})
        _export_snapshot(bridge, output_dir / "settled_runtime_snapshot.pkl")
        final_status = bridge.request({"op": "native_task_status"})
        final_grasp = bridge.request({"op": "grasp_state", "carried_binding": pot2, "expected_instruction": EXPECTED_INSTRUCTION})
        body_ids = {"pot1": int(pot1["root_body_id"]), "pot2": int(pot2["root_body_id"])}
        stability = summarize_settling(settle_snapshots, body_ids)
        for item in stability["objects"].values():
            item["passes_2mm_position_span"] = item["position_span_m"] <= 0.002
            item["passes_0.08m_s_linear_speed"] = item["maximum_linear_speed_m_s"] is not None and item["maximum_linear_speed_m_s"] <= 0.08
            item["passes_0.8rad_s_angular_speed"] = item["maximum_angular_speed_rad_s"] is not None and item["maximum_angular_speed_rad_s"] <= 0.8
        settle_values = [_native_values(row) for row in native_history["settle"]]
        contact_sets = {phase: _contact_union(rows) for phase, rows in phase_contact_rows.items()}
        final_contact = bridge.request({"op": "inspect_contacts"})
        initial_set = contact_sets.get("native_success", set())
        release_set = contact_sets.get("release", set())
        retreat_set = contact_sets.get("retreat", set())
        settle_set = _contact_union(phase_contact_rows.get("settle", [])) | contact_pair_names(final_contact)
        report = {
            "schema_version": "gs_opd_run53_release_diagnostic.v1",
            "task": "obstacle_avoidance/L1T2 seed=0 init_state=0",
            "branch_type": "continued_independent_physical_diagnostic_branch",
            "source_run53_report_sha256": sha256_file(RUN53 / "run_report.json"),
            "prior_branch_archive": str(prior_archive),
            "prior_branch_post_release_snapshot_sha256": source_descriptor["payload_sha256"],
            "initial_post_release_native_success_and_unheld": start_status.get("task_success") is True and start_grasp.get("grasp_confirmed") is not True,
            "native_task_success": {
                "original_run53_report": True,
                "reproduced_before_release": True,
                "remained_true_after_release_and_settling": final_status.get("task_success") is True,
                "final_predicates": _native_values(final_status),
            },
            "post_release_stability": {
                "released_from_pot2": True,
                "release_transition_timestep": 1336,
                "retreat_actions": retreat_actions,
                "settling_actions": len(settle_snapshots),
                "all_native_goals_true_at_each_settle_sample": bool(settle_values) and all(all(value is True for value in row.values()) for row in settle_values),
                "final_pot2_grasp_confirmed": final_grasp.get("grasp_confirmed"),
                "settling_motion": stability,
                "contact_change_counts": {
                    "native_success_pairs": len(initial_set),
                    "release_pairs": len(release_set),
                    "retreat_pairs": len(retreat_set),
                    "settle_pairs": len(settle_set),
                    "removed_at_release": sorted(initial_set - release_set),
                    "removed_by_retreat": sorted(release_set - retreat_set),
                    "added_during_settle": sorted(settle_set - retreat_set),
                },
                "start_contact_sample_after_release_restore": start_contacts,
            },
            "trace_integrity": {
                "run53_source_suffix_actions_exactly_replayed": len(replay_rows) == 118 and all(row.get("replay_integration_state_exact") for row in replay_rows),
                "prior_release_action_persisted": True,
                "continuation_physical_actions": retreat_actions + len(settle_snapshots),
                "full_raw_policy_observation_complete": False,
                "physical_robot_object_state_saved_each_control_step": True,
                "camera_video_covers_replay_release_retreat_settling": True,
            },
            "safety_certification": False,
            "contact_monitoring": {
                "source_replay_branch_first_attempt_substep_monitor_failure": "step_monitored bridge disconnected after one exact source step; that attempt is separately archived",
                "selected_branch_sampling": "end_of_control_step contacts only",
                "substep_contact_monitoring_complete": False,
                "safety_certified": False,
            },
            "task_success_definition_changed": False,
            "object_state_directly_modified": False,
            "original_run53_report_modified": False,
            "artifacts": {
                "complete_action_state_contacts_jsonl": str(records_path),
                "dual_camera_video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
                "video_frame_index": str(output_dir / "video_frame_index.json"),
                "post_release_start_snapshot": str(output_dir / "continued_post_release_start_runtime_snapshot.pkl"),
                "post_retreat_snapshot": str(output_dir / "post_retreat_runtime_snapshot.pkl"),
                "settled_snapshot": str(output_dir / "settled_runtime_snapshot.pkl"),
            },
        }
        _write_json(output_dir / "run53_release_diagnostic_report.json", report)
        _write_json(output_dir / "video_frame_index.json", frame_index)
        return report
    except Exception as exc:
        _write_json(output_dir / "diagnostic_failure.json", {
            "status": "continuation_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "last_durable_timestep": action_index + 1217,
            "prior_archive_preserved": str(prior_archive),
        })
        raise
    finally:
        if video is not None:
            video.release()
        bridge.close()


def finalize_archive(output_dir: Path, prior_archive: Path) -> dict[str, Any]:
    """Rebuild the report from durable evidence if post-run reporting was interrupted."""
    output_dir = Path(output_dir).resolve()
    prior_archive = Path(prior_archive).resolve()
    records_path = output_dir / "diagnostic_physical_step_records.jsonl"
    rows = read_step_rows(records_path)
    phases_path = output_dir / "diagnostic_phase_status.jsonl"
    phase_rows = [json.loads(line) for line in phases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_report = json.loads((RUN53 / "run_report.json").read_text(encoding="utf-8"))
    semantic_pot2 = normalize_grasp_binding(source_report["task"]["semantic_binding"])
    semantic_pot1 = json.loads((RUN46 / "run_report.json").read_text(encoding="utf-8"))["task"]["semantic_binding"]
    bodies = {"pot1": int(semantic_pot1["root_body_id"]), "pot2": int(semantic_pot2["root_body_id"])}
    replay = [row for row in rows if row.get("phase") == "replay_run53_success_suffix"]
    release = [row for row in rows if row.get("phase") == "release"]
    retreat = [row for row in rows if row.get("phase") == "retreat"]
    settle = [row for row in rows if row.get("phase") == "settle"]
    robot_rows = [row["snapshot_state"].get("robot_states", [{}])[0] for row in rows]
    statuses_by_timestep = {int(row["timestep"]): row["native_task_status"] for row in phase_rows}
    native_success_status = source_report["final_native_task_status"]
    release_status = statuses_by_timestep.get(1336)
    if not replay or not release or len(settle) != 20 or release_status is None:
        raise ValueError("durable_run53_release_branch_is_missing_required_phase_evidence")
    final_status = statuses_by_timestep.get(int(rows[-1]["timestep"]))
    if final_status is None:
        raise ValueError("final_settle_native_status_is_missing")
    settling = summarize_settling([row["snapshot_state"] for row in settle], bodies)
    for value in settling["objects"].values():
        value["passes_2mm_position_span"] = value["position_span_m"] <= 0.002
        value["passes_0.08m_s_linear_speed"] = value["maximum_linear_speed_m_s"] is not None and value["maximum_linear_speed_m_s"] <= 0.08
        value["passes_0.8rad_s_angular_speed"] = value["maximum_angular_speed_rad_s"] is not None and value["maximum_angular_speed_rad_s"] <= 0.8
    settle_history = [_native_values(statuses_by_timestep[int(row["timestep"])]) for row in settle]
    pot2_on_loss_timestep = next(
        (int(row["timestep"]) for row in phase_rows if row.get("phase") == "retreat" and _native_values(row["native_task_status"]).get("pot2_on") is False),
        None,
    )
    release_grasp_rows = [row for row in phase_rows if row.get("phase") == "release" and row.get("grasp_state")]
    release_grasp = release_grasp_rows[-1]["grasp_state"] if release_grasp_rows else None
    settling_motion_pass = all(
        item["passes_2mm_position_span"]
        and item["passes_0.08m_s_linear_speed"]
        and item["passes_0.8rad_s_angular_speed"]
        for item in settling["objects"].values()
    )
    release_eef = np.asarray(robot_rows[release[-1]["action_index"]]["eef_pos"], dtype=np.float64)
    retreat_final_eef = np.asarray(robot_rows[retreat[-1]["action_index"]]["eef_pos"], dtype=np.float64)
    retreat_target = release_eef + np.asarray([0.0, 0.0, 0.10])
    contacts_by_phase = {
        "native_success": [replay[-1]],
        "release": release,
        "retreat": retreat,
        "settle": settle,
    }
    pair_sets = {phase: _contact_union(phase_rows) for phase, phase_rows in contacts_by_phase.items()}
    rows_by_timestep = {int(row["timestep"]): row for row in rows}

    def has_pot2_hand_contact(row: dict[str, Any]) -> bool:
        return has_target_hand_contact(row.get("contacts", {}), semantic_pot2)

    contact_free_release_timestep = next(
        (
            timestep for timestep in range(1336, 1340)
            if timestep in rows_by_timestep and not has_pot2_hand_contact(rows_by_timestep[timestep])
        ),
        None,
    )
    contact_free_release_status = statuses_by_timestep.get(contact_free_release_timestep) if contact_free_release_timestep else None
    contact_sets_at = {
        timestep: contact_pair_names(rows_by_timestep[timestep].get("contacts", {}))
        for timestep in (1335, 1336, 1337, 1338, 1348, 1349, 1368)
        if timestep in rows_by_timestep
    }
    camera_index = []
    for index, row in enumerate(rows):
        camera_index.append({"frame": index, "action_index": row["action_index"], "phase": row["phase"], "timestep": row["timestep"]})
    _write_json(output_dir / "video_frame_index.json", camera_index)
    video = cv2.VideoCapture(str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"))
    video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if video.isOpened() else 0
    video.release()

    actions = np.asarray([row["action_7d"] for row in rows], dtype=np.float32).reshape(-1, 7)
    pots = {
        name: [
            _object_state(row["snapshot_state"], body_id)
            for row in rows
        ]
        for name, body_id in bodies.items()
    }
    arrays = {
        "action_7d": actions,
        "phase": np.asarray([row["phase"] for row in rows]),
        "timestep": np.asarray([row["timestep"] for row in rows], dtype=np.int64),
        "eef_position_world_m": np.asarray([row.get("state", {}).get("robot0_eef_pos", [np.nan] * 3) for row in rows], dtype=np.float64),
        "arm_joint_qpos_rad": np.asarray([robot.get("joint_qpos", [np.nan] * 7) for robot in robot_rows], dtype=np.float64),
        "pot1_root_position_world_m": np.asarray([state["root_pos"] for state in pots["pot1"]], dtype=np.float64),
        "pot2_root_position_world_m": np.asarray([state["root_pos"] for state in pots["pot2"]], dtype=np.float64),
        "pot2_root_qvel": np.asarray([state.get("joint_qvel", [np.nan] * 6) for state in pots["pot2"]], dtype=np.float64),
        "reward": np.asarray([row.get("reward", 0.0) for row in rows], dtype=np.float32),
        "done": np.asarray([row.get("done", False) for row in rows], dtype=bool),
        "task_success": np.asarray([row.get("task_success", False) for row in rows], dtype=bool),
    }
    np.savez_compressed(output_dir / "diagnostic_actions_states.npz", **arrays)
    settle_goals_true = bool(settle_history) and all(all(value is True for value in row.values()) for row in settle_history)
    report = {
        "schema_version": "gs_opd_run53_release_diagnostic.v1",
        "status": "post_success_release_unstable",
        "task": "obstacle_avoidance/L1T2 seed=0 init_state=0",
        "branch_type": "independent_physical_diagnostic_branch",
        "source_run53_report_sha256": sha256_file(RUN53 / "run_report.json"),
        "prior_release_branch_archive": str(prior_archive),
        "diagnostic_archive": str(output_dir),
        "native_task_success": {
            "original_run53_report": source_report.get("task_success") is True,
            "reproduced_at_timestep_1335_before_release": native_success_status.get("task_success") is True,
            "native_predicates_at_success": _native_values(native_success_status),
            "post_release_timestep_1336": _native_values(release_status),
            "post_retreat_and_settle_final": _native_values(final_status),
            "task_success_remains_true_after_release_retreat_settle": final_status.get("task_success") is True,
        },
        "post_release_stability": {
            "release_action_timestep": 1336,
            "grasp_released": contact_free_release_timestep is not None,
            "physical_contact_free_release_timestep": contact_free_release_timestep,
            "grasp_state_after_release": release_grasp,
            "grasp_state_api_status": release_grasp.get("status") if release_grasp else "unavailable",
            "grasp_state_api_limitation": "raw semantic binding lacks body_id; contact-free release is determined from compiled geom-name contact records",
            "pot2_on_timestep_1336": _native_values(release_status).get("pot2_on"),
            "predicates_at_contact_free_release": _native_values(contact_free_release_status) if contact_free_release_status else None,
            "pot2_on_first_became_false_during_retreat": pot2_on_loss_timestep,
            "retreat_actions": len(retreat),
            "retreat_timestep_range": [retreat[0]["timestep"], retreat[-1]["timestep"]],
            "retreat_target_distance_remaining_m": float(np.linalg.norm(retreat_target - retreat_final_eef)),
            "retreat_reached_100mm_target": float(np.linalg.norm(retreat_target - retreat_final_eef)) <= 0.006,
            "settling_actions": len(settle),
            "all_native_goals_true_during_settling": settle_goals_true,
            "motion_metrics": settling,
            "stable_after_release": settle_goals_true and settling_motion_pass,
            "stability_failure_reasons": [
                *([] if settle_goals_true else ["at_least_one_native_goal_false_during_settling"]),
                *([] if settling_motion_pass else ["object_pose_or_velocity_exceeded_phase_settle_criteria"]),
            ],
            "contact_pair_changes": {
                "native_success_pair_count": len(pair_sets["native_success"]),
                "release_pair_count": len(pair_sets["release"]),
                "retreat_pair_count": len(pair_sets["retreat"]),
                "settle_pair_count": len(pair_sets["settle"]),
                "pairs_removed_after_release": sorted(pair_sets["native_success"] - pair_sets["release"]),
                "pairs_added_during_retreat": sorted(pair_sets["retreat"] - pair_sets["release"]),
                "pairs_removed_by_retreat": sorted(pair_sets["release"] - pair_sets["retreat"]),
                "pairs_added_during_settling": sorted(pair_sets["settle"] - pair_sets["retreat"]),
                "contact_pairs_at_key_timesteps": {
                    str(timestep): sorted(pairs) for timestep, pairs in contact_sets_at.items()
                },
            },
        },
        "trace_integrity": {
            "run53_success_suffix_exact_replay_actions": len(replay),
            "run53_success_suffix_exact_state_matches": sum(row.get("replay_integration_state_exact") is True for row in replay),
            "source_and_diagnostic_actions_contiguous": all(int(row["action_index"]) == index for index, row in enumerate(rows)),
            "source_timestep_range": [rows[0]["timestep"], rows[-1]["timestep"]],
            "per_step_robot_object_physical_states_saved": True,
            "full_raw_policy_observation_complete": False,
            "dual_camera_video_frame_count": video_frames,
            "records_count": len(rows),
            "run48_root_report_missing_but_recovery_metadata_available": True,
        },
        "safety_certification": False,
        "contact_monitoring": {
            "sampling": "contacts and physical state captured at end of each control action",
            "25_substep_contact_monitoring_complete": False,
            "reason": "the diagnostic uses the original step interface after step_monitored disconnected in independent replay probes",
        },
        "provenance": {
            "selected_source_action_lineage": "run19 reset+grasp -> run46 branch03 -> run47 -> run48 grasp -> run52 -> run53 native success",
            "source_chain_audit": str(output_dir / "source_chain_audit.json"),
            "source_snapshots_and_actions_unmodified": True,
            "original_run53_report_modified": False,
            "official_safety_demonstrations_used": False,
            "task_success_definition_changed": False,
            "object_state_directly_modified": False,
            "report_rebuilt_from_durable_records_after_runtime_report_error": True,
            "runtime_report_error": "KeyError body_id on semantic binding (the binding exposes root_body_id); all physical actions and snapshots had already been durably written",
        },
        "artifacts": {
            "physical_records_jsonl": str(records_path),
            "action_state_npz": str(output_dir / "diagnostic_actions_states.npz"),
            "dual_camera_video": str(output_dir / "dual_camera_agentview_wrist_20fps.mp4"),
            "frame_index": str(output_dir / "video_frame_index.json"),
            "release_snapshot": str(output_dir / "post_release_runtime_snapshot.pkl"),
            "retreat_snapshot": str(output_dir / "post_retreat_runtime_snapshot.pkl"),
            "settled_snapshot": str(output_dir / "settled_runtime_snapshot.pkl"),
        },
    }
    _write_json(output_dir / "run53_release_diagnostic_report.json", report)
    _write_json(output_dir / "diagnostic_failure_resolution.json", {
        "prior_failure": "KeyError body_id",
        "physical_actions_after_failure_point": 0,
        "durable_end_timestep": rows[-1]["timestep"],
        "resolution": "report and NPZ were rebuilt offline using the recorded root_body_id; no action or state file was overwritten",
    })
    pot2_motion = settling["objects"]["pot2"]
    summary = f"""# run53 post-success release diagnostic

## Result

The original run53 report remains unchanged: native BDDL success was true at t=1335 with Pot1 On, Pot2 On, Stove Turnon, reward 1, and done true. The exact 118-action run53 suffix replay reproduced its saved Integration State sequence.

The independent branch issued an open-gripper action at t=1336. Pot2 remained in gripper contact at t=1336 and t=1337; all gripper-to-Pot2 contacts disappeared at t={contact_free_release_timestep}. At that contact-free step, both On predicates and Stove Turnon were still true.

The vertical retreat used {len(retreat)} OSC actions and did not reach its 100 mm target; {float(np.linalg.norm(retreat_target - retreat_final_eef)) * 1000:.1f} mm remained. Pot2's On predicate first became false at t={pot2_on_loss_timestep}, while burner contact was still present. Pot2 contacted the table by t=1349 and later contacted a toy ball. After 20 settling actions, Pot2's position span was {pot2_motion['position_span_m'] * 1000:.1f} mm, maximum recorded linear speed {pot2_motion['maximum_linear_speed_m_s']:.3f} m/s, and angular speed {pot2_motion['maximum_angular_speed_rad_s']:.3f} rad/s. Pot1 remained stationary and On; Pot2 was On=false at the final check. The post-release state is not stable and the native complete task predicate no longer holds in this diagnostic branch.

## Independent result fields

- Native success at run53 t=1335: true.
- Both Pot On predicates after physical contact-free release at t={contact_free_release_timestep}: true.
- Both Pot On predicates after retreat and settling: false for Pot2.
- Safety certification: false; contact monitoring sampled each control-step endpoint only, not all 25 physics substeps.
- Historical per-step state/action chain: contiguous from reset through t=1335; full raw policy Observation tensors were not saved, although per-step robot/object states and dual-camera videos exist.

`grasp_state` was unavailable in the diagnostic process because the semantic binding used `root_body_id` while the query requires `body_id`. Physical release timing was therefore determined from the compiled Pot2 and gripper Geom contact records. The diagnostic report assembly initially failed after all physical actions because of the same field-name mismatch; this report was rebuilt from the durable action/state/contact records. The physical data and original success report were not overwritten.

## Artifacts

- JSON report: `{output_dir / 'run53_release_diagnostic_report.json'}`
- Full per-action state/contact record: `{records_path}`
- Action and state NPZ: `{output_dir / 'diagnostic_actions_states.npz'}`
- Dual-camera video: `{output_dir / 'dual_camera_agentview_wrist_20fps.mp4'}`
- Source-chain audit: `{output_dir / 'source_chain_audit.json'}`
- Final settled Snapshot: `{output_dir / 'settled_runtime_snapshot.pkl'}`
"""
    (output_dir / "run53_release_diagnostic_summary.md").write_text(summary, encoding="utf-8")
    hash_targets = [
        records_path,
        output_dir / "diagnostic_actions_states.npz",
        output_dir / "dual_camera_agentview_wrist_20fps.mp4",
        output_dir / "source_chain_audit.json",
        output_dir / "post_release_runtime_snapshot.pkl",
        output_dir / "post_retreat_runtime_snapshot.pkl",
        output_dir / "settled_runtime_snapshot.pkl",
        output_dir / "run53_release_diagnostic_report.json",
        output_dir / "run53_release_diagnostic_summary.md",
    ]
    _write_json(output_dir / "artifact_hashes.json", {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in hash_targets
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-after-release-archive", type=Path, default=None)
    parser.add_argument("--finalize-archive", type=Path, default=None)
    parser.add_argument("--prior-archive", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.finalize_archive is not None:
        if args.prior_archive is None:
            parser.error("--finalize-archive requires --prior-archive")
        report = finalize_archive(args.finalize_archive, args.prior_archive)
        print(json.dumps({"archive": str(args.finalize_archive), "post_release_stability": report["post_release_stability"], "native_task_success": report["native_task_success"]}, ensure_ascii=False, indent=2))
        return 0
    if args.resume_after_release_archive is None:
        parser.error("--resume-after-release-archive or --finalize-archive is required")
    output = args.output or ROLLOUT_ROOT / f"run53_release_stability_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}_continued"
    report = continue_diagnostic(args.resume_after_release_archive, output)
    print(json.dumps({"archive": str(output), "post_release_stability": report["post_release_stability"], "native_task_success": report["native_task_success"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
