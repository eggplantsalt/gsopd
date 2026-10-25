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
"""Convert an accepted GS-OPD Round 0 chain into one LeRobot v2.1 episode.

This is a data-only adapter for the heterogeneous, already accepted OA traces.
It does not run MuJoCo, alter actions, or treat predicted actions as labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ACCEPTANCE_REPORT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "round0_l0t2_complete_episode_readonly_acceptance_20261025/"
    "round0_complete_episode_acceptance.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/opt/data/private/gs_opd/datasets/"
    "lerobot_l0t2_round0_operational_success_v21_20261025"
)
FPS = 20
ACTION_HORIZON = 10
EXPECTED_ACTIONS = 1460
EXPECTED_SEGMENTS = 25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_converter_helpers() -> Any:
    path = REPOSITORY_ROOT / "research/gs_opd/data/npz_to_lerobot.py"
    spec = importlib.util.spec_from_file_location("gs_opd_existing_npz_converter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_load_existing_converter_helpers:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_compat_helper() -> Any:
    path = REPOSITORY_ROOT / "rlinf/data/storage/lerobot/compat.py"
    spec = importlib.util.spec_from_file_location("gs_opd_lerobot_compat", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_load_lerobot_compat_helper:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash_file_entry(files: dict[str, dict[str, str]], path_value: str | Path) -> str:
    path = Path(path_value).resolve()
    key = str(path)
    if key not in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        files[key] = {"sha256": _sha256(path), "role": "source_artifact"}
    return files[key]["sha256"]


def _quat_xyzw_to_axis_angle(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Match the existing LIBERO/LeRobot robosuite XYZW state transform."""
    quat = np.asarray(quaternion_xyzw, dtype=np.float64).copy()
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"invalid_eef_quaternion:{quat.shape}")
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * (2.0 * math.acos(quat[3]))) / denominator


def _state_vector(
    position: np.ndarray,
    quaternion: np.ndarray,
    gripper_qpos: np.ndarray,
    *,
    quaternion_order: str,
) -> np.ndarray:
    pos = np.asarray(position, dtype=np.float64)
    quat = np.asarray(quaternion, dtype=np.float64)
    grip = np.asarray(gripper_qpos, dtype=np.float64)
    if pos.shape != (3,) or grip.shape != (2,):
        raise ValueError(f"invalid_state_components:{pos.shape}:{quat.shape}:{grip.shape}")
    if quaternion_order == "wxyz":
        quat = quat[[1, 2, 3, 0]]
    elif quaternion_order != "xyzw":
        raise ValueError(f"unsupported_quaternion_order:{quaternion_order}")
    state = np.concatenate((pos, _quat_xyzw_to_axis_angle(quat), grip))
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("invalid_8d_robot_state")
    return state.astype(np.float32)


def _npz_scalar_string(array: np.ndarray) -> str:
    return str(np.asarray(array).item())


def _expected_action_key(segment_index: int) -> str:
    return "actions" if segment_index in (0, 2) else "executed_actions"


def _expected_timestep_key(segment_index: int) -> str:
    if segment_index == 0:
        return "timestep"
    if segment_index == 2:
        return "timesteps"
    return "step_timesteps"


def _load_segment_arrays(segment_index: int, archive: Any) -> dict[str, np.ndarray]:
    """Decompress each needed NPZ member once; skip large raw audit members."""
    keys = {_expected_action_key(segment_index), _expected_timestep_key(segment_index)}
    if segment_index == 0:
        keys.update(
            {
                "pre_agentview",
                "pre_eye_in_hand",
                "pre_robot0_eef_pos",
                "pre_robot0_eef_quat",
                "pre_robot0_gripper_qpos",
            }
        )
    elif segment_index == 1:
        keys.update({"pre_agentview", "pre_eye_in_hand", "task_phase"})
    elif segment_index == 2:
        keys.update(
            {
                "agentview_rgb",
                "wrist_rgb",
                "eef_position_world_m",
                "eef_quaternion_wxyz",
                "gripper_qpos_m",
            }
        )
    else:
        keys.update(
            {
                "decision_agentview_image",
                "decision_robot0_eye_in_hand_image",
                "decision_robot0_eef_pos",
                "decision_robot0_eef_quat",
                "decision_robot0_gripper_qpos",
                "decision_instructions",
                "step_task_phase",
            }
        )
    for optional in (
        "step_rewards",
        "rewards",
        "step_dones",
        "dones",
        "step_task_success",
        "task_success",
    ):
        if optional in archive.files:
            keys.add(optional)
    missing = sorted(key for key in keys if key not in archive.files)
    if missing:
        raise ValueError(f"missing_required_npz_members_segment_{segment_index}:{missing}")
    return {key: np.array(archive[key], copy=True) for key in sorted(keys)}


def _preflight(report: dict[str, Any]) -> None:
    if report.get("round0_safe_success_data_candidate") is not True:
        raise ValueError("acceptance_report_does_not_accept_round0_candidate")
    if report.get("outcome") != "round0_operational_safe_success_data_candidate":
        raise ValueError(f"unexpected_acceptance_outcome:{report.get('outcome')}")
    if report.get("audit_errors"):
        raise ValueError("acceptance_report_contains_audit_errors")
    totals = report.get("trajectory_totals", {})
    if int(totals.get("actions", -1)) != EXPECTED_ACTIONS:
        raise ValueError(f"unexpected_accepted_action_count:{totals.get('actions')}")
    if int(totals.get("expected_physics_substeps", -1)) != EXPECTED_ACTIONS * 25:
        raise ValueError("accepted_substep_total_mismatch")
    safety = report.get("safety_acceptance", {})
    if safety.get("native_safety") != "pass" or safety.get("operational_safety") != "pass":
        raise ValueError("accepted_chain_native_or_operational_safety_failed")
    if safety.get("all_actions_have_25_substep_monitor_coverage") is not True:
        raise ValueError("accepted_chain_missing_full_substep_coverage")
    if safety.get("strict_10mm") != "not_passed_additional_metric_only":
        raise ValueError("unexpected_strict_10mm_status")
    terminal = report.get("native_terminal_acceptance", {})
    if not (
        terminal.get("task_success") is True
        and terminal.get("reward") == 1.0
        and terminal.get("done") is True
        and terminal.get("demonstration_only_1800_step_success") is True
        and terminal.get("standard_1000_step_success") is False
    ):
        raise ValueError("native_terminal_acceptance_not_as_expected")
    segments = report.get("segments", [])
    if len(segments) != EXPECTED_SEGMENTS:
        raise ValueError(f"expected_{EXPECTED_SEGMENTS}_segments_found_{len(segments)}")
    cursor = 0
    action_total = 0
    for index, segment in enumerate(segments):
        start = int(segment["start_timestep"])
        end = int(segment["end_timestep"])
        count = int(segment["physical_actions"])
        if start != cursor or end - start != count:
            raise ValueError(f"noncontiguous_segment_{index}:{start}:{end}:{count}:{cursor}")
        if int(segment["physical_substeps_expected"]) != count * 25:
            raise ValueError(f"segment_substep_count_mismatch_{index}")
        attestation = segment.get("safety_attestation", {})
        if not (
            attestation.get("evidence_exists") is True
            and attestation.get("safety_status") == "pass"
            and attestation.get("native_safety") == "pass"
            and attestation.get("operational_safety") == "pass"
            and int(attestation.get("expected_substeps", -1)) == count * 25
            and attestation.get("strict_10mm") == "additional_metric_not_passed"
        ):
            raise ValueError(f"segment_safety_attestation_invalid_{index}")
        trace_path = Path(segment["trace"])
        if not trace_path.is_file() or _sha256(trace_path) != segment["trace_sha256"]:
            raise ValueError(f"segment_trace_hash_mismatch_{index}")
        cursor = end
        action_total += count
    if cursor != EXPECTED_ACTIONS or action_total != EXPECTED_ACTIONS:
        raise ValueError("accepted_segments_do_not_cover_t0_through_t1460")
    seams = report.get("snapshot_boundaries", [])
    if len(seams) != EXPECTED_SEGMENTS - 1 or not all(seam.get("pass") is True for seam in seams):
        raise ValueError("accepted_snapshot_seam_attestation_failed")


def _source_task_instruction(report: dict[str, Any]) -> tuple[str, dict[str, np.ndarray]]:
    path = Path(report["task_source"]["raw_pi05_trace"])
    with np.load(path, allow_pickle=False) as archive:
        instructions = np.asarray(archive["decision_instructions"]).astype(str)
    if _sha256(path) != report["task_source"]["raw_pi05_trace_sha256"]:
        raise ValueError("raw_pi05_source_trace_hash_mismatch")
    unique = sorted(set(instructions.tolist()))
    if unique != ["put both moka pots on the stove"]:
        raise ValueError(f"unexpected_task_instruction:{unique}")
    return unique[0], {}


def _phase_rows(
    segment_index: int,
    source: Any,
    count: int,
    branch_report: dict[str, Any] | None,
) -> tuple[list[str], str]:
    if segment_index == 0:
        return ["raw_pi05_initial_prefix"] * count, "raw_pi05_action_source"
    if segment_index == 1:
        phases = np.asarray(source["task_phase"]).astype(str).tolist()
        return phases, "accepted_prefix_replay.task_phase"
    if segment_index == 2:
        if branch_report is None or branch_report.get("local_lift_30mm_success") is not True:
            raise ValueError("t249_to_t329_lift_phase_not_attested")
        return ["lift"] * count, "accepted_branch_report.local_lift_30mm_success"
    if "step_task_phase" not in source:
        raise ValueError(f"missing_archived_task_phase_in_segment_{segment_index}")
    return np.asarray(source["step_task_phase"]).astype(str).tolist(), "accepted_trace.step_task_phase"


def _observation_components(
    segment_index: int,
    row: int,
    source: Any,
    *,
    instruction: str,
    original_prefix_state: dict[str, np.ndarray] | None,
    previous_segment_end_state: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return two raw pre-action camera frames, pre-action 8D state, state source."""
    if segment_index == 0:
        main = source["pre_agentview"][row]
        wrist = source["pre_eye_in_hand"][row]
        state = _state_vector(
            source["pre_robot0_eef_pos"][row],
            source["pre_robot0_eef_quat"][row],
            source["pre_robot0_gripper_qpos"][row],
            quaternion_order="xyzw",
        )
        return main, wrist, state, "accepted_trace.pre_robot0_*"

    if segment_index == 1:
        if original_prefix_state is None:
            raise ValueError("missing_episode9_state_companion_for_t10_t249")
        main = source["pre_agentview"][row]
        wrist = source["pre_eye_in_hand"][row]
        state = _state_vector(
            original_prefix_state["decision_robot0_eef_pos"][row],
            original_prefix_state["decision_robot0_eef_quat"][row],
            original_prefix_state["decision_robot0_gripper_qpos"][row],
            quaternion_order="xyzw",
        )
        return main, wrist, state, "episode9_original_OA.decision_robot0_*"

    if segment_index == 2:
        if row == 0:
            if previous_segment_end_state is None:
                raise ValueError("missing_t249_pre_action_state_from_adjacent_accepted_OA")
            state = previous_segment_end_state
            state_source = "episode9_original_OA.post_robot0_*_at_t249"
        else:
            previous_row = row - 1
            state = _state_vector(
                source["eef_position_world_m"][previous_row],
                source["eef_quaternion_wxyz"][previous_row],
                source["gripper_qpos_m"][previous_row],
                quaternion_order="wxyz",
            )
            state_source = "accepted_branch_previous_action_post_state"
        return (
            source["agentview_rgb"][row],
            source["wrist_rgb"][row],
            state,
            state_source,
        )

    return (
        source["decision_agentview_image"][row],
        source["decision_robot0_eye_in_hand_image"][row],
        _state_vector(
            source["decision_robot0_eef_pos"][row],
            source["decision_robot0_eef_quat"][row],
            source["decision_robot0_gripper_qpos"][row],
            quaternion_order="xyzw",
        ),
        "accepted_trace.decision_robot0_*",
    )


def _terminal_values(segment_index: int, source: Any, row: int) -> tuple[float | None, bool | None, bool | None]:
    reward_key = "step_rewards" if "step_rewards" in source else "rewards"
    done_key = "step_dones" if "step_dones" in source else "dones"
    success_key = "step_task_success" if "step_task_success" in source else "task_success"
    reward = float(source[reward_key][row]) if reward_key in source else None
    done = bool(source[done_key][row]) if done_key in source else None
    if success_key in source:
        raw_success = source[success_key][row]
        if np.issubdtype(np.asarray(raw_success).dtype, np.integer):
            success = int(raw_success) == 1
        else:
            success = bool(raw_success)
    else:
        success = None
    if segment_index == 0:
        return None, None, None
    return reward, done, success


def _iter_source_frames(
    report: dict[str, Any],
    instruction: str,
    original_prefix_state: dict[str, np.ndarray],
    *,
    converter_helpers: Any,
    source_file_hashes: dict[str, dict[str, str]],
) -> Iterator[tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]]:
    segments = report["segments"]
    seam_by_timestep = {int(item["timestep"]): item for item in report["snapshot_boundaries"]}
    previous_segment_end_state: np.ndarray | None = None
    for segment_index, segment in enumerate(segments):
        trace_path = Path(segment["trace"])
        evidence_path = Path(segment["safety_attestation"]["evidence_path"])
        trace_sha = _hash_file_entry(source_file_hashes, trace_path)
        evidence_sha = _hash_file_entry(source_file_hashes, evidence_path)
        action_key = _expected_action_key(segment_index)
        timestep_key = _expected_timestep_key(segment_index)
        branch_report = None
        if segment_index == 2:
            branch_report_path = evidence_path
            branch_report = json.loads(branch_report_path.read_text(encoding="utf-8"))
        with np.load(trace_path, allow_pickle=False) as archive:
            source = _load_segment_arrays(segment_index, archive)
            count = int(segment["physical_actions"])
            start = int(segment["start_timestep"])
            end = int(segment["end_timestep"])
            actions = np.asarray(source[action_key])
            timesteps = np.asarray(source[timestep_key], dtype=np.int64)
            if actions.shape != (count, 7) or not np.isfinite(actions).all():
                raise ValueError(f"invalid_executed_action_array_segment_{segment_index}")
            if not np.array_equal(timesteps, np.arange(start + 1, end + 1, dtype=np.int64)):
                raise ValueError(f"source_timestep_alignment_failed_segment_{segment_index}")
            if segment["action_array"].get("shape") != list(actions.shape):
                raise ValueError(f"accepted_report_action_shape_mismatch_segment_{segment_index}")
            if actions.dtype != np.float32:
                raise ValueError(f"executed_action_dtype_not_float32_segment_{segment_index}:{actions.dtype}")
            phases, phase_source = _phase_rows(segment_index, source, count, branch_report)
            if len(phases) != count or any(not phase for phase in phases):
                raise ValueError(f"phase_alignment_failed_segment_{segment_index}")
            for row in range(count):
                global_step = start + row
                main_raw, wrist_raw, state, state_source = _observation_components(
                    segment_index,
                    row,
                    source,
                    instruction=instruction,
                    original_prefix_state=original_prefix_state,
                    previous_segment_end_state=previous_segment_end_state,
                )
                # Use the exact RGB rotation and feature names of the existing
                # GS-OPD/OpenPI LeRobot exporter.
                main = converter_helpers._frame_image({"camera": main_raw}, "camera")
                wrist = converter_helpers._frame_image({"camera": wrist_raw}, "camera")
                action = np.asarray(actions[row], dtype=np.float32).copy()
                frame = {
                    "image": main,
                    "wrist_image": wrist,
                    "state": state,
                    "actions": action,
                    "task": instruction,
                }
                snapshot_seam = None
                if global_step in seam_by_timestep:
                    snapshot_seam = seam_by_timestep[global_step]
                reward, done, native_success = _terminal_values(segment_index, source, row)
                provenance = {
                    "dataset_episode_index": 0,
                    "dataset_frame_index": global_step,
                    "global_pre_action_timestep": global_step,
                    "global_post_action_timestep": global_step + 1,
                    "source_segment_index": segment_index,
                    "source_segment_start_timestep": start,
                    "source_segment_end_timestep": end,
                    "source_action_row": row,
                    "source_action_key": action_key,
                    "source_action_array_dtype": str(actions.dtype),
                    "source_action_array_file": str(trace_path),
                    "source_action_trace_sha256": trace_sha,
                    "source_recorded_post_action_timestep": int(timesteps[row]),
                    "action_is_recorded_executed_action": True,
                    "predicted_action_used_as_label": False,
                    "observation_camera_source": {
                        "agentview": (
                            "pre_agentview" if segment_index in (0, 1)
                            else "agentview_rgb[pre-action index]" if segment_index == 2
                            else "decision_agentview_image"
                        ),
                        "wrist": (
                            "pre_eye_in_hand" if segment_index in (0, 1)
                            else "wrist_rgb[pre-action index]" if segment_index == 2
                            else "decision_robot0_eye_in_hand_image"
                        ),
                        "state_source": state_source,
                        "pre_action": True,
                    },
                    "task_instruction": instruction,
                    "task_phase": phases[row],
                    "task_phase_source": phase_source,
                    "physical_safety": {
                        "native_safety": "pass",
                        "operational_safety": "pass",
                        "monitored_physics_substeps": 25,
                        "safety_evidence_file": str(evidence_path),
                        "safety_evidence_sha256": evidence_sha,
                    },
                    "strict_10mm": {
                        "status": "not_passed_additional_metric_only",
                        "is_operational_gate": False,
                    },
                    "snapshot_seam_before_action": snapshot_seam,
                    "reward": reward,
                    "done": done,
                    "native_task_success": native_success,
                    "episode_scope": "one_demo_only_episode_terminal_at_timestep_1460",
                    "episode_terminal_acceptance": {
                        "native_task_success": True,
                        "terminal_reward": 1.0,
                        "terminal_done": True,
                        "demo_only_1800_step_success": True,
                        "standard_1000_step_success": False,
                        "strict_10mm": "not_passed_additional_metric_only",
                    },
                }
                yield frame, provenance, action, state, main_raw
            if segment_index == 1:
                # Segment 1 was physically accepted from the original Episode 9
                # OA trace. Its states accompany the same action/image rows.
                previous_segment_end_state = _state_vector(
                    original_prefix_state["post_robot0_eef_pos"][-1],
                    original_prefix_state["post_robot0_eef_quat"][-1],
                    original_prefix_state["post_robot0_gripper_qpos"][-1],
                    quaternion_order="xyzw",
                )


def _as_hwc(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.shape == (3, 256, 256):
        result = np.transpose(result, (1, 2, 0))
    if np.issubdtype(result.dtype, np.floating) and result.size and result.max() <= 1.0:
        result = np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def convert_and_verify(
    acceptance_report: Path, output_root: Path, *, resume_existing: bool = False
) -> dict[str, Any]:
    if output_root.exists() and not resume_existing:
        raise FileExistsError(f"refusing_to_overwrite_dataset:{output_root}")
    manifest_path = output_root / "data_manifest.json"
    provenance_path = output_root / "per_action_provenance.jsonl"
    acceptance_output_path = output_root / "round0_lerobot_acceptance.json"
    if resume_existing:
        if not output_root.is_dir() or not (output_root / "meta/info.json").is_file():
            raise FileNotFoundError(f"resume_dataset_metadata_missing:{output_root}")
        if manifest_path.exists() or acceptance_output_path.exists():
            raise FileExistsError("refusing_to_replace_existing_dataset_manifest_or_acceptance")
        if not provenance_path.is_file():
            raise FileNotFoundError(f"resume_provenance_missing:{provenance_path}")
    report_hash = _sha256(acceptance_report)
    report = json.loads(acceptance_report.read_text(encoding="utf-8"))
    _preflight(report)

    instruction, task_source = _source_task_instruction(report)
    prefix_trace_path = Path(
        "/opt/data/private/gs_opd/rollouts/task_success_first_safety_reconstruction_l0t2_20261023_attempt1/"
        "episode9/observation_action_trace.npz"
    )
    with np.load(prefix_trace_path, allow_pickle=False) as prefix_npz:
        original_prefix_state = {
            key: np.array(prefix_npz[key], copy=True)
            for key in (
                "executed_actions",
                "step_task_phase",
                "decision_agentview_image",
                "decision_robot0_eye_in_hand_image",
                "decision_robot0_eef_pos",
                "decision_robot0_eef_quat",
                "decision_robot0_gripper_qpos",
                "post_robot0_eef_pos",
                "post_robot0_eef_quat",
                "post_robot0_gripper_qpos",
            )
        }

    # Verify the accepted t=10 replay preserved the original successful OA
    # prefix exactly; the original OA file supplies robot-state fields omitted
    # from the accepted replay NPZ.
    segment_1 = report["segments"][1]
    with np.load(segment_1["trace"], allow_pickle=False) as accepted_prefix:
        if not np.array_equal(accepted_prefix["executed_actions"], original_prefix_state["executed_actions"]):
            raise ValueError("t10_t249_actions_differ_from_accepted_episode9_source")
        if not np.array_equal(accepted_prefix["pre_agentview"], original_prefix_state["decision_agentview_image"]):
            raise ValueError("t10_t249_agentview_differs_from_original_pre_action_observation")
        if not np.array_equal(accepted_prefix["pre_eye_in_hand"], original_prefix_state["decision_robot0_eye_in_hand_image"]):
            raise ValueError("t10_t249_wrist_camera_differs_from_original_pre_action_observation")
        if not np.array_equal(accepted_prefix["task_phase"], original_prefix_state["step_task_phase"]):
            raise ValueError("t10_t249_phase_values_differ_from_episode9_source")

    source_file_hashes: dict[str, dict[str, str]] = {}
    acceptance_hash = _hash_file_entry(source_file_hashes, acceptance_report)
    task_source_path = Path(report["task_source"]["raw_pi05_trace"])
    task_source_hash = _hash_file_entry(source_file_hashes, task_source_path)
    if task_source_hash != report["task_source"]["raw_pi05_trace_sha256"]:
        raise ValueError("task_source_trace_hash_mismatch")
    episode9_prefix_hash = _hash_file_entry(source_file_hashes, prefix_trace_path)
    for segment in report["segments"]:
        _hash_file_entry(source_file_hashes, segment["trace"])
        _hash_file_entry(source_file_hashes, segment["safety_attestation"]["evidence_path"])
        for boundary_name in ("start_snapshot", "end_snapshot"):
            snapshot = segment.get(boundary_name)
            if snapshot:
                actual = _hash_file_entry(source_file_hashes, snapshot["path"])
                if actual != snapshot["sha256"]:
                    raise ValueError(f"snapshot_hash_mismatch:{boundary_name}:{snapshot['path']}")
    for path in report.get("source_reports", {}).values():
        if Path(path).is_file():
            _hash_file_entry(source_file_hashes, path)

    converter_helpers = _load_converter_helpers()
    compat = _load_compat_helper()
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    features = {
        "image": {
            "dtype": "image",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
        },
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = None
    if not resume_existing:
        dataset = LeRobotDataset.create(
            repo_id=output_root.name,
            root=output_root,
            robot_type="franka_panda",
            fps=FPS,
            features=features,
            use_videos=False,
            image_writer_threads=0,
            image_writer_processes=0,
        )

    expected_actions = np.empty((EXPECTED_ACTIONS, 7), dtype=np.float32)
    expected_states = np.empty((EXPECTED_ACTIONS, 8), dtype=np.float32)
    phases: list[str] = [""] * EXPECTED_ACTIONS
    provenance_rows = 0
    provenance_mode = "r" if resume_existing else "x"
    with provenance_path.open(provenance_mode, encoding="utf-8") as provenance_file:
        for frame, provenance, action, state, _raw_main in _iter_source_frames(
            report,
            instruction,
            original_prefix_state,
            converter_helpers=converter_helpers,
            source_file_hashes=source_file_hashes,
        ):
            index = int(provenance["dataset_frame_index"])
            expected_actions[index] = action
            expected_states[index] = state
            phases[index] = str(provenance["task_phase"])
            if resume_existing:
                stored = json.loads(provenance_file.readline())
                if stored != provenance:
                    raise ValueError(f"resume_per_action_provenance_mismatch_at_{index}")
            else:
                compat.add_frame_to_dataset(dataset, frame)
                provenance_file.write(json.dumps(provenance, ensure_ascii=False) + "\n")
            provenance_rows += 1
        if resume_existing and provenance_file.readline():
            raise ValueError("resume_per_action_provenance_has_extra_rows")
        if not resume_existing:
            dataset.save_episode()
    if dataset is not None:
        image_writer = getattr(dataset, "image_writer", None)
        if image_writer is not None:
            image_writer.wait_until_done()
            image_writer.stop()
            dataset.image_writer = None
    if provenance_rows != EXPECTED_ACTIONS:
        raise RuntimeError(f"provenance_count_mismatch:{provenance_rows}")

    # Reader round-trip: all 1,460 action/state/image rows through LeRobot's
    # dataset reader, plus every valid unpadded OpenPI 10-action horizon.
    reader = LeRobotDataset(repo_id=output_root.name, root=output_root)
    starts, ends = compat.episode_boundaries(reader)
    if (
        len(reader) != EXPECTED_ACTIONS
        or int(reader.num_episodes) != 1
        or starts != [0]
        or ends != [EXPECTED_ACTIONS]
        or reader.meta.info.get("codebase_version") != "v2.1"
        or CODEBASE_VERSION != "v2.1"
    ):
        raise RuntimeError(
            f"lerobot_episode_metadata_mismatch:{len(reader)}:{reader.num_episodes}:{starts}:{ends}"
        )

    aligned = 0
    max_timestamp_error_s = 0.0
    for index in range(EXPECTED_ACTIONS):
        item = reader[index]
        action = np.asarray(item["actions"], dtype=np.float32)
        state = np.asarray(item["state"], dtype=np.float32)
        image = _as_hwc(item["image"])
        wrist = _as_hwc(item["wrist_image"])
        task = item["task"]
        if not np.array_equal(action, expected_actions[index]):
            raise RuntimeError(f"readback_action_mismatch_at_{index}")
        if not np.array_equal(state, expected_states[index]):
            raise RuntimeError(f"readback_state_mismatch_at_{index}")
        if image.shape != (256, 256, 3) or wrist.shape != (256, 256, 3):
            raise RuntimeError(f"readback_image_shape_mismatch_at_{index}")
        if task != instruction:
            raise RuntimeError(f"readback_task_mismatch_at_{index}")
        timestamp = float(np.asarray(item["timestamp"]).reshape(-1)[0])
        timestamp_error = abs(timestamp - index / float(FPS))
        max_timestamp_error_s = max(max_timestamp_error_s, timestamp_error)
        if timestamp_error > 5e-6:
            raise RuntimeError(f"readback_timestamp_mismatch_at_{index}:{timestamp}")
        aligned += 1

    horizon_reader = LeRobotDataset(
        repo_id=output_root.name,
        root=output_root,
        delta_timestamps={"actions": [offset / float(FPS) for offset in range(ACTION_HORIZON)]},
    )
    valid_windows = EXPECTED_ACTIONS - ACTION_HORIZON + 1
    for start in range(valid_windows):
        item = horizon_reader[start]
        window = np.asarray(item["actions"], dtype=np.float32)
        if window.shape != (ACTION_HORIZON, 7):
            raise RuntimeError(f"readback_horizon_shape_mismatch_at_{start}:{window.shape}")
        if not np.array_equal(window, expected_actions[start : start + ACTION_HORIZON]):
            raise RuntimeError(f"readback_horizon_action_mismatch_at_{start}")
        if start + ACTION_HORIZON - 1 >= EXPECTED_ACTIONS:
            raise RuntimeError(f"invalid_window_crosses_terminal:{start}")

    # Reopen accepted source images one segment at a time and compare every
    # LeRobot image pixel with the same pre-action source row and transform.
    image_reader = LeRobotDataset(repo_id=output_root.name, root=output_root)
    image_aligned = 0
    source_row_cursor = 0
    for segment_index, segment in enumerate(report["segments"]):
        with np.load(segment["trace"], allow_pickle=False) as archive:
            source = _load_segment_arrays(segment_index, archive)
            count = int(segment["physical_actions"])
            for row in range(count):
                main, wrist, _, _ = _observation_components(
                    segment_index,
                    row,
                    source,
                    instruction=instruction,
                    original_prefix_state=original_prefix_state,
                    previous_segment_end_state=(
                        _state_vector(
                            original_prefix_state["post_robot0_eef_pos"][-1],
                            original_prefix_state["post_robot0_eef_quat"][-1],
                            original_prefix_state["post_robot0_gripper_qpos"][-1],
                            quaternion_order="xyzw",
                        )
                        if segment_index == 2
                        else None
                    ),
                )
                expected_main = converter_helpers._frame_image({"camera": main}, "camera")
                expected_wrist = converter_helpers._frame_image({"camera": wrist}, "camera")
                item = image_reader[source_row_cursor]
                if not np.array_equal(_as_hwc(item["image"]), expected_main):
                    raise RuntimeError(f"readback_agentview_pixel_mismatch_at_{source_row_cursor}")
                if not np.array_equal(_as_hwc(item["wrist_image"]), expected_wrist):
                    raise RuntimeError(f"readback_wrist_pixel_mismatch_at_{source_row_cursor}")
                source_row_cursor += 1
                image_aligned += 1
    if image_aligned != EXPECTED_ACTIONS:
        raise RuntimeError(f"image_readback_count_mismatch:{image_aligned}")

    segment_action_counts = {
        str(i): int(segment["physical_actions"])
        for i, segment in enumerate(report["segments"])
    }
    phase_action_counts: dict[str, int] = {}
    for phase in phases:
        phase_action_counts[phase] = phase_action_counts.get(phase, 0) + 1
    phase_window_counts: dict[str, int] = {}
    for start in range(valid_windows):
        phase = phases[start]
        phase_window_counts[phase] = phase_window_counts.get(phase, 0) + 1
    seam_timesteps = [int(item["timestep"]) for item in report["snapshot_boundaries"]]
    windows_crossing_any_seam = sum(
        any(start < seam <= start + ACTION_HORIZON - 1 for seam in seam_timesteps)
        for start in range(valid_windows)
    )
    window_counts_per_seam = {
        str(seam): sum(start < seam <= start + ACTION_HORIZON - 1 for start in range(valid_windows))
        for seam in seam_timesteps
    }

    # Include action traces, safety evidence, report, initial source, original
    # t=10 state source, all segment snapshots, and report-level source files.
    manifest = {
        "schema": "gs_opd_round0_l0t2_lerobot_v21_data_manifest.v1",
        "dataset_path": str(output_root.resolve()),
        "format": "LeRobot v2.1",
        "lerobot_package_version": importlib.metadata.version("lerobot"),
        "fps": FPS,
        "episode_count": 1,
        "episode_length": EXPECTED_ACTIONS,
        "total_action_samples": EXPECTED_ACTIONS,
        "action_horizon": ACTION_HORIZON,
        "task": instruction,
        "camera_features": {
            "image": "pre-action agentview; 180-degree RGB rotation from the existing GS-OPD OpenPI converter",
            "wrist_image": "pre-action robot0_eye_in_hand; 180-degree RGB rotation from the existing GS-OPD OpenPI converter",
        },
        "state_feature": "pre-action eef position xyz + robosuite axis-angle xyz + gripper qpos[2], float32; existing LeRobot 8D convention",
        "state_quaternion_convention": "robosuite observation quaternions are XYZW; t249-to-t329 branch WXYZ is converted to XYZW before the existing axis-angle transform",
        "action_feature": "recorded executed float32 7D action, unnormalized; predicted_actions excluded",
        "segments": [
            {
                "segment_index": i,
                "global_timestep_range_exclusive": [int(s["start_timestep"]), int(s["end_timestep"])],
                "physical_actions": int(s["physical_actions"]),
                "trace": s["trace"],
                "trace_sha256": s["trace_sha256"],
                "safety_status": {
                    "native_safety": s["safety_attestation"]["native_safety"],
                    "operational_safety": s["safety_attestation"]["operational_safety"],
                    "substeps": int(s["safety_attestation"]["expected_substeps"]),
                    "strict_10mm": "not_passed_additional_metric_only",
                },
                "start_snapshot": s.get("start_snapshot"),
                "end_snapshot": s.get("end_snapshot"),
            }
            for i, s in enumerate(report["segments"])
        ],
        "snapshot_seams": report["snapshot_boundaries"],
        "source_files": source_file_hashes,
        "source_file_hashes": {
            "acceptance_report_sha256": acceptance_hash,
            "raw_pi05_trace_sha256": task_source_hash,
            "episode9_original_OA_trace_sha256": episode9_prefix_hash,
        },
        "source_candidate_count": 1,
        "source_segment_count": EXPECTED_SEGMENTS,
        "successful_episodes": 1,
        "strict_10mm_pass": False,
        "operational_safety_pass": True,
        "native_task_success": True,
        "reward": 1.0,
        "done": True,
        "standard_1000_step_success": False,
        "demonstration_only_1800_step_success": True,
        "lora_training_started": False,
        "episode_acceptance_repeated_in_each_action_provenance_row": True,
    }
    _write_json(manifest_path, manifest)

    acceptance = {
        "schema": "gs_opd_round0_l0t2_lerobot_v21_acceptance.v1",
        "status": "verified",
        "source_acceptance_report": str(acceptance_report),
        "source_acceptance_report_sha256": report_hash,
        "dataset_path": str(output_root.resolve()),
        "format": "LeRobot v2.1",
        "lerobot_package_version": importlib.metadata.version("lerobot"),
        "episode_count": int(reader.num_episodes),
        "sample_count": int(len(reader)),
        "episode_lengths": [EXPECTED_ACTIONS],
        "episode_boundaries": {"starts": starts, "ends": ends},
        "fps": FPS,
        "task": instruction,
        "fields": features,
        "data_provenance": {
            "accepted_source_segments": EXPECTED_SEGMENTS,
            "one_successful_episode_not_25_episodes": True,
            "per_action_provenance_rows": provenance_rows,
            "episode_success_horizon_and_strict10_provenance_repeated_per_action": True,
            "provenance_sidecar": str(provenance_path),
            "source_file_hash_manifest": str(manifest_path),
        },
        "source_trajectory": {
            "physical_actions": EXPECTED_ACTIONS,
            "physical_substeps": EXPECTED_ACTIONS * 25,
            "compatible_snapshot_seams": len(report["snapshot_boundaries"]),
            "terminal_timestep": 1460,
            "native_task_success": True,
            "reward": 1.0,
            "done": True,
            "demonstration_only_horizon_success": True,
            "standard_1000_step_success": False,
        },
        "safety_acceptance": {
            "native_safety": "pass",
            "operational_safety": "pass",
            "all_action_substep_records_covered": True,
            "substeps_per_action": 25,
            "hard_safety_violations": 0,
            "strict_10mm": "not_passed_additional_metric_only",
            "strict_10mm_is_not_the_operational_acceptance_gate": True,
        },
        "action_readback": {
            "actions_are_recorded_executed_actions": True,
            "predicted_actions_used": False,
            "normalization_applied": False,
            "readback_exact_samples": aligned,
            "readback_total_samples": EXPECTED_ACTIONS,
            "all_actions_bitwise_equal_float32": aligned == EXPECTED_ACTIONS,
        },
        "observation_readback": {
            "images_are_pre_action": True,
            "all_camera_pixels_roundtrip_exact_after_existing_rotation": image_aligned == EXPECTED_ACTIONS,
            "all_state_vectors_roundtrip_exact_float32": aligned == EXPECTED_ACTIONS,
            "timestamp_max_absolute_error_seconds": max_timestamp_error_s,
            "all_tasks_match": True,
        },
        "training_windows": {
            "action_horizon": ACTION_HORIZON,
            "valid_unpadded_windows": valid_windows,
            "last_valid_start_index": valid_windows - 1,
            "last_valid_action_index": EXPECTED_ACTIONS - 1,
            "valid_windows_cross_terminal": False,
            "windows_crossing_one_or_more_accepted_snapshot_seams": windows_crossing_any_seam,
            "windows_per_snapshot_seam": window_counts_per_seam,
            "action_samples_by_start_phase": phase_window_counts,
            "action_counts_by_phase": phase_action_counts,
            "phase_counts_are_start_step_labels": True,
        },
        "lora_training_started": False,
    }
    _write_json(acceptance_output_path, acceptance)
    return acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance-report", type=Path, default=DEFAULT_ACCEPTANCE_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="verify a complete dataset created by this converter after an interrupted read-back",
    )
    args = parser.parse_args()
    result = convert_and_verify(
        args.acceptance_report, args.output_root, resume_existing=args.resume_existing
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
