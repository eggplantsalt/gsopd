# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert a trajectory-writer trace to fixed-size supervised action chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
CHUNK_LENGTH = 5
EPISODE_ID = "obstacle_avoidance_l0_t0_20261019_adaptive_attempt1"


def _scalar_string(value: Any) -> str:
    return str(np.asarray(value).item())


def _load_source(source_path: Path) -> dict[str, np.ndarray]:
    with np.load(source_path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _decision_observation_lookup(
    source: dict[str, np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    lookup: dict[int, dict[str, np.ndarray]] = {}
    for row, step_index in enumerate(source["decision_step_indices"].tolist()):
        step_index = int(step_index)
        observation = {
            f"{camera}": source[f"decision_{camera}"][row]
            for camera in CAMERAS
        }
        observation.update({
            key: source[f"decision_{key}"][row] for key in STATE_KEYS
        })
        if step_index in lookup:
            raise ValueError(f"duplicate_decision_observation_at_step_{step_index}")
        lookup[step_index] = observation
    return lookup


def _source_ranges(
    actions: np.ndarray, sources: np.ndarray
) -> dict[str, dict[str, Any]]:
    result = {}
    for source_name in dict.fromkeys(sources.tolist()):
        values = actions[sources == source_name]
        result[str(source_name)] = {
            "action_count": int(len(values)),
            "minimum_by_dimension": values.min(axis=0).astype(float).tolist(),
            "maximum_by_dimension": values.max(axis=0).astype(float).tolist(),
        }
    return result


def _action_chunks(
    step_indices: np.ndarray,
    stages: np.ndarray,
    source_chunk_indices: np.ndarray,
    *,
    chunk_length: int,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Make non-overlapping chunks inside continuous, source-consistent runs."""
    chunks: list[np.ndarray] = []
    dropped: dict[str, int] = {}
    start = 0
    while start < len(step_indices):
        stage = str(stages[start])
        original_chunk = int(source_chunk_indices[start])
        end = start + 1
        while end < len(step_indices):
            same_stage = str(stages[end]) == stage
            same_student_decision = (
                stage == "student"
                and int(source_chunk_indices[end]) == original_chunk
            )
            continuous = int(step_indices[end]) == int(step_indices[end - 1]) + 1
            same_run = same_stage and continuous and (
                stage != "student" or same_student_decision
            )
            if not same_run:
                break
            end += 1
        run_length = end - start
        complete_count = run_length // chunk_length
        for offset in range(complete_count):
            first = start + offset * chunk_length
            chunks.append(np.arange(first, first + chunk_length, dtype=np.int64))
        tail = run_length % chunk_length
        if tail:
            dropped[stage] = dropped.get(stage, 0) + tail
        start = end
    return chunks, dropped


def convert_episode(
    source_path: Path,
    report_path: Path,
    output_path: Path,
    *,
    chunk_length: int = CHUNK_LENGTH,
) -> dict[str, Any]:
    """Create one immutable-source Episode chunk archive; refuse to overwrite."""
    if chunk_length <= 0:
        raise ValueError("chunk_length_must_be_positive")
    if output_path.exists():
        raise FileExistsError(f"refusing_to_overwrite_episode_archive:{output_path}")

    source = _load_source(source_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trace_metadata = json.loads(_scalar_string(source["metadata_json"]))
    episode_rollout = report.get("episode_rollout") or {}
    if (
        report.get("route_completed") is not True
        or episode_rollout.get("success") is not True
        or episode_rollout.get("done") is not True
    ):
        raise ValueError("source_episode_is_not_a_completed_successful_left_bypass_rollout")

    actions = np.asarray(source["executed_actions"])
    step_indices = np.asarray(source["step_indices"], dtype=np.int64)
    rewards = np.asarray(source["step_rewards"], dtype=np.float32)
    dones = np.asarray(source["step_dones"], dtype=np.bool_)
    original_chunk_indices = np.asarray(source["step_chunk_indices"], dtype=np.int64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("source_executed_actions_must_be_finite_N_by_7")
    if not (
        len(step_indices) == len(actions) == len(rewards) == len(dones)
        == len(original_chunk_indices)
    ):
        raise ValueError("source_transition_arrays_have_inconsistent_lengths")
    if len(step_indices) == 0 or not np.array_equal(
        step_indices, np.arange(step_indices[0], step_indices[0] + len(step_indices))
    ):
        raise ValueError("source_episode_has_noncontiguous_steps; refusing_to_join_action_chunks")
    if int(episode_rollout.get("executed_steps", -1)) != len(actions):
        raise ValueError("source_trace_transition_count_does_not_match_success_report")
    if trace_metadata.get("transition_count") != len(actions):
        raise ValueError("trajectory_writer_transition_count_mismatch")

    direct_start = int(report["decision_step"])
    direct_stop = direct_start + int(report["real_execution_steps"])
    if int(report["real_execution_steps"]) <= 0:
        raise ValueError("source_report_has_no_real_direct_OSC_actions")
    expected_steps = np.arange(step_indices[0], step_indices[0] + len(actions))
    if direct_start < expected_steps[0] or direct_stop > expected_steps[-1] + 1:
        raise ValueError("direct_OSC_stage_bounds_fall_outside_source_episode")

    task_instruction = str(trace_metadata["instruction"])
    if not task_instruction or task_instruction != episode_rollout.get("instruction"):
        raise ValueError("source_task_instruction_mismatch")
    stages = np.where(
        step_indices < direct_start,
        "student_pre_bypass",
        np.where(step_indices < direct_stop, "bspline_osc", "student_post_bypass"),
    ).astype("U24")
    action_sources = np.where(
        stages == "bspline_osc", "native_bspline_osc_pose", "pi05_student"
    ).astype("U32")
    expected_direct_count = int(np.count_nonzero(stages == "bspline_osc"))
    if expected_direct_count != int(report["real_execution_steps"]):
        raise ValueError("source_stage_boundaries_do_not_match_real_OSC_action_count")

    decision_lookup = _decision_observation_lookup(source)
    if int(step_indices[0]) not in decision_lookup:
        raise ValueError("source_episode_has_no_initial_observation_at_first_action")
    for camera in CAMERAS:
        next_images = source[f"post_{camera}"]
        if len(next_images) != len(actions) or next_images.ndim != 4:
            raise ValueError(f"invalid_post_observation_camera_array:{camera}")
    for key in STATE_KEYS:
        if len(source[f"post_{key}"]) != len(actions):
            raise ValueError(f"invalid_post_observation_state_array:{key}")

    chunk_indices, dropped_tails = _action_chunks(
        step_indices,
        np.where(stages == "bspline_osc", "bspline_osc", "student"),
        original_chunk_indices,
        chunk_length=chunk_length,
    )
    if not chunk_indices:
        raise ValueError("source_episode_has_no_complete_action_chunks")

    sample_observations: dict[str, list[np.ndarray]] = {
        key: [] for key in (*CAMERAS, *STATE_KEYS)
    }
    sample_next_observations: dict[str, list[np.ndarray]] = {
        key: [] for key in (*CAMERAS, *STATE_KEYS)
    }
    sample_step_indices = []
    sample_actions = []
    sample_rewards = []
    sample_dones = []
    sample_stages = []
    sample_sources = []
    sample_original_chunk_indices = []
    sample_observation_origins = []

    row_by_step = {int(step): index for index, step in enumerate(step_indices.tolist())}
    for rows in chunk_indices:
        first_row, final_row = int(rows[0]), int(rows[-1])
        first_step = int(step_indices[first_row])
        if first_step in decision_lookup:
            initial_observation = decision_lookup[first_step]
            observation_origin = "trajectory_writer_decision_observation"
        else:
            previous_row = row_by_step.get(first_step - 1)
            if previous_row is None:
                raise ValueError(f"action_chunk_has_no_observation_at_step_{first_step}")
            initial_observation = {
                key: source[f"post_{key}"][previous_row]
                for key in (*CAMERAS, *STATE_KEYS)
            }
            observation_origin = "previous_transition_next_observation"
        for camera in CAMERAS:
            sample_observations[camera].append(initial_observation[camera])
            sample_next_observations[camera].append(source[f"post_{camera}"][final_row])
        for key in STATE_KEYS:
            sample_observations[key].append(initial_observation[key])
            sample_next_observations[key].append(source[f"post_{key}"][final_row])

        sample_step_indices.append(step_indices[rows])
        sample_actions.append(actions[rows])
        sample_rewards.append(rewards[rows])
        sample_dones.append(dones[rows])
        sample_stages.append(str(stages[first_row]))
        sample_sources.append(str(action_sources[first_row]))
        sample_original_chunk_indices.append(int(original_chunk_indices[first_row]))
        sample_observation_origins.append(observation_origin)

    stages_per_action = stages.tolist()
    action_source_stats = {}
    action_source_action_totals: dict[str, int] = {}
    action_source_sample_totals: dict[str, int] = {}
    for stage in dict.fromkeys(stages_per_action):
        mask = stages == stage
        values = actions[mask]
        source_name = str(action_sources[np.flatnonzero(mask)[0]])
        source_sample_count = int(sum(item == stage for item in sample_stages))
        action_source_action_totals[source_name] = (
            action_source_action_totals.get(source_name, 0) + int(mask.sum())
        )
        action_source_sample_totals[source_name] = (
            action_source_sample_totals.get(source_name, 0) + source_sample_count
        )
        action_source_stats[str(stage)] = {
            "action_source": source_name,
            "action_count": int(mask.sum()),
            "sample_count": source_sample_count,
            "minimum_by_action_dimension": values.min(axis=0).astype(float).tolist(),
            "maximum_by_action_dimension": values.max(axis=0).astype(float).tolist(),
        }

    episode_id = EPISODE_ID
    success = bool(episode_rollout["success"])
    output_metadata = {
        "schema": "gsopd_episode_action_chunks",
        "schema_version": 1,
        "episode_id": episode_id,
        "task_instruction": task_instruction,
        "episode_success": success,
        "supervision_kind": "successful_episode_behavior_cloning",
        "safety_correction_label": "not_assigned",
        "source_trajectory_writer_schema_version": trace_metadata.get("trace_schema_version"),
        "source_archive": str(source_path.resolve()),
        "source_report": str(report_path.resolve()),
        "source_transition_count": int(len(actions)),
        "transition_step_indices": [int(step_indices[0]), int(step_indices[-1])],
        "chunk_length": int(chunk_length),
        "chunk_stride": int(chunk_length),
        "chunk_policy": "nonoverlapping; Student windows preserve original decision chunk boundaries; native OSC windows split only within the contiguous OSC stage",
        "partial_action_tails_omitted_from_samples": dropped_tails,
        "sample_count": len(chunk_indices),
        "sample_action_count": int(len(chunk_indices) * chunk_length),
        "camera_keys": list(CAMERAS),
        "camera_shape": list(source[f"post_{CAMERAS[0]}"][0].shape),
        "state_keys": list(STATE_KEYS),
        "action_shape_per_sample": [int(chunk_length), 7],
        "action_dtype": str(actions.dtype),
        "action_transform": "identity; raw executed_actions preserved as recorded; no normalization or rescaling",
        "task_success_reward_sum": float(rewards.sum()),
        "terminal_transition_count": int(dones.sum()),
        "action_source_statistics": action_source_stats,
        "action_source_action_totals": action_source_action_totals,
        "action_source_sample_totals": action_source_sample_totals,
    }

    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(output_metadata, ensure_ascii=False, sort_keys=True)),
        "sample_episode_id": np.asarray([episode_id] * len(chunk_indices), dtype=np.str_),
        "sample_task_instruction": np.asarray([task_instruction] * len(chunk_indices), dtype=np.str_),
        "sample_episode_success": np.full(len(chunk_indices), success, dtype=np.bool_),
        "sample_supervision_kind": np.asarray(
            ["successful_episode_behavior_cloning"] * len(chunk_indices), dtype=np.str_
        ),
        "sample_safety_correction_label": np.asarray(
            ["not_assigned"] * len(chunk_indices), dtype=np.str_
        ),
        "sample_stage": np.asarray(sample_stages, dtype=np.str_),
        "sample_action_source": np.asarray(sample_sources, dtype=np.str_),
        "sample_observation_origin": np.asarray(sample_observation_origins, dtype=np.str_),
        "sample_source_chunk_index": np.asarray(sample_original_chunk_indices, dtype=np.int64),
        "sample_step_indices": np.stack(sample_step_indices),
        "action_chunks": np.stack(sample_actions).astype(np.float32, copy=False),
        "reward_chunks": np.stack(sample_rewards).astype(np.float32, copy=False),
        "done_chunks": np.stack(sample_dones).astype(np.bool_, copy=False),
    }
    for camera in CAMERAS:
        arrays[f"observation_{camera}"] = np.stack(sample_observations[camera])
        arrays[f"next_observation_{camera}"] = np.stack(sample_next_observations[camera])
    for key in STATE_KEYS:
        arrays[f"observation_{key}"] = np.stack(sample_observations[key])
        arrays[f"next_observation_{key}"] = np.stack(sample_next_observations[key])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output_file:
        np.savez_compressed(output_file, **arrays)
    return {
        "status": "converted",
        "output_path": str(output_path.resolve()),
        **output_metadata,
        "action_source_statistics": action_source_stats,
    }


class EpisodeChunkDataset(Dataset):
    """PyTorch Dataset for one converted successful Episode."""

    def __init__(self, archive_path: str | Path):
        self.archive_path = Path(archive_path)
        with np.load(self.archive_path, allow_pickle=False) as archive:
            self.arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
        self.metadata = json.loads(_scalar_string(self.arrays["metadata_json"]))
        self.sample_count = len(self.arrays["action_chunks"])

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        arrays = self.arrays

        def tensor(name: str) -> torch.Tensor:
            return torch.from_numpy(np.array(arrays[name][index], copy=True))

        observation = {
            "images": {
                camera: tensor(f"observation_{camera}") for camera in CAMERAS
            },
            "state": {key: tensor(f"observation_{key}") for key in STATE_KEYS},
        }
        next_observation = {
            "images": {
                camera: tensor(f"next_observation_{camera}") for camera in CAMERAS
            },
            "state": {
                key: tensor(f"next_observation_{key}") for key in STATE_KEYS
            },
        }
        return {
            "episode_id": str(arrays["sample_episode_id"][index]),
            "task_instruction": str(arrays["sample_task_instruction"][index]),
            "episode_success": bool(arrays["sample_episode_success"][index]),
            "stage": str(arrays["sample_stage"][index]),
            "action_source": str(arrays["sample_action_source"][index]),
            "supervision_kind": str(arrays["sample_supervision_kind"][index]),
            "safety_correction_label": str(arrays["sample_safety_correction_label"][index]),
            "observation_origin": str(arrays["sample_observation_origin"][index]),
            "source_chunk_index": int(arrays["sample_source_chunk_index"][index]),
            "step_indices": tensor("sample_step_indices"),
            "observation": observation,
            "action_chunk": tensor("action_chunks"),
            "reward_chunk": tensor("reward_chunks"),
            "done_chunk": tensor("done_chunks"),
            "next_observation": next_observation,
        }


def run_conversion_and_loader_smoke(
    source_path: Path,
    report_path: Path,
    output_path: Path,
    smoke_report_path: Path,
    *,
    seed: int = 20261019,
) -> dict[str, Any]:
    if smoke_report_path.exists():
        raise FileExistsError(f"refusing_to_overwrite_smoke_report:{smoke_report_path}")
    conversion = convert_episode(source_path, report_path, output_path)
    dataset = EpisodeChunkDataset(output_path)
    sample_index = int(np.random.default_rng(seed).integers(0, len(dataset)))
    loader = DataLoader(dataset, batch_size=1, sampler=[sample_index], num_workers=0)
    batch = next(iter(loader))

    first = dataset[sample_index]
    input_shapes = {
        "agentview_image": list(batch["observation"]["images"][CAMERAS[0]].shape[1:]),
        "wrist_image": list(batch["observation"]["images"][CAMERAS[1]].shape[1:]),
        "state": {
            key: list(batch["observation"]["state"][key].shape[1:])
            for key in STATE_KEYS
        },
    }
    output_shapes = {
        "action_chunk": list(batch["action_chunk"].shape[1:]),
        "reward_chunk": list(batch["reward_chunk"].shape[1:]),
        "done_chunk": list(batch["done_chunk"].shape[1:]),
        "next_agentview_image": list(
            batch["next_observation"]["images"][CAMERAS[0]].shape[1:]
        ),
        "next_wrist_image": list(
            batch["next_observation"]["images"][CAMERAS[1]].shape[1:]
        ),
    }
    summary = {
        "status": "smoke_passed",
        "conversion": conversion,
        "loader": {
            "dataset_type": type(dataset).__name__,
            "loader_type": type(loader).__name__,
            "sample_count": len(dataset),
            "random_seed": seed,
            "random_sample_index": sample_index,
            "input_shapes": input_shapes,
            "label_shapes": output_shapes,
            "random_sample": {
                "episode_id": first["episode_id"],
                "task_instruction": first["task_instruction"],
                "episode_success": first["episode_success"],
                "stage": first["stage"],
                "action_source": first["action_source"],
                "supervision_kind": first["supervision_kind"],
                "safety_correction_label": first["safety_correction_label"],
                "observation_origin": first["observation_origin"],
                "step_indices": first["step_indices"].tolist(),
                "input": {
                    "images": {
                        key: {
                            "shape": list(first["observation"]["images"][key].shape),
                            "dtype": str(first["observation"]["images"][key].dtype),
                        }
                        for key in CAMERAS
                    },
                    "state": {
                        key: {
                            "shape": list(first["observation"]["state"][key].shape),
                            "dtype": str(first["observation"]["state"][key].dtype),
                        }
                        for key in STATE_KEYS
                    },
                },
                "label": {
                    "action_chunk_shape": list(first["action_chunk"].shape),
                    "action_dtype": str(first["action_chunk"].dtype),
                    "action_chunk_raw_values": first["action_chunk"].tolist(),
                    "reward_chunk": first["reward_chunk"].tolist(),
                    "done_chunk": first["done_chunk"].tolist(),
                },
            },
        },
    }
    smoke_report_path.parent.mkdir(parents=True, exist_ok=True)
    with smoke_report_path.open("x", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        output_file.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    result = run_conversion_and_loader_smoke(
        args.source_archive,
        args.source_report,
        args.output,
        args.smoke_report,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
