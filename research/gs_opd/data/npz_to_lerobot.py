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
"""Convert the full successful GS-OPD episode to one LeRobot episode."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


RUN_DIR = Path(
    "logs/gs_opd/mink_panda_physical_v3/"
    "live_left_bspline_osc_adaptive_20261019_attempt1"
)
SOURCE_NPZ = RUN_DIR / "direct_bspline_osc_trajectory.npz"
SOURCE_REPORT = RUN_DIR / "live_left_bspline_osc_report.json"
OUTPUT_ROOT = Path("/opt/data/private/gs_opd/datasets/lerobot_full_episode_v21")
STEP_COUNT = 845
ACTION_HORIZON = 10
PHASE_RANGES = {
    "student_grasp": (0, 55),
    "bspline_osc_carry": (55, 804),
    "student_place": (804, 845),
}
FPS = 20

CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
LEROBOT_FEATURES = {
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _state_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    # Use the same robosuite axis-angle conversion used by LIBERO observations.
    from rlinf.envs.sim.libero.utils import quat2axisangle

    axis_angle = quat2axisangle(
        np.asarray(observation["robot0_eef_quat"], dtype=np.float64).copy()
    )
    return np.concatenate(
        [
            observation["robot0_eef_pos"],
            axis_angle,
            observation["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)


def _frame_image(
    observation: dict[str, np.ndarray], key: str
) -> np.ndarray:
    # These are the exact 180-degree image rotations used by pi05_rollout.py.
    value = np.asarray(observation[key])[::-1, ::-1]
    if value.shape != (256, 256, 3) or value.dtype != np.uint8:
        raise ValueError(f"unexpected_image_{key}_{value.shape}_{value.dtype}")
    return np.ascontiguousarray(value)


def _observation(
    source: dict[str, np.ndarray], step: int, decision_rows: dict[int, int], step_rows: dict[int, int]
) -> dict[str, np.ndarray]:
    if step in decision_rows:
        prefix, row = "decision_", decision_rows[step]
    else:
        previous_step = step - 1
        if previous_step not in step_rows:
            raise ValueError(f"missing_previous_next_observation_for_step_{step}")
        prefix, row = "post_", step_rows[previous_step]
    result = {
        key: source[f"{prefix}{key}"][row]
        for key in (*CAMERA_KEYS, *STATE_KEYS)
    }
    return result


def _as_hwc(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.shape == (3, 256, 256):
        result = np.transpose(result, (1, 2, 0))
    # LeRobot's image reader returns float CHW values in [0, 1], even though
    # the image feature is stored losslessly from the original uint8 pixels.
    if np.issubdtype(result.dtype, np.floating) and result.size and result.max() <= 1.0:
        result = np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)
    return result


def convert_and_verify(
    source_npz: Path, source_report: Path, output_root: Path
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing_to_overwrite_dataset:{output_root}")

    report = json.loads(source_report.read_text(encoding="utf-8"))
    osc_steps = int(report.get("real_execution_steps", -1))
    decision_step = int(report.get("decision_step", -1))
    if (
        report.get("route_completed") is not True
        or osc_steps != PHASE_RANGES["bspline_osc_carry"][1] - PHASE_RANGES["bspline_osc_carry"][0]
        or decision_step != PHASE_RANGES["bspline_osc_carry"][0]
    ):
        raise ValueError("source_report_does_not_confirm_successful_osc_segment_boundaries")

    source = _load_npz(source_npz)
    trace_metadata = json.loads(str(source["metadata_json"].item()))
    instruction = str(trace_metadata["instruction"])
    if trace_metadata.get("steps") != STEP_COUNT:
        raise ValueError(f"expected_{STEP_COUNT}_source_steps_found_{trace_metadata.get('steps')}")
    source_episode_success = bool(
        np.any((source["step_rewards"] > 0) & source["step_dones"])
    )
    if not source_episode_success:
        raise ValueError("source_npz_does_not_confirm_successful_episode")
    step_indices = np.asarray(source["step_indices"], dtype=np.int64)
    selected_steps = np.arange(STEP_COUNT, dtype=np.int64)
    if not np.array_equal(step_indices, selected_steps):
        raise ValueError("raw_npz_does_not_contain_steps_0_through_844_contiguously")
    if source["executed_actions"].shape != (STEP_COUNT, 7):
        raise ValueError(f"unexpected_full_episode_action_shape_{source['executed_actions'].shape}")
    if source["step_dones"].sum() != 1 or not source["step_dones"][-1]:
        raise ValueError("source_episode_terminal_transition_must_be_step_844")
    if float(source["step_rewards"][-1]) != 1.0 or np.any(source["step_rewards"][:-1] != 0):
        raise ValueError("source_episode_reward_must_be_sparse_success_at_step_844")

    report_steps = report.get("steps", [])
    osc_start, osc_end = PHASE_RANGES["bspline_osc_carry"]
    report_indices = np.asarray([int(item["step_index"]) for item in report_steps])
    expected_osc_steps = np.arange(osc_start, osc_end, dtype=np.int64)
    if not np.array_equal(report_indices, expected_osc_steps):
        raise ValueError("report_steps_do_not_match_osc_steps_55_through_803")
    report_step_by_index = {int(item["step_index"]): item for item in report_steps}

    decision_rows: dict[int, int] = {}
    for row, step in enumerate(source["decision_step_indices"].tolist()):
        step = int(step)
        if step in decision_rows:
            raise ValueError(f"duplicate_decision_observation_at_step_{step}")
        decision_rows[step] = row
    step_rows = {int(step): row for row, step in enumerate(step_indices.tolist())}
    if len(step_rows) != len(step_indices):
        raise ValueError("duplicate_raw_step_indices")
    for required_step in (0, osc_start, osc_end):
        if required_step not in decision_rows:
            raise ValueError(f"missing_phase_boundary_decision_observation_at_{required_step}")

    # Frames are pre-action Observation + the actual 7D environment action.
    frames: list[dict[str, Any]] = []
    source_frames: list[dict[str, np.ndarray]] = []
    action_rows = np.arange(STEP_COUNT, dtype=np.int64)
    for step in selected_steps.tolist():
        observation = _observation(source, step, decision_rows, step_rows)
        action = np.asarray(source["executed_actions"][step_rows[step]], dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"unexpected_action_shape_at_step_{step}_{action.shape}")
        if step in report_step_by_index:
            report_action = np.asarray(report_step_by_index[step]["osc_action_7d"], dtype=np.float32)
            if not np.allclose(action, report_action, atol=1e-7, rtol=0):
                raise ValueError(f"report_action_mismatch_at_step_{step}")
        if step > 0:
            previous = _observation(source, step - 1, decision_rows, step_rows)
            for key in (*CAMERA_KEYS, *STATE_KEYS):
                if not np.array_equal(observation[key], source[f"post_{key}"][step_rows[step - 1]]):
                    raise ValueError(f"observation_transition_alignment_failed_at_step_{step}_{key}")

        frames.append(
            {
                "image": _frame_image(observation, "agentview_image"),
                "wrist_image": _frame_image(observation, "robot0_eye_in_hand_image"),
                "state": _state_from_observation(observation),
                "actions": action,
                "task": instruction,
            }
        )
        source_frames.append(observation)

    # Writer.create does not expose LeRobot's root argument; point its normal
    # HF_LEROBOT_HOME/repo_id path at the requested local output directory.
    repo_id = output_root.name
    os.environ["HF_LEROBOT_HOME"] = str(output_root.parent.resolve())

    from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

    writer = LeRobotDatasetWriter()
    writer.create(
        repo_id=repo_id,
        robot_type="franka_panda",
        fps=FPS,
        features=LEROBOT_FEATURES,
        image_writer_threads=0,
        image_writer_processes=0,
    )
    writer.add_episode(frames)
    writer.finalize()

    # Verify the just-written episode through LeRobot's own dataset reader.
    import lerobot
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from rlinf.data.storage.lerobot.compat import episode_boundaries

    dataset = LeRobotDataset(repo_id=repo_id, root=output_root)
    starts, ends = episode_boundaries(dataset)
    if len(dataset) != STEP_COUNT or starts != [0] or ends != [STEP_COUNT]:
        raise RuntimeError(
            f"lerobot_episode_boundary_mismatch_len_{len(dataset)}_{starts}_{ends}"
        )

    first = dataset[0]
    last = dataset[STEP_COUNT - 1]
    selected_indices = (0, STEP_COUNT - 1)
    alignment = []
    aligned_steps = 0
    for source_index in range(STEP_COUNT):
        item = dataset[source_index]
        expected_observation = source_frames[source_index]
        expected_action = source["executed_actions"][step_rows[source_index]]
        expected_state = _state_from_observation(expected_observation)
        main_image = _as_hwc(item["image"])
        wrist_image = _as_hwc(item["wrist_image"])
        action = np.asarray(item["actions"], dtype=np.float32)
        state = np.asarray(item["state"], dtype=np.float32)
        task = item["task"]
        if main_image.shape != (256, 256, 3) or wrist_image.shape != (256, 256, 3):
            raise RuntimeError("lerobot_image_shape_mismatch")
        if not np.array_equal(action, expected_action):
            raise RuntimeError(f"lerobot_action_alignment_failed_at_{source_index}")
        if not np.array_equal(state, expected_state):
            raise RuntimeError(f"lerobot_state_alignment_failed_at_{source_index}")
        if task != instruction:
            raise RuntimeError(f"lerobot_task_text_mismatch_at_{source_index}")
        expected_main = _frame_image(expected_observation, "agentview_image")
        expected_wrist = _frame_image(expected_observation, "robot0_eye_in_hand_image")
        main_exact = bool(np.array_equal(main_image, expected_main))
        wrist_exact = bool(np.array_equal(wrist_image, expected_wrist))
        if not main_exact or not wrist_exact:
            raise RuntimeError(f"lerobot_image_alignment_failed_at_{source_index}")
        aligned_steps += 1
        if source_index in selected_indices:
            alignment.append({
                "dataset_index": int(source_index),
                "source_step_index": int(selected_steps[source_index]),
                "action_exact": True,
                "state_exact": True,
                "task_exact": True,
                "image_exact": {"image": main_exact, "wrist_image": wrist_exact},
            })

    valid_window_count = STEP_COUNT - ACTION_HORIZON + 1
    phase_window_counts = {
        name: max(0, min(end, valid_window_count) - start)
        for name, (start, end) in PHASE_RANGES.items()
    }

    return {
        "status": "verified",
        "source_npz": str(source_npz),
        "source_report": str(source_report),
        "dataset_path": str(output_root.resolve()),
        "source_episode_id": "obstacle_avoidance_l0_t0_20261019_adaptive_attempt1",
        "lerobot_package_version": importlib.metadata.version("lerobot"),
        "lerobot_format_version": dataset.meta.info["codebase_version"],
        "lerobot_codebase_version": CODEBASE_VERSION,
        "fps": FPS,
        "episode_count": int(dataset.num_episodes),
        "sample_count": int(len(dataset)),
        "source_step_range_inclusive": [0, STEP_COUNT - 1],
        "stage_ranges_step_inclusive": {
            name: [start, end - 1] for name, (start, end) in PHASE_RANGES.items()
        },
        "stage_action_counts": {
            name: end - start for name, (start, end) in PHASE_RANGES.items()
        },
        "stage_unpadded_window_counts_by_start_step": phase_window_counts,
        "retained_cross_stage_window_start_ranges_inclusive": {
            "student_grasp_to_bspline_osc_carry": [
                PHASE_RANGES["student_grasp"][1] - ACTION_HORIZON + 1,
                PHASE_RANGES["student_grasp"][1] - 1,
            ],
            "bspline_osc_carry_to_student_place": [
                PHASE_RANGES["student_place"][0] - ACTION_HORIZON + 1,
                PHASE_RANGES["student_place"][0] - 1,
            ],
        },
        "action_horizon": ACTION_HORIZON,
        "valid_unpadded_training_windows": valid_window_count,
        "all_step_observation_action_alignment_verified": aligned_steps == STEP_COUNT,
        "all_step_alignment_count": aligned_steps,
        "task": instruction,
        "prompt_source": "LeRobot task text; pi05_libero prompt_from_task=True",
        "fields": {
            key: {
                "dtype": value["dtype"],
                "shape": list(value["shape"]),
            }
            for key, value in LEROBOT_FEATURES.items()
        },
        "reader_first_last_alignment": alignment,
        "actions_are_raw_executed_actions": True,
        "normalization_applied": False,
        "source_episode_success_confirmed_by_reward_and_done": source_episode_success,
        "full_episode_reward_sum": float(source["step_rewards"].sum()),
        "full_episode_done_count": int(source["step_dones"].sum()),
        "le_robot_episode_boundaries": {"starts": starts, "ends": ends},
        "source_actions_shape": list(source["executed_actions"][action_rows].shape),
        "readback_first_fields": sorted(first.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, default=SOURCE_NPZ)
    parser.add_argument("--source-report", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    report_path = args.output_root.with_name(args.output_root.name + "_conversion_report.json")
    if report_path.exists():
        raise FileExistsError(f"refusing_to_overwrite_conversion_report:{report_path}")
    result = convert_and_verify(args.source_npz, args.source_report, args.output_root)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"conversion_report={report_path}")


if __name__ == "__main__":
    main()
