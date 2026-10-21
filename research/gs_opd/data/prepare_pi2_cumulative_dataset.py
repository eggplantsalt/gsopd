#!/usr/bin/env python3
"""Build and verify the cumulative Round 0/1 GS-OPD LeRobot dataset.

This is an offline data operation. It does not create an environment, train a
model, or execute a rollout. The original LeRobot conversion helpers are reused
for image/state/action conversion; provenance and phase-balanced indices are
stored beside the standard LeRobot dataset so OpenPI's schema stays unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gs_opd.data.npz_to_lerobot import (  # noqa: E402
    FPS,
    LEROBOT_FEATURES,
    _as_hwc,
    _frame_image,
    _load_npz,
    _observation,
    _state_from_observation,
)


PHASES = ("student_grasp", "bspline_osc_carry", "student_place")
ACTION_HORIZON = 10
BASE = ROOT / "logs/gs_opd/mink_panda_physical_v3"
DEFAULT_OUTPUT = ROOT / "logs/gs_opd/datasets/"
DEFAULT_OUTPUT /= "pi2_cumulative_lerobot_v21_round0_1_20261021"

SOURCES = (
    {
        "round": 0,
        "episode_id": "round0_original_student_left_trajectory001",
        "run_dir": BASE / "live_left_bspline_osc_adaptive_20261019_attempt1",
        "npz_name": "direct_bspline_osc_trajectory.npz",
        "report_name": "live_left_bspline_osc_report.json",
        "result_name": None,
        "run_manifest_name": "launch.json",
        "route_artifact_name": None,
        "policy_version": "original_pi05_student_without_lora",
        "student_action_source": "pi05_base_checkpoint_live_observation_inference",
        "expected_steps": 845,
        "expected_osc_steps": 749,
        "expected_candidate_id": "trajectory-001",
    },
    {
        "round": 1,
        "episode_id": "round1_attempt3_lora_left_interp10_toward004",
        "run_dir": BASE
        / "live_left_bspline_candidate001_interp10_toward004_attempt3_lora_20261021_attempt1",
        "npz_name": "direct_bspline_osc_trajectory.npz",
        "report_name": "trajectory_validation.json",
        "result_name": "result.json",
        "run_manifest_name": "run_manifest.json",
        "route_artifact_name": "interpolated_left_route.json",
        "policy_version": "pi05_base_plus_attempt3_lora_adapter",
        "student_action_source": "attempt3_lora_student_live_observation_inference",
        "expected_steps": 859,
        "expected_osc_steps": 767,
        "expected_candidate_id": "trajectory-001-interp-10pct-toward-trajectory-004",
    },
    {
        "round": 1,
        "episode_id": "round1_attempt3_lora_left_interp20_toward004",
        "run_dir": BASE
        / "live_left_bspline_candidate001_interp20_toward004_attempt3_lora_20261021_attempt1",
        "npz_name": "direct_bspline_osc_trajectory.npz",
        "report_name": "trajectory_validation.json",
        "result_name": "result.json",
        "run_manifest_name": "run_manifest.json",
        "route_artifact_name": "interpolated_left_route.json",
        "policy_version": "pi05_base_plus_attempt3_lora_adapter",
        "student_action_source": "attempt3_lora_student_live_observation_inference",
        "expected_steps": 875,
        "expected_osc_steps": 783,
        "expected_candidate_id": "trajectory-001-interp-20pct-toward-trajectory-004",
    },
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_and_validate_source(spec: dict[str, Any]) -> dict[str, Any]:
    run_dir: Path = spec["run_dir"]
    npz_path = run_dir / spec["npz_name"]
    report_path = run_dir / spec["report_name"]
    if not npz_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"missing_source_artifact:{npz_path}:{report_path}")
    report = _json(report_path)
    result_path = run_dir / spec["result_name"] if spec["result_name"] else None
    result = _json(result_path) if result_path else {}
    manifest_path = run_dir / spec["run_manifest_name"]
    run_manifest = _json(manifest_path) if manifest_path.is_file() else {}
    route_path = (
        run_dir / spec["route_artifact_name"]
        if spec["route_artifact_name"]
        else None
    )
    route_artifact = _json(route_path) if route_path and route_path.is_file() else {}

    source = _load_npz(npz_path)
    trace_meta = json.loads(str(source["metadata_json"].item()))
    step_count = int(spec["expected_steps"])
    if trace_meta.get("steps") != step_count:
        raise ValueError(
            f"{spec['episode_id']}:expected_{step_count}_steps_got_{trace_meta.get('steps')}"
        )
    if not np.array_equal(source["step_indices"], np.arange(step_count, dtype=np.int64)):
        raise ValueError(f"{spec['episode_id']}:source_steps_are_not_contiguous")
    if source["executed_actions"].shape != (step_count, 7):
        raise ValueError(f"{spec['episode_id']}:unexpected_action_shape")
    if source["step_rewards"].shape != (step_count,) or source["step_dones"].shape != (
        step_count,
    ):
        raise ValueError(f"{spec['episode_id']}:unexpected_reward_or_done_shape")
    if (
        float(source["step_rewards"].sum()) != 1.0
        or int(source["step_dones"].sum()) != 1
        or not bool(source["step_dones"][-1])
        or float(source["step_rewards"][-1]) != 1.0
        or np.any(source["step_rewards"][:-1] != 0)
        or np.any(source["step_dones"][:-1])
    ):
        raise ValueError(f"{spec['episode_id']}:source_episode_not_terminal_success")
    if report.get("route_completed") is not True:
        raise ValueError(f"{spec['episode_id']}:route_not_confirmed_complete")
    if int(report.get("real_execution_steps", -1)) != int(spec["expected_osc_steps"]):
        raise ValueError(f"{spec['episode_id']}:osc_step_count_mismatch")

    osc_start = int(report["decision_step"])
    osc_end = osc_start + int(report["real_execution_steps"])
    if not (0 < osc_start < osc_end < step_count):
        raise ValueError(f"{spec['episode_id']}:invalid_stage_boundaries")
    report_steps = report.get("steps", [])
    expected_report_steps = np.arange(osc_start, osc_end, dtype=np.int64)
    actual_report_steps = np.asarray(
        [int(step["step_index"]) for step in report_steps], dtype=np.int64
    )
    if not np.array_equal(actual_report_steps, expected_report_steps):
        raise ValueError(f"{spec['episode_id']}:report_osc_step_indices_mismatch")
    report_by_step = {int(step["step_index"]): step for step in report_steps}
    for step_index, report_step in report_by_step.items():
        report_action = np.asarray(report_step["osc_action_7d"], dtype=np.float32)
        actual_action = np.asarray(source["executed_actions"][step_index], dtype=np.float32)
        if report_action.shape != (7,) or not np.allclose(
            report_action, actual_action, atol=1e-7, rtol=0
        ):
            raise ValueError(f"{spec['episode_id']}:osc_action_alignment_failed_at_{step_index}")

    candidate_id = str(
        trace_meta["direct_bspline_osc_execution"].get("route_candidate_id", "")
    )
    if candidate_id != spec["expected_candidate_id"]:
        raise ValueError(
            f"{spec['episode_id']}:expected_route_{spec['expected_candidate_id']}_got_{candidate_id}"
        )
    if spec["round"] == 1:
        direct_tag = 1_000_005
        chunk_ids = np.asarray(source["step_chunk_indices"], dtype=np.int64)
        if not np.array_equal(np.flatnonzero(chunk_ids == direct_tag), expected_report_steps):
            raise ValueError(f"{spec['episode_id']}:actual_osc_action_source_not_marked_exactly")
        if result.get("done") is not True or float(result.get("reward_sum", -1)) != 1.0:
            raise ValueError(f"{spec['episode_id']}:rollout_result_not_successful")
        selected_route = route_artifact.get("selected_candidate", {})
        route_candidate_id = str(selected_route.get("candidate_id", ""))
        if route_candidate_id != candidate_id or selected_route.get("direction") != "left":
            raise ValueError(f"{spec['episode_id']}:route_artifact_mismatch")
        if not selected_route.get("passes_geometric_screen"):
            raise ValueError(f"{spec['episode_id']}:route_did_not_pass_existing_geometric_screen")
        if not trace_meta.get("lora_checkpoint_path"):
            raise ValueError(f"{spec['episode_id']}:attempt3_adapter_not_recorded_in_npz")
    else:
        rollout = report.get("episode_rollout", {})
        if rollout.get("success") is not True or rollout.get("done") is not True:
            raise ValueError(f"{spec['episode_id']}:original_rollout_result_not_successful")

    decision_rows: dict[int, int] = {}
    for row, step in enumerate(source["decision_step_indices"].tolist()):
        step = int(step)
        if step in decision_rows:
            raise ValueError(f"{spec['episode_id']}:duplicate_decision_observation_{step}")
        decision_rows[step] = row
    step_rows = {int(step): row for row, step in enumerate(source["step_indices"].tolist())}
    if len(step_rows) != step_count:
        raise ValueError(f"{spec['episode_id']}:duplicate_source_step_indices")
    for required in (0, osc_start, osc_end):
        if required not in decision_rows:
            raise ValueError(f"{spec['episode_id']}:missing_boundary_observation_{required}")

    instruction = str(trace_meta["instruction"])
    phase_segments = [
        {
            "phase": PHASES[0],
            "start_step": 0,
            "end_step_exclusive": osc_start,
            "steps": osc_start,
            "action_source": spec["student_action_source"],
        },
        {
            "phase": PHASES[1],
            "start_step": osc_start,
            "end_step_exclusive": osc_end,
            "steps": osc_end - osc_start,
            "action_source": report.get(
                "actual_action_source", "direct_b_spline_closed_loop_native_osc_pose"
            ),
        },
        {
            "phase": PHASES[2],
            "start_step": osc_end,
            "end_step_exclusive": step_count,
            "steps": step_count - osc_end,
            "action_source": spec["student_action_source"],
        },
    ]

    if route_artifact:
        route_definition = {
            "interpolation": route_artifact.get("route_interpolation"),
            "selected_candidate": route_artifact.get("selected_candidate"),
        }
    else:
        selected_route = report.get("selected_route", {})
        route_definition = {
            "selected_route_summary": selected_route,
            # The source report stores each real OSC reference target. Retain the
            # executed target sequence because the historical source has no
            # separate control-point JSON artifact.
            "executed_reference_target_eef_positions_world_m": [
                step["target_eef_position_world_m"] for step in report_steps
            ],
        }

    adapter_path = trace_meta.get("lora_checkpoint_path")
    if adapter_path and Path(adapter_path).is_file():
        adapter_sha256 = _sha256(Path(adapter_path))
    else:
        adapter_sha256 = None
    if spec["round"] == 1 and adapter_sha256 is None:
        raise FileNotFoundError(f"attempt3_adapter_missing:{adapter_path}")

    safety_summary = {
        "source_report_path": _relative(report_path),
        "route_completed": bool(report.get("route_completed")),
        "physical_collision_observed": report.get("physical_collision_observed"),
        "failure_reason": report.get("failure_reason"),
        "execution_summary": report.get("execution_summary"),
        "source_safety_claim_field": report.get("physical_safety_claim"),
        "grasp_final_status": report.get("execution_summary", {}).get(
            "final_dual_finger_grasp_status"
        ),
    }

    episode_reward_sum = float(source["step_rewards"].sum())
    episode_done = bool(source["step_dones"][-1])
    del source
    return {
        "spec": spec,
        "source": None,
        "source_paths": {
            "npz": npz_path,
            "report": report_path,
            "result": result_path,
            "run_manifest": manifest_path,
            "route_artifact": route_path,
        },
        "report": report,
        "result": result,
        "run_manifest": run_manifest,
        "trace_metadata": trace_meta,
        "instruction": instruction,
        "step_count": step_count,
        "episode_reward_sum": episode_reward_sum,
        "episode_done": episode_done,
        "osc_start": osc_start,
        "osc_end": osc_end,
        "phase_segments": phase_segments,
        "route_definition": route_definition,
        "adapter_path": adapter_path,
        "adapter_sha256": adapter_sha256,
        "safety_summary": safety_summary,
    }


def _episode_frames(item: dict[str, Any]) -> list[dict[str, Any]]:
    source = _load_npz(item["source_paths"]["npz"])
    decision_rows = {
        int(step): row
        for row, step in enumerate(source["decision_step_indices"].tolist())
    }
    step_rows = {int(step): row for row, step in enumerate(source["step_indices"].tolist())}
    frames: list[dict[str, Any]] = []
    for step in range(item["step_count"]):
        observation = _observation(source, step, decision_rows, step_rows)
        action = np.asarray(source["executed_actions"][step], dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"{item['spec']['episode_id']}:bad_action_at_{step}")
        if step > 0:
            previous = _observation(
                source, step - 1, decision_rows, step_rows
            )
            for key in (
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ):
                if not np.array_equal(observation[key], source[f"post_{key}"][step - 1]):
                    raise ValueError(
                        f"{item['spec']['episode_id']}:observation_transition_alignment_failed_at_{step}_{key}"
                    )
        frames.append(
            {
                "image": _frame_image(observation, "agentview_image"),
                "wrist_image": _frame_image(observation, "robot0_eye_in_hand_image"),
                "state": _state_from_observation(observation),
                "actions": action,
                "task": item["instruction"],
            }
        )
    return frames


def _phase_windows(items: list[dict[str, Any]]) -> dict[str, list[int]]:
    phase_indices = {name: [] for name in PHASES}
    frame_offset = 0
    for item in items:
        valid_count = item["step_count"] - ACTION_HORIZON + 1
        bounds = (0, item["osc_start"], item["osc_end"], valid_count)
        if not (0 < bounds[1] < bounds[2] < bounds[3]):
            raise ValueError(f"{item['spec']['episode_id']}:invalid_horizon_phase_bounds")
        for phase, start, end in zip(PHASES, bounds[:-1], bounds[1:]):
            phase_indices[phase].extend(
                range(frame_offset + start, frame_offset + end)
            )
        item["global_frame_start"] = frame_offset
        item["valid_window_count"] = valid_count
        frame_offset += item["step_count"]
    return phase_indices


def _balanced_sampling(items: list[dict[str, Any]], phase_indices: dict[str, list[int]]) -> dict[str, Any]:
    target = max(len(indices) for indices in phase_indices.values())
    if target <= 0 or any(not indices for indices in phase_indices.values()):
        raise ValueError("cannot_balance_empty_phase")
    unique_windows = [
        index
        for item in items
        for index in range(
            item["global_frame_start"],
            item["global_frame_start"] + item["valid_window_count"],
        )
    ]
    schedule = list(unique_windows)
    for phase in PHASES:
        candidates = np.asarray(phase_indices[phase], dtype=np.int64)
        repeats = target - int(candidates.size)
        if repeats < 0:
            raise ValueError(f"target_below_unique_count_for_{phase}")
        if repeats:
            schedule.extend(np.resize(candidates, repeats).tolist())
    exposures = {phase: target for phase in PHASES}
    eval_per_phase = 16
    evaluation = {}
    for phase in PHASES:
        candidates = np.asarray(phase_indices[phase], dtype=np.int64)
        if candidates.size < eval_per_phase:
            raise ValueError(f"not_enough_{phase}_windows_for_fixed_eval")
        candidate_positions = np.linspace(
            0, candidates.size - 1, eval_per_phase, dtype=np.int64
        )
        selected = candidates[candidate_positions]
        if np.unique(selected).size != eval_per_phase:
            raise ValueError(f"fixed_{phase}_evaluation_indices_are_not_unique")
        evaluation[phase] = selected.tolist()
    return {
        "method": "attempt3_full_episode_phase_balance_with_all_unique_windows",
        "action_horizon": ACTION_HORIZON,
        "unique_window_count": len(unique_windows),
        "unique_window_starts_by_phase": {
            phase: len(phase_indices[phase]) for phase in PHASES
        },
        "target_exposures_per_phase": target,
        "balanced_exposures_by_phase": exposures,
        "training_schedule_window_count": len(schedule),
        "training_schedule_global_frame_indices": schedule,
        "evaluation_windows_per_phase": eval_per_phase,
        "evaluation_global_frame_indices_by_phase": evaluation,
        "window_boundary_policy": (
            "full horizon windows only; each window is anchored inside its source episode; "
            "no action window crosses an episode boundary"
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_dataset(output_root: Path, items: list[dict[str, Any]]) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing_to_overwrite_dataset:{output_root}")
    os.environ["HF_LEROBOT_HOME"] = str(output_root.parent.resolve())
    from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

    writer = LeRobotDatasetWriter()
    writer.create(
        repo_id=output_root.name,
        robot_type="franka_panda",
        fps=FPS,
        features=LEROBOT_FEATURES,
        image_writer_threads=0,
        image_writer_processes=0,
    )
    for item in items:
        frames = _episode_frames(item)
        if len(frames) != item["step_count"]:
            raise ValueError(f"{item['spec']['episode_id']}:frame_count_mismatch")
        writer.add_episode(frames)
        print(
            f"written episode={item['spec']['episode_id']} frames={len(frames)}",
            flush=True,
        )
        del frames
    writer.finalize()


def _verify_lerobot_data(output_root: Path, items: list[dict[str, Any]]) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from rlinf.data.storage.lerobot.compat import episode_boundaries

    dataset = LeRobotDataset(repo_id=output_root.name, root=output_root)
    starts, ends = episode_boundaries(dataset)
    expected_starts = []
    expected_ends = []
    offset = 0
    for item in items:
        expected_starts.append(offset)
        offset += item["step_count"]
        expected_ends.append(offset)
    if len(dataset) != offset or starts != expected_starts or ends != expected_ends:
        raise ValueError(
            f"episode_boundary_mismatch:len={len(dataset)} starts={starts} ends={ends}"
        )
    exact_steps = 0
    episode_checks = []
    for episode_index, item in enumerate(items):
        source = _load_npz(item["source_paths"]["npz"])
        decision_rows = {
            int(step): row
            for row, step in enumerate(source["decision_step_indices"].tolist())
        }
        step_rows = {int(step): row for row, step in enumerate(source["step_indices"].tolist())}
        image_checks = 0
        for local_step in range(item["step_count"]):
            observation = _observation(source, local_step, decision_rows, step_rows)
            actual = dataset[expected_starts[episode_index] + local_step]
            action = np.asarray(actual["actions"], dtype=np.float32)
            state = np.asarray(actual["state"], dtype=np.float32)
            expected_action = np.asarray(source["executed_actions"][local_step], dtype=np.float32)
            expected_state = _state_from_observation(observation)
            if action.shape != (7,) or not np.array_equal(action, expected_action):
                raise ValueError(
                    f"{item['spec']['episode_id']}:readback_action_mismatch_at_{local_step}"
                )
            if state.shape != (8,) or not np.array_equal(state, expected_state):
                raise ValueError(
                    f"{item['spec']['episode_id']}:readback_state_mismatch_at_{local_step}"
                )
            if actual["task"] != item["instruction"]:
                raise ValueError(
                    f"{item['spec']['episode_id']}:readback_task_mismatch_at_{local_step}"
                )
            expected_main = _frame_image(observation, "agentview_image")
            expected_wrist = _frame_image(observation, "robot0_eye_in_hand_image")
            actual_main = _as_hwc(actual["image"])
            actual_wrist = _as_hwc(actual["wrist_image"])
            if (
                actual_main.shape != (256, 256, 3)
                or actual_wrist.shape != (256, 256, 3)
                or not np.array_equal(actual_main, expected_main)
                or not np.array_equal(actual_wrist, expected_wrist)
            ):
                raise ValueError(
                    f"{item['spec']['episode_id']}:readback_image_mismatch_at_{local_step}"
                )
            image_checks += 2
            exact_steps += 1
        episode_checks.append(
            {
                "episode_index": episode_index,
                "episode_id": item["spec"]["episode_id"],
                "start_frame": expected_starts[episode_index],
                "end_frame_exclusive": expected_ends[episode_index],
                "frames_checked_exactly": item["step_count"],
                "images_checked_exactly": image_checks,
                "actions_states_and_task_exact": True,
            }
        )
        del source
    return {
        "status": "passed",
        "lerobot_version_expected": "0.3.3",
        "lerobot_metadata_version": str(dataset.meta.info.get("codebase_version")),
        "fps": FPS,
        "frame_count": len(dataset),
        "episode_count": len(items),
        "episode_starts": starts,
        "episode_ends_exclusive": ends,
        "exact_full_frame_readback_count": exact_steps,
        "episode_checks": episode_checks,
    }


def _verify_official_training_loader(
    output_root: Path, items: list[dict[str, Any]]
) -> dict[str, Any]:
    os.environ.setdefault("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    import logging

    from hydra import compose, initialize_config_dir
    import torch
    from research.gs_opd.data.run_single_episode_sft import SingleEpisodeSftWorker

    logging.getLogger().setLevel(logging.WARNING)

    with initialize_config_dir(
        config_dir=str(ROOT / "examples/sft/config"), version_base="1.2"
    ):
        cfg = compose(config_name="gs_opd_full_episode_pi05_lora")
    cfg.actor.eval_batch_size = 1
    cfg.data.train_data_paths = str(output_root)
    cfg.data.val_data_paths = str(output_root)
    from omegaconf import OmegaConf

    OmegaConf.update(
        cfg,
        "data.phase_manifest_path",
        str(output_root / "pi2_cumulative_dataset_manifest.json"),
        force_add=True,
    )

    def build_worker() -> SingleEpisodeSftWorker:
        worker = object.__new__(SingleEpisodeSftWorker)
        worker.cfg = cfg
        worker._world_size = 1
        worker._rank = 0
        return worker

    train_worker = build_worker()
    train_loader, _ = train_worker.build_dataloader(str(output_root), eval_dataset=False)
    sampling = json.loads(
        (output_root / "pi2_stage_balanced_sampling.json").read_text(encoding="utf-8")
    )
    if train_worker._selected_phase_window_indices != sampling[
        "training_schedule_global_frame_indices"
    ]:
        raise ValueError("existing_sft_worker_schedule_differs_from_saved_sampler_manifest")
    if train_worker._balanced_phase_window_exposure != sampling[
        "balanced_exposures_by_phase"
    ]:
        raise ValueError("existing_sft_worker_phase_exposure_differs_from_manifest")
    train_batch = next(iter(train_loader))
    if not isinstance(train_batch, (tuple, list)) or len(train_batch) != 2:
        raise TypeError(f"unexpected_training_batch_type:{type(train_batch).__name__}")
    train_observation, train_actions = train_batch
    if tuple(train_actions.shape) != (2, ACTION_HORIZON, 32):
        raise ValueError(f"training_loader_action_shape_mismatch:{tuple(train_actions.shape)}")
    if not bool(torch.isfinite(train_actions).all()):
        raise ValueError("training_loader_actions_contain_nonfinite_values")

    eval_worker = build_worker()
    eval_loader, _ = eval_worker.build_dataloader(str(output_root), eval_dataset=True)
    eval_indices = [
        index
        for phase in PHASES
        for index in eval_worker._fixed_eval_phase_indices[phase]
    ]
    eval_phase_sequence = list(eval_worker._fixed_eval_phase_sequence)
    expected_episode_ids: list[str] = []
    episode_frame_boundaries = np.cumsum([0, *[item["step_count"] for item in items]])
    for index in eval_indices:
        episode_index = int(np.searchsorted(episode_frame_boundaries[1:], index, side="right"))
        expected_episode_ids.append(items[episode_index]["spec"]["episode_id"])
    batch_shapes = []
    for batch_index, batch in enumerate(eval_loader):
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise TypeError(f"unexpected_eval_batch_type:{type(batch).__name__}")
        observation, actions = batch
        if tuple(actions.shape) != (1, ACTION_HORIZON, 32):
            raise ValueError(f"evaluation_loader_action_shape_mismatch:{tuple(actions.shape)}")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("evaluation_loader_actions_contain_nonfinite_values")
        batch_shapes.append(
            {
                "batch_index": batch_index,
                "batch_type": [type(observation).__name__, type(actions).__name__],
                "observation_state_shape": list(observation.state.shape),
                "observation_image_shapes": {
                    key: list(value.shape) for key, value in observation.images.items()
                },
                "actions_shape": list(actions.shape),
                "actions_dtype": str(actions.dtype),
            }
        )
        if len(batch_shapes) == len(eval_indices):
            break
    if len(batch_shapes) != len(eval_indices):
        raise ValueError(
            f"evaluation_loader_returned_{len(batch_shapes)}_of_{len(eval_indices)}_batches"
        )
    return {
        "status": "passed",
        "loader": "SingleEpisodeSftWorker + RLinf official OpenPI / pi05_libero",
        "training_mode_batch_read": {
            "status": "passed",
            "batch_size": 2,
            "actions_shape": list(train_actions.shape),
            "observation_state_shape": list(train_observation.state.shape),
            "actions_finite": True,
        },
        "evaluation_mode": {
            "status": "passed",
            "windows_per_phase": int(cfg.data.eval_windows_per_phase),
        },
        "world_size": 1,
        "num_workers": 0,
        "raw_frame_count": sum(item["step_count"] for item in items),
        "episode_count": len(items),
        "balanced_training_schedule_window_count": train_worker._training_schedule_window_count,
        "unique_training_window_count": train_worker._unique_training_window_count,
        "balanced_window_exposure_by_phase": train_worker._balanced_phase_window_exposure,
        "evaluation_global_frame_indices": eval_indices,
        "evaluation_phase_sequence": eval_phase_sequence,
        "sampled_batch_count": len(batch_shapes),
        "expected_episode_sample_order": expected_episode_ids,
        "batch_shape_checks": batch_shapes,
    }


def prepare(output_root: Path, *, verify_existing: bool = False) -> dict[str, Any]:
    if output_root.exists() and not verify_existing:
        raise FileExistsError(f"refusing_to_overwrite_dataset:{output_root}")
    items = [_load_and_validate_source(spec) for spec in SOURCES]
    if len({item["instruction"] for item in items}) != 1:
        raise ValueError("source_episodes_do_not_share_the_same_task_instruction")
    phase_indices = _phase_windows(items)
    sampling = _balanced_sampling(items, phase_indices)
    if not output_root.exists():
        print(
            "validated sources; writing 3 episodes / "
            f"{sum(item['step_count'] for item in items)} frames",
            flush=True,
        )
        _write_dataset(output_root, items)
    else:
        print(f"verifying existing cumulative dataset without rewriting: {output_root}", flush=True)

    manifest_episodes = []
    for episode_index, item in enumerate(items):
        source_paths = item["source_paths"]
        npz_path: Path = source_paths["npz"]
        checkpoint_path = item["trace_metadata"].get("checkpoint_path")
        adapter_path = item["adapter_path"]
        run_model = item["run_manifest"].get("model", {})
        manifest_episodes.append(
            {
                "episode_index": episode_index,
                "episode_id": item["spec"]["episode_id"],
                "round": item["spec"]["round"],
                "frame_count": item["step_count"],
                "source": {
                    "npz_path": _relative(npz_path),
                    "npz_size_bytes": npz_path.stat().st_size,
                    "npz_sha256": _sha256(npz_path),
                    "report_path": _relative(source_paths["report"]),
                    "report_sha256": _sha256(source_paths["report"]),
                    "result_path": _relative(source_paths["result"])
                    if source_paths["result"]
                    else None,
                    "run_manifest_path": _relative(source_paths["run_manifest"]),
                    "route_artifact_path": _relative(source_paths["route_artifact"])
                    if source_paths["route_artifact"]
                    else None,
                    "route_artifact_sha256": _sha256(source_paths["route_artifact"])
                    if source_paths["route_artifact"]
                    else None,
                },
                "policy": {
                    "version": item["spec"]["policy_version"],
                    "base_checkpoint_path": checkpoint_path,
                    "checkpoint_file": item["trace_metadata"].get("checkpoint_file"),
                    "adapter_path": adapter_path,
                    "adapter_sha256": item["adapter_sha256"],
                    "run_manifest_model": run_model,
                },
                "route": {
                    "candidate_id": item["spec"]["expected_candidate_id"],
                    "direction": "left",
                    "actual_action_source": item["report"].get(
                        "actual_action_source",
                        "direct_b_spline_closed_loop_native_osc_pose",
                    ),
                    "reference_speed_m_s": item["report"].get(
                        "reference_speed_m_s",
                        item["trace_metadata"]["direct_bspline_osc_execution"].get(
                            "reference_speed_m_s"
                        ),
                    ),
                    "definition": item["route_definition"],
                },
                "task_instruction": item["instruction"],
                "episode_success": {
                    "reward_sum": item["episode_reward_sum"],
                    "done": item["episode_done"],
                },
                "phase_segments": item["phase_segments"],
                "phase_boundaries_local_step": [item["osc_start"], item["osc_end"]],
                "horizon_10_unique_windows_by_phase": {
                    phase: int(
                        max(
                            0,
                            min(item["valid_window_count"], end)
                            - start,
                        )
                    )
                    for phase, start, end in zip(
                        PHASES,
                        (0, item["osc_start"], item["osc_end"]),
                        (item["osc_start"], item["osc_end"], item["valid_window_count"]),
                    )
                },
                "safety_source_evidence": item["safety_summary"],
            }
        )
    manifest = {
        "schema_version": "gsopd.pi2_cumulative_lerobot.v1",
        "dataset_path": str(output_root.resolve()),
        "format": "LeRobot v2.1",
        "writer": "RLinf LeRobotDatasetWriter / LeRobot 0.3.3",
        "robot_type": "franka_panda",
        "fps": FPS,
        "image_preprocessing": "same 180-degree rotation as pi05_rollout.py",
        "state_encoding": "EEF xyz + quaternion axis-angle + two gripper qpos (float32, 8)",
        "action_encoding": "raw executed 7D environment action (float32), unchanged",
        "episode_count": len(items),
        "frame_count": sum(item["step_count"] for item in items),
        "task_instruction": items[0]["instruction"],
        "episode_sources": manifest_episodes,
        "stage_balanced_sampling_report": "pi2_stage_balanced_sampling.json",
        "dataset_validation_report": "pi2_cumulative_dataset_validation.json",
        "no_training_or_rollout": True,
    }
    _write_json(output_root / "pi2_cumulative_dataset_manifest.json", manifest)
    _write_json(output_root / "pi2_stage_balanced_sampling.json", sampling)
    dataset_validation = _verify_lerobot_data(output_root, items)
    loader_validation = _verify_official_training_loader(output_root, items)
    validation = {
        "status": "passed",
        "source_episode_validation": "passed",
        "lerobot_full_readback": dataset_validation,
        "official_training_loader_smoke": loader_validation,
        "no_training_or_rollout": True,
    }
    _write_json(output_root / "pi2_cumulative_dataset_validation.json", validation)
    return {
        "manifest": manifest,
        "sampling": {
            key: value
            for key, value in sampling.items()
            if key != "training_schedule_global_frame_indices"
        },
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="verify a dataset already created at --output-root without rewriting its LeRobot files",
    )
    args = parser.parse_args()
    report = prepare(args.output_root, verify_existing=args.verify_existing)
    print(
        json.dumps(
            {
                "dataset_path": report["manifest"]["dataset_path"],
                "episode_count": report["manifest"]["episode_count"],
                "frame_count": report["manifest"]["frame_count"],
                "unique_horizon_windows": report["sampling"]["unique_window_count"],
                "balanced_schedule_windows": report["sampling"][
                    "training_schedule_window_count"
                ],
                "validation_status": report["validation"]["status"],
                "official_loader_samples": report["validation"][
                    "official_training_loader_smoke"
                ]["sampled_batch_count"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
