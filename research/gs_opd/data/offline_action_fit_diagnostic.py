#!/usr/bin/env python3
"""Offline paired action and Flow Matching diagnostics for the GS-OPD demo.

This reads the existing full-episode LeRobot dataset and saved NPZ traces. It
does not create an environment, take actions, train, or modify checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import Subset


ROOT = Path("/workspace/zy_workspace/RLinf")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE_CHECKPOINT = Path("/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft")
ADAPTER = ROOT / "logs/gs_opd/sft/pi05_libero_full_trajectory_lora_20261019_attempt1/lora_adapter.pt"
NORM_STATS = BASE_CHECKPOINT / "physical-intelligence/libero/norm_stats.json"
DEMO_NPZ = ROOT / "logs/gs_opd/live_left_bspline_osc_adaptive_20261019_attempt1/direct_bspline_osc_trajectory.npz"
STUDENT_NPZ = ROOT / "logs/gs_opd/eval/pi05_libero_full_trajectory_lora_20261019_attempt1/student_trajectory.npz"
TRAIN_REPORT = ROOT / "logs/gs_opd/sft/pi05_libero_full_trajectory_lora_20261019_attempt1/report.json"
ROUTE_REPORT = ROOT / "logs/gs_opd/live_left_bspline_osc_adaptive_20261019_attempt1/live_left_bspline_osc_report.json"
DATASET = ROOT / "logs/gs_opd/mink_panda_physical_v3/live_left_bspline_osc_adaptive_20261019_attempt1/lerobot_full_episode_v21"
DEFAULT_OUT = ROOT / "logs/gs_opd/diagnostics/offline_action_fit_20261020_attempt1"
PHASES = ("student_grasp", "bspline_osc_carry", "student_place")
PHASE_BOUNDS = ((0, 55), (55, 804), (804, 836))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def _selected_indices() -> tuple[list[int], list[str]]:
    indices: list[int] = []
    phases: list[str] = []
    for name, (start, end) in zip(PHASES, PHASE_BOUNDS):
        candidates = np.arange(start, end, dtype=np.int64)
        chosen = np.linspace(candidates[0], candidates[-1], 16, dtype=np.int64)
        indices.extend(int(value) for value in chosen)
        phases.extend([name] * len(chosen))
    return indices, phases


def _build_fixed_loader(indices: list[int]):
    from rlinf.data.datasets.openpi.official_sft_data_loader import (
        build_official_openpi_sft_dataloader,
    )

    with initialize_config_dir(
        config_dir=str(ROOT / "examples/sft/config"), version_base="1.2"
    ):
        cfg = compose(config_name="gs_opd_full_episode_pi05_lora")
    cfg.actor.eval_batch_size = 1
    cfg.data.train_data_paths = str(DATASET)
    cfg.data.val_data_paths = str(DATASET)
    loader, _ = build_official_openpi_sft_dataloader(
        cfg, world_size=1, rank=0, data_paths=str(DATASET), eval_dataset=True
    )
    torch_loader = loader._data_loader.torch_loader
    transformed = torch_loader.dataset
    raw = transformed
    while hasattr(raw, "_dataset"):
        raw = raw._dataset
    if len(raw) != 845:
        raise ValueError(f"expected_845_frame_episode_found_{len(raw)}")
    parent = transformed
    while hasattr(parent, "_dataset") and parent._dataset is not raw:
        parent = parent._dataset
    if not hasattr(parent, "_dataset") or parent._dataset is not raw:
        raise RuntimeError("could_not_locate_raw_lerobot_dataset_leaf")
    parent._dataset = Subset(raw, indices)
    sampler = torch_loader.sampler
    if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
        if sampler.num_replicas != 1:
            raise ValueError("diagnostic_loader_must_be_single_process")
        sampler.num_samples = len(indices)
        sampler.total_size = len(indices)
    batches = []
    for i, batch in enumerate(loader):
        batches.append(batch)
        if len(batches) == len(indices):
            break
    if len(batches) != len(indices):
        raise RuntimeError(f"expected_48_batches_found_{len(batches)}")
    return cfg, batches


def _build_model(*, with_adapter: bool, device: str):
    from rlinf.models import get_model
    from research.gs_opd.data.lora_checkpoint import load_lora_adapter

    with initialize_config_dir(
        config_dir=str(ROOT / "evaluations/libero"), version_base=None
    ):
        cfg = compose(config_name="libero_spatial_openpi_pi05_smoke")
    model_cfg = cfg.rollout.model
    model_cfg.model_path = str(BASE_CHECKPOINT)
    OmegaConf.update(model_cfg, "openpi.task", "eval", force_add=True)
    OmegaConf.update(model_cfg, "openpi.config_name", "pi05_libero", force_add=True)
    OmegaConf.update(model_cfg, "openpi.torch_compile", False, force_add=True)
    OmegaConf.update(model_cfg, "openpi.action_chunk", 10, force_add=True)
    OmegaConf.update(model_cfg, "openpi.action_env_dim", 7, force_add=True)
    OmegaConf.update(model_cfg, "openpi.model_action_dim", 32, force_add=True)
    OmegaConf.update(
        model_cfg,
        "openpi_data.norm_stats_path",
        str(NORM_STATS),
        force_add=True,
    )
    if with_adapter:
        OmegaConf.update(
            model_cfg,
            "openpi.action_expert_variant",
            "gemma_300m_lora",
            force_add=True,
        )
        OmegaConf.update(model_cfg, "is_lora", False, force_add=True)
        OmegaConf.update(model_cfg, "lora_rank", 32, force_add=True)
    else:
        OmegaConf.update(
            model_cfg,
            "openpi.action_expert_variant",
            "gemma_300m",
            force_add=True,
        )
    model = get_model(model_cfg)
    adapter_metadata = None
    if with_adapter:
        adapter_metadata = load_lora_adapter(model, ADAPTER)
    model = model.to(device).eval()
    return model, adapter_metadata


def _tensor_to_device(value, device):
    if isinstance(value, dict):
        return {key: _tensor_to_device(item, device) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, device=device)
    return value


def _action_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    error = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    per_sample_l2 = np.linalg.norm(error.reshape(error.shape[0], -1), axis=-1)
    return {
        "sample_count": int(error.shape[0]),
        "action_horizon": int(error.shape[1]),
        "raw_action_dim": int(error.shape[2]),
        "mae_all_7d": float(np.abs(error).mean()),
        "rmse_all_7d": float(np.sqrt(np.square(error).mean())),
        "mean_chunk_l2": float(per_sample_l2.mean()),
        "mae_by_dim": np.abs(error).mean(axis=(0, 1)).tolist(),
        "rmse_by_dim": np.sqrt(np.square(error).mean(axis=(0, 1))).tolist(),
        "mae_translation_0_2": float(np.abs(error[..., :3]).mean()),
        "mae_rotation_3_5": float(np.abs(error[..., 3:6]).mean()),
        "mae_gripper_6": float(np.abs(error[..., 6]).mean()),
    }


def _flow_stats(losses: list[np.ndarray]) -> dict:
    per_element = np.concatenate(losses, axis=0).astype(np.float64)
    real = per_element[..., :7]
    padding = per_element[..., 7:]
    full_mean = float(per_element.mean())
    real_mean = float(real.mean())
    padding_mean = float(padding.mean())
    return {
        "examples": int(per_element.shape[0]),
        "loss_tensor_shape_per_example": list(per_element.shape[1:]),
        "real_action_dims_0_6": {
            "mean_squared_flow_loss_per_element": real_mean,
            "dimensions": 7,
            "weighted_share_of_default_32d_mean": float(7 * real_mean / (32 * full_mean)),
        },
        "zero_padding_dims_7_31": {
            "mean_squared_flow_loss_per_element": padding_mean,
            "dimensions": 25,
            "target_max_abs": None,
            "weighted_share_of_default_32d_mean": float(25 * padding_mean / (32 * full_mean)),
        },
        "default_sft_mean_over_all_10x32_elements": full_mean,
        "reconstructed_weighted_mean": float((7 * real_mean + 25 * padding_mean) / 32),
    }


def _evaluate_model(
    batches,
    indices: list[int],
    phase_names: list[str],
    demo_actions: np.ndarray,
    *,
    with_adapter: bool,
    device: str,
    noise: np.ndarray,
    times: np.ndarray,
) -> tuple[dict, dict]:
    model, adapter_metadata = _build_model(with_adapter=with_adapter, device=device)
    if model.action_horizon != 10 or model.action_dim != 32 or model.action_env_dim != 7:
        raise RuntimeError(
            "model_action_contract_mismatch: "
            f"horizon={model.action_horizon}, model_dim={model.action_dim}, "
            f"env_dim={model.action_env_dim}"
        )
    if model.action_chunk != 10 or model.num_steps != 3:
        raise RuntimeError(
            f"expected_action_chunk_10_and_3_flow_steps_got_{model.action_chunk}_{model.num_steps}"
        )

    prediction_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    flow_by_phase = {name: [] for name in PHASES}
    padding_target_by_phase = {name: 0.0 for name in PHASES}
    actions_by_phase = {name: {"pred": [], "target": []} for name in PHASES}
    observed_target_max_padding = 0.0
    alignment_max_abs = 0.0
    example_shapes = None
    default_loss_reduction_matches = True
    for position, ((observation, actions), step_index, phase) in enumerate(
        zip(batches, indices, phase_names)
    ):
        observation = model._observation_to_device(observation)
        actions = model._actions_to_device(actions)
        noise_tensor = torch.as_tensor(noise[position : position + 1], device=device)
        time_tensor = torch.as_tensor(times[position : position + 1], device=device)
        with torch.no_grad():
            model_actions = model.sample_actions(
                observation,
                num_steps=3,
                noise=noise_tensor,
            )
            predicted_raw = model.decode_actions(model_actions, observation.state)
            target_raw = model.decode_actions(actions, observation.state)
            element_loss = model.compute_loss(
                observation,
                actions,
                train=True,
                noise=noise_tensor,
                time=time_tensor,
            )
            default_reduced = model._reduce_sft_loss(
                element_loss, use_action_chunk_loss=False
            )

        target_raw_np = target_raw.detach().float().cpu().numpy()[0]
        predicted_raw_np = predicted_raw.detach().float().cpu().numpy()[0]
        target_npz = demo_actions[step_index : step_index + 10]
        if target_raw_np.shape != (10, 7) or predicted_raw_np.shape != (10, 7):
            raise RuntimeError("decoded_environment_actions_not_10_by_7")
        alignment_max_abs = max(
            alignment_max_abs,
            float(np.max(np.abs(target_raw_np - target_npz))),
        )
        observed_target_max_padding = max(
            observed_target_max_padding,
            float(actions[..., 7:].abs().max().detach().cpu().item()),
        )
        padding_target_by_phase[phase] = max(
            padding_target_by_phase[phase],
            float(actions[..., 7:].abs().max().detach().cpu().item()),
        )
        loss_np = element_loss.detach().float().cpu().numpy()
        flow_by_phase[phase].append(loss_np)
        default_from_elements = float(loss_np.mean())
        if abs(float(default_reduced.detach().float().cpu().item()) - default_from_elements) > 1e-6:
            default_loss_reduction_matches = False
        prediction_chunks.append(predicted_raw_np)
        target_chunks.append(target_npz.astype(np.float32))
        actions_by_phase[phase]["pred"].append(predicted_raw_np)
        actions_by_phase[phase]["target"].append(target_npz.astype(np.float32))
        if example_shapes is None:
            example_shapes = {
                "observation_state": list(observation.state.shape),
                "observation_images": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in observation.images.items()
                },
                "normalized_action_target": list(actions.shape),
                "predicted_raw_action_chunk": list(predicted_raw.shape),
                "flow_loss_per_element": list(element_loss.shape),
            }

    all_pred = np.stack(prediction_chunks)
    all_target = np.stack(target_chunks)
    phase_report = {}
    phase_flow = {}
    for name in PHASES:
        pred = np.stack(actions_by_phase[name]["pred"])
        target = np.stack(actions_by_phase[name]["target"])
        phase_report[name] = {
            "start_index_inclusive": PHASE_BOUNDS[PHASES.index(name)][0],
            "end_index_exclusive": PHASE_BOUNDS[PHASES.index(name)][1],
            "window_count": len(pred),
            "full_10_step_chunk": _action_metrics(pred, target),
            "first_action_at_observation": _action_metrics(pred[:, :1, :], target[:, :1, :]),
            "mae_reduction_vs_base_percent": None,
        }
        phase_flow[name] = _flow_stats(flow_by_phase[name])
        phase_flow[name]["zero_padding_dims_7_31"]["target_max_abs"] = padding_target_by_phase[name]

    report = {
        "model_variant": "rank32_action_expert_lora" if with_adapter else "original_pi05_base",
        "adapter_metadata": adapter_metadata,
        "inference": {
            "same_observation_order": True,
            "fixed_initial_noise_seed": None,
            "fixed_noise_shape": [1, 10, 32],
            "fixed_flow_matching_time_seed": None,
            "flow_solver_steps": int(model.num_steps),
            "action_horizon": int(model.action_horizon),
            "model_action_dim": int(model.action_dim),
            "environment_action_dim": int(model.action_env_dim),
            "action_chunk": int(model.action_chunk),
            "comparison": "raw decoded 10x7 OSC environment action chunk against the 10 saved actions beginning at each fixed observation",
        },
        "sample_shapes": example_shapes,
        "target_alignment": {
            "dataset_normalized_target_decoded_vs_original_npz_max_abs": alignment_max_abs,
            "dataset_padding_target_max_abs_dims_7_31": observed_target_max_padding,
            "fixed_eval_indices": {name: [i for i, p in zip(indices, phase_names) if p == name] for name in PHASES},
        },
        "sft_flow_matching": {
            "default_sft_forward_use_action_chunk_loss": False,
            "default_reduction_verified_as_full_mean": default_loss_reduction_matches,
            "default_reduction_formula": "mean over batch x action_horizon(10) x model_action_dim(32)",
            "by_phase": phase_flow,
        },
        "actions_vs_demonstration": {
            "raw_units": "the original seven environment action coordinates after the Checkpoint's existing OpenPI output unnormalization; no new scaling",
            "by_phase": phase_report,
            "all_samples": _action_metrics(all_pred, all_target),
        },
    }
    arrays = {
        "observation_step_indices": np.asarray(indices, dtype=np.int64),
        "phase_names": np.asarray(phase_names, dtype="U32"),
        "target_action_chunks_raw": all_target.astype(np.float32),
        "predicted_action_chunks_raw": all_pred.astype(np.float32),
        "fixed_noise": noise.astype(np.float32),
        "fixed_time": times.astype(np.float32),
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report, arrays


def _paths_and_plot(out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    demo = np.load(DEMO_NPZ, allow_pickle=True)
    student = np.load(STUDENT_NPZ, allow_pickle=True)
    route = json.loads(ROUTE_REPORT.read_text(encoding="utf-8"))
    demo_eef = np.asarray(demo["post_robot0_eef_pos"], dtype=np.float64)
    student_eef = np.asarray(student["post_robot0_eef_pos"], dtype=np.float64)
    demo_object = np.asarray(
        [row["actual_sphere_center_world_m"] for row in route["steps"]],
        dtype=np.float64,
    )
    student_object = []
    for state_json in student["post_physical_state_json"]:
        state = json.loads(str(state_json))
        body = next(
            item
            for item in state["object_states"].values()
            if item.get("body_name") == "akita_black_bowl_1_main"
        )
        student_object.append(body["root_pos"])
    student_object = np.asarray(student_object, dtype=np.float64)

    fig = plt.figure(figsize=(14, 6.5), constrained_layout=True)
    ax_eef_3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_obj_3d = fig.add_subplot(1, 2, 2, projection="3d")
    for ax, title, first, second, first_label, second_label in (
        (ax_eef_3d, "EEF paths", demo_eef, student_eef, "845-step successful demo", "Student closed loop"),
        (ax_obj_3d, "Carried bowl paths", demo_object, student_object, "B-Spline OSC carry (749 steps)", "Student bowl body (full episode)"),
    ):
        ax.plot(first[:, 0], first[:, 1], first[:, 2], color="#2166ac", linewidth=1.6, label=first_label)
        ax.plot(second[:, 0], second[:, 1], second[:, 2], color="#b2182b", linewidth=1.8, label=second_label)
        ax.scatter(*first[0], color="#2166ac", marker="o", s=28)
        ax.scatter(*first[-1], color="#2166ac", marker="x", s=42)
        ax.scatter(*second[0], color="#b2182b", marker="o", s=28)
        ax.scatter(*second[-1], color="#b2182b", marker="x", s=42)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Saved task paths only; no obstacle-contact safety classification is inferred",
        fontsize=11,
    )
    plot_path = out_dir / "spatial_paths_comparison.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    def path_summary(points: np.ndarray) -> dict:
        return {
            "point_count": int(len(points)),
            "start_xyz_m": points[0].tolist(),
            "end_xyz_m": points[-1].tolist(),
            "path_length_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
        }

    return {
        "plot": str(plot_path),
        "comparison": {
            "eef_successful_demo": path_summary(demo_eef),
            "eef_pure_student": path_summary(student_eef),
            "demo_bowl_center": {
                **path_summary(demo_object),
                "source": "live_left_bspline_osc_report.actual_sphere_center_world_m",
                "coverage": "the 749-step B-Spline OSC carry segment only",
            },
            "student_bowl_body": {
                **path_summary(student_object),
                "source": "saved pure Student MuJoCo physical state root_pos",
                "coverage": "full saved Student episode",
            },
        },
        "safety_interpretation": "not_assessed; plot contains no obstacle labels and does not reinterpret contact or semantic bindings",
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    logging.disable(logging.CRITICAL)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the saved Pi0.5 Checkpoint inference")

    indices, phase_names = _selected_indices()
    train_report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    demo_npz = np.load(DEMO_NPZ, allow_pickle=True)
    demo_actions = np.asarray(demo_npz["executed_actions"], dtype=np.float32)
    if demo_actions.shape != (845, 7):
        raise ValueError(f"expected_demo_actions_845x7_found_{demo_actions.shape}")
    if indices != [i for phase in PHASES for i in train_report["fixed_eval_indices_by_phase"][phase]]:
        raise ValueError("selected_indices_do_not_match_training_fixed_eval_set")
    cfg, batches = _build_fixed_loader(indices)
    if len(batches) != 48:
        raise RuntimeError("fixed_eval_loader_did_not_yield_48_examples")

    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal((len(indices), 10, 32), dtype=np.float32)
    # Inverse CDF for Beta(1.5, 1), matching the model's Flow Matching time law.
    uniform = rng.random(len(indices), dtype=np.float32)
    times = np.power(uniform, 2.0 / 3.0).astype(np.float32) * 0.999 + 0.001

    base_report, base_arrays = _evaluate_model(
        batches,
        indices,
        phase_names,
        demo_actions,
        with_adapter=False,
        device=args.device,
        noise=noise,
        times=times,
    )
    lora_report, lora_arrays = _evaluate_model(
        batches,
        indices,
        phase_names,
        demo_actions,
        with_adapter=True,
        device=args.device,
        noise=noise,
        times=times,
    )

    for phase in PHASES:
        base_mae = base_report["actions_vs_demonstration"]["by_phase"][phase]["first_action_at_observation"]["mae_all_7d"]
        lora_mae = lora_report["actions_vs_demonstration"]["by_phase"][phase]["first_action_at_observation"]["mae_all_7d"]
        lora_report["actions_vs_demonstration"]["by_phase"][phase]["mae_reduction_vs_base_percent"] = (
            100.0 * (base_mae - lora_mae) / base_mae if base_mae else None
        )

    if not np.array_equal(base_arrays["fixed_noise"], lora_arrays["fixed_noise"]):
        raise RuntimeError("paired_inference_noise_changed_between_models")
    for model_report in (base_report, lora_report):
        model_report["inference"]["fixed_initial_noise_seed"] = int(args.seed)
        model_report["inference"]["fixed_flow_matching_time_seed"] = int(args.seed)
        model_report["sft_flow_matching"]["target_max_abs_padding_action"] = max(
            value["zero_padding_dims_7_31"]["target_max_abs"]
            for value in model_report["sft_flow_matching"]["by_phase"].values()
        )

    paths = _paths_and_plot(out_dir)
    np.savez_compressed(out_dir / "paired_action_predictions.npz", **{
        "observation_step_indices": lora_arrays["observation_step_indices"],
        "phase_names": lora_arrays["phase_names"],
        "target_action_chunks_raw": lora_arrays["target_action_chunks_raw"],
        "base_predicted_action_chunks_raw": base_arrays["predicted_action_chunks_raw"],
        "lora_predicted_action_chunks_raw": lora_arrays["predicted_action_chunks_raw"],
        "fixed_noise": noise,
        "fixed_time": times,
    })

    original_train_fit = {
        "fixed_eval_before": train_report["fixed_eval_before"],
        "fixed_eval_after": train_report["fixed_eval_after"],
        "evaluation_seed": train_report["fixed_eval_seed"],
    }
    paired_comparison = {}
    for phase in PHASES:
        base_phase = base_report["actions_vs_demonstration"]["by_phase"][phase]
        lora_phase = lora_report["actions_vs_demonstration"]["by_phase"][phase]
        for metric in ("first_action_at_observation", "full_10_step_chunk"):
            base_mae = base_phase[metric]["mae_all_7d"]
            lora_mae = lora_phase[metric]["mae_all_7d"]
            paired_comparison.setdefault(phase, {})[metric] = {
                "base_mae": base_mae,
                "lora_mae": lora_mae,
                "relative_mae_reduction_percent": (
                    100.0 * (base_mae - lora_mae) / base_mae if base_mae else None
                ),
                "closer_to_recorded_demonstration": lora_mae < base_mae,
            }
    report = {
        "title": "GS-OPD offline paired action and Flow Matching diagnostic",
        "created_utc": "2026-10-20",
        "scope": "offline only; no MuJoCo, Student rollout, training, checkpoint modification, collision verdict, or semantic binding change",
        "source_artifacts": {
            "complete_845_step_demo_npz": str(DEMO_NPZ),
            "original_checkpoint": str(BASE_CHECKPOINT),
            "rank32_lora_adapter": str(ADAPTER),
            "normalization_stats_path": str(NORM_STATS),
            "normalization_stats_sha256": hashlib.sha256(NORM_STATS.read_bytes()).hexdigest(),
            "saved_pure_student_trace": str(STUDENT_NPZ),
            "official_openpi_lerobot_dataset": str(DATASET),
        },
        "demonstration": {
            "actions_shape": list(demo_actions.shape),
            "valid_full_horizon_windows": 836,
            "phases_by_window_start": {
                name: {"range_start_inclusive": lo, "range_end_exclusive": hi, "count": hi - lo}
                for name, (lo, hi) in zip(PHASES, PHASE_BOUNDS)
            },
            "phase_assignment": "grouped by the action-window start index, matching the existing training report",
            "windows_crossing_phase_boundaries": {
                "grasp_to_carry_start_indices": list(range(46, 55)),
                "carry_to_place_start_indices": list(range(795, 804)),
                "first_action_metrics_remain_in_anchor_phase": True,
                "full_chunk_and_flow_metrics_include_these_cross_phase_windows": True,
            },
            "fixed_comparison_windows_per_phase": 16,
            "fixed_noise_and_flow_time_seed": args.seed,
        },
        "original_training_report_fit": original_train_fit,
        "action_configuration_consistency": {
            "training": {
                "openpi_config_name": "pi05_libero",
                "model_action_horizon": 10,
                "sft_dataset_action_horizon": 10,
                "action_chunk": 10,
                "environment_action_dim": 7,
                "model_action_dim": 32,
                "source": "examples/sft/config/gs_opd_full_episode_pi05_lora.yaml plus pi05_libero DataConfig",
            },
            "closed_loop_inference": {
                "model_action_horizon": 10,
                "action_chunk": 10,
                "environment_action_dim": 7,
                "model_action_dim": 32,
                "flow_solver_steps": 3,
                "source": "pi05_rollout.py explicitly overrides openpi.action_chunk=10; model horizon comes from the pi05_libero OpenPI DataConfig",
            },
            "horizon_and_chunk_match": True,
        },
        "sft_forward_source_audit": {
            "implementation": "rlinf/models/embodiment/openpi/pi0.py::Pi0.sft_forward",
            "default_use_action_chunk_loss": False,
            "SFT_runner_call": "rlinf/workers/sft/fsdp_vla_sft_worker.py passes data=batch and no use_action_chunk_loss override",
            "reduction": "_reduce_sft_loss returns per_element_loss.mean() when use_action_chunk_loss=False",
            "mean_shape": "[B, action_horizon=10, action_dim=32]",
            "result": "all 7 real dimensions and all 25 zero-padded dimensions enter the same elementwise mean; padding targets are verified exactly zero",
        },
        "paired_comparison": paired_comparison,
        "base_model": base_report,
        "lora_model": lora_report,
        "path_comparison": paths,
        "interpretation": {
            "toward_demonstrated_actions": "determined by paired raw 7D action error reductions on the same fixed demonstration observations; this is imitation evidence on training states, not a collision-safety verdict",
            "flow_loss_padding": "the final 25 dimensions are zero-padded targets but still contribute to default SFT mean loss",
            "contacts_and_unknown_binding": "saved raw contact records are not reclassified; missing semantic object binding prevents claiming either safe or violation",
        },
    }
    report_path = out_dir / "offline_action_fit_diagnostic.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "plot": paths["plot"],
        "base_first_action_mae": {
            phase: base_report["actions_vs_demonstration"]["by_phase"][phase]["first_action_at_observation"]["mae_all_7d"]
            for phase in PHASES
        },
        "lora_first_action_mae": {
            phase: lora_report["actions_vs_demonstration"]["by_phase"][phase]["first_action_at_observation"]["mae_all_7d"]
            for phase in PHASES
        },
        "lora_flow_loss_7d_vs_padding25": {
            phase: {
                "real7": lora_report["sft_flow_matching"]["by_phase"][phase]["real_action_dims_0_6"]["mean_squared_flow_loss_per_element"],
                "padding25": lora_report["sft_flow_matching"]["by_phase"][phase]["zero_padding_dims_7_31"]["mean_squared_flow_loss_per_element"],
                "default32": lora_report["sft_flow_matching"]["by_phase"][phase]["default_sft_mean_over_all_10x32_elements"],
            }
            for phase in PHASES
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
