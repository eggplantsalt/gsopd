#!/usr/bin/env python3
"""Export the final RLinf SFT adapter and compare composed inference to training weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models import get_model
from rlinf.models.embodiment.openpi.checkpoint import (
    _normalize_state_dict,
    resolve_model_safetensors,
)
from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

from research.gs_opd.data.lora_checkpoint import load_lora_adapter, save_lora_adapter


ROOT = Path(__file__).resolve().parents[3]
BASE_CHECKPOINT = Path("/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft")
DEFAULT_NPZ = ROOT / "logs/gs_opd/mink_panda_physical_v3/live_left_bspline_osc_adaptive_20261019_attempt1/direct_bspline_osc_trajectory.npz"


def _model_config(model_path: Path):
    with initialize_config_dir(config_dir=str(ROOT / "evaluations/libero"), version_base=None):
        cfg = compose(config_name="libero_spatial_openpi_pi05_smoke")
    model_cfg = cfg.rollout.model
    model_cfg.model_path = str(model_path)
    OmegaConf.update(model_cfg, "is_lora", False, force_add=True)
    OmegaConf.update(model_cfg, "lora_rank", 32, force_add=True)
    OmegaConf.update(model_cfg, "num_action_chunks", 10, force_add=True)
    OmegaConf.update(model_cfg, "action_dim", 7, force_add=True)
    OmegaConf.update(model_cfg, "openpi.task", "eval", force_add=True)
    OmegaConf.update(model_cfg, "openpi.config_name", "pi05_libero", force_add=True)
    OmegaConf.update(model_cfg, "openpi.action_expert_variant", "gemma_300m_lora", force_add=True)
    OmegaConf.update(model_cfg, "openpi.action_chunk", 10, force_add=True)
    OmegaConf.update(model_cfg, "openpi.action_env_dim", 7, force_add=True)
    OmegaConf.update(model_cfg, "openpi.model_action_dim", 32, force_add=True)
    OmegaConf.update(model_cfg, "openpi.num_steps", 5, force_add=True)
    OmegaConf.update(model_cfg, "openpi.torch_compile", False, force_add=True)
    OmegaConf.update(
        model_cfg,
        "openpi_data.norm_stats_path",
        str(BASE_CHECKPOINT / "physical-intelligence/libero/norm_stats.json"),
        force_add=True,
    )
    return model_cfg


def _sample_observation(npz_path: Path):
    from rlinf.envs.sim.libero.utils import get_libero_image, get_libero_wrist_image, quat2axisangle

    with np.load(npz_path, allow_pickle=False) as archive:
        step = 0
        image = archive[f"decision_agentview_image"][0]
        wrist = archive[f"decision_robot0_eye_in_hand_image"][0]
        position = archive[f"decision_robot0_eef_pos"][0]
        quaternion = archive[f"decision_robot0_eef_quat"][0]
        gripper = archive[f"decision_robot0_gripper_qpos"][0]
        metadata = json.loads(str(archive["metadata_json"].item()))
    state = np.concatenate([position, quat2axisangle(quaternion.copy()), gripper]).astype(np.float32)
    observation = {
        "main_images": torch.from_numpy(np.ascontiguousarray(get_libero_image({"agentview_image": image}))).unsqueeze(0),
        "wrist_images": torch.from_numpy(np.ascontiguousarray(get_libero_wrist_image({"robot0_eye_in_hand_image": wrist}))).unsqueeze(0),
        "states": torch.from_numpy(state).unsqueeze(0),
        "task_descriptions": [metadata["instruction"]],
    }
    return observation


def _predict(model_path: Path, observation: dict, adapter_path: Path | None, seed: int) -> np.ndarray:
    model = get_model(_model_config(model_path))
    if adapter_path is not None:
        load_lora_adapter(model, adapter_path)
    model = model.to("cuda:0").eval()
    rng = torch.Generator(device="cuda:0").manual_seed(seed)
    with torch.inference_mode():
        actions, _ = model.predict_action_batch(observation, mode="eval", rng=rng)
    result = actions[0].detach().float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="Final RLinf full_weights.pt")
    parser.add_argument("--output", type=Path, help="Standalone adapter output")
    parser.add_argument("--report", type=Path, required=True, help="Training report.json")
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.report.parent / "lora_adapter.pt"
    if args.result is None:
        args.result = args.output.with_name("adapter_reload_verification.json")

    raw = torch.load(args.weights, map_location="cpu", weights_only=False)
    full_state = _normalize_state_dict(as_state_dict(raw))
    del raw
    report = json.loads(args.report.read_text(encoding="utf-8"))
    metadata = save_lora_adapter(
        full_state,
        args.output,
        base_checkpoint=report["base_checkpoint"],
        rank=32,
        training_checkpoint=str(args.weights),
    )
    expected_names = set(report["trainable_parameter_names"])
    actual_names = set(metadata["tensor_names"])
    if actual_names != expected_names:
        raise RuntimeError(
            "exported_lora_names_do_not_match_training_manifest: "
            f"missing={sorted(expected_names - actual_names)[:8]} "
            f"unexpected={sorted(actual_names - expected_names)[:8]}"
        )
    del full_state

    observation = _sample_observation(args.npz)
    full_prediction = _predict(args.weights, observation, None, seed=20261019)
    composed_prediction = _predict(BASE_CHECKPOINT, observation, args.output, seed=20261019)
    if full_prediction.shape != composed_prediction.shape:
        raise RuntimeError("composed_inference_action_shape_mismatch")
    difference = np.abs(full_prediction - composed_prediction)
    result = {
        "adapter_path": str(args.output.resolve()),
        "adapter_metadata": metadata,
        "adapter_names_match_training_manifest": True,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "inference_comparison": {
            "observation_source": str(args.npz),
            "source_step": 0,
            "actions_shape": list(full_prediction.shape),
            "seed": 20261019,
            "max_abs_action_difference": float(difference.max(initial=0.0)),
            "mean_abs_action_difference": float(difference.mean()),
            "allclose_atol_1e_5": bool(np.allclose(full_prediction, composed_prediction, atol=1e-5, rtol=1e-5)),
        },
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["inference_comparison"]["allclose_atol_1e_5"]:
        raise RuntimeError("standalone_adapter_inference_does_not_match_full_training_checkpoint")


if __name__ == "__main__":
    main()
