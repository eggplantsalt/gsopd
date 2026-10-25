#!/usr/bin/env python3
"""Run one official OpenPI SFT batch and verify fresh Action Expert LoRA gradients."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
from research.gs_opd.data.run_single_episode_sft import SingleEpisodeSftWorker


REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "examples/sft/config"
CONFIG_NAME = "gs_opd_l0t2_round0_fresh_lora_20261025"
OUTPUT_DIR = Path(
    "/opt/data/private/gs_opd/sft/round0_l0t2_fresh_lora_preflight_20261025"
)
REPORT_PATH = OUTPUT_DIR / "preflight_report.json"


class OneBatchWorker(SingleEpisodeSftWorker):
    """Initialize the official RLinf worker but skip training and baseline eval."""

    def init_worker(self) -> None:
        # Keep the production FSDP/OpenPI model, optimizer, and data-loader setup.
        # This preflight only avoids the normal training-run initialization/evals.
        FSDPSftWorker.init_worker(self)

    @staticmethod
    def _trainable_digest(module: torch.nn.Module) -> str:
        digest = hashlib.sha256()
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                digest.update(name.encode("utf-8"))
                digest.update(parameter.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def run_one_batch(self) -> dict[str, Any]:
        loader = self.data_loader._data_loader.torch_loader
        torch.cuda.reset_peak_memory_stats()
        allocated_before_batch = int(torch.cuda.memory_allocated())
        batch = next(iter(self.data_loader))
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise TypeError(f"unexpected_official_openpi_batch:{type(batch).__name__}")
        observation, actions = batch
        action_shape = list(actions.shape)
        expected_shape = [
            int(self.cfg.actor.micro_batch_size),
            int(self.cfg.data.action_horizon),
            int(self.cfg.actor.model.openpi.model_action_dim),
        ]
        if action_shape != expected_shape:
            raise RuntimeError(f"action_tensor_shape_mismatch:{action_shape}:{expected_shape}")
        if int(self.cfg.actor.global_batch_size) != int(self.cfg.actor.micro_batch_size) * self._world_size:
            raise RuntimeError("global_batch_size_does_not_match_single_replica_micro_batch")
        if self.gradient_accumulation != 1:
            raise RuntimeError(f"unexpected_gradient_accumulation:{self.gradient_accumulation}")
        if not torch.isfinite(actions).all().item():
            raise RuntimeError("nonfinite_normalized_action_batch")

        subset = _find_subset(loader.dataset)
        selected = [int(index) for index in subset.indices]
        if selected != self._selected_train_phase_window_indices:
            raise RuntimeError("official_loader_subset_differs_from_provenance_schedule")
        before_digest = self._trainable_digest(self.model.module)
        self.model.train()
        self.model.module.img.eval()
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.get_train_model_output(batch)
        if not torch.isfinite(loss).all().item():
            raise RuntimeError(f"nonfinite_flow_matching_loss:{float(loss.detach().float().item())}")
        backward_context = self.before_micro_batch(self.model, is_last_micro_batch=True)
        with backward_context:
            self.grad_scaler.scale(loss).backward()
        torch.cuda.synchronize()

        grad_tensor_count = 0
        nonzero_grad_tensor_count = 0
        nonfinite_gradients: list[str] = []
        grad_squared_norm = 0.0
        for name, parameter in self.model.module.named_parameters():
            if parameter.requires_grad:
                grad = parameter.grad
                if grad is not None:
                    grad_tensor_count += 1
                    grad32 = grad.detach().float()
                    if not torch.isfinite(grad32).all().item():
                        nonfinite_gradients.append(name)
                    else:
                        grad_squared_norm += float(torch.sum(grad32 * grad32).item())
                        if torch.count_nonzero(grad32).item() > 0:
                            nonzero_grad_tensor_count += 1
        if grad_tensor_count == 0 or nonfinite_gradients or nonzero_grad_tensor_count == 0:
            raise RuntimeError(
                "lora_gradient_check_failed:"
                f"with_grad={grad_tensor_count}:nonzero={nonzero_grad_tensor_count}:"
                f"nonfinite={nonfinite_gradients[:8]}"
            )
        after_digest = self._trainable_digest(self.model.module)
        if before_digest != after_digest:
            raise RuntimeError("trainable_weights_changed_without_optimizer_step")

        phase_exposure = {}
        for phase, indices in self._phase_indices_for_preflight.items():
            index_set = set(int(i) for i in indices)
            phase_exposure[phase] = sum(index in index_set for index in selected)

        norm_stats_path = Path(self.cfg.actor.model.openpi_data.norm_stats_path)
        checkpoint_path = Path(self.cfg.actor.model.model_path) / "model.safetensors"
        image_masks = getattr(observation, "image_masks", {})
        image_mask_values = {
            name: value.detach().cpu().bool().tolist()
            if torch.is_tensor(value)
            else bool(value)
            for name, value in image_masks.items()
        }
        norm_stats = getattr(self.data_config, "norm_stats", {})
        state_norm = norm_stats.get("state")
        action_norm = norm_stats.get("actions")
        result = {
            "status": "passed",
            "actual_training_batch_forward_and_backward": True,
            "optimizer_step_performed": False,
            "data": {
                "path": str(self.cfg.data.train_data_paths),
                "acceptance_report": str(
                    Path(self.cfg.data.train_data_paths) / "round0_lerobot_acceptance.json"
                ),
                "provenance_path": str(self._phase_source),
                "frame_count": int(self._episode_frame_count),
                "episode_count": 1,
                "fps": 20,
                "action_horizon": int(self.cfg.data.action_horizon),
                "valid_unpadded_windows": int(self._unique_training_window_count),
                "scheduled_window_exposures": int(self._training_schedule_window_count),
                "phase_sampling_mode": self._phase_sampling_mode,
                "unique_windows_by_task_phase": self._unique_phase_window_counts,
                "scheduled_exposures_by_task_phase": phase_exposure,
                "batch_size": int(loader.batch_size),
                "batches_per_epoch": len(loader),
            },
            "model": {
                "base_checkpoint": str(checkpoint_path),
                "base_checkpoint_sha256": _sha256(checkpoint_path),
                "base_checkpoint_load_audit": self._base_checkpoint_load_audit,
                "norm_stats_path": str(norm_stats_path),
                "norm_stats_sha256": _sha256(norm_stats_path),
                "norm_stats_source": "original_pi05_libero_checkpoint",
                "action_horizon": int(self.cfg.actor.model.num_action_chunks),
                "action_dim_environment": int(self.cfg.actor.model.action_dim),
                "action_dim_model_padded": int(self.cfg.actor.model.openpi.model_action_dim),
                "input_camera_count": int(self.cfg.actor.model.openpi.num_images_in_input),
                "input_state_dim_after_official_transform": int(observation.state.shape[-1]),
                "input_image_tensor_shapes": _find_image_shapes(observation),
                "official_image_slot_masks": image_mask_values,
                "camera_compatibility": {
                    "stored_views": ["agentview", "robot0_eye_in_hand"],
                    "official_libero_slots": [
                        "base_0_rgb",
                        "left_wrist_0_rgb",
                        "right_wrist_0_rgb",
                    ],
                    "right_wrist_slot": "zero-padded by official LiberoInputs and masked false for non-FAST Pi0.5",
                },
                "normalization_transform": {
                    "asset_id": getattr(self.data_config, "asset_id", None),
                    "uses_quantile_action_norm": bool(
                        getattr(self.data_config, "use_quantile_norm", False)
                    ),
                    "state_stats_dimensions": int(len(state_norm.mean))
                    if state_norm is not None
                    else None,
                    "action_stats_dimensions": int(len(action_norm.mean))
                    if action_norm is not None
                    else None,
                    "norm_stats_sha256": _sha256(norm_stats_path),
                },
                "batch_normalized_action_tensor_shape": action_shape,
                "batch_normalized_action_is_finite": True,
                "action_labels": "executed 7D OSC_POSE labels from accepted LeRobot data; official OpenPI transform normalizes and pads to model dimension 32",
                "base_weights_frozen": True,
                "initial_adapter": self._initial_adapter_load,
                "optimizer_state_loaded": False,
                "trainable_parameter_count": int(self._pre_fsdp_trainable_parameters),
                "trainable_tensor_count": len(self._lora_trainable_names),
                "trainable_parameter_names": self._lora_trainable_names,
                "only_action_expert_lora_trainable": True,
                "lora_rank": int(self.cfg.actor.model.lora_rank),
            },
            "batch": {
                "flow_matching_loss": float(loss.detach().float().item()),
                "forward_metrics": metrics,
                "gradient_tensor_count": grad_tensor_count,
                "nonzero_gradient_tensor_count": nonzero_grad_tensor_count,
                "gradient_global_l2_norm": float(grad_squared_norm**0.5),
                "nonfinite_gradient_tensors": nonfinite_gradients,
                "trainable_weights_digest_before": before_digest,
                "trainable_weights_digest_after": after_digest,
                "trainable_weights_unchanged": True,
            },
            "training_parameters": OmegaConf.to_container(
                self.cfg.actor.optim, resolve=True
            ),
            "runtime": {
                "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
                "gpu_index": torch.cuda.current_device(),
                "allocated_bytes_before_batch": allocated_before_batch,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "fsdp_sharding_strategy": str(self.cfg.actor.fsdp_config.sharding_strategy),
                "world_size": int(self._world_size),
                "gradient_accumulation": int(self.gradient_accumulation),
            },
        }
        self.optimizer.zero_grad(set_to_none=True)
        return result


def _find_subset(dataset: Any) -> torch.utils.data.Subset:
    current = dataset
    for _ in range(16):
        child = getattr(current, "_dataset", None)
        if isinstance(child, torch.utils.data.Subset):
            return child
        if child is None:
            break
        current = child
    raise RuntimeError("official_loader_did_not_retain_provenance_selected_subset")


def _find_image_shapes(value: Any) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    if torch.is_tensor(value) and value.ndim == 4 and value.shape[1] in (1, 3, 4):
        result["image"] = list(value.shape)
    elif isinstance(value, dict):
        for key, item in value.items():
            child = _find_image_shapes(item)
            for image_index, shape in enumerate(child.values()):
                result[f"{key}.{image_index}"] = shape
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            for key, shape in _find_image_shapes(item).items():
                result[f"{index}.{key}"] = shape
    elif hasattr(value, "__dict__"):
        for key, item in vars(value).items():
            for child_key, shape in _find_image_shapes(item).items():
                result[f"{key}.{child_key}"] = shape
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(config_name=CONFIG_NAME)
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster).get_strategy("actor")
    workers = OneBatchWorker.create_group(cfg).launch(
        cluster,
        name="GSOPD_L0T2_ROUND0_ONE_BATCH_PREFLIGHT",
        placement_strategy=placement,
    )
    try:
        workers.init_worker().wait()
        rows = workers.run_one_batch().wait()
        result = rows[0]
        if result.get("status") != "passed":
            raise RuntimeError(f"preflight_result_not_passed:{result}")
        result["resolved_config"] = OmegaConf.to_container(cfg, resolve=True)
        config_path = CONFIG_DIR / f"{CONFIG_NAME}.yaml"
        result["config_path"] = str(config_path)
        result["config_sha256"] = _sha256(config_path)
        result["preflight_script"] = str(Path(__file__).resolve())
        result["preflight_script_sha256"] = _sha256(Path(__file__).resolve())
        training_entry = REPO / "research/gs_opd/data/run_single_episode_sft.py"
        result["training_entry"] = str(training_entry)
        result["training_entry_sha256"] = _sha256(training_entry)
        result["preflight_command"] = (
            "CUDA_VISIBLE_DEVICES=0 "
            "EMBODIED_PATH=/workspace/zy_workspace/RLinf/examples/embodiment "
            "PYTHONPATH=/workspace/zy_workspace/RLinf-gs-opd-overfit-recovery:"
            "/workspace/zy_workspace/RLinf:/workspace/zy_workspace/LIBERO-Safety "
            "/workspace/zy_workspace/RLinf/.venv/bin/python "
            "research/gs_opd/data/preflight_round0_lora.py"
        )
        result["formal_training_command"] = (
            "CUDA_VISIBLE_DEVICES=0 "
            "EMBODIED_PATH=/workspace/zy_workspace/RLinf/examples/embodiment "
            "PYTHONPATH=/workspace/zy_workspace/RLinf-gs-opd-overfit-recovery:"
            "/workspace/zy_workspace/RLinf:/workspace/zy_workspace/LIBERO-Safety "
            "/workspace/zy_workspace/RLinf/.venv/bin/python "
            "research/gs_opd/data/run_single_episode_sft.py "
            "--config-name gs_opd_l0t2_round0_fresh_lora_20261025"
        )
        result["formal_training_started"] = False
        manifest_path = Path(cfg.data.train_data_paths) / "data_manifest.json"
        acceptance_path = Path(cfg.data.train_data_paths) / "round0_lerobot_acceptance.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        result["dataset_manifest_sha256"] = _sha256(manifest_path)
        result["dataset_acceptance_sha256"] = _sha256(acceptance_path)
        result["dataset_manifest_summary"] = {
            key: manifest.get(key)
            for key in (
                "schema",
                "format",
                "lerobot_package_version",
                "fps",
                "episode_count",
                "episode_length",
                "total_action_samples",
                "action_horizon",
                "task",
                "camera_features",
                "state_feature",
                "action_feature",
            )
        }
        result["dataset_acceptance_summary"] = {
            "status": acceptance.get("status"),
            "safety_acceptance": acceptance.get("safety_acceptance"),
            "action_readback": acceptance.get("action_readback"),
            "observation_readback": acceptance.get("observation_readback"),
            "training_windows": acceptance.get("training_windows"),
            "lora_training_started_before_preflight": acceptance.get(
                "lora_training_started"
            ),
        }
        source_run_config = Path(
            "/opt/data/private/gs_opd/rollouts/"
            "l0t2_round0_tsfs_safe_seed_20261024/episode1/run_config.json"
        )
        if source_run_config.is_file():
            source_run = json.loads(source_run_config.read_text(encoding="utf-8"))
            result["model"]["student_checkpoint_provenance"] = {
                "archived_run_config": str(source_run_config),
                "archived_student_checkpoint": source_run.get("student_checkpoint"),
                "path_matches_base_checkpoint": (
                    str(source_run.get("student_checkpoint"))
                    == str(cfg.actor.model.model_path)
                ),
            }
        inventory = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        result["parallelism"] = {
            "host_gpu_inventory": [
                {
                    "index": int(line.split(",")[0].strip()),
                    "name": line.split(",")[1].strip(),
                    "memory_total_mib": int(line.split(",")[2].strip()),
                    "memory_free_mib_at_report": int(line.split(",")[3].strip()),
                }
                for line in inventory
                if line.strip()
            ],
            "rlinf_fsdp_sft_data_parallel_backend": True,
            "gs_opd_provenance_sampler_world_size_limit": 1,
            "selected_world_size": 1,
            "selected_accelerator": "GPU 0",
            "model_sharding_strategy": str(cfg.actor.fsdp_config.sharding_strategy),
            "reason": "single RTX 4090 completed the real batch preflight; current provenance sampler explicitly requires one replica, so no multi-GPU placement or model partition is configured",
        }
        REPORT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "report_path": str(REPORT_PATH),
                    "loss": result["batch"]["flow_matching_loss"],
                    "lora_gradient_tensors": result["batch"]["gradient_tensor_count"],
                    "nonzero_gradient_tensors": result["batch"]["nonzero_gradient_tensor_count"],
                    "trainable_parameter_count": result["model"]["trainable_parameter_count"],
                    "peak_allocated_bytes": result["runtime"]["peak_allocated_bytes"],
                },
                indent=2,
            )
        )
        return 0
    except BaseException as exc:
        failure = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "optimizer_step_performed": False,
        }
        REPORT_PATH.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
        raise
    finally:
        try:
            import ray

            for info in workers.worker_info_list:
                ray.kill(info.worker, no_restart=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
