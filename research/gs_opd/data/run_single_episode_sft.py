#!/usr/bin/env python3
"""Run the official RLinf OpenPI SFT Runner on one complete LeRobot episode."""

from __future__ import annotations

import json
import hashlib
import itertools
import logging
import os
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf
from torch.utils.data import Subset

from rlinf.config import SupportedModel, validate_cfg
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


class SingleEpisodeSftWorker(FSDPVlaSftWorker):
    """Use official OpenPI transforms with complete, phase-balanced episode windows."""

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        loader, data_config = super().build_dataloader(data_paths, eval_dataset)
        if SupportedModel(self.cfg.actor.model.model_type) != SupportedModel.OPENPI:
            raise TypeError("GS-OPD single-episode SFT requires the OpenPI loader")

        torch_loader = loader._data_loader.torch_loader
        transformed = torch_loader.dataset
        raw = transformed
        while hasattr(raw, "_dataset"):
            raw = raw._dataset

        phase_manifest_path = OmegaConf.select(
            self.cfg, "data.phase_manifest_path", default=None
        )
        if phase_manifest_path:
            return self._build_manifest_phase_dataloader(
                loader,
                data_config,
                transformed,
                raw,
                Path(phase_manifest_path),
                eval_dataset,
            )

        frame_count = len(raw)
        expected_frames = int(self.cfg.data.expected_frames)
        horizon = int(self.cfg.data.action_horizon)
        if frame_count != expected_frames:
            raise ValueError(
                f"expected the preserved {expected_frames}-frame episode, found {frame_count} frames"
            )
        meta = getattr(raw, "meta", None)
        episodes = getattr(meta, "episodes", None)
        if episodes is not None and len(episodes) != 1:
            raise ValueError(f"expected one episode, found {len(episodes)}")

        valid_count = frame_count - horizon + 1
        if valid_count <= 0:
            raise ValueError("episode is shorter than the OpenPI action horizon")
        boundaries = [int(value) for value in self.cfg.data.phase_boundaries]
        if len(boundaries) != 2 or not (0 < boundaries[0] < boundaries[1] < frame_count):
            raise ValueError(f"invalid_episode_phase_boundaries:{boundaries}")
        phase_names = ("student_grasp", "bspline_osc_carry", "student_place")
        phase_ranges = (
            (0, boundaries[0]),
            (boundaries[0], boundaries[1]),
            (boundaries[1], valid_count),
        )
        phase_indices = {
            name: np.arange(start, min(end, valid_count), dtype=np.int64)
            for name, (start, end) in zip(phase_names, phase_ranges)
        }
        if any(indices.size == 0 for indices in phase_indices.values()):
            raise ValueError("every demonstration phase must contain a complete action window")

        if eval_dataset:
            per_phase = int(self.cfg.data.eval_windows_per_phase)
            selected = []
            phase_sequence = []
            phase_eval_indices = {}
            for name in phase_names:
                candidates = phase_indices[name]
                if candidates.size < per_phase:
                    raise ValueError(f"not_enough_{name}_windows_for_fixed_eval")
                indices = np.linspace(
                    candidates[0], candidates[-1], per_phase, dtype=np.int64
                )
                if len(np.unique(indices)) != per_phase:
                    raise ValueError(f"fixed_{name}_evaluation_indices_are_not_unique")
                selected.extend(indices.tolist())
                phase_sequence.extend([name] * per_phase)
                phase_eval_indices[name] = indices.tolist()
            self._fixed_eval_phase_sequence = phase_sequence
            self._fixed_eval_phase_indices = phase_eval_indices
        else:
            target_per_phase = int(self.cfg.data.balanced_windows_per_phase)
            selected = list(range(valid_count))
            phase_exposures = {}
            for name in phase_names:
                candidates = phase_indices[name]
                if candidates.size > target_per_phase:
                    raise ValueError(
                        f"{name}_has_{candidates.size}_windows_but_balance_target_is_{target_per_phase}"
                    )
                repeats = target_per_phase - int(candidates.size)
                if repeats:
                    selected.extend(np.resize(candidates, repeats).tolist())
                phase_exposures[name] = target_per_phase
            self._phase_ranges = {
                name: [int(start), int(end - 1)]
                for name, (start, end) in zip(phase_names, phase_ranges)
            }
            self._unique_phase_window_counts = {
                name: int(indices.size) for name, indices in phase_indices.items()
            }
            self._balanced_phase_window_exposure = phase_exposures
            self._unique_training_window_count = valid_count
            self._training_schedule_window_count = len(selected)

        # The official OpenPI loader retains this same transformed Dataset and its
        # sampler. Replacing the raw LeRobot leaf with an index view leaves every
        # image/state/action transform and the checkpoint statistics untouched.
        parent = transformed
        while hasattr(parent, "_dataset") and parent._dataset is not raw:
            parent = parent._dataset
        if not hasattr(parent, "_dataset") or parent._dataset is not raw:
            raise RuntimeError("could not locate the raw LeRobot dataset leaf")
        parent._dataset = Subset(raw, selected)
        sampler = torch_loader.sampler
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            if sampler.num_replicas != 1:
                raise ValueError(
                    "single-episode GS-OPD SFT is configured for one GPU, but "
                    f"the official loader initialized {sampler.num_replicas} sampler replicas"
                )
            sampler.num_samples = len(selected)
            sampler.total_size = len(selected)

        return loader, data_config

    def _build_manifest_phase_dataloader(
        self,
        loader: Any,
        data_config: Any,
        transformed: Any,
        raw: Any,
        manifest_path: Path,
        eval_dataset: bool,
    ):
        """Balance horizon windows across multiple full episodes from a sidecar."""
        from rlinf.data.storage.lerobot.compat import episode_boundaries

        if not manifest_path.is_file():
            raise FileNotFoundError(f"episode_phase_manifest_not_found:{manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_episodes = manifest.get("episode_sources", [])
        starts, ends = episode_boundaries(raw)
        if len(starts) != len(manifest_episodes) or len(raw.meta.episodes) != len(
            manifest_episodes
        ):
            raise ValueError(
                "phase_manifest_episode_count_mismatch:"
                f"dataset={len(starts)} manifest={len(manifest_episodes)}"
            )

        horizon = int(self.cfg.data.action_horizon)
        phase_names = ("student_grasp", "bspline_osc_carry", "student_place")
        phase_indices: dict[str, list[int]] = {name: [] for name in phase_names}
        all_unique_indices: list[int] = []
        phase_segments: dict[str, list[list[int]]] = {name: [] for name in phase_names}
        for episode_index, (episode, start, end) in enumerate(
            zip(manifest_episodes, starts, ends)
        ):
            frame_count = end - start
            if int(episode.get("episode_index", -1)) != episode_index or int(
                episode.get("frame_count", -1)
            ) != frame_count:
                raise ValueError(
                    f"phase_manifest_boundary_mismatch_episode_{episode_index}:"
                    f"dataset={frame_count} manifest={episode.get('frame_count')}"
                )
            valid_count = frame_count - horizon + 1
            boundaries = [
                0,
                *[int(value) for value in episode["phase_boundaries_local_step"]],
                valid_count,
            ]
            if len(boundaries) != 4 or not (
                0 < boundaries[1] < boundaries[2] < boundaries[3]
            ):
                raise ValueError(
                    f"invalid_phase_manifest_boundaries_episode_{episode_index}:{boundaries}"
                )
            all_unique_indices.extend(range(start, start + valid_count))
            for phase, local_start, local_end in zip(
                phase_names, boundaries[:-1], boundaries[1:]
            ):
                if local_end <= local_start:
                    raise ValueError(
                        f"empty_{phase}_window_range_episode_{episode_index}"
                    )
                global_start = start + local_start
                global_end = start + local_end
                phase_indices[phase].extend(range(global_start, global_end))
                phase_segments[phase].append([global_start, global_end - 1])

        if any(not values for values in phase_indices.values()):
            raise ValueError("every cumulative episode phase must have valid windows")
        if eval_dataset:
            per_phase = int(self.cfg.data.eval_windows_per_phase)
            selected: list[int] = []
            phase_sequence: list[str] = []
            phase_eval_indices: dict[str, list[int]] = {}
            for name in phase_names:
                candidates = np.asarray(phase_indices[name], dtype=np.int64)
                if candidates.size < per_phase:
                    raise ValueError(f"not_enough_{name}_windows_for_fixed_eval")
                positions = np.linspace(
                    0, candidates.size - 1, per_phase, dtype=np.int64
                )
                indices = candidates[positions]
                if len(np.unique(indices)) != per_phase:
                    raise ValueError(f"fixed_{name}_evaluation_indices_are_not_unique")
                selected.extend(indices.tolist())
                phase_sequence.extend([name] * per_phase)
                phase_eval_indices[name] = indices.tolist()
            self._fixed_eval_phase_sequence = phase_sequence
            self._fixed_eval_phase_indices = phase_eval_indices
        else:
            configured_target = int(
                OmegaConf.select(
                    self.cfg, "data.balanced_windows_per_phase", default=0
                )
            )
            target_per_phase = max(
                configured_target, max(len(values) for values in phase_indices.values())
            )
            selected = list(all_unique_indices)
            for name in phase_names:
                candidates = np.asarray(phase_indices[name], dtype=np.int64)
                repeats = target_per_phase - int(candidates.size)
                if repeats:
                    selected.extend(np.resize(candidates, repeats).tolist())
            self._phase_ranges = phase_segments
            self._unique_phase_window_counts = {
                name: len(values) for name, values in phase_indices.items()
            }
            self._balanced_phase_window_exposure = {
                name: target_per_phase for name in phase_names
            }
            self._unique_training_window_count = len(all_unique_indices)
            self._training_schedule_window_count = len(selected)

        parent = transformed
        while hasattr(parent, "_dataset") and parent._dataset is not raw:
            parent = parent._dataset
        if not hasattr(parent, "_dataset") or parent._dataset is not raw:
            raise RuntimeError("could_not_locate_raw_LeRobot_dataset_leaf")
        parent._dataset = Subset(raw, selected)
        sampler = loader._data_loader.torch_loader.sampler
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            if sampler.num_replicas != 1:
                raise ValueError(
                    "cumulative GS-OPD SFT is configured for one GPU, but "
                    f"the official loader initialized {sampler.num_replicas} sampler replicas"
                )
            sampler.num_samples = len(selected)
            sampler.total_size = len(selected)
        if eval_dataset:
            self._unique_training_window_count = len(all_unique_indices)
            self._training_schedule_window_count = len(selected)
            self._unique_phase_window_counts = {
                name: len(values) for name, values in phase_indices.items()
            }
            self._balanced_phase_window_exposure = {
                name: int(self.cfg.data.eval_windows_per_phase)
                for name in phase_names
            }
            self._phase_ranges = phase_segments
        self._episode_phase_manifest_path = str(manifest_path)
        self._selected_phase_window_indices = [int(index) for index in selected]
        return loader, data_config

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is None:
            return super().model_provider_func()
        from research.gs_opd.data.lora_checkpoint import load_lora_adapter
        from openpi.transforms import compose
        from rlinf.models.embodiment.openpi.transforms.pipeline import (
            build_openpi_transforms,
        )
        from safetensors import safe_open
        from rlinf.models.embodiment.openpi.checkpoint import (
            _convert_official_openpi_keys,
            resolve_model_safetensors,
        )

        checkpoint = resolve_model_safetensors(self.cfg.actor.model.model_path)
        if checkpoint is None:
            raise FileNotFoundError("the original Student model.safetensors was not found")
        model_state = model.state_dict()
        model_keys = set(model_state)
        adapter_keys = {key for key in model_keys if "lora_" in key}
        with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
            raw_source_keys = set(source.keys())
            # Audit the same legacy OpenPI -> RLinf Pi0 key conversion used by
            # get_model(), using meta tensors so the 7.5 GiB file is not loaded
            # into host memory a second time.
            meta_state = {
                key: torch.empty(source.get_slice(key).get_shape(), device="meta")
                for key in raw_source_keys
            }
            converted_source = _convert_official_openpi_keys(meta_state, checkpoint)
            source_keys = set(converted_source)
            missing_base = sorted(
                key for key in model_keys if "lora_" not in key and key not in source_keys
            )
            unexpected_base = sorted(source_keys - model_keys)
            if missing_base or unexpected_base:
                raise RuntimeError(
                    "initial_checkpoint_non_lora_key_coverage_failed: "
                    f"missing={missing_base[:12]} unexpected={unexpected_base[:12]}"
                )
            if not adapter_keys or adapter_keys & source_keys:
                raise RuntimeError("the selected Action Expert LoRA modules are not fresh adapters")
            sample_key = next(
                (
                    key for key in ("action_out_proj.weight", "state_proj.weight", "img.stem.weight")
                    if key in raw_source_keys and key in model_keys
                ),
                None,
            )
            sample_diff = None
            if sample_key is not None:
                source_tensor = source.get_tensor(sample_key)
                loaded_tensor = model_state[sample_key].detach().cpu().to(source_tensor.dtype)
                sample_diff = float(
                    (loaded_tensor.float() - source_tensor.float()).abs().max().item()
                )
                if sample_diff != 0.0:
                    raise RuntimeError(
                        f"initial_checkpoint_sample_weight_mismatch:{sample_key}:{sample_diff}"
                    )

        adapter_path_value = self.cfg.gs_opd.get("initial_adapter_path", None)
        if adapter_path_value:
            adapter_path = Path(adapter_path_value).expanduser().resolve()
            if not adapter_path.is_file():
                raise FileNotFoundError(f"initial_lora_adapter_not_found:{adapter_path}")
            adapter_metadata = load_lora_adapter(model, adapter_path)
            if int(adapter_metadata.get("rank", -1)) != int(self.cfg.actor.model.lora_rank):
                raise RuntimeError(
                    f"initial_adapter_rank_mismatch:{adapter_metadata.get('rank')}"
                )
            loaded_adapter_names = set(adapter_metadata.get("tensor_names", []))
            if loaded_adapter_names != adapter_keys:
                raise RuntimeError("initial_adapter_does_not_cover_all_action_expert_lora_tensors")
            self._initial_adapter_load = {
                "status": "loaded_existing_adapter",
                "path": str(adapter_path),
                "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
                "metadata": adapter_metadata,
                "all_action_expert_lora_tensors_loaded": True,
            }
        else:
            self._initial_adapter_load = {
                "status": "fresh_lora_initialization",
                "path": None,
                "reason": "no prior adapter was available; rank-32 Action Expert LoRA tensors use the model initializer",
                "rank": int(self.cfg.actor.model.lora_rank),
                "tensor_count": len(adapter_keys),
            }

        trainable_names = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad_("lora_" in name)
            if parameter.requires_grad:
                trainable_names.append(name)
        if not trainable_names or any(".1." not in name for name in trainable_names):
            raise RuntimeError("trainable_parameters_must_be_action_expert_lora_only")
        non_lora_trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]
        if non_lora_trainable:
            raise RuntimeError(f"non_lora_parameters_are_trainable:{non_lora_trainable[:8]}")
        model.img.eval()
        frozen_tensors = sum(1 for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logging.info(
            "GS-OPD full-episode LoRA SFT: trainable=%d LoRA / total=%d parameters; "
            "frozen_tensors=%d; base_weight_keys=%d",
            trainable,
            total,
            frozen_tensors,
            len(source_keys),
        )
        self._lora_trainable_names = trainable_names
        self._base_checkpoint_load_audit = {
            "checkpoint": str(checkpoint),
            "raw_source_weight_key_count": len(raw_source_keys),
            "source_weight_key_count": len(source_keys),
            "model_non_lora_key_count": len(model_keys - adapter_keys),
            "missing_non_lora_keys": [],
            "unexpected_source_keys": [],
            "loader_strict_missing_unexpected_check_passed": True,
            "adapter_key_count_initialized_for_training": len(adapter_keys),
            "sample_weight_key": sample_key,
            "sample_weight_max_abs_diff": sample_diff,
            "complete_non_lora_load": True,
        }
        _, output_transforms = build_openpi_transforms(
            str(self.cfg.actor.model.model_path),
            str(self.cfg.actor.model.openpi.config_name),
            data_kwargs=OmegaConf.to_container(
                self.cfg.actor.model.openpi_data, resolve=True
            ),
        )
        self._eval_output_transform = compose(output_transforms)
        self._pre_fsdp_trainable_parameters = trainable
        self._pre_fsdp_frozen_tensor_count = frozen_tensors
        return model

    def init_worker(self):
        super().init_worker()
        self._steps_per_epoch = self.get_max_steps_per_epoch()
        step_limits = []
        max_epochs = int(self.cfg.runner.get("max_epochs", -1))
        max_steps = int(self.cfg.runner.get("max_steps", -1))
        if max_epochs > 0:
            step_limits.append(self._steps_per_epoch * max_epochs)
        if max_steps >= 0:
            step_limits.append(max_steps)
        self._expected_total_steps = min(step_limits) if step_limits else self._steps_per_epoch
        configured_total_steps = int(self.cfg.actor.optim.total_training_steps)
        if configured_total_steps != self._expected_total_steps:
            raise ValueError(
                "optimizer schedule must span every configured trajectory step: "
                f"total_training_steps={configured_total_steps}, "
                f"expected={self._expected_total_steps}"
            )
        self._report_dir = Path(self.cfg.runner.logger.log_path) / str(
            self.cfg.runner.logger.experiment_name
        )
        self._report_dir.mkdir(parents=True, exist_ok=True)
        self._training_started = time.monotonic()
        self._step_metrics_path = self._report_dir / "step_metrics.jsonl"
        self._step_metrics_path.write_text("", encoding="utf-8")
        self._epoch_eval_metrics_path = self._report_dir / "epoch_eval.jsonl"
        self._epoch_eval_metrics_path.write_text("", encoding="utf-8")
        self._flow_eval_seed = int(self.cfg.gs_opd.flow_eval_seed)
        self._action_eval_seed = int(self.cfg.gs_opd.action_eval_seed)
        self._action_eval_solver_steps = int(self.cfg.gs_opd.action_eval_solver_steps)
        self._fixed_eval_steps = [
            int(step) for step in self.cfg.gs_opd.get("fixed_eval_steps", [])
        ]
        if self._fixed_eval_steps != sorted(set(self._fixed_eval_steps)) or any(
            step <= 0 or step > self._expected_total_steps
            for step in self._fixed_eval_steps
        ):
            raise ValueError(f"invalid_fixed_eval_steps:{self._fixed_eval_steps}")
        self._fixed_eval_metrics_path = self._report_dir / "fixed_eval_by_step.jsonl"
        self._fixed_eval_metrics_path.write_text("", encoding="utf-8")
        self._epoch_evaluations: list[dict[str, Any]] = []
        self._fixed_eval_by_step: list[dict[str, Any]] = []
        self._training_loss_history: list[float] = []
        self._first_failure_reason: str | None = None
        self._stop_training_requested = False

        baseline = self._evaluate_fixed_fit()
        if not self._tree_is_finite(baseline):
            raise RuntimeError("nonfinite_fixed_evaluation_before_training")
        baseline_record = {"step": 0, "fixed_eval": baseline}
        self._fixed_eval_by_step.append(baseline_record)
        with self._fixed_eval_metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(baseline_record, ensure_ascii=False) + "\n")
        initial_parameter = self._trainable_parameter_signature()
        initial_lr = float(self.optimizer.param_groups[0]["lr"])
        if not np.isfinite(initial_lr) or initial_lr != float(self.cfg.actor.optim.lr):
            raise RuntimeError(f"configured_initial_learning_rate_mismatch:{initial_lr}")
        if str(self.cfg.actor.optim.lr_scheduler) == "constant" and int(
            self.cfg.actor.optim.get("lr_warmup_steps", -1)
        ) != 0:
            raise RuntimeError("constant_learning_rate_requires_zero_warmup")
        report = {
            "dataset_path": str(self.cfg.data.train_data_paths),
            "episode_frames": int(self.cfg.data.expected_frames),
            "action_horizon": int(self.cfg.data.action_horizon),
            "unique_valid_training_windows": int(self._unique_training_window_count),
            "valid_training_windows_per_phase": self._unique_phase_window_counts,
            "balanced_window_exposure_per_phase_per_epoch": self._balanced_phase_window_exposure,
            "balanced_training_schedule_samples_per_epoch": int(self._training_schedule_window_count),
            "phase_ranges_step_inclusive": self._phase_ranges,
            "training_batch_size": int(self.cfg.actor.micro_batch_size),
            "steps_per_epoch": int(self._steps_per_epoch),
            "epochs": int(self.cfg.runner.max_epochs),
            "total_optimizer_steps": int(self._expected_total_steps),
            "optimizer_schedule_steps": configured_total_steps,
            "fixed_eval_indices_by_phase": self._fixed_eval_phase_indices,
            "fixed_eval_steps": [0, *self._fixed_eval_steps],
            "fixed_eval_seed": self._flow_eval_seed,
            "fixed_action_eval_seed": self._action_eval_seed,
            "fixed_action_eval_solver_steps": self._action_eval_solver_steps,
            "base_checkpoint": str(self.cfg.actor.model.model_path),
            "initial_adapter_load": self._initial_adapter_load,
            "optimizer_state": {
                "status": "reinitialized",
                "reason": "a new AdamW optimizer was created for this run; no optimizer state was loaded",
                "optimizer": "RLinf SFTRunner configured AdamW",
                "initial_optimizer_step": 0,
            },
            "learning_rate_schedule": {
                "scheduler": str(self.cfg.actor.optim.lr_scheduler),
                "peak_lr": float(self.cfg.actor.optim.lr),
                "minimum_lr": float(self.cfg.actor.optim.min_lr),
                "warmup_steps": int(self.cfg.actor.optim.lr_warmup_steps),
                "schedule_steps": configured_total_steps,
                "initial_lr_after_scheduler_setup": initial_lr,
                "constant_learning_rate": str(self.cfg.actor.optim.lr_scheduler) == "constant"
                and int(self.cfg.actor.optim.lr_warmup_steps) == 0,
            },
            "norm_stats_path": str(self.cfg.actor.model.openpi_data.norm_stats_path),
            "frozen_original_parameter_tensors": int(
                getattr(self, "_pre_fsdp_frozen_tensor_count", 0)
            ),
            "trainable_parameters": int(self._pre_fsdp_trainable_parameters),
            "trainable_parameter_names": self._lora_trainable_names,
            "base_checkpoint_load_audit": self._base_checkpoint_load_audit,
            "only_action_expert_lora_trainable": True,
            "lora_rank": 32,
            "trainable_parameter_signature_before": initial_parameter,
            "fixed_eval_before": baseline,
            "fixed_eval_by_step": self._fixed_eval_by_step,
            "fixed_eval_by_epoch": self._epoch_evaluations,
            "divergence_guard": {
                "stop_on_nonfinite_loss_gradient_or_learning_rate": True,
                "stop_on_finite_batch_loss_spike": False,
            },
            "early_stop_policy": {
                "fit_plateau_or_regression_stopping": False,
                "stop_only_on_nonfinite_training_or_fixed_evaluation_metrics": True,
            },
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "gpu_peak_memory_bytes_before_training": int(
                torch.cuda.max_memory_allocated()
            ),
        }
        self._write_report(report)

    def get_train_model_output(self, batch: Any):
        # FSDPSftWorker calls model.train() for each update; keep the frozen
        # image tower in inference mode, as Pi0.freeze_vlm() intends.
        self.model.module.img.eval()
        return super().get_train_model_output(batch)

    def run_training(self):
        metrics = super().run_training()
        completed_step = int(self.global_step) + 1
        scalar_metrics = {}
        nonfinite_metrics = []
        for key, value in metrics.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                scalar_metrics[key] = number
            else:
                nonfinite_metrics.append(key)
        current_loss = scalar_metrics.get("loss")
        failure_reason = None
        if nonfinite_metrics:
            failure_reason = "nonfinite_training_metrics:" + ",".join(sorted(nonfinite_metrics))
        if failure_reason is None and str(self.cfg.actor.optim.lr_scheduler) == "constant":
            actual_lr = scalar_metrics.get("learning_rate")
            expected_lr = float(self.cfg.actor.optim.lr)
            if actual_lr is None or not np.isclose(actual_lr, expected_lr, rtol=1e-6, atol=0.0):
                failure_reason = f"constant_learning_rate_changed:{actual_lr}"
        if current_loss is not None and np.isfinite(current_loss):
            self._training_loss_history.append(current_loss)

        step_record = {
            "step": completed_step,
            "epoch": int((completed_step - 1) // self._steps_per_epoch + 1),
            **scalar_metrics,
            "nonfinite_metrics": nonfinite_metrics,
            "stop_reason": failure_reason,
            "elapsed_seconds": float(time.monotonic() - self._training_started),
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            "gpu_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        if failure_reason is None and completed_step in self._fixed_eval_steps:
            fixed_eval = self._evaluate_fixed_fit()
            if not self._tree_is_finite(fixed_eval):
                failure_reason = "nonfinite_fixed_evaluation_at_step_" + str(completed_step)
            eval_record = {
                "step": completed_step,
                "fixed_eval": fixed_eval,
                "elapsed_seconds": float(time.monotonic() - self._training_started),
            }
            self._fixed_eval_by_step.append(eval_record)
            with self._fixed_eval_metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

        if failure_reason is not None:
            self._first_failure_reason = failure_reason
            self._stop_training_requested = True
            step_record["stop_reason"] = failure_reason

        with self._step_metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(step_record, ensure_ascii=False) + "\n")

        if self._stop_training_requested:
            metrics["gs_opd_stop_training"] = 1.0

        if completed_step == self._expected_total_steps or self._stop_training_requested:
            final_fit = self._fixed_eval_by_step[-1]["fixed_eval"]
            final_parameter = self._trainable_parameter_signature()
            final = json.loads((self._report_dir / "report.json").read_text())
            final.update(
                {
                    "fixed_eval_after": final_fit,
                    "fixed_eval_by_step": self._fixed_eval_by_step,
                    "fixed_eval_by_epoch": self._epoch_evaluations,
                    "stop_reason": self._first_failure_reason
                    or "configured_optimizer_step_budget_completed",
                    "completed_optimizer_steps": completed_step,
                    "trainable_parameter_signature_after": final_parameter,
                    "trainable_parameter_changed": (
                        final_parameter["signature"]
                        != final["trainable_parameter_signature_before"]["signature"]
                    ),
                    "elapsed_seconds": float(
                        time.monotonic() - self._training_started
                    ),
                    "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "gpu_peak_reserved_memory_bytes": int(
                        torch.cuda.max_memory_reserved()
                    ),
                    "checkpoint_path": str(
                        self._report_dir
                        / "checkpoints"
                        / f"global_step_{completed_step}"
                        / "actor"
                    ),
                }
            )
            self._write_report(final)
        return metrics

    @classmethod
    def _tree_is_finite(cls, value: Any) -> bool:
        if isinstance(value, dict):
            return all(cls._tree_is_finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(cls._tree_is_finite(item) for item in value)
        if isinstance(value, (int, float, np.number)):
            return bool(np.isfinite(value))
        return True

    @staticmethod
    def _fit_metric_spiked(current: dict[str, Any], previous: dict[str, Any]) -> bool:
        for phase, metrics in current["by_phase"].items():
            old = previous["by_phase"][phase]
            old_values = (
                float(old["flow_matching_mse"]),
                float(old["raw_7d_action_error_full_10_step_chunk"]["mae_all_7d"]),
            )
            new_values = (
                float(metrics["flow_matching_mse"]),
                float(metrics["raw_7d_action_error_full_10_step_chunk"]["mae_all_7d"]),
            )
            if any(new > max(1e-8, 10.0 * old) for old, new in zip(old_values, new_values)):
                return True
        return False

    def _evaluate_fixed_fit(self) -> dict[str, Any]:
        self.model.eval()
        torch.manual_seed(self._flow_eval_seed)
        torch.cuda.manual_seed_all(self._flow_eval_seed)
        loss_sum = 0.0
        examples = 0
        phase_losses = {name: [] for name in self._fixed_eval_phase_indices}
        phase_actions = {name: {"pred": [], "target": []} for name in phase_losses}
        rng = np.random.default_rng(self._action_eval_seed)
        fixed_noise = rng.standard_normal((48, 10, 32), dtype=np.float32)
        with torch.no_grad():
            eval_torch_loader = self.eval_data_loader._data_loader.torch_loader
            if len(self._fixed_eval_phase_sequence) != len(eval_torch_loader):
                raise RuntimeError("fixed_phase_eval_order_does_not_match_loader_length")
            if len(eval_torch_loader) != 48:
                raise RuntimeError(
                    f"fixed_eval_expected_48_samples_got_{len(eval_torch_loader)}"
                )
            module = self.model.module
            for eval_index, batch in enumerate(
                itertools.islice(self.eval_data_loader, len(eval_torch_loader))
            ):
                # Call through the FSDP root for the native loss forward so its
                # lazy initialization and mixed-precision hooks stay active.
                output = self.model(forward_type=ForwardType.SFT, data=batch)
                loss = output["loss"] if isinstance(output, dict) else output
                raw_observation, raw_actions = module._unpack_sft_batch(batch)
                observation = module._observation_to_device(raw_observation)
                actions = module._actions_to_device(raw_actions)
                noise = torch.as_tensor(
                    fixed_noise[eval_index : eval_index + 1], device=self.device
                )
                predicted_model_actions = module.sample_actions(
                    observation,
                    num_steps=self._action_eval_solver_steps,
                    noise=noise,
                )
                predicted_raw_actions = self._decode_raw_actions(
                    predicted_model_actions, observation.state
                )
                target_raw_actions = self._decode_raw_actions(actions, observation.state)
                batch_size = int(actions.shape[0])
                if batch_size != 1:
                    raise ValueError("phase-specific fixed evaluation requires eval_batch_size=1")
                phase = self._fixed_eval_phase_sequence[eval_index]
                step_loss = float(loss.detach().float().item())
                loss_sum += step_loss * batch_size
                examples += batch_size
                phase_losses[phase].append(step_loss)
                predicted_np = predicted_raw_actions.detach().float().cpu().numpy()
                target_np = target_raw_actions.detach().float().cpu().numpy()
                if predicted_np.shape != (1, 10, 7) or target_np.shape != (1, 10, 7):
                    raise RuntimeError(
                        "raw_action_eval_shape_mismatch:"
                        f"pred={predicted_np.shape}:target={target_np.shape}"
                    )
                phase_actions[phase]["pred"].append(predicted_np[0])
                phase_actions[phase]["target"].append(target_np[0])
        self.model.train()
        self.model.module.img.eval()
        if examples == 0:
            raise RuntimeError("fixed evaluation subset is empty")
        mean_loss = loss_sum / examples

        def action_metrics(predictions: list[np.ndarray], targets: list[np.ndarray]):
            pred = np.asarray(predictions, dtype=np.float64)
            target = np.asarray(targets, dtype=np.float64)
            error = pred - target
            return {
                "sample_count": int(error.shape[0]),
                "action_horizon": int(error.shape[1]),
                "raw_action_dim": int(error.shape[2]),
                "mae_all_7d": float(np.abs(error).mean()),
                "rmse_all_7d": float(np.sqrt(np.square(error).mean())),
                "mae_by_dim": np.abs(error).mean(axis=(0, 1)).tolist(),
                "rmse_by_dim": np.sqrt(np.square(error).mean(axis=(0, 1))).tolist(),
            }

        by_phase = {}
        for name, losses in phase_losses.items():
            predictions = phase_actions[name]["pred"]
            targets = phase_actions[name]["target"]
            by_phase[name] = {
                "flow_matching_mse": float(np.mean(losses)),
                "flow_matching_rmse_normalized_model_space": float(
                    np.sqrt(np.mean(losses))
                ),
                "examples": len(losses),
                "raw_7d_action_error_full_10_step_chunk": action_metrics(
                    predictions, targets
                ),
                "raw_7d_action_error_first_action": action_metrics(
                    [item[:1] for item in predictions],
                    [item[:1] for item in targets],
                ),
            }
        return {
            "flow_matching_mse": mean_loss,
            "flow_matching_rmse_normalized_model_space": float(np.sqrt(mean_loss)),
            "examples": examples,
            "flow_matching_seed": self._flow_eval_seed,
            "fixed_action_noise_seed": self._action_eval_seed,
            "action_solver_steps": self._action_eval_solver_steps,
            "action_error_target": "official OpenPI loader targets decoded by the original pi05_libero output transform to raw environment 7D actions",
            "by_phase": by_phase,
        }

    def _decode_raw_actions(
        self, model_actions: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        from rlinf.models.embodiment.openpi.transforms.apply import (
            apply_output_transform,
        )

        decoded = apply_output_transform(
            self._eval_output_transform,
            {"actions": model_actions, "state": state},
            action_chunk=int(self.cfg.actor.model.num_action_chunks),
        )["actions"]
        return decoded.to(device=self.device, dtype=torch.float32)

    def _evaluate_carry_progress(
        self, previous: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any]:
        phase = "bspline_osc_carry"
        previous_phase = previous["by_phase"][phase]
        current_phase = current["by_phase"][phase]
        previous_mse = float(previous_phase["flow_matching_mse"])
        current_mse = float(current_phase["flow_matching_mse"])
        previous_mae = float(
            previous_phase["raw_7d_action_error_full_10_step_chunk"]["mae_all_7d"]
        )
        current_mae = float(
            current_phase["raw_7d_action_error_full_10_step_chunk"]["mae_all_7d"]
        )
        values = (previous_mse, current_mse, previous_mae, current_mae)
        if not all(np.isfinite(value) for value in values):
            return {
                "stop_training": True,
                "reason": "nonfinite_carry_fit_metric",
                "relative_flow_mse_reduction": None,
                "relative_raw_7d_chunk_mae_reduction": None,
            }
        mse_reduction = (previous_mse - current_mse) / max(abs(previous_mse), 1e-12)
        mae_reduction = (previous_mae - current_mae) / max(abs(previous_mae), 1e-12)
        regression = max(-mse_reduction, -mae_reduction)
        if regression > self._regression_stop_relative_threshold:
            stop, reason = True, "carry_fit_metric_regression_exceeded_stop_threshold"
        elif (
            mse_reduction < self._plateau_relative_threshold
            and mae_reduction < self._plateau_relative_threshold
        ):
            stop, reason = True, "carry_flow_mse_and_raw_7d_action_error_improvement_stalled"
        else:
            stop, reason = False, "carry_fit_still_improving"
        return {
            "stop_training": stop,
            "reason": reason,
            "previous_flow_matching_mse": previous_mse,
            "current_flow_matching_mse": current_mse,
            "relative_flow_mse_reduction": float(mse_reduction),
            "previous_raw_7d_chunk_mae": previous_mae,
            "current_raw_7d_chunk_mae": current_mae,
            "relative_raw_7d_chunk_mae_reduction": float(mae_reduction),
            "plateau_relative_threshold": self._plateau_relative_threshold,
            "regression_stop_relative_threshold": self._regression_stop_relative_threshold,
        }

    def _trainable_parameter_signature(self) -> dict[str, Any]:
        module = self.model.module
        for name, parameter in module.named_parameters():
            if parameter.requires_grad and parameter.numel() >= 16:
                values = parameter.detach().float().reshape(-1)[:4096].cpu()
                signature = float(values.sum().item() + values.square().sum().item())
                return {"name": name, "signature": signature}
        raise RuntimeError("no trainable Action Expert parameter was found")

    def _write_report(self, report: dict[str, Any]) -> None:
        destination = self._report_dir / "report.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, destination)


@hydra.main(
    version_base="1.1",
    config_path="../../../examples/sft/config",
    config_name="gs_opd_single_episode_pi05",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    logging.info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_placement = component_placement.get_strategy("actor")
    actor_group = SingleEpisodeSftWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    runner = SFTRunner(cfg=cfg, actor=actor_group)
    runner.init_workers()
    runner.run()

    report_dir = Path(cfg.runner.logger.log_path) / str(
        cfg.runner.logger.experiment_name
    )
    report_path = report_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        step_rows = [
            json.loads(line)
            for line in (report_dir / "step_metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        epochs: dict[int, list[float]] = {}
        for row in step_rows:
            epochs.setdefault(int(row["epoch"]), []).append(float(row["loss"]))
        completed_steps = len(step_rows)
        report["checkpoint_path"] = str(
            report_dir
            / "checkpoints"
            / f"global_step_{completed_steps}"
            / "actor"
        )
        report["loss_curve_by_epoch"] = [
            {
                "epoch": epoch,
                "optimizer_steps": len(losses),
                "mean_flow_matching_loss": float(np.mean(losses)),
                "first_step_loss": float(losses[0]),
                "last_step_loss": float(losses[-1]),
            }
            for epoch, losses in sorted(epochs.items())
        ]
        report["completed_optimizer_steps"] = completed_steps
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, loss_axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
        steps = [int(row["step"]) for row in step_rows]
        losses = [float(row["loss"]) for row in step_rows]
        loss_axis.plot(steps, losses, linewidth=0.7, alpha=0.65, label="optimizer-step Flow MSE")
        loss_axis.set_xlabel("Optimizer step")
        loss_axis.set_ylabel("Flow Matching loss")
        loss_axis.grid(True, alpha=0.25)
        lr_axis = loss_axis.twinx()
        learning_rates = [float(row["learning_rate"]) for row in step_rows]
        lr_axis.plot(steps, learning_rates, color="tab:orange", linewidth=1.0, label="learning rate")
        lr_axis.set_ylabel("Learning rate")
        for epoch in sorted(epochs):
            loss_axis.axvline(epoch * int(report["steps_per_epoch"]), color="gray", linestyle="--", linewidth=0.7)
        handles_a, labels_a = loss_axis.get_legend_handles_labels()
        handles_b, labels_b = lr_axis.get_legend_handles_labels()
        loss_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="upper right")
        figure.savefig(report_dir / "loss_curve.png", dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    main()
