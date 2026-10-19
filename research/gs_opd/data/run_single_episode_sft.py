#!/usr/bin/env python3
"""Run the official RLinf OpenPI SFT Runner on one complete LeRobot episode."""

from __future__ import annotations

import json
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

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is None:
            return super().model_provider_func()
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
        self._pre_fsdp_trainable_parameters = trainable
        self._pre_fsdp_frozen_tensor_count = frozen_tensors
        return model

    def init_worker(self):
        super().init_worker()
        self._steps_per_epoch = self.get_max_steps_per_epoch()
        self._expected_total_steps = self._steps_per_epoch * int(
            self.cfg.runner.max_epochs
        )
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
        self._eval_seed = 20261019

        baseline = self._evaluate_fixed_flow_loss()
        initial_parameter = self._trainable_parameter_signature()
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
            "fixed_eval_seed": self._eval_seed,
            "base_checkpoint": str(self.cfg.actor.model.model_path),
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
        step_record = {
            "step": completed_step,
            "epoch": int((completed_step - 1) // self._steps_per_epoch + 1),
            **{key: float(value) for key, value in metrics.items()},
            "elapsed_seconds": float(time.monotonic() - self._training_started),
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
            "gpu_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        with self._step_metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(step_record, ensure_ascii=False) + "\n")

        if completed_step == self._expected_total_steps:
            final_fit = self._evaluate_fixed_flow_loss()
            final_parameter = self._trainable_parameter_signature()
            final = json.loads((self._report_dir / "report.json").read_text())
            final.update(
                {
                    "fixed_eval_after": final_fit,
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
                        / f"global_step_{self._expected_total_steps}"
                        / "actor"
                    ),
                }
            )
            self._write_report(final)
        return metrics

    def _evaluate_fixed_flow_loss(self) -> dict[str, Any]:
        self.model.eval()
        torch.manual_seed(self._eval_seed)
        torch.cuda.manual_seed_all(self._eval_seed)
        loss_sum = 0.0
        examples = 0
        phase_loss_sums = {
            name: {"loss_sum": 0.0, "examples": 0}
            for name in ("student_grasp", "bspline_osc_carry", "student_place")
        }
        with torch.no_grad():
            eval_torch_loader = self.eval_data_loader._data_loader.torch_loader
            if len(self._fixed_eval_phase_sequence) != len(eval_torch_loader):
                raise RuntimeError("fixed_phase_eval_order_does_not_match_loader_length")
            for eval_index, batch in enumerate(
                itertools.islice(self.eval_data_loader, len(eval_torch_loader))
            ):
                with self.amp_context:
                    output = self.model(forward_type=ForwardType.SFT, data=batch)
                loss = output["loss"] if isinstance(output, dict) else output
                batch_size = int(batch[1].shape[0])
                if batch_size != 1:
                    raise ValueError("phase-specific fixed evaluation requires eval_batch_size=1")
                phase = self._fixed_eval_phase_sequence[eval_index]
                step_loss = float(loss.detach().float().item())
                loss_sum += step_loss * batch_size
                examples += batch_size
                phase_loss_sums[phase]["loss_sum"] += step_loss
                phase_loss_sums[phase]["examples"] += batch_size
        self.model.train()
        if examples == 0:
            raise RuntimeError("fixed evaluation subset is empty")
        mean_loss = loss_sum / examples
        return {
            "flow_matching_mse": mean_loss,
            "flow_matching_rmse_normalized_model_space": float(np.sqrt(mean_loss)),
            "examples": examples,
            "by_phase": {
                name: {
                    "flow_matching_mse": values["loss_sum"] / values["examples"],
                    "flow_matching_rmse_normalized_model_space": float(
                        np.sqrt(values["loss_sum"] / values["examples"])
                    ),
                    "examples": int(values["examples"]),
                }
                for name, values in phase_loss_sums.items()
            },
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
        report["completed_optimizer_steps"] = len(step_rows)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
