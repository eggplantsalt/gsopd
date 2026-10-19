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
    """Keep OpenPI transforms, but select only complete action windows."""

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
                f"expected the preserved 749-frame episode, found {frame_count} frames"
            )
        meta = getattr(raw, "meta", None)
        episodes = getattr(meta, "episodes", None)
        if episodes is not None and len(episodes) != 1:
            raise ValueError(f"expected one episode, found {len(episodes)}")

        valid_count = frame_count - horizon + 1
        if valid_count <= 0:
            raise ValueError("episode is shorter than the OpenPI action horizon")
        if eval_dataset:
            eval_count = int(self.cfg.data.eval_window_count)
            indices = np.linspace(0, valid_count - 1, eval_count, dtype=np.int64)
            if len(np.unique(indices)) != eval_count:
                raise ValueError("fixed evaluation indices are not unique")
            selected = indices.tolist()
        else:
            selected = list(range(valid_count))

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

        if not eval_dataset:
            self._single_episode_window_count = valid_count
            self._fixed_eval_window_indices = np.linspace(
                0, valid_count - 1, int(self.cfg.data.eval_window_count), dtype=np.int64
            ).tolist()
        return loader, data_config

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is None:
            return super().model_provider_func()
        frozen_tensors = model.freeze_vlm()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logging.info(
            "GS-OPD single-episode SFT: froze %d VLM parameter tensors; "
            "trainable=%d / total=%d parameters",
            frozen_tensors,
            trainable,
            total,
        )
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
            "valid_training_windows": int(self._single_episode_window_count),
            "training_batch_size": int(self.cfg.actor.micro_batch_size),
            "steps_per_epoch": int(self._steps_per_epoch),
            "epochs": int(self.cfg.runner.max_epochs),
            "total_optimizer_steps": int(self._expected_total_steps),
            "optimizer_schedule_steps": configured_total_steps,
            "fixed_eval_indices": self._fixed_eval_window_indices,
            "fixed_eval_seed": self._eval_seed,
            "base_checkpoint": str(self.cfg.actor.model.model_path),
            "norm_stats_path": str(self.cfg.actor.model.openpi_data.norm_stats_path),
            "frozen_vlm_parameter_tensors": int(
                getattr(self, "_pre_fsdp_frozen_tensor_count", 0)
            ),
            "trainable_parameters": int(self._pre_fsdp_trainable_parameters),
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

    def _evaluate_fixed_flow_loss(self) -> dict[str, float]:
        self.model.eval()
        torch.manual_seed(self._eval_seed)
        torch.cuda.manual_seed_all(self._eval_seed)
        loss_sum = 0.0
        examples = 0
        with torch.no_grad():
            eval_torch_loader = self.eval_data_loader._data_loader.torch_loader
            for batch in itertools.islice(
                self.eval_data_loader, len(eval_torch_loader)
            ):
                with self.amp_context:
                    output = self.model(forward_type=ForwardType.SFT, data=batch)
                loss = output["loss"] if isinstance(output, dict) else output
                batch_size = int(batch[1].shape[0])
                loss_sum += float(loss.detach().float().item()) * batch_size
                examples += batch_size
        self.model.train()
        if examples == 0:
            raise RuntimeError("fixed evaluation subset is empty")
        mean_loss = loss_sum / examples
        return {
            "flow_matching_mse": mean_loss,
            "flow_matching_rmse_normalized_model_space": float(np.sqrt(mean_loss)),
            "examples": examples,
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
