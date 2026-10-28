# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Validate successful L0T0 traces as structural, non-replay references."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SuccessReference:
    """Identity and verified task outcome for one archived physical episode."""

    reference_id: str
    trace_path: Path
    trace_sha256: str
    report_path: Path | None
    report_sha256: str | None
    metadata: dict[str, Any]
    native_task_success: bool
    final_reward: float
    final_done: bool
    route_prior: dict[str, Any] | None
    source_limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary without exposing source action arrays."""
        return {
            "reference_id": self.reference_id,
            "trace_path": str(self.trace_path),
            "trace_sha256": self.trace_sha256,
            "report_path": str(self.report_path) if self.report_path else None,
            "report_sha256": self.report_sha256,
            "task": {
                "suite": self.metadata.get("suite"),
                "level": self.metadata.get("level"),
                "task_id": self.metadata.get("task_id"),
                "initial_state_index": self.metadata.get("initial_state_index"),
                "instruction": self.metadata.get("instruction"),
                "steps": self.metadata.get("steps"),
                "action_dim": self.metadata.get("action_dim"),
            },
            "checkpoint_path": self.metadata.get("checkpoint_path"),
            "lora_checkpoint_path": self.metadata.get("lora_checkpoint_path"),
            "native_task_success": self.native_task_success,
            "final_reward": self.final_reward,
            "final_done": self.final_done,
            "route_prior": self.route_prior,
            "historical_actions_replayed": False,
            "stable_release_verified": False,
            "strict_10mm_pass": False,
            "complete_substep_monitoring": False,
            "safe_training_eligible": False,
            "source_limitations": list(self.source_limitations),
        }


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_from_npz(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item()
    if not isinstance(value, str):
        raise ValueError("success_reference_metadata_json_must_be_a_string")
    metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise ValueError("success_reference_metadata_json_must_decode_to_an_object")
    return metadata


def load_success_reference(
    trace_path: Path,
    report_path: Path | None = None,
    *,
    expected_suite: str = "obstacle_avoidance",
    expected_level: int = 0,
    expected_task_id: int = 0,
    expected_initial_state_index: int = 0,
) -> SuccessReference:
    """Verify an executed 7D episode and preserve its provenance boundaries."""
    trace_path = Path(trace_path).resolve()
    report_path = Path(report_path).resolve() if report_path is not None else None
    if not trace_path.is_file():
        raise FileNotFoundError(f"success_reference_trace_missing:{trace_path}")
    if report_path is not None and not report_path.is_file():
        raise FileNotFoundError(f"success_reference_report_missing:{report_path}")

    with np.load(trace_path, allow_pickle=False) as archive:
        required = {"metadata_json", "executed_actions", "step_rewards", "step_dones"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"success_reference_npz_missing_arrays:{','.join(missing)}")
        metadata = _metadata_from_npz(archive["metadata_json"])
        actions = np.asarray(archive["executed_actions"], dtype=np.float64)
        rewards = np.asarray(archive["step_rewards"], dtype=np.float64).reshape(-1)
        dones = np.asarray(archive["step_dones"], dtype=bool).reshape(-1)

    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("success_reference_actions_must_be_finite_Nx7")
    if not (len(actions) == len(rewards) == len(dones)):
        raise ValueError("success_reference_action_reward_done_lengths_differ")
    if not len(actions):
        raise ValueError("success_reference_episode_is_empty")
    expected_identity = {
        "suite": expected_suite,
        "level": expected_level,
        "task_id": expected_task_id,
        "initial_state_index": expected_initial_state_index,
    }
    for field, expected in expected_identity.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"success_reference_{field}_mismatch:"
                f"expected={expected!r}:found={metadata.get(field)!r}"
            )
    if metadata.get("action_dim") != 7 or metadata.get("steps") != len(actions):
        raise ValueError("success_reference_metadata_action_contract_mismatch")
    if metadata.get("transition_count") != len(actions):
        raise ValueError("success_reference_transition_count_mismatch")

    final_reward = float(rewards[-1])
    final_done = bool(dones[-1])
    if not np.isfinite(rewards).all() or not final_done or final_reward <= 0.0:
        raise ValueError("success_reference_has_no_terminal_native_success_evidence")

    report: dict[str, Any] = {}
    if report_path is not None:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        episode = report.get("episode_rollout")
        if not isinstance(episode, dict):
            raise ValueError("success_reference_run_report_has_no_episode_rollout")
        if (
            episode.get("success") is not True
            or episode.get("done") is not True
            or episode.get("executed_steps") != len(actions)
        ):
            raise ValueError("success_reference_run_report_disagrees_with_npz_terminal_state")

    route = metadata.get("direct_bspline_osc_execution")
    route_prior = None
    limitations = [
        "source safety is not transferable to a new scene",
        "source does not establish a stable released-object episode",
        "historical actions are never replayed by this reference interface",
    ]
    if isinstance(route, dict):
        direction = route.get("route_direction")
        speed = route.get("reference_speed_m_s")
        if (
            route.get("source") != "direct_b_spline_closed_loop_native_osc_pose"
            or direction not in {"left", "right", "above"}
            or not isinstance(speed, (int, float))
            or not np.isfinite(float(speed))
            or float(speed) <= 0.0
        ):
            raise ValueError("success_reference_direct_route_metadata_is_invalid")
        route_report_steps = report.get("real_execution_steps")
        historical_steps = route.get("direct_osc_steps", route_report_steps)
        if not isinstance(historical_steps, int) or historical_steps <= 0:
            raise ValueError("success_reference_route_step_count_is_missing")
        route_prior = {
            "route_family": "current_scene_b_spline_then_native_osc_pose",
            "preferred_direction": direction,
            "reference_speed_m_s": float(speed),
            "historical_route_steps": historical_steps,
            "retarget_from_live_state": True,
            "historical_actions_replayed": False,
        }
        if report:
            route_report = report.get("execution_summary", {})
            if (
                report.get("route_completed") is not True
                or report.get("real_execution_steps") != route_prior["historical_route_steps"]
                or route_report.get("actual_osc_steps") != route_prior["historical_route_steps"]
            ):
                raise ValueError("success_reference_run_report_disagrees_with_route_metadata")
    if report.get("physical_safety_claim") is not True:
        limitations.append("source report does not certify physical or strict 10 mm safety")

    return SuccessReference(
        reference_id=(
            f"{expected_suite}/L{expected_level}T{expected_task_id}/"
            f"state{expected_initial_state_index}/{trace_path.parent.name}"
        ),
        trace_path=trace_path,
        trace_sha256=sha256_file(trace_path),
        report_path=report_path,
        report_sha256=sha256_file(report_path) if report_path is not None else None,
        metadata=metadata,
        native_task_success=True,
        final_reward=final_reward,
        final_done=final_done,
        route_prior=route_prior,
        source_limitations=tuple(limitations),
    )
