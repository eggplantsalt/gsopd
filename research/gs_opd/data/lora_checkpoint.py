"""Save and load standalone RLinf OpenPI Action Expert LoRA checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

FORMAT = "rlinf_openpi_action_expert_lora_v1"


def save_lora_adapter(
    state_dict: dict[str, torch.Tensor],
    output_path: Path,
    *,
    base_checkpoint: str,
    rank: int,
    training_checkpoint: str,
) -> dict[str, Any]:
    adapter = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
        if "lora_" in key
    }
    if not adapter:
        raise ValueError("checkpoint_contains_no_lora_tensors")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": FORMAT,
        "base_checkpoint": str(base_checkpoint),
        "training_checkpoint": str(training_checkpoint),
        "rank": int(rank),
        "tensor_count": len(adapter),
        "parameter_count": sum(value.numel() for value in adapter.values()),
        "tensor_names": sorted(adapter),
    }
    torch.save({"metadata": metadata, "state_dict": adapter}, output_path)
    return metadata


def load_lora_adapter(model, adapter_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(adapter_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("metadata", {}).get("format") != FORMAT:
        raise ValueError("unsupported_openpi_lora_checkpoint_format")
    adapter = checkpoint.get("state_dict")
    if not isinstance(adapter, dict) or not adapter:
        raise ValueError("openpi_lora_checkpoint_has_no_state_dict")

    expected = {key for key in model.state_dict() if "lora_" in key}
    provided = set(adapter)
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if missing or unexpected:
        raise ValueError(
            f"openpi_lora_tensor_coverage_mismatch:missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    incompatible = model.load_state_dict(adapter, strict=False)
    adapter_missing = sorted(key for key in incompatible.missing_keys if "lora_" in key)
    if adapter_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "openpi_lora_load_failed: "
            f"missing={adapter_missing[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    return checkpoint["metadata"]
