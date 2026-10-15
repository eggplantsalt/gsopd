"""Run a single-object text-prompted SAM 3 image segmentation smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


def _load_processor(checkpoint_path: Path) -> Any:
    """Load the official SAM 3 image model from a local checkpoint only."""
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        device=device,
    )
    return Sam3Processor(model, device=device)


def _to_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
        import torch

        if value.dtype == torch.bfloat16:
            value = value.to(dtype=torch.float32)
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _normalize_predictions(output: dict[str, Any], image_size: tuple[int, int]):
    """Return masks [N,H,W], boxes [N,4], and scores [N] as NumPy arrays."""
    import numpy as np

    for key in ("masks", "boxes", "scores"):
        if key not in output:
            raise RuntimeError(f"SAM 3 output is missing {key!r}")

    masks = _to_numpy(output["masks"])
    boxes = _to_numpy(output["boxes"])
    scores = _to_numpy(output["scores"])

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise RuntimeError(f"Unexpected SAM 3 masks shape: {masks.shape}")

    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    elif boxes.ndim == 1:
        boxes = boxes[None, :]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise RuntimeError(f"Unexpected SAM 3 boxes shape: {boxes.shape}")
    scores = scores.reshape(-1)

    count = masks.shape[0]
    if boxes.shape[0] != count or scores.shape[0] != count:
        raise RuntimeError(
            "SAM 3 candidate count differs across masks, boxes, and scores: "
            f"{count}, {boxes.shape[0]}, {scores.shape[0]}"
        )
    expected_size = (image_size[1], image_size[0])
    if masks.shape[1:] != expected_size:
        raise RuntimeError(
            f"SAM 3 masks have size {masks.shape[1:]}, expected {expected_size}"
        )
    return masks > 0.5, boxes, scores


def _save_results(
    image: Any,
    masks: Any,
    boxes: Any,
    scores: Any,
    output_dir: Path,
    teacher_json: Path,
    image_path: Path,
    object_id: str,
    object_name: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    candidates = []

    for index, mask in enumerate(masks):
        binary_mask = np.asarray(mask, dtype=bool)
        if binary_mask.shape != (height, width):
            raise RuntimeError(
                f"Candidate {index} mask shape {binary_mask.shape} does not match "
                f"image size {(height, width)}"
            )

        candidate_name = f"candidate_{index:03d}"
        mask_file = f"{candidate_name}_mask.png"
        overlay_file = f"{candidate_name}_overlay.png"
        Image.fromarray(binary_mask.astype(np.uint8) * 255).save(
            output_dir / mask_file
        )

        overlay = image_rgb.copy()
        tint = np.asarray([255, 64, 64], dtype=np.float32)
        overlay[binary_mask] = (
            0.55 * overlay[binary_mask].astype(np.float32)
            + 0.45 * tint
        ).astype(np.uint8)
        Image.fromarray(overlay).save(output_dir / overlay_file)

        candidates.append(
            {
                "candidate_index": index,
                "score": float(scores[index]),
                "box_xyxy_pixels": [float(value) for value in boxes[index]],
                "mask_file": mask_file,
                "overlay_file": overlay_file,
                "mask_area_pixels": int(binary_mask.sum()),
            }
        )

    result = {
        "status": "ok" if candidates else "no_candidates",
        "teacher_json": str(teacher_json.resolve()),
        "object_id": object_id,
        "object_name": object_name,
        "text_prompt": object_name,
        "image": str(image_path.resolve()),
        "image_width": width,
        "image_height": height,
        "checkpoint": str(checkpoint_path.resolve()),
        "candidate_count": len(candidates),
        "selected_candidate": None,
        "candidates": candidates,
    }
    (output_dir / "sam3_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_smoke(
    teacher_json: Path,
    object_id: str,
    image_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    teacher_output = parse_teacher_output_v2(
        teacher_json.read_text(encoding="utf-8")
    )
    matched = next(
        (obj for obj in teacher_output["objects"] if obj["id"] == object_id), None
    )
    if matched is None:
        known_ids = [obj["id"] for obj in teacher_output["objects"]]
        raise ValueError(
            f"object-id {object_id!r} was not found in Teacher JSON; "
            f"available IDs: {known_ids}"
        )

    object_name = matched["name"]
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    processor = _load_processor(checkpoint_path)
    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(image)
                # Only the Teacher object's semantic name is sent to SAM 3.
                output = processor.set_text_prompt(
                    state=state, prompt=object_name
                )
        else:
            state = processor.set_image(image)
            # Only the Teacher object's semantic name is sent to SAM 3.
            output = processor.set_text_prompt(state=state, prompt=object_name)
    masks, boxes, scores = _normalize_predictions(output, image.size)
    return _save_results(
        image=image,
        masks=masks,
        boxes=boxes,
        scores=scores,
        output_dir=output_dir,
        teacher_json=teacher_json,
        image_path=image_path,
        object_id=object_id,
        object_name=object_name,
        checkpoint_path=checkpoint_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-json", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    for label, path in (
        ("Teacher JSON", args.teacher_json),
        ("image", args.image),
        ("SAM 3 checkpoint", args.checkpoint),
    ):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")

    try:
        result = run_smoke(
            teacher_json=args.teacher_json,
            object_id=args.object_id,
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            output_dir=args.out_dir,
        )
    except Exception as exc:
        print(f"SAM 3 Smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        f"SAM 3 result: status={result['status']}, "
        f"object_id={result['object_id']!r}, prompt={result['text_prompt']!r}, "
        f"candidates={result['candidate_count']}"
    )
    if not result["candidates"]:
        print("No candidates were returned; no instance was selected.")
    for candidate in result["candidates"]:
        print(
            f"  candidate {candidate['candidate_index']}: "
            f"score={candidate['score']:.4f}, "
            f"box={candidate['box_xyxy_pixels']}, "
            f"mask_area={candidate['mask_area_pixels']} px"
        )
    print(f"Results saved under: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
