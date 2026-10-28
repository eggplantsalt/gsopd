"""Match SAM 3 candidate masks to visible MuJoCo Body groups by pixel IoU."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} JSON must contain an object")
    return value


def _resolve_recorded_file(base_dir: Path, metadata: dict, field: str) -> Path:
    name = metadata.get(field)
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"Geom metadata is missing {field!r}")
    path = Path(name)
    return path if path.is_absolute() else base_dir / path


def _check_forward_direction(metadata: dict, filename_field: str) -> str:
    filename = metadata.get(filename_field)
    if not isinstance(filename, str) or not filename:
        raise RuntimeError(f"Geom metadata is missing {filename_field!r}")
    directions = metadata.get("coordinate_directions")
    direction = directions.get(filename) if isinstance(directions, dict) else None
    if not isinstance(direction, str):
        raise RuntimeError(
            f"Geom metadata has no coordinate direction for {filename!r}"
        )
    normalized = direction.lower()
    identity_alignment = (
        metadata.get("pixel_alignment_transform") == "identity"
        and metadata.get("rgb_segmentation_pixel_consistency") is True
        and metadata.get("physical_state_unchanged") is True
        and "identity" in normalized and "forward" in normalized
    )
    rotated_alignment = (metadata.get("pixel_alignment_transform") != "identity"
                         and "180" in normalized and "forward" in normalized)
    if not rotated_alignment and not identity_alignment:
        raise RuntimeError(
            f"{filename!r} is not marked as the 180-degree forward orientation: "
            f"{direction!r}"
        )
    return filename


def _load_rgb(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(f"Cannot load {label} image {path}: {exc}") from exc


def _load_inputs(
    sam3_results_path: Path,
    geom_ids_path: Path,
    geom_metadata_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    sam_results = _load_json(sam3_results_path, "SAM 3 results")
    geom_metadata = _load_json(geom_metadata_path, "Geom metadata")

    identity = geom_metadata.get("pixel_alignment_transform") == "identity"
    rgb_field = "rgb_forward_file" if identity else "rgb_rotated_180_file"
    geom_field = "geom_ids_forward_file" if identity else "geom_ids_rotated_180_file"
    rgb_filename = _check_forward_direction(geom_metadata, rgb_field)
    _check_forward_direction(geom_metadata, geom_field)
    recorded_geom_ids_path = _resolve_recorded_file(
        geom_metadata_path.parent, geom_metadata, geom_field
    ).resolve()
    if geom_ids_path.resolve() != recorded_geom_ids_path:
        raise RuntimeError(
            "Provided Geom ID map does not match the rotated map recorded in "
            f"metadata: expected {recorded_geom_ids_path}, got {geom_ids_path.resolve()}"
        )

    rgb_path = _resolve_recorded_file(
        geom_metadata_path.parent, geom_metadata, rgb_field
    )
    geom_rgb = _load_rgb(rgb_path, "forward Geom capture RGB")
    sam_image_value = sam_results.get("image")
    if not isinstance(sam_image_value, str) or not sam_image_value:
        raise RuntimeError("SAM 3 results are missing the actual input image path")
    sam_image_path = Path(sam_image_value)
    if not sam_image_path.is_absolute():
        sam_image_path = sam3_results_path.parent / sam_image_path
    sam_rgb = _load_rgb(sam_image_path, "SAM 3 input RGB")

    if geom_rgb.shape != sam_rgb.shape:
        raise RuntimeError(
            "Image coordinate check failed: SAM 3 image shape "
            f"{sam_rgb.shape} differs from forward Geom RGB shape {geom_rgb.shape}"
        )
    if not np.array_equal(geom_rgb, sam_rgb):
        raise RuntimeError(
            "Image coordinate check failed: SAM 3 input RGB is not pixel-identical "
            "to the 180-degree forward RGB recorded with the Geom ID map"
        )

    height, width = geom_rgb.shape[:2]
    if (
        sam_results.get("image_width") != width
        or sam_results.get("image_height") != height
        or geom_metadata.get("width") != width
        or geom_metadata.get("height") != height
    ):
        raise RuntimeError(
            "Image dimensions do not agree across SAM 3 results and Geom metadata"
        )

    try:
        geom_ids = np.load(geom_ids_path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Cannot load Geom ID array {geom_ids_path}: {exc}") from exc
    if geom_ids.ndim != 2 or not np.issubdtype(geom_ids.dtype, np.integer):
        raise RuntimeError(
            f"Geom ID map must be a 2D integer array, got "
            f"shape={geom_ids.shape}, dtype={geom_ids.dtype}"
        )
    if geom_ids.shape != (height, width):
        raise RuntimeError(
            f"Geom ID map shape {geom_ids.shape} differs from RGB size {(height, width)}"
        )
    if np.any(geom_ids < -2):
        raise RuntimeError("Geom ID map contains unsupported negative labels")

    model_geom_count = geom_metadata.get("model_geom_count")
    if isinstance(model_geom_count, bool) or not isinstance(model_geom_count, int):
        raise RuntimeError("Geom metadata has an invalid model_geom_count")
    if np.any(geom_ids >= model_geom_count):
        raise RuntimeError("Geom ID map contains an ID outside model_geom_count")

    raw_visible_geoms = geom_metadata.get("visible_geoms")
    if not isinstance(raw_visible_geoms, list):
        raise RuntimeError("Geom metadata is missing visible_geoms")
    visible_geoms = []
    seen_geom_ids = set()
    for item in raw_visible_geoms:
        if not isinstance(item, dict):
            raise RuntimeError("Each visible Geom record must be an object")
        geom_id = item.get("geom_id")
        body_id = item.get("body_id")
        if (
            isinstance(geom_id, bool)
            or not isinstance(geom_id, int)
            or not 0 <= geom_id < model_geom_count
        ):
            raise RuntimeError(f"Invalid visible Geom ID: {geom_id!r}")
        if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id < 0:
            raise RuntimeError(
                f"Geom {geom_id} has no reliable owning Body ID: {body_id!r}"
            )
        if geom_id in seen_geom_ids:
            raise RuntimeError(f"Duplicate visible Geom metadata for ID {geom_id}")
        seen_geom_ids.add(geom_id)
        pixel_count = int(np.count_nonzero(geom_ids == geom_id))
        if pixel_count == 0:
            raise RuntimeError(f"Geom {geom_id} is listed as visible but has no pixels")
        if item.get("pixel_count") != pixel_count:
            raise RuntimeError(
                f"Geom {geom_id} pixel count disagrees with metadata: "
                f"map={pixel_count}, metadata={item.get('pixel_count')!r}"
            )
        visible_geoms.append(
            {
                "geom_id": geom_id,
                "geom_name": item.get("geom_name"),
                "body_id": body_id,
                "body_name": item.get("body_name"),
            }
        )

    map_geom_ids = set(int(value) for value in np.unique(geom_ids) if value >= 0)
    if map_geom_ids != seen_geom_ids:
        raise RuntimeError(
            "Visible Geom metadata does not cover exactly the IDs in the map: "
            f"map={sorted(map_geom_ids)}, metadata={sorted(seen_geom_ids)}"
        )

    raw_candidates = sam_results.get("candidates")
    if not isinstance(raw_candidates, list):
        raise RuntimeError("SAM 3 results are missing the candidates list")
    candidates = []
    seen_candidate_ids = set()
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            raise RuntimeError(f"Candidate {index} must be an object")
        candidate_index = item.get("candidate_index")
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise RuntimeError(f"Candidate {index} has an invalid candidate_index")
        if candidate_index in seen_candidate_ids:
            raise RuntimeError(f"Duplicate candidate_index {candidate_index}")
        seen_candidate_ids.add(candidate_index)
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise RuntimeError(f"Candidate {candidate_index} has an invalid SAM score")
        mask_value = item.get("mask_file")
        if not isinstance(mask_value, str) or not mask_value:
            raise RuntimeError(f"Candidate {candidate_index} has no mask_file")
        mask_path = Path(mask_value)
        if not mask_path.is_absolute():
            mask_path = sam3_results_path.parent / mask_path
        try:
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L")) > 0
        except Exception as exc:
            raise RuntimeError(
                f"Cannot load Candidate {candidate_index} Mask {mask_path}: {exc}"
            ) from exc
        if mask.shape != (height, width):
            raise RuntimeError(
                f"Candidate {candidate_index} Mask shape {mask.shape} differs from "
                f"image size {(height, width)}"
            )
        candidates.append(
            {
                "candidate_index": candidate_index,
                "sam_score": float(score),
                "mask": mask,
                "mask_file": str(mask_path.resolve()),
                "mask_area_pixels": int(mask.sum()),
            }
        )

    if sam_results.get("candidate_count") != len(candidates):
        raise RuntimeError("SAM 3 candidate_count does not match candidates list")
    if sam_results.get("status") == "no_candidates" and candidates:
        raise RuntimeError("SAM 3 status says no_candidates but candidates are present")

    return geom_rgb, geom_ids.astype(np.int32, copy=False), visible_geoms, candidates


def _group_visible_geoms(geom_ids: np.ndarray, visible_geoms: list[dict]) -> list[dict]:
    """Group visible Geoms only when they share the same owning MuJoCo Body."""
    groups: dict[int, dict] = {}
    for geom in visible_geoms:
        body_id = geom["body_id"]
        group = groups.setdefault(
            body_id,
            {
                "physical_object_id": f"body_{body_id}",
                "body_id": body_id,
                "body_name": geom["body_name"],
                "geoms": [],
            },
        )
        if group["body_name"] != geom["body_name"]:
            raise RuntimeError(
                f"Body {body_id} has inconsistent names in Geom metadata"
            )
        group["geoms"].append((geom["geom_id"], geom["geom_name"]))

    result = []
    for body_id in sorted(groups):
        group = groups[body_id]
        sorted_geoms = sorted(group.pop("geoms"), key=lambda value: value[0])
        group["geom_ids"] = [geom_id for geom_id, _ in sorted_geoms]
        group["geom_names"] = [name for _, name in sorted_geoms]
        group["pixel_count"] = int(
            np.count_nonzero(np.isin(geom_ids, group["geom_ids"]))
        )
        result.append(group)
    return result


def _score_candidates(
    geom_ids: np.ndarray,
    physical_objects: list[dict],
    candidates: list[dict],
    min_iou: float,
) -> tuple[list[list[float]], list[dict]]:
    object_masks = [
        np.isin(geom_ids, item["geom_ids"]) for item in physical_objects
    ]
    matrix: list[list[float]] = []
    candidate_results = []
    for candidate in candidates:
        candidate_mask = candidate["mask"]
        ious = []
        for object_mask in object_masks:
            intersection = int(np.count_nonzero(candidate_mask & object_mask))
            union = int(np.count_nonzero(candidate_mask | object_mask))
            ious.append(float(intersection / union) if union else 0.0)
        matrix.append(ious)

        max_iou = max(ious, default=0.0)
        top_indices = [
            index
            for index, iou in enumerate(ious)
            if math.isclose(iou, max_iou, rel_tol=0.0, abs_tol=1e-12)
        ]
        if max_iou <= 0.0 or candidate["mask_area_pixels"] == 0:
            top_indices = []
        top_overlaps = [
            {
                "physical_object_id": physical_objects[index]["physical_object_id"],
                "body_id": physical_objects[index]["body_id"],
                "body_name": physical_objects[index]["body_name"],
                "geom_ids": physical_objects[index]["geom_ids"],
                "geom_names": physical_objects[index]["geom_names"],
                "iou": ious[index],
            }
            for index in top_indices
        ]

        if not physical_objects:
            status = "unresolved"
            reason = "no_visible_physical_objects"
        elif candidate["mask_area_pixels"] == 0:
            status = "unresolved"
            reason = "empty_candidate_mask"
        elif max_iou <= 0.0:
            status = "unresolved"
            reason = "no_pixel_overlap"
        elif len(top_indices) > 1:
            status = "unresolved"
            reason = "tied_best_iou"
        elif max_iou < min_iou:
            status = "unresolved"
            reason = "best_iou_below_min_iou"
        else:
            status = "matched_by_iou"
            reason = None

        candidate_results.append(
            {
                "candidate_index": candidate["candidate_index"],
                "sam_score": candidate["sam_score"],
                "mask_file": candidate["mask_file"],
                "mask_area_pixels": candidate["mask_area_pixels"],
                "status": status,
                "unresolved_reason": reason,
                "best_iou": max_iou,
                "best_overlap_objects": top_overlaps,
                "best_match": (
                    top_overlaps[0] if status == "matched_by_iou" else None
                ),
            }
        )
    return matrix, candidate_results


def _blend_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    tint = np.asarray(color, dtype=np.float32)
    result[mask] = 0.45 * result[mask] + 0.55 * tint
    return np.clip(result, 0, 255).astype(np.uint8)


def run_matching(
    sam3_results_path: Path,
    geom_ids_path: Path,
    geom_metadata_path: Path,
    output_dir: Path,
    min_iou: float = 0.1,
) -> dict:
    if not 0.0 <= min_iou <= 1.0:
        raise ValueError("min_iou must be in [0, 1]")
    rgb, geom_ids, visible_geoms, candidates = _load_inputs(
        sam3_results_path, geom_ids_path, geom_metadata_path
    )
    physical_objects = _group_visible_geoms(geom_ids, visible_geoms)
    iou_matrix, candidate_results = _score_candidates(
        geom_ids, physical_objects, candidates, min_iou
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_images = []
    for candidate, result in zip(candidates, candidate_results):
        left = _blend_mask(rgb, candidate["mask"], (255, 64, 64))
        best_mask = np.zeros(candidate["mask"].shape, dtype=bool)
        for overlap in result["best_overlap_objects"]:
            if overlap["iou"] > 0:
                matching_object = next(
                    item
                    for item in physical_objects
                    if item["physical_object_id"] == overlap["physical_object_id"]
                )
                best_mask |= np.isin(geom_ids, matching_object["geom_ids"])
        right = _blend_mask(rgb, best_mask, (64, 220, 255))
        canvas = Image.new("RGB", (rgb.shape[1] * 2, rgb.shape[0] + 28), (24, 24, 24))
        canvas.paste(Image.fromarray(left), (0, 28))
        canvas.paste(Image.fromarray(right), (rgb.shape[1], 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"SAM candidate {result['candidate_index']} mask", fill="white")
        draw.text(
            (rgb.shape[1] + 8, 7),
            f"Top physical overlap: IoU={result['best_iou']:.4f} ({result['status']})",
            fill="white",
        )
        filename = f"candidate_{result['candidate_index']:03d}_iou_comparison.png"
        canvas.save(output_dir / filename)
        result["comparison_image"] = filename
        comparison_images.append(filename)

    matrix_data = {
        "candidate_indices": [item["candidate_index"] for item in candidates],
        "physical_object_ids": [item["physical_object_id"] for item in physical_objects],
        "values": iou_matrix,
    }
    result = {
        "status": "ok" if candidates else "no_candidates",
        "matching_basis": "pixel IoU only; SAM score is reported but not used for matching",
        "grouping_method": "visible Geoms sharing the same owning MuJoCo Body ID are combined",
        "grouping_limitation": (
            "Capture metadata does not include Body parent links, so Geoms on distinct "
            "Bodies are not merged into articulated multi-Body assemblies."
        ),
        "coordinate_direction": (
            "identity forward orientation shared with SAM 3 input"
            if _load_json(geom_metadata_path, "Geom metadata").get("pixel_alignment_transform") == "identity"
            else "180-degree forward orientation shared with SAM 3 input"
        ),
        "image_width": int(rgb.shape[1]),
        "image_height": int(rgb.shape[0]),
        "min_iou": min_iou,
        "sam3_results": str(sam3_results_path.resolve()),
        "geom_ids": str(geom_ids_path.resolve()),
        "geom_metadata": str(geom_metadata_path.resolve()),
        "physical_objects": physical_objects,
        "iou_matrix": matrix_data,
        "candidates": candidate_results,
        "comparison_images": comparison_images,
    }
    (output_dir / "sam3_mujoco_iou_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_min_iou(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number in [0, 1]") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-results", type=Path, required=True)
    parser.add_argument("--geom-ids", type=Path, required=True)
    parser.add_argument("--geom-metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--min-iou",
        type=_parse_min_iou,
        default=0.1,
        help="minimum unique IoU for a spatial match (default: 0.1)",
    )
    args = parser.parse_args(argv)
    for label, path in (
        ("SAM 3 results", args.sam3_results),
        ("rotated Geom ID map", args.geom_ids),
        ("Geom metadata", args.geom_metadata),
    ):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    try:
        result = run_matching(
            args.sam3_results,
            args.geom_ids,
            args.geom_metadata,
            args.out_dir,
            min_iou=args.min_iou,
        )
    except Exception as exc:
        print(f"SAM 3 × MuJoCo IoU Smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        f"IoU matching: candidates={len(result['candidates'])}, "
        f"physical_body_groups={len(result['physical_objects'])}, "
        f"min_iou={result['min_iou']:.3f}"
    )
    for candidate in result["candidates"]:
        best = candidate["best_overlap_objects"]
        best_text = ", ".join(
            f"body={item['body_id']} geoms={item['geom_ids']} IoU={item['iou']:.4f}"
            for item in best
        ) or "no visible object overlap"
        print(
            f"  candidate {candidate['candidate_index']}: "
            f"status={candidate['status']}, SAM score={candidate['sam_score']:.4f}, "
            f"best overlap: {best_text}"
        )
    print(f"Results saved under: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
