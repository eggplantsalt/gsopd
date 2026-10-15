"""Locate a Teacher v2 object with one VLM point and map it locally to a Mask."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from research.gs_opd.teacher.backend import (
    BackendConfigError,
    RGBFrame,
    TeacherBackendError,
    _rgb_frame_to_data_url,
    create_backend_from_env,
)
from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


SYSTEM_PROMPT = """You locate one semantically described object in a single original RGB image.
Return one point on a clearly visible part of that object, not a bounding box.
Coordinates must be normalized to [0, 1], with the origin at the top-left:
x increases left to right and y increases top to bottom. Use only the supplied
object description and image. If the object cannot be identified confidently
or no reliable point lies on its visible pixels, return unresolved with x and y
both null. Do not guess. Return only the required JSON object.
"""


LOCATION_OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "x", "y", "reason"],
    "properties": {
        "status": {"enum": ["located", "unresolved"]},
        "x": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "y": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1},
    },
}


class VisualPointOutputError(ValueError):
    """Raised when the VLM point response is not a valid strict JSON result."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_visual_point_output(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise VisualPointOutputError(f"Invalid VLM JSON response: {exc}") from exc
    if not isinstance(value, dict):
        raise VisualPointOutputError("VLM response must be a JSON object")
    expected = {"status", "x", "y", "reason"}
    if set(value) != expected:
        raise VisualPointOutputError(
            f"VLM response fields must be exactly {sorted(expected)}"
        )
    status = value["status"]
    if status not in ("located", "unresolved"):
        raise VisualPointOutputError("status must be 'located' or 'unresolved'")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise VisualPointOutputError("reason must be a non-empty string")

    x, y = value["x"], value["y"]
    if status == "unresolved":
        if x is not None or y is not None:
            raise VisualPointOutputError("unresolved output must set x and y to null")
    else:
        for name, coordinate in (("x", x), ("y", y)):
            if (
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(coordinate)
                or not 0 <= coordinate <= 1
            ):
                raise VisualPointOutputError(
                    f"located output {name} must be a finite number in [0, 1]"
                )
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot load {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} JSON must contain an object")
    return value


def _resolve_path(value: Any, base_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Missing {label} path")
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _read_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(f"Cannot read RGB image {path}: {exc}") from exc


def _object_context(target: dict[str, Any], teacher: dict[str, Any]) -> dict[str, Any]:
    related_rules = [
        rule for rule in teacher["rules"] if rule["target_id"] == target["id"]
    ]
    return {
        "name": target["name"],
        "role": target["role"],
        "visibility": target["visibility"],
        "mobility": target["mobility"],
        "visual_facts": target["visual_facts"],
        "inferences": target["inferences"],
        "uncertainties": target["uncertainties"],
        "related_contact_rules": related_rules,
        "global_uncertainties": teacher["global_uncertainties"],
    }


def _build_user_prompt(target: dict[str, Any], teacher: dict[str, Any]) -> str:
    context = _object_context(target, teacher)
    return (
        "Find one confidently visible pixel point on the described object in the "
        "attached original RGB image. Coordinates are normalized to [0, 1], "
        "origin top-left; x is horizontal and y is vertical.\n"
        "Teacher object semantics (Safety IR v2):\n"
        f"{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}\n"
        "If visual evidence does not support a reliable point, return unresolved "
        "with x and y both null. Do not infer a point from task convention alone."
    )


def _request_vlm(backend: Any, prompt: str, rgb: np.ndarray) -> str:
    required = ("model", "temperature", "max_tokens", "_post_payload")
    if any(not hasattr(backend, name) for name in required):
        raise TeacherBackendError(
            "This Smoke requires the configured OpenAI-compatible Teacher Backend"
        )
    height, width = rgb.shape[:2]
    frame = RGBFrame(
        camera_name="agentview_image",
        width=width,
        height=height,
        rgb_bytes=np.ascontiguousarray(rgb).tobytes(),
    )
    payload = {
        "model": backend.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Image: original forward RGB."},
                    {
                        "type": "image_url",
                        "image_url": {"url": _rgb_frame_to_data_url(frame)},
                    },
                ],
            },
        ],
        "temperature": backend.temperature,
        "max_tokens": backend.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "gs_opd_teacher_visual_point_v1",
                "schema": LOCATION_OUTPUT_JSON_SCHEMA,
            },
        },
    }
    thinking_enabled = getattr(backend, "thinking_enabled", None)
    if thinking_enabled is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": thinking_enabled}
    # Send one schema-constrained request through the existing backend transport;
    # never retry with an unconstrained or json_object response format.
    return backend._post_payload(payload, schema_constrained=True)


def _load_case(
    teacher_json: Path,
    object_id: str,
    image_path: Path,
    sam3_results_path: Path,
    iou_results_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    teacher = parse_teacher_output_v2(teacher_json.read_text(encoding="utf-8"))
    target = next((item for item in teacher["objects"] if item["id"] == object_id), None)
    if target is None:
        available = [item["id"] for item in teacher["objects"]]
        raise RuntimeError(f"Teacher object_id {object_id!r} not found; available IDs: {available}")

    sam_results = _load_json(sam3_results_path, "SAM 3 results")
    iou_results = _load_json(iou_results_path, "MuJoCo IoU results")
    if iou_results.get("status") not in ("ok", "no_candidates"):
        raise RuntimeError(f"Unsupported IoU results status: {iou_results.get('status')!r}")
    if iou_results.get("coordinate_direction") != (
        "180-degree forward orientation shared with SAM 3 input"
    ):
        raise RuntimeError("IoU results do not confirm the SAM 3 forward image orientation")
    if "sam3_results" in iou_results:
        recorded_sam = Path(iou_results["sam3_results"]).resolve()
        if recorded_sam != sam3_results_path.resolve():
            raise RuntimeError("IoU results refer to a different SAM 3 results file")

    sam_object_id = sam_results.get("object_id")
    if sam_object_id is not None and sam_object_id != object_id:
        raise RuntimeError(
            f"SAM 3 results belong to object_id {sam_object_id!r}, not {object_id!r}"
        )
    sam_image_path = _resolve_path(
        sam_results.get("image"), sam3_results_path.parent, "SAM 3 input RGB"
    )
    rgb = _read_rgb(image_path)
    sam_rgb = _read_rgb(sam_image_path)
    if rgb.shape != sam_rgb.shape or not np.array_equal(rgb, sam_rgb):
        raise RuntimeError("--image is not pixel-identical to the RGB used by SAM 3")

    geom_metadata_value = iou_results.get("geom_metadata")
    if not isinstance(geom_metadata_value, str) or not geom_metadata_value:
        raise RuntimeError("IoU results are missing their source Geom metadata path")
    geom_metadata_path = Path(geom_metadata_value).resolve()
    geom_metadata = _load_json(geom_metadata_path, "Geom metadata")
    forward_name = geom_metadata.get("rgb_rotated_180_file")
    directions = geom_metadata.get("coordinate_directions")
    direction = directions.get(forward_name) if isinstance(directions, dict) else None
    if (
        not isinstance(forward_name, str)
        or not isinstance(direction, str)
        or "180" not in direction.lower()
        or "forward" not in direction.lower()
    ):
        raise RuntimeError("Geom metadata does not mark a 180-degree forward RGB image")
    forward_rgb_path = Path(forward_name)
    if not forward_rgb_path.is_absolute():
        forward_rgb_path = geom_metadata_path.parent / forward_rgb_path
    forward_rgb = _read_rgb(forward_rgb_path)
    if rgb.shape != forward_rgb.shape or not np.array_equal(rgb, forward_rgb):
        raise RuntimeError("--image does not match the forward RGB used for Geom IoU capture")

    height, width = rgb.shape[:2]
    if width <= 0 or height <= 0:
        raise RuntimeError("Input RGB image has invalid dimensions")
    for label, payload in (("SAM 3", sam_results), ("IoU", iou_results)):
        if payload.get("image_width") != width or payload.get("image_height") != height:
            raise RuntimeError(f"{label} dimensions do not match the original RGB image")

    raw_candidates = sam_results.get("candidates")
    iou_candidates = iou_results.get("candidates")
    if not isinstance(raw_candidates, list) or not isinstance(iou_candidates, list):
        raise RuntimeError("SAM 3 and IoU results must both contain candidate lists")
    if sam_results.get("candidate_count") != len(raw_candidates):
        raise RuntimeError("SAM 3 candidate_count does not match its candidate list")
    if iou_results.get("status") == "no_candidates" and raw_candidates:
        raise RuntimeError("IoU reports no_candidates but SAM 3 contains candidates")

    iou_by_index: dict[int, dict[str, Any]] = {}
    for item in iou_candidates:
        if not isinstance(item, dict):
            raise RuntimeError("IoU candidate entries must be objects")
        index = item.get("candidate_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RuntimeError(f"Invalid IoU candidate_index: {index!r}")
        if index in iou_by_index:
            raise RuntimeError(f"Duplicate IoU candidate_index {index}")
        iou_by_index[index] = item

    candidates: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for item in raw_candidates:
        if not isinstance(item, dict):
            raise RuntimeError("SAM 3 candidate entries must be objects")
        index = item.get("candidate_index")
        score = item.get("score")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RuntimeError(f"Invalid SAM candidate_index: {index!r}")
        if index in seen_indices:
            raise RuntimeError(f"Duplicate SAM candidate_index {index}")
        seen_indices.add(index)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise RuntimeError(f"Invalid SAM score for candidate {index}")
        mask_path = _resolve_path(
            item.get("mask_file"), sam3_results_path.parent, f"candidate {index} Mask"
        )
        try:
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert("L")) > 0
        except Exception as exc:
            raise RuntimeError(f"Cannot read candidate {index} Mask {mask_path}: {exc}") from exc
        if mask.shape != (height, width):
            raise RuntimeError(f"Candidate {index} Mask dimensions do not match original RGB")
        iou_entry = iou_by_index.get(index)
        if iou_entry is None:
            raise RuntimeError(f"IoU results are missing SAM candidate {index}")
        iou_score = iou_entry.get("sam_score")
        if not isinstance(iou_score, (int, float)) or not math.isclose(
            float(iou_score), float(score), rel_tol=0.0, abs_tol=1e-8
        ):
            raise RuntimeError(f"IoU score does not match SAM candidate {index}")
        recorded_mask = iou_entry.get("mask_file")
        if recorded_mask:
            recorded_mask_path = _resolve_path(
                recorded_mask, iou_results_path.parent, f"IoU candidate {index} Mask"
            )
            if recorded_mask_path.resolve() != mask_path.resolve():
                raise RuntimeError(f"IoU result refers to a different Mask for candidate {index}")
        if iou_entry.get("status") not in ("matched_by_iou", "unresolved"):
            raise RuntimeError(f"Unsupported IoU status for candidate {index}")
        candidates.append({"candidate_index": index, "mask": mask, "iou_result": iou_entry})
    if set(iou_by_index) != seen_indices:
        raise RuntimeError("SAM 3 and IoU candidate index sets do not match")
    return target, {"teacher": teacher, "iou": iou_results}, rgb, candidates


def _normalized_to_pixel(value: float, size: int) -> int:
    # Map normalized endpoints to the first/last pixel with deterministic
    # round-half-up behavior rather than Python's ties-to-even round().
    return min(size - 1, int(math.floor(value * (size - 1) + 0.5)))


def _physical_association(iou_result: dict[str, Any], physical_objects: dict[str, Any]) -> dict[str, Any]:
    if iou_result.get("status") != "matched_by_iou":
        return {
            "status": "unresolved",
            "reason": iou_result.get("unresolved_reason") or "candidate_iou_unresolved",
        }
    best = iou_result.get("best_match")
    if not isinstance(best, dict):
        return {"status": "unresolved", "reason": "missing_best_match"}
    physical_id = best.get("physical_object_id")
    physical = physical_objects.get(physical_id) if isinstance(physical_id, str) else None
    body_id = best.get("body_id")
    geom_ids = best.get("geom_ids")
    iou = best.get("iou")
    if physical is None:
        return {"status": "unresolved", "reason": "unknown_physical_object"}
    if (
        isinstance(body_id, bool)
        or not isinstance(body_id, int)
        or body_id < 0
        or not isinstance(geom_ids, list)
        or not geom_ids
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in geom_ids)
        or isinstance(iou, bool)
        or not isinstance(iou, (int, float))
        or not math.isfinite(iou)
        or not 0 < iou <= 1
        or body_id != physical.get("body_id")
        or geom_ids != physical.get("geom_ids")
    ):
        return {"status": "unresolved", "reason": "invalid_best_match_metadata"}
    return {
        "status": "matched_by_iou",
        "physical_object_id": physical_id,
        "body_id": body_id,
        "geom_ids": geom_ids,
        "iou": float(iou),
    }


def _load_semantic_reference(path: Path | None, object_id: str) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _load_json(path, "existing semantic disambiguation result")
    teacher_object = payload.get("teacher_object")
    decision = payload.get("semantic_decision")
    association = payload.get("physical_association")
    if not isinstance(teacher_object, dict) or teacher_object.get("object_id") != object_id:
        raise RuntimeError("Semantic comparison result belongs to a different Teacher object")
    if not isinstance(decision, dict) or decision.get("object_id") != object_id:
        raise RuntimeError("Semantic comparison result has no matching semantic_decision")
    status = decision.get("status")
    index = decision.get("candidate_index")
    if status == "resolved":
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RuntimeError("Resolved semantic comparison has an invalid candidate_index")
    elif status == "unresolved":
        if index is not None:
            raise RuntimeError("Unresolved semantic comparison must have a null candidate_index")
    else:
        raise RuntimeError("Semantic comparison status must be resolved or unresolved")
    if not isinstance(association, dict):
        association = {"status": "unresolved"}
    return {
        "candidate_index": index,
        "status": status,
        "physical_association": association,
        "path": str(path.resolve()),
    }


def _save_visualization(
    rgb: np.ndarray,
    output_path: Path,
    pixel: tuple[int, int] | None,
) -> None:
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    if pixel is None:
        draw.text((8, 8), "unresolved", fill=(255, 40, 40), stroke_width=2, stroke_fill=(0, 0, 0))
    else:
        x, y = pixel
        radius = max(5, min(12, min(image.size) // 32))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 40, 40), width=2)
        draw.line((x - radius - 3, y, x + radius + 3, y), fill=(255, 40, 40), width=1)
        draw.line((x, y - radius - 3, x, y + radius + 3), fill=(255, 40, 40), width=1)
    image.save(output_path)


def run_smoke(
    teacher_json: Path,
    object_id: str,
    image_path: Path,
    sam3_results_path: Path,
    iou_results_path: Path,
    output_dir: Path,
    backend: Any | None = None,
    semantic_result_path: Path | None = None,
) -> dict[str, Any]:
    target, loaded, rgb, candidates = _load_case(
        teacher_json, object_id, image_path, sam3_results_path, iou_results_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if backend is None:
        backend = create_backend_from_env()
    prompt = _build_user_prompt(target, loaded["teacher"])
    started = time.perf_counter()
    raw_output = _request_vlm(backend, prompt, rgb)
    elapsed_s = time.perf_counter() - started
    if not isinstance(raw_output, str):
        raise TeacherBackendError("Teacher backend returned non-text output")
    raw_path = output_dir / "teacher_visual_point_raw.txt"
    raw_path.write_text(raw_output, encoding="utf-8")
    location = parse_visual_point_output(raw_output)

    pixel: tuple[int, int] | None = None
    member_indices: list[int] = []
    point_candidate_index: int | None = None
    association: dict[str, Any] = {"status": "unresolved", "reason": "no_location_point"}
    selected_candidate = None
    if location["status"] == "located":
        height, width = rgb.shape[:2]
        pixel = (
            _normalized_to_pixel(location["x"], width),
            _normalized_to_pixel(location["y"], height),
        )
        px, py = pixel
        member_indices = [
            item["candidate_index"] for item in candidates if bool(item["mask"][py, px])
        ]
        if len(member_indices) == 1:
            point_candidate_index = member_indices[0]
            iou_entry = next(
                item["iou_result"]
                for item in candidates
                if item["candidate_index"] == point_candidate_index
            )
            physical_objects = {
                item["physical_object_id"]: item
                for item in loaded["iou"].get("physical_objects", [])
                if isinstance(item, dict) and isinstance(item.get("physical_object_id"), str)
            }
            association = _physical_association(iou_entry, physical_objects)
            if association["status"] == "matched_by_iou":
                selected_candidate = {
                    "candidate_index": point_candidate_index,
                    "body_id": association["body_id"],
                    "geom_ids": association["geom_ids"],
                }
        elif not member_indices:
            association = {"status": "unresolved", "reason": "point_outside_all_candidate_masks"}
        else:
            association = {"status": "unresolved", "reason": "point_inside_multiple_candidate_masks"}

    visualization_path = output_dir / "teacher_visual_point.png"
    _save_visualization(rgb, visualization_path, pixel)
    reference = _load_semantic_reference(semantic_result_path, object_id)
    comparison = None
    if reference is not None:
        reference_association = reference["physical_association"]
        candidate_agreement = (
            point_candidate_index == reference["candidate_index"]
            if point_candidate_index is not None and reference["candidate_index"] is not None
            else None
        )
        current_has_physical = association.get("status") == "matched_by_iou"
        reference_has_physical = reference_association.get("status") == "matched_by_iou"
        physical_agreement = None
        if current_has_physical and reference_has_physical:
            physical_agreement = (
                association.get("body_id") == reference_association.get("body_id")
                and association.get("geom_ids") == reference_association.get("geom_ids")
            )
        comparison = {
            "reference_file": reference["path"],
            "reference_status": reference["status"],
            "reference_candidate_index": reference["candidate_index"],
            "point_candidate_index": point_candidate_index,
            "candidate_agreement": candidate_agreement,
            "physical_association_agreement": physical_agreement,
        }

    result = {
        "status": "resolved" if selected_candidate is not None else "unresolved",
        "teacher_object": {
            "object_id": target["id"],
            "name": target["name"],
            "role": target["role"],
        },
        "location": {
            "status": location["status"],
            "x": location["x"],
            "y": location["y"],
            "pixel_x": pixel[0] if pixel is not None else None,
            "pixel_y": pixel[1] if pixel is not None else None,
            "coordinate_convention": "normalized [0,1], origin top-left; x horizontal, y vertical",
            "reason": location["reason"],
        },
        "candidate_membership": {
            "status": (
                "unique" if len(member_indices) == 1
                else "ambiguous" if len(member_indices) > 1
                else "no_match"
            ),
            "candidate_indices": member_indices,
        },
        "selected_candidate": selected_candidate,
        "physical_association": association,
        "semantic_comparison": comparison,
        "visualization": visualization_path.name,
        "raw_response_file": raw_path.name,
        "sam3_results": str(sam3_results_path.resolve()),
        "iou_results": str(iou_results_path.resolve()),
        "teacher_json": str(teacher_json.resolve()),
        "image": str(image_path.resolve()),
        "vlm_request_seconds": elapsed_s,
    }
    result_path = output_dir / "teacher_visual_point_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-json", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--image", type=Path, required=True, help="Original forward RGB used by SAM 3")
    parser.add_argument("--sam3-results", type=Path, required=True)
    parser.add_argument("--iou-results", type=Path, required=True)
    parser.add_argument("--semantic-result", type=Path, help="Optional existing VLM semantic disambiguation JSON")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for label, path in (
        ("Teacher v2 JSON", args.teacher_json),
        ("original forward RGB", args.image),
        ("SAM 3 results", args.sam3_results),
        ("MuJoCo IoU results", args.iou_results),
    ):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    if args.semantic_result is not None and not args.semantic_result.is_file():
        parser.error(f"Semantic comparison JSON does not exist: {args.semantic_result}")
    try:
        result = run_smoke(
            args.teacher_json,
            args.object_id,
            args.image,
            args.sam3_results,
            args.iou_results,
            args.out_dir,
            semantic_result_path=args.semantic_result,
        )
    except (
        BackendConfigError,
        TeacherBackendError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"Teacher visual point Smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Visual point localization: status={result['status']}")
    print(json.dumps(result["location"], ensure_ascii=False, indent=2))
    print(f"Candidate membership: {result['candidate_membership']}")
    if result["selected_candidate"] is not None:
        print(f"Selected physical match: {result['selected_candidate']}")
    if result["semantic_comparison"] is not None:
        print(f"Semantic result comparison: {result['semantic_comparison']}")
    print(f"Results saved under: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
