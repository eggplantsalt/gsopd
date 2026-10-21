"""One-shot task-conditioned selection among all SAM candidate crops.

This Smoke deliberately builds its VLM request from only the source RGB,
candidate indices, candidate Masks and candidate Boxes, plus caller-provided
task text. Teacher JSON is never read. Body/Geom/IoU data are opened only after
the single VLM response has been persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
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


SYSTEM_PROMPT = """You are doing a task-conditioned visual candidate verification.
Image 1 is the original full-scene RGB image. Each following image is an
automatically generated local crop of the same RGB image. Each crop corresponds
to the Candidate Index stated immediately before it. All supplied candidates
are shown.

If a target description is provided, your primary task is to identify an object
matching that description among all supplied candidates. The described object
may be the item to manipulate, a receptacle, an obstacle, or any other scene
object. Treat the task instruction only as context; it must not override the
target description, and it is not a reason to reject a candidate merely because
that candidate is not the operation target named in the instruction.

If no target description is provided, identify the operation target requested
by the task instruction.

In either mode, use visible image evidence. The candidate list may omit the
target, and having only one candidate does not make it correct. Select one
listed Candidate Index only when supported by the image; otherwise return
unresolved. Do not infer correctness from candidate order. Return only a
compact JSON object with keys status, candidate_index, and visual_basis. Use
status "selected" with one of the listed indices, or status "unresolved" with
a null candidate_index."""


_COLOR_WORDS = (
    "white", "black", "green", "brown", "red", "blue", "yellow", "orange",
    "purple", "pink", "gray", "grey", "silver", "gold", "beige", "tan",
    "ivory", "cream", "dark", "light", "clear", "transparent",
)
_COLOR_WORDS_ZH = ("白色", "黑色", "绿色", "棕色", "褐色", "红色", "蓝色", "黄色", "橙色", "紫色", "粉色", "灰色", "银色", "金色")


def _validate_target_description(value: str | None) -> str | None:
    if value is None:
        return None
    description = value.strip()
    if not description:
        raise ValueError("Target description cannot be empty when supplied")
    color_pattern = r"\b(?:" + "|".join(re.escape(word) for word in _COLOR_WORDS) + r")\b"
    if re.search(color_pattern, description, flags=re.IGNORECASE) or any(
        word in description for word in _COLOR_WORDS_ZH
    ):
        raise ValueError("Target description must omit color/appearance attributes")
    if re.search(r"\b(?:body|geom|iou)\b|black_book|white_place_box", description, flags=re.IGNORECASE):
        raise ValueError("Target description must not contain simulator identity or IoU metadata")
    return description


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} JSON must contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resolve(path_value: Any, base_dir: Path, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeError(f"Missing {label} path")
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def _candidate_inputs(sam3_results_path: Path, rgb: np.ndarray) -> list[dict[str, Any]]:
    # Only these fields are used. In particular, Teacher-facing names and
    # confidence scores in the same JSON are intentionally ignored.
    sam = _read_json(sam3_results_path, "SAM candidate")
    raw_candidates = sam.get("candidates")
    if not isinstance(raw_candidates, list):
        raise RuntimeError("SAM results must contain a candidate list")
    if sam.get("candidate_count") is not None and sam.get("candidate_count") != len(raw_candidates):
        raise RuntimeError("SAM candidate_count does not match the candidate list")
    height, width = rgb.shape[:2]
    if sam.get("image_width") != width or sam.get("image_height") != height:
        raise RuntimeError("SAM image dimensions do not match the source RGB")
    sam_rgb_path = _resolve(sam.get("image"), sam3_results_path.parent, "SAM input RGB")
    sam_rgb = _read_rgb(sam_rgb_path)
    if sam_rgb.shape != rgb.shape or not np.array_equal(sam_rgb, rgb):
        raise RuntimeError("Provided original RGB is not pixel-identical to the SAM input RGB")

    candidates: list[dict[str, Any]] = []
    indices: set[int] = set()
    for item in raw_candidates:
        if not isinstance(item, dict):
            raise RuntimeError("SAM candidate must be an object")
        index = item.get("candidate_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RuntimeError(f"Invalid candidate index: {index!r}")
        if index in indices:
            raise RuntimeError(f"Duplicate candidate index {index}")
        indices.add(index)
        box = item.get("box_xyxy_pixels")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)
        ):
            raise RuntimeError(f"Candidate {index} has an invalid pixel box")
        mask_path = _resolve(item.get("mask_file"), sam3_results_path.parent, f"Candidate {index} Mask")
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
        if mask.shape != (height, width):
            raise RuntimeError(f"Candidate {index} Mask dimensions do not match the source RGB")
        ys, xs = np.where(mask)
        if xs.size == 0:
            raise RuntimeError(f"Candidate {index} Mask is empty")

        # Build the crop automatically from the union of the Mask bounds and
        # supplied Box, then add 20% context (minimum 24 px) on every side.
        # The crop contains unmodified source RGB pixels; no resize or overlay.
        box_x0, box_y0, box_x1, box_y1 = [float(v) for v in box]
        union_x0 = min(float(xs.min()), box_x0)
        union_y0 = min(float(ys.min()), box_y0)
        union_x1 = max(float(xs.max() + 1), box_x1)
        union_y1 = max(float(ys.max() + 1), box_y1)
        pad = max(24, int(math.ceil(0.20 * max(union_x1 - union_x0, union_y1 - union_y0))))
        x0 = max(0, int(math.floor(union_x0)) - pad)
        y0 = max(0, int(math.floor(union_y0)) - pad)
        x1 = min(width, int(math.ceil(union_x1)) + pad)
        y1 = min(height, int(math.ceil(union_y1)) + pad)
        candidates.append(
            {
                "candidate_index": index,
                "mask_path": mask_path,
                "mask_sha256": _sha256(mask_path),
                "box_xyxy_pixels": [float(v) for v in box],
                "mask_bounds_xyxy_pixels": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
                "crop_xyxy_pixels": [x0, y0, x1, y1],
                "crop_rgb": rgb[y0:y1, x0:x1].copy(),
            }
        )
    candidates.sort(key=lambda item: item["candidate_index"])
    return candidates


def _save_crops_and_montage(
    rgb: np.ndarray, candidates: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    crops_dir = output_dir / "candidate_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for candidate in candidates:
        index = candidate["candidate_index"]
        path = crops_dir / f"candidate_{index:03d}_crop.png"
        crop = candidate["crop_rgb"]
        Image.fromarray(crop).save(path)
        candidate["crop_path"] = path
        entries.append(
            {
                "candidate_index": index,
                "box_xyxy_pixels": candidate["box_xyxy_pixels"],
                "mask_bounds_xyxy_pixels": candidate["mask_bounds_xyxy_pixels"],
                "crop_xyxy_pixels": candidate["crop_xyxy_pixels"],
                "crop_width": int(crop.shape[1]),
                "crop_height": int(crop.shape[0]),
                "mask_sha256": candidate["mask_sha256"],
                "crop_file": str(path),
                "crop_sha256": _sha256(path),
            }
        )

    # Reviewer-only overview. This contact sheet is not sent to the VLM.
    tile_w, label_h = 360, 36
    source = Image.fromarray(rgb)
    scale = tile_w / source.width
    full_thumb = source.resize((tile_w, int(source.height * scale)), Image.Resampling.LANCZOS)
    crops = []
    for candidate in candidates:
        crop = Image.open(candidate["crop_path"]).convert("RGB")
        thumb_h = min(460, int(round(crop.height * tile_w / crop.width)))
        thumb = crop.resize((tile_w, thumb_h), Image.Resampling.LANCZOS)
        crops.append((candidate["candidate_index"], thumb))
    max_h = max([full_thumb.height, *(image.height for _, image in crops)])
    cols = 3
    rows = math.ceil((len(crops) + 1) / cols)
    sheet = Image.new("RGB", (cols * tile_w, rows * (max_h + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    tiles = [("Original full scene (review overview only)", full_thumb)] + [
        (f"Candidate {index} crop (review overview only)", image) for index, image in crops
    ]
    for slot, (title, image) in enumerate(tiles):
        col, row = slot % cols, slot // cols
        x, y = col * tile_w, row * (max_h + label_h)
        draw.text((x + 6, y + 7), title, fill=(0, 0, 0))
        sheet.paste(image, (x, y + label_h))
    sheet.save(output_dir / "crop_overview_not_sent_to_vlm.png")
    return entries


def _frame_data_url(rgb: np.ndarray) -> str:
    height, width = rgb.shape[:2]
    frame = RGBFrame(
        camera_name="agentview_image",
        width=width,
        height=height,
        rgb_bytes=np.ascontiguousarray(rgb).tobytes(),
    )
    return _rgb_frame_to_data_url(frame)


def _build_payload(
    backend: Any,
    instruction: str,
    target_description: str | None,
    original_rgb: np.ndarray,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    target_description = _validate_target_description(target_description)
    candidate_indices = [item["candidate_index"] for item in candidates]
    system_prompt = SYSTEM_PROMPT + f"\nAvailable candidate indices: {json.dumps(candidate_indices)}."
    user_intro = [f"Task instruction (verbatim): {instruction}"]
    if target_description is not None:
        user_intro.append(f"Neutral target description: {target_description}")
    content: list[dict[str, Any]] = [
        *({"type": "text", "text": text} for text in user_intro),
        {"type": "text", "text": "Image 1: original full-scene RGB."},
        {"type": "image_url", "image_url": {"url": _frame_data_url(original_rgb)}},
    ]
    for image_number, candidate in enumerate(candidates, start=2):
        index = candidate["candidate_index"]
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"Image {image_number}: Candidate {index}.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": _frame_data_url(candidate["crop_rgb"])},
                },
            ]
        )
    text_only = system_prompt + "\n" + "\n".join(
        block["text"] for block in content if block["type"] == "text"
    )
    forbidden_terms = ("black_book", "white_place_box", "body_id", "geom_id", "candidate score")
    leaked = [term for term in forbidden_terms if term.casefold() in text_only.casefold()]
    if leaked:
        raise RuntimeError(f"VLM text includes prohibited metadata terms: {leaked}")
    return {
        "model": backend.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": backend.temperature,
        "max_tokens": backend.max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }


def _single_request(backend: Any, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    request = urllib.request.Request(
        backend.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    if backend.api_key:
        request.add_header("Authorization", f"Bearer {backend.api_key}")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=backend.timeout_s) as response:
            status = int(response.status)
            body = response.read()
        elapsed = time.perf_counter() - start
        response_json = json.loads(body.decode("utf-8"))
        content = response_json["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise RuntimeError("VLM response content was not a string")
        return content, {
            "status": "completed",
            "http_status": status,
            "elapsed_seconds": elapsed,
            "model": payload["model"],
            "id": response_json.get("id"),
            "usage": response_json.get("usage"),
            "request_count_sent": 1,
            "response_json": response_json,
        }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start
        detail = exc.read().decode("utf-8", errors="replace")
        return "", {
            "status": "http_error",
            "http_status": exc.code,
            "elapsed_seconds": elapsed,
            "model": payload["model"],
            "request_count_sent": 1,
            "error_body": detail,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return "", {
            "status": "request_error",
            "http_status": None,
            "elapsed_seconds": elapsed,
            "model": payload["model"],
            "request_count_sent": 1,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _parse_decision(raw: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except Exception as exc:
        return {"status": "unresolved", "candidate_index": None, "visual_basis": None, "parse_error": str(exc)}
    index = value.get("candidate_index") if isinstance(value, dict) else None
    status = value.get("status") if isinstance(value, dict) else None
    basis = value.get("visual_basis") if isinstance(value, dict) else None
    allowed = {item["candidate_index"] for item in candidates}
    if status == "selected" and isinstance(index, int) and not isinstance(index, bool) and index in allowed:
        return {"status": "selected", "candidate_index": index, "visual_basis": basis if isinstance(basis, str) else None}
    if status == "unresolved" and index is None:
        return {"status": "unresolved", "candidate_index": None, "visual_basis": basis if isinstance(basis, str) else None}
    return {
        "status": "unresolved",
        "candidate_index": None,
        "visual_basis": basis if isinstance(basis, str) else None,
        "parse_error": "Response did not match the allowed candidate-selection schema.",
    }


def _read_iou_after_request(
    iou_results_path: Path, decision: dict[str, Any]
) -> dict[str, Any]:
    # This function is intentionally called only after a VLM response is
    # available and its raw content or parsed decision has been persisted.
    iou = _read_json(iou_results_path, "Body-level IoU")
    candidate_index = decision.get("candidate_index")
    if decision.get("status") != "selected" or candidate_index is None:
        return {
            "status": "not_compared_unresolved",
            "candidate_index": None,
            "best_body_geom_overlap": None,
            "verified_body_geom_match": None,
        }
    entry = next(
        (item for item in iou.get("candidates", []) if item.get("candidate_index") == candidate_index),
        None,
    )
    if not isinstance(entry, dict):
        return {
            "status": "candidate_missing_from_iou_record",
            "candidate_index": candidate_index,
            "best_body_geom_overlap": None,
            "verified_body_geom_match": None,
        }
    best = entry.get("best_match") if isinstance(entry.get("best_match"), dict) else None
    overlaps = entry.get("best_overlap_objects")
    if best is None and isinstance(overlaps, list) and overlaps:
        best = next((item for item in overlaps if isinstance(item, dict)), None)
    if best is None:
        matrix = iou.get("iou_matrix")
        if isinstance(matrix, dict):
            candidate_ids = matrix.get("candidate_indices")
            physical_ids = matrix.get("physical_object_ids")
            values = matrix.get("values")
            if (
                isinstance(candidate_ids, list)
                and isinstance(physical_ids, list)
                and isinstance(values, list)
                and candidate_index in candidate_ids
            ):
                row_index = candidate_ids.index(candidate_index)
                row = values[row_index] if row_index < len(values) else None
                if isinstance(row, list) and len(row) == len(physical_ids) and row:
                    column_index = max(range(len(row)), key=lambda column: row[column])
                    physical_id = physical_ids[column_index]
                    physical = next(
                        (item for item in iou.get("physical_objects", []) if item.get("physical_object_id") == physical_id),
                        {},
                    )
                    if isinstance(physical, dict):
                        best = {
                            **physical,
                            "iou": row[column_index],
                        }
    best_summary = None
    if isinstance(best, dict):
        best_summary = {
            "physical_object_id": best.get("physical_object_id"),
            "body_id": best.get("body_id"),
            "body_name": best.get("body_name"),
            "geom_ids": best.get("geom_ids"),
            "geom_names": best.get("geom_names"),
            "iou": best.get("iou", entry.get("best_iou")),
        }
    match_status = entry.get("status", iou.get("status", "unknown"))
    return {
        "status": match_status,
        "candidate_index": candidate_index,
        "sam_score": entry.get("sam_score"),
        "best_body_geom_overlap": best_summary,
        "verified_body_geom_match": best_summary if match_status == "matched_by_iou" else None,
        "iou_threshold": iou.get("min_iou"),
        "interpretation": (
            "Post-hoc geometry comparison only; it does not establish target semantics or create a binding."
        ),
    }


def run_smoke(
    rgb_path: Path,
    sam3_results_path: Path,
    iou_results_path: Path,
    output_dir: Path,
    instruction: str,
    target_description: str | None = None,
) -> dict[str, Any]:
    if not instruction.strip():
        raise ValueError("Task instruction cannot be empty")
    target_description = _validate_target_description(target_description)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = _read_rgb(rgb_path)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.shape[0] or not rgb.shape[1]:
        raise RuntimeError(f"Expected a non-empty HxWx3 RGB image, got {rgb.shape}")
    height, width = rgb.shape[:2]
    candidates = _candidate_inputs(sam3_results_path, rgb)
    crop_manifest = _save_crops_and_montage(rgb, candidates, output_dir)
    original_copy = output_dir / f"original_rgb_{width}x{height}.png"
    Image.fromarray(rgb).save(original_copy)

    if not candidates:
        decision = {
            "status": "unresolved",
            "candidate_index": None,
            "visual_basis": "SAM returned no candidates; no VLM request was made.",
        }
        (output_dir / "candidate_selection.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        result = {
            "status": "unresolved",
            "candidate_selection": decision,
            "posthoc_geometry_association": {"status": "not_compared_no_candidates"},
            "task_instruction": instruction,
            "target_description": target_description,
            "image_width": width,
            "image_height": height,
            "candidate_indices_sent": [],
            "request_count_sent": 0,
            "teacher_ir_modified": False,
            "formal_semantic_binding_written": False,
        }
        (output_dir / "experiment_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result

    # Resolve the model config only after all image preparation is complete.
    backend = create_backend_from_env()
    if backend.model != "Qwen3.8-27B-FP8":
        raise BackendConfigError(f"Expected frozen Qwen3.8-27B-FP8, got {backend.model!r}")
    if backend.temperature != 0.0 or backend.max_tokens != 4096 or backend.thinking_enabled is not False:
        raise BackendConfigError(
            "Expected the archived non-thinking, temperature=0, max_tokens=4096 configuration"
        )
    payload = _build_payload(backend, instruction, target_description, rgb, candidates)
    payload_path = output_dir / "actual_vlm_request_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Record the exact visible text and image hashes before making the sole call.
    input_manifest = {
        "model": backend.model,
        "endpoint": backend.endpoint,
        "temperature": backend.temperature,
        "max_tokens": backend.max_tokens,
        "thinking_enabled": backend.thinking_enabled,
        "instruction": instruction,
        "target_description": target_description,
        "system_prompt": payload["messages"][0]["content"],
        "source_rgb": {"path": str(rgb_path.resolve()), "sha256": _sha256(rgb_path), "width": width, "height": height},
        "source_sam3_results": {"path": str(sam3_results_path.resolve()), "sha256": _sha256(sam3_results_path)},
        "candidate_count": len(candidates),
        "candidate_indices": [item["candidate_index"] for item in candidates],
        "crops": crop_manifest,
        "request_payload_file": payload_path.name,
        "request_payload_sha256": _sha256(payload_path),
        "teacher_json_used": False,
        "body_geom_iou_used_before_vlm": False,
        "candidate_scores_in_vlm_request": False,
        "candidate_prefiltering": False,
        "request_count_planned": 1,
    }
    (output_dir / "vlm_input_manifest.json").write_text(
        json.dumps(input_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Persist a pre-call marker so the output records that post-hoc IoU parsing
    # was not allowed to happen until the model answer existed.
    (output_dir / "single_request_prepared.json").write_text(
        json.dumps({"ready": True, "request_count_planned": 1, "candidate_indices": input_manifest["candidate_indices"]}, indent=2) + "\n",
        encoding="utf-8",
    )

    raw, call_result = _single_request(backend, payload)
    call_result_path = output_dir / "vlm_call_result.json"
    call_result_path.write_text(json.dumps(call_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "vlm_output_raw.txt").write_text(raw, encoding="utf-8")
    decision = _parse_decision(raw, candidates) if raw else {
        "status": "unresolved", "candidate_index": None, "visual_basis": None,
        "parse_error": "No assistant content returned by the single request.",
    }
    (output_dir / "candidate_selection.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # This is the first point at which Body-level IoU is opened.
    iou_comparison = _read_iou_after_request(iou_results_path, decision)
    result = {
        "status": "completed" if call_result.get("status") == "completed" else call_result.get("status"),
        "candidate_selection": decision,
        "posthoc_geometry_association": iou_comparison,
        "task_instruction": instruction,
        "target_description": target_description,
        "image_width": width,
        "image_height": height,
        "candidate_indices_sent": input_manifest["candidate_indices"],
        "candidate_crop_formula": "union of each full-image Mask bounds and candidate Box; pad each edge by max(24 px, ceil(20% of the larger union dimension)); clamp to image; crop raw RGB without resize",
        "body_iou_source_read_after_vlm": {"path": str(iou_results_path.resolve()), "sha256": _sha256(iou_results_path)},
        "raw_response_file": "vlm_output_raw.txt",
        "actual_request_file": payload_path.name,
        "request_count_sent": call_result.get("request_count_sent", 1),
        "teacher_ir_modified": False,
        "formal_semantic_binding_written": False,
    }
    (output_dir / "experiment_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_offline_regression(
    rgb_path: Path,
    sam3_results_path: Path,
    iou_results_path: Path,
    output_dir: Path,
    instruction: str,
    raw_response_path: Path,
    target_description: str | None = None,
) -> dict[str, Any]:
    """Re-evaluate a saved answer without creating a backend or sending a request."""
    if not instruction.strip():
        raise ValueError("Task instruction cannot be empty")
    target_description = _validate_target_description(target_description)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = _read_rgb(rgb_path)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.shape[0] or not rgb.shape[1]:
        raise RuntimeError(f"Expected a non-empty HxWx3 RGB image, got {rgb.shape}")
    height, width = rgb.shape[:2]
    candidates = _candidate_inputs(sam3_results_path, rgb)
    crops = _save_crops_and_montage(rgb, candidates, output_dir)
    Image.fromarray(rgb).save(output_dir / f"original_rgb_{width}x{height}.png")
    raw = raw_response_path.read_text(encoding="utf-8")
    decision = _parse_decision(raw, candidates)
    (output_dir / "existing_vlm_response_raw.txt").write_text(raw, encoding="utf-8")
    (output_dir / "candidate_selection.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # The archived answer is established before reading geometry results.
    association = _read_iou_after_request(iou_results_path, decision)
    result = {
        "status": "offline_regression_completed",
        "inference_performed": False,
        "request_count_sent": 0,
        "source_response": {"path": str(raw_response_path.resolve()), "sha256": _sha256(raw_response_path)},
        "source_rgb": {"path": str(rgb_path.resolve()), "sha256": _sha256(rgb_path)},
        "source_sam3_results": {"path": str(sam3_results_path.resolve()), "sha256": _sha256(sam3_results_path)},
        "source_iou_results_after_response": {"path": str(iou_results_path.resolve()), "sha256": _sha256(iou_results_path)},
        "candidate_selection": decision,
        "posthoc_geometry_association": association,
        "task_instruction": instruction,
        "target_description": target_description,
        "image_width": width,
        "image_height": height,
        "candidate_count": len(candidates),
        "candidate_indices": [item["candidate_index"] for item in candidates],
        "crops": crops,
        "teacher_ir_modified": False,
        "formal_semantic_binding_written": False,
    }
    (output_dir / "offline_regression_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--sam3-results", type=Path, required=True)
    parser.add_argument("--iou-results", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--target-description",
        help="optional neutral description, e.g. book, bottle, caddy; omit Teacher colors and simulator metadata",
    )
    parser.add_argument(
        "--offline-response",
        type=Path,
        help="reuse this saved VLM answer without initializing a backend or sending a request",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    required_files = [
        ("RGB", args.rgb),
        ("SAM results", args.sam3_results),
        ("IoU results", args.iou_results),
    ]
    if args.offline_response is not None:
        required_files.append(("Saved VLM response", args.offline_response))
    for label, path in required_files:
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    try:
        if args.offline_response is None:
            result = run_smoke(
                args.rgb,
                args.sam3_results,
                args.iou_results,
                args.out_dir,
                args.instruction,
                target_description=args.target_description,
            )
        else:
            result = run_offline_regression(
                args.rgb,
                args.sam3_results,
                args.iou_results,
                args.out_dir,
                args.instruction,
                args.offline_response,
                target_description=args.target_description,
            )
    except (BackendConfigError, TeacherBackendError, OSError, ValueError, RuntimeError) as exc:
        print(f"Task-conditioned candidate verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved under: {args.out_dir.resolve()}")
    return 0 if result["status"] in ("completed", "offline_regression_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
