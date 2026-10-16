"""Resolve a Teacher v2 object to a numbered SAM 3 Candidate using a VLM."""

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


SYSTEM_PROMPT = """You perform visual semantic disambiguation for robot manipulation.
Use the supplied Teacher object description and both supplied views of the same
RGB frame. The first image is the unmodified original RGB and is the only source
for judging genuine object appearance, including color and material. The second
image adds thin candidate boundaries and integer labels only to help locate SAM
segmentation candidates. These artificial boundaries and labels are not visual
evidence and must never be used to judge color, material, or object appearance.
Decide which numbered candidate, if any, visually depicts the requested
semantic object. Do not infer identity from segmentation confidence, simulator
names, or hidden physical IDs.
If the image is ambiguous, occluded, or no candidate depicts the object, return
unresolved and candidate_index=null. Never guess. Return only the required JSON.
"""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot load {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} JSON must contain an object")
    return payload


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


def _load_case(
    teacher_json: Path,
    object_id: str,
    sam3_results_path: Path,
    iou_results_path: Path,
    geometry_inventory_path: Path | None = None,
) -> tuple[dict, dict, np.ndarray, list[dict]]:
    teacher = parse_teacher_output_v2(teacher_json.read_text(encoding="utf-8"))
    target = next((item for item in teacher["objects"] if item["id"] == object_id), None)
    if target is None:
        known_ids = [item["id"] for item in teacher["objects"]]
        raise RuntimeError(
            f"Teacher object_id {object_id!r} was not found; available IDs: {known_ids}"
        )

    sam_results = _load_json(sam3_results_path, "SAM 3 results")
    iou_results = _load_json(iou_results_path, "MuJoCo IoU results")
    if iou_results.get("status") not in ("ok", "no_candidates"):
        raise RuntimeError(f"IoU results have unsupported status: {iou_results.get('status')!r}")
    if iou_results.get("coordinate_direction") != (
        "180-degree forward orientation shared with SAM 3 input"
    ):
        raise RuntimeError("IoU results do not confirm the SAM 3 forward image orientation")
    if "sam3_results" in iou_results:
        recorded_sam_results = Path(iou_results["sam3_results"]).resolve()
        if recorded_sam_results != sam3_results_path.resolve():
            raise RuntimeError(
                "IoU results were generated from a different SAM 3 results file"
            )
    sam_object_id = sam_results.get("object_id")
    if sam_object_id is not None and sam_object_id != object_id:
        raise RuntimeError(
            f"SAM 3 results were generated for object_id {sam_object_id!r}, "
            f"not requested Teacher object_id {object_id!r}"
        )

    image_path = _resolve_path(
        sam_results.get("image"), sam3_results_path.parent, "SAM 3 input image"
    )
    rgb = _read_rgb(image_path)
    height, width = rgb.shape[:2]
    geom_metadata_value = iou_results.get("geom_metadata")
    if not isinstance(geom_metadata_value, str) or not geom_metadata_value:
        raise RuntimeError("IoU results are missing their source Geom metadata path")
    geom_metadata_path = Path(geom_metadata_value).resolve()
    geom_metadata = _load_json(geom_metadata_path, "Geom metadata")
    rgb_filename = geom_metadata.get("rgb_rotated_180_file")
    directions = geom_metadata.get("coordinate_directions")
    direction = directions.get(rgb_filename) if isinstance(directions, dict) else None
    if (
        not isinstance(rgb_filename, str)
        or not isinstance(direction, str)
        or "180" not in direction.lower()
        or "forward" not in direction.lower()
    ):
        raise RuntimeError("Source Geom RGB is not marked as 180-degree forward orientation")
    geom_rgb_path = Path(rgb_filename)
    if not geom_rgb_path.is_absolute():
        geom_rgb_path = geom_metadata_path.parent / geom_rgb_path
    geom_rgb = _read_rgb(geom_rgb_path)
    if geom_rgb.shape != rgb.shape or not np.array_equal(geom_rgb, rgb):
        raise RuntimeError(
            "SAM 3 input RGB is not pixel-identical to the forward RGB used by "
            "the MuJoCo IoU capture"
        )
    for label, payload in (("SAM 3", sam_results), ("IoU", iou_results)):
        if payload.get("image_width") != width or payload.get("image_height") != height:
            raise RuntimeError(
                f"{label} metadata image size does not match source RGB: "
                f"expected {width}x{height}"
            )

    raw_candidates = sam_results.get("candidates")
    iou_candidates = iou_results.get("candidates")
    if not isinstance(raw_candidates, list) or not isinstance(iou_candidates, list):
        raise RuntimeError("SAM 3 and IoU results must both contain candidate lists")
    if sam_results.get("candidate_count") != len(raw_candidates):
        raise RuntimeError("SAM 3 candidate_count does not match its candidate list")

    iou_by_index = {}
    for item in iou_candidates:
        if not isinstance(item, dict):
            raise RuntimeError("IoU candidate entries must be objects")
        index = item.get("candidate_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RuntimeError(f"Invalid IoU candidate_index: {index!r}")
        if index in iou_by_index:
            raise RuntimeError(f"Duplicate IoU candidate_index {index}")
        iou_by_index[index] = item

    candidates = []
    seen_indices = set()
    robot_candidates_filtered: list[int] = []
    unresolved_iou_candidates_filtered: list[int] = []
    body_categories: dict[int, str] = {}
    geom_identities: dict[int, tuple[int, str, str]] = {}
    if geometry_inventory_path is not None:
        inventory = _load_json(geometry_inventory_path, "Bridge geometry inventory")
        if (
            inventory.get("status") != "complete"
            or inventory.get("robot_classification_status") != "available"
            or not isinstance(inventory.get("geometries"), list)
        ):
            raise RuntimeError("Bridge geometry inventory is not complete for candidate filtering")
        for geometry in inventory["geometries"]:
            if not isinstance(geometry, dict):
                raise RuntimeError("Bridge geometry inventory contains an invalid geometry entry")
            geom_id = geometry.get("geom_id")
            body_id = geometry.get("body_id")
            category = geometry.get("category")
            geom_name = geometry.get("geom_name")
            body_name = geometry.get("body_name")
            if (
                isinstance(geom_id, bool)
                or not isinstance(geom_id, int)
                or isinstance(body_id, bool)
                or not isinstance(body_id, int)
                or category not in ("robot", "scene")
                or not isinstance(geom_name, str)
                or not isinstance(body_name, str)
                or geom_id in geom_identities
            ):
                raise RuntimeError("Bridge geometry inventory has invalid body/category data")
            geom_identities[geom_id] = (body_id, geom_name, category)
            previous = body_categories.setdefault(body_id, category)
            if previous != category:
                raise RuntimeError("Bridge geometry inventory has inconsistent body categories")
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
            raise RuntimeError(
                f"Candidate {index} Mask size {mask.shape} does not match RGB "
                f"size {(height, width)}"
            )
        iou_entry = iou_by_index.get(index)
        if iou_entry is None:
            raise RuntimeError(f"IoU results are missing SAM candidate {index}")
        recorded_score = iou_entry.get("sam_score")
        if not isinstance(recorded_score, (int, float)) or not math.isclose(
            float(recorded_score), float(score), rel_tol=0.0, abs_tol=1e-8
        ):
            raise RuntimeError(f"IoU result score does not match candidate {index}")
        recorded_mask = iou_entry.get("mask_file")
        if recorded_mask:
            recorded_mask_path = _resolve_path(
                recorded_mask, iou_results_path.parent, f"IoU candidate {index} Mask"
            )
            if recorded_mask_path.resolve() != mask_path.resolve():
                raise RuntimeError(f"IoU result references a different Mask for candidate {index}")
        if iou_entry.get("status") not in ("matched_by_iou", "unresolved"):
            raise RuntimeError(
                f"Unsupported IoU status for candidate {index}: {iou_entry.get('status')!r}"
            )
        best_match = iou_entry.get("best_match")
        matched_body_id = best_match.get("body_id") if isinstance(best_match, dict) else None
        if geometry_inventory_path is not None and iou_entry.get("status") == "matched_by_iou":
            if not isinstance(best_match, dict):
                raise RuntimeError(f"IoU matched candidate {index} has no best_match")
            match_geom_ids = best_match.get("geom_ids")
            match_geom_names = best_match.get("geom_names")
            if (
                not isinstance(match_geom_ids, list)
                or not match_geom_ids
                or not isinstance(match_geom_names, list)
                or len(match_geom_names) != len(match_geom_ids)
            ):
                raise RuntimeError(f"IoU candidate {index} has malformed physical geometry identity")
            for geom_id, geom_name in zip(match_geom_ids, match_geom_names):
                identity = geom_identities.get(geom_id) if isinstance(geom_id, int) else None
                if (
                    identity is None
                    or identity[0] != matched_body_id
                    or identity[1] != geom_name
                ):
                    raise RuntimeError(
                        f"IoU candidate {index} disagrees with the active Bridge geometry inventory"
                    )
        physical_category = (
            body_categories.get(matched_body_id)
            if isinstance(matched_body_id, int) and not isinstance(matched_body_id, bool)
            else None
        )
        if geometry_inventory_path is not None:
            if iou_entry.get("status") != "matched_by_iou":
                unresolved_iou_candidates_filtered.append(index)
                continue
            if physical_category == "robot":
                # The active model hierarchy, not a geom-name prefix or SAM
                # score, establishes robot membership.
                robot_candidates_filtered.append(index)
                continue
        candidates.append(
            {
                "candidate_index": index,
                "sam_score": float(score),
                "mask": mask,
                "mask_file": str(mask_path.resolve()),
                "iou_result": iou_entry,
                "physical_category": physical_category,
            }
        )
    if set(iou_by_index) != seen_indices:
        raise RuntimeError("SAM 3 and IoU candidate index sets do not match")
    return target, {
        "teacher": teacher,
        "sam": sam_results,
        "iou": iou_results,
        "geometry_inventory_path": str(geometry_inventory_path.resolve()) if geometry_inventory_path else None,
        "robot_candidates_filtered": robot_candidates_filtered,
        "unresolved_iou_candidates_filtered": unresolved_iou_candidates_filtered,
    }, rgb, candidates


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = mask.copy()
    height, width = mask.shape
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            interior &= padded[dy : dy + height, dx : dx + width]
    return mask & ~interior


def _candidate_color(index: int) -> tuple[int, int, int]:
    return (
        (index * 73 + 61) % 210 + 35,
        (index * 151 + 127) % 210 + 35,
        (index * 199 + 193) % 210 + 35,
    )


def _annotate_candidates(rgb: np.ndarray, candidates: list[dict], output_path: Path) -> np.ndarray:
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for candidate in candidates:
        index = candidate["candidate_index"]
        mask = candidate["mask"]
        color = _candidate_color(index)
        edge_y, edge_x = np.where(_edge_mask(mask))
        if edge_x.size:
            draw.point(list(zip(edge_x.tolist(), edge_y.tolist())), fill=color)
            label_x = int(np.min(np.where(mask)[1]))
            label_y = int(np.min(np.where(mask)[0]))
        else:
            label_x = 3
            label_y = 3 + 18 * index
        label = str(index)
        draw.text(
            (label_x, label_y),
            label,
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    image.save(output_path)
    return np.asarray(image, dtype=np.uint8)


def _teacher_object_context(target: dict, teacher: dict) -> dict:
    related_rules = [
        rule for rule in teacher["rules"] if rule["target_id"] == target["id"]
    ]
    return {
        "object_id": target["id"],
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


def _build_user_prompt(
    target: dict,
    teacher: dict,
    candidate_indices: list[int],
    task_instruction: str | None = None,
) -> str:
    context = _teacher_object_context(target, teacher)
    instruction = (
        f"Task instruction: {task_instruction.strip()}\n"
        if isinstance(task_instruction, str) and task_instruction.strip()
        else ""
    )
    return (
        "Determine whether the Teacher object shown below corresponds to one of "
        "the numbered candidate regions in the two attached views of one RGB frame.\n"
        f"{instruction}"
        "Teacher object semantics (Safety IR v2):\n"
        f"{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}\n"
        f"Candidate indices shown in the image: {candidate_indices}\n"
        "Image 1 is the completely unmodified original RGB; use it to judge genuine "
        "appearance, including color and material. Image 2 is the same frame with "
        "thin candidate boundaries and numbers added only to locate regions. The "
        "numbers and boundaries are artificial location aids, not appearance "
        "evidence, and must not be used to judge color or material. Use the visible "
        "appearance in Image 1 together with the supplied name/description. The "
        "numbers do not rank candidates. Return "
        "resolved with the matching candidate index only when visual evidence is "
        "clear; otherwise return unresolved with candidate_index null. Give a short reason."
    )


def _response_schema(object_id: str, candidate_indices: list[int]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["object_id", "candidate_index", "reason", "status"],
        "properties": {
            "object_id": {"type": "string", "const": object_id},
            "candidate_index": {"enum": [*candidate_indices, None]},
            "reason": {"type": "string", "minLength": 1},
            "status": {"type": "string", "enum": ["resolved", "unresolved"]},
        },
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_disambiguation_output(
    raw: str, object_id: str, candidate_indices: list[int]
) -> dict:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid VLM JSON response: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("VLM response must be a JSON object")
    expected_fields = {"object_id", "candidate_index", "reason", "status"}
    if set(value) != expected_fields:
        raise ValueError(
            "VLM response fields must be exactly "
            f"{sorted(expected_fields)}; received {sorted(value)}"
        )
    if value["object_id"] != object_id:
        raise ValueError("VLM response object_id does not match requested Teacher object")
    status = value["status"]
    if status not in ("resolved", "unresolved"):
        raise ValueError("VLM status must be 'resolved' or 'unresolved'")
    candidate_index = value["candidate_index"]
    if status == "resolved":
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index not in candidate_indices
        ):
            raise ValueError("resolved output must reference an available candidate_index")
    elif candidate_index is not None:
        raise ValueError("unresolved output must set candidate_index to null")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("VLM reason must be a non-empty string")
    return value


def _validate_physical_match(iou_result: dict, physical_objects: dict) -> dict:
    if iou_result.get("status") != "matched_by_iou":
        return {"status": "unresolved", "reason": iou_result.get("unresolved_reason")}
    best_match = iou_result.get("best_match")
    if not isinstance(best_match, dict):
        raise RuntimeError("IoU candidate marked matched_by_iou but has no best_match")
    physical_id = best_match.get("physical_object_id")
    physical = physical_objects.get(physical_id)
    if physical is None:
        raise RuntimeError("IoU best_match references an unknown physical object")
    body_id = best_match.get("body_id")
    geom_ids = best_match.get("geom_ids")
    iou = best_match.get("iou")
    if (
        isinstance(body_id, bool)
        or not isinstance(body_id, int)
        or not isinstance(geom_ids, list)
        or not geom_ids
        or any(isinstance(item, bool) or not isinstance(item, int) for item in geom_ids)
        or isinstance(iou, bool)
        or not isinstance(iou, (int, float))
        or not math.isfinite(iou)
        or not 0 < iou <= 1
    ):
        raise RuntimeError("IoU best_match has invalid Body/Geom IDs or IoU")
    if body_id != physical.get("body_id") or geom_ids != physical.get("geom_ids"):
        raise RuntimeError("IoU best_match disagrees with physical_objects metadata")
    return {
        "status": "matched_by_iou",
        "physical_object_id": physical_id,
        "body_id": body_id,
        "body_name": physical.get("body_name"),
        "geom_ids": geom_ids,
        "geom_names": physical.get("geom_names"),
        "iou": float(iou),
    }


def _request_vlm(
    backend: Any,
    prompt: str,
    original_rgb: np.ndarray,
    annotated_rgb: np.ndarray,
    object_id: str,
    candidate_indices: list[int],
) -> str:
    required = ("model", "temperature", "max_tokens", "_post_payload")
    if any(not hasattr(backend, name) for name in required):
        raise TeacherBackendError(
            "This Smoke requires the configured OpenAI-compatible Teacher Backend"
        )
    frames = []
    for rgb in (original_rgb, annotated_rgb):
        height, width = rgb.shape[:2]
        frames.append(
            RGBFrame(
                camera_name="agentview_image",
                width=width,
                height=height,
                rgb_bytes=np.ascontiguousarray(rgb).tobytes(),
            )
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "text",
                    "text": "Image 1: completely unmodified original RGB; use this image for color and material evidence.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": _rgb_frame_to_data_url(frames[0])},
                },
                {
                    "type": "text",
                    "text": "Image 2: same RGB with thin candidate boundaries and numbers added only as location aids; do not use these markings for color or material judgments.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": _rgb_frame_to_data_url(frames[1])},
                },
            ],
        },
    ]
    payload = {
        "model": backend.model,
        "messages": messages,
        "temperature": backend.temperature,
        "max_tokens": backend.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "gs_opd_visual_candidate_disambiguation_v1",
                "schema": _response_schema(object_id, candidate_indices),
            },
        },
    }
    thinking_enabled = getattr(backend, "thinking_enabled", None)
    if thinking_enabled is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": thinking_enabled}
    # This uses the existing configured backend transport. The request is made
    # exactly once; no repair or unstructured fallback is attempted.
    return backend._post_payload(payload)


def run_smoke(
    teacher_json: Path,
    object_id: str,
    sam3_results_path: Path,
    iou_results_path: Path,
    output_dir: Path,
    backend: Any | None = None,
    geometry_inventory_path: Path | None = None,
    task_instruction: str | None = None,
) -> dict:
    target, loaded, rgb, candidates = _load_case(
        teacher_json,
        object_id,
        sam3_results_path,
        iou_results_path,
        geometry_inventory_path=geometry_inventory_path,
    )
    teacher = loaded["teacher"]
    sam_results = loaded["sam"]
    iou_results = loaded["iou"]
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "original_rgb.png"
    Image.fromarray(rgb.copy()).save(original_path)
    annotated_path = output_dir / "numbered_candidates.png"
    annotated_rgb = _annotate_candidates(rgb, candidates, annotated_path)
    candidate_indices = [item["candidate_index"] for item in candidates]

    if not candidates:
        decision = {
            "object_id": object_id,
            "candidate_index": None,
            "reason": "SAM 3 returned no candidate masks.",
            "status": "unresolved",
        }
        raw_output = None
    else:
        if backend is None:
            backend = create_backend_from_env()
        prompt = _build_user_prompt(target, teacher, candidate_indices, task_instruction)
        started = time.perf_counter()
        raw_output = _request_vlm(
            backend, prompt, rgb, annotated_rgb, object_id, candidate_indices
        )
        elapsed_s = time.perf_counter() - started
        if not isinstance(raw_output, str):
            raise TeacherBackendError("Teacher backend returned non-text output")
        raw_path = output_dir / "semantic_disambiguation_raw.txt"
        raw_path.write_text(raw_output, encoding="utf-8")
        decision = parse_disambiguation_output(raw_output, object_id, candidate_indices)

    selected_iou = None
    association = {"status": "unresolved", "reason": "semantic_decision_unresolved"}
    if decision["status"] == "resolved":
        selected_iou = next(
            item
            for item in iou_results["candidates"]
            if item["candidate_index"] == decision["candidate_index"]
        )
        physical_objects = {
            item["physical_object_id"]: item
            for item in iou_results.get("physical_objects", [])
            if isinstance(item, dict) and isinstance(item.get("physical_object_id"), str)
        }
        association = _validate_physical_match(selected_iou, physical_objects)

    if decision["status"] == "unresolved":
        overall_status = "unresolved"
    elif association["status"] == "matched_by_iou":
        overall_status = "resolved"
    else:
        overall_status = "candidate_resolved_physical_unresolved"

    result = {
        "status": overall_status,
        "semantic_decision": decision,
        "teacher_object": {
            "object_id": target["id"],
            "name": target["name"],
            "role": target["role"],
        },
        "selected_candidate": (
            {
                "candidate_index": decision["candidate_index"],
                "sam_score": next(
                    item["sam_score"]
                    for item in candidates
                    if item["candidate_index"] == decision["candidate_index"]
                ),
                "mask_file": next(
                    item["mask_file"]
                    for item in candidates
                    if item["candidate_index"] == decision["candidate_index"]
                ),
            }
            if decision["status"] == "resolved"
            else None
        ),
        "physical_association": association,
        "original_rgb_image": original_path.name,
        "annotated_image": annotated_path.name,
        "visual_input_mode": "unmodified_original_rgb_plus_thin_candidate_boundaries_and_numbers",
        "robot_candidates_filtered": loaded["robot_candidates_filtered"],
        "unresolved_iou_candidates_filtered": loaded["unresolved_iou_candidates_filtered"],
        "geometry_inventory": loaded["geometry_inventory_path"],
        "task_instruction": task_instruction,
        "sam3_results": str(sam3_results_path.resolve()),
        "iou_results": str(iou_results_path.resolve()),
        "teacher_json": str(teacher_json.resolve()),
        "vlm_request_seconds": elapsed_s if candidates else None,
        "raw_response_file": "semantic_disambiguation_raw.txt" if raw_output is not None else None,
    }
    (output_dir / "semantic_disambiguation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-json", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--sam3-results", type=Path, required=True)
    parser.add_argument("--iou-results", type=Path, required=True)
    parser.add_argument(
        "--geometry-inventory",
        type=Path,
        help="optional active Bridge inventory used only to remove robot-part masks before VLM selection",
    )
    parser.add_argument(
        "--instruction",
        help="optional task instruction used with the Teacher semantics during visual candidate disambiguation",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    required_files = [
        ("Teacher v2 JSON", args.teacher_json),
        ("SAM 3 results", args.sam3_results),
        ("MuJoCo IoU results", args.iou_results),
    ]
    if args.geometry_inventory is not None:
        required_files.append(("Bridge geometry inventory", args.geometry_inventory))
    for label, path in required_files:
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    try:
        result = run_smoke(
            args.teacher_json,
            args.object_id,
            args.sam3_results,
            args.iou_results,
            args.out_dir,
            geometry_inventory_path=args.geometry_inventory,
            task_instruction=args.instruction,
        )
    except (BackendConfigError, TeacherBackendError, OSError, ValueError, RuntimeError) as exc:
        print(f"VLM semantic disambiguation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Semantic disambiguation: status={result['status']}")
    print(json.dumps(result["semantic_decision"], ensure_ascii=False, indent=2))
    if result["physical_association"].get("status") == "matched_by_iou":
        association = result["physical_association"]
        print(
            f"Physical association: body_id={association['body_id']}, "
            f"geom_ids={association['geom_ids']}, IoU={association['iou']:.4f}"
        )
    elif result["selected_candidate"] is not None:
        print("Candidate resolved visually, but its physical IoU match remains unresolved.")
    print(f"Results saved under: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
