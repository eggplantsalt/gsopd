"""Capture same-state agentview RGB and MuJoCo Geom ID segmentation."""

import argparse
import json
import sys
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
from PIL import Image


AUTHKEY = b"gs-opd-local"


def _decode_payload(payload, expected_shape, expected_dtype, label):
    if not isinstance(payload, dict):
        raise RuntimeError(f"Bridge response is missing {label} payload")
    shape = tuple(payload.get("shape", ()))
    dtype = payload.get("dtype")
    data = payload.get("data")
    if shape != expected_shape or dtype != expected_dtype:
        raise RuntimeError(
            f"Unsupported {label} payload: shape={shape}, dtype={dtype!r}; "
            f"expected {expected_shape} {expected_dtype}"
        )
    if not isinstance(data, bytes):
        raise RuntimeError(f"Invalid byte payload for {label}")
    array = np.frombuffer(data, dtype=np.dtype(expected_dtype))
    expected_size = int(np.prod(expected_shape))
    if array.size != expected_size:
        raise RuntimeError(
            f"Invalid {label} payload length: {array.size} values, "
            f"expected {expected_size}"
        )
    return array.reshape(expected_shape).copy()


def _compare_rgb(rgb, image_path):
    try:
        with Image.open(image_path) as image_file:
            existing = np.asarray(image_file.convert("RGB"))
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"Cannot read comparison RGB {image_path}: {type(exc).__name__}: {exc}",
        }

    if existing.shape != rgb.shape:
        return {
            "status": "unaligned",
            "error": (
                f"Comparison RGB shape {existing.shape} does not match "
                f"rendered RGB shape {rgb.shape}"
            ),
        }
    difference = np.abs(existing.astype(np.int16) - rgb.astype(np.int16))
    differing_pixels = int(np.count_nonzero(np.any(difference != 0, axis=2)))
    max_abs_channel_difference = int(difference.max(initial=0))
    return {
        "status": "matched" if differing_pixels == 0 else "mismatch",
        "file": str(Path(image_path).resolve()),
        "pixel_equal": differing_pixels == 0,
        "differing_pixel_count": differing_pixels,
        "max_abs_channel_difference": max_abs_channel_difference,
    }


def _overlay_segmentation(rgb, geom_ids, visible_geoms):
    overlay = rgb.astype(np.float32).copy()
    for geom in visible_geoms:
        geom_id = int(geom["geom_id"])
        color = np.asarray(
            [
                (geom_id * 73 + 61) % 256,
                (geom_id * 151 + 127) % 256,
                (geom_id * 199 + 193) % 256,
            ],
            dtype=np.float32,
        )
        mask = geom_ids == geom_id
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * color

    non_geom_mask = geom_ids == -2
    non_geom_color = np.asarray([255, 0, 255], dtype=np.float32)
    overlay[non_geom_mask] = (
        0.45 * overlay[non_geom_mask] + 0.55 * non_geom_color
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_capture(response, output_dir, compare_rgb_path=None):
    if not isinstance(response, dict):
        raise RuntimeError("Bridge response is not an object")
    if "error" in response:
        raise RuntimeError(response["error"])
    if response.get("camera_name") != "agentview":
        raise RuntimeError(
            f"Unsupported camera {response.get('camera_name')!r}; expected 'agentview'"
        )
    if response.get("width") != 512 or response.get("height") != 512:
        raise RuntimeError("Bridge did not return the requested 512x512 capture")
    if response.get("physical_state_unchanged") is not True:
        raise RuntimeError("Bridge could not confirm unchanged physical state")
    if response.get("rgb_segmentation_pixel_consistency") is not True:
        raise RuntimeError("Bridge could not confirm RGB/segmentation alignment")

    images = response.get("images")
    if not isinstance(images, dict):
        raise RuntimeError("Bridge response has no image payload mapping")
    rgb = _decode_payload(images.get("rgb"), (512, 512, 3), "uint8", "RGB")
    geom_ids = _decode_payload(
        images.get("geom_ids"), (512, 512), "int32", "Geom ID map"
    )
    if np.any(geom_ids < -2):
        raise RuntimeError("Geom ID map contains an unknown negative label")

    visible_geoms = response.get("visible_geoms")
    if not isinstance(visible_geoms, list):
        raise RuntimeError("Bridge response has no visible Geom metadata")
    geom_count = response.get("model_geom_count")
    if not isinstance(geom_count, int) or geom_count < 0:
        raise RuntimeError("Bridge response has invalid model Geom count")
    if np.any(geom_ids >= geom_count):
        raise RuntimeError("Geom ID map contains IDs outside the current model")

    # Keep the renderer-native arrays for debugging, and apply the exact same
    # 180-degree rotation to RGB and labels for alignment with the SAM 3 input.
    rgb_rotated_180 = np.ascontiguousarray(np.rot90(rgb, k=2))
    geom_ids_rotated_180 = np.ascontiguousarray(np.rot90(geom_ids, k=2))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_file = "agentview_512_rgb.png"
    geom_ids_file = "geom_ids.npy"
    overlay_file = "geom_segmentation_overlay.png"
    rgb_rotated_file = "agentview_512_rgb_rotated_180.png"
    geom_ids_rotated_file = "geom_ids_rotated_180.npy"
    overlay_rotated_file = "geom_segmentation_overlay_rotated_180.png"
    Image.fromarray(rgb).save(output_dir / rgb_file)
    np.save(output_dir / geom_ids_file, geom_ids, allow_pickle=False)
    overlay = _overlay_segmentation(rgb, geom_ids, visible_geoms)
    Image.fromarray(overlay).save(output_dir / overlay_file)
    Image.fromarray(rgb_rotated_180).save(output_dir / rgb_rotated_file)
    np.save(
        output_dir / geom_ids_rotated_file,
        geom_ids_rotated_180,
        allow_pickle=False,
    )
    overlay_rotated_180 = _overlay_segmentation(
        rgb_rotated_180, geom_ids_rotated_180, visible_geoms
    )
    Image.fromarray(overlay_rotated_180).save(output_dir / overlay_rotated_file)

    comparison = (
        _compare_rgb(rgb_rotated_180, compare_rgb_path)
        if compare_rgb_path is not None
        else {"status": "not_requested"}
    )
    metadata = {
        "camera_name": "agentview",
        "render_api": response.get("render_api"),
        "width": 512,
        "height": 512,
        "color_space": "RGB",
        "physical_state_unchanged": True,
        "rgb_segmentation_pixel_consistency": True,
        "geom_id_background_value": -1,
        "non_geom_object_value": -2,
        "model_geom_count": geom_count,
        "background_pixel_count": int(np.count_nonzero(geom_ids == -1)),
        "non_geom_pixel_count": int(np.count_nonzero(geom_ids == -2)),
        "visible_geom_count": len(visible_geoms),
        "visible_geoms": visible_geoms,
        "rgb_file": rgb_file,
        "geom_ids_file": geom_ids_file,
        "overlay_file": overlay_file,
        "rgb_rotated_180_file": rgb_rotated_file,
        "geom_ids_rotated_180_file": geom_ids_rotated_file,
        "overlay_rotated_180_file": overlay_rotated_file,
        "coordinate_directions": {
            rgb_file: "MuJoCo/robosuite raw renderer row/column order",
            geom_ids_file: "MuJoCo/robosuite raw renderer row/column order",
            overlay_file: "MuJoCo/robosuite raw renderer row/column order",
            rgb_rotated_file: "raw image rotated 180 degrees; SAM 3 forward orientation",
            geom_ids_rotated_file: "raw label map rotated 180 degrees; aligned with SAM 3 forward orientation",
            overlay_rotated_file: "raw overlay rotated 180 degrees; SAM 3 forward orientation",
        },
        "compare_rgb_coordinate_direction": (
            "raw image rotated 180 degrees; SAM 3 forward orientation"
            if compare_rgb_path is not None
            else None
        ),
        "comparison": comparison,
    }
    (output_dir / "geom_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Safety Bridge host")
    parser.add_argument("--port", type=int, default=38567, help="Safety Bridge port")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--compare-rgb",
        type=Path,
        help=(
            "optional existing forward-oriented 512x512 RGB PNG to compare "
            "pixel-for-pixel against the raw capture rotated 180 degrees"
        ),
    )
    args = parser.parse_args(argv)

    try:
        connection = Client((args.host, args.port), authkey=AUTHKEY)
        try:
            connection.send({"op": "geom_segmentation"})
            response = connection.recv()
        finally:
            connection.close()
        metadata = save_capture(response, args.out_dir, args.compare_rgb)
    except Exception as exc:
        print(f"Geom segmentation capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Geom segmentation saved under: {args.out_dir.resolve()}")
    print(
        f"visible_geoms={metadata['visible_geom_count']}, "
        f"background_pixels={metadata['background_pixel_count']}, "
        f"non_geom_pixels={metadata['non_geom_pixel_count']}"
    )
    comparison = metadata["comparison"]
    if comparison["status"] == "mismatch":
        print(
            "RGB comparison failed: "
            f"{comparison['differing_pixel_count']} pixels differ; "
            f"max channel delta={comparison['max_abs_channel_difference']}",
            file=sys.stderr,
        )
        return 1
    if comparison["status"] in ("unaligned", "unavailable"):
        print(f"RGB comparison failed: {comparison['error']}", file=sys.stderr)
        return 1
    if comparison["status"] == "matched":
        print("RGB comparison: exact pixel match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
