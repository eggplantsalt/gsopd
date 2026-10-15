"""Read a real Safety Bridge observation and query the configured VLM Teacher."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

from .backend import (
    BackendConfigError,
    RGBFrame,
    TeacherBackendError,
    _rgb_frame_to_data_url,
    create_backend_from_env,
)
from .schema import CAMERA_NAMES, TeacherOutputError, parse_teacher_output
from .schema_v2 import parse_teacher_output_v2

AUTHKEY = b"gs-opd-local"


def _receive_reset(connection: Any) -> dict[str, Any]:
    connection.send({"op": "reset"})
    result = connection.recv()
    if not isinstance(result, dict):
        raise RuntimeError("Safety Bridge reset response is not an object")
    if "error" in result:
        raise RuntimeError(f"Safety Bridge reset failed: {result['error']}")
    return result


def _decode_bridge_observation(result: dict[str, Any]) -> tuple[str, list[RGBFrame]]:
    instruction = result.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise RuntimeError("Safety Bridge response has no task instruction")
    image_payloads = result.get("images")
    if not isinstance(image_payloads, dict):
        raise RuntimeError("Safety Bridge response has no image mapping")

    frames = []
    for camera_name in CAMERA_NAMES:
        payload = image_payloads.get(camera_name)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Safety Bridge observation is missing {camera_name}")
        shape = payload.get("shape")
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) != 3
            or not all(isinstance(size, int) and size > 0 for size in shape)
            or shape[2] != 3
        ):
            raise RuntimeError(
                f"{camera_name} must be an H×W×3 RGB image; received shape={shape!r}"
            )
        if payload.get("dtype") != "uint8":
            raise RuntimeError(
                f"{camera_name} must be raw uint8 RGB; received dtype={payload.get('dtype')!r}"
            )
        raw_bytes = payload.get("data")
        if not isinstance(raw_bytes, bytes):
            raise RuntimeError(f"{camera_name} image payload is not bytes")
        height, width, _channels = shape
        frames.append(
            RGBFrame(
                camera_name=camera_name,
                width=width,
                height=height,
                rgb_bytes=raw_bytes,
            )
        )
    return instruction, frames


def _save_capture(
    output_dir: Path, instruction: str, images: list[RGBFrame]
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_metadata = {}
    for frame in images:
        data_url = _rgb_frame_to_data_url(frame)
        header, separator, encoded_png = data_url.partition(",")
        if header != "data:image/png;base64" or not separator:
            raise RuntimeError("Backend RGB encoder returned an invalid PNG data URL")
        filename = f"{frame.camera_name}.png"
        (output_dir / filename).write_bytes(base64.b64decode(encoded_png))
        image_metadata[frame.camera_name] = {
            "file": filename,
            "width": frame.width,
            "height": frame.height,
        }

    metadata = {
        "instruction": instruction,
        "color_space": "RGB",
        "dtype": "uint8",
        "encoding": "PNG",
        "images": image_metadata,
    }
    (output_dir / "capture_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Safety Bridge host")
    parser.add_argument("--port", type=int, default=38567, help="Safety Bridge port")
    parser.add_argument(
        "--raw-out",
        type=Path,
        help="save the raw VLM response text to this JSON-named file before validation",
    )
    parser.add_argument(
        "--images-out-dir",
        type=Path,
        help="save the raw RGB camera images and capture metadata in this directory",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="save the Bridge observation and exit without calling the VLM",
    )
    parser.add_argument(
        "--teacher-version",
        choices=("v1", "v2"),
        default="v1",
        help="Safety Teacher prompt and output schema version (default: v1)",
    )
    args = parser.parse_args(argv)

    if args.capture_only and args.images_out_dir is None:
        parser.error("--capture-only requires --images-out-dir")

    backend = None
    if not args.capture_only:
        try:
            backend = create_backend_from_env()
        except BackendConfigError as exc:
            print(f"Teacher configuration error: {exc}", file=sys.stderr)
            return 2

    try:
        connection = Client((args.host, args.port), authkey=AUTHKEY)
    except OSError as exc:
        print(f"Cannot connect to Safety Bridge at {args.host}:{args.port}: {exc}")
        return 1
    try:
        bridge_result = _receive_reset(connection)
    except Exception as exc:
        print(f"Safety Bridge error: {type(exc).__name__}: {exc}")
        return 1
    finally:
        connection.close()

    try:
        instruction, images = _decode_bridge_observation(bridge_result)
    except (ValueError, RuntimeError) as exc:
        print(f"Teacher Smoke failed: {type(exc).__name__}: {exc}")
        return 1

    if args.images_out_dir is not None:
        try:
            capture_metadata = _save_capture(
                args.images_out_dir, instruction, images
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Cannot save Bridge observation: {type(exc).__name__}: {exc}")
            return 1
        print(f"Camera images saved under: {args.images_out_dir}")
        for camera_name, image_info in capture_metadata["images"].items():
            print(
                f"  {camera_name}: {image_info['file']} "
                f"({image_info['width']}x{image_info['height']}, RGB uint8)"
            )
        print(
            "Task instruction and image sizes saved: "
            f"{args.images_out_dir / 'capture_metadata.json'}"
        )

    if args.capture_only:
        print("Capture-only complete; VLM was not called.")
        return 0

    request_started = time.perf_counter()
    try:
        if args.teacher_version == "v2":
            generate_v2 = getattr(backend, "generate_v2", None)
            if not callable(generate_v2):
                raise TeacherBackendError(
                    "Configured Teacher backend does not support Safety IR v2"
                )
            raw_output = generate_v2(instruction, images)
        else:
            raw_output = backend.generate(instruction, images)
    except (TeacherBackendError, ValueError, RuntimeError) as exc:
        elapsed_s = time.perf_counter() - request_started
        print(f"VLM request elapsed: {elapsed_s:.3f}s")
        print(f"Teacher Smoke failed: {type(exc).__name__}: {exc}")
        return 1
    elapsed_s = time.perf_counter() - request_started
    print(f"VLM request elapsed: {elapsed_s:.3f}s")

    if args.raw_out is not None:
        try:
            args.raw_out.parent.mkdir(parents=True, exist_ok=True)
            args.raw_out.write_text(raw_output, encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            print(f"Cannot save raw Teacher output to {args.raw_out}: {exc}")
            return 1
        print(f"Raw Teacher output saved: {args.raw_out}")

    try:
        if args.teacher_version == "v2":
            output = parse_teacher_output_v2(raw_output)
        else:
            output = parse_teacher_output(raw_output)
    except (TeacherOutputError, ValueError) as exc:
        print(f"Teacher Smoke failed: {type(exc).__name__}: {exc}")
        return 1

    print("Teacher output passed schema validation:")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
