"""Read a real Safety Bridge observation and query the configured VLM Teacher."""

from __future__ import annotations

import argparse
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
    create_backend_from_env,
)
from .schema import CAMERA_NAMES, TeacherOutputError, parse_teacher_output

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Safety Bridge host")
    parser.add_argument("--port", type=int, default=38567, help="Safety Bridge port")
    parser.add_argument(
        "--raw-out",
        type=Path,
        help="save the raw VLM response text to this JSON-named file before validation",
    )
    args = parser.parse_args(argv)

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

    request_started = time.perf_counter()
    try:
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
        output = parse_teacher_output(raw_output)
    except (TeacherOutputError, ValueError) as exc:
        print(f"Teacher Smoke failed: {type(exc).__name__}: {exc}")
        return 1

    print("Teacher output passed schema validation:")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
