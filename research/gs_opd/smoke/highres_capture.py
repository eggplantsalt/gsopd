"""Save raw and 180-degree Safety Bridge renders from its primary camera."""

import argparse
import json
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
from PIL import Image


AUTHKEY = b"gs-opd-local"
IMAGE_FILES = {
    "raw_rgb": "agentview_512_raw.png",
    "rotated_180_rgb": "agentview_512_rotated_180.png",
}


def _decode_rgb(payload, key):
    if not isinstance(payload, dict):
        raise RuntimeError(f"Bridge response is missing image payload {key!r}")
    shape = payload.get("shape")
    if tuple(shape or ()) != (512, 512, 3) or payload.get("dtype") != "uint8":
        raise RuntimeError(
            f"Unexpected {key} image metadata: shape={shape!r}, "
            f"dtype={payload.get('dtype')!r}"
        )
    data = payload.get("data")
    if not isinstance(data, bytes) or len(data) != 512 * 512 * 3:
        raise RuntimeError(f"Invalid RGB byte payload for {key!r}")
    return np.frombuffer(data, dtype=np.uint8).reshape(512, 512, 3)


def save_capture(response, output_dir):
    if not isinstance(response, dict):
        raise RuntimeError("Bridge response is not an object")
    if "error" in response:
        raise RuntimeError(response["error"])
    if response.get("physical_state_unchanged") is not True:
        raise RuntimeError("Bridge did not confirm that physical state was unchanged")
    if response.get("width") != 512 or response.get("height") != 512:
        raise RuntimeError("Bridge did not return the requested 512x512 render")

    images = response.get("images")
    if not isinstance(images, dict):
        raise RuntimeError("Bridge response has no image mapping")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_metadata = {}
    for key, filename in IMAGE_FILES.items():
        rgb = _decode_rgb(images.get(key), key)
        Image.fromarray(rgb).save(output_dir / filename)
        image_metadata[key] = {
            "file": filename,
            "width": 512,
            "height": 512,
            "color_space": "RGB",
            "dtype": "uint8",
        }

    metadata = {
        "camera_name": response.get("camera_name"),
        "render_api": response.get("render_api"),
        "physical_state_unchanged": True,
        "raw_orientation": response.get("raw_orientation"),
        "rotated_orientation": response.get("rotated_orientation"),
        "images": image_metadata,
    }
    (output_dir / "capture_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Safety Bridge host")
    parser.add_argument("--port", type=int, default=38567, help="Safety Bridge port")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    connection = Client((args.host, args.port), authkey=AUTHKEY)
    try:
        connection.send({"op": "render_highres"})
        response = connection.recv()
    finally:
        connection.close()

    metadata = save_capture(response, args.out_dir)
    print(f"Saved 512x512 RGB captures under: {args.out_dir.resolve()}")
    for image in metadata["images"].values():
        print(f"  {image['file']} ({image['width']}x{image['height']} RGB uint8)")
    print(f"Metadata: {args.out_dir / 'capture_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
