"""Save the active Safety Bridge model's MuJoCo geometry inventory as JSON."""
import argparse
import json
from multiprocessing.connection import Client
from pathlib import Path


AUTHKEY = b"gs-opd-local"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Safety Bridge host")
    parser.add_argument("--port", type=int, default=38567, help="Safety Bridge port")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("geometry_inventory.json"),
        help="output JSON path (default: ./geometry_inventory.json)",
    )
    args = parser.parse_args()

    connection = Client((args.host, args.port), authkey=AUTHKEY)
    try:
        connection.send({"op": "geometry_inventory"})
        inventory = connection.recv()
    finally:
        connection.close()

    if not isinstance(inventory, dict):
        raise RuntimeError("Bridge returned a non-object geometry inventory response")
    if "error" in inventory:
        raise RuntimeError(inventory["error"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    geometries = inventory.get("geometries", [])
    counts = {category: 0 for category in ("robot", "scene", "unknown")}
    for geometry in geometries:
        category = geometry.get("category", "unknown")
        counts[category if category in counts else "unknown"] += 1
    print(
        f"Geometry inventory: status={inventory.get('status', 'unknown')}, "
        f"geoms={len(geometries)}, robot={counts['robot']}, "
        f"scene={counts['scene']}, unknown={counts['unknown']}"
    )
    print(f"Saved JSON: {args.out.resolve()}")
    if inventory.get("robot_classification_status") != "available":
        print("Robot/scene classification is unavailable; entries are marked unknown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
