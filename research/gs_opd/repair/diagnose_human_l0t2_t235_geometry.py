# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0.

"""Read-only MuJoCo geometry and endpoint-state audit for the Human L0T2 t=235 Snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.gs_opd.repair.task_success_first_grasp_search import (  # noqa: E402
    BRIDGE_PYTHON,
    BridgeSession,
    _PythonObjectState,
    _snapshot_state_vector,
)

SOURCE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "gs_opd_human_l0t2_curobo_transport_20261027_020247"
)
EXPECTED_MODEL_SHA = "633b8b0956f2f38e7f1c62067608f42743d0bccdc6bfeaca11976b5c7f301606"
EXPECTED_BDDL_SHA = "2c4010e44700b0d2069dcd47f489a7411ec364f31cfd69141801b49db3398af7"
INSTRUCTION = "put both moka pots on the stove"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _as_rows(rows: list[dict], wanted: set[int]) -> list[dict]:
    return [row for row in rows if int(row.get("geom_id", -1)) in wanted]


def run(port: int, output_dir: Path, robot_clearance_only: bool = False) -> dict:
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    snapshot_path = SOURCE / "terminal_runtime_snapshot.pkl"
    payload = snapshot_path.read_bytes()
    envelope = pickle.loads(payload)
    snapshot = envelope.get("snapshot", {})
    scene = envelope.get("scene_metadata", {})
    source_state = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    if envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
        raise ValueError("t235_snapshot_schema_mismatch")
    if envelope.get("model_geometry_sha256") != EXPECTED_MODEL_SHA:
        raise ValueError("t235_snapshot_model_SHA_mismatch")
    if scene.get("bddl_sha256") != EXPECTED_BDDL_SHA or scene.get("instruction") != INSTRUCTION:
        raise ValueError("t235_snapshot_BDDL_or_instruction_mismatch")
    timestep = int(snapshot.get("task_bookkeeping", {}).get("timestep", -1))
    if timestep != 235:
        raise ValueError(f"expected_t235_snapshot_got_{timestep}")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source_t235_runtime_snapshot.pkl").write_bytes(payload)
    args = argparse.Namespace(
        benchmark="obstacle_avoidance_human",
        level=0,
        task_id=2,
        init_state=0,
        seed=0,
        camera_resolution=256,
        demonstration_horizon=3600,
    )
    bridge = BridgeSession(("127.0.0.1", port), output_dir / "safety_bridge.log", args)
    report: dict = {
        "schema": "gs_opd_human_l0t2_t235_geometry_endpoint_diagnostic.v1",
        "status": "in_progress",
        "source_snapshot": {
            "path": str(snapshot_path),
            "payload_sha256": _sha(payload),
            "model_geometry_sha256": envelope.get("model_geometry_sha256"),
            "bddl_sha256": scene.get("bddl_sha256"),
            "timestep": timestep,
            "integration_state_shape": list(source_state.shape),
            "integration_state_sha256": _sha(np.ascontiguousarray(source_state).tobytes()),
        },
        "actions_issued": 0,
        "physics_substeps": 0,
    }
    try:
        bridge.start()
        bridge.request({"op": "reset"})
        fingerprint = bridge.request({"op": "compiled_model_fingerprint_details"})
        report["compiled_model_fingerprint_details"] = fingerprint
        if fingerprint.get("model_geometry_sha256") != EXPECTED_MODEL_SHA:
            raise RuntimeError("compiled_model_SHA_does_not_match_t235_snapshot")
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
        restored_state = _snapshot_state_vector(restored.get("snapshot_state", {}))
        if restored.get("status") != "restored" or not np.array_equal(restored_state, source_state):
            raise RuntimeError("t235_runtime_snapshot_restore_not_exact")
        report["restore"] = {
            "status": restored.get("status"),
            "model_geometry_sha256": restored.get("model_geometry_sha256"),
            "integration_state_exact_match": True,
            "integration_state_shape": list(restored_state.shape),
            "integration_state_sha256": _sha(np.ascontiguousarray(restored_state).tobytes()),
        }

        observed = bridge.request({"op": "observe", "include_snapshot_state": True})
        semantics = bridge.request({"op": "task_semantic_geometry_bindings"})
        inventory = bridge.request({"op": "geometry_inventory"})
        export = bridge.request({"op": "collision_scene_export"})
        if observed.get("timestep") != 235 or semantics.get("status") != "complete":
            raise RuntimeError("restored_state_or_semantic_geometry_binding_incomplete")
        bindings = semantics.get("bindings", {})
        pot = bindings["moka_pot_1"]
        bottle_names = [name for name in bindings if "tomato" in name.lower() or "paste" in name.lower()]
        bottle_rows = {name: bindings[name] for name in bottle_names}
        bottle_geom_ids = sorted({
            int(gid)
            for row in bottle_rows.values()
            for gid in row.get("geom_ids", row.get("root_geom_ids", []))
        })
        pot_geom_ids = sorted(int(x) for x in pot.get("root_geom_ids", []))
        geom_rows = inventory.get("geometries", [])
        id_rows = {int(row["geom_id"]): row for row in geom_rows if "geom_id" in row}
        selected_ids = set(pot_geom_ids) | set(bottle_geom_ids)
        report["semantic_bindings"] = {
            "status": semantics.get("status"),
            "Pot1": pot,
            "tomato_bottle_candidates": bottle_rows,
        }
        report["selected_compiled_geometries"] = _as_rows(geom_rows, selected_ids)

        distance_rows = []
        if not robot_clearance_only:
            for pot_geom in pot_geom_ids:
                for bottle_geom in bottle_geom_ids:
                    result = bridge.request({
                        "op": "geom_set_min_distance",
                        "geom_ids_a": [pot_geom],
                        "geom_ids_b": [bottle_geom],
                        "distmax_m": 1.0,
                    })
                    distance_rows.append({
                        "pot_geom": id_rows.get(pot_geom, {"geom_id": pot_geom}),
                        "bottle_geom": id_rows.get(bottle_geom, {"geom_id": bottle_geom}),
                        "distance_query": result,
                    })
        report["exact_mujoco_geom_pair_distances"] = distance_rows
        robot_geom_ids = sorted(
            int(row["geom_id"])
            for row in export.get("robot_collision_geometries", [])
        )
        report["robot_to_tomato_bottle_exact_distance"] = bridge.request({
            "op": "geom_set_min_distance",
            "geom_ids_a": robot_geom_ids,
            "geom_ids_b": bottle_geom_ids,
            "distmax_m": 10.0,
        })
        report["collision_scene_export_status"] = {
            key: export.get(key)
            for key in ["status", "query_complete", "physical_state_unchanged", "model_geometry_sha256"]
        }
        report["collision_scene_selected_meshes"] = [
            row for row in export.get("scene_collision_geometries", [])
            if set(int(x) for x in row.get("geom_ids", [row.get("geom_id")]) if x is not None) & set(bottle_geom_ids)
        ]
        report["collision_scene_selected_pot_components"] = [
            row for row in export.get("scene_collision_geometries", [])
            if set(int(x) for x in row.get("geom_ids", [row.get("geom_id")]) if x is not None) & set(pot_geom_ids)
        ]
        report["restored_runtime_state"] = observed.get("snapshot_state", {})
        report["read_only_state_unchanged"] = np.array_equal(
            _snapshot_state_vector(observed.get("snapshot_state", {})), source_state
        )
        grasp = bridge.request({
            "op": "grasp_state",
            "carried_binding": {
                "object_id": "moka_pot_1",
                "body_id": int(pot["root_body_id"]),
                "body_name": str(pot["root_body_name"]),
                "geom_ids": pot_geom_ids,
                "geom_names": list(pot.get("root_geom_names", [])),
            },
            "expected_instruction": INSTRUCTION,
        })
        report["Pot1_grasp_state"] = grasp
        report["status"] = "complete" if report["read_only_state_unchanged"] and export.get("physical_state_unchanged") is True else "incomplete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        bridge.close()
        (output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=38779)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robot-clearance-only", action="store_true")
    args = parser.parse_args()
    result = run(args.port, args.output_dir, args.robot_clearance_only)
    print(json.dumps({k: result.get(k) for k in ["status", "error", "restore", "collision_scene_export_status", "read_only_state_unchanged", "output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
