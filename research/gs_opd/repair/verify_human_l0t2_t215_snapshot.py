# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0.

"""Verify exact Human L0T2 t=215 Runtime Snapshot recovery after the legacy reset sequence.

This is a read-only recovery check. It performs Bridge startup, the same
additional reset used by the historical executor, strict model fingerprint
checks, Runtime Snapshot restore, and read-only Pot1 grasp verification. It
does not issue actions or advance MuJoCo physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.gs_opd.repair.task_success_first_grasp_search import (
    BRIDGE_PYTHON,
    BridgeSession,
    _PythonObjectState,
)


DEFAULT_SNAPSHOT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "gs_opd_human_l0t2_collision_aware_rrt_20261026_run1/"
    "obstacle_avoidance_human_L0T2/stage_00/"
    "task_success_first_executor_obstacle_avoidance_human_L0T2_"
    "20261026_224008_781508768/expert_native_goal_stage_control/"
    "placement_branch_00_region_layout_04/"
    "native_goal_transport_start_runtime_snapshot.pkl"
)
EXPECTED_MODEL_SHA256 = (
    "633b8b0956f2f38e7f1c62067608f42743d0bccdc6bfeaca11976b5c7f301606"
)
EXPECTED_BDDL_SHA256 = (
    "2c4010e44700b0d2069dcd47f489a7411ec364f31cfd69141801b49db3398af7"
)
INSTRUCTION = "put both moka pots on the stove"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_identity(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=False,
            capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    relative = path.relative_to(REPO_ROOT)
    return {
        "path": str(relative),
        "sha256": _sha256(path.read_bytes()),
        "last_change_commit": git("log", "-1", "--format=%H", "--", str(relative)),
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--port", type=int, default=38745)
    parser.add_argument("--horizon", type=int, default=3600)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    snapshot_path = args.snapshot.resolve()
    payload = snapshot_path.read_bytes()
    envelope = pickle.loads(payload)
    scene = envelope.get("scene_metadata", {})
    snapshot = envelope.get("snapshot", {})
    source_state = np.asarray(snapshot.get("mujoco_state"), dtype=np.float64)
    if envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
        raise ValueError("source_Runtime_Snapshot_schema_mismatch")
    if envelope.get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("source_Runtime_Snapshot_model_SHA_mismatch")
    expected_scene = {
        "benchmark": "obstacle_avoidance_human",
        "level": 0,
        "task_id": 2,
        "init_state_index": 0,
        "environment_seed": 0,
        "bddl_sha256": EXPECTED_BDDL_SHA256,
        "instruction": INSTRUCTION,
    }
    for key, expected in expected_scene.items():
        if scene.get(key) != expected:
            raise ValueError(f"source_Runtime_Snapshot_{key}_mismatch")
    timestep = int(snapshot.get("task_bookkeeping", {}).get("timestep", -1))
    if timestep != 215 or source_state.ndim != 1 or not np.isfinite(source_state).all():
        raise ValueError("source_Runtime_Snapshot_timestep_or_state_invalid")

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(
            f"/opt/data/private/gs_opd/rollouts/"
            f"gs_opd_human_l0t2_t215_bridge_recovery_{stamp}"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source_runtime_snapshot.pkl").write_bytes(payload)

    bridge_args = argparse.Namespace(
        benchmark="obstacle_avoidance_human",
        level=0,
        task_id=2,
        init_state=0,
        seed=0,
        camera_resolution=256,
        demonstration_horizon=int(args.horizon),
    )
    report: dict[str, Any] = {
        "schema": "gs_opd_t215_bridge_recovery.v1",
        "status": "in_progress",
        "repo_root": str(REPO_ROOT),
        "repo_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "relevant_code": [
            _git_identity(REPO_ROOT / "research/gs_opd/smoke/safety_bridge.py"),
            _git_identity(REPO_ROOT / "research/gs_opd/repair/task_success_first_grasp_search.py"),
            _git_identity(REPO_ROOT / "research/gs_opd/repair/task_success_first_executor.py"),
        ],
        "snapshot": {
            "source_path": str(snapshot_path),
            "archived_copy": str(output_dir / "source_runtime_snapshot.pkl"),
            "payload_sha256": _sha256(payload),
            "model_geometry_sha256": envelope["model_geometry_sha256"],
            "bddl_sha256": scene["bddl_sha256"],
            "timestep": timestep,
            "integration_state_shape": list(source_state.shape),
            "integration_state_sha256": _sha256(np.ascontiguousarray(source_state).tobytes()),
        },
        "launch": {
            "python": str(BRIDGE_PYTHON),
            "benchmark": "obstacle_avoidance_human",
            "level": 0,
            "task_id": 2,
            "init_state": 0,
            "seed": 0,
            "camera_resolution": 256,
            "demonstration_horizon": int(args.horizon),
            "startup_reset_sequence": "seed -> env.reset -> set_init_state",
            "required_executor_sequence": (
                "BridgeSession.start -> Bridge reset -> compiled fingerprint check -> "
                "exact Runtime Snapshot restore"
            ),
        },
        "fingerprints": {},
        "component_differences": [],
        "restore": {},
        "validation": {},
        "actions_issued": 0,
        "mujoco_physics_substeps": 0,
    }

    bridge = BridgeSession(
        ("127.0.0.1", int(args.port)), output_dir / "safety_bridge.log", bridge_args
    )
    try:
        bridge.start()
        pre_reset = bridge.request({"op": "compiled_model_fingerprint_details"})
        reset = bridge.request({"op": "reset"})
        post_reset = bridge.request({"op": "compiled_model_fingerprint_details"})
        report["fingerprints"] = {
            "after_bridge_start_seed_reset_set_init_state": pre_reset,
            "after_historical_executor_additional_reset": post_reset,
            "source_expected_model_geometry_sha256": EXPECTED_MODEL_SHA256,
        }
        pre_components = pre_reset.get("component_digests", {})
        post_components = post_reset.get("component_digests", {})
        report["component_differences"] = [
            {
                "field": key,
                "before": pre_components.get(key),
                "after": post_components.get(key),
            }
            for key in sorted(set(pre_components) | set(post_components))
            if pre_components.get(key) != post_components.get(key)
        ]
        report["reset_scene_metadata"] = reset.get("scene_metadata")
        if post_reset.get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
            report["status"] = "model_fingerprint_not_reproduced"
            report["restore"] = {
                "status": "not_attempted",
                "reason": "post_reset_model_fingerprint_mismatch",
            }
            return report

        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
        restored_state = np.asarray(
            restored.get("snapshot_state", {}).get("mujoco_integration_state", []),
            dtype=np.float64,
        )
        exact = restored_state.shape == source_state.shape and np.array_equal(
            restored_state, source_state
        )
        report["restore"] = {
            "status": restored.get("status"),
            "model_geometry_sha256": restored.get("model_geometry_sha256"),
            "integration_state_shape": list(restored_state.shape),
            "integration_state_sha256": _sha256(
                np.ascontiguousarray(restored_state).tobytes()
            ),
            "integration_state_exact_match": bool(exact),
        }
        if restored.get("status") != "restored" or not exact:
            report["status"] = "snapshot_restore_not_exact"
            return report

        native = bridge.request({"op": "native_task_status"})
        semantics = bridge.request({"op": "task_semantic_geometry_bindings"})
        pot = semantics.get("bindings", {}).get("moka_pot_1", {})
        carried_binding = {
            "object_id": "moka_pot_1",
            "body_id": int(pot["root_body_id"]),
            "body_name": str(pot["root_body_name"]),
            "geom_ids": [int(value) for value in pot["root_geom_ids"]],
            "geom_names": [str(value) for value in pot["root_geom_names"]],
        }
        grasp = bridge.request({
            "op": "grasp_state",
            "carried_binding": carried_binding,
            "expected_instruction": INSTRUCTION,
        })
        final_export = bridge.request({"op": "export_runtime_snapshot"})
        final_state = np.asarray(
            final_export.get("snapshot_state", {}).get("mujoco_integration_state", []),
            dtype=np.float64,
        )
        report["validation"] = {
            "native_task_status": native,
            "semantic_binding_status": semantics.get("status"),
            "pot1_binding": carried_binding,
            "grasp_state": grasp,
            "grasp_verified_bilateral": bool(
                grasp.get("grasp_confirmed") is True
                and grasp.get("finger_contacts", {}).get("left") is True
                and grasp.get("finger_contacts", {}).get("right") is True
                and grasp.get("gripper_closed") is True
            ),
            "grasp_read_physical_state_unchanged": grasp.get("physical_state_unchanged"),
            "post_readonly_integration_state_exact_match": bool(
                final_state.shape == source_state.shape
                and np.array_equal(final_state, source_state)
            ),
            "post_restore_model_geometry_sha256": final_export.get(
                "model_geometry_sha256"
            ),
            "actions_issued": 0,
            "mujoco_physics_substeps": 0,
        }
        ok = bool(
            native.get("status") == "complete"
            and native.get("timestep") == timestep
            and native.get("task_success") is False
            and semantics.get("status") == "complete"
            and report["validation"]["grasp_verified_bilateral"]
            and grasp.get("physical_state_unchanged") is True
            and report["validation"]["post_readonly_integration_state_exact_match"]
            and final_export.get("model_geometry_sha256") == EXPECTED_MODEL_SHA256
        )
        report["status"] = "verified_exact_restore_and_bilateral_grasp" if ok else "restore_or_grasp_validation_failed"
        return report
    finally:
        bridge.close()


def main() -> int:
    args = _args()
    try:
        report = run(args)
    except Exception as exc:
        report = {
            "schema": "gs_opd_t215_bridge_recovery.v1",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "actions_issued": 0,
            "mujoco_physics_substeps": 0,
        }
        if args.output_dir is not None and args.output_dir.is_dir():
            (args.output_dir / "recovery_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(report["snapshot"]["archived_copy"]).parent
    else:
        output_dir = output_dir.resolve()
    (output_dir / "recovery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "output_dir": str(output_dir),
        "source_model_sha256": report.get("snapshot", {}).get("model_geometry_sha256"),
        "pre_reset_model_sha256": report.get("fingerprints", {}).get(
            "after_bridge_start_seed_reset_set_init_state", {}
        ).get("model_geometry_sha256"),
        "post_reset_model_sha256": report.get("fingerprints", {}).get(
            "after_historical_executor_additional_reset", {}
        ).get("model_geometry_sha256"),
        "component_differences": [row["field"] for row in report.get("component_differences", [])],
        "restore": report.get("restore"),
        "validation": {
            "timestep": report.get("validation", {}).get("native_task_status", {}).get("timestep"),
            "pot1_bilateral_grasp": report.get("validation", {}).get("grasp_verified_bilateral"),
            "state_unchanged": report.get("validation", {}).get("grasp_read_physical_state_unchanged"),
            "actions": report.get("actions_issued"),
            "substeps": report.get("mujoco_physics_substeps"),
        },
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "verified_exact_restore_and_bilateral_grasp" else 1


if __name__ == "__main__":
    raise SystemExit(main())
