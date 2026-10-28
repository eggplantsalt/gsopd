"""Normalize a verified L0T2 physical grasp trace for the strict scratch screen.

The output is a pose/joint prior only. It is not an episode, safety certificate,
or source of replayed actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


TASK = "put both moka pots on the stove"
OBJECT_ID = "moka_pot_2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def build_reference_adapter(
    *,
    branch_report_path: Path,
    dataset_manifest_path: Path,
    snapshot_manifest_path: Path,
    target_snapshot_path: Path,
    bddl_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate source identities and write the reference schema consumed by strict screening."""
    branch_report_path = branch_report_path.resolve()
    dataset_manifest_path = dataset_manifest_path.resolve()
    snapshot_manifest_path = snapshot_manifest_path.resolve()
    target_snapshot_path = target_snapshot_path.resolve()
    bddl_path = bddl_path.resolve()
    output_dir = output_dir.resolve()

    branch = _read_json(branch_report_path)
    dataset = _read_json(dataset_manifest_path)
    snapshot_manifest = _read_json(snapshot_manifest_path)
    scene = snapshot_manifest.get("scene_metadata", {})
    if (
        dataset.get("task") != TASK
        or dataset.get("native_task_success") is not True
        or dataset.get("operational_safety_pass") is not True
        or dataset.get("strict_10mm_pass") is not False
    ):
        raise ValueError("round0_source_acceptance_or_task_identity_mismatch")
    expected_scene = {
        "benchmark": "obstacle_avoidance",
        "level": 0,
        "task_id": 2,
        "task_name": "put_both_moka_pots_on_the_stove",
        "instruction": TASK,
        "init_state_index": 0,
        "environment_seed": 0,
        "standard_evaluation_horizon": 1000,
    }
    scene_mismatches = {
        key: {"expected": value, "actual": scene.get(key)}
        for key, value in expected_scene.items()
        if scene.get(key) != value
    }
    if scene_mismatches:
        raise ValueError(f"active_snapshot_scene_identity_mismatch:{scene_mismatches}")
    if not target_snapshot_path.is_file() or _sha256(target_snapshot_path) != snapshot_manifest.get("snapshots", {}).get("initial", {}).get("payload_sha256"):
        raise ValueError("clean_target_snapshot_hash_mismatch")
    if (
        not bddl_path.is_file()
        or str(bddl_path) != scene.get("bddl_path")
        or _sha256(bddl_path) != scene.get("bddl_sha256")
    ):
        raise ValueError("active_bddl_identity_mismatch")

    model_sha = str(scene.get("model_geometry_sha256", ""))
    if (
        branch.get("expected_source_model_geometry_sha256") != model_sha
        or branch.get("native_reset_model_geometry_sha256") != model_sha
        or branch.get("native_task_success") is not False
        or branch.get("physical_actions") != 80
        or branch.get("terminal_timestep") != 329
        or branch.get("source_snapshot_sha256") is None
    ):
        raise ValueError("source_grasp_branch_model_or_stage_mismatch")
    source_safety = branch.get("acceptance_verdicts", {})
    if not (
        source_safety.get("native_safety") is True
        and source_safety.get("operational_safety") is True
        and source_safety.get("strict_10mm") is False
    ):
        raise ValueError("source_grasp_branch_safety_labels_mismatch")
    grasp = branch.get("live_grasp_state", {})
    if not (
        grasp.get("status") == "pass"
        and grasp.get("object_id") == OBJECT_ID
        and grasp.get("grasp_confirmed") is True
        and grasp.get("finger_contacts", {}).get("left") is True
        and grasp.get("finger_contacts", {}).get("right") is True
    ):
        raise ValueError("source_grasp_branch_lacks_verified_bilateral_contact")
    lift_m = float(branch.get("measured_pot2_vertical_rise_m", 0.0))
    if lift_m < 0.030:
        raise ValueError("source_grasp_branch_lift_below_30mm")

    segment = next((
        row for row in dataset.get("segments", [])
        if row.get("global_timestep_range_exclusive") == [249, 329]
    ), None)
    if segment is None:
        raise ValueError("accepted_round0_episode_has_no_249_to_329_lift_segment")
    trace_path = Path(str(branch.get("data_paths", {}).get("trace_npz", ""))).resolve()
    source_snapshot = Path(str(branch.get("source_snapshot", ""))).resolve()
    segment_start = segment.get("start_snapshot", {})
    if (
        not trace_path.is_file()
        or str(trace_path) != str(Path(str(segment.get("trace", ""))).resolve())
        or _sha256(trace_path) != segment.get("trace_sha256")
        or branch.get("source_snapshot_sha256") != segment_start.get("sha256")
        or not source_snapshot.is_file()
        or _sha256(source_snapshot) != segment_start.get("sha256")
    ):
        raise ValueError("accepted_lift_segment_trace_or_snapshot_hash_mismatch")

    with np.load(trace_path, allow_pickle=False) as source:
        required = {
            "arm_joint_qpos", "gripper_qpos_m",
            "pot2_root_position_world_m", "pot2_root_quaternion_wxyz",
        }
        if not required.issubset(source.files):
            raise ValueError("accepted_lift_trace_pose_fields_missing")
        arm = np.asarray(source["arm_joint_qpos"], dtype=np.float64)
        gripper = np.asarray(source["gripper_qpos_m"], dtype=np.float64)
        position = np.asarray(source["pot2_root_position_world_m"], dtype=np.float64)
        quaternion = np.asarray(source["pot2_root_quaternion_wxyz"], dtype=np.float64)
        if (
            arm.ndim != 2 or arm.shape[1] != 7
            or gripper.ndim != 2 or gripper.shape[1] != 2
            or position.ndim != 2 or position.shape[1] != 3
            or quaternion.ndim != 2 or quaternion.shape[1] != 4
            or len({len(arm), len(gripper), len(position), len(quaternion)}) != 1
            or len(arm) != 80
            or not all(np.isfinite(value).all() for value in (arm, gripper, position, quaternion))
            or np.any(np.linalg.norm(quaternion, axis=1) < 1e-8)
        ):
            raise ValueError("accepted_lift_trace_pose_arrays_malformed")

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_trace = output_dir / "l0t2_pot2_physical_grasp_pose_prior.npz"
    # Alias names only; values and row order are copied directly from the accepted source trace.
    np.savez_compressed(
        normalized_trace,
        arm_joint_qpos_rad=arm,
        gripper_qpos=gripper,
        target_root_position_world_m=position,
        target_root_quaternion_wxyz=quaternion,
    )
    trace_sha = _sha256(normalized_trace)
    artifact_hashes = {
        normalized_trace.name: {"sha256": trace_sha, "bytes": normalized_trace.stat().st_size}
    }
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    adapter = {
        "schema_version": "gs_opd_strict_grasp_reference_adapter.v1",
        "status": "verified_physical_grasp_pose_prior",
        "archive": str(output_dir),
        "scope": {
            "benchmark": "obstacle_avoidance",
            "level": 0,
            "task_id": 2,
            "init_state": 0,
            "environment_seed": 0,
            "model_geometry_sha256": model_sha,
            "bddl_sha256": scene["bddl_sha256"],
        },
        "success": {
            "target_object_id": OBJECT_ID,
            "stable_bilateral_grasp": {"stable_grasp_observed": True},
            "target_lift_delta_m": lift_m,
        },
        "artifacts": {"actions_and_robot_states_npz": str(normalized_trace)},
        "source_provenance": {
            "branch_report": str(branch_report_path),
            "branch_report_sha256": _sha256(branch_report_path),
            "dataset_manifest": str(dataset_manifest_path),
            "dataset_manifest_sha256": _sha256(dataset_manifest_path),
            "accepted_source_trace": str(trace_path),
            "accepted_source_trace_sha256": segment["trace_sha256"],
            "source_snapshot": str(source_snapshot),
            "source_snapshot_sha256": segment_start["sha256"],
            "clean_target_snapshot": str(target_snapshot_path),
            "clean_target_snapshot_sha256": _sha256(target_snapshot_path),
            "operational_safety_pass_for_source_episode": True,
            "strict_10mm_pass_for_source_episode": False,
            "safety_qualification_inherited": False,
            "source_actions_replayed": False,
            "normalized_trace_values_copied_without_transform": True,
            "pose_prior_only": True,
            "training_label_valid": False,
        },
    }
    adapter_path = output_dir / "reference_adapter.json"
    adapter_path.write_text(json.dumps(adapter, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    adapter["adapter_report"] = str(adapter_path)
    adapter["adapter_report_sha256"] = _sha256(adapter_path)
    return adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-report", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--target-snapshot", type=Path, required=True)
    parser.add_argument("--bddl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_reference_adapter(
        branch_report_path=args.branch_report,
        dataset_manifest_path=args.dataset_manifest,
        snapshot_manifest_path=args.snapshot_manifest,
        target_snapshot_path=args.target_snapshot,
        bddl_path=args.bddl,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "status": result["status"],
        "adapter_report": result["adapter_report"],
        "adapter_report_sha256": result["adapter_report_sha256"],
        "source_trace_sha256": result["source_provenance"]["accepted_source_trace_sha256"],
        "strict_10mm_pass_for_source_episode": result["source_provenance"]["strict_10mm_pass_for_source_episode"],
        "training_label_valid": result["source_provenance"]["training_label_valid"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
