# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Replay the archived Episode 9 t=10..249 actions under current safety rules."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from research.gs_opd.repair.automatic_recovery import RecoveryArtifactWriter
from research.gs_opd.repair.automatic_recovery_runtime import (
    AutomaticRecoveryBridgeBackend,
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
)
from research.gs_opd.repair.run_episode9_t249_lift_branch import (
    EXPECTED_MODEL_SHA256,
    JOINT_LIMIT_TOLERANCE_POLICY,
    _grasp_state,
    _target_binding,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROLLOUT_ROOT = Path("/opt/data/private/gs_opd/rollouts")
EPISODE9_DIR = ROLLOUT_ROOT / "task_success_first_safety_reconstruction_l0t2_20261023_attempt1/episode9"
T10_AUDIT_DIR = ROLLOUT_ROOT / "l0t2_round1_safety_recovery_20261023_attempt1/evidence"
RAW_STUDENT_DIR = ROLLOUT_ROOT / "obstacle_avoidance_L0T2_base_pi05_rollout_400_20261022_attempt2"
T10_VERIFICATION = ROLLOUT_ROOT / "l0t2_round1_jointspace_rrt_connect_20261023_attempt1/source/t10_replay_snapshot_verification.json"
T10_SNAPSHOT = T10_AUDIT_DIR / "raw_pi05_t10_runtime_snapshot.pkl"
EPISODE9_T249_SNAPSHOT = EPISODE9_DIR / "snapshots/main_timestep_000249.pkl"
T249_CHAIN_POLICY_REPORT = ROLLOUT_ROOT / "l0t2_round0_t249_operational_lift_20261024_branch1/branch_report.json"
DEFAULT_CONFIG = REPO_ROOT / "research/gs_opd/configs/l0t2_task_success_first_safety_reconstruction.json"
EXPECTED_BDDL_SHA256 = "901feec9b1ade569500c06f738f07900cb0f1db0de8b91c467d860315e7d675c"
TARGET_OBJECT = "moka_pot_2"
T0_TO_T10_TRACE = T10_AUDIT_DIR / "raw_pi05_t0_to_t10_replay.npz"
T10_TO_T249_TRACE = EPISODE9_DIR / "observation_action_trace.npz"
START_STATE_JSON = EPISODE9_DIR / "start_state.json"
EXPECTED_ACTION_COUNT = 239
EXPECTED_SUBSTEPS_PER_ACTION = 25
CHECKPOINT_STRIDE = 25


class _PythonObjectState:
    """Placeholder for helper objects serialized by the standalone Bridge."""


class _SnapshotUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "_PythonObjectState":
            return _PythonObjectState
        return super().find_class(module, name)


def _jsonable(value: Any, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item, _seen) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, _seen) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item, _seen) for item in value)
    if hasattr(value, "__dict__"):
        identifier = id(value)
        if identifier in _seen:
            return {"__cycle__": f"{type(value).__module__}.{type(value).__name__}"}
        _seen.add(identifier)
        result = {
            "__class__": f"{type(value).__module__}.{type(value).__name__}",
            "__dict__": _jsonable(vars(value), _seen),
        }
        _seen.remove(identifier)
        return result
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _integration_hash(state: dict[str, Any]) -> str:
    values = np.asarray(state["mujoco_integration_state"], dtype=np.float64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _max_abs(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return math.inf
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _camera_frame(data: dict[str, Any], key: str) -> np.ndarray:
    image = data.get("images", {}).get(key)
    if not isinstance(image, dict):
        raise ValueError(f"required_camera_observation_missing:{key}")
    return np.frombuffer(image["data"], dtype=np.dtype(image["dtype"])).reshape(image["shape"]).copy()


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"video_writer_failed:{path.name}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _load_source_audit() -> dict[str, Any]:
    student_trace_path = RAW_STUDENT_DIR / "rollout_trace.npz"
    with np.load(student_trace_path, allow_pickle=False) as student_trace, np.load(
        T0_TO_T10_TRACE, allow_pickle=False
    ) as prefix_trace, np.load(T10_TO_T249_TRACE, allow_pickle=False) as episode_trace:
        student_actions = np.asarray(student_trace["executed_actions"][:10], dtype=np.float32)
        t0_actions = np.asarray(prefix_trace["actions"], dtype=np.float32)
        episode_actions = np.asarray(episode_trace["executed_actions"], dtype=np.float32)
        timesteps = np.asarray(episode_trace["step_timesteps"], dtype=np.int64)
        phase_labels = np.asarray(episode_trace["step_task_phase"]).astype(str)
        metadata = json.loads(str(episode_trace["metadata_json"]))
        if t0_actions.shape != (10, 7) or student_actions.shape != (10, 7):
            raise ValueError("original_pi05_t0_to_t10_action_rows_are_not_exactly_10x7")
        if episode_actions.shape != (EXPECTED_ACTION_COUNT, 7):
            raise ValueError("episode9_t10_to_t249_action_rows_are_not_exactly_239x7")
        if not np.array_equal(timesteps, np.arange(11, 250, dtype=np.int64)):
            raise ValueError("episode9_action_timestep_range_is_not_11_through_249")
        if not np.array_equal(t0_actions, student_actions):
            raise ValueError("t0_to_t10_replayed_actions_differ_from_raw_student_rollout")
        if phase_labels.shape != (EXPECTED_ACTION_COUNT,):
            raise ValueError("episode9_task_phase_rows_are_incomplete")
        return {
            "student_trace_path": str(student_trace_path),
            "student_trace_sha256": hashlib.sha256(student_trace_path.read_bytes()).hexdigest(),
            "student_trace_metadata": json.loads(str(student_trace["metadata_json"])),
            "t0_to_t10_action_archive": str(T0_TO_T10_TRACE),
            "t0_to_t10_action_count": int(len(t0_actions)),
            "t0_to_t10_action_hash": hashlib.sha256(t0_actions.tobytes()).hexdigest(),
            "t0_to_t10_actions_exactly_match_raw_pi05": True,
            "t10_snapshot_path": str(T10_SNAPSHOT),
            "t10_snapshot_sha256": hashlib.sha256(T10_SNAPSHOT.read_bytes()).hexdigest(),
            "t10_verification_path": str(T10_VERIFICATION),
            "t10_verification": json.loads(T10_VERIFICATION.read_text()),
            "episode9_trace_path": str(T10_TO_T249_TRACE),
            "episode9_trace_sha256": hashlib.sha256(T10_TO_T249_TRACE.read_bytes()).hexdigest(),
            "episode9_metadata": metadata,
            "episode9_action_count": int(len(episode_actions)),
            "episode9_timestep_range": [int(timesteps[0]), int(timesteps[-1])],
            "episode9_action_sha256": hashlib.sha256(episode_actions.tobytes()).hexdigest(),
            "episode9_phase_counts": {
                phase: int(np.count_nonzero(phase_labels == phase))
                for phase in sorted(set(phase_labels.tolist()))
            },
            "episode9_phase_first_last_timestep": {
                phase: [int(timesteps[np.flatnonzero(phase_labels == phase)[0]]),
                        int(timesteps[np.flatnonzero(phase == phase_labels)[-1]])]
                for phase in sorted(set(phase_labels.tolist()))
            },
            "episode9_actions": episode_actions,
            "episode9_timesteps": timesteps,
            "episode9_phase_labels": phase_labels,
        }


def _make_current_policy(backend, frozen_policy: dict[str, Any]) -> dict[str, Any]:
    """Reuse the archived current policy and rebind it to this compiled scene."""
    policy = copy.deepcopy(frozen_policy)
    policy["carried_object"] = None
    policy["joint_limit_tolerance_policy"] = copy.deepcopy(JOINT_LIMIT_TOLERANCE_POLICY)
    policy["source"] = "episode9_t10_to_t249_replay_current_frozen_object_level_policy"
    if policy.get("allowed_target_objects") is None or policy.get("forbidden_objects") is None:
        raise ValueError("frozen_current_operational_policy_is_incomplete")
    binding = backend.bindings[TARGET_OBJECT]
    finger_ids = set(int(value) for value in backend.context_static["gripper_geometry_ids"])
    if len(finger_ids) != 4:
        raise ValueError("current_model_does_not_bind_exactly_four_finger_collision_geoms")
    target_ids = set(int(value) for value in binding["geom_ids"])
    declared_pairs = {
        (int(item["robot_geom_id"]), int(item["scene_geom_id"]))
        for item in policy.get("allowed_contact_pairs", [])
    }
    if not declared_pairs or any(robot not in finger_ids or scene not in target_ids for robot, scene in declared_pairs):
        raise ValueError("frozen_policy_contact_allowlist_is_not_finger_only_pot2")
    if any(float(item.get("minimum_clearance_m", 1.0)) != -0.002 for item in policy["allowed_contact_pairs"]):
        raise ValueError("frozen_policy_finger_contact_penetration_is_not_minus_2mm")
    margins_forbidden = {
        str(item["object_id"]): float(item["minimum_clearance_m"])
        for item in policy["forbidden_objects"]
    }
    if margins_forbidden.get("bottle_of_ginger_beer_1") != 0.010 or margins_forbidden.get("white_place_box_1") != 0.010:
        raise ValueError("frozen_policy_ginger_beer_or_place_box_clearance_is_not_10mm")
    margins_targets = {
        str(item["object_id"]): float(item["minimum_clearance_m"])
        for item in policy["allowed_target_objects"]
    }
    if margins_targets.get(TARGET_OBJECT) != 0.002:
        raise ValueError("frozen_policy_pot2_nonfinger_margin_is_not_2mm")
    # Geometry identities must still be those resolved from the active BDDL.
    frozen_pot2 = next(item for item in policy["allowed_target_objects"] if item["object_id"] == TARGET_OBJECT)
    if set(int(value) for value in frozen_pot2["geom_ids"]) != target_ids:
        raise ValueError("frozen_policy_pot2_geometry_binding_differs_from_active_BDDL")
    for group in policy.get("forbidden_objects", []) + policy.get("allowed_target_objects", []):
        object_id = str(group["object_id"])
        active_binding = backend.bindings.get(object_id)
        if not isinstance(active_binding, dict) or set(int(value) for value in group.get("geom_ids", [])) != set(
            int(value) for value in active_binding.get("geom_ids", [])
        ):
            raise ValueError(f"frozen_policy_geometry_binding_differs_from_active_BDDL:{object_id}")
    robot_geom_ids = {
        int(item["geom_id"])
        for item in backend.bridge.request({"op": "geometry_inventory"}).get("geometries", [])
        if item.get("category") == "robot" and item.get("collision_enabled") is True
    }
    if not robot_geom_ids:
        raise ValueError("active_full_robot_collision_geometry_inventory_is_missing")
    # The frozen policy already enumerates all full robot and scene geometry.
    # Reject the run if it does not cover the current compiled robot inventory.
    frozen_robot_geom_ids = set(int(value) for value in policy.get("robot_geom_ids", []))
    if frozen_robot_geom_ids and frozen_robot_geom_ids != robot_geom_ids:
        raise ValueError("frozen_policy_robot_collision_geometry_differs_from_active_model")
    backend.safety_policy = copy.deepcopy(policy)
    return policy


def _active_policy_for_step(
    backend,
    base_policy: dict[str, Any],
    held: str | None,
    phase: str,
    carried_support_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    policy = copy.deepcopy(base_policy)
    if held is None:
        policy["carried_object"] = None
        return policy
    if held != TARGET_OBJECT:
        raise ValueError(f"unexpected_object_is_reported_as_held:{held}")
    backend.safety_policy = copy.deepcopy(base_policy)
    backend._refresh_carried_geometry_policy(held)
    carried = backend.safety_policy.get("carried_object")
    if not isinstance(carried, dict):
        raise ValueError("verified_grasp_has_no_carried_geometry_binding")
    # The archived Task Phase Controller explicitly permits the held pot to
    # remain on its support while stable_grasp/lift is being established.
    # This is the same named table support pair already in the frozen policy.
    if phase in {"stable_grasp", "lift"}:
        carried["allowed_support_contact_pairs"] = copy.deepcopy(carried_support_pairs)
    else:
        carried["allowed_support_contact_pairs"] = []
    carried["allowed_contact_pairs"] = []
    policy["carried_object"] = carried
    return policy


def _robot_contact_audit(
    substep: dict[str, Any],
    robot_geom_ids: set[int],
    finger_geom_ids: set[int],
    target_body_id: int,
    target_geom_ids: set[int],
) -> dict[str, Any]:
    robot_contacts = []
    illegal = []
    max_penetration_m = 0.0
    for contact in substep.get("contact_identities", []):
        geom1, geom2 = contact.get("geom1", {}), contact.get("geom2", {})
        body1, body2 = contact.get("body1", {}), contact.get("body2", {})
        id1, id2 = int(geom1.get("id", -1)), int(geom2.get("id", -1))
        b1, b2 = int(body1.get("id", -1)), int(body2.get("id", -1))
        robot1, robot2 = id1 in robot_geom_ids, id2 in robot_geom_ids
        if not (robot1 or robot2):
            continue
        robot_geom, scene_geom = (geom1, geom2) if robot1 else (geom2, geom1)
        scene_body = body2 if robot1 else body1
        distance = contact.get("signed_contact_distance_m")
        distance = float(distance) if distance is not None else None
        valid = (
            not (robot1 and robot2)
            and int(robot_geom.get("id", -1)) in finger_geom_ids
            and int(scene_body.get("id", -1)) == target_body_id
            and int(scene_geom.get("id", -1)) in target_geom_ids
            and distance is not None
            and distance >= -0.002
        )
        row = {
            "geom1": geom1,
            "geom2": geom2,
            "body1": body1,
            "body2": body2,
            "signed_contact_distance_m": distance,
            "finger_only_pot2_contact_allowed": bool(valid),
        }
        robot_contacts.append(row)
        if distance is not None:
            max_penetration_m = max(max_penetration_m, max(0.0, -distance))
        if not valid:
            illegal.append(row)
    return {
        "robot_contact_count": len(robot_contacts),
        "robot_contacts": robot_contacts,
        "illegal_robot_contact_count": len(illegal),
        "illegal_robot_contacts": illegal,
        "maximum_robot_contact_penetration_m": max_penetration_m,
    }


def _native_substep_pass(substep: dict[str, Any], robot_contact: dict[str, Any]) -> bool:
    def is_actual_contact_violation(row: dict[str, Any]) -> bool:
        if any(row.get(key) is True for key in (
            "geometry_clearance_requirement_violated",
            "allowed_pair_penetration_exceeded",
            "carried_object_clearance_requirement_violated",
            "carried_object_allowed_pair_penetration_exceeded",
        )):
            return False
        return True

    return bool(
        substep.get("status") != "incomplete"
        and not any(is_actual_contact_violation(row) for row in substep.get("forbidden_contacts", []))
        and not substep.get("self_contacts")
        and not substep.get("unresolved_contacts")
        and not substep.get("forbidden_object_contacts")
        and not substep.get("unallowed_scene_contacts")
        and robot_contact.get("illegal_robot_contact_count") == 0
    )


def _operational_substep_pass(substep: dict[str, Any], required_margins: dict[str, float]) -> bool:
    if substep.get("status") == "incomplete" or substep.get("joint_limit_violations"):
        return False
    for object_id, margin in required_margins.items():
        row = substep.get("clearances_by_object", {}).get(object_id)
        distance = row.get("minimum_distance_m") if isinstance(row, dict) else None
        if distance is None or float(distance) < margin:
            return False
    for object_id, row in substep.get("carried_clearances_by_object", {}).items():
        distance = row.get("minimum_distance_m") if isinstance(row, dict) else None
        if distance is None:
            return False
        if object_id in {"bottle_of_ginger_beer_1", "white_place_box_1"} and float(distance) < 0.010:
            return False
        if object_id not in {"bottle_of_ginger_beer_1", "white_place_box_1"} and float(distance) < 0.0:
            return False
    if any(float(value) < -0.002 for value in substep.get("allowed_contact_clearances_m", {}).values()):
        return False
    return True


def _snapshot_runtime_fields(payload: bytes) -> dict[str, Any]:
    envelope = _SnapshotUnpickler(io.BytesIO(payload)).load()
    snap = envelope.get("snapshot", {})
    robots = snap.get("robots", [])
    robot = robots[0] if robots else {}
    return {
        "model_geometry_sha256": envelope.get("model_geometry_sha256"),
        "integration_state": snap.get("mujoco_state"),
        "integration_state_sha256": hashlib.sha256(
            np.asarray(snap.get("mujoco_state"), dtype=np.float64).tobytes()
        ).hexdigest() if snap.get("mujoco_state") is not None else None,
        "task_bookkeeping": snap.get("task_bookkeeping"),
        "controller": robot.get("controller"),
        "robot_buffers": robot.get("buffers"),
        "gripper_current_action": robot.get("gripper_current_action"),
        "has_gripper_current_action": robot.get("has_gripper_current_action"),
    }


def _runtime_state_comparison(reference: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    for field in ("controller", "robot_buffers", "gripper_current_action", "task_bookkeeping"):
        ref = reference.get(field)
        got = replay.get(field)
        comparisons[field] = {
            "exact": _jsonable(ref) == _jsonable(got),
            "reference_sha256": _canonical_sha256(ref),
            "replay_sha256": _canonical_sha256(got),
        }
    return {
        "status": "pass" if all(item["exact"] for item in comparisons.values()) else "mismatch",
        "fields": comparisons,
        "task_phase_controller_state": "not_part_of_MuJoCo_runtime_snapshot;source_phase_is_audited_from_archived_events_and_replay_rows",
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38783)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROLLOUT_ROOT / f"l0t2_episode9_t10_to_t249_current_safety_replay_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir()
    source = _load_source_audit()
    with np.load(T10_TO_T249_TRACE, allow_pickle=False) as trace:
        actions = np.asarray(trace["executed_actions"], dtype=np.float32)
        archived_timesteps = np.asarray(trace["step_timesteps"], dtype=np.int64)
        phase_labels = np.asarray(trace["step_task_phase"]).astype(str)
    episode9_start = json.loads(START_STATE_JSON.read_text())
    t10_verification = source["t10_verification"]
    t249_reference_report = json.loads(T249_CHAIN_POLICY_REPORT.read_text())
    frozen_policy = t249_reference_report.get("policy")
    if not isinstance(frozen_policy, dict):
        raise ValueError("t249_success_chain_source_has_no_frozen_operational_policy")
    source_snapshot_bytes = T10_SNAPSHOT.read_bytes()
    t249_snapshot_bytes = EPISODE9_T249_SNAPSHOT.read_bytes()
    config_data, recovery = load_runtime_config(args.config)
    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend = None
    rows: list[dict[str, Any]] = []
    substep_rows: list[dict[str, Any]] = []
    agent_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    pre_agent_frames: list[np.ndarray] = []
    pre_wrist_frames: list[np.ndarray] = []
    start_snapshot_bytes: bytes | None = None
    end_snapshot_bytes: bytes | None = None
    report: dict[str, Any] = {
        "schema": "gs_opd_episode9_t10_to_t249_current_safety_replay.v1",
        "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "physical_replay_scope": "strictly_t10_to_t249_archived_episode9_actions_only",
        "later_successful_suffix_replayed": False,
        "trajectory_search_or_controller_optimization": False,
        "native_resets_before_restore": 0,
        "physical_action_count": 0,
        "physical_substep_count": 0,
        "expected_actions": EXPECTED_ACTION_COUNT,
        "expected_substeps_per_action": EXPECTED_SUBSTEPS_PER_ACTION,
        "source_audit": {
            key: value for key, value in source.items()
            if key not in {"episode9_actions", "episode9_timesteps", "episode9_phase_labels"}
        },
        "t249_success_chain_source": {
            "source_snapshot": t249_reference_report.get("source_snapshot"),
            "snapshot_sha256": hashlib.sha256(t249_snapshot_bytes).hexdigest(),
            "t249_lift_branch_report": str(T249_CHAIN_POLICY_REPORT),
            "policy_sha256": _canonical_sha256(frozen_policy),
        },
        "safety_policy": {
            "source": "frozen_policy_from_verified_t249_operational_lift_branch",
            "finger_geom_ids": sorted({int(item["robot_geom_id"]) for item in frozen_policy.get("allowed_contact_pairs", [])}),
            "allowed_finger_to_pot2_max_penetration_m": 0.002,
            "forbidden_clearances_m": {
                str(item["object_id"]): float(item["minimum_clearance_m"])
                for item in frozen_policy.get("forbidden_objects", [])
            },
            "target_object_clearances_m": {
                str(item["object_id"]): float(item["minimum_clearance_m"])
                for item in frozen_policy.get("allowed_target_objects", [])
            },
            "joint_limit_tolerance_policy": copy.deepcopy(JOINT_LIMIT_TOLERANCE_POLICY),
            "self_collision_check": True,
            "native_bddl_forbidden_object_pairs": True,
        },
        "outcome": None,
        "failure": None,
    }
    try:
        bridge.start(
            benchmark=config_data["benchmark"],
            level=int(config_data["level"]),
            task_id=int(config_data["task_id"]),
            init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=1800,
        )
        startup = bridge.request({"op": "observe", "include_snapshot_state": True})
        startup_fingerprint = bridge.request({"op": "compiled_model_fingerprint_details"})
        native_reset = bridge.request({"op": "reset"})
        report["native_resets_before_restore"] = 1
        reset_hash = native_reset.get("scene_metadata", {}).get("model_geometry_sha256")
        if reset_hash != EXPECTED_MODEL_SHA256 or int(native_reset.get("timestep", -1)) != 0:
            raise RuntimeError("native_reset_geometry_or_timestep_mismatch")

        # Read the exact t=249 chain source on this isolated Bridge, capture its
        # runtime controller/gripper reference, then return to the verified t=10.
        restore249 = bridge.request({"op": "restore_runtime_snapshot", "payload": t249_snapshot_bytes})
        if restore249.get("status") != "restored":
            raise RuntimeError(f"t249_chain_source_snapshot_restore_failed:{restore249.get('error')}")
        source249_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
        source249_contacts = bridge.request({"op": "inspect_contacts"})
        source249_native = bridge.request({"op": "native_task_status"})
        source249_repair = bridge.request({"op": "repair_context"})
        source249_runtime = _snapshot_runtime_fields(t249_snapshot_bytes)
        if int(source249_observation.get("timestep", -1)) != 249:
            raise RuntimeError("t249_chain_source_snapshot_has_wrong_timestep")
        if source249_observation.get("scene_metadata", {}).get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError("t249_chain_source_snapshot_model_fingerprint_mismatch")

        backend = AutomaticRecoveryBridgeBackend(
            bridge,
            config_data,
            recovery,
            output_dir,
            RecoveryArtifactWriter(output_dir),
            mode="runtime-snapshot",
            student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=T10_SNAPSHOT,
            t10_verification_path=T10_VERIFICATION,
        )
        backend.latest_data = source249_observation
        backend._load_context_inputs(source249_observation)
        source249_grasp = _grasp_state(bridge, backend)

        restore10 = bridge.request({"op": "restore_runtime_snapshot", "payload": source_snapshot_bytes})
        if restore10.get("status") != "restored":
            raise RuntimeError(f"verified_t10_snapshot_restore_failed:{restore10.get('error')}")
        current = bridge.request({"op": "observe", "include_snapshot_state": True})
        start_state = current.get("snapshot_state")
        expected_start_state = episode9_start.get("snapshot_state", {})
        start_integration_error = _max_abs(
            start_state.get("mujoco_integration_state"),
            expected_start_state.get("mujoco_integration_state"),
        )
        report["restore_validation"] = {
            "t10_restore_status": restore10.get("status"),
            "timestep": int(current.get("timestep", -1)),
            "model_geometry_sha256": current.get("scene_metadata", {}).get("model_geometry_sha256"),
            "t10_snapshot_sha256": hashlib.sha256(source_snapshot_bytes).hexdigest(),
            "integration_state_sha256": _integration_hash(start_state),
            "episode9_archived_start_state_match_max_abs": start_integration_error,
            "episode9_start_state_match_exact": start_integration_error == 0.0,
            "previous_t10_verification_status": t10_verification.get("status"),
            "previous_t10_verification_full_physical_state_match": t10_verification.get("first_10_replay_state_max_abs_error") and all(
                float(value) == 0.0 for value in t10_verification["first_10_replay_state_max_abs_error"].values()
            ),
            "previous_t10_substep_contact_coverage": t10_verification.get("substep_contact_monitoring"),
            "prior_t10_fields_not_used_as_current_policy_safety_proof": True,
        }
        if int(current.get("timestep", -1)) != 10 or current.get("scene_metadata", {}).get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError("verified_t10_restore_timestep_or_model_fingerprint_mismatch")
        if start_integration_error != 0.0:
            raise RuntimeError("t10_snapshot_does_not_match_episode9_archived_t10_start_state")

        backend.latest_data = current
        policy = _make_current_policy(backend, frozen_policy)
        geometry_inventory = bridge.request({"op": "geometry_inventory"})
        robot_geom_ids = {
            int(item["geom_id"])
            for item in geometry_inventory.get("geometries", [])
            if item.get("category") == "robot" and item.get("collision_enabled") is True
        }
        finger_geom_ids = set(int(item) for item in backend.context_static["gripper_geometry_ids"])
        gripper_inventory = bridge.request({"op": "gripper_geometry_inventory"})
        actual_finger_ids = {
            int(item["geom_id"])
            for gripper in gripper_inventory.get("grippers", [])
            for item in gripper.get("geometries", [])
            if str(item.get("geom_name", "")).endswith((
                "finger1_collision", "finger1_pad_collision",
                "finger2_collision", "finger2_pad_collision",
            ))
        }
        if gripper_inventory.get("status") != "complete" or actual_finger_ids != finger_geom_ids:
            raise RuntimeError("current_four_finger_geom_inventory_does_not_match_frozen_allowlist")
        target = _target_binding(backend)
        carried_support_pairs = copy.deepcopy(
            frozen_policy.get("carried_object", {}).get("allowed_support_contact_pairs", [])
        )
        # Verify t=10 against the same complete policy before the first replayed action.
        policy["carried_object"] = None
        initial_geometry = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": policy,
        }).get("safety_geometry")
        initial_contacts = bridge.request({"op": "inspect_contacts"})
        initial_grasp = _grasp_state(bridge, backend)
        initial_native = bridge.request({"op": "native_task_status"})
        initial_contact_audit = _robot_contact_audit(
            {"contact_identities": initial_geometry.get("contact_identities", [])},
            robot_geom_ids, finger_geom_ids, int(target["body_id"]), set(target["geom_ids"]),
        )
        report["t10_current_policy_preflight"] = {
            "geometry_status": initial_geometry.get("status"),
            "geometry_clearances_by_object_m": initial_geometry.get("clearances_by_object"),
            "forbidden_contacts": initial_geometry.get("forbidden_contacts"),
            "self_contacts": initial_geometry.get("self_contacts"),
            "forbidden_object_contacts": initial_geometry.get("forbidden_object_contacts"),
            "unallowed_scene_contacts": initial_geometry.get("unallowed_scene_contacts"),
            "unresolved_contacts": initial_geometry.get("unresolved_contacts"),
            "joint_limit_violations": initial_geometry.get("joint_limit_violations"),
            "robot_contact_audit": initial_contact_audit,
            "observed_contacts": initial_contacts.get("contacts", []),
            "grasp_state": initial_grasp,
            "native_task_status": initial_native,
            "native_safety_pass": _native_substep_pass(initial_geometry, initial_contact_audit),
            "operational_safety_pass": _operational_substep_pass(initial_geometry, {
                str(item["object_id"]): float(item["minimum_clearance_m"])
                for group in (policy["forbidden_objects"], policy["allowed_target_objects"], policy["other_scene_objects"])
                for item in group
            }),
        }
        if initial_geometry.get("status") != "complete" or not report["t10_current_policy_preflight"]["native_safety_pass"] or not report["t10_current_policy_preflight"]["operational_safety_pass"]:
            report["outcome"] = "stopped_before_action_t10_snapshot_fails_current_safety_preflight"
            return report

        start_export = bridge.request({"op": "export_runtime_snapshot"})
        if start_export.get("status") != "complete":
            raise RuntimeError("t10_start_snapshot_export_failed")
        start_snapshot_bytes = start_export["payload"]
        (snapshots_dir / "start_timestep_000010.pkl").write_bytes(start_snapshot_bytes)
        pre_agent = _camera_frame(current, "agentview_image")
        pre_wrist = _camera_frame(current, "robot0_eye_in_hand_image")
        pre_agent_frames.append(pre_agent)
        pre_wrist_frames.append(pre_wrist)
        required_margins = {
            str(item["object_id"]): float(item["minimum_clearance_m"])
            for group in (policy["forbidden_objects"], policy["allowed_target_objects"], policy["other_scene_objects"])
            for item in group
        }
        report["physical_replay_started"] = True
        for index, action in enumerate(actions):
            archive_timestep = int(archived_timesteps[index])
            phase = str(phase_labels[index])
            if archive_timestep != int(current.get("timestep", -1)) + 1:
                raise RuntimeError(f"archive_timestep_discontinuity_at_action_{index}:t{archive_timestep}")
            held, grasp_inventory_complete, grasp_states = backend._grasp_states()
            if not grasp_inventory_complete:
                raise RuntimeError(f"grasp_inventory_incomplete_before_t{archive_timestep}")
            active_policy = _active_policy_for_step(
                backend, policy, held, phase, carried_support_pairs
            )
            reply = bridge.request({
                "op": "step_monitored",
                "action": action.tolist(),
                "action_index": index,
                "safety_geometry_policy": active_policy,
            })
            terminal = reply
            terminal_timestep = int(reply.get("timestep", -1))
            if terminal_timestep != archive_timestep:
                raise RuntimeError(f"replay_timestep_mismatch_expected_{archive_timestep}_got_{terminal_timestep}")
            monitor = reply.get("contact_monitoring", {})
            geometry_monitor = monitor.get("geometry_monitoring", {})
            substeps = geometry_monitor.get("substeps", [])
            action_native_pass = True
            action_operational_pass = True
            action_strict10_pass = True
            action_minimums: dict[str, float] = {}
            illegal_first_substep = None
            for substep_index, substep in enumerate(substeps):
                contact_audit = _robot_contact_audit(
                    substep, robot_geom_ids, finger_geom_ids,
                    int(target["body_id"]), set(target["geom_ids"]),
                )
                native_pass = _native_substep_pass(substep, contact_audit)
                operational_pass = _operational_substep_pass(substep, required_margins)
                all_distances = []
                for group_id, clearance in substep.get("clearances_by_object", {}).items():
                    distance = clearance.get("minimum_distance_m") if isinstance(clearance, dict) else None
                    if distance is not None:
                        distance = float(distance)
                        action_minimums[group_id] = min(distance, action_minimums.get(group_id, math.inf))
                        all_distances.append(distance)
                for group_id, clearance in substep.get("carried_clearances_by_object", {}).items():
                    distance = clearance.get("minimum_distance_m") if isinstance(clearance, dict) else None
                    if distance is not None:
                        distance = float(distance)
                        action_minimums[f"carried:{TARGET_OBJECT}->{group_id}"] = min(
                            distance, action_minimums.get(f"carried:{TARGET_OBJECT}->{group_id}", math.inf)
                        )
                        all_distances.append(distance)
                for pair_clearance in substep.get("allowed_contact_clearances_m", {}).values():
                    all_distances.append(float(pair_clearance))
                strict10_pass = bool(all_distances) and min(all_distances) >= 0.010
                action_native_pass &= native_pass
                action_operational_pass &= operational_pass
                action_strict10_pass &= strict10_pass
                substep_rows.append({
                    "timestep": archive_timestep,
                    "action_index_0based": index,
                    "phase_from_archived_event": phase,
                    "physics_substep_index_0based": substep_index,
                    "native_safety_pass": native_pass,
                    "operational_safety_pass": operational_pass,
                    "strict_10mm_clearance_pass": strict10_pass,
                    "robot_contact_audit": contact_audit,
                    "geometry": substep,
                })
                if not native_pass or not operational_pass:
                    if illegal_first_substep is None:
                        illegal_first_substep = {
                            "substep_index_0based": substep_index,
                            "native_safety_pass": native_pass,
                            "operational_safety_pass": operational_pass,
                            "robot_contact_audit": contact_audit,
                            "geometry": substep,
                        }
            post_grasp = backend._grasp_states()
            post_native = bridge.request({"op": "native_task_status"})
            post_contacts = reply.get("contacts", {})
            expected_capture = monitor.get("captured_physics_substeps") == EXPECTED_SUBSTEPS_PER_ACTION
            complete = (
                monitor.get("status") == "complete"
                and geometry_monitor.get("status") in {"complete", "unsafe"}
                and len(substeps) == EXPECTED_SUBSTEPS_PER_ACTION
                and expected_capture
                and all(item.get("status") in {"complete", "unsafe"} for item in substeps)
            )
            row = {
                "timestep": archive_timestep,
                "action_index_0based": index,
                "archived_phase": phase,
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "pre_observation": {
                    "timestep": archive_timestep - 1,
                    "robot_states": current.get("snapshot_state", {}).get("robot_states"),
                    "object_states": current.get("snapshot_state", {}).get("object_states"),
                    "native_task_status": initial_native if index == 0 else rows[-1]["native_task_status_post"],
                },
                "post_snapshot_state": reply.get("snapshot_state"),
                "native_task_status_post": post_native,
                "pre_grasp_states": grasp_states,
                "post_grasp_states": post_grasp[2],
                "post_contacts": post_contacts,
                "contact_monitoring_status": monitor.get("status"),
                "captured_physics_substeps": monitor.get("captured_physics_substeps"),
                "geometry_monitoring_status": geometry_monitor.get("status"),
                "substep_count": len(substeps),
                "native_safety_pass": bool(complete and action_native_pass),
                "operational_safety_pass": bool(complete and action_operational_pass),
                "strict_10mm_all_geometry_clearance_pass": bool(action_strict10_pass),
                "minimum_clearance_by_object_m": action_minimums,
                "first_failure_substep": illegal_first_substep,
                "reward": float(reply.get("reward", 0.0)),
                "done": bool(reply.get("done", False)),
                "task_success": bool(post_native.get("task_success", False)),
                "monitor_complete": bool(complete),
            }
            rows.append(row)
            post_agent = _camera_frame(reply, "agentview_image")
            post_wrist = _camera_frame(reply, "robot0_eye_in_hand_image")
            agent_frames.append(post_agent)
            wrist_frames.append(post_wrist)
            pre_agent_frames.append(post_agent)
            pre_wrist_frames.append(post_wrist)
            report["physical_action_count"] = len(rows)
            report["physical_substep_count"] = len(substep_rows)
            current = reply
            if (index + 1) % CHECKPOINT_STRIDE == 0:
                checkpoint = bridge.request({"op": "export_runtime_snapshot"})
                if checkpoint.get("status") == "complete":
                    (snapshots_dir / f"checkpoint_timestep_{archive_timestep:06d}.pkl").write_bytes(checkpoint["payload"])
            if not complete:
                report["failure"] = {
                    "category": "monitor_incomplete",
                    "timestep": archive_timestep,
                    "contact_monitoring_status": monitor.get("status"),
                    "geometry_monitoring_status": geometry_monitor.get("status"),
                    "captured_substeps": monitor.get("captured_physics_substeps"),
                    "expected_substeps": EXPECTED_SUBSTEPS_PER_ACTION,
                }
                break
            if illegal_first_substep is not None:
                report["failure"] = {
                    "category": "real_safety_violation",
                    "timestep": archive_timestep,
                    "first_substep": illegal_first_substep,
                }
                break
            if bool(reply.get("done")) or post_native.get("task_success") is True:
                report["failure"] = {
                    "category": "unexpected_terminal_during_prefix_replay",
                    "timestep": archive_timestep,
                    "done": bool(reply.get("done")),
                    "task_success": post_native.get("task_success"),
                }
                break
        final = bridge.request({"op": "observe", "include_snapshot_state": True})
        end_export = bridge.request({"op": "export_runtime_snapshot"})
        if end_export.get("status") == "complete":
            end_snapshot_bytes = end_export["payload"]
            (snapshots_dir / f"end_timestep_{int(final.get('timestep', -1)):06d}.pkl").write_bytes(end_snapshot_bytes)
        terminal_held, _, terminal_grasps = backend._grasp_states()
        terminal_native = bridge.request({"op": "native_task_status"})
        terminal_repair = bridge.request({"op": "repair_context"})
        terminal_grasp = _grasp_state(bridge, backend)
        terminal_contacts = bridge.request({"op": "inspect_contacts"})
        final_state = final.get("snapshot_state", {})
        ref_state = source249_observation.get("snapshot_state", {})
        integration_error = _max_abs(
            final_state.get("mujoco_integration_state"),
            ref_state.get("mujoco_integration_state"),
        )
        final_objects = final_state.get("object_states", {})
        ref_objects = ref_state.get("object_states", {})
        object_pose_diffs = {}
        for key in sorted(set(final_objects) & set(ref_objects)):
            left, right = final_objects[key], ref_objects[key]
            object_pose_diffs[key] = {
                "root_pos_max_abs_m": _max_abs(left.get("root_pos"), right.get("root_pos")),
                "root_quat_max_abs": _max_abs(left.get("root_quat"), right.get("root_quat")),
                "joint_qpos_max_abs": _max_abs(left.get("joint_qpos"), right.get("joint_qpos")),
            }
        final_robot = (final_state.get("robot_states") or [{}])[0]
        ref_robot = (ref_state.get("robot_states") or [{}])[0]
        robot_state_diffs = {
            "joint_qpos_max_abs_rad": _max_abs(final_robot.get("joint_qpos"), ref_robot.get("joint_qpos")),
            "joint_qvel_max_abs": _max_abs(final_robot.get("joint_qvel"), ref_robot.get("joint_qvel")),
            "eef_position_max_abs_m": _max_abs(final_robot.get("eef_pos"), ref_robot.get("eef_pos")),
            "eef_quaternion_max_abs": _max_abs(final_robot.get("eef_quat"), ref_robot.get("eef_quat")),
            "gripper_qpos_max_abs_m": _max_abs(final_robot.get("gripper_qpos"), ref_robot.get("gripper_qpos")),
        }
        final_runtime = _snapshot_runtime_fields(end_snapshot_bytes) if end_snapshot_bytes else {}
        runtime_compare = _runtime_state_comparison(source249_runtime, final_runtime) if final_runtime else {"status": "unavailable"}
        source_contact_identities = source249_contacts.get("contacts", [])
        replay_contact_identities = terminal_contacts.get("contacts", [])
        report["terminal_comparison"] = {
            "replay_timestep": int(final.get("timestep", -1)),
            "source_timestep": int(source249_observation.get("timestep", -1)),
            "source_t249_snapshot_sha256": hashlib.sha256(t249_snapshot_bytes).hexdigest(),
            "replay_integration_state_sha256": _integration_hash(final_state),
            "source_integration_state_sha256": _integration_hash(ref_state),
            "integration_state_exact": integration_error == 0.0,
            "integration_state_max_abs_difference": integration_error,
            "robot_state_max_abs_differences": robot_state_diffs,
            "object_pose_max_abs_differences": object_pose_diffs,
            "runtime_controller_and_gripper_state": runtime_compare,
            "replay_task_phase": str(phase_labels[len(rows) - 1]) if rows else None,
            "source_episode9_task_phase": str(source["episode9_phase_labels"][-1]),
            "task_phase_controller_state_in_runtime_snapshot": False,
            "terminal_grasp_state": terminal_grasp,
            "source_grasp_state": source249_grasp,
            "replay_all_object_grasp_states": terminal_grasps,
            "source_native_task_status": source249_native,
            "replay_native_task_status": terminal_native,
            "source_robot_contact_identities": source_contact_identities,
            "replay_robot_contact_identities": replay_contact_identities,
            "controller_diagnostic_context_source": source249_repair,
            "controller_diagnostic_context_replay": terminal_repair,
        }
        complete_trace = len(rows) == EXPECTED_ACTION_COUNT and all(row["monitor_complete"] for row in rows)
        all_native = bool(rows) and all(row["native_safety_pass"] for row in rows)
        all_operational = bool(rows) and all(row["operational_safety_pass"] for row in rows)
        state_compatible = (
            int(final.get("timestep", -1)) == 249
            and integration_error <= 1e-10
            and all(value <= 1e-10 for row in object_pose_diffs.values() for value in row.values())
            and all(value <= 1e-10 for value in robot_state_diffs.values())
            and runtime_compare.get("status") == "pass"
            and terminal_grasp.get("status") == "pass"
            and terminal_grasp.get("grasp_confirmed") is True
            and terminal_held == TARGET_OBJECT
        )
        report["acceptance"] = {
            "all_239_actions_replayed": len(rows) == EXPECTED_ACTION_COUNT,
            "all_replayed_actions_have_25_substep_records": complete_trace and len(substep_rows) == EXPECTED_ACTION_COUNT * EXPECTED_SUBSTEPS_PER_ACTION,
            "native_safety_pass": all_native,
            "operational_safety_pass": all_operational,
            "strict_10mm_all_geometry_clearance_pass": bool(rows) and all(
                row["strict_10mm_all_geometry_clearance_pass"] for row in rows
            ),
            "endpoint_t249_state_compatible_with_success_chain_source": state_compatible,
            "stable_dual_finger_grasp_at_t249": terminal_grasp.get("grasp_confirmed") is True and terminal_grasp.get("status") == "pass",
            "pot2_task_success_at_t249_expected_false": terminal_native.get("task_success") is False,
            "complete_round0_seed": False,
        }
        if report.get("failure"):
            report["outcome"] = report["failure"]["category"]
        elif not complete_trace:
            report["outcome"] = "incomplete_replay_or_missing_substep_evidence"
        elif not all_native or not all_operational:
            report["outcome"] = "current_safety_failure_in_prefix"
        elif not state_compatible:
            report["outcome"] = "safe_replay_but_t249_state_incompatible_with_success_chain"
        else:
            report["outcome"] = "prefix_current_safety_pass_and_t249_state_continuity_verified"
        report["continuity_interpretation"] = (
            "TaskPhaseController is not serialized in MuJoCo Runtime Snapshots; next replay must initialize it from the restored physical state. "
            "This report certifies only the exact archived t=10..249 physical prefix under the frozen current Native/Operational policy."
        )
        report["physical_action_count"] = len(rows)
        report["physical_substep_count"] = len(substep_rows)
        report["minimum_clearance_by_object_m"] = {
            key: min(row.get("minimum_clearance_by_object_m", {}).get(key, math.inf) for row in rows)
            for key in sorted({key for row in rows for key in row.get("minimum_clearance_by_object_m", {})})
        }
        report["wall_time_seconds"] = time.monotonic() - started
        return report
    except Exception as exc:
        report["outcome"] = "audit_or_replay_exception"
        report["failure"] = {"category": "runtime_or_validation_error", "error": f"{type(exc).__name__}: {exc}"}
        report["physical_action_count"] = len(rows)
        report["physical_substep_count"] = len(substep_rows)
        return report
    finally:
        if not (snapshots_dir / "start_timestep_000010.pkl").exists():
            (snapshots_dir / "start_timestep_000010.pkl").write_bytes(
                start_snapshot_bytes if start_snapshot_bytes is not None else source_snapshot_bytes
            )
        if end_snapshot_bytes is None and bridge.connection is not None:
            try:
                current_export = bridge.request({"op": "export_runtime_snapshot"})
                if current_export.get("status") == "complete":
                    end_snapshot_bytes = current_export["payload"]
            except Exception:
                pass
        if end_snapshot_bytes is None and not rows:
            end_snapshot_bytes = source_snapshot_bytes
        if end_snapshot_bytes is not None and not any(snapshots_dir.glob("end_timestep_*.pkl")):
            timestep = rows[-1]["timestep"] if rows else 10
            (snapshots_dir / f"end_timestep_{timestep:06d}.pkl").write_bytes(end_snapshot_bytes)
        np.savez_compressed(
            output_dir / "observation_action_trace.npz",
            step_timesteps=np.asarray([row["timestep"] for row in rows], dtype=np.int32),
            executed_actions=np.asarray([row["action"] for row in rows], dtype=np.float32).reshape((-1, 7)),
            rewards=np.asarray([row["reward"] for row in rows], dtype=np.float32),
            dones=np.asarray([row["done"] for row in rows], dtype=np.bool_),
            task_success=np.asarray([row["task_success"] for row in rows], dtype=np.bool_),
            task_phase=np.asarray([row["archived_phase"] for row in rows], dtype="U32"),
            pre_agentview=np.asarray(pre_agent_frames[:-1], dtype=np.uint8),
            post_agentview=np.asarray(agent_frames, dtype=np.uint8),
            pre_eye_in_hand=np.asarray(pre_wrist_frames[:-1], dtype=np.uint8),
            post_eye_in_hand=np.asarray(wrist_frames, dtype=np.uint8),
            metadata_json=np.asarray(json.dumps({
                "schema": "gs_opd_episode9_t10_to_t249_current_safety_replay.v1",
                "source_action_trace": str(T10_TO_T249_TRACE),
                "source_action_trace_sha256": hashlib.sha256(T10_TO_T249_TRACE.read_bytes()).hexdigest(),
                "start_snapshot": str(T10_SNAPSHOT),
                "end_source_snapshot": str(EPISODE9_T249_SNAPSHOT),
                "camera_keys": ["agentview", "robot0_eye_in_hand"],
                "substeps_per_action": EXPECTED_SUBSTEPS_PER_ACTION,
            }, sort_keys=True)),
        )
        with (output_dir / "action_records.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        with (output_dir / "substep_safety.jsonl").open("w", encoding="utf-8") as stream:
            for row in substep_rows:
                stream.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
        for field, frames in (("agentview", agent_frames), ("wrist", wrist_frames)):
            try:
                _write_video(output_dir / f"{field}.mp4", frames)
            except Exception as exc:
                report["video_write_error"] = f"{type(exc).__name__}: {exc}"
        report["data_paths"] = {
            "report": str(output_dir / "prefix_replay_report.json"),
            "observation_action_trace": str(output_dir / "observation_action_trace.npz"),
            "action_records": str(output_dir / "action_records.jsonl"),
            "substep_safety": str(output_dir / "substep_safety.jsonl"),
            "start_snapshot": str(snapshots_dir / "start_timestep_000010.pkl"),
            "end_snapshot": str(next(iter(sorted(snapshots_dir.glob("end_timestep_*.pkl"))), snapshots_dir / "end_timestep_000010.pkl")),
            "agentview_video": str(output_dir / "agentview.mp4"),
            "wrist_video": str(output_dir / "wrist.mp4"),
            "safety_bridge_log": str(output_dir / "safety_bridge.log"),
        }
        try:
            bridge.close()
        finally:
            (output_dir / "prefix_replay_report.json").write_text(
                json.dumps(_jsonable(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


def main() -> int:
    args = _arguments()
    report = run(args)
    print(json.dumps(_jsonable(report), ensure_ascii=False, sort_keys=True))
    return 0 if report.get("outcome") == "prefix_current_safety_pass_and_t249_state_continuity_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
