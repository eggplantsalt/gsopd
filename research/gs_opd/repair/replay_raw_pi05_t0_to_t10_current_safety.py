# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Rebuild the raw L0T2 initial state and replay only raw Pi0.5 actions 0..9."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
from research.gs_opd.repair.replay_episode9_t10_to_t249_current_safety import (
    EXPECTED_BDDL_SHA256,
    EXPECTED_MODEL_SHA256,
    EPISODE9_DIR,
    RAW_STUDENT_DIR,
    T0_TO_T10_TRACE,
    T10_AUDIT_DIR,
    T10_SNAPSHOT,
    T10_VERIFICATION,
    T249_CHAIN_POLICY_REPORT,
    _canonical_sha256,
    _camera_frame,
    _grasp_state,
    _integration_hash,
    _jsonable,
    _make_current_policy,
    _max_abs,
    _native_substep_pass,
    _operational_substep_pass,
    _robot_contact_audit,
    _runtime_state_comparison,
    _snapshot_runtime_fields,
    _write_video,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROLLOUT_ROOT = Path("/opt/data/private/gs_opd/rollouts")
DEFAULT_CONFIG = REPO_ROOT / "research/gs_opd/configs/l0t2_task_success_first_safety_reconstruction.json"
T10_EPISODE9_START = EPISODE9_DIR / "start_state.json"
EXPECTED_ACTIONS = 10
EXPECTED_SUBSTEPS_PER_ACTION = 25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n")


def _camera_image(observation: dict[str, Any], key: str) -> np.ndarray:
    value = observation.get("images", {}).get(key)
    if not isinstance(value, dict):
        raise ValueError(f"required_camera_observation_missing:{key}")
    return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"]).copy()


def _margins(policy: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for group_name in ("forbidden_objects", "allowed_target_objects", "other_scene_objects"):
        for item in policy.get(group_name, []):
            result[str(item["object_id"])] = float(item["minimum_clearance_m"])
    return result


def _all_robot_ids(bridge: BridgeProcess) -> set[int]:
    inventory = bridge.request({"op": "geometry_inventory"})
    if inventory.get("status") != "complete":
        raise ValueError("full_robot_collision_geometry_inventory_incomplete")
    return {
        int(item["geom_id"])
        for item in inventory.get("geometries", [])
        if item.get("category") == "robot" and item.get("collision_enabled") is True
    }


def _compare_object_state(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    details = {}
    for semantic_id in (
        "objects_dict:moka_pot_1",
        "objects_dict:moka_pot_2",
        "objects_dict:bottle_of_ginger_beer_1",
        "fixtures_dict:white_place_box_1",
    ):
        got = actual.get(semantic_id)
        ref = expected.get(semantic_id)
        if got is None or ref is None:
            details[semantic_id] = {"available": False}
            continue
        details[semantic_id] = {
            "available": True,
            "root_position_max_abs_m": _max_abs(got.get("root_pos"), ref.get("root_pos")),
            "root_quaternion_max_abs": _max_abs(got.get("root_quat"), ref.get("root_quat")),
            "joint_qpos_max_abs": _max_abs(got.get("joint_qpos", []), ref.get("joint_qpos", [])),
        }
    return {
        "objects": details,
        "all_available_objects_exact": bool(details) and all(
            item.get("available")
            and item.get("root_position_max_abs_m") == 0.0
            and item.get("root_quaternion_max_abs") == 0.0
            and item.get("joint_qpos_max_abs") == 0.0
            for item in details.values()
        ),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38784)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROLLOUT_ROOT / f"l0t2_round0_raw_pi05_t0_to_t10_current_safety_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir()

    raw_trace_path = RAW_STUDENT_DIR / "rollout_trace.npz"
    rollout_manifest_path = RAW_STUDENT_DIR / "run_manifest.json"
    episode_start = json.loads(T10_EPISODE9_START.read_text())
    expected_t10_state = episode_start["snapshot_state"]
    t10_source_bytes = T10_SNAPSHOT.read_bytes()
    t10_source_runtime = _snapshot_runtime_fields(t10_source_bytes)
    frozen_policy_report = json.loads(T249_CHAIN_POLICY_REPORT.read_text())
    frozen_policy = frozen_policy_report.get("policy")
    if not isinstance(frozen_policy, dict):
        raise ValueError("frozen_current_operational_policy_missing")

    with np.load(raw_trace_path, allow_pickle=False) as raw, np.load(T0_TO_T10_TRACE, allow_pickle=False) as archived:
        raw_metadata = json.loads(str(raw["metadata_json"]))
        raw_actions = np.asarray(raw["executed_actions"][:EXPECTED_ACTIONS], dtype=np.float32)
        actions = np.asarray(archived["actions"], dtype=np.float32)
        if actions.shape != (EXPECTED_ACTIONS, 7) or raw_actions.shape != (EXPECTED_ACTIONS, 7):
            raise ValueError("raw_pi05_prefix_is_not_exactly_10x7")
        actions_exact = bool(np.array_equal(actions, raw_actions))
        if not actions_exact:
            raise ValueError("archived_t0_to_t10_actions_do_not_match_raw_pi05_executed_actions")
        expected_initial_integration = np.asarray(
            raw_metadata["initial_physical_state"]["mujoco_integration_state"], dtype=np.float64
        )
        expected_t10_archive = {
            "pre_agentview": np.asarray(archived["pre_agentview"], dtype=np.uint8),
            "pre_eye_in_hand": np.asarray(archived["pre_eye_in_hand"], dtype=np.uint8),
        }

    config_data, recovery = load_runtime_config(args.config)
    if int(config_data["environment_seed"]) != int(raw_metadata["env_seed"]):
        raise ValueError("configured_seed_does_not_match_raw_student_seed")
    if int(config_data["init_state"]) != 0 or int(raw_metadata.get("index_base", -1)) != 0:
        raise ValueError("t0_replay_requires_original_init_state_zero_and_zero_based_trace")

    report: dict[str, Any] = {
        "schema": "gs_opd_round0_raw_pi05_t0_to_t10_current_safety.v1",
        "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "one_native_initialization_from_raw_seed_init_state_then_only_raw_pi05_actions_0_through_9",
        "after_t10_actions_executed": False,
        "search_rrt_controller_optimization_or_training": False,
        "source_provenance": {
            "raw_rollout_trace": str(raw_trace_path),
            "raw_rollout_trace_sha256": _sha256(raw_trace_path),
            "raw_rollout_manifest": str(rollout_manifest_path),
            "raw_rollout_manifest_sha256": _sha256(rollout_manifest_path),
            "raw_student_action_source": raw_metadata.get("action_source"),
            "raw_pi05_seed": raw_metadata.get("env_seed"),
            "init_state_index": config_data["init_state"],
            "model_checkpoint_sha256": raw_metadata.get("checkpoint_sha256"),
            "t0_to_t10_action_archive": str(T0_TO_T10_TRACE),
            "t0_to_t10_action_archive_sha256": _sha256(T0_TO_T10_TRACE),
            "t0_to_t10_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "actions_exactly_match_raw_pi05_executed_actions": actions_exact,
            "action_rows_shape": list(actions.shape),
            "raw_initial_integration_state_sha256": hashlib.sha256(expected_initial_integration.tobytes()).hexdigest(),
            "verified_t10_runtime_snapshot": str(T10_SNAPSHOT),
            "verified_t10_runtime_snapshot_sha256": hashlib.sha256(t10_source_bytes).hexdigest(),
        },
        "initialization": {
            "bridge_native_startup_matches_original_rollout_sequence": "Bridge OffScreenRenderEnv(seed=0)/reset/set_init_state, then the original Pi0.5 Rollout reset RPC (reset/set_init_state) before action 0",
            "original_rollout_reset_rpc_after_bridge_startup": True,
            "expected_model_geometry_sha256": EXPECTED_MODEL_SHA256,
            "expected_bddl_sha256": EXPECTED_BDDL_SHA256,
        },
        "safety_policy": {
            "source": str(T249_CHAIN_POLICY_REPORT),
            "policy_sha256": _canonical_sha256(frozen_policy),
            "pot2_nonfinger_min_clearance_m": 0.002,
            "ginger_beer_min_clearance_m": 0.010,
            "place_box_min_clearance_m": 0.010,
            "legal_finger_to_pot2_max_penetration_m": 0.002,
            "self_collision_check": True,
            "full_robot_geometry_check": True,
            "joint_limit_operational_calibration": copy.deepcopy(frozen_policy.get("joint_limit_tolerance_policy")),
            "strict_10mm_is_additional_metric_only": True,
        },
        "physical_action_count": 0,
        "physical_substep_count": 0,
        "outcome": None,
        "failure": None,
    }

    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend = None
    rows: list[dict[str, Any]] = []
    substep_rows: list[dict[str, Any]] = []
    pre_agent_frames: list[np.ndarray] = []
    post_agent_frames: list[np.ndarray] = []
    pre_wrist_frames: list[np.ndarray] = []
    post_wrist_frames: list[np.ndarray] = []
    pre_states: list[list[float]] = []
    post_states: list[list[float]] = []
    start_snapshot_payload: bytes | None = None
    final_snapshot_payload: bytes | None = None

    try:
        bridge.start(
            benchmark=config_data["benchmark"], level=int(config_data["level"]),
            task_id=int(config_data["task_id"]), init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]), demonstration_horizon=1000,
        )
        first_compile_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
        first_compile_fingerprint = bridge.request({"op": "compiled_model_fingerprint_details"})
        native_reset = bridge.request({"op": "reset"})
        startup = bridge.request({"op": "observe", "include_snapshot_state": True})
        fingerprint_details = bridge.request({"op": "compiled_model_fingerprint_details"})
        scene = startup.get("scene_metadata", {})
        initial_state = startup.get("snapshot_state", {})
        actual_initial_integration = np.asarray(initial_state.get("mujoco_integration_state", []), dtype=np.float64)
        initial_state_error = _max_abs(actual_initial_integration, expected_initial_integration)
        report["initialization"].update({
            "first_compile_model_geometry_sha256": first_compile_observation.get("scene_metadata", {}).get("model_geometry_sha256"),
            "first_compile_integration_state_sha256": _integration_hash(first_compile_observation.get("snapshot_state", {})),
            "first_compile_fingerprint_details": first_compile_fingerprint,
            "original_rollout_reset_rpc_status": native_reset.get("status", "response_without_status_field"),
            "original_rollout_reset_rpc_timestep": native_reset.get("timestep"),
            "post_rollout_reset_timestep": int(startup.get("timestep", -1)),
            "post_rollout_reset_fingerprint_details": fingerprint_details,
            "bridge_timestep": int(startup.get("timestep", -1)),
            "model_geometry_sha256": scene.get("model_geometry_sha256"),
            "bddl_sha256": scene.get("bddl_sha256"),
            "compiled_model_fingerprint_details": fingerprint_details,
            "actual_initial_integration_state_sha256": _integration_hash(initial_state),
            "raw_rollout_initial_integration_state_max_abs_error": initial_state_error,
            "raw_rollout_initial_integration_state_exact": initial_state_error == 0.0,
            "native_task_status_t0": bridge.request({"op": "native_task_status"}),
        })
        if int(native_reset.get("timestep", -1)) != 0 or int(startup.get("timestep", -1)) != 0:
            raise RuntimeError("native_initialization_did_not_start_at_timestep_0")
        if scene.get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError("native_initialization_model_geometry_fingerprint_mismatch")
        if scene.get("bddl_sha256") != EXPECTED_BDDL_SHA256:
            raise RuntimeError("native_initialization_bddl_fingerprint_mismatch")
        if initial_state_error != 0.0:
            raise RuntimeError("native_initialization_does_not_match_raw_student_t0_integration_state")

        backend = AutomaticRecoveryBridgeBackend(
            bridge, config_data, recovery, output_dir, RecoveryArtifactWriter(output_dir),
            mode="episode", student=None,
            archived_trace_path=raw_trace_path,
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=T10_SNAPSHOT,
            t10_verification_path=T10_VERIFICATION,
        )
        backend.latest_data = startup
        backend._load_context_inputs(startup)
        policy = _make_current_policy(backend, frozen_policy)
        margins = _margins(policy)
        geometry_inventory = bridge.request({"op": "geometry_inventory"})
        robot_ids = _all_robot_ids(bridge)
        finger_ids = set(int(value) for value in backend.context_static["gripper_geometry_ids"])
        target_binding = {
            "body_id": int(backend.bindings["moka_pot_2"]["root_body_id"]),
            "geom_ids": [int(x) for x in backend.bindings["moka_pot_2"]["root_geom_ids"]],
        }
        finger_inventory = bridge.request({"op": "gripper_geometry_inventory"})
        reported_fingers = {
            int(item["geom_id"])
            for group in finger_inventory.get("grippers", [])
            for item in group.get("geometries", [])
            if str(item.get("geom_name", "")).endswith((
                "finger1_collision", "finger1_pad_collision",
                "finger2_collision", "finger2_pad_collision",
            ))
        }
        if finger_inventory.get("status") != "complete" or reported_fingers != finger_ids or len(finger_ids) != 4:
            raise RuntimeError("four_finger_collision_geometry_binding_mismatch")
        robot_ids_from_policy = set(int(x) for x in policy.get("robot_geom_ids", []))
        if robot_ids_from_policy and robot_ids_from_policy != robot_ids:
            raise RuntimeError("full_robot_geometry_inventory_differs_from_frozen_safety_policy")

        preflight = bridge.request({
            "op": "observe", "include_snapshot_state": True,
            "safety_geometry_policy": policy,
        })
        geometry = preflight.get("safety_geometry", {})
        target = backend.bindings["moka_pot_2"]
        preflight_contact = _robot_contact_audit(
            {"contact_identities": geometry.get("contact_identities", [])},
            robot_ids, finger_ids, int(target["root_body_id"]),
            set(int(value) for value in target["root_geom_ids"]),
        )
        grasp = _grasp_state(bridge, backend)
        native = bridge.request({"op": "native_task_status"})
        preflight_native = _native_substep_pass(geometry, preflight_contact)
        preflight_operational = _operational_substep_pass(geometry, margins)
        report["t0_preflight"] = {
            "geometry_status": geometry.get("status"),
            "native_safety_pass": preflight_native,
            "operational_safety_pass": preflight_operational,
            "geometry_clearances_by_object_m": geometry.get("clearances_by_object"),
            "forbidden_contacts": geometry.get("forbidden_contacts"),
            "self_contacts": geometry.get("self_contacts"),
            "forbidden_object_contacts": geometry.get("forbidden_object_contacts"),
            "unallowed_scene_contacts": geometry.get("unallowed_scene_contacts"),
            "unresolved_contacts": geometry.get("unresolved_contacts"),
            "joint_limit_violations": geometry.get("joint_limit_violations"),
            "robot_contact_audit": preflight_contact,
            "grasp_state": grasp,
            "native_task_status": native,
            "geometry_inventory_status": geometry_inventory.get("status"),
            "full_collision_robot_geom_count": len(robot_ids),
            "four_finger_geom_ids": sorted(finger_ids),
            "current_clearance_thresholds_m": margins,
        }
        if geometry.get("status") != "complete" or not preflight_native or not preflight_operational:
            report["outcome"] = "stopped_before_actions_t0_preflight_failed"
            raise RuntimeError("t0_current_native_or_operational_safety_preflight_failed")

        start_export = bridge.request({"op": "export_runtime_snapshot"})
        if start_export.get("status") != "complete":
            raise RuntimeError("t0_runtime_snapshot_export_failed")
        start_snapshot_payload = start_export["payload"]
        (snapshots_dir / "start_timestep_000000.pkl").write_bytes(start_snapshot_payload)
        for index, action in enumerate(actions):
            current_timestep = index
            expected_timestep = index + 1
            before = bridge.request({"op": "observe", "include_snapshot_state": True}) if index else startup
            if int(before.get("timestep", -1)) != current_timestep:
                raise RuntimeError(f"pre_action_timestep_mismatch_at_index_{index}")
            before_state = before.get("snapshot_state", {})
            reply = bridge.request({
                "op": "step_monitored",
                "action": action.tolist(),
                "action_index": index,
                "safety_geometry_policy": policy,
            })
            if int(reply.get("timestep", -1)) != expected_timestep:
                raise RuntimeError(f"post_action_timestep_mismatch_at_index_{index}")
            monitor = reply.get("contact_monitoring", {})
            geometry_monitor = monitor.get("geometry_monitoring", {})
            substeps = geometry_monitor.get("substeps", [])
            complete = (
                monitor.get("status") == "complete"
                and geometry_monitor.get("status") in {"complete", "unsafe"}
                and len(substeps) == EXPECTED_SUBSTEPS_PER_ACTION
                and monitor.get("captured_physics_substeps") == EXPECTED_SUBSTEPS_PER_ACTION
                and all(row.get("status") in {"complete", "unsafe"} for row in substeps)
            )
            action_native = bool(complete)
            action_operational = bool(complete)
            action_strict10 = bool(complete)
            action_minimums: dict[str, float] = {}
            first_failure = None
            action_substeps: list[dict[str, Any]] = []
            for substep_index, substep in enumerate(substeps):
                contact_audit = _robot_contact_audit(
                    substep, robot_ids, finger_ids,
                    int(target["root_body_id"]), set(int(x) for x in target["root_geom_ids"]),
                )
                native_pass = _native_substep_pass(substep, contact_audit)
                operational_pass = _operational_substep_pass(substep, margins)
                distances = []
                for object_id, value in substep.get("clearances_by_object", {}).items():
                    distance = value.get("minimum_distance_m") if isinstance(value, dict) else None
                    if distance is not None:
                        distance = float(distance)
                        action_minimums[str(object_id)] = min(action_minimums.get(str(object_id), math.inf), distance)
                        distances.append(distance)
                for pair_name, distance_value in substep.get("allowed_contact_clearances_m", {}).items():
                    distance = float(distance_value)
                    action_minimums[f"allowed_pair:{pair_name}"] = min(
                        action_minimums.get(f"allowed_pair:{pair_name}", math.inf), distance
                    )
                    distances.append(distance)
                strict_pass = bool(distances) and min(distances) >= 0.010
                action_native &= native_pass
                action_operational &= operational_pass
                action_strict10 &= strict_pass
                row = {
                    "timestep": expected_timestep,
                    "action_index_0based": index,
                    "physics_substep_index_0based": substep_index,
                    "native_safety_pass": native_pass,
                    "operational_safety_pass": operational_pass,
                    "strict_10mm_additional_metric_pass": strict_pass,
                    "robot_contact_audit": contact_audit,
                    "geometry": substep,
                }
                substep_rows.append(row)
                action_substeps.append(row)
                if first_failure is None and (not native_pass or not operational_pass):
                    first_failure = {
                        "physics_substep_index_0based": substep_index,
                        "native_safety_pass": native_pass,
                        "operational_safety_pass": operational_pass,
                        "robot_contact_audit": contact_audit,
                        "geometry": substep,
                    }
            post_native = bridge.request({"op": "native_task_status"})
            held, grasp_complete, grasp_states = backend._grasp_states()
            post_state = reply.get("snapshot_state", {})
            action_pass = bool(complete and action_native and action_operational)
            record = {
                "timestep": expected_timestep,
                "action_index_0based": index,
                "action_source": "raw_pi05_executed_actions_exact_archive",
                "action": action.tolist(),
                "pre_integration_state_sha256": _integration_hash(before_state),
                "post_integration_state_sha256": _integration_hash(post_state),
                "pre_robot_states": before_state.get("robot_states"),
                "post_robot_states": post_state.get("robot_states"),
                "pre_object_states": before_state.get("object_states"),
                "post_object_states": post_state.get("object_states"),
                "grasp_inventory_complete": grasp_complete,
                "grasp_states_post": grasp_states,
                "held_object_post": held,
                "native_task_status_post": post_native,
                "reward": float(reply.get("reward", 0.0)),
                "done": bool(reply.get("done", False)),
                "contact_monitoring_status": monitor.get("status"),
                "captured_physics_substeps": monitor.get("captured_physics_substeps"),
                "native_safety_pass": bool(complete and action_native),
                "operational_safety_pass": bool(complete and action_operational),
                "strict_10mm_additional_metric_pass": bool(complete and action_strict10),
                "minimum_clearance_by_object_m": action_minimums,
                "first_failure_substep": first_failure,
                "substep_rows": action_substeps,
            }
            rows.append(record)
            report["physical_action_count"] += 1
            report["physical_substep_count"] += len(substeps)

            pre_agent_frames.append(_camera_image(before, "agentview_image"))
            pre_wrist_frames.append(_camera_image(before, "robot0_eye_in_hand_image"))
            post_agent_frames.append(_camera_image(reply, "agentview_image"))
            post_wrist_frames.append(_camera_image(reply, "robot0_eye_in_hand_image"))
            pre_states.append(np.asarray(before_state.get("mujoco_integration_state"), dtype=np.float64).tolist())
            post_states.append(np.asarray(post_state.get("mujoco_integration_state"), dtype=np.float64).tolist())

            snapshot = bridge.request({"op": "export_runtime_snapshot"})
            if snapshot.get("status") != "complete":
                raise RuntimeError(f"runtime_snapshot_export_failed_after_t{expected_timestep}")
            final_snapshot_payload = snapshot["payload"]
            (snapshots_dir / f"timestep_{expected_timestep:06d}.pkl").write_bytes(final_snapshot_payload)

            if not action_pass:
                report["outcome"] = "stopped_at_first_current_safety_failure"
                report["failure"] = {
                    "timestep": expected_timestep,
                    "action_index_0based": index,
                    "first_failure_substep": first_failure,
                }
                break
            if bool(reply.get("done", False)):
                report["outcome"] = "stopped_on_native_done_before_expected_t10_boundary"
                report["failure"] = {"timestep": expected_timestep, "done": True}
                break
            startup = reply

        if final_snapshot_payload is None:
            final_snapshot_payload = start_snapshot_payload
        end_timestep = int(rows[-1]["timestep"] if rows else 0)
        (snapshots_dir / f"end_timestep_{end_timestep:06d}.pkl").write_bytes(final_snapshot_payload)
        final_observation = bridge.request({"op": "observe", "include_snapshot_state": True})
        final_state = final_observation.get("snapshot_state", {})
        replay_runtime = _snapshot_runtime_fields(final_snapshot_payload)
        expected_objects = expected_t10_state.get("object_states", {})
        object_compare = _compare_object_state(final_state.get("object_states", {}), expected_objects)
        integration_error = _max_abs(final_state.get("mujoco_integration_state"), t10_source_runtime.get("integration_state"))
        controller_compare = _runtime_state_comparison(t10_source_runtime, replay_runtime)
        report["terminal_comparison"] = {
            "expected_timestep": 10,
            "actual_timestep": int(final_observation.get("timestep", -1)),
            "t10_integration_state_max_abs_error": integration_error,
            "t10_integration_state_exact": integration_error == 0.0,
            "t10_integration_state_sha256_expected": t10_source_runtime.get("integration_state_sha256"),
            "t10_integration_state_sha256_replay": replay_runtime.get("integration_state_sha256"),
            "runtime_controller_gripper_and_task_bookkeeping": controller_compare,
            "object_state_comparison_to_episode9_t10_boundary": object_compare,
            "native_task_status": bridge.request({"op": "native_task_status"}),
            "grasp_state": _grasp_state(bridge, backend),
            "final_runtime_snapshot_sha256": hashlib.sha256(final_snapshot_payload).hexdigest(),
            "expected_t10_runtime_snapshot_sha256": hashlib.sha256(t10_source_bytes).hexdigest(),
            "runtime_snapshot_byte_identical_to_t10_source": final_snapshot_payload == t10_source_bytes,
            "task_phase_controller_python_state_in_runtime_snapshot": False,
        }

        all_actions_pass = len(rows) == EXPECTED_ACTIONS and all(
            row["native_safety_pass"] and row["operational_safety_pass"] for row in rows
        )
        exact_boundary = (
            int(final_observation.get("timestep", -1)) == 10
            and integration_error == 0.0
            and controller_compare["status"] == "pass"
            and object_compare["all_available_objects_exact"]
        )
        if report["outcome"] is None:
            report["outcome"] = "current_safety_pass_and_t10_full_state_exact" if all_actions_pass and exact_boundary else "t0_to_t10_replay_failed_acceptance"
        report["acceptance"] = {
            "10_of_10_archived_actions_executed": len(rows) == EXPECTED_ACTIONS,
            "all_250_substeps_have_complete_current_safety_records": len(substep_rows) == 250 and all(
                row["geometry"].get("status") in {"complete", "unsafe"} for row in substep_rows
            ),
            "all_actions_native_safety_pass": all_actions_pass,
            "all_actions_operational_safety_pass": all_actions_pass,
            "t10_physical_integration_state_and_runtime_compatible": exact_boundary,
            "safety_prefix_t0_to_t10_pass": bool(all_actions_pass and exact_boundary),
        }
        report["strict_metric_summary"] = {
            "strict_10mm_is_not_the_operational_acceptance_rule": True,
            "strict_10mm_all_required_pairs_passed_every_substep": bool(rows) and all(
                row["strict_10mm_additional_metric_pass"] for row in rows
            ),
            "strict_10mm_per_action_pass_count": sum(bool(row["strict_10mm_additional_metric_pass"]) for row in rows),
            "operational_rule_pass_count": sum(bool(row["operational_safety_pass"]) for row in rows),
        }
    except Exception as exc:
        if report.get("outcome") is None:
            report["outcome"] = "failed_or_stopped_with_evidence"
        report["failure"] = report.get("failure") or {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "completed_actions": int(report.get("physical_action_count", 0)),
        }
        report["exception_traceback"] = __import__("traceback").format_exc()
    finally:
        if rows:
            _save_jsonl(output_dir / "action_records.jsonl", rows)
            _save_jsonl(output_dir / "substep_safety.jsonl", substep_rows)
            np.savez_compressed(
                output_dir / "observation_action_trace.npz",
                actions=actions[:len(rows)].astype(np.float32),
                timestep=np.asarray([row["timestep"] for row in rows], dtype=np.int64),
                pre_integration_state=np.asarray(pre_states[:len(rows)], dtype=np.float64),
                post_integration_state=np.asarray(post_states[:len(rows)], dtype=np.float64),
                pre_agentview=np.asarray(pre_agent_frames[:len(rows)], dtype=np.uint8),
                post_agentview=np.asarray(post_agent_frames[:len(rows)], dtype=np.uint8),
                pre_eye_in_hand=np.asarray(pre_wrist_frames[:len(rows)], dtype=np.uint8),
                post_eye_in_hand=np.asarray(post_wrist_frames[:len(rows)], dtype=np.uint8),
                pre_robot0_eef_pos=np.asarray([row["pre_robot_states"][0]["eef_pos"] for row in rows], dtype=np.float64),
                post_robot0_eef_pos=np.asarray([row["post_robot_states"][0]["eef_pos"] for row in rows], dtype=np.float64),
                pre_robot0_eef_quat=np.asarray([row["pre_robot_states"][0]["eef_quat"] for row in rows], dtype=np.float64),
                post_robot0_eef_quat=np.asarray([row["post_robot_states"][0]["eef_quat"] for row in rows], dtype=np.float64),
                pre_robot0_gripper_qpos=np.asarray([row["pre_robot_states"][0]["gripper_qpos"] for row in rows], dtype=np.float64),
                post_robot0_gripper_qpos=np.asarray([row["post_robot_states"][0]["gripper_qpos"] for row in rows], dtype=np.float64),
                metadata_json=np.asarray(json.dumps({
                    "schema": report["schema"],
                    "action_source": "raw_pi05_executed_actions_exact_archive",
                    "raw_rollout_sha256": report["source_provenance"]["raw_rollout_trace_sha256"],
                    "safety_policy_sha256": report["safety_policy"]["policy_sha256"],
                    "control_steps": len(rows),
                }, sort_keys=True)),
            )
            _write_video(output_dir / "agentview.mp4", pre_agent_frames[:1] + post_agent_frames[:len(rows)])
            _write_video(output_dir / "wrist.mp4", pre_wrist_frames[:1] + post_wrist_frames[:len(rows)])
        report["physical_wall_seconds"] = time.monotonic() - started
        report["archive_dir"] = str(output_dir)
        report_path = output_dir / "t0_to_t10_current_safety_report.json"
        _json_dump(report_path, report)
        if bridge is not None:
            bridge.close()
        manifest = {}
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.name != "sha256_manifest.json":
                manifest[str(path.relative_to(output_dir))] = _sha256(path)
        _json_dump(output_dir / "sha256_manifest.json", manifest)
    return report


def main() -> int:
    report = run(_arguments())
    print(json.dumps({
        "outcome": report.get("outcome"),
        "physical_action_count": report.get("physical_action_count"),
        "physical_substep_count": report.get("physical_substep_count"),
        "acceptance": report.get("acceptance"),
        "archive_dir": report.get("archive_dir"),
        "failure": report.get("failure"),
    }, indent=2, sort_keys=True))
    return 0 if report.get("outcome") == "current_safety_pass_and_t10_full_state_exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())
