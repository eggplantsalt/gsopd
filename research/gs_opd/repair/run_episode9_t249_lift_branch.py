# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Run one bounded, isolated OSC lift branch from Episode 9 t=249."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from research.gs_opd.repair.automatic_recovery import RecoveryArtifactWriter
from research.gs_opd.repair.automatic_recovery_runtime import (
    AutomaticRecoveryBridgeBackend,
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
)
from research.gs_opd.repair.mujoco_jointspace_planning import (
    validate_finger_only_contact_geoms,
)
from research.gs_opd.repair.trajectory_tracking import world_pose_tracking_action
from research.gs_opd.smoke.joint_limit_policy import (
    acceptance_results,
    contact_safety_pass,
    normalize_joint_limit_tolerance_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "task_success_first_safety_reconstruction_l0t2_20261023_attempt1"
)
EPISODE_DIR = ARCHIVE_ROOT / "episode9"
DEFAULT_CONFIG = REPO_ROOT / "research/gs_opd/configs/l0t2_task_success_first_safety_reconstruction.json"
DEFAULT_SNAPSHOT = EPISODE_DIR / "snapshots/main_timestep_000249.pkl"
DEFAULT_OUTPUT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_round0_t249_operational_lift_20261024"
)
EXPECTED_MODEL_SHA256 = "1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752"
EXPECTED_STATE_SHA256 = "ff38bea1ff53331bbf97d4f8f70fe6301ab5e81635b7187ee70ffbc2bc258094"
TARGET_OBJECT = "moka_pot_2"
EXPECTED_SUBSTEPS = 25
MAX_ACTIONS = 80
LIFT_INCREMENT_M = 0.0025
SUCCESS_LIFT_M = 0.030
SAFE_EXIT_LIFT_M = 0.040
JOINT_LIMIT_TOLERANCE_POLICY = normalize_joint_limit_tolerance_policy({
    "by_joint_type": {
        "hinge": {
            "strict_tolerance": 1e-8,
            "operational_tolerance": 1e-8,
            "unit": "rad",
        },
        "slide": {
            "strict_tolerance": 1e-8,
            "operational_tolerance": 1e-8,
            "unit": "m",
        },
    },
    "by_joint_name": {
        name: {
            "operational_tolerance": 8e-6,
            "unit": "m",
            "calibration_id": (
                "episode9_observation_action_trace_239_endpoints_"
                "max_excursion_7.374um_20261024"
            ),
        }
        for name in ("gripper0_finger_joint1", "gripper0_finger_joint2")
    },
})


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "bytes_length": len(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _state_hash(data: dict[str, Any]) -> str:
    state = np.asarray(data["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
    return hashlib.sha256(state.tobytes()).hexdigest()


def _camera_frame(data: dict[str, Any], key: str) -> np.ndarray:
    image = data.get("images", {}).get(key)
    if not isinstance(image, dict):
        raise ValueError(f"required_camera_observation_missing:{key}")
    return np.frombuffer(image["data"], dtype=np.dtype(image["dtype"])).reshape(image["shape"]).copy()


def _target_binding(backend) -> dict[str, Any]:
    binding = backend.bindings[TARGET_OBJECT]
    return {
        "object_id": TARGET_OBJECT,
        "body_id": int(binding["root_body_id"]),
        "body_name": str(binding["root_body_name"]),
        "geom_ids": [int(value) for value in binding["root_geom_ids"]],
        "geom_names": [str(value) for value in binding["root_geom_names"]],
    }


def _grasp_state(bridge, backend) -> dict[str, Any]:
    return bridge.request({
        "op": "grasp_state",
        "expected_instruction": backend.context_static["instruction"],
        "carried_binding": _target_binding(backend),
    })


def _finger_only_contact_audit(contacts: dict[str, Any], finger_ids: set[int], binding: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    for contact in contacts.get("contacts", []):
        body1, body2 = contact.get("body1", {}), contact.get("body2", {})
        if not (
            str(body1.get("name", "")).startswith(("robot0_", "gripper0_"))
            or str(body2.get("name", "")).startswith(("robot0_", "gripper0_"))
        ):
            continue
        robot_is_first = str(body1.get("name", "")).startswith(("robot0_", "gripper0_"))
        robot_geom, scene_geom = (
            (contact.get("geom1", {}), contact.get("geom2", {}))
            if robot_is_first
            else (contact.get("geom2", {}), contact.get("geom1", {}))
        )
        scene_body = body2 if robot_is_first else body1
        pairs.append({
            "robot_geom": robot_geom,
            "scene_geom": scene_geom,
            "scene_body": scene_body,
            "signed_contact_distance_m": contact.get("signed_contact_distance_m"),
        })
    valid = bool(pairs) and all(
        int(pair["robot_geom"].get("id", -1)) in finger_ids
        and int(pair["scene_body"].get("id", -1)) == int(binding["body_id"])
        and int(pair["scene_geom"].get("id", -1)) in set(binding["geom_ids"])
        for pair in pairs
    )
    return {
        "status": "pass" if valid else "fail",
        "finger_geom_ids": sorted(finger_ids),
        "target_binding": binding,
        "robot_contact_pairs": pairs,
        "all_current_robot_contacts_are_whitelisted_finger_to_pot2": valid,
    }


def _clearances_by_object(geometry: dict[str, Any]) -> dict[str, float]:
    result = {}
    for object_id, detail in geometry.get("clearances_by_object", {}).items():
        value = detail.get("minimum_distance_m") if isinstance(detail, dict) else None
        if value is not None:
            result[str(object_id)] = float(value)
    carried_object = geometry.get("carried_object_id", TARGET_OBJECT)
    for object_id, detail in geometry.get("carried_clearances_by_object", {}).items():
        value = detail.get("minimum_distance_m") if isinstance(detail, dict) else None
        if value is not None:
            result[f"carried:{carried_object}->{object_id}"] = float(value)
    return result


def _calibration_from_episode9(joint_inventory: dict[str, Any]) -> dict[str, Any]:
    """Calibrate gripper slide envelopes from the archived normal control trace."""
    trace_path = EPISODE_DIR / "observation_action_trace.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        gripper_qpos = np.asarray(trace["post_robot0_gripper_qpos"], dtype=np.float64)
        timesteps = np.asarray(trace["step_timesteps"], dtype=np.int64)
    joints = {
        item["joint_name"]: item
        for item in joint_inventory.get("joints", [])
        if item.get("joint_name") in {
            "gripper0_finger_joint1",
            "gripper0_finger_joint2",
        }
    }
    result = {
        "source_trace": str(trace_path),
        "source_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "observed_control_endpoints": int(len(timesteps)),
        "timestep_range": [int(timesteps[0]), int(timesteps[-1])],
        "sampling_scope": "archived_post_control_qpos_only;historical_substep_qpos_not_logged",
        "per_joint": {},
    }
    if gripper_qpos.shape != (len(timesteps), 2) or set(joints) != {
        "gripper0_finger_joint1",
        "gripper0_finger_joint2",
    }:
        raise ValueError("episode9_gripper_limit_calibration_source_is_incomplete")
    for column, name in enumerate(("gripper0_finger_joint1", "gripper0_finger_joint2")):
        joint = joints[name]
        lower, upper = float(joint["lower"]), float(joint["upper"])
        values = gripper_qpos[:, column]
        maximum_excursion = max(
            0.0,
            lower - float(values.min()),
            float(values.max()) - upper,
        )
        tolerance = float(
            JOINT_LIMIT_TOLERANCE_POLICY["by_joint_name"][name]["operational_tolerance"]
        )
        result["per_joint"][name] = {
            "joint_type": joint["joint_type"],
            "unit": joint["unit"],
            "model_range": [lower, upper],
            "recorded_min_qpos": float(values.min()),
            "recorded_max_qpos": float(values.max()),
            "maximum_recorded_range_excursion": maximum_excursion,
            "operational_tolerance": tolerance,
            "tolerance_headroom_over_recorded_maximum": tolerance - maximum_excursion,
            "historical_endpoint_samples_outside_range": int(
                np.count_nonzero((values < lower) | (values > upper))
            ),
            "calibration_pass": tolerance >= maximum_excursion,
        }
    if not all(item["calibration_pass"] for item in result["per_joint"].values()):
        raise ValueError("configured_gripper_slide_tolerance_does_not_cover_episode9_recorded_excursion")
    t236_path = Path(
        "/opt/data/private/gs_opd/rollouts/l0t2_round0_tsfs_safe_seed_20261024/"
        "episode1_t236_gripper_limit_fix_branches_20261024/"
        "first_substep_controller_state_reconstruction.json"
    )
    if t236_path.is_file():
        t236 = json.loads(t236_path.read_text())
        result["t236_snapshot_first_substep_read_only_reference"] = {
            "joint_limit_violation": {
                **(t236.get("joint_limit_violation") or {}),
                "joint_type": "slide",
                "unit": "m",
                "previous_unit_label_was_incorrect": True,
            },
            "first_action_substep_excursions": [
                {
                    "joint_name": item.get("joint_name"),
                    "unit": "m",
                    "qpos": item.get("qpos_rad"),
                    "lower": item.get("lower_rad"),
                    "upper": item.get("upper_rad"),
                }
                for item in t236.get("joint_limit_violation_substeps", [])[:1]
            ],
            "source_kind": "read_only_previous_branch_evidence_not_replayed",
        }
    return result


def _verdicts(
    geometry: dict[str, Any],
    substeps: list[dict[str, Any]],
    clearances: dict[str, float],
    policy: dict[str, Any],
) -> dict[str, bool]:
    contact_safe = all(contact_safety_pass(item) for item in substeps)
    native = bool(
        len(substeps) == EXPECTED_SUBSTEPS
        and all(item.get("status") in {"complete", "unsafe"} for item in substeps)
        and contact_safe
    )
    operational = bool(
        geometry.get("status") == "complete"
        and len(substeps) == EXPECTED_SUBSTEPS
        and all(item.get("status") == "complete" for item in substeps)
        and contact_safe
    )
    required_objects = {
        str(item["object_id"])
        for group_name in ("forbidden_objects", "allowed_target_objects")
        for item in policy.get(group_name, [])
    }
    strict10 = bool(
        operational
        and required_objects.issubset(clearances)
        and all(value >= 0.010 for value in clearances.values())
    )
    return acceptance_results(
        native_safety=native,
        operational_safety=operational,
        strict_10mm=strict10,
    )


class LiftRecorder:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.snapshots_dir = output_dir / "snapshots"
        self.snapshots_dir.mkdir()
        self.rows: list[dict[str, Any]] = []
        self.agent_frames: list[np.ndarray] = []
        self.wrist_frames: list[np.ndarray] = []
        self.substeps_path = output_dir / "substep_safety.jsonl"
        self.events_path = output_dir / "events.jsonl"

    def add_initial(self, data: dict[str, Any], payload: bytes) -> None:
        self.agent_frames.append(_camera_frame(data, "agentview_image"))
        self.wrist_frames.append(_camera_frame(data, "robot0_eye_in_hand_image"))
        (self.snapshots_dir / "branch_start_timestep_000249.pkl").write_bytes(payload)
        (self.output_dir / "initial_observation.json").write_text(
            json.dumps(_jsonable(data), ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.event({"event": "branch_start_snapshot_saved", "timestep": 249})

    def event(self, value: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) + "\n")

    def add_action(self, action: np.ndarray, data: dict[str, Any], grasp: dict[str, Any], native: dict[str, Any], phase: str, object_state: dict[str, Any], snapshot_payload: bytes) -> dict[str, Any]:
        monitor = data.get("contact_monitoring", {})
        geometry = monitor.get("geometry_monitoring", {})
        substeps = geometry.get("substeps", [])
        if len(substeps) != EXPECTED_SUBSTEPS:
            raise RuntimeError(f"expected_25_physics_substeps_got_{len(substeps)}")
        robot = data["snapshot_state"]["robot_states"][0]
        row = {
            "action_index": len(self.rows),
            "timestep": int(data["timestep"]),
            "phase": phase,
            "action": np.asarray(action, dtype=np.float32).tolist(),
            "eef_position_world_m": robot["eef_pos"],
            "eef_quaternion_wxyz": robot["eef_quat"],
            "arm_joint_qpos": robot["joint_qpos"],
            "gripper_qpos_m": robot["gripper_qpos"],
            "pot2_root_position_world_m": object_state["root_pos"],
            "pot2_root_quaternion_wxyz": object_state["root_quat"],
            "reward": data.get("reward"),
            "done": data.get("done"),
            "task_success": data.get("task_success"),
            "native_task_status": native,
            "grasp_state": grasp,
            "gripper_current_action": grasp.get("gripper_current_action"),
            "contact_monitoring": monitor,
            "raw_contacts": data.get("contacts"),
            "runtime_snapshot_sha256": hashlib.sha256(snapshot_payload).hexdigest(),
            "runtime_snapshot_path": str(self.snapshots_dir / f"timestep_{int(data['timestep']):06d}.pkl"),
        }
        Path(row["runtime_snapshot_path"]).write_bytes(snapshot_payload)
        self.rows.append(row)
        self.agent_frames.append(_camera_frame(data, "agentview_image"))
        self.wrist_frames.append(_camera_frame(data, "robot0_eye_in_hand_image"))
        with self.substeps_path.open("a", encoding="utf-8") as stream:
            for substep_index, item in enumerate(substeps):
                stream.write(json.dumps(_jsonable({
                    "action_index": row["action_index"],
                    "physics_substep_index_zero_based": substep_index,
                    **item,
                }), ensure_ascii=False, sort_keys=True) + "\n")
        self.event({"event": "physical_action", **row})
        return row

    def save(self) -> None:
        if not self.rows:
            return
        np.savez_compressed(
            self.output_dir / "observation_action_trace.npz",
            actions=np.asarray([row["action"] for row in self.rows], dtype=np.float32),
            timesteps=np.asarray([row["timestep"] for row in self.rows], dtype=np.int32),
            eef_position_world_m=np.asarray([row["eef_position_world_m"] for row in self.rows]),
            eef_quaternion_wxyz=np.asarray([row["eef_quaternion_wxyz"] for row in self.rows]),
            arm_joint_qpos=np.asarray([row["arm_joint_qpos"] for row in self.rows]),
            gripper_qpos_m=np.asarray([row["gripper_qpos_m"] for row in self.rows]),
            pot2_root_position_world_m=np.asarray([row["pot2_root_position_world_m"] for row in self.rows]),
            pot2_root_quaternion_wxyz=np.asarray([row["pot2_root_quaternion_wxyz"] for row in self.rows]),
            rewards=np.asarray([row["reward"] for row in self.rows], dtype=np.float32),
            dones=np.asarray([row["done"] for row in self.rows], dtype=np.bool_),
            agentview_rgb=np.stack(self.agent_frames),
            wrist_rgb=np.stack(self.wrist_frames),
            post_contact_monitoring_json=np.asarray([
                json.dumps(_jsonable(row["contact_monitoring"]), sort_keys=True)
                for row in self.rows
            ]),
            post_contacts_json=np.asarray([
                json.dumps(_jsonable(row["raw_contacts"]), sort_keys=True)
                for row in self.rows
            ]),
            native_task_status_json=np.asarray([
                json.dumps(_jsonable(row["native_task_status"]), sort_keys=True)
                for row in self.rows
            ]),
            grasp_state_json=np.asarray([
                json.dumps(_jsonable(row["grasp_state"]), sort_keys=True)
                for row in self.rows
            ]),
        )
        for frames, filename in ((self.agent_frames, "agentview.mp4"), (self.wrist_frames, "wrist.mp4")):
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(self.output_dir / filename), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"video_writer_failed:{filename}")
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        (self.output_dir / "steps.json").write_text(
            json.dumps(_jsonable(self.rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _make_policy(backend) -> dict[str, Any]:
    policy = json.loads(json.dumps(backend.safety_policy))
    binding = backend.bindings[TARGET_OBJECT]
    finger_ids = sorted(int(value) for value in backend.context_static["gripper_geometry_ids"])
    inventory = backend.bridge.request({"op": "gripper_geometry_inventory"})
    geoms = inventory["grippers"][0]["geometries"]
    left, right = set(), set()
    for item in geoms:
        name, geom_id = str(item.get("geom_name", "")), int(item["geom_id"])
        if name.endswith(("finger1_collision", "finger1_pad_collision")):
            left.add(geom_id)
        elif name.endswith(("finger2_collision", "finger2_pad_collision")):
            right.add(geom_id)
    if validate_finger_only_contact_geoms(sorted(left), sorted(right), finger_ids) != set(finger_ids) or len(finger_ids) != 4:
        raise ValueError("finger_only_allowlist_does_not_equal_four_verified_geoms")
    for group in policy["forbidden_objects"]:
        group["minimum_clearance_m"] = 0.010
    for group in policy["allowed_target_objects"]:
        group["minimum_clearance_m"] = 0.002 if group["object_id"] == TARGET_OBJECT else 0.010
    policy["allowed_contact_pairs"] = [
        {"robot_geom_id": finger, "scene_geom_id": int(scene), "minimum_clearance_m": -0.002}
        for finger in finger_ids
        for scene in binding["geom_ids"]
    ]
    policy["minimum_clearance_m"] = 0.010
    policy["joint_limit_tolerance_policy"] = JOINT_LIMIT_TOLERANCE_POLICY
    policy["source"] = (
        "active_BDDL_full_geometry_finger_only_t249_lift;"
        "calibrated_gripper_slide_operational_tolerance"
    )
    backend._refresh_carried_geometry_policy(TARGET_OBJECT)
    policy["carried_object"] = backend.safety_policy["carried_object"]
    if any(
        "kitchen_table" in pair
        for pair in backend.context_static.get("forbidden_pairs", [])
    ) or "kitchen_table" in backend.context_static.get("forbidden_names", []):
        raise ValueError("table_support_contact_is_declared_as_BDDL_forbidden")
    inventory = backend.bridge.request({"op": "geometry_inventory"})
    table_matches = [
        int(item["geom_id"])
        for item in inventory.get("geometries", [])
        if item.get("geom_name") == "table_collision"
        and item.get("collision_enabled") is True
    ]
    if inventory.get("status") != "complete" or len(table_matches) != 1:
        raise ValueError("unique_active_table_collision_geom_is_unavailable")
    table_geom = table_matches[0]
    target_geoms = [int(value) for value in binding["geom_ids"]]
    policy["allowed_carried_support_contact_geom_ids"] = [table_geom]
    policy["carried_object"]["allowed_support_contact_pairs"] = [
        {
            "carried_geom_id": geom_id,
            "scene_geom_id": table_geom,
            "minimum_clearance_m": -0.002,
        }
        for geom_id in target_geoms
    ]
    policy["carried_object"]["support_contact_scope"] = {
        "scene_geom_id": table_geom,
        "scene_geom_name": "table_collision",
        "maximum_penetration_m": 0.002,
        "source": "observed_t249_Pot2_table_support_pair;not_named_in_active_BDDL_constraints",
    }
    return policy


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_data, recovery = load_runtime_config(args.config)
    source_payload = args.snapshot.read_bytes()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "gs_opd_l0t2_episode9_t249_local_lift.v1",
        "source_snapshot": str(args.snapshot),
        "source_snapshot_sha256": hashlib.sha256(source_payload).hexdigest(),
        "expected_source_model_geometry_sha256": EXPECTED_MODEL_SHA256,
        "expected_integration_state_sha256": EXPECTED_STATE_SHA256,
        "maximum_control_actions": min(MAX_ACTIONS, int(args.maximum_actions)),
        "physical_actions": 0,
        "physical_substeps": 0,
        "native_reset_operations": 0,
        "task_phase_controller_state_used": False,
        "complete_task_episode": False,
        "round0_seed_candidate": False,
        "outcome": None,
        "calibrated_joint_limit_tolerance_policy": JOINT_LIMIT_TOLERANCE_POLICY,
    }
    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend = None
    recorder = LiftRecorder(output_dir)
    started = time.monotonic()
    try:
        bridge.start(
            benchmark=config_data["benchmark"],
            level=int(config_data["level"]),
            task_id=int(config_data["task_id"]),
            init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=1800,
        )
        reset = bridge.request({"op": "reset"})
        report["native_reset_operations"] = 1
        report["native_reset_model_geometry_sha256"] = reset.get("scene_metadata", {}).get("model_geometry_sha256")
        report["native_reset_timestep"] = reset.get("timestep")
        if reset.get("timestep") != 0 or report["native_reset_model_geometry_sha256"] != EXPECTED_MODEL_SHA256:
            raise RuntimeError("native_reset_model_or_timestep_mismatch_before_t249_restore")
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": source_payload})
        if restored.get("status") != "restored":
            raise RuntimeError(f"strict_t249_snapshot_restore_failed:{restored.get('error')}")
        current = bridge.request({"op": "observe", "include_snapshot_state": True})
        if current.get("timestep") != 249 or _state_hash(current) != EXPECTED_STATE_SHA256:
            raise RuntimeError("restored_t249_integration_state_identity_mismatch")
        if current.get("scene_metadata", {}).get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError("restored_t249_model_geometry_fingerprint_mismatch")

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
            runtime_snapshot_path=Path(config_data["artifacts"]["t10_runtime_snapshot"]),
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
        )
        backend.latest_data = current
        backend._load_context_inputs(current)
        policy = _make_policy(backend)
        report["policy"] = policy
        report["finger_geom_ids"] = sorted(int(value) for value in backend.context_static["gripper_geometry_ids"])
        report["compiled_fingerprint_details"] = bridge.request({"op": "compiled_model_fingerprint_details"})
        report["joint_limit_inventory"] = bridge.request({
            "op": "robot_joint_limit_inventory",
            "joint_limit_tolerance_policy": JOINT_LIMIT_TOLERANCE_POLICY,
        })
        report["gripper_soft_limit_calibration"] = _calibration_from_episode9(
            report["joint_limit_inventory"]
        )
        report["safety_policy_geometry_preflight"] = bridge.request({
            "op": "observe",
            "include_snapshot_state": True,
            "safety_geometry_policy": policy,
        }).get("safety_geometry")
        contacts = bridge.request({"op": "inspect_contacts"})
        binding = _target_binding(backend)
        grasp = _grasp_state(bridge, backend)
        report["current_finger_only_contact_audit"] = _finger_only_contact_audit(
            contacts, set(report["finger_geom_ids"]), binding
        )
        report["live_grasp_state"] = grasp
        geometry = report["safety_policy_geometry_preflight"] or {}
        preflight_clearances = _clearances_by_object(geometry)
        report["preflight_minimum_clearance_by_object_m"] = preflight_clearances
        initial_joints = report["joint_limit_inventory"].get("joints", [])
        preflight_operational = bool(
            geometry.get("status") == "complete"
            and report["current_finger_only_contact_audit"]["status"] == "pass"
            and grasp.get("status") == "pass"
            and grasp.get("grasp_confirmed") is True
            and all(item.get("operational_pass") for item in initial_joints)
        )
        contacts_in_geometry = geometry.get("contact_identities", [])
        native_contact_evidence = {
            "forbidden_contacts": geometry.get("forbidden_contacts", []),
            "self_contacts": geometry.get("self_contacts", []),
            "unresolved_contacts": geometry.get("unresolved_contacts", []),
            "forbidden_object_contacts": geometry.get("forbidden_object_contacts", []),
            "unallowed_scene_contacts": geometry.get("unallowed_scene_contacts", []),
        }
        no_initial_native_contact_violation = bool(
            contact_safety_pass(native_contact_evidence)
            and report["current_finger_only_contact_audit"]["status"] == "pass"
        )
        report["preflight_contact_identities"] = contacts_in_geometry
        required_objects = {
            str(item["object_id"])
            for group_name in ("forbidden_objects", "allowed_target_objects")
            for item in policy.get(group_name, [])
        }
        operational_margin_pass = bool(
            {str(item["object_id"]): float(item["minimum_clearance_m"])
             for group_name in ("forbidden_objects", "allowed_target_objects")
             for item in policy.get(group_name, [])}.keys() <= preflight_clearances.keys()
            and all(
                preflight_clearances[str(item["object_id"])] >= float(item["minimum_clearance_m"])
                for group_name in ("forbidden_objects", "allowed_target_objects")
                for item in policy.get(group_name, [])
            )
        )
        preflight_operational = preflight_operational and operational_margin_pass and no_initial_native_contact_violation
        report["preflight_verdicts"] = acceptance_results(
            native_safety=no_initial_native_contact_violation,
            operational_safety=preflight_operational,
            strict_10mm=bool(
                preflight_operational
                and required_objects.issubset(preflight_clearances)
                and all(value >= 0.010 for value in preflight_clearances.values())
            ),
        )
        controller_fields = bridge.request({"op": "repair_context"})["robots"][0]["fields"]
        report["controller_mode"] = "local_snapshot_anchored_vertical_lift_only;TaskPhaseController_not_used"
        report["gripper_hold_strategy"] = {
            "command": 0.0,
            "source": "PandaGripper.format_action uses sign(action); zero preserves current_action",
            "restored_current_action": grasp.get("gripper_current_action"),
            "restored_closed": grasp.get("gripper_closed"),
        }
        start_object = backend._live_object_state(current, TARGET_OBJECT)
        start_z = float(start_object["root_pos"][2])
        start_robot = current["snapshot_state"]["robot_states"][0]
        start_eef = np.asarray(start_robot["eef_pos"], dtype=np.float64)
        start_pot = np.asarray(start_object["root_pos"], dtype=np.float64)
        start_snapshot = bridge.request({"op": "export_runtime_snapshot"})
        if start_snapshot.get("status") != "complete" or not isinstance(start_snapshot.get("payload"), bytes):
            raise RuntimeError("restored_t249_runtime_snapshot_reexport_failed")
        reexport_payload = start_snapshot["payload"]
        (recorder.snapshots_dir / "source_episode9_timestep_000249.pkl").write_bytes(source_payload)
        (recorder.snapshots_dir / "restored_runtime_reexport_timestep_000249.pkl").write_bytes(reexport_payload)
        report["snapshot_restore_roundtrip"] = {
            "restore_status": restored.get("status"),
            "integration_state_exactly_restored": _state_hash(current) == EXPECTED_STATE_SHA256,
            "model_geometry_fingerprint_matches": current.get("scene_metadata", {}).get("model_geometry_sha256") == EXPECTED_MODEL_SHA256,
            "restored_timestep_matches": current.get("timestep") == 249,
            "source_payload_sha256": hashlib.sha256(source_payload).hexdigest(),
            "reexport_payload_sha256": hashlib.sha256(reexport_payload).hexdigest(),
            "payload_bytes_identical": reexport_payload == source_payload,
            "byte_identity_required_for_state_acceptance": False,
            "precedent": "read_only_episode9_t249_finger_only_audit_also_found_exact_integration_restore_with_nonidentical_reexport_bytes",
        }
        recorder.add_initial(current, source_payload)
        report["restore_validation"] = {
            "status": "pass",
            "timestep": 249,
            "integration_state_sha256": _state_hash(current),
            "model_geometry_sha256": EXPECTED_MODEL_SHA256,
            "snapshot_reexport_sha256": hashlib.sha256(reexport_payload).hexdigest(),
            "snapshot_reexport_byte_identical": reexport_payload == source_payload,
            "robot_joint_qpos": start_robot["joint_qpos"],
            "gripper_qpos_m": start_robot["gripper_qpos"],
            "pot2_root_position_world_m": start_object["root_pos"],
            "pot2_root_quaternion_wxyz": start_object["root_quat"],
            "pot2_initial_eef_relative_position_m": (start_eef - start_pot).tolist(),
            "native_task_status": bridge.request({"op": "native_task_status"}),
            "controller_state_recorded_for_audit_only": bridge.request({"op": "repair_context"}),
        }
        if not preflight_operational:
            report["outcome"] = "stopped_before_action_start_snapshot_failed_frozen_operational_safety"
            report["physical_branch_started"] = False
            return report

        report["physical_branch_started"] = True
        report["lift_increment_m"] = LIFT_INCREMENT_M
        report["target_object_vertical_rise_m"] = {"success": SUCCESS_LIFT_M, "safe_exit": SAFE_EXIT_LIFT_M}
        min_clearances: dict[str, float] = {}
        current = bridge.request({
            "op": "observe",
            "include_snapshot_state": True,
            "safety_geometry_policy": policy,
        })
        last_pot_z = start_z
        stalled = 0
        for action_index in range(min(MAX_ACTIONS, int(args.maximum_actions))):
            robot = current["snapshot_state"]["robot_states"][0]
            object_now = backend._live_object_state(current, TARGET_OBJECT)
            rise = float(object_now["root_pos"][2]) - start_z
            phase = "lift" if rise < SUCCESS_LIFT_M else "safe_exit"
            increment = min(LIFT_INCREMENT_M, max(0.0, SAFE_EXIT_LIFT_M - rise))
            if increment <= 1e-6:
                report["safe_exit"] = True
                break
            eef_now = np.asarray(robot["eef_pos"], dtype=np.float64)
            eef_quat = np.asarray(robot["eef_quat"], dtype=np.float64)
            target_eef = eef_now + np.asarray([0.0, 0.0, increment], dtype=np.float64)
            action = world_pose_tracking_action(
                eef_now,
                eef_quat,
                target_eef,
                eef_quat,
                controller_fields,
                gripper_action=0.0,
            )
            reply = bridge.request({
                "op": "step_monitored",
                "action": action.tolist(),
                "action_index": action_index,
                "safety_geometry_policy": policy,
            })
            grasp_now = _grasp_state(bridge, backend)
            native_now = bridge.request({"op": "native_task_status"})
            object_post = backend._live_object_state(reply, TARGET_OBJECT)
            snapshot_post = bridge.request({"op": "export_runtime_snapshot"})
            if snapshot_post.get("status") != "complete":
                raise RuntimeError("post_action_runtime_snapshot_export_failed")
            row = recorder.add_action(
                action, reply, grasp_now, native_now, phase, object_post, snapshot_post["payload"]
            )
            report["physical_actions"] += 1
            report["physical_substeps"] += EXPECTED_SUBSTEPS
            substeps = reply.get("contact_monitoring", {}).get("geometry_monitoring", {}).get("substeps", [])
            geometry_monitor = reply.get("contact_monitoring", {}).get("geometry_monitoring", {})
            action_clearances: dict[str, float] = {}
            for item in substeps:
                all_clearance_rows = dict(item.get("clearances_by_object", {}))
                for object_id, detail in item.get("carried_clearances_by_object", {}).items():
                    all_clearance_rows[f"carried:{TARGET_OBJECT}->{object_id}"] = detail
                for object_id, detail in all_clearance_rows.items():
                    distance = detail.get("minimum_distance_m") if isinstance(detail, dict) else None
                    if distance is not None:
                        min_clearances[object_id] = min(float(distance), min_clearances.get(object_id, math.inf))
                        action_clearances[object_id] = min(
                            float(distance), action_clearances.get(object_id, math.inf)
                        )
            verdicts = _verdicts(
                geometry_monitor,
                substeps,
                action_clearances,
                policy,
            )
            row["acceptance_verdicts"] = verdicts
            row["pot2_vertical_rise_m"] = float(object_post["root_pos"][2]) - start_z
            row["gripper_hold_current_action_unchanged"] = (
                grasp_now.get("gripper_current_action") == grasp.get("gripper_current_action")
            )
            if (
                reply.get("contact_monitoring", {}).get("status") != "complete"
                or len(substeps) != EXPECTED_SUBSTEPS
                or geometry_monitor.get("status") != "complete"
                or verdicts["operational_safety"] is not True
            ):
                report["failure_evidence"] = {
                    "phase": phase,
                    "action_index": action_index,
                    "timestep": reply.get("timestep"),
                    "contact_monitoring_status": reply.get("contact_monitoring", {}).get("status"),
                    "geometry_monitoring_status": geometry_monitor.get("status"),
                    "failed_substeps": [
                        {"index": i, **item} for i, item in enumerate(substeps)
                        if item.get("status") != "complete"
                    ],
                    "acceptance_verdicts": verdicts,
                }
                report["outcome"] = "fail_closed_on_physical_substep_or_operational_safety_violation"
                current = reply
                break
            if grasp_now.get("status") != "pass" or grasp_now.get("grasp_confirmed") is not True:
                report["outcome"] = "physical_grasp_lost_during_lift"
                report["failure_phase"] = phase
                current = reply
                break
            if not row["gripper_hold_current_action_unchanged"]:
                report["outcome"] = "gripper_internal_hold_state_changed_unexpectedly"
                report["failure_phase"] = phase
                current = reply
                break
            current = reply
            new_z = float(object_post["root_pos"][2])
            if new_z - last_pot_z < 0.00015:
                stalled += 1
            else:
                stalled = 0
                last_pot_z = new_z
            if stalled >= 6:
                report["outcome"] = "local_lift_controller_stalled_6_actions"
                break
            if row["pot2_vertical_rise_m"] >= SUCCESS_LIFT_M and "lift_30mm_event" not in report:
                report["lift_30mm_event"] = {
                    "timestep": row["timestep"],
                    "pot2_start_position_world_m": start_pot.tolist(),
                    "pot2_position_world_m": object_post["root_pos"],
                    "measured_vertical_rise_m": row["pot2_vertical_rise_m"],
                }
                recorder.event({"event": "pot2_real_vertical_lift_at_least_30mm", **report["lift_30mm_event"]})
                table_contacts = [
                    contact
                    for substep in substeps
                    for contact in substep.get("contact_identities", [])
                    if {
                        contact.get("body1", {}).get("name"),
                        contact.get("body2", {}).get("name"),
                    } == {"table", "moka_pot_2_main"}
                ]
                if not table_contacts:
                    report["safe_exit"] = True
                    report["safe_exit_evidence"] = {
                        "timestep": row["timestep"],
                        "measured_vertical_rise_m": row["pot2_vertical_rise_m"],
                        "post_lift_control_substeps": len(substeps),
                        "pot2_table_contacts_in_threshold_action": 0,
                        "criterion": "30mm_real_lift_and_support_contact_cleared_for_all_25_substeps",
                    }
                    recorder.event({
                        "event": "pot2_contact_free_vertical_safe_exit_complete",
                        **report["safe_exit_evidence"],
                    })
                    break
            if row["pot2_vertical_rise_m"] >= SAFE_EXIT_LIFT_M:
                report["extra_vertical_clearance_40mm_reached"] = True
                recorder.event({
                    "event": "extra_vertical_clearance_40mm_reached",
                    "timestep": row["timestep"],
                    "measured_vertical_rise_m": row["pot2_vertical_rise_m"],
                })
                break
        report["minimum_clearance_by_object_m"] = min_clearances
        report["native_task_status_terminal"] = bridge.request({"op": "native_task_status"})
        report["terminal_grasp_state"] = _grasp_state(bridge, backend)
        report["terminal_timestep"] = current.get("timestep")
        report["terminal_pot2_root_position_world_m"] = backend._live_object_state(current, TARGET_OBJECT)["root_pos"]
        report["measured_pot2_vertical_rise_m"] = (
            float(report["terminal_pot2_root_position_world_m"][2]) - start_z
        )
        report["action_count_with_25_substep_records"] = (
            len(recorder.rows) == report["physical_actions"]
            and all(len(row["contact_monitoring"].get("geometry_monitoring", {}).get("substeps", [])) == EXPECTED_SUBSTEPS for row in recorder.rows)
        )
        history_verdicts = [row.get("acceptance_verdicts", {}) for row in recorder.rows]
        report["acceptance_verdicts"] = acceptance_results(
            native_safety=bool(history_verdicts) and all(row.get("native_safety") is True for row in history_verdicts),
            operational_safety=bool(history_verdicts) and all(row.get("operational_safety") is True for row in history_verdicts),
            strict_10mm=bool(history_verdicts) and all(row.get("strict_10mm") is True for row in history_verdicts),
        )
        report["native_task_success"] = report["native_task_status_terminal"].get("task_success") is True
        report["local_lift_30mm_success"] = report.get("measured_pot2_vertical_rise_m", 0.0) >= SUCCESS_LIFT_M
        report["local_safe_exit_success"] = report.get("safe_exit") is True
        report["extra_vertical_clearance_40mm_success"] = (
            report.get("measured_pot2_vertical_rise_m", 0.0) >= SAFE_EXIT_LIFT_M
        )
        if report["outcome"] is None:
            report["outcome"] = (
                "t249_local_lift_30mm_and_contact_free_safe_exit_success"
                if report["local_lift_30mm_success"] and report["local_safe_exit_success"]
                else "action_budget_exhausted_before_lift_and_safe_exit"
            )
        report["wall_time_seconds"] = time.monotonic() - started
        return report
    except Exception as exc:
        report["outcome"] = "preflight_or_runtime_failure"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["wall_time_seconds"] = time.monotonic() - started
        return report
    finally:
        try:
            recorder.save()
        except Exception as exc:
            report["recording_save_error"] = f"{type(exc).__name__}: {exc}"
        report["physical_actions"] = len(recorder.rows)
        report["physical_substeps"] = sum(
            len(row.get("contact_monitoring", {}).get("geometry_monitoring", {}).get("substeps", []))
            for row in recorder.rows
        )
        report["data_paths"] = {
            "report": str(output_dir / "branch_report.json"),
            "trace_npz": str(output_dir / "observation_action_trace.npz"),
            "steps_json": str(output_dir / "steps.json"),
            "substep_safety_jsonl": str(output_dir / "substep_safety.jsonl"),
            "events_jsonl": str(output_dir / "events.jsonl"),
            "agentview_video": str(output_dir / "agentview.mp4"),
            "wrist_video": str(output_dir / "wrist.mp4"),
        }
        try:
            bridge.close()
        finally:
            (output_dir / "branch_report.json").write_text(
                json.dumps(_jsonable(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38784)
    parser.add_argument("--maximum-actions", type=int, default=MAX_ACTIONS)
    return parser.parse_args()


def main() -> int:
    report = run(_arguments())
    print(json.dumps(_jsonable(report), ensure_ascii=False, sort_keys=True))
    return 0 if report.get("outcome") == "t249_local_lift_30mm_and_contact_free_safe_exit_success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
