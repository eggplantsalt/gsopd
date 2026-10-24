# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Resume the archived L0T2 Pot 2 transport route from its t=359 Snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import signal
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from research.gs_opd.repair.automatic_recovery import (
    AutomaticSafetyRecoveryEngine,
    JointSpacePlan,
    RecoveryArtifactWriter,
    RiskBoundaryState,
)
from research.gs_opd.repair.automatic_recovery_runtime import (
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
    sha256_file,
)
from research.gs_opd.repair.run_t329_native_task_suffix import (
    ARCHIVED_ROUTE_DIR,
    CONFIG_PATH,
    EXPECTED_SUBSTEPS,
    MODEL_SHA256,
    OBJECT_ID,
    SOURCE_REPORT_PATH,
    ArchivedRouteTaskSuffixBackend,
    _all_robot_contacts_finger_target,
    _canonical_sha256,
    _jsonable,
    _load_archived_route,
    _suffix_acceptance,
    _write_json,
)


PREFIX_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_round0_t329_native_task_suffix_20261024_205500"
)
RESUME_BUNDLE = PREFIX_ARCHIVE / "route_resume_bundle_timestep_000359.npz"
RESUME_SNAPSHOT = PREFIX_ARCHIVE / "snapshots/main_timestep_000359.pkl"
RESUME_AUDIT = PREFIX_ARCHIVE / "route_reuse_prefix_audit.json"
MAX_RESUME_ACTIONS = 450
MAX_TASK_TIMESTEP = 1800
STANDARD_HORIZON = 1000
RESUME_TIMESTEP = 359
RESUME_PATH_INDEX = 30
LEFT_FINGER_GEOMS = {76, 77}
RIGHT_FINGER_GEOMS = {79, 80}


class _PythonObjectState:
    """Snapshot pickle helper class emitted by safety_bridge.py as __main__."""

    def __init__(self, module=None, qualname=None, attributes=None):
        self.module = module
        self.qualname = qualname
        self.attributes = attributes or {}


def _canonical_array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _normalize_snapshot_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _normalize_snapshot_value(value.item())
    if isinstance(value, _PythonObjectState):
        return {
            "kind": "python_object_state",
            "module": value.module,
            "qualname": value.qualname,
            "attributes": _normalize_snapshot_value(value.attributes),
        }
    if isinstance(value, dict):
        rows = [
            (str(key), _normalize_snapshot_value(item))
            for key, item in value.items()
        ]
        return {key: item for key, item in sorted(rows, key=lambda row: row[0])}
    if isinstance(value, (tuple, list)):
        return [_normalize_snapshot_value(item) for item in value]
    if isinstance(value, bytes):
        return {"kind": "bytes", "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"kind": "float", "hex": value.hex()}
        return {"kind": "float", "hex": value.hex()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return {"kind": "repr", "value": repr(value)}


def _normalized_state_hash(value: Any) -> str:
    payload = json.dumps(
        _normalize_snapshot_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quat_error_rad(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("resume quaternion comparison requires two WXYZ quaternions")
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(2.0 * math.acos(float(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def _load_resume_sources(
    prefix_archive: Path,
    route_archive: Path,
    bundle_path: Path,
    snapshot_path: Path,
) -> dict[str, Any]:
    audit = json.loads((prefix_archive / "route_reuse_prefix_audit.json").read_text())
    bundle_file_sha = sha256_file(bundle_path)
    snapshot_file_sha = sha256_file(snapshot_path)
    if bundle_file_sha != audit["continuation_handoff"]["resume_bundle_sha256"]:
        raise ValueError("t359_resume_bundle_sha256_mismatch")
    if snapshot_file_sha != audit["continuation_handoff"]["resume_snapshot_sha256"]:
        raise ValueError("t359_runtime_snapshot_sha256_mismatch")
    if audit["continuation_handoff"].get("resume_bundle_restore_preverified") is not False:
        raise ValueError("resume_bundle_metadata_has_unexpected_prior_restore_claim")

    route = _load_archived_route(route_archive)
    archived_events = [
        json.loads(line)
        for line in Path(route["events_path"]).read_text(encoding="utf-8").splitlines()
    ]
    task_goal_events = [
        event for event in archived_events
        if event.get("event") == "task_phase_rrt_route_requested"
        and event.get("phase") == "transport"
    ]
    if len(task_goal_events) != 1 or not isinstance(task_goal_events[0].get("goal"), dict):
        raise ValueError("archived_transport_goal_source_event_is_not_unique_or_complete")
    route["task_goal_event"] = task_goal_events[0]
    archived_goal = task_goal_events[0]["goal"]
    archived_goal_position_error = float(np.linalg.norm(
        route["eef_reference_poses_world"][-1, :3]
        - np.asarray(archived_goal["position_world_m"], dtype=np.float64)
    ))
    archived_goal_orientation_error = _quat_error_rad(
        route["eef_reference_poses_world"][-1, 3:], archived_goal["quaternion_wxyz"]
    )
    if archived_goal_position_error > 1e-5 or archived_goal_orientation_error > 1e-5:
        raise ValueError("archived_route_endpoint_does_not_match_its_transport_goal_event")
    route_file = np.load(prefix_archive / "reused_archived_route.npz", allow_pickle=False)
    qpos_path = np.asarray(route_file["qpos_path"], dtype=np.float64)
    eef_path = np.asarray(route_file["eef_reference_poses_world"], dtype=np.float64)
    if not np.array_equal(qpos_path, route["qpos_path"]):
        raise ValueError("reused_route_qpos_does_not_match_archived_planner_event")
    if not np.array_equal(eef_path, route["eef_reference_poses_world"]):
        raise ValueError("reused_route_EEF_path_does_not_match_archived_planner_event")
    if _canonical_array_hash(qpos_path) != route["qpos_path_sha256"]:
        raise ValueError("reused_route_qpos_hash_mismatch")
    if _canonical_array_hash(eef_path) != route["eef_reference_path_sha256"]:
        raise ValueError("reused_route_EEF_hash_mismatch")
    if route_file["metadata_json"].shape != ():
        raise ValueError("reused_route_metadata_is_not_scalar_json")
    route_metadata = json.loads(str(route_file["metadata_json"]))
    if (
        route_metadata.get("source_start_state_sha256")
        != route["planner_result"].get("start_state_sha256")
        or route_metadata.get("model_geometry_sha256") != MODEL_SHA256
        or route_metadata.get("safety_policy_sha256") != route["archived_policy_sha256"]
        or route_metadata.get("qpos_path_sha256") != route["qpos_path_sha256"]
        or route_metadata.get("eef_reference_path_sha256")
        != route["eef_reference_path_sha256"]
    ):
        raise ValueError("reused_route_provenance_metadata_mismatch")

    bundle_file = np.load(bundle_path, allow_pickle=False)
    bundle_meta = json.loads(str(bundle_file["metadata_json"]))
    resume_index = int(bundle_meta.get("next_path_start_index", -1))
    if resume_index != RESUME_PATH_INDEX:
        raise ValueError("t359_resume_path_cursor_is_not_index_30")
    if int(bundle_meta.get("snapshot_timestep", -1)) != RESUME_TIMESTEP:
        raise ValueError("t359_resume_bundle_timestep_mismatch")
    if int(bundle_meta.get("original_path_points", -1)) != len(qpos_path):
        raise ValueError("t359_resume_bundle_original_route_length_mismatch")
    if int(bundle_meta.get("remaining_path_points", -1)) != len(qpos_path) - resume_index:
        raise ValueError("t359_resume_bundle_remaining_route_length_mismatch")
    if (
        bundle_meta.get("source_qpos_path_sha256") != route["qpos_path_sha256"]
        or bundle_meta.get("source_eef_reference_path_sha256")
        != route["eef_reference_path_sha256"]
        or bundle_meta.get("source_route_start_state_sha256")
        != route["planner_result"].get("start_state_sha256")
        or bundle_meta.get("model_geometry_sha256") != MODEL_SHA256
        or bundle_meta.get("safety_policy_sha256") != route["archived_policy_sha256"]
    ):
        raise ValueError("t359_resume_bundle_route_provenance_mismatch")
    remaining_qpos = np.asarray(bundle_file["remaining_qpos_path"], dtype=np.float64)
    remaining_eef = np.asarray(
        bundle_file["remaining_eef_reference_poses_world"], dtype=np.float64
    )
    if not np.array_equal(remaining_qpos, qpos_path[resume_index:]):
        raise ValueError("t359_bundle_qpos_is_not_exact_route_suffix_from_index_30")
    if not np.array_equal(remaining_eef, eef_path[resume_index:]):
        raise ValueError("t359_bundle_EEF_path_is_not_exact_route_suffix_from_index_30")

    envelope = pickle.loads(snapshot_path.read_bytes())
    if not isinstance(envelope, dict) or envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
        raise ValueError("t359_runtime_snapshot_schema_mismatch")
    if envelope.get("model_geometry_sha256") != MODEL_SHA256:
        raise ValueError("t359_runtime_snapshot_model_fingerprint_mismatch")
    snapshot_state = envelope.get("snapshot")
    if not isinstance(snapshot_state, dict):
        raise ValueError("t359_runtime_snapshot_payload_is_incomplete")
    integration_state = np.asarray(snapshot_state.get("mujoco_state"), dtype=np.float64)
    integration_sha = _canonical_array_hash(integration_state)
    if integration_sha != bundle_meta.get("integration_state_sha256"):
        raise ValueError("t359_snapshot_integration_state_does_not_match_resume_bundle")
    if int(snapshot_state.get("task_bookkeeping", {}).get("timestep", -1)) != RESUME_TIMESTEP:
        raise ValueError("t359_snapshot_internal_timestep_mismatch")

    trace = np.load(prefix_archive / "observation_action_trace.npz", allow_pickle=False)
    if len(trace["executed_actions"]) != 30 or len(trace["step_timesteps"]) != 30:
        raise ValueError("t329_t359_action_trace_does_not_contain_exactly_30_steps")
    if not np.array_equal(trace["step_timesteps"], np.arange(330, 360, dtype=np.int64)):
        raise ValueError("t329_t359_action_trace_timesteps_are_not_contiguous")
    if trace["executed_actions"].shape != (30, 7):
        raise ValueError("t329_t359_action_trace_is_not_7d")
    if not np.allclose(trace["executed_actions"][:, 6], 0.0, rtol=0.0, atol=1e-7):
        raise ValueError("t329_t359_action_trace_did_not_neutral_hold_the_gripper")
    final_physics = json.loads(str(trace["post_physical_state_json"][-1]))
    final_robot = final_physics["robot_states"][0]
    final_object = final_physics["object_states"][f"objects_dict:{OBJECT_ID}"]
    saved_robot = snapshot_state.get("bridge_observation", {}).get("snapshot_state", {}).get(
        "robot_states", [None]
    )[0]
    # The durable action trace is canonical for the terminal observation; the
    # runtime Snapshot is independently checked below after native restore.
    if saved_robot is not None:
        for field, expected in (
            ("joint_qpos", final_robot["joint_qpos"]),
            ("eef_pos", final_robot["eef_pos"]),
            ("eef_quat", final_robot["eef_quat"]),
            ("gripper_qpos", final_robot["gripper_qpos"]),
        ):
            if not np.array_equal(
                np.asarray(saved_robot[field], dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
            ):
                raise ValueError(f"t359_trace_and_snapshot_{field}_mismatch")

    actual_qpos = np.asarray(final_robot["joint_qpos"], dtype=np.float64)
    actual_eef = np.asarray(final_robot["eef_pos"], dtype=np.float64)
    actual_eef_quat = np.asarray(final_robot["eef_quat"], dtype=np.float64)
    qpos_errors, eef_position_errors, eef_orientation_errors = [], [], []
    for action_index, state_json in enumerate(trace["post_physical_state_json"], start=1):
        robot_state = json.loads(str(state_json))["robot_states"][0]
        qpos_errors.append(float(np.linalg.norm(
            np.asarray(robot_state["joint_qpos"], dtype=np.float64) - qpos_path[action_index]
        )))
        eef_position_errors.append(float(np.linalg.norm(
            np.asarray(robot_state["eef_pos"], dtype=np.float64) - eef_path[action_index, :3]
        )))
        eef_orientation_errors.append(_quat_error_rad(
            robot_state["eef_quat"], eef_path[action_index, 3:]
        ))
    if max(qpos_errors) > 0.18 or max(eef_position_errors) > 0.025 or max(eef_orientation_errors) > 0.08:
        raise ValueError("archived_t359_route_tracking_prefix_exceeded_frozen_limits")

    safety_complete = True
    bilateral_substeps = 0
    robot_target_contact_pairs = set()
    observed_substeps = 0
    maximum_target_penetration_m = 0.0
    for action_index, raw_monitor in enumerate(trace["post_contact_monitoring_json"]):
        monitor = json.loads(str(raw_monitor))
        substeps = monitor.get("geometry_monitoring", {}).get("substeps", [])
        if monitor.get("status") != "complete" or len(substeps) != EXPECTED_SUBSTEPS:
            safety_complete = False
        for substep in substeps:
            observed_substeps += 1
            if substep.get("status") != "complete" or any(
                substep.get(key)
                for key in (
                    "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
                    "unallowed_scene_contacts", "unresolved_contacts",
                    "joint_limit_violations",
                )
            ):
                safety_complete = False
            left_contact = False
            right_contact = False
            for contact in substep.get("contact_identities", []):
                body1 = contact.get("body1", {})
                body2 = contact.get("body2", {})
                geom1 = contact.get("geom1", {})
                geom2 = contact.get("geom2", {})
                robot1 = str(body1.get("name", "")).startswith(("robot0_", "gripper0_"))
                robot2 = str(body2.get("name", "")).startswith(("robot0_", "gripper0_"))
                if not robot1 and not robot2:
                    continue
                robot_geom = geom1 if robot1 else geom2
                scene_body = body2 if robot1 else body1
                scene_geom = geom2 if robot1 else geom1
                if (
                    str(scene_body.get("name", "")).startswith(f"{OBJECT_ID}_")
                    and str(scene_geom.get("name", "")).startswith(f"{OBJECT_ID}_")
                    and int(robot_geom.get("id", -1)) in LEFT_FINGER_GEOMS | RIGHT_FINGER_GEOMS
                ):
                    robot_target_contact_pairs.add((
                        int(robot_geom["id"]), int(scene_geom["id"])
                    ))
                    left_contact |= int(robot_geom["id"]) in LEFT_FINGER_GEOMS
                    right_contact |= int(robot_geom["id"]) in RIGHT_FINGER_GEOMS
                else:
                    safety_complete = False
            bilateral_substeps += int(left_contact and right_contact)
            for distance in substep.get("allowed_contact_clearances_m", {}).values():
                penetration = max(0.0, -float(distance))
                maximum_target_penetration_m = max(
                    maximum_target_penetration_m, penetration
                )
                if penetration > 0.002:
                    safety_complete = False
    if observed_substeps != 750 or bilateral_substeps != observed_substeps or not safety_complete:
        raise ValueError("archived_t329_t359_prefix_failed_frozen_contact_or_substep_audit")
    saved_gripper_internal = snapshot_state["robots"][0].get("gripper_current_action")
    if not snapshot_state["robots"][0].get("has_gripper_current_action") or saved_gripper_internal is None:
        raise ValueError("t359_snapshot_does_not_preserve_native_gripper_internal_action")

    result = {
        "status": "read_only_resume_sources_verified",
        "prefix_archive": str(prefix_archive),
        "route_archive": str(route_archive),
        "bundle_path": str(bundle_path),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": snapshot_file_sha,
        "bundle_sha256": bundle_file_sha,
        "timestep": RESUME_TIMESTEP,
        "integration_state_sha256": integration_sha,
        "model_geometry_sha256": envelope["model_geometry_sha256"],
        "route": {
            "qpos_points": len(qpos_path),
            "eef_reference_points": len(eef_path),
            "path_cursor": resume_index,
            "remaining_points_in_bundle": len(remaining_qpos),
            "source_start_state_sha256": route["planner_result"].get("start_state_sha256"),
            "qpos_path_sha256": route["qpos_path_sha256"],
            "eef_reference_path_sha256": route["eef_reference_path_sha256"],
            "safety_policy_sha256": route["archived_policy_sha256"],
            "first_feasible_iteration": None,
            "minimum_candidate_iterations": route["first_feasible_iteration"],
            "candidate_iteration_order_is_chronological": False,
            "first_feasible_elapsed_time_s": None,
            "first_feasible_elapsed_time_evidence": (
                "not recorded in the archived planner_result_details event;"
                " only total planner wall time and per-route iteration counts are available"
            ),
            "archived_planner_total_wall_time_s": float(
                route["planner_result"].get("planning_seconds", float("nan"))
            ),
            "archived_route_candidate_iterations": [
                {
                    "route_index": int(candidate.get("route_index", -1)),
                    "iterations": int(candidate.get("iterations", -1)),
                }
                for candidate in route["planner_result"].get("route_candidates", [])
            ],
            "planner_search_repeated": False,
            "full_path_geometry_check": "reuse_archived_validation; live suffix is previewed segment by segment",
        },
        "execution_prefix": {
            "actions": len(trace["executed_actions"]),
            "timesteps_contiguous": [int(trace["step_timesteps"][0]), int(trace["step_timesteps"][-1])],
            "observed_substeps": observed_substeps,
            "bilateral_finger_substeps": bilateral_substeps,
            "finger_only_contact_pairs": sorted([list(pair) for pair in robot_target_contact_pairs]),
            "all_substep_safety_pass": safety_complete,
            "all_gripper_commands_neutral": True,
            "max_joint_error_rad": max(qpos_errors),
            "joint_error_last5_rad": qpos_errors[-5:],
            "max_eef_position_error_m": max(eef_position_errors),
            "eef_position_error_end_m": eef_position_errors[-1],
            "eef_orientation_error_end_rad": eef_orientation_errors[-1],
        },
        "resume_state": {
            "arm_joint_qpos": actual_qpos.tolist(),
            "eef_position_world_m": actual_eef.tolist(),
            "eef_quaternion_wxyz": actual_eef_quat.tolist(),
            "gripper_qpos_m": final_robot["gripper_qpos"],
            "gripper_current_action": _jsonable(saved_gripper_internal),
            "pot2_root_position_world_m": final_object["root_pos"],
            "pot2_root_quaternion_wxyz": final_object["root_quat"],
            "qpos_to_route_index30_l2_rad": float(np.linalg.norm(actual_qpos - qpos_path[resume_index])),
            "eef_position_to_route_index30_m": float(np.linalg.norm(actual_eef - eef_path[resume_index, :3])),
            "eef_orientation_to_route_index30_rad": _quat_error_rad(actual_eef_quat, eef_path[resume_index, 3:]),
            "maximum_target_penetration_m": maximum_target_penetration_m,
        },
        "task_goal_source_event": route["task_goal_event"],
        "route_endpoint_to_archived_transport_goal": {
            "position_error_m": float(np.linalg.norm(
                eef_path[-1, :3] - np.asarray(
                    route["task_goal_event"]["goal"]["position_world_m"], dtype=np.float64
                )
            )),
            "orientation_error_rad": _quat_error_rad(
                eef_path[-1, 3:], route["task_goal_event"]["goal"]["quaternion_wxyz"]
            ),
        },
        "bundle_metadata": bundle_meta,
        "audit_acceptance": audit["acceptance"],
        "snapshot_python_controller_state_sha256": _normalized_state_hash(snapshot_state["robots"][0]["controller"]),
        "snapshot_python_gripper_state_sha256": _normalized_state_hash(snapshot_state["robots"][0]["gripper_current_action"]),
    }
    return {
        "audit": result,
        "route": route,
        "bundle_metadata": bundle_meta,
        "snapshot_state": snapshot_state,
        "trace": trace,
        "final_robot": final_robot,
        "final_object": final_object,
        "qpos_path": qpos_path,
        "eef_path": eef_path,
    }


@dataclass
class ResumedJointSpacePlan(JointSpacePlan):
    """Full original path plus an attested cursor into that path."""

    path_start_index: int = 0

    def validate(self, start_qpos: np.ndarray, goal, config) -> None:
        path = np.asarray(self.qpos_path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1:] != (7,) or len(path) < 2:
            raise ValueError("resumed_route_must_preserve_complete_original_7d_path")
        if not np.isfinite(path).all():
            raise ValueError("resumed_route_contains_nonfinite_qpos")
        if not 0 <= int(self.path_start_index) < len(path) - 1:
            raise ValueError("resumed_route_cursor_is_out_of_bounds")
        current_error = float(np.linalg.norm(path[self.path_start_index] - start_qpos))
        if current_error > config.maximum_joint_path_deviation_rad:
            raise ValueError("resumed_route_cursor_does_not_match_current_qpos_within_frozen_limit")
        if self.planner_status != "offline_feasible" or not self.planner_source:
            raise ValueError("resumed_route_has_no_archived_offline_feasibility_attestation")
        if not math.isfinite(float(self.planning_seconds)) or self.planning_seconds < 0:
            raise ValueError("resumed_route_planning_time_is_invalid")
        if self.planning_seconds > config.planning_time_budget_s:
            raise ValueError("resumed_route_load_exceeded_planning_budget")
        if (
            not math.isfinite(float(self.minimum_forbidden_clearance_m))
            or self.minimum_forbidden_clearance_m < config.hard_minimum_clearance_m
        ):
            raise ValueError("resumed_route_clearance_is_below_frozen_hard_threshold")
        goal.validate()


class T359ResumeBackend(ArchivedRouteTaskSuffixBackend):
    def __init__(self, *args, resume_sources: dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.resume_sources = resume_sources
        self.resume_cursor = RESUME_PATH_INDEX
        self.prepared_resume_plan: ResumedJointSpacePlan | None = None
        self.resume_plan_delivered = False
        self._milestone_status: dict[str, Any] | None = None
        self._post_release_stability_samples = deque(maxlen=5)
        self._last_post_release_sample_timestep = -1

    def prepare_resume_plan(self, observation, goal, context, config) -> ResumedJointSpacePlan:
        if self.prepared_resume_plan is not None:
            raise RuntimeError("t359_resume_route_was_prepared_more_than_once")
        if context.phase != "transport" or context.held_object != OBJECT_ID:
            raise RuntimeError("t359_resume_requires_live_pot2_transport_context")
        planner = self.cached_route["planner_result"]
        path = self.cached_route["qpos_path"].copy()
        poses = self.cached_route["eef_reference_poses_world"].copy()
        index = self.resume_cursor
        state = observation.raw["snapshot_state"]["mujoco_integration_state"]
        current_state_sha = _canonical_array_hash(np.asarray(state, dtype=np.float64))
        if current_state_sha != self.resume_sources["audit"]["integration_state_sha256"]:
            raise RuntimeError("t359_resume_live_integration_state_changed_before_route_attach")
        model_sha = observation.raw.get("scene_metadata", {}).get("model_geometry_sha256")
        if model_sha != MODEL_SHA256 or model_sha != planner.get("model_geometry_sha256"):
            raise RuntimeError("t359_resume_model_fingerprint_mismatch")
        if _canonical_sha256(self.safety_policy) != self.cached_route["archived_policy_sha256"]:
            raise RuntimeError("t359_resume_live_safety_policy_fingerprint_mismatch")
        if len(path) != 257 or not np.array_equal(
            path[index:], self.resume_sources["qpos_path"][index:]
        ):
            raise RuntimeError("t359_resume_path_index_or_full_route_integrity_mismatch")
        q_error = float(np.linalg.norm(observation.qpos - path[index]))
        eef_error = float(np.linalg.norm(observation.eef_position_world_m - poses[index, :3]))
        orientation_error = float(_quat_error_rad(observation.eef_quaternion_wxyz, poses[index, 3:]))
        if q_error > config.maximum_joint_path_deviation_rad:
            raise RuntimeError("t359_resume_qpos_cursor_deviation_exceeds_frozen_limit")
        if (
            eef_error > config.maximum_eef_tracking_error_m
            or orientation_error > config.maximum_eef_orientation_error_rad
        ):
            raise RuntimeError("t359_resume_EEF_cursor_deviation_exceeds_frozen_limit")
        transport_position_tolerance, transport_rotation_tolerance = (
            self.task_controller.pose_goal_tolerances("transport")
        )
        endpoint_position_error = float(np.linalg.norm(
            poses[-1, :3] - np.asarray(goal.position_world_m, dtype=np.float64)
        ))
        endpoint_orientation_error = _quat_error_rad(poses[-1, 3:], goal.quaternion_wxyz)
        if (
            endpoint_position_error > transport_position_tolerance
            or endpoint_orientation_error > transport_rotation_tolerance
        ):
            raise RuntimeError("archived_route_terminal_pose_is_outside_current_transport_goal_tolerance")
        if float(planner.get("minimum_forbidden_clearance_m", -1.0)) < config.hard_minimum_clearance_m:
            raise RuntimeError("archived_route_clearance_is_below_current_frozen_operational_policy")

        self._plan_poses = poses
        self.plan_cursor = index
        self.plan_counter += 1
        plan = ResumedJointSpacePlan(
            qpos_path=path,
            minimum_forbidden_clearance_m=float(planner["minimum_forbidden_clearance_m"]),
            planner_status=str(planner["status"]),
            planner_source=str(planner["planner_source"]),
            planning_seconds=0.0,
            diagnostics={
                "route_reused_from_archive": True,
                "route_resume_cursor": index,
                "route_original_path_points": len(path),
                "archived_planner_seconds": float(planner["planning_seconds"]),
                "archived_planner_iterations": int(planner["iterations"]),
                "archived_configuration_checks": int(planner["config_checks"]),
                "qpos_path_sha256": self.cached_route["qpos_path_sha256"],
                "eef_reference_path_sha256": self.cached_route["eef_reference_path_sha256"],
                "resume_current_qpos_to_path_cursor_l2_rad": q_error,
                "resume_current_eef_to_path_cursor_position_m": eef_error,
                "resume_current_eef_to_path_cursor_orientation_rad": orientation_error,
                "current_transport_goal_endpoint_position_error_m": endpoint_position_error,
                "current_transport_goal_endpoint_orientation_error_rad": endpoint_orientation_error,
                "current_transport_phase_tolerances": {
                    "position_m": transport_position_tolerance,
                    "orientation_rad": transport_rotation_tolerance,
                },
            },
            path_start_index=index,
        )
        plan.validate(observation.qpos, goal, config)
        self.prepared_resume_plan = plan
        self.artifacts.event({
            "event": "t359_archived_transport_route_resume_attested",
            "timestep": observation.timestep,
            "route_path_cursor": index,
            "route_original_path_points": len(path),
            "current_integration_state_sha256": current_state_sha,
            "model_geometry_sha256": model_sha,
            "safety_policy_sha256": self.cached_route["archived_policy_sha256"],
            "route_goal_source": goal.source,
            **plan.diagnostics,
            "planner_search_repeated": False,
        })
        return plan

    def plan_rrt_connect(self, observation, goal, context, config):
        if self.prepared_resume_plan is None:
            raise RuntimeError("t359_resume_route_requires_preflight_plan_attestation")
        if self.resume_plan_delivered:
            raise RuntimeError("resume_route_replan_not_authorized_without_new_evidence")
        self.resume_plan_delivered = True
        return self.prepared_resume_plan

    def task_expert_action(self, observation, context):
        if (
            context.phase == "withdrawal"
            and context.held_object is None
            and self.current_native_task_status is not None
            and any(
                tuple(str(token).lower() for token in item.get("predicate", []))
                == ("on", OBJECT_ID, "flat_stove_1_cook_region")
                and item.get("value") is True
                for item in self.current_native_task_status.get("native_goal_values", [])
            )
        ):
            # The user requested a Pot 2 local placement endpoint. Keep the
            # already released robot still while the existing stability
            # thresholds collect five physical samples; do not enter Pot 1 or
            # create a new withdrawal plan. Neutral gripper input preserves
            # the released state without continuing to drive the open limit.
            action = np.zeros(7, dtype=np.float32)
            self._current_action_source = "pot2_native_on_post_release_stability_hold"
            self.current_forecast = action[None, :].copy()
            return action
        return super().task_expert_action(observation, context)

    def task_milestone_reached(self, observation) -> dict[str, Any] | None:
        """Stop immediately after Pot 2 is natively on the burner and settled."""
        if self.task_controller is None:
            return None
        context = self.current_context()
        status = self.current_native_task_status or {}
        predicate = ("on", OBJECT_ID, "flat_stove_1_cook_region")
        value = next((
            item.get("value") for item in status.get("native_goal_values", [])
            if tuple(str(token).lower() for token in item.get("predicate", [])) == predicate
        ), None)
        if (
            value is not True
            or context is None
            or context.held_object is not None
            or context.phase not in {"withdrawal", "settle", "second_object_recovery"}
        ):
            self._post_release_stability_samples.clear()
            self._last_post_release_sample_timestep = -1
            return None
        timestep = int(observation.timestep)
        if timestep != self._last_post_release_sample_timestep:
            pot2_state = self._live_object_state(self.latest_data or {}, OBJECT_ID)
            position = np.asarray(pot2_state.get("root_pos"), dtype=np.float64)
            quaternion = np.asarray(pot2_state.get("root_quat"), dtype=np.float64)
            qvel = np.asarray(pot2_state.get("joint_qvel", []), dtype=np.float64)
            if position.shape != (3,) or quaternion.shape != (4,) or qvel.shape != (6,):
                self._post_release_stability_samples.clear()
                return None
            self._post_release_stability_samples.append((timestep, position, quaternion, qvel))
            self._last_post_release_sample_timestep = timestep
        samples = list(self._post_release_stability_samples)
        if len(samples) < self._post_release_stability_samples.maxlen:
            return None
        positions = np.stack([item[1] for item in samples])
        position_span = float(np.max(np.linalg.norm(positions - positions[-1], axis=1)))
        qvel = samples[-1][3]
        linear_limit = float(self.config_data["task_execution"].get("settle_linear_speed_m_s", 0.05))
        angular_limit = float(self.config_data["task_execution"].get("settle_angular_speed_rad_s", 0.5))
        position_limit = float(self.config_data["task_execution"].get("settle_position_span_m", 0.001))
        linear_speed = float(np.linalg.norm(qvel[:3]))
        angular_speed = float(np.linalg.norm(qvel[3:]))
        if position_span > position_limit or linear_speed > linear_limit or angular_speed > angular_limit:
            return None
        self._milestone_status = {
            "milestone": "pot2_on_native_predicate_and_settled_after_release",
            "timestep": int(observation.timestep),
            "task_phase_controller_phase_after_release": self.task_controller.phase,
            "withdrawal_motion_skipped": True,
            "held_object": context.held_object,
            "native_predicate": ["on", OBJECT_ID, "flat_stove_1_cook_region"],
            "native_predicate_value": True,
            "native_task_success": status.get("task_success"),
            "settle_samples": len(samples),
            "settle_position_span_m": position_span,
            "settle_position_span_limit_m": position_limit,
            "pot2_linear_speed_m_s": linear_speed,
            "settle_linear_speed_limit_m_s": linear_limit,
            "pot2_angular_speed_rad_s": angular_speed,
            "settle_angular_speed_limit_rad_s": angular_limit,
        }
        return dict(self._milestone_status)

    def execute_action(self, action: np.ndarray, context):
        if self.transition_count >= MAX_RESUME_ACTIONS:
            raise RuntimeError("t359_resume_450_physical_action_budget_exhausted")
        observation = super().execute_action(action, context)
        if context.held_object == OBJECT_ID and context.phase not in {"release", "withdrawal", "settle"}:
            evidence = self._grasp_states()[2].get(OBJECT_ID, {})
            if evidence.get("status") != "pass" or evidence.get("grasp_confirmed") is not True:
                self.artifacts.event({
                    "event": "t359_resume_live_grasp_lost_fail_closed",
                    "timestep": observation.get("timestep"),
                    "phase": context.phase,
                    "grasp_evidence": evidence,
                })
                raise RuntimeError("t359_resume_live_pot2_grasp_lost_after_physical_transition")
        return observation


def _contact_sides(contacts: dict[str, Any], pot_geom_ids: set[int]) -> tuple[bool, bool, list[dict[str, Any]]]:
    left = right = False
    robot_contacts = []
    for contact in contacts.get("contacts", []):
        body1, body2 = contact.get("body1", {}), contact.get("body2", {})
        geom1, geom2 = contact.get("geom1", {}), contact.get("geom2", {})
        robot1 = str(body1.get("name", "")).startswith(("robot0_", "gripper0_"))
        robot2 = str(body2.get("name", "")).startswith(("robot0_", "gripper0_"))
        if not robot1 and not robot2:
            continue
        robot_geom = geom1 if robot1 else geom2
        scene_body = body2 if robot1 else body1
        scene_geom = geom2 if robot1 else geom1
        robot_contacts.append(contact)
        if (
            str(scene_body.get("name", "")).startswith(f"{OBJECT_ID}_")
            and int(scene_geom.get("id", -1)) in pot_geom_ids
            and int(robot_geom.get("id", -1)) in LEFT_FINGER_GEOMS | RIGHT_FINGER_GEOMS
        ):
            left |= int(robot_geom["id"]) in LEFT_FINGER_GEOMS
            right |= int(robot_geom["id"]) in RIGHT_FINGER_GEOMS
        else:
            return False, False, robot_contacts
    return left, right, robot_contacts


def _resume_attestation(sources: dict[str, Any]) -> dict[str, Any]:
    snapshot_state = sources["snapshot_state"]
    robot = sources["final_robot"]
    obj = sources["final_object"]
    return {
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(RESUME_SNAPSHOT),
        "source_snapshot_sha256": sources["audit"]["snapshot_sha256"],
        "source_branch_report": str(SOURCE_REPORT_PATH),
        "source_action_record": str(PREFIX_ARCHIVE / "observation_action_trace.npz"),
        "timestep": RESUME_TIMESTEP,
        "model_geometry_sha256": MODEL_SHA256,
        "object_id": OBJECT_ID,
        "arm_joint_qpos": robot["joint_qpos"],
        "eef_position_world_m": robot["eef_pos"],
        "eef_quaternion_wxyz": robot["eef_quat"],
        "gripper_qpos_m": robot["gripper_qpos"],
        "object_position_world_m": obj["root_pos"],
        "object_quaternion_wxyz": obj["root_quat"],
        "restored_gripper_current_action": _jsonable(
            snapshot_state["robots"][0]["gripper_current_action"]
        ),
        "integration_state_sha256": sources["audit"]["integration_state_sha256"],
        "controller_state_sha256": sources["audit"]["snapshot_python_controller_state_sha256"],
        "source_route_path_cursor": RESUME_PATH_INDEX,
        "previous_branch_rise_from_t249_m": 0.039129220337848,
    }


def _parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-archive", type=Path, default=PREFIX_ARCHIVE)
    parser.add_argument("--route-archive", type=Path, default=ARCHIVED_ROUTE_DIR)
    parser.add_argument("--resume-bundle", type=Path, default=RESUME_BUNDLE)
    parser.add_argument("--resume-snapshot", type=Path, default=RESUME_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38742)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    sources = _load_resume_sources(
        args.prefix_archive, args.route_archive, args.resume_bundle, args.resume_snapshot
    )
    if args.audit_only:
        return sources["audit"]
    source_report = json.loads(SOURCE_REPORT_PATH.read_text(encoding="utf-8"))
    source_policy = source_report["policy"]
    config_data, recovery = load_runtime_config(CONFIG_PATH)
    config_data = json.loads(json.dumps(config_data))
    config_data["candidate_search"]["enabled"] = False
    object_clearances = {
        str(item["object_id"]): float(item["minimum_clearance_m"])
        for item in source_policy["allowed_target_objects"]
    }
    recovery = replace(
        recovery,
        hard_minimum_clearance_m=float(source_policy["minimum_clearance_m"]),
        target_object_clearance_m=float(source_policy["minimum_clearance_m"]),
        object_clearance_m=object_clearances,
        execution_step_budget=RESUME_TIMESTEP + MAX_RESUME_ACTIONS,
        horizon_steps=MAX_TASK_TIMESTEP,
        standard_horizon_steps=STANDARD_HORIZON,
        allow_horizon_extension=True,
    )
    recovery.validate()
    config_data["recovery"] = {
        **vars(recovery),
        "object_clearance_m": dict(recovery.object_clearance_m),
    }
    attestation = _resume_attestation(sources)
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(
        f"/opt/data/private/gs_opd/rollouts/l0t2_round0_t359_pot2_placement_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "resume_source_read_only_audit.json", sources["audit"])
    _write_json(output_dir / "run_config.json", {
        "mode": "task-runtime-snapshot-resume",
        "config_path": str(CONFIG_PATH),
        "source_snapshot_attestation": attestation,
        "source_route_events": str(sources["route"]["events_path"]),
        "source_route_events_sha256": sources["route"]["events_sha256"],
        "full_original_route_preserved": True,
        "resume_path_start_index": RESUME_PATH_INDEX,
        "resume_cursor_is_validated_against_current_state": True,
        "resume_first_connection_requires_isolated_osc_preview": True,
        "planner_search_repeated": False,
        "maximum_new_physical_actions": MAX_RESUME_ACTIONS,
        "maximum_absolute_timestep": RESUME_TIMESTEP + MAX_RESUME_ACTIONS,
        "demonstration_horizon": MAX_TASK_TIMESTEP,
        "standard_benchmark_horizon": STANDARD_HORIZON,
        "stop_at_native_pot2_on_and_settled": True,
        "operational_safety_policy_sha256": sources["route"]["archived_policy_sha256"],
        "strict_10mm_role": "supplementary_statistic_only",
        "effective_recovery_config": config_data["recovery"],
        "no_timestep_or_physical_state_edit": True,
    })

    artifacts = RecoveryArtifactWriter(output_dir)
    bridge = BridgeProcess(
        output_dir, python=args.python, port=args.bridge_port
    )
    backend: T359ResumeBackend | None = None
    engine: AutomaticSafetyRecoveryEngine | None = None
    result_json: dict[str, Any] | None = None
    video_released = False
    original_handlers = {}

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal_{signum}_received_after_durable_action_boundary")

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.signal(signum, interrupt)
    started = time.monotonic()
    try:
        bridge.start(
            benchmark=config_data["benchmark"],
            level=int(config_data["level"]),
            task_id=int(config_data["task_id"]),
            init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=MAX_TASK_TIMESTEP,
        )
        cached_route = sources["route"]
        backend = T359ResumeBackend(
            bridge,
            config_data,
            recovery,
            output_dir,
            artifacts,
            mode="task-runtime-snapshot",
            student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=args.resume_snapshot,
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
            runtime_snapshot_attestation=attestation,
            journal_dir=output_dir / "durable_action_journal",
            cached_route=cached_route,
            resume_sources=sources,
        )
        restored = backend.restore_start_state()
        if int(restored.get("timestep", -1)) != RESUME_TIMESTEP:
            raise RuntimeError("strictly_restored_resume_timestep_is_not_359")
        if restored.get("scene_metadata", {}).get("model_geometry_sha256") != MODEL_SHA256:
            raise RuntimeError("strictly_restored_t359_model_geometry_fingerprint_mismatch")
        restored_state = restored.get("snapshot_state", {}).get("mujoco_integration_state")
        restored_state_hash = _canonical_array_hash(np.asarray(restored_state, dtype=np.float64))
        if restored_state_hash != sources["audit"]["integration_state_sha256"]:
            raise RuntimeError("strictly_restored_t359_integration_state_hash_mismatch")

        restored_export = bridge.request({"op": "export_runtime_snapshot"})
        if restored_export.get("status") != "complete":
            raise RuntimeError("restored_t359_runtime_controller_state_export_failed")
        restored_envelope = pickle.loads(restored_export["payload"])
        restored_snapshot = restored_envelope.get("snapshot", {})
        source_snapshot = sources["snapshot_state"]
        expected_internal_hash = _normalized_state_hash(source_snapshot)
        restored_internal_hash = _normalized_state_hash(restored_snapshot)
        if restored_internal_hash != expected_internal_hash:
            raise RuntimeError("restored_t359_controller_or_gripper_internal_state_mismatch")

        preflight_data = bridge.request({"op": "observe", "include_snapshot_state": True})
        if int(preflight_data.get("timestep", -1)) != RESUME_TIMESTEP:
            raise RuntimeError("t359_read_only_observation_timestep_mismatch")
        backend._load_context_inputs(preflight_data)
        backend.safety_policy["joint_limit_tolerance_policy"] = source_policy[
            "joint_limit_tolerance_policy"
        ]
        backend.safety_policy["source"] = (
            "frozen_t249_t329_native_operational_policy;"
            "unit_aware_joint_limits;finger_only_live_grasp;strict10mm_statistic_only"
        )
        backend.latest_data = preflight_data
        native_status = bridge.request({"op": "native_task_status"})
        if native_status.get("status") != "complete" or native_status.get("task_success") is not False:
            raise RuntimeError("t359_native_task_state_is_not_pending_and_known")
        backend.latest_data["native_task_status"] = native_status
        context = backend.current_context()
        if context.phase != "transport" or context.held_object != OBJECT_ID:
            raise RuntimeError("t359_live_task_phase_or_held_object_mismatch")
        if _canonical_sha256(backend.safety_policy) != cached_route["archived_policy_sha256"]:
            raise RuntimeError("t359_live_safety_policy_fingerprint_does_not_match_route")
        grasp_evidence = context.metadata.get("held_object_state_evidence", {}).get(OBJECT_ID, {})
        expected_gripper_action = np.asarray(
            attestation["restored_gripper_current_action"], dtype=np.float64
        )
        actual_gripper_action = np.asarray(
            grasp_evidence.get("gripper_current_action"), dtype=np.float64
        )
        if (
            grasp_evidence.get("grasp_confirmed") is not True
            or grasp_evidence.get("finger_contacts", {}).get("left") is not True
            or grasp_evidence.get("finger_contacts", {}).get("right") is not True
            or actual_gripper_action.shape != expected_gripper_action.shape
            or not np.array_equal(actual_gripper_action, expected_gripper_action)
        ):
            raise RuntimeError("t359_live_gripper_or_bilateral_finger_grasp_restore_mismatch")

        contacts = bridge.request({"op": "inspect_contacts"})
        pot_binding = backend.bindings[OBJECT_ID]
        pot_geom_ids = set(int(item) for item in pot_binding["geom_ids"])
        if contacts.get("status") not in {"contact", "no_contact"}:
            raise RuntimeError("t359_live_contact_inventory_unavailable")
        left_now, right_now, live_robot_contacts = _contact_sides(contacts, pot_geom_ids)
        if not left_now or not right_now:
            raise RuntimeError("t359_live_contact_inventory_does_not_confirm_bilateral_finger_hold")
        if not _all_robot_contacts_finger_target(contacts, backend):
            raise RuntimeError("t359_live_robot_contact_is_not_finger_only_to_pot2")

        backend.latest_data = preflight_data
        preflight_data["native_task_status"] = native_status
        preflight_data["contacts"] = contacts
        preflight_data["contact_monitoring"] = None
        preflight_data["safety_geometry"] = bridge.request({
            "op": "observe",
            "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        }).get("safety_geometry")
        if preflight_data["safety_geometry"].get("status") != "complete":
            raise RuntimeError("t359_current_state_fails_frozen_operational_safety_preflight")
        observation = backend._observation(preflight_data)
        context = backend.current_context()
        goal = backend.resolve_goal(observation, context)
        plan = backend.prepare_resume_plan(observation, goal, context, recovery)
        risk_boundary = RiskBoundaryState.from_clearance(
            observation.minimum_forbidden_clearance_m,
            recovery.risk_trigger_distance_m,
            recovery.risk_rearm_distance_m,
        )
        isolated_preview = backend.preview_prefix(
            observation,
            plan,
            context,
            recovery,
            risk_boundary_armed=risk_boundary.armed,
        )
        preview_errors = {"joint_path_rad": [], "eef_position_m": [], "eef_orientation_rad": []}
        preview_bilateral_substeps = 0
        preview_substeps = 0
        preview_contact_safe = True
        for preview_observation in isolated_preview.observations:
            reference_qpos = preview_observation.raw.get("reference_qpos")
            reference_pose = preview_observation.raw.get("reference_eef_pose")
            if reference_qpos is None or reference_pose is None:
                preview_contact_safe = False
                continue
            preview_errors["joint_path_rad"].append(float(np.linalg.norm(
                preview_observation.qpos - np.asarray(reference_qpos, dtype=np.float64)
            )))
            preview_errors["eef_position_m"].append(float(np.linalg.norm(
                preview_observation.eef_position_world_m - np.asarray(reference_pose, dtype=np.float64)[:3]
            )))
            preview_errors["eef_orientation_rad"].append(_quat_error_rad(
                preview_observation.eef_quaternion_wxyz,
                np.asarray(reference_pose, dtype=np.float64)[3:],
            ))
            monitor = preview_observation.raw.get("contact_monitoring", {})
            substeps = monitor.get("geometry_monitoring", {}).get("substeps", [])
            if monitor.get("status") != "complete" or len(substeps) != EXPECTED_SUBSTEPS:
                preview_contact_safe = False
            for substep in substeps:
                preview_substeps += 1
                if substep.get("status") != "complete" or any(
                    substep.get(key)
                    for key in (
                        "forbidden_contacts", "forbidden_object_contacts", "self_contacts",
                        "unallowed_scene_contacts", "unresolved_contacts", "joint_limit_violations",
                    )
                ):
                    preview_contact_safe = False
                left, right, _ = _contact_sides(
                    {"contacts": substep.get("contact_identities", [])}, pot_geom_ids
                )
                preview_bilateral_substeps += int(left and right)
                if not left or not right:
                    preview_contact_safe = False
                if any(float(value) < -0.002 for value in substep.get(
                    "allowed_contact_clearances_m", {}
                ).values()):
                    preview_contact_safe = False
        if (
            not isolated_preview.complete
            or not isolated_preview.safe
            or not isolated_preview.main_state_restored_verified
            or isolated_preview.artifacts.get("stop_reason") is not None
            or len(isolated_preview.actions) != recovery.segment_control_steps
            or not np.allclose(isolated_preview.actions[:, 6], 0.0, rtol=0.0, atol=1e-7)
            or not preview_contact_safe
            or preview_substeps != len(isolated_preview.actions) * EXPECTED_SUBSTEPS
            or preview_bilateral_substeps != preview_substeps
            or not preview_errors["joint_path_rad"]
            or max(preview_errors["joint_path_rad"]) > recovery.maximum_joint_path_deviation_rad
            or max(preview_errors["eef_position_m"]) > recovery.maximum_eef_tracking_error_m
            or max(preview_errors["eef_orientation_rad"]) > recovery.maximum_eef_orientation_error_rad
        ):
            raise RuntimeError(
                "t359_first_route_connection_isolated_preview_did_not_pass:"+
                str(isolated_preview.failure_reason or isolated_preview.artifacts.get("stop_reason"))
            )

        preflight_report = {
            "status": "t359_resume_and_first_route_connection_preview_passed",
            "restored_timestep": RESUME_TIMESTEP,
            "restored_integration_state_sha256": restored_state_hash,
            "model_geometry_sha256": MODEL_SHA256,
            "snapshot_internal_state_sha256_before_restore": expected_internal_hash,
            "snapshot_internal_state_sha256_after_restore": restored_internal_hash,
            "gripper_current_action": actual_gripper_action.tolist(),
            "gripper_qpos_m": grasp_evidence.get("gripper_qpos_m"),
            "held_object": context.held_object,
            "bilateral_live_finger_contacts": {"left": left_now, "right": right_now},
            "live_robot_contact_count": len(live_robot_contacts),
            "safety_policy_sha256": cached_route["archived_policy_sha256"],
            "route_full_path_preserved": True,
            "route_path_cursor": RESUME_PATH_INDEX,
            "route_qpos_cursor_deviation_rad": plan.diagnostics[
                "resume_current_qpos_to_path_cursor_l2_rad"
            ],
            "route_eef_cursor_position_error_m": plan.diagnostics[
                "resume_current_eef_to_path_cursor_position_m"
            ],
            "route_eef_cursor_orientation_error_rad": plan.diagnostics[
                "resume_current_eef_to_path_cursor_orientation_rad"
            ],
            "current_goal_endpoint_position_error_m": plan.diagnostics[
                "current_transport_goal_endpoint_position_error_m"
            ],
            "current_goal_endpoint_orientation_error_rad": plan.diagnostics[
                "current_transport_goal_endpoint_orientation_error_rad"
            ],
            "current_transport_phase_tolerances": plan.diagnostics[
                "current_transport_phase_tolerances"
            ],
            "first_connection_preview_path_indices": [RESUME_PATH_INDEX + 1,
                RESUME_PATH_INDEX + len(isolated_preview.actions)],
            "first_connection_preview_actions": len(isolated_preview.actions),
            "first_connection_preview_substeps": preview_substeps,
            "first_connection_preview_bilateral_finger_substeps": preview_bilateral_substeps,
            "first_connection_preview_tracking_maxima": {
                key: max(values) for key, values in preview_errors.items()
            },
            "first_connection_preview_safe": isolated_preview.safe,
            "preview_main_snapshot_restored": isolated_preview.main_state_restored_verified,
            "next_stage_execution_authorized_by_user": True,
            "remaining_action_budget": MAX_RESUME_ACTIONS,
        }
        _write_json(output_dir / "t359_resume_preflight.json", preflight_report)
        artifacts.event({"event": "t359_resume_preflight_passed", **preflight_report})
        backend.latest_data = preflight_data
        backend.initialize_trajectory()
        backend._save_video_frame(preflight_data)

        engine = AutomaticSafetyRecoveryEngine(backend, recovery, artifacts=artifacts)
        engine_result = engine.run()
        result_json = _jsonable(engine_result.__dict__)
        suffix_verdict = _suffix_acceptance(backend.trajectory, backend.safety_policy)
        terminal = backend.observe()
        terminal_native = terminal.raw.get("native_task_status")
        pot2_on = next((
            item.get("value") for item in (terminal_native or {}).get("native_goal_values", [])
            if tuple(str(token).lower() for token in item.get("predicate", []))
            == ("on", OBJECT_ID, "flat_stove_1_cook_region")
        ), None)
        milestone = backend._milestone_status
        result_json.update({
            "native_task_status": terminal_native,
            "native_task_success": terminal.task_success,
            "pot2_native_on_predicate": pot2_on,
            "pot2_local_placement_success": bool(milestone and pot2_on is True),
            "pot2_milestone_evidence": milestone,
            "physical_actions": backend.transition_count,
            "maximum_authorized_physical_actions": MAX_RESUME_ACTIONS,
            "initial_timestep": RESUME_TIMESTEP,
            "final_timestep": terminal.timestep,
            "suffix_native_safety": suffix_verdict["native_safety"],
            "suffix_operational_safety": suffix_verdict["operational_safety"],
            "suffix_strict_10mm": suffix_verdict["strict_10mm"],
            "suffix_acceptance": suffix_verdict,
            "complete_task_success": terminal.task_success is True,
            "complete_round0_seed": False,
            "historical_prefix_fully_verified_under_current_contact_schema": False,
            "pot1_phase_started": any(
                event.get("event") == "task_phase_transition"
                and event.get("to_phase") == "second_object_recovery"
                for event in engine._events
            ),
            "planner_search_repeated": False,
            "resume_source_snapshot_sha256": sources["audit"]["snapshot_sha256"],
            "resume_integration_state_sha256": sources["audit"]["integration_state_sha256"],
            "resume_route_qpos_sha256": cached_route["qpos_path_sha256"],
            "resume_route_eef_sha256": cached_route["eef_reference_path_sha256"],
            "failure_category": (
                "none_pot2_local_milestone_reached"
                if milestone
                else "safety_or_control_fail_closed"
                if suffix_verdict["native_safety"] is False
                or suffix_verdict["operational_safety"] is False
                else "route_or_task_stage_budget_or_progress_stop"
            ),
        })
        return result_json
    except BaseException as exc:
        result_json = {
            "status": "interrupted_or_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_timestep": (
                int(backend.latest_data.get("timestep", -1))
                if backend is not None and backend.latest_data else None
            ),
            "failure_phase": getattr(engine, "_phase", None) if engine else None,
            "physical_transitions_recorded": backend.transition_count if backend else 0,
            "task_success": False,
            "complete_round0_seed": False,
            "preflight_report": str(output_dir / "t359_resume_preflight.json"),
        }
        return result_json
    finally:
        if backend is not None:
            try:
                if hasattr(backend, "trajectory"):
                    backend.finalize(result_json or {"complete_task_success": False})
                    video_released = True
            except Exception as finalize_exc:
                (output_dir / "finalize_error.txt").write_text(
                    f"{type(finalize_exc).__name__}: {finalize_exc}\n", encoding="utf-8"
                )
                try:
                    if hasattr(backend, "trajectory"):
                        backend.trajectory.save(
                            steps=backend.transition_count,
                            chunks=backend.transition_count,
                        )
                except Exception as save_exc:
                    (output_dir / "trajectory_save_error.txt").write_text(
                        f"{type(save_exc).__name__}: {save_exc}\n", encoding="utf-8"
                    )
            if not video_released:
                for writer in backend._video_writers.values():
                    writer.release()
        bridge.close()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        if result_json is not None:
            result_json["output_dir"] = str(output_dir)
            result_json["elapsed_seconds"] = time.monotonic() - started
            _write_json(output_dir / "suffix_result.json", result_json)


def main() -> int:
    args = _parser()
    try:
        result = run(args)
    except Exception as exc:
        result = {"status": "read_only_audit_failed", "failure_reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True))
    if args.audit_only:
        return 0 if result.get("status") == "read_only_resume_sources_verified" else 2
    return 0 if result.get("pot2_local_placement_success") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
