# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.

"""Continue L0T2 from the archived, physically verified Episode 9 t=329 state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from research.gs_opd.repair.automatic_recovery import (
    AutomaticSafetyRecoveryEngine,
    JointSpacePlan,
    RecoveryArtifactWriter,
)
from research.gs_opd.repair.automatic_recovery_runtime import (
    AutomaticRecoveryBridgeBackend,
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    load_runtime_config,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "research/gs_opd/configs/l0t2_task_success_first_safety_reconstruction.json"
SOURCE_BRANCH = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_round0_t249_operational_lift_20261024_branch1"
)
SOURCE_REPORT_PATH = SOURCE_BRANCH / "branch_report.json"
SOURCE_STEPS_PATH = SOURCE_BRANCH / "steps.json"
ARCHIVED_ROUTE_DIR = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "l0t2_round0_t329_native_task_suffix_20261024_194230"
)
OBJECT_ID = "moka_pot_2"
MODEL_SHA256 = "1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752"
MAX_TASK_TIMESTEP = 1800
MAX_PREFIX_ACTIONS = 30
T329_INITIAL_TIMESTEP = 329
PREFIX_END_TIMESTEP = T329_INITIAL_TIMESTEP + MAX_PREFIX_ACTIONS
EXPECTED_SUBSTEPS = 25


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _source_attestation() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source_report = json.loads(SOURCE_REPORT_PATH.read_text(encoding="utf-8"))
    rows = json.loads(SOURCE_STEPS_PATH.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("source_t249_branch_has_no_action_records")
    row = rows[-1]
    if int(row.get("timestep", -1)) != 329:
        raise ValueError("source_branch_terminal_action_is_not_t329")
    snapshot_path = Path(row["runtime_snapshot_path"])
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    snapshot_sha = sha256_file(snapshot_path)
    if snapshot_sha != row.get("runtime_snapshot_sha256"):
        raise ValueError("t329_source_action_snapshot_hash_mismatch")
    if source_report.get("expected_source_model_geometry_sha256") != MODEL_SHA256:
        raise ValueError("t249_branch_model_fingerprint_does_not_match_expected_model")
    if source_report.get("local_lift_30mm_success") is not True or source_report.get(
        "local_safe_exit_success"
    ) is not True:
        raise ValueError("t249_to_t329_local_safe_lift_attestation_is_not_successful")
    if source_report.get("terminal_grasp_state", {}).get("grasp_confirmed") is not True:
        raise ValueError("t329_source_branch_does_not_attest_a_held_pot2")
    attestation = {
        "status": "verified_source_action_snapshot",
        "source_snapshot_path": str(snapshot_path),
        "source_snapshot_sha256": snapshot_sha,
        "source_branch_report": str(SOURCE_REPORT_PATH),
        "source_action_record": str(SOURCE_STEPS_PATH),
        "timestep": 329,
        "model_geometry_sha256": MODEL_SHA256,
        "object_id": OBJECT_ID,
        "arm_joint_qpos": row["arm_joint_qpos"],
        "eef_position_world_m": row["eef_position_world_m"],
        "eef_quaternion_wxyz": row["eef_quaternion_wxyz"],
        "gripper_qpos_m": row["gripper_qpos_m"],
        "object_position_world_m": row["pot2_root_position_world_m"],
        "object_quaternion_wxyz": row["pot2_root_quaternion_wxyz"],
        "restored_gripper_current_action": row.get("gripper_current_action"),
        "previous_branch_operational_verdict": row.get("acceptance_verdicts"),
        "previous_branch_rise_from_t249_m": row.get("pot2_vertical_rise_m"),
    }
    return snapshot_path, attestation, source_report


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_archived_route(route_dir: Path) -> dict[str, Any]:
    events_path = route_dir / "recovery_events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(events_path)
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    plan_events = [
        event for event in events
        if event.get("event") == "planner_result_details"
        and event.get("task_phase") == "transport"
    ]
    start_events = [
        event for event in events
        if event.get("event") == "t329_snapshot_continuation_started"
    ]
    if len(plan_events) != 1 or len(start_events) != 1:
        raise ValueError("archived_t329_route_or_start_policy_event_is_not_unique")
    planner = plan_events[0]["planner_result"]
    if (
        planner.get("status") != "offline_feasible"
        or planner.get("operation") != "rrt_connect"
        or planner.get("live_state_unchanged") is not True
        or planner.get("model_geometry_sha256") != MODEL_SHA256
    ):
        raise ValueError("archived_t329_route_does_not_have_a_valid_feasibility_attestation")
    path = np.asarray(planner.get("qpos_path"), dtype=np.float64)
    poses = np.asarray(planner.get("eef_reference_poses_world"), dtype=np.float64)
    if (
        path.ndim != 2 or path.shape[1:] != (7,) or len(path) < 2
        or poses.shape != (len(path), 7)
        or not np.isfinite(path).all() or not np.isfinite(poses).all()
        or np.any(np.abs(np.linalg.norm(poses[:, 3:], axis=1) - 1.0) > 1e-5)
    ):
        raise ValueError("archived_t329_route_path_or_EEF_reference_is_incomplete")
    route_candidates = planner.get("route_candidates")
    selected_index = planner.get("selected_route_index")
    if (
        not isinstance(route_candidates, list)
        or isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or not 0 <= selected_index < len(route_candidates)
    ):
        raise ValueError("archived_t329_route_candidate_selection_is_malformed")
    route_metrics = route_candidates[selected_index].get("metrics", {})
    policy = start_events[0].get("operational_policy")
    if not isinstance(policy, dict):
        raise ValueError("archived_t329_safety_policy_is_missing")
    if planner.get("safety_policy_source") != policy.get("source"):
        raise ValueError("archived_t329_route_safety_policy_provenance_mismatch")
    return {
        "route_dir": str(route_dir),
        "events_path": str(events_path),
        "events_sha256": sha256_file(events_path),
        "planner_event": plan_events[0],
        "planner_result": planner,
        "archived_policy": policy,
        "archived_policy_sha256": _canonical_sha256(policy),
        "route_metrics": route_metrics,
        "route_candidate_count": len(route_candidates),
        "first_feasible_iteration": min(
            int(item.get("iterations", 0)) for item in route_candidates
        ),
        "first_feasible_wall_time_recorded": False,
        "qpos_path": path,
        "eef_reference_poses_world": poses,
        "qpos_path_sha256": hashlib.sha256(np.ascontiguousarray(path).tobytes()).hexdigest(),
        "eef_reference_path_sha256": hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest(),
    }


class DurableTaskSuffixBackend(AutomaticRecoveryBridgeBackend):
    """Use the existing backend and keep an atomic one-action recovery journal."""

    def __init__(self, *args, journal_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.journal_index_path = self.journal_dir / "index.jsonl"
        self.pending_path = self.journal_dir / "pending_action.json"

    @staticmethod
    def _record_value(prefix: str, source: dict[str, Any], arrays: dict[str, Any]) -> None:
        for key, value in source.items():
            array_key = f"{prefix}{key}"
            if isinstance(value, np.ndarray):
                arrays[array_key] = value
            elif isinstance(value, (str, int, float, bool, np.generic)) or value is None:
                arrays[array_key] = np.asarray(json.dumps(_jsonable(value)))
            elif isinstance(value, (dict, list, tuple)):
                arrays[array_key + "_json"] = np.asarray(
                    json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                )

    def execute_action(self, action: np.ndarray, context):
        prior = self.latest_data or {}
        pending = {
            "status": "action_transition_pending",
            "pre_timestep": int(prior.get("timestep", -1)),
            "task_phase": context.phase,
            "action": np.asarray(action, dtype=np.float32).tolist(),
            "note": "if retained after interruption, the physical transition may be incomplete or its response uncommitted",
        }
        _write_json(self.pending_path, pending)
        observation = super().execute_action(action, context)
        decision = self.trajectory.decisions[-1]
        step = self.trajectory.steps[-1]
        arrays: dict[str, Any] = {}
        self._record_value("pre_", decision, arrays)
        self._record_value("post_", step, arrays)
        destination = self.journal_dir / f"action_{self.transition_count:05d}_t{observation.timestep:06d}.npz"
        _atomic_npz(destination, arrays)
        entry = {
            "journal_index": self.transition_count,
            "timestep": int(observation.timestep),
            "task_phase": context.phase,
            "journal_path": str(destination),
            "journal_sha256": sha256_file(destination),
            "runtime_snapshot_path": str(
                self.output_dir / f"snapshots/main_timestep_{int(observation.timestep):06d}.pkl"
            ),
            "action": np.asarray(action, dtype=np.float32).tolist(),
        }
        with self.journal_index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if self.pending_path.exists():
            self.pending_path.unlink()
        return observation


class ArchivedRouteTaskSuffixBackend(DurableTaskSuffixBackend):
    """Reuse one exact-state RRT route; never launch a second search."""

    def __init__(self, *args, cached_route: dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_route = cached_route
        self.cached_route_consumed = False

    def plan_rrt_connect(self, observation, goal, context, config):
        if self.cached_route_consumed:
            raise RuntimeError("archived_route_replan_not_authorized_in_30_action_prefix")
        self.cached_route_consumed = True
        planner = self.cached_route["planner_result"]
        integration = np.asarray(
            observation.raw["snapshot_state"]["mujoco_integration_state"], dtype=np.float64
        )
        live_state_sha = hashlib.sha256(integration.tobytes()).hexdigest()
        live_model_sha = observation.raw.get("scene_metadata", {}).get("model_geometry_sha256")
        path = self.cached_route["qpos_path"].copy()
        poses = self.cached_route["eef_reference_poses_world"].copy()
        if live_state_sha != planner.get("start_state_sha256"):
            raise RuntimeError("archived_route_start_integration_state_fingerprint_mismatch")
        if live_model_sha != planner.get("model_geometry_sha256"):
            raise RuntimeError("archived_route_model_geometry_fingerprint_mismatch")
        if self.safety_policy.get("source") != planner.get("safety_policy_source"):
            raise RuntimeError("archived_route_active_safety_policy_source_mismatch")
        if _canonical_sha256(self.safety_policy) != self.cached_route["archived_policy_sha256"]:
            raise RuntimeError("archived_route_active_safety_policy_fingerprint_mismatch")
        if not np.allclose(path[0], observation.qpos, rtol=0.0, atol=1e-6):
            raise RuntimeError("archived_route_first_qpos_does_not_match_live_t329_state")
        if (
            np.linalg.norm(poses[0, :3] - observation.eef_position_world_m) > 1e-6
            or np.linalg.norm(poses[-1, :3] - np.asarray(goal.position_world_m)) > 1e-5
        ):
            raise RuntimeError("archived_route_EEF_endpoint_does_not_match_live_state_or_goal")
        if np.linalg.norm(poses[-1, 3:] - np.asarray(goal.quaternion_wxyz)) > 1e-5:
            raise RuntimeError("archived_route_EEF_goal_orientation_mismatch")
        if float(planner.get("minimum_forbidden_clearance_m", -1.0)) < config.hard_minimum_clearance_m:
            raise RuntimeError("archived_route_clearance_is_below_current_hard_policy")
        self._plan_poses = poses
        self.plan_cursor = 0
        self.plan_counter += 1
        path_hash = self.cached_route["qpos_path_sha256"]
        pose_hash = self.cached_route["eef_reference_path_sha256"]
        self.artifacts.event({
            "event": "archived_rrt_route_reused_and_start_attested",
            "source_events_path": self.cached_route["events_path"],
            "source_events_sha256": self.cached_route["events_sha256"],
            "source_plan_index": self.cached_route["planner_event"].get("plan_index"),
            "source_start_state_sha256": live_state_sha,
            "source_model_geometry_sha256": live_model_sha,
            "active_safety_policy_sha256": self.cached_route["archived_policy_sha256"],
            "qpos_path_sha256": path_hash,
            "eef_reference_path_sha256": pose_hash,
            "qpos_path_points": len(path),
            "eef_reference_points": len(poses),
            "planner_search_repeated": False,
            "archived_planner_search_seconds": planner.get("planning_seconds"),
            "path_current_state_checks": {
                "model_fingerprint_match": True,
                "integration_state_match": True,
                "safety_policy_fingerprint_match": True,
                "qpos_start_match": True,
                "eef_start_and_goal_match": True,
                "archived_offline_clearance_above_hard_threshold": True,
            },
        })
        _atomic_npz(
            self.output_dir / "reused_archived_route.npz",
            {
                "qpos_path": path,
                "eef_reference_poses_world": poses,
                "metadata_json": np.asarray(json.dumps({
                    "source_events_path": self.cached_route["events_path"],
                    "source_events_sha256": self.cached_route["events_sha256"],
                    "source_start_state_sha256": live_state_sha,
                    "model_geometry_sha256": live_model_sha,
                    "safety_policy_sha256": self.cached_route["archived_policy_sha256"],
                    "qpos_path_sha256": path_hash,
                    "eef_reference_path_sha256": pose_hash,
                    "planner_search_repeated": False,
                }, sort_keys=True)),
            },
        )
        return JointSpacePlan(
            qpos_path=path,
            minimum_forbidden_clearance_m=float(planner["minimum_forbidden_clearance_m"]),
            planner_status=str(planner["status"]),
            planner_source=str(planner["planner_source"]),
            # This is an already-produced plan. Keep its original 360.055 s
            # in provenance, while giving the Recovery execution budget only
            # the negligible route-load/attestation cost.
            planning_seconds=0.0,
            diagnostics={
                "route_reused_from_archive": True,
                "archived_planner_seconds": float(planner["planning_seconds"]),
                "archived_planner_iterations": int(planner["iterations"]),
                "archived_configuration_checks": int(planner["config_checks"]),
                "archived_route_metrics": self.cached_route["route_metrics"],
                "qpos_path_sha256": path_hash,
                "eef_reference_path_sha256": pose_hash,
                "archived_route_source_events_sha256": self.cached_route["events_sha256"],
            },
        )

    def preview_prefix(self, observation, plan, context, config, *, risk_boundary_armed=True):
        preview = super().preview_prefix(
            observation, plan, context, config,
            risk_boundary_armed=risk_boundary_armed,
        )
        if context.held_object is not None:
            if not np.isclose(float(preview.artifacts.get("preview_gripper_action", np.nan)), 0.0):
                preview.safe = False
                preview.failure_reason = "held_object_preview_did_not_use_neutral_gripper_hold"
            elif preview.actions.ndim == 2 and len(preview.actions) and not np.allclose(
                preview.actions[:, 6], 0.0, rtol=0.0, atol=1e-7
            ):
                preview.safe = False
                preview.failure_reason = "held_object_preview_actions_do_not_hold_gripper"
        return preview


def _all_robot_contacts_finger_target(contacts: dict[str, Any], backend) -> bool:
    finger_ids = set(int(item) for item in backend.context_static["gripper_geometry_ids"])
    binding = backend.bindings[OBJECT_ID]
    target_ids = set(int(item) for item in binding["geom_ids"])
    for contact in contacts.get("contacts", []):
        first = contact.get("body1", {})
        second = contact.get("body2", {})
        first_robot = str(first.get("name", "")).startswith(("robot0_", "gripper0_"))
        second_robot = str(second.get("name", "")).startswith(("robot0_", "gripper0_"))
        if not first_robot and not second_robot:
            continue
        robot_geom = contact.get("geom1", {}) if first_robot else contact.get("geom2", {})
        scene_body = second if first_robot else first
        scene_geom = contact.get("geom2", {}) if first_robot else contact.get("geom1", {})
        if (
            int(robot_geom.get("id", -1)) not in finger_ids
            or int(scene_body.get("id", -1)) != int(binding["root_body_id"])
            or int(scene_geom.get("id", -1)) not in target_ids
        ):
            return False
    return True


def _suffix_acceptance(trajectory, operational_policy: dict[str, Any]) -> dict[str, Any]:
    has_physical_actions = bool(trajectory.steps)
    # Zero-action planner failures are not safety passes or violations. Keep
    # the suffix-level checks explicitly unassessed instead of conflating
    # "nothing executed" with a failed physical safety check.
    operational_ok: bool | None = True if has_physical_actions else None
    native_ok: bool | None = True if has_physical_actions else None
    strict_ok: bool | None = True if has_physical_actions else None
    min_clearance: dict[str, float] = {}
    observed_substeps = 0
    expected_substeps = len(trajectory.steps) * EXPECTED_SUBSTEPS
    substep_joint_violations = 0
    for step in trajectory.steps:
        raw_monitor = step.get("contact_monitoring_json")
        monitor = json.loads(raw_monitor) if isinstance(raw_monitor, str) else {}
        geometry = monitor.get("geometry_monitoring", {})
        substeps = geometry.get("substeps", [])
        observed_substeps += len(substeps)
        if (
            monitor.get("status") != "complete"
            or geometry.get("status") != "complete"
            or len(substeps) != EXPECTED_SUBSTEPS
            or monitor.get("captured_physics_substeps") != EXPECTED_SUBSTEPS
        ):
            if operational_ok is not None:
                operational_ok = False
            if native_ok is not None:
                native_ok = False
        for item in substeps:
            if item.get("status") != "complete":
                if operational_ok is not None:
                    operational_ok = False
                if native_ok is not None:
                    native_ok = False
            if any(
                item.get(key)
                for key in (
                    "forbidden_contacts", "self_contacts", "forbidden_object_contacts",
                    "unallowed_scene_contacts", "unresolved_contacts",
                )
            ):
                if native_ok is not None:
                    native_ok = False
            joint_violations = item.get("joint_limit_violations", [])
            substep_joint_violations += len(joint_violations)
            if joint_violations:
                if operational_ok is not None:
                    operational_ok = False
            measured = {}
            measured.update(item.get("clearances_by_object", {}))
            measured.update({
                f"carried:{OBJECT_ID}->{name}": value
                for name, value in item.get("carried_clearances_by_object", {}).items()
            })
            for name, clearance in measured.items():
                distance = clearance.get("minimum_distance_m") if isinstance(clearance, dict) else None
                if distance is None:
                    continue
                value = float(distance)
                min_clearance[str(name)] = min(value, min_clearance.get(str(name), float("inf")))
                if strict_ok is not None and value < 0.010:
                    strict_ok = False
        if step.get("done") and step.get("task_success") is not True:
            if native_ok is not None:
                native_ok = False
    return {
        "native_safety": native_ok,
        "operational_safety": operational_ok,
        "strict_10mm": strict_ok,
        "strict_10mm_is_supplementary_only": True,
        "policy": operational_policy,
        "physical_actions": len(trajectory.steps),
        "expected_substeps": expected_substeps,
        "observed_substeps": observed_substeps,
        "substep_coverage_complete": (
            observed_substeps == expected_substeps if has_physical_actions else None
        ),
        "execution_status": "complete" if has_physical_actions else "not_run_zero_physical_actions",
        "operational_joint_limit_violations": substep_joint_violations,
        "minimum_clearance_by_object_m": min_clearance,
        "native_task_success_is_separately_reported": True,
    }


def _parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--route-archive", type=Path, default=ARCHIVED_ROUTE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38741)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    snapshot_path, attestation, previous_report = _source_attestation()
    cached_route = _load_archived_route(args.route_archive)
    config_data, recovery = load_runtime_config(args.config)
    config_data = json.loads(json.dumps(config_data))
    config_data["candidate_search"]["enabled"] = False
    safety_policy = previous_report["policy"]
    target_clearances = {
        str(item["object_id"]): float(item["minimum_clearance_m"])
        for item in safety_policy["allowed_target_objects"]
    }
    recovery = replace(
        recovery,
        hard_minimum_clearance_m=float(safety_policy["minimum_clearance_m"]),
        target_object_clearance_m=float(safety_policy["minimum_clearance_m"]),
        object_clearance_m=target_clearances,
        # RecoveryConfig uses absolute Episode timesteps because the engine
        # initializes ``steps`` from the restored task timestep.
        execution_step_budget=PREFIX_END_TIMESTEP,
        horizon_steps=MAX_TASK_TIMESTEP,
        standard_horizon_steps=1000,
        allow_horizon_extension=True,
    )
    recovery.validate()
    config_data["recovery"] = asdict(recovery)
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(
        f"/opt/data/private/gs_opd/rollouts/l0t2_round0_t329_native_task_suffix_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "run_config.json", {
        "mode": "task-runtime-snapshot",
        "config_path": str(args.config),
        "effective_config": config_data,
        "recovery_config": asdict(recovery),
        "source_snapshot_attestation": attestation,
        "native_operational_safety_policy_source": "exact_policy_from_t249_t329_verified_local_lift_branch",
        "strict_10mm_role": "supplementary_statistic_only_not_a_start_or_execution_blocker",
        "task_phase_controller": "fresh_instance_adopts_live_verified_held_object_and_native_predicates",
        "no_historical_action_or_controller_state_replayed": True,
        "maximum_total_task_timestep": MAX_TASK_TIMESTEP,
        "maximum_physical_prefix_actions": MAX_PREFIX_ACTIONS,
        "maximum_prefix_episode_timestep": PREFIX_END_TIMESTEP,
        "standard_benchmark_horizon": 1000,
        "reused_route_source": cached_route["events_path"],
        "reused_route_source_events_sha256": cached_route["events_sha256"],
        "reused_route_qpos_path_sha256": cached_route["qpos_path_sha256"],
        "reused_route_eef_reference_path_sha256": cached_route["eef_reference_path_sha256"],
        "planner_search_repeated": False,
    })

    artifacts = RecoveryArtifactWriter(output_dir)
    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend: ArchivedRouteTaskSuffixBackend | None = None
    engine: AutomaticSafetyRecoveryEngine | None = None
    result_json: dict[str, Any] | None = None
    video_released = False
    original_handlers = {}

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal_{signum}_received_after_durable_action_boundary")

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.signal(signum, interrupt)
    start_time = time.monotonic()
    try:
        bridge.start(
            benchmark=config_data["benchmark"],
            level=int(config_data["level"]),
            task_id=int(config_data["task_id"]),
            init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
            demonstration_horizon=MAX_TASK_TIMESTEP,
        )
        backend = ArchivedRouteTaskSuffixBackend(
            bridge,
            config_data,
            recovery,
            output_dir,
            artifacts,
            mode="task-runtime-snapshot",
            student=None,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=snapshot_path,
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
            runtime_snapshot_attestation=attestation,
            journal_dir=output_dir / "durable_action_journal",
            cached_route=cached_route,
        )
        restored = backend.restore_start_state()
        backend.latest_data = restored
        if int(restored.get("timestep", -1)) != int(attestation["timestep"]):
            raise RuntimeError("restored_t329_timestep_mismatch")
        if restored.get("scene_metadata", {}).get("model_geometry_sha256") != MODEL_SHA256:
            raise RuntimeError("restored_t329_model_geometry_fingerprint_mismatch")
        (output_dir / "start_state.json").write_text(
            json.dumps(_jsonable({
                "status": "strict_source_action_snapshot_restore_verified",
                "source_attestation": attestation,
                "restore_status": restored.get("runtime_snapshot_restore_status"),
                "timestep": restored.get("timestep"),
                "model_geometry_sha256": restored.get("scene_metadata", {}).get("model_geometry_sha256"),
                "robot_states": restored.get("snapshot_state", {}).get("robot_states"),
                "pot2_state": [
                    value for key, value in restored.get("snapshot_state", {}).get("object_states", {}).items()
                    if str(key).endswith(":" + OBJECT_ID)
                ],
            }), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        context = backend.current_context()
        context.validate()
        if context.held_object != OBJECT_ID or context.phase != "transport":
            raise RuntimeError(
                f"live_t329_task_adoption_failed:held={context.held_object},phase={context.phase}"
            )
        if backend.latest_data is None:
            raise RuntimeError("t329_live_observation_missing_after_restore")
        if not _all_robot_contacts_finger_target(backend.latest_data.get("contacts", {}), backend):
            raise RuntimeError("t329_preflight_found_nonfinger_or_nonpot2_robot_contact")
        grasp = context.metadata.get("held_object_state_evidence", {}).get(OBJECT_ID, {})
        if grasp.get("grasp_confirmed") is not True:
            raise RuntimeError("t329_live_pot2_bilateral_grasp_not_confirmed")
        if not np.isclose(
            float(attestation["previous_branch_rise_from_t249_m"]),
            float(previous_report["measured_pot2_vertical_rise_m"]),
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError("source_t329_lift_measurement_does_not_match_source_report")

        # Carry forward the exact calibrated soft-limit policy and the
        # object-specific operational clearances from the verified branch.
        backend.safety_policy["joint_limit_tolerance_policy"] = safety_policy[
            "joint_limit_tolerance_policy"
        ]
        backend.safety_policy["source"] = (
            "frozen_t249_t329_native_operational_policy;"
            "unit_aware_joint_limits;finger_only_live_grasp;strict10mm_statistic_only"
        )
        preflight = bridge.request({
            "op": "observe",
            "include_snapshot_state": True,
            "safety_geometry_policy": backend.safety_policy,
        })
        backend.latest_data = preflight
        if preflight.get("safety_geometry", {}).get("status") not in {"complete", "unsafe"}:
            raise RuntimeError("t329_operational_safety_preflight_is_incomplete")
        if preflight.get("safety_geometry", {}).get("status") != "complete":
            raise RuntimeError("t329_current_state_fails_frozen_operational_safety")
        if preflight.get("contact_monitoring") is not None:
            if preflight["contact_monitoring"].get("status") != "complete":
                raise RuntimeError("t329_contact_preflight_is_incomplete")
        native_status = bridge.request({"op": "native_task_status"})
        required_goals = {
            tuple(str(value).lower() for value in item["predicate"]): item["value"]
            for item in native_status.get("native_goal_values", [])
        }
        if native_status.get("status") != "complete" or native_status.get("task_success") is not False:
            raise RuntimeError("t329_native_task_state_is_not_a_verified_pending_task")
        backend.latest_data["native_task_status"] = native_status
        backend.initialize_trajectory()
        artifacts.event({
            "event": "t329_snapshot_continuation_started",
            "timestep": 329,
            "snapshot_sha256": attestation["source_snapshot_sha256"],
            "task_phase": context.phase,
            "held_object": context.held_object,
            "grasp_evidence": grasp,
            "native_task_status": native_status,
            "operational_policy": backend.safety_policy,
            "strict_10mm_is_supplementary_only": True,
            "historical_prefix_safety_verified": False,
            "maximum_physical_prefix_actions": MAX_PREFIX_ACTIONS,
            "maximum_prefix_episode_timestep": PREFIX_END_TIMESTEP,
            "archived_route_reuse_required": True,
        })
        engine = AutomaticSafetyRecoveryEngine(backend, recovery, artifacts=artifacts)
        result = engine.run()
        result_json = asdict(result)
        suffix_verdict = _suffix_acceptance(backend.trajectory, backend.safety_policy)
        terminal = backend.observe()
        result_json.update({
            "native_task_status": terminal.raw.get("native_task_status"),
            "task_success": terminal.task_success,
            "suffix_native_safety": suffix_verdict["native_safety"],
            "suffix_operational_safety": suffix_verdict["operational_safety"],
            "suffix_strict_10mm": suffix_verdict["strict_10mm"],
            "verified_safe_success_suffix": bool(
                result.complete_task_success
                and suffix_verdict["native_safety"]
                and suffix_verdict["operational_safety"]
            ),
            "suffix_label": (
                "verified_safe_success_suffix"
                if result.complete_task_success
                and suffix_verdict["native_safety"]
                and suffix_verdict["operational_safety"]
                else "not_a_verified_safe_success_suffix"
            ),
            "suffix_acceptance": suffix_verdict,
            "complete_round0_seed": False,
            "prefix_safety_verified_under_current_contact_identity_schema": False,
            "maximum_physical_prefix_actions": MAX_PREFIX_ACTIONS,
            "archived_route_source": cached_route["events_path"],
            "archived_route_qpos_path_sha256": cached_route["qpos_path_sha256"],
            "archived_route_eef_reference_path_sha256": cached_route["eef_reference_path_sha256"],
            "planner_search_repeated": False,
            "required_next_step_for_complete_seed": (
                "replay_and_audit_one_continuous_trajectory_from_the_verified_raw_t10_state;"
                "do_not_concatenate_this_snapshot_suffix_as_if_it_were_the_missing_prefix"
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
            "snapshot_restore_was_strictly_attested": bool(backend is not None and backend.latest_data),
        }
        return result_json
    finally:
        if backend is not None:
            try:
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
            result_json["elapsed_seconds"] = time.monotonic() - start_time
            _write_json(output_dir / "suffix_result.json", result_json)
            if result_json.get("task_success") is True:
                artifacts.event({"event": "native_task_success", "timestep": result_json.get("final_episode_timestep")})


def main() -> int:
    args = _parser()
    result = run(args)
    print(json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.get("task_success") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
