#!/usr/bin/env python3
"""Continue a physical pusher route through verified IK joint references.

This runner restores an exact same-Episode RuntimeSnapshot, computes a
read-only endpoint IK path from the live MuJoCo scene, and executes the
resulting bounded joint targets through the Safety Bridge. It never writes
object state or sets MuJoCo qpos directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.gs_opd.repair import task_success_first_grasp_search as search
from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-snapshot", required=True, type=Path)
    parser.add_argument("--t0-manifest", required=True, type=Path)
    parser.add_argument("--parent-run-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--level", required=True, type=int)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--target-predicate", required=True, nargs=3)
    parser.add_argument("--model-sha", required=True)
    parser.add_argument("--bddl-sha", required=True)
    parser.add_argument("--demonstration-horizon", type=int, default=10000)
    parser.add_argument("--maximum-actions", type=int, default=1000)
    parser.add_argument("--maximum-segments", type=int, default=8)
    parser.add_argument("--segment-distance-m", type=float, default=0.08)
    parser.add_argument("--push-direction-world-xy", nargs=2, type=float)
    parser.add_argument("--pose-waypoint-json", type=Path,
                        help="optional list of explicit EEF poses to reach through read-only IK and real joint control")
    parser.add_argument("--select-first-ik-feasible", action="store_true",
                        help="treat explicit poses as read-only alternatives for one physical endpoint")
    parser.add_argument("--continue-after-native-target-goal", action="store_true")
    parser.add_argument("--collision-aware-rrt", action="store_true",
                        help="plan a collision-checked joint-space route before real joint control")
    parser.add_argument("--rrt-start-joint-limit-tolerance-rad", type=float, default=0.0,
                        help=("allow only a small measured source-state hard-stop overshoot; "
                              "the offline start is projected to the model limit"))
    parser.add_argument("--joint-limit-recovery-tolerance-rad", type=float, default=0.0,
                        help=("bounded physical joint-step tolerance for moving an already "
                              "measured near-limit source back inside its model range"))
    parser.add_argument("--rrt-allowed-penetration", nargs=2, action="append",
                        metavar=("OBJECT_ID", "DEPTH_M"), default=[],
                        help=("Stage A planner allowance for bounded geometry overlap with a named "
                              "scene object; physical MuJoCo contacts remain authoritative"))
    parser.add_argument("--rrt-allow-target-support-object", action="append", default=[],
                        metavar="OBJECT_ID",
                        help=("allow the native In target to retain contact with its verified "
                              "region-parent support object during planning; robot-object checks remain enabled"))
    parser.add_argument("--rrt-search-time-budget-s", type=float, default=15.0)
    parser.add_argument("--rrt-max-iterations", type=int, default=400)
    parser.add_argument("--rrt-max-config-checks", type=int, default=5000)
    parser.add_argument("--gripper-action", type=float, default=-1.0,
                        help="physical gripper command sent with each joint-space action")
    parser.add_argument("--ik-solver-max-nfev", type=int, default=500)
    parser.add_argument("--ik-reference-max-step-rad", type=float, default=0.08)
    parser.add_argument("--ik-reference-time-profile", choices=("linear", "smoothstep"), default="linear")
    parser.add_argument("--ik-joint-limit-margin-rad", type=float, default=1e-7)
    parser.add_argument("--joint-tracking-error-limit-rad", type=float, default=0.15)
    parser.add_argument("--rrt-waypoint-action-repeats", type=int, default=3,
                        help="hold each RRT joint waypoint for multiple real actions so the physical controller can track it")
    args = parser.parse_args()

    if (
        args.maximum_actions < 1 or args.maximum_segments < 1
        or not 0.01 <= args.segment_distance_m <= 0.20
        or not 1 <= args.ik_solver_max_nfev <= 1000
        or not 0.01 <= args.ik_reference_max_step_rad <= 0.5
        or not np.isfinite(args.ik_joint_limit_margin_rad)
        or not 1e-7 <= args.ik_joint_limit_margin_rad <= 0.20
        or not 0.01 <= args.joint_tracking_error_limit_rad <= 0.5
        or not 1 <= args.rrt_waypoint_action_repeats <= 100
        or not np.isfinite(args.rrt_start_joint_limit_tolerance_rad)
        or not 0.0 <= args.rrt_start_joint_limit_tolerance_rad <= 0.01
        or not np.isfinite(args.joint_limit_recovery_tolerance_rad)
        or not 0.0 <= args.joint_limit_recovery_tolerance_rad <= 0.01
        or any(
            not row[0] or not np.isfinite(float(row[1])) or not 0.0 <= float(row[1]) <= 0.20
            for row in args.rrt_allowed_penetration
        )
        or len({row[0] for row in args.rrt_allowed_penetration})
        != len(args.rrt_allowed_penetration)
        or len(set(args.rrt_allow_target_support_object))
        != len(args.rrt_allow_target_support_object)
        or (args.rrt_allow_target_support_object and not args.collision_aware_rrt)
        or not np.isfinite(args.rrt_search_time_budget_s)
        or not 0.5 <= args.rrt_search_time_budget_s <= 600.0
        or not 1 <= args.rrt_max_iterations <= 100000
        or not 1 <= args.rrt_max_config_checks <= 100000
        or not 1000 <= args.demonstration_horizon <= search.MAX_DEMONSTRATION_HORIZON
        or not -1.0 <= args.gripper_action <= 1.0
        or (args.push_direction_world_xy is not None and (
            len(args.push_direction_world_xy) != 2
            or not np.isfinite(args.push_direction_world_xy).all()
            or np.linalg.norm(args.push_direction_world_xy) < 1e-8
        ))
    ):
        raise ValueError("joint_push_chain_limits_are_invalid")

    manifest = json.loads(args.t0_manifest.read_text(encoding="utf-8"))
    scene = manifest.get("scene_metadata", {})
    expected_identity = {
        "benchmark": args.benchmark,
        "level": args.level,
        "task_id": args.task_id,
        "environment_seed": 0,
        "init_state_index": 0,
        "model_geometry_sha256": args.model_sha,
        "bddl_sha256": args.bddl_sha,
        "instruction": args.instruction,
    }
    for key, expected in expected_identity.items():
        if scene.get(key) != expected:
            raise ValueError(f"t0_manifest_identity_mismatch:{key}:{scene.get(key)}")

    resume = search._verified_resume_runtime_snapshot(
        args.resume_snapshot,
        benchmark=args.benchmark, level=args.level, task_id=args.task_id,
        seed=0, init_state=0, model_sha=args.model_sha, bddl_sha=args.bddl_sha,
        instruction=args.instruction,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    allowed_penetration_by_object = {
        str(object_id): float(depth)
        for object_id, depth in args.rrt_allowed_penetration
    }
    explicit_waypoints = None
    if args.pose_waypoint_json is not None:
        explicit_waypoints = json.loads(args.pose_waypoint_json.read_text(encoding="utf-8"))
        if not isinstance(explicit_waypoints, list) or not explicit_waypoints:
            raise ValueError("pose_waypoint_json_must_be_nonempty_list")
        if len(explicit_waypoints) > args.maximum_segments and not args.select_first_ik_feasible:
            raise ValueError("pose_waypoint_json_exceeds_maximum_segments")
        for row in explicit_waypoints:
            if not isinstance(row, dict):
                raise ValueError("pose_waypoint_must_be_object")
            pos = np.asarray(row.get("position_world_m"), dtype=np.float64)
            quat = np.asarray(row.get("quaternion_wxyz"), dtype=np.float64)
            if (pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all()
                    or not np.isfinite(quat).all()
                    or abs(float(np.linalg.norm(quat)) - 1.0) > 1e-3
                    or not str(row.get("phase", ""))):
                raise ValueError("pose_waypoint_is_invalid")
    (args.output / "source_resume_runtime_snapshot.pkl").write_bytes(resume["payload"])
    (args.output / "source_snapshot_manifest.json").write_text(
        json.dumps({
            "schema_version": "gs_opd_runtime_snapshot_manifest.v1",
            "scene_metadata": resume["scene_metadata"],
            "snapshot": {
                "path": str(args.resume_snapshot.resolve()),
                "payload_sha256": resume["payload_sha256"],
                "integration_state_sha256": resume["integration_state_sha256"],
            },
            "t0_manifest": str(args.t0_manifest.resolve()),
            "t0_manifest_sha256": _sha(args.t0_manifest.read_bytes()),
            "parent_run_report": str(args.parent_run_report.resolve()),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    bridge_args = SimpleNamespace(
        benchmark=args.benchmark, level=args.level, task_id=args.task_id,
        init_state=0, seed=0, camera_resolution=256,
        demonstration_horizon=args.demonstration_horizon,
    )
    bridge = search.BridgeSession(
        ("127.0.0.1", _free_port()), args.output / "safety_bridge.log", bridge_args,
    )
    records: list[dict] = []
    frames: list[dict] = []
    report: dict = {
        "schema_version": "gs_opd_snapshot_ik_joint_push_chain.v1",
        "status": "in_progress",
        "identity": {
            "benchmark": args.benchmark, "level": args.level,
            "task_id": args.task_id, "seed": 0, "init_state": 0,
            "model_sha": args.model_sha, "bddl_sha": args.bddl_sha,
            "instruction": args.instruction,
        },
        "parent_run_report": str(args.parent_run_report.resolve()),
        "resume_snapshot": {
            "path": str(args.resume_snapshot.resolve()),
            "payload_sha256": resume["payload_sha256"],
            "integration_state_sha256": resume["integration_state_sha256"],
        },
        "planning_is_read_only": True,
        "physical_execution_uses_joint_trajectory_step": True,
        "direct_qpos_or_object_state_writes": False,
        "segment_distance_m": args.segment_distance_m,
        "pose_waypoint_json": str(args.pose_waypoint_json.resolve()) if args.pose_waypoint_json else None,
        "select_first_ik_feasible": bool(args.select_first_ik_feasible),
        "collision_aware_rrt": bool(args.collision_aware_rrt),
        "rrt_start_joint_limit_tolerance_rad": float(
            args.rrt_start_joint_limit_tolerance_rad
        ),
        "joint_limit_recovery_tolerance_rad": float(
            args.joint_limit_recovery_tolerance_rad
        ),
        "rrt_allowed_penetration_by_object": allowed_penetration_by_object,
        "rrt_search_limits": {
            "time_budget_s": float(args.rrt_search_time_budget_s),
            "maximum_iterations": int(args.rrt_max_iterations),
            "maximum_config_checks": int(args.rrt_max_config_checks),
        },
        "gripper_action": float(args.gripper_action),
        "maximum_segments": args.maximum_segments,
        "ik_reference_max_step_rad": args.ik_reference_max_step_rad,
        "ik_reference_time_profile": args.ik_reference_time_profile,
        "ik_joint_limit_margin_rad": args.ik_joint_limit_margin_rad,
        "maximum_actions": args.maximum_actions,
        "segments": [],
        "actions_executed": 0,
        "safety_certified": False,
        "started_at_epoch": time.time(),
    }

    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        reset_meta = reset.get("scene_metadata", {})
        if reset_meta.get("model_geometry_sha256") != args.model_sha:
            raise RuntimeError("reset_model_sha_mismatch")
        if reset_meta.get("bddl_sha256") != args.bddl_sha:
            raise RuntimeError("reset_bddl_sha_mismatch")
        search._restore_initial(
            bridge, resume["payload"], resume["snapshot_state"]["mujoco_integration_state"],
            args.model_sha,
        )
        observed = bridge.request({"op": "observe", "include_snapshot_state": True})
        live_state = observed["snapshot_state"]
        live_vector = search._snapshot_state_vector(live_state)
        if not np.array_equal(live_vector, resume["snapshot_state"]["mujoco_integration_state"]):
            raise RuntimeError("resume_integration_state_not_exact")
        initial_native = bridge.request({"op": "native_task_status"})
        if initial_native.get("status") != "complete" or initial_native.get("task_success") is not False:
            raise RuntimeError(f"resume_native_state_invalid:{initial_native}")
        if int(initial_native.get("timestep", -1)) + args.maximum_actions > args.demonstration_horizon:
            raise ValueError("joint_push_budget_exceeds_episode_horizon")
        context = bridge.request({
            "op": "task_success_first_context",
            "expected_instruction": args.instruction,
            "target_predicate": args.target_predicate,
        })
        if (context.get("status") != "complete"
                or (context.get("native_goal_value") is not False
                    and not (args.continue_after_native_target_goal and context.get("native_goal_value") is True))):
            raise RuntimeError(
                f"active_target_context_invalid:status={context.get('status')};"
                f"goal_value={context.get('native_goal_value')}"
            )
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        geometry = bridge.request({"op": "geometry_inventory"})
        gripper_geometry = bridge.request({"op": "gripper_geometry_inventory"})
        if any(item.get("status") != "complete" for item in (semantic, geometry, gripper_geometry)):
            raise RuntimeError("active_geometry_inventory_incomplete")
        policy = search._build_safety_policy(context, semantic, geometry, gripper_geometry)
        target_object_id = str(context["target_object_id"])
        allowed_target_support_objects = set(args.rrt_allow_target_support_object)
        if allowed_target_support_objects:
            support_query = bridge.request({
                "op": "native_goal_query", "predicate": args.target_predicate,
            })
            region = support_query.get("region", {})
            if (args.target_predicate[0] != "in"
                    or support_query.get("predicate_is_native_goal") is not True):
                raise ValueError("target_support_contact_requires_native_In_goal")
            region_parent = str(region.get("parent_name", ""))
            if (not region_parent
                    or allowed_target_support_objects != {region_parent}):
                raise ValueError(
                    "allowed_target_support_must_match_native_In_region_parent:"
                    f"{region_parent}:{sorted(allowed_target_support_objects)}"
                )
            support_pair_before = len(policy.get("forbidden_object_pairs", []))
            policy["forbidden_object_pairs"] = [
                pair for pair in policy.get("forbidden_object_pairs", [])
                if not (
                    pair.get("object_id_a") == target_object_id
                    and pair.get("object_id_b") in allowed_target_support_objects
                )
            ]
            support_pair_after = len(policy["forbidden_object_pairs"])
            if support_pair_after == support_pair_before:
                raise ValueError("native_target_support_pair_not_present_in_collision_policy")
            report["rrt_allowed_target_support_pairs"] = [{
                "target_object_id": target_object_id,
                "support_object_id": region_parent,
                "native_In_predicate": args.target_predicate,
                "native_In_value": support_query.get(
                    "value", support_query.get("predicate_value")
                ),
                "scope": "target-support object pair only; robot-object and robot-robot checks unchanged",
                "safety_certified": False,
            }]
        target_semantic_binding = semantic.get("bindings", {}).get(target_object_id)
        if not isinstance(target_semantic_binding, dict):
            raise RuntimeError("active_target_has_no_semantic_collision_binding")
        target_grasp_binding = {
            "object_id": target_object_id,
            "body_id": int(target_semantic_binding["root_body_id"]),
            "body_name": str(target_semantic_binding["root_body_name"]),
            "geom_ids": sorted(int(value) for value in target_semantic_binding["root_geom_ids"]),
            "geom_names": list(target_semantic_binding["root_geom_names"]),
        }
        target_support_geom_ids = []
        if allowed_target_support_objects:
            support_semantic_binding = semantic.get("bindings", {}).get(region_parent)
            if not isinstance(support_semantic_binding, dict):
                raise RuntimeError(f"native_In_region_parent_has_no_semantic_binding:{region_parent}")
            support_body_ids = set(int(value) for value in support_semantic_binding.get(
                "collision_body_ids", support_semantic_binding.get("body_ids", []),
            ))
            target_support_geom_ids = sorted({
                int(item["geom_id"])
                for item in geometry.get("geometries", [])
                if item.get("collision_enabled") is True
                and int(item.get("body_id", -1)) in support_body_ids
            })
            if not support_body_ids or not target_support_geom_ids:
                raise RuntimeError(f"native_In_region_parent_collision_geometry_unavailable:{region_parent}")
            policy["allowed_carried_support_object_ids"] = [region_parent]
            policy["allowed_carried_support_contact_geom_ids"] = target_support_geom_ids
            report["rrt_allowed_target_support_pairs"][0]["support_collision_geom_ids"] = target_support_geom_ids
            report["rrt_allowed_target_support_pairs"][0]["carried_geometry_contact_minimum_clearance_m"] = -0.002
        initial_grasp = bridge.request({
            "op": "grasp_state",
            "expected_instruction": args.instruction,
            "carried_binding": target_grasp_binding,
        })
        if (initial_grasp.get("status") not in {"pass", "not_grasped"}
                or initial_grasp.get("physical_state_unchanged") is not True):
            raise RuntimeError(f"target_grasp_state_unavailable:{initial_grasp}")
        report["initial_target_grasp_state"] = initial_grasp
        if initial_grasp.get("grasp_confirmed") is True:
            # The active scene planner already supports carrying a measured,
            # bridge-verified object transform. Populate that binding here so
            # its scratch scene moves the grasped target with the EEF instead
            # of leaving the object behind at its source pose.
            policy["carried_object"] = {
                "object_id": target_object_id,
                "grasp_binding": target_grasp_binding,
                "allowed_contact_pairs": [],
                "allowed_support_contact_pairs":[
                    {
                        "carried_geom_id": int(target_geom_id),
                        "scene_geom_id": int(support_geom_id),
                        "minimum_clearance_m": -0.002,
                    }
                    for target_geom_id in target_grasp_binding["geom_ids"]
                    for support_geom_id in target_support_geom_ids
                ],
            }
            report["carried_object_planning"] = {
                "status": "bridge_verified_physical_grasp",
                "object_id": target_object_id,
                "body_id": target_grasp_binding["body_id"],
                "geometry_ids": target_grasp_binding["geom_ids"],
                "transform_source": "measured_live_eef_to_object_transform",
                "planning_only_object_motion": True,
                "live_object_qpos_or_pose_writes": False,
            }
        else:
            report["carried_object_planning"] = {
                "status": "target_not_physically_held",
                "object_id": target_object_id,
            }
        if args.collision_aware_rrt:
            # Stage A permits safety violations. Keep geometric collision
            # checking enabled while removing the default 10 mm safety margin
            # that would reject a valid no-contact start in the human scene.
            policy = json.loads(json.dumps(policy))
            policy["minimum_clearance_m"] = 0.0
            for group_name in (
                "forbidden_objects", "allowed_target_objects", "other_scene_objects",
            ):
                for group in policy.get(group_name, []):
                    group["minimum_clearance_m"] = 0.0
            known_object_ids = {
                str(group.get("object_id"))
                for group_name in (
                    "forbidden_objects", "allowed_target_objects", "other_scene_objects",
                )
                for group in policy.get(group_name, [])
            }
            unknown_penetration_objects = sorted(
                set(allowed_penetration_by_object) - known_object_ids
            )
            if unknown_penetration_objects:
                raise ValueError(
                    f"rrt_allowed_penetration_objects_are_unbound:{unknown_penetration_objects}"
                )
            report["collision_aware_rrt_clearance_policy"] = {
                "minimum_clearance_m": 0.0,
                "geometric_collision_checks_enabled": True,
                "safety_certified": False,
                "allowed_penetration_by_object": allowed_penetration_by_object,
                "reason": (
                    "Stage A permits bounded named-object overlap in offline planning; "
                    "MuJoCo execution contact remains physical and fully recorded"
                ),
            }
        controller = bridge.request({"op": "repair_context"})
        if controller.get("status") not in {"complete", "available"}:
            raise RuntimeError(f"controller_context_unavailable:{controller}")
        search._record_joint_trajectory_step.controller_fields = controller["robots"][0]["fields"]
        search.configure_stage_a_contact_only_substep_audit(True)
        monitor = GraspStabilityMonitor(required_samples=3)

        target_binding = target_grasp_binding
        if target_binding["body_id"] < 1:
            raise RuntimeError("active_target_root_body_invalid")
        report["target_binding"] = target_binding
        report["resume_native_task_status"] = initial_native
        report["seed_policy_source"] = "current_configuration_plus_bounded_alternate_Panda_elbow_seeds"
        report["segments"] = []

        start_status = bridge.request({
            "op": "native_goal_query", "predicate": args.target_predicate,
        })
        if start_status.get("status") != "complete" or start_status.get("predicate_is_native_goal") is not True:
            raise RuntimeError(f"native_target_predicate_query_invalid:{start_status}")
        native_region = start_status.get("region", {})
        native_center = np.asarray(native_region.get("site_position_world_m", []), dtype=np.float64)
        if native_center.shape != (3,) or not np.isfinite(native_center).all():
            raise RuntimeError("native_target_region_center_unavailable")
        report["native_target_region"] = native_region
        report["initial_native_goal_query"] = start_status
        if str(native_region.get("name", "")).endswith("_heating_region"):
            fixture_context = bridge.request({
                "op": "task_success_first_articulation_context",
                "expected_instruction": args.instruction,
                "predicate": ["close", native_region["parent_name"]],
            })
            report["fixture_aperture_geometry"] = fixture_context.get("closed_door_aperture")
            after_context = bridge.request({"op": "observe", "include_snapshot_state": True})
            unchanged = search._state_sha(search._snapshot_state_vector(
                after_context["snapshot_state"],
            )) == search._state_sha(search._snapshot_state_vector(live_state))
            report["fixture_aperture_geometry_live_state_unchanged"] = unchanged
            if not unchanged:
                raise RuntimeError("read_only_fixture_aperture_query_changed_live_state")
        if args.push_direction_world_xy is None:
            root = search._object_state(live_state, target_binding["body_id"])["root_pos"]
            direction_xy = native_center[:2] - np.asarray(root, dtype=np.float64)[:2]
        else:
            direction_xy = np.asarray(args.push_direction_world_xy, dtype=np.float64)
        direction_xy = direction_xy / max(float(np.linalg.norm(direction_xy)), 1e-12)
        report["push_direction_world_xy"] = direction_xy.tolist()

        action_index = 0
        no_progress_segments = 0
        last_export = bridge.request({"op": "export_runtime_snapshot"})
        if last_export.get("status") != "complete" or not isinstance(last_export.get("payload"), bytes):
            raise RuntimeError("initial_branch_snapshot_export_failed")
        (args.output / "branch_start_runtime_snapshot.pkl").write_bytes(last_export["payload"])

        segment_limit = (
            1 if args.select_first_ik_feasible else
            len(explicit_waypoints) if explicit_waypoints is not None else
            args.maximum_segments
        )
        for segment_index in range(segment_limit):
            live = bridge.request({"op": "observe", "include_snapshot_state": True})
            live_state = live["snapshot_state"]
            robot = live_state["robot_states"][0]
            start_q = np.asarray(robot["joint_qpos"], dtype=np.float64)
            start_eef = np.asarray(robot["eef_pos"], dtype=np.float64)
            start_quat = np.asarray(robot["eef_quat"], dtype=np.float64)
            start_target = search._object_state(live_state, target_binding["body_id"])
            start_root = np.asarray(start_target["root_pos"], dtype=np.float64)
            goal_before = bridge.request({
                "op": "native_goal_query", "predicate": args.target_predicate,
            })
            full_native_before = bridge.request({"op": "native_task_status"})
            if full_native_before.get("task_success") is True:
                report["status"] = "native_task_success"
                break
            if (goal_before.get("value") is True
                    and not args.continue_after_native_target_goal):
                report["status"] = "native_target_goal_reached"
                break
            remaining_actions = args.maximum_actions - action_index
            if remaining_actions < 1:
                break

            if explicit_waypoints is None:
                push_axis_3d = np.r_[direction_xy, 0.0]
                endpoint = start_eef + push_axis_3d * args.segment_distance_m
                endpoint_quat_wxyz = start_quat.copy()
                waypoint_phase = f"push_segment_{segment_index}"
            else:
                waypoint = explicit_waypoints[segment_index]
                endpoint = np.asarray(waypoint["position_world_m"], dtype=np.float64)
                endpoint_quat_wxyz = np.asarray(waypoint["quaternion_wxyz"], dtype=np.float64)
                waypoint_phase = str(waypoint["phase"])
            state_sha = _sha(np.ascontiguousarray(
                search._snapshot_state_vector(live_state), dtype=np.float64,
            ).tobytes())
            alternate_seeds = [
                start_q.tolist(),
                [-0.8, 1.0, 0.8, -1.6, 0.5, 1.8, 0.3],
                [0.0, 0.8, 0.0, -1.8, 0.0, 1.8, 0.0],
                [-0.5, 1.2, 0.4, -2.2, 0.2, 2.0, 0.3],
            ]
            candidate_poses = explicit_waypoints if args.select_first_ik_feasible else [None]
            preflight_alternatives = []
            selected_pose = None
            preflight = None
            for candidate_pose in candidate_poses:
                if candidate_pose is not None:
                    endpoint = np.asarray(candidate_pose["position_world_m"], dtype=np.float64)
                    endpoint_quat_wxyz = np.asarray(candidate_pose["quaternion_wxyz"], dtype=np.float64)
                    waypoint_phase = str(candidate_pose["phase"])
                candidate_preflight = bridge.request({
                    "op": "task_success_first_ik_preflight",
                    "expected_instruction": args.instruction,
                    "model_geometry_sha256": args.model_sha,
                    "start_state_sha256": state_sha,
                    "goal_pose_world": {
                        "position_world_m": endpoint.tolist(),
                        "quaternion_wxyz": endpoint_quat_wxyz.tolist(),
                    },
                    "ik_seed_qpos": alternate_seeds,
                    "ik_selection_seed_qpos": start_q.tolist(),
                    "goal_ik_only": not args.collision_aware_rrt,
                    "planner_config": (
                        {
                            "max_config_checks": args.rrt_max_config_checks,
                            "max_candidate_routes": 1,
                            "search_time_budget_s": args.rrt_search_time_budget_s,
                            "max_iterations": args.rrt_max_iterations,
                            "shortcut_attempts": 50,
                            "minimum_clearance_m": 0.0,
                            "start_joint_limit_tolerance_rad": (
                                args.rrt_start_joint_limit_tolerance_rad
                            ),
                            "allowed_penetration_by_object": allowed_penetration_by_object,
                            "ik_joint_limit_margin_rad": args.ik_joint_limit_margin_rad,
                        }
                        if args.collision_aware_rrt else {
                            "ik_joint_limit_margin_rad": args.ik_joint_limit_margin_rad,
                            "start_joint_limit_tolerance_rad": (
                                args.rrt_start_joint_limit_tolerance_rad
                            ),
                        }
                    ),
                    "ik_solver_max_nfev": args.ik_solver_max_nfev,
                    "jointspace_reference_start_qpos": (
                        None if args.collision_aware_rrt else start_q.tolist()
                    ),
                    "jointspace_reference_max_step_rad": args.ik_reference_max_step_rad,
                    "safety_geometry_policy": policy,
                })
                preflight_alternatives.append({
                    "phase": waypoint_phase,
                    "position_world_m": endpoint.tolist(),
                    "quaternion_wxyz": endpoint_quat_wxyz.tolist(),
                    "preflight": candidate_preflight,
                })
                preflight = candidate_preflight
                accepted_status = (
                    preflight.get("status") == "offline_feasible"
                    if args.collision_aware_rrt else
                    preflight.get("status") == "goal_ik_preflight_passed"
                )
                has_route = (
                    bool(preflight.get("route_candidates"))
                    and isinstance(preflight.get("route_candidates")[0].get(
                        "joint_space_qpos_path_rad"), list)
                    if args.collision_aware_rrt else
                    isinstance(preflight.get("eef_jointspace_reference_path_world"), list)
                )
                if (accepted_status and preflight.get("live_state_unchanged") is True
                        and isinstance(preflight.get("ik_goal_qpos"), list)
                        and has_route):
                    selected_pose = candidate_pose
                    break
            segment = {
                "segment_index": segment_index,
                "episode_timestep_start": full_native_before.get("timestep"),
                "target_root_start_world_m": start_root.tolist(),
                "eef_start_world_m": start_eef.tolist(),
                "eef_endpoint_world_m": endpoint.tolist(),
                "eef_quaternion_wxyz": endpoint_quat_wxyz.tolist(),
                "waypoint_phase": waypoint_phase,
                "start_joint_qpos_rad": start_q.tolist(),
                "ik_preflight": preflight,
                "ik_preflight_alternatives": preflight_alternatives if args.select_first_ik_feasible else None,
                "physical_actions": [],
            }
            if (
                preflight is None
                or preflight.get("status") != (
                    "offline_feasible" if args.collision_aware_rrt
                    else "goal_ik_preflight_passed"
                )
                or preflight.get("live_state_unchanged") is not True
                or not isinstance(preflight.get("ik_goal_qpos"), list)
                or (args.collision_aware_rrt and not preflight.get("route_candidates"))
                or (not args.collision_aware_rrt and not isinstance(
                    preflight.get("eef_jointspace_reference_path_world"), list
                ))
            ):
                segment["status"] = "ik_preflight_failed"
                report["segments"].append(segment)
                report["next_specific_action"] = "reduce endpoint distance or select a physically different contact pose"
                break
            if selected_pose is not None:
                endpoint = np.asarray(selected_pose["position_world_m"], dtype=np.float64)
                endpoint_quat_wxyz = np.asarray(selected_pose["quaternion_wxyz"], dtype=np.float64)
                waypoint_phase = str(selected_pose["phase"])
                segment["eef_endpoint_world_m"] = endpoint.tolist()
                segment["eef_quaternion_wxyz"] = endpoint_quat_wxyz.tolist()
                segment["waypoint_phase"] = waypoint_phase
            goal_q = np.asarray(preflight["ik_goal_qpos"], dtype=np.float64)
            if args.collision_aware_rrt:
                selected_route_index = int(preflight.get("selected_route_index", 0))
                route = next((row for row in preflight["route_candidates"]
                              if int(row.get("route_index", -1)) == selected_route_index),
                             preflight["route_candidates"][0])
                joint_path = np.asarray(route["joint_space_qpos_path_rad"], dtype=np.float64)
                reference_path = route["joint_space_qpos_path_rad"]
                q_waypoints = [row.copy() for row in joint_path[1:]]
                route_reference_waypoint_count = len(q_waypoints) + 1
                if args.rrt_waypoint_action_repeats > 1:
                    q_waypoints = [
                        waypoint.copy()
                        for waypoint in q_waypoints
                        for _ in range(args.rrt_waypoint_action_repeats)
                    ]
                segment["rrt_waypoint_action_repeats"] = args.rrt_waypoint_action_repeats
                segment["physical_action_targets_scheduled"] = len(q_waypoints)
                segment["selected_rrt_route"] = {
                    "route_index": route.get("route_index"),
                    "metrics": route.get("metrics"),
                    "compact_vertex_count": route.get("joint_space_compact_vertex_count"),
                    "dense_sample_count": route.get("joint_space_dense_sample_count"),
                    "maximum_sample_delta_rad": route.get("joint_space_maximum_sample_delta_rad"),
                }
            else:
                reference_path = preflight["eef_jointspace_reference_path_world"]
                step_count = len(reference_path) - 1
                if args.ik_reference_time_profile == "smoothstep":
                    step_count = max(step_count, int(np.ceil(
                        1.5 * np.max(np.abs(goal_q - start_q)) / args.ik_reference_max_step_rad,
                    )))
                fractions = np.arange(1, step_count + 1, dtype=np.float64) / step_count
                if args.ik_reference_time_profile == "smoothstep":
                    fractions = fractions ** 2 * (3.0 - 2.0 * fractions)
                q_waypoints = [
                    start_q + (goal_q - start_q) * fraction
                    for fraction in fractions
                ]
                route_reference_waypoint_count = len(q_waypoints) + 1
            step_count = len(q_waypoints)
            if step_count < 1 or step_count > remaining_actions:
                segment["status"] = "joint_reference_path_outside_action_budget"
                segment["reference_path_waypoints"] = route_reference_waypoint_count
                report["segments"].append(segment)
                break
            max_reference_delta = float(np.max(np.abs(np.diff(
                np.vstack([start_q, *q_waypoints]), axis=0,
            ))))
            if (not args.collision_aware_rrt
                    and max_reference_delta > args.ik_reference_max_step_rad + 1e-8):
                raise RuntimeError("generated_joint_reference_exceeds_preflight_step_bound")
            segment["status"] = "physical_execution_in_progress"
            segment["reference_path_waypoints"] = route_reference_waypoint_count
            segment["joint_reference_max_step_rad"] = max_reference_delta
            segment["ik_goal_qpos_rad"] = goal_q.tolist()
            segment["object_progress_along_goal_m_start"] = 0.0
            report["segments"].append(segment)
            _atomic_json(args.output / "run_report.json", report)
            lost_grasp_states = 0

            for path_index, q_target in enumerate(q_waypoints):
                if action_index >= args.maximum_actions:
                    break
                search._record_joint_trajectory_step(
                    bridge, q_target, f"snapshot_ik_jointspace_{waypoint_phase}", "ik_jointspace_native_goal_push",
                    action_index, target_binding, args.instruction, policy, monitor,
                    records, frames,
                    gripper_action=args.gripper_action,
                    joint_limit_recovery_tolerance_rad=(
                        args.joint_limit_recovery_tolerance_rad
                    ),
                    planning_tracking={
                        "route_source": (
                            "active_scene_jointspace_rrt_connect_dense_path"
                            if args.collision_aware_rrt else
                            f"active_scene_task_success_first_ik_preflight_{args.ik_reference_time_profile}_joint_reference"
                        ),
                        "segment_index": segment_index,
                        "reference_path_index": (
                            path_index // args.rrt_waypoint_action_repeats + 1
                            if args.collision_aware_rrt else path_index + 1
                        ),
                        "reference_path_length": len(reference_path),
                        "waypoint_action_repeat_index": (
                            path_index % args.rrt_waypoint_action_repeats + 1
                            if args.collision_aware_rrt else 1
                        ),
                        "push_direction_world_xy": direction_xy.tolist(),
                        "target_eef_endpoint_world_m": endpoint.tolist(),
                        "native_BDDL_is_success_oracle": True,
                        "safety_certified": False,
                    },
                )
                action_index += 1
                record = records[-1]
                record["native_goal_query"] = bridge.request({
                    "op": "native_goal_query", "predicate": args.target_predicate,
                })
                record["native_task_status"] = bridge.request({"op": "native_task_status"})
                tracking_error = record.get("planning_tracking", {}).get("joint_target_error_l2_rad")
                action_entry = {
                    "timestep": record.get("timestep"),
                    "joint_target_qpos_rad": q_target.tolist(),
                    "actual_joint_target_error_l2_rad": tracking_error,
                    "target_root_world_m": search._object_state(
                        record.get("snapshot_state", {}), target_binding["body_id"],
                    )["root_pos"],
                    "native_goal_value": record["native_goal_query"].get("predicate_value"),
                    "task_success": record["native_task_status"].get("task_success"),
                    "target_contact_pairs": [
                        [row.get("geom1", {}).get("name"), row.get("geom2", {}).get("name")]
                        for row in record.get("contacts", {}).get("contacts", [])
                        if (
                            int(row.get("body1", {}).get("id", -1)) == target_binding["body_id"]
                            or int(row.get("body2", {}).get("id", -1)) == target_binding["body_id"]
                        )
                        and (
                            str(row.get("geom1", {}).get("name", "")).startswith(("robot0_", "gripper0_"))
                            or str(row.get("geom2", {}).get("name", "")).startswith(("robot0_", "gripper0_"))
                        )
                    ],
                }
                segment["physical_actions"].append(action_entry)
                if not record.get("substep_contact_monitoring_complete"):
                    segment["status"] = "incomplete_contact_substep_monitoring"
                    raise RuntimeError("joint_route_contact_substep_monitoring_incomplete")
                if tracking_error is not None and float(tracking_error) > args.joint_tracking_error_limit_rad:
                    segment["status"] = "joint_tracking_diverged"
                    segment["failure_reason"] = f"joint_target_error_l2_rad={tracking_error}"
                    break
                if action_entry["task_success"] is True:
                    segment["status"] = "native_task_success"
                    break
                if action_entry["native_goal_value"] is True and not args.continue_after_native_target_goal:
                    segment["status"] = "native_target_goal_reached"
                    break
                if (report.get("initial_target_grasp_state", {}).get("grasp_confirmed") is True
                        and args.gripper_action > 0.5):
                    lost_grasp_states = (
                        0 if record.get("grasp_state", {}).get("grasp_confirmed") is True
                        else lost_grasp_states + 1
                    )
                    if lost_grasp_states >= 2:
                        segment["status"] = "carried_object_grasp_lost"
                        break
                if record.get("done"):
                    segment["status"] = "environment_done"
                    break

            end_live = bridge.request({"op": "observe", "include_snapshot_state": True})
            end_state = end_live["snapshot_state"]
            end_root = np.asarray(search._object_state(
                end_state, target_binding["body_id"],
            )["root_pos"], dtype=np.float64)
            end_eef = end_state["robot_states"][0]
            progress = float(np.dot(end_root[:2] - start_root[:2], direction_xy))
            segment["episode_timestep_end"] = bridge.request({"op": "native_task_status"}).get("timestep")
            segment["target_root_end_world_m"] = end_root.tolist()
            segment["target_progress_along_goal_m"] = progress
            segment["eef_end_world_m"] = end_eef["eef_pos"]
            segment["end_joint_qpos_rad"] = end_eef["joint_qpos"]
            segment["actions_executed"] = len(segment["physical_actions"])
            segment["eef_endpoint_position_error_m"] = float(np.linalg.norm(
                np.asarray(end_eef["eef_pos"], dtype=np.float64) - endpoint,
            ))
            actual_quat = np.asarray(end_eef["eef_quat"], dtype=np.float64)
            quaternion_dot = abs(float(np.dot(actual_quat, endpoint_quat_wxyz))) / (
                float(np.linalg.norm(actual_quat)) * float(np.linalg.norm(endpoint_quat_wxyz))
            )
            segment["eef_endpoint_rotation_error_rad"] = float(
                2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))
            )
            if segment.get("status") == "physical_execution_in_progress":
                segment["status"] = (
                    "physical_segment_completed"
                    if (segment["eef_endpoint_position_error_m"] <= 0.012
                        and segment["eef_endpoint_rotation_error_rad"] <= 0.08)
                    else "physical_endpoint_tracking_failed"
                )

            exported = bridge.request({"op": "export_runtime_snapshot"})
            if exported.get("status") != "complete" or not isinstance(exported.get("payload"), bytes):
                raise RuntimeError("segment_terminal_snapshot_export_failed")
            snapshot_path = args.output / f"segment_{segment_index:02d}_terminal_runtime_snapshot.pkl"
            snapshot_path.write_bytes(exported["payload"])
            segment["terminal_snapshot"] = {
                "path": str(snapshot_path),
                "payload_sha256": _sha(exported["payload"]),
                "integration_state_sha256": _sha(np.ascontiguousarray(
                    search._snapshot_state_vector(end_state), dtype=np.float64,
                ).tobytes()),
            }
            if segment.get("status") in {
                "native_task_success", "native_target_goal_reached", "joint_tracking_diverged",
                "environment_done", "carried_object_grasp_lost", "physical_endpoint_tracking_failed",
            }:
                break
            no_progress_segments = no_progress_segments + 1 if progress < 0.002 else 0
            if explicit_waypoints is None and no_progress_segments >= 1:
                segment["status"] = "no_native_goal_progress"
                report["next_specific_action"] = "stop this IK contact route and choose a different physical interaction"
                break
            if action_index >= args.maximum_actions:
                break

        report["actions_executed"] = len(records)
        if (explicit_waypoints is not None and len(report["segments"]) == len(explicit_waypoints)
                and all(row.get("status") == "physical_segment_completed" for row in report["segments"])):
            report["pose_route_status"] = "all_explicit_waypoints_physically_reached"
        report["final_native_task_status"] = bridge.request({"op": "native_task_status"})
        report["final_native_goal_query"] = bridge.request({
            "op": "native_goal_query", "predicate": args.target_predicate,
        })
        terminal = bridge.request({"op": "export_runtime_snapshot"})
        if terminal.get("status") == "complete" and isinstance(terminal.get("payload"), bytes):
            terminal_path = args.output / "terminal_runtime_snapshot.pkl"
            terminal_path.write_bytes(terminal["payload"])
            report["terminal_snapshot"] = {
                "path": str(terminal_path),
                "payload_sha256": _sha(terminal["payload"]),
                "integration_state_sha256": _sha(np.ascontiguousarray(
                    search._snapshot_state_vector(terminal.get("snapshot_state", {})), dtype=np.float64,
                ).tobytes()),
            }
        if report.get("status") not in {"native_task_success", "native_target_goal_reached"}:
            if report["final_native_task_status"].get("task_success") is True:
                report["status"] = "native_task_success"
            elif report["final_native_goal_query"].get("predicate_value") is True:
                report["status"] = "native_target_goal_reached_task_incomplete"
            else:
                report["status"] = "physical_joint_push_incomplete"
    except Exception as exc:
        report["status"] = "probe_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if report["segments"] and report["segments"][-1].get("status") == "physical_execution_in_progress":
            report["segments"][-1]["status"] = "physical_execution_aborted"
        try:
            report["final_native_task_status"] = bridge.request({"op": "native_task_status"})
            terminal = bridge.request({"op": "export_runtime_snapshot"})
            if terminal.get("status") == "complete" and isinstance(terminal.get("payload"), bytes):
                terminal_path = args.output / "terminal_runtime_snapshot.pkl"
                terminal_path.write_bytes(terminal["payload"])
                report["terminal_snapshot"] = {
                    "path": str(terminal_path),
                    "payload_sha256": _sha(terminal["payload"]),
                    "integration_state_sha256": _sha(np.ascontiguousarray(
                        search._snapshot_state_vector(terminal.get("snapshot_state", {})), dtype=np.float64,
                    ).tobytes()),
                }
        except Exception as recovery_exc:
            report["terminal_snapshot_error"] = f"{type(recovery_exc).__name__}: {recovery_exc}"
    finally:
        bridge.close()
        report["actions_executed"] = len(records)
        initial_timestep = report.get("resume_native_task_status", {}).get("timestep")
        final_timestep = report.get("final_native_task_status", {}).get("timestep")
        if initial_timestep is not None and final_timestep is not None:
            timestep_advance = int(final_timestep) - int(initial_timestep)
            report["physical_action_continuity"] = {
                "episode_timestep_advance": timestep_advance,
                "saved_action_records": len(records),
                "unrecorded_timestep_advance": timestep_advance - len(records),
                "all_advanced_timesteps_recorded": timestep_advance == len(records),
            }
        report["finished_at_epoch"] = time.time()
        if records:
            report["action_record_archive"] = search._write_recording(args.output, records, frames)
            report["physical_execution"] = {
                "MuJoCo_actions": len(records),
                "captured_contact_substeps": sum(
                    int(row.get("contact_monitoring", {}).get("captured_physics_substeps", 0))
                    for row in records
                ),
                "all_contact_substeps_complete": all(
                    bool(row.get("substep_contact_monitoring_complete")) for row in records
                ),
                "safety_certified": False,
                "task_success_source": "native LIBERO BDDL task_success and active native_goal_query",
            }
        _atomic_json(args.output / "run_report.json", report)

    print(json.dumps({
        "status": report.get("status"),
        "actions": report.get("actions_executed"),
        "timestep": report.get("final_native_task_status", {}).get("timestep"),
        "task_success": report.get("final_native_task_status", {}).get("task_success"),
        "native_goal": report.get("final_native_goal_query", {}).get("predicate_value"),
        "output": str(args.output),
    }))
    return 0 if report.get("status") in {"native_task_success", "native_target_goal_reached", "native_target_goal_reached_task_incomplete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
