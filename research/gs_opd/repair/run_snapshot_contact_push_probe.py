#!/usr/bin/env python3
"""Branch a real MuJoCo Snapshot and probe a short contact-driven goal push.

The probe never writes object or joint state. It only issues ordinary OSC pose
actions through the Safety Bridge and records native BDDL status, substep
contacts, observations, and a terminal runtime Snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from research.gs_opd.repair import task_success_first_grasp_search as search
from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-snapshot", required=True, type=Path)
    parser.add_argument("--t0-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--target-predicate", required=True, nargs=3)
    parser.add_argument("--model-sha", required=True)
    parser.add_argument("--bddl-sha", required=True)
    parser.add_argument("--maximum-actions", type=int, default=18)
    parser.add_argument("--demonstration-horizon", type=int, default=1000)
    parser.add_argument("--approach-actions", type=int, default=18)
    parser.add_argument("--approach-stagnation-window-actions", type=int, default=20)
    parser.add_argument("--approach-position-tolerance-m", type=float, default=0.008,
                        help="position tolerance for high transit waypoints; final contact remains tighter")
    parser.add_argument("--approach-gripper-action", type=float, default=-1.0)
    parser.add_argument("--push-gripper-action", type=float, default=-1.0)
    parser.add_argument("--skip-contact-approach", action="store_true")
    parser.add_argument("--parent-run-report", type=Path)
    parser.add_argument("--resume-approach-report", type=Path,
                        help="continue the named waypoint and remaining route from a prior probe branch")
    parser.add_argument("--resume-at-waypoint", type=str)
    parser.add_argument("--approach-waypoint", nargs=4, action="append",
                        metavar=("NAME", "X_M", "Y_M", "Z_M"),
                        help="override with an explicit physical OSC waypoint; repeat in route order")
    parser.add_argument("--contact-standoff-m", type=float, default=0.045)
    parser.add_argument("--contact-height-offset-m", type=float, default=0.015)
    parser.add_argument("--approach-lateral-detour-m", type=float, default=0.0,
                        help="signed lateral offset in the push frame; nonzero routes high around the target")
    parser.add_argument("--approach-clearance-m", type=float, default=0.10)
    parser.add_argument("--retain-source-orientation", action="store_true",
                        help="keep the live Snapshot wrist orientation for a side-of-gripper push")
    parser.add_argument("--goal-world-x-offset-m", type=float, default=0.0,
                        help="shift the requested point within the native region along world X")
    parser.add_argument("--goal-region-local-offset-m", nargs=2, type=float,
                        help="shift the requested point along the native region's local X and Y axes")
    parser.add_argument("--goal-region-object-margin-m", type=float, default=0.035)
    parser.add_argument("--push-chunk-m", type=float, default=0.012)
    parser.add_argument("--push-distance-m", type=float, default=0.072)
    args = parser.parse_args()

    if (
        args.maximum_actions < 1 or args.approach_actions < 1
        or args.approach_stagnation_window_actions < 0
        or (0 < args.approach_stagnation_window_actions < 2)
        or not 0.002 <= args.approach_position_tolerance_m <= 0.02
        or bool(args.resume_approach_report) != bool(args.resume_at_waypoint)
        or not 1000 <= args.demonstration_horizon <= search.MAX_DEMONSTRATION_HORIZON
        or not -1.0 <= args.approach_gripper_action <= 1.0
        or not -1.0 <= args.push_gripper_action <= 1.0
        or not 0.01 <= args.contact_standoff_m <= 0.12
        or not -0.03 <= args.contact_height_offset_m <= 0.10
        or abs(args.approach_lateral_detour_m) > 0.25
        or not 0.04 <= args.approach_clearance_m <= 0.30
        or abs(args.goal_world_x_offset_m) > 0.20
        or not 0.0 <= args.goal_region_object_margin_m <= 0.10
        or (args.goal_region_local_offset_m is not None
            and (len(args.goal_region_local_offset_m) != 2
                 or not np.isfinite(args.goal_region_local_offset_m).all()
                 or max(abs(x) for x in args.goal_region_local_offset_m) > 0.20))
        or (args.approach_waypoint is not None
            and (args.resume_approach_report is not None
                 or any(not np.isfinite([float(x) for x in row[1:]]).all()
                        for row in args.approach_waypoint)))
        or args.push_chunk_m <= 0 or args.push_distance_m <= 0
    ):
        raise ValueError("push_probe_limits_must_be_positive")
    manifest = json.loads(args.t0_manifest.read_text(encoding="utf-8"))
    scene = manifest.get("scene_metadata", {})
    identity = {
        "benchmark": args.benchmark,
        "level": args.level,
        "task_id": args.task_id,
        "seed": 0,
        "init_state": 0,
        "model_sha": args.model_sha,
        "bddl_sha": args.bddl_sha,
        "instruction": args.instruction,
    }
    for key, expected in {
        "benchmark": identity["benchmark"], "level": identity["level"],
        "task_id": identity["task_id"], "environment_seed": 0,
        "init_state_index": 0, "model_geometry_sha256": identity["model_sha"],
        "bddl_sha256": identity["bddl_sha"], "instruction": identity["instruction"],
    }.items():
        if scene.get(key) != expected:
            raise ValueError(f"t0_manifest_identity_mismatch:{key}:{scene.get(key)}")

    resume = search._verified_resume_runtime_snapshot(
        args.resume_snapshot, benchmark=identity["benchmark"], level=identity["level"],
        task_id=identity["task_id"], seed=0, init_state=0,
        model_sha=identity["model_sha"], bddl_sha=identity["bddl_sha"],
        instruction=identity["instruction"],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source_resume_runtime_snapshot.pkl").write_bytes(resume["payload"])
    (args.output / "source_snapshot_manifest.json").write_text(
        json.dumps({
            "schema_version": "gs_opd_runtime_snapshot_manifest.v1",
            "scene_metadata": resume["scene_metadata"],
            "snapshot": {"path": str(args.resume_snapshot.resolve()),
                         "payload_sha256": resume["payload_sha256"],
                         "integration_state_sha256": resume["integration_state_sha256"]},
            "t0_manifest": str(args.t0_manifest.resolve()),
            "t0_manifest_sha256": sha256(args.t0_manifest.read_bytes()),
            "physical_ancestor_records": "referenced by prior run reports; Snapshot is an exact state branch",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    bridge_args = SimpleNamespace(
        benchmark=args.benchmark, level=args.level, task_id=args.task_id,
        init_state=0, seed=0, camera_resolution=256,
        demonstration_horizon=args.demonstration_horizon, port=free_port(),
    )
    bridge = search.BridgeSession(
        ("127.0.0.1", bridge_args.port), args.output / "safety_bridge.log", bridge_args,
    )
    records: list[dict] = []
    frames: list[dict] = []
    report: dict = {
        "schema_version": "gs_opd_snapshot_contact_push_probe.v1",
        "status": "in_progress",
        "hypothesis": "a bounded real OSC surface push from this exact snapshot can move the target toward its native region without paying for a new grasp prefix",
        "identity": identity,
        "resume_snapshot": {
            "path": str(args.resume_snapshot.resolve()),
            "payload_sha256": resume["payload_sha256"],
            "integration_state_sha256": resume["integration_state_sha256"],
        },
        "configured_action_budget": args.maximum_actions,
        "demonstration_horizon": args.demonstration_horizon,
        "approach_action_budget": args.approach_actions,
        "approach_gripper_action": args.approach_gripper_action,
        "push_gripper_action": args.push_gripper_action,
        "actions_executed": 0,
        "safety_certified": False,
        "contact_substep_clearance_sweeps": "disabled; all physical contact substeps retained",
        "parent_run_report": str(args.parent_run_report.resolve()) if args.parent_run_report else None,
        "started_at_epoch": time.time(),
    }
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        if reset.get("scene_metadata", {}).get("model_geometry_sha256") != args.model_sha:
            raise RuntimeError("reset_model_sha_mismatch")
        if reset.get("scene_metadata", {}).get("bddl_sha256") != args.bddl_sha:
            raise RuntimeError("reset_bddl_sha_mismatch")
        restored = search._restore_initial(
            bridge, resume["payload"], resume["snapshot_state"]["mujoco_integration_state"],
            args.model_sha,
        )
        obs = bridge.request({"op": "observe", "include_snapshot_state": True})
        restored_state = search._snapshot_state_vector(obs.get("snapshot_state", {}))
        if not np.array_equal(restored_state, resume["snapshot_state"]["mujoco_integration_state"]):
            raise RuntimeError("resume_integration_state_not_exact")
        initial_native = bridge.request({"op": "native_task_status"})
        if initial_native.get("status") != "complete" or initial_native.get("task_success") is not False:
            raise RuntimeError(f"resume_native_state_invalid:{initial_native}")
        if int(initial_native.get("timestep", -1)) + args.maximum_actions > args.demonstration_horizon:
            raise ValueError("probe_budget_exceeds_configured_demonstration_horizon")
        context = bridge.request({
            "op": "task_success_first_context", "expected_instruction": args.instruction,
            "target_predicate": args.target_predicate,
        })
        if context.get("status") != "complete" or context.get("native_goal_value") is not False:
            raise RuntimeError(f"active_target_context_invalid:{context}")
        target_binding = dict(context["target_binding"])
        if target_binding.get("root_body_id") is None:
            raise RuntimeError("active_target_has_no_root_body")
        # The context operation names this field root_body_id; the shared
        # physical recorder and trajectory tracker use the canonical body_id.
        target_binding["body_id"] = int(target_binding["root_body_id"])
        target_binding["body_name"] = str(target_binding["root_body_name"])
        controller = bridge.request({"op": "repair_context"})
        if controller.get("status") not in {"complete", "available"}:
            raise RuntimeError(f"controller_context_unavailable:{controller}")
        search._advance_pose.controller_fields = controller["robots"][0]["fields"]
        search.configure_stage_a_contact_only_substep_audit(True)
        monitor = GraspStabilityMonitor(required_samples=3)
        policy = {}
        candidate_id = "snapshot_native_goal_region_contact_push"
        status = bridge.request({"op": "native_goal_query", "predicate": args.target_predicate})
        snapshot_state = obs["snapshot_state"]
        target_object = search._object_state(snapshot_state, int(target_binding["root_body_id"]))
        target_position = np.asarray(target_object["root_pos"], dtype=np.float64)
        region = status.get("region", {})
        region_position = np.asarray(region.get("site_position_world_m", []), dtype=np.float64)
        if region_position.shape != (3,) or not np.isfinite(region_position).all():
            raise RuntimeError(f"native_target_region_center_unavailable:{region}")
        region_half_extent = np.asarray(region.get("site_size_m", [0.0, 0.0]), dtype=np.float64)
        applied_goal_x_offset = float(args.goal_world_x_offset_m)
        applied_local_goal_offset = None
        if args.goal_region_local_offset_m is not None:
            site_rotation = np.asarray(region.get("site_rotation_world", []), dtype=np.float64)
            if site_rotation.shape != (3, 3) or not np.isfinite(site_rotation).all():
                raise RuntimeError("native_region_rotation_unavailable_for_local_goal_offset")
            applied_local_goal_offset = np.asarray(args.goal_region_local_offset_m, dtype=np.float64)
            if region_half_extent.shape[0] >= 2:
                max_center_offsets = np.maximum(
                    0.0, region_half_extent[:2] - args.goal_region_object_margin_m,
                )
                applied_local_goal_offset = np.clip(
                    applied_local_goal_offset, -max_center_offsets, max_center_offsets,
                )
            goal_xy = region_position[:2] + site_rotation[:2, :2] @ applied_local_goal_offset
        else:
            if region_half_extent.shape[0] >= 1 and applied_goal_x_offset:
                # Keep the chosen object center within the native region's
                # reported X span with a margin for the soup collision geometry.
                maximum_center_offset = max(0.0, float(region_half_extent[0]) - 0.045)
                applied_goal_x_offset = float(np.clip(
                    applied_goal_x_offset, -maximum_center_offset, maximum_center_offset,
                ))
            goal_xy = region_position[:2] + np.array([applied_goal_x_offset, 0.0])
        direction = goal_xy - target_position[:2]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-5:
            raise RuntimeError("target_xy_already_at_region_center; contact_push_direction_undefined")
        direction /= norm
        lateral_axis = np.array([-direction[1], direction[0]], dtype=np.float64)
        report.update({
            "resume_native_task_status": initial_native,
            "target_binding": target_binding,
            "native_target_region": region,
            "initial_target_root_pose": target_object,
            "initial_eef_pose": snapshot_state["robot_states"][0],
            "push_direction_world_xy": direction.tolist(),
            "requested_goal_point_world_xy": goal_xy.tolist(),
            "requested_goal_world_x_offset_m": args.goal_world_x_offset_m,
            "applied_goal_world_x_offset_m": applied_goal_x_offset,
            "requested_goal_region_local_offset_m": args.goal_region_local_offset_m,
            "applied_goal_region_local_offset_m": (
                applied_local_goal_offset.tolist() if applied_local_goal_offset is not None else None
            ),
            "approach_lateral_detour_m": args.approach_lateral_detour_m,
            "approach_clearance_m": args.approach_clearance_m,
            "retain_source_orientation": args.retain_source_orientation,
            "region_xy_half_extents_m": region_half_extent[:2].tolist(),
            "initial_native_goal_query": status,
            "push_chunk_m": args.push_chunk_m,
            "push_distance_m": args.push_distance_m,
            "contact_standoff_m": args.contact_standoff_m,
            "contact_height_offset_m": args.contact_height_offset_m,
            "native_task_status_after_each_action": True,
        })
        start = bridge.request({"op": "export_runtime_snapshot"})
        if start.get("status") != "complete" or not isinstance(start.get("payload"), bytes):
            raise RuntimeError("branch_start_export_failed")
        (args.output / "branch_start_runtime_snapshot.pkl").write_bytes(start["payload"])
        action_counter = [0]
        initial_eef = snapshot_state["robot_states"][0]
        current_position = np.asarray(initial_eef["eef_pos"], dtype=np.float64)
        current_quaternion = np.asarray(initial_eef["eef_quat"], dtype=np.float64)
        initial_target_xy = target_position[:2].copy()
        contact_pose = current_position.copy()
        contact_pose[:2] = target_position[:2] - direction * args.contact_standoff_m
        contact_pose[2] = target_position[2] + args.contact_height_offset_m
        approach_budget = min(args.approach_actions, args.maximum_actions)
        def target_finger_contact_seen() -> bool:
            if not records:
                return False
            fingers = records[-1].get("grasp_state", {}).get("finger_contacts", {})
            return bool(fingers.get("left") or fingers.get("right"))
        def target_robot_contact_seen() -> bool:
            if not records:
                return False
            contact_rows = records[-1].get("contacts", {}).get("contacts", [])
            target_id = int(target_binding["body_id"])
            for contact in contact_rows if isinstance(contact_rows, list) else []:
                body1 = contact.get("body1", {})
                body2 = contact.get("body2", {})
                geom1 = str(contact.get("geom1", {}).get("name", ""))
                geom2 = str(contact.get("geom2", {}).get("name", ""))
                body1_id = int(body1.get("id", -1))
                body2_id = int(body2.get("id", -1))
                robot1 = geom1.startswith(("robot0_", "gripper0_"))
                robot2 = geom2.startswith(("robot0_", "gripper0_"))
                if (robot1 and body2_id == target_id) or (robot2 and body1_id == target_id):
                    return True
            return False
        def target_contact_seen() -> bool:
            return target_finger_contact_seen() or target_robot_contact_seen()

        def target_pusher_contact_seen() -> bool:
            if not records:
                return False
            contact_rows = records[-1].get("contacts", {}).get("contacts", [])
            target_id = int(target_binding["body_id"])
            for contact in contact_rows if isinstance(contact_rows, list) else []:
                body1 = contact.get("body1", {})
                body2 = contact.get("body2", {})
                geom1 = str(contact.get("geom1", {}).get("name", ""))
                geom2 = str(contact.get("geom2", {}).get("name", ""))
                body1_id = int(body1.get("id", -1))
                body2_id = int(body2.get("id", -1))
                if body1_id == target_id and geom2.startswith(("robot0_", "gripper0_")):
                    robot_geom = geom2
                elif body2_id == target_id and geom1.startswith(("robot0_", "gripper0_")):
                    robot_geom = geom1
                else:
                    continue
                # The planned push face is the gripper palm. Open fingers may
                # brush the mug while the source grasp is being cleared; count
                # that as acquisition only near the actual pusher waypoint.
                if robot_geom == "gripper0_hand_collision":
                    return True
                if "finger" in robot_geom:
                    live_eef = np.asarray(
                        records[-1].get("state", {}).get("robot0_eef_pos", []),
                        dtype=np.float64,
                    )
                    if live_eef.shape == (3,) and np.isfinite(live_eef).all():
                        if float(np.linalg.norm(live_eef - contact_pose)) <= 0.025:
                            return True
            return False

        initial_grasp = bridge.request({
            "op": "grasp_state",
            "carried_binding": target_binding,
            "expected_instruction": args.instruction,
        })
        # A push branch may resume from a Snapshot that still holds the target.
        # Those source finger contacts are not evidence that the newly planned
        # pusher reached its contact pose. Keep retreat active until the real
        # opening/retreat actions clear all target contacts.
        source_grasp_contacts_pending = [
            initial_grasp.get("grasp_confirmed") is True
            or initial_grasp.get("finger_contacts", {}).get("left") is True
            or initial_grasp.get("finger_contacts", {}).get("right") is True
        ]

        def approach_contact_seen() -> bool:
            if source_grasp_contacts_pending[0]:
                if records and not target_contact_seen():
                    source_grasp_contacts_pending[0] = False
                if source_grasp_contacts_pending[0]:
                    return False
            return target_pusher_contact_seen()

        report["initial_target_grasp_state"] = initial_grasp
        report["source_grasp_contacts_pending_release"] = source_grasp_contacts_pending[0]

        contact_quaternion = current_quaternion.copy()
        if args.approach_lateral_detour_m and not args.retain_source_orientation:
            # The gripper's local +Z is its approach axis. Orient it toward the
            # push direction while keeping the open jaw axis approximately
            # vertical, so first contact is made by the front of the tool.
            push_axis = np.r_[direction, 0.0]
            world_up = np.array([0.0, 0.0, 1.0])
            jaw_axis = world_up - push_axis * float(np.dot(world_up, push_axis))
            jaw_axis /= max(float(np.linalg.norm(jaw_axis)), 1e-12)
            local_x_world = np.cross(jaw_axis, push_axis)
            local_x_world /= max(float(np.linalg.norm(local_x_world)), 1e-12)
            rotation_matrix = np.column_stack((local_x_world, jaw_axis, push_axis))
            quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
            contact_quaternion = quat_xyzw[[3, 0, 1, 2]]
        contact_approach_ready = False
        if args.skip_contact_approach:
            report["contact_approach"] = {
                "status": "skipped_from_exact_physical_contact_snapshot",
                "actions": 0,
                "target_finger_contact_seen_at_branch_start": target_finger_contact_seen(),
            }
            contact_approach_ready = True
        elif float(np.linalg.norm(contact_pose - current_position)) > 0.008:
            detour = lateral_axis * args.approach_lateral_detour_m
            high_z = max(
                float(current_position[2] + args.approach_clearance_m),
                float(contact_pose[2] + args.approach_clearance_m),
            )
            behind_distance = max(args.contact_standoff_m + 0.05, 0.06)
            behind_xy = target_position[:2] - direction * behind_distance
            waypoints = []
            if args.approach_lateral_detour_m:
                waypoints = [
                    (np.array([current_position[0], current_position[1], high_z]), current_quaternion,
                     "lift_above_scene"),
                    (np.array([current_position[0], current_position[1], high_z]), contact_quaternion,
                     "align_open_tool_to_push_axis"),
                    (np.array([current_position[0] + detour[0], current_position[1] + detour[1], high_z]), contact_quaternion,
                     "lateral_high_detour"),
                    (np.array([behind_xy[0] + detour[0], behind_xy[1] + detour[1], high_z]), contact_quaternion,
                     "transit_behind_target_high"),
                    (np.array([behind_xy[0], behind_xy[1], high_z]), contact_quaternion,
                     "align_behind_target_high"),
                    (np.array([behind_xy[0], behind_xy[1], contact_pose[2]]), contact_quaternion,
                     "descend_behind_target"),
                    (contact_pose.copy(), contact_quaternion,
                     "acquire_pusher_contact"),
                ]
            else:
                waypoints = [(contact_pose.copy(), current_quaternion, "direct_contact_approach")]
            if args.resume_approach_report:
                previous = json.loads(args.resume_approach_report.read_text(encoding="utf-8"))
                if previous.get("identity") != identity:
                    raise ValueError("resume_approach_report_scene_identity_mismatch")
                previous_direction = np.asarray(previous.get("push_direction_world_xy", []), dtype=np.float64)
                previous_target_position = np.asarray(
                    previous.get("initial_target_root_pose", {}).get("root_pos", []),
                    dtype=np.float64,
                )
                if previous_direction.shape != (2,) or previous_target_position.shape != (3,):
                    raise ValueError("resume_approach_report_has_no_target_route_anchor")
                direction_angle_degrees = float(np.rad2deg(np.arccos(np.clip(
                    float(np.dot(previous_direction, direction)), -1.0, 1.0,
                ))))
                target_xy_shift = target_position[:2] - previous_target_position[:2]
                target_xy_shift_m = float(np.linalg.norm(target_xy_shift))
                # Physical continuation can move a freely supported target by
                # a few micrometers while the arm clears its old grasp. Allow
                # only a small, auditable route rebase; material object motion
                # or a materially different push direction still requires a
                # fresh approach plan.
                if direction_angle_degrees > 1.0 or target_xy_shift_m > 0.01:
                    raise ValueError(
                        "resume_approach_report_route_changed_beyond_rebase_tolerance:"
                        f"direction_delta_deg={direction_angle_degrees:.6f};"
                        f"target_xy_shift_m={target_xy_shift_m:.6f}"
                    )
                report["resume_approach_route_rebase"] = {
                    "status": "accepted_small_physical_target_drift",
                    "direction_delta_degrees": direction_angle_degrees,
                    "target_xy_shift_m": target_xy_shift_m,
                    "target_xy_shift_world_m": target_xy_shift.tolist(),
                    "maximum_direction_delta_degrees": 1.0,
                    "maximum_target_xy_shift_m": 0.01,
                }
                saved_waypoints = previous.get("contact_approach", {}).get("waypoints", [])
                matching = [i for i, row in enumerate(saved_waypoints)
                            if row.get("phase") == args.resume_at_waypoint]
                if len(matching) != 1:
                    raise ValueError(f"resume_approach_waypoint_not_unique:{args.resume_at_waypoint}")
                waypoints = []
                for row in saved_waypoints[matching[0]:]:
                    pose = np.asarray(row["position_world_m"], dtype=np.float64)
                    if pose.shape == (3,) and np.isfinite(pose).all():
                        pose[:2] += target_xy_shift
                    waypoints.append((pose, contact_quaternion, str(row["phase"])))
                if any(pose.shape != (3,) or not np.isfinite(pose).all()
                       for pose, _quat, _phase in waypoints):
                    raise ValueError("resume_approach_report_has_invalid_waypoint")
            if args.approach_waypoint:
                waypoints = [
                    (np.asarray([float(x) for x in row[1:]], dtype=np.float64),
                     contact_quaternion, str(row[0]))
                    for row in args.approach_waypoint
                ]
                if any(not phase or pose.shape != (3,) or not np.isfinite(pose).all()
                       for pose, _quat, phase in waypoints):
                    raise ValueError("custom_approach_waypoint_invalid")
            report["contact_approach"] = {
                "status": "in_progress",
                "action_budget": approach_budget,
                "target_eef_position_world_m": contact_pose.tolist(),
                "waypoints": [
                    {"position_world_m": pose.tolist(), "phase": name}
                    for pose, _quat, name in waypoints
                ],
                "resume_approach_report": (
                    str(args.resume_approach_report.resolve())
                    if args.resume_approach_report else None
                ),
                "resume_at_waypoint": args.resume_at_waypoint,
                "stagnation_window_actions": args.approach_stagnation_window_actions,
                "noncontact_waypoint_position_tolerance_m": args.approach_position_tolerance_m,
                "native_BDDL_remains_success_oracle": True,
            }
            approach_start_actions = action_counter[0]
            approached = True
            for waypoint_position, waypoint_quaternion, waypoint_phase in waypoints:
                final_contact_phase = waypoint_phase == "acquire_pusher_contact"
                if final_contact_phase and approach_contact_seen():
                    approached = False
                    break
                remaining_approach = approach_budget - (action_counter[0] - approach_start_actions)
                if remaining_approach <= 0:
                    approached = False
                    break
                waypoint_reached = search._advance_pose(
                    bridge, waypoint_position, waypoint_quaternion,
                    f"snapshot_contact_push_{waypoint_phase}", candidate_id,
                    action_counter, remaining_approach, target_binding, args.instruction,
                    policy, monitor, records, frames,
                    stagnation_window_actions=args.approach_stagnation_window_actions,
                    position_tolerance_m=(
                        0.004 if "contact" in waypoint_phase
                        else args.approach_position_tolerance_m
                    ),
                    rotation_tolerance_rad=0.06,
                    max_translation_step_m=0.008,
                    gripper_action=args.approach_gripper_action,
                    stop_condition=(
                        approach_contact_seen if final_contact_phase else None
                    ),
                    planning_tracking={
                        "tracking_mode": "snapshot_contact_push_lateral_detour",
                        "target_position_world_m": target_position.tolist(),
                        "push_direction_world_xy": direction.tolist(),
                        "lateral_axis_world_xy": lateral_axis.tolist(),
                        "contact_pose_world_m": contact_pose.tolist(),
                        "contact_standoff_m": args.contact_standoff_m,
                        "contact_height_offset_m": args.contact_height_offset_m,
                        "physical_collision_response_enabled": True,
                        "safety_certified": False,
                        "waypoint_phase": waypoint_phase,
                    },
                    absolute_action_limit=args.maximum_actions,
                )
                if not waypoint_reached:
                    approached = False
                    break
            report["contact_approach"] = {
                "status": (
                    "first_target_pusher_contact" if target_pusher_contact_seen() else
                    "reached" if approached else
                    "not_reached"
                ),
                "actions": action_counter[0],
                "target_eef_position_world_m": contact_pose.tolist(),
                "target_finger_contact_seen": target_finger_contact_seen(),
                "target_robot_contact_seen": target_robot_contact_seen(),
                "native_BDDL_remains_success_oracle": True,
            }
            contact_approach_ready = approached or target_pusher_contact_seen()
        else:
            contact_approach_ready = target_pusher_contact_seen()
        after_approach = bridge.request({"op": "observe", "include_snapshot_state": True})
        approach_state = after_approach["snapshot_state"]
        approach_target = search._object_state(
            approach_state, int(target_binding["root_body_id"]),
        )
        push_baseline_target_xy = np.asarray(approach_target["root_pos"], dtype=np.float64)[:2]
        push_action_start = action_counter[0]
        approach_contacts = records[-1].get("contacts", {}) if records else {}
        report["contact_approach_live_evidence"] = {
            "target_root_pose": approach_target,
            "eef_pose": approach_state["robot_states"][0],
            "target_finger_contacts": records[-1].get("grasp_state", {}).get(
                "finger_contacts", {}
            ) if records else {},
            "last_contact_sample": approach_contacts,
            "physical_actions_before_push": action_counter[0],
            "push_baseline_target_xy": push_baseline_target_xy.tolist(),
        }
        goal_after_approach = bridge.request({
            "op": "native_goal_query", "predicate": args.target_predicate,
        })
        report["native_target_goal_after_approach"] = goal_after_approach
        max_chunks = (
            int(np.ceil(args.push_distance_m / args.push_chunk_m))
            if contact_approach_ready else 0
        )
        if not contact_approach_ready:
            report["early_stop_reason"] = "push_skipped_because_contact_approach_did_not_reach_pusher_pose"
        no_progress_chunks = 0
        for chunk in range(max_chunks):
            if goal_after_approach.get("value") is True:
                report["status"] = "native_target_goal_reached_after_contact_approach"
                break
            if action_counter[0] >= args.maximum_actions:
                break
            live = bridge.request({"op": "observe", "include_snapshot_state": True})
            live_robot = live["snapshot_state"]["robot_states"][0]
            current_position = np.asarray(live_robot["eef_pos"], dtype=np.float64)
            current_quaternion = np.asarray(live_robot["eef_quat"], dtype=np.float64)
            step_len = min(args.push_chunk_m, args.push_distance_m - chunk * args.push_chunk_m)
            target_position = current_position + np.r_[direction * step_len, 0.0]
            chunk_start_xy = np.asarray(
                search._object_state(live["snapshot_state"], int(target_binding["root_body_id"]))["root_pos"],
                dtype=np.float64,
            )[:2]
            chunk_action_start = action_counter[0]
            remaining = min(12, args.maximum_actions - action_counter[0])

            def chunk_progress_or_success() -> bool:
                if records and records[-1].get("task_success") is True:
                    return True
                if not records or action_counter[0] <= chunk_action_start:
                    return False
                latest_object = search._object_state(
                    records[-1].get("snapshot_state", {}), int(target_binding["root_body_id"]),
                )
                latest_xy = np.asarray(latest_object["root_pos"], dtype=np.float64)[:2]
                return float(np.dot(latest_xy - chunk_start_xy, direction)) >= min(
                    0.010, step_len * 0.80,
                )

            try:
                search._advance_pose(
                    bridge, target_position, current_quaternion,
                    "snapshot_contact_push_to_native_region", candidate_id,
                    action_counter, remaining, target_binding, args.instruction,
                    policy, monitor, records, frames,
                    max_translation_step_m=min(0.008, args.push_chunk_m),
                    position_tolerance_m=0.003, rotation_tolerance_rad=0.06,
                    gripper_action=args.push_gripper_action,
                    stop_condition=chunk_progress_or_success,
                    absolute_action_limit=args.maximum_actions,
                )
            except TimeoutError:
                # Under contact, the object can advance while OSC cannot close
                # the commanded EEF waypoint. Keep the real actions and decide
                # from measured object motion below instead of discarding them.
                pass
            if records and records[-1].get("task_success") is True:
                report["status"] = "native_task_success"
                break
            after = bridge.request({"op": "observe", "include_snapshot_state": True})
            after_target = search._object_state(
                after["snapshot_state"], int(target_binding["root_body_id"])
            )
            after_xy = np.asarray(after_target["root_pos"], dtype=np.float64)[:2]
            progress = float(np.dot(after_xy - initial_target_xy, direction))
            push_progress = float(np.dot(after_xy - push_baseline_target_xy, direction))
            chunk_progress = float(np.dot(after_xy - chunk_start_xy, direction))
            goal_state = bridge.request({"op": "native_goal_query", "predicate": args.target_predicate})
            report.setdefault("chunk_diagnostics", []).append({
                "chunk_index": chunk,
                "actions_executed_total": action_counter[0],
                "target_root_pose": after_target,
                "target_progress_along_push_m": progress,
                "push_progress_since_contact_approach_m": push_progress,
                "chunk_progress_along_push_m": chunk_progress,
                "target_xy_to_region_center_m": float(np.linalg.norm(region_position[:2] - after_xy)),
                "native_goal_query": goal_state,
                "last_grasp_state": records[-1].get("grasp_state") if records else None,
                "last_eef_pose": after["snapshot_state"]["robot_states"][0],
            })
            if goal_state.get("value") is True:
                report["status"] = "native_target_goal_reached"
                goal_after_approach = goal_state
                break
            if action_counter[0] == 0 or not records:
                break
            recent = records[-min(3, len(records)):]
            object_progress = float(np.linalg.norm(after_xy - push_baseline_target_xy))
            live_contacts = records[-1].get("grasp_state", {}).get("finger_contacts", {})
            has_target_contact = target_contact_seen()
            if chunk_progress < 0.0015:
                no_progress_chunks += 1
            else:
                no_progress_chunks = 0
            if no_progress_chunks >= 1 and not has_target_contact:
                report["early_stop_reason"] = "push_contact_lost_without_object_progress"
                break
            if no_progress_chunks >= 1 and action_counter[0] - chunk_action_start >= 8:
                report["early_stop_reason"] = "target_stalled_for_a_full_contact_chunk; change push geometry before spending more actions"
                break
        report["actions_executed"] = action_counter[0]
        report["final_native_task_status"] = bridge.request({"op": "native_task_status"})
        final_goal = bridge.request({"op": "native_goal_query", "predicate": args.target_predicate})
        report["final_native_goal_query"] = final_goal
        terminal = bridge.request({"op": "export_runtime_snapshot"})
        if terminal.get("status") == "complete" and isinstance(terminal.get("payload"), bytes):
            (args.output / "terminal_runtime_snapshot.pkl").write_bytes(terminal["payload"])
            report["terminal_snapshot"] = {
                "path": str(args.output / "terminal_runtime_snapshot.pkl"),
                "payload_sha256": sha256(terminal["payload"]),
                "timestep": terminal.get("snapshot_state", {}).get("timestep"),
            }
        report["action_record_archive"] = search._write_recording(args.output, records, frames)
        report["actions_executed"] = len(records)
        if report["final_native_task_status"].get("task_success") is True:
            report["status"] = "native_task_success"
        elif report["status"] not in {
            "native_target_goal_reached",
            "native_target_goal_reached_after_contact_approach",
        }:
            report["status"] = "bounded_physical_push_failed"
        report["physical_execution"] = {
            "MuJoCo_actions": len(records),
            "captured_contact_substeps": sum(
                int(row.get("contact_monitoring", {}).get("captured_physics_substeps", 0))
                for row in records
            ),
            "all_contact_substeps_complete": all(
                row.get("substep_contact_monitoring_complete") is True for row in records
            ),
            "safety_certified": False,
            "task_success_source": "native LIBERO BDDL task_success and active native_goal_query",
        }
    except Exception as exc:
        report["status"] = "probe_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        try:
            native = bridge.request({"op": "native_task_status"})
            report["final_native_task_status"] = native
            snap = bridge.request({"op": "export_runtime_snapshot"})
            if snap.get("status") == "complete" and isinstance(snap.get("payload"), bytes):
                (args.output / "terminal_runtime_snapshot.pkl").write_bytes(snap["payload"])
                report["terminal_snapshot"] = {
                    "path": str(args.output / "terminal_runtime_snapshot.pkl"),
                    "payload_sha256": sha256(snap["payload"]),
                    "timestep": snap.get("snapshot_state", {}).get("timestep"),
                }
            report["actions_executed"] = len(records)
            report["action_record_archive"] = search._write_recording(args.output, records, frames)
            report["physical_execution"] = {
                "MuJoCo_actions": len(records),
                "captured_contact_substeps": sum(
                    int(row.get("contact_monitoring", {}).get("captured_physics_substeps", 0))
                    for row in records
                ),
                "all_contact_substeps_complete": all(
                    row.get("substep_contact_monitoring_complete") is True for row in records
                ),
                "safety_certified": False,
                "task_success_source": "native LIBERO BDDL task_success and active native_goal_query",
            }
        except Exception as archive_exc:
            report["archive_error"] = f"{type(archive_exc).__name__}: {archive_exc}"
        raise
    finally:
        report["wall_time_seconds"] = time.time() - report["started_at_epoch"]
        report["finished_at_epoch"] = time.time()
        (args.output / "run_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)) + "\n",
            encoding="utf-8",
        )
        bridge.close()
    print(json.dumps({"status": report["status"], "actions": report.get("actions_executed"),
                      "task_success": report.get("final_native_task_status", {}).get("task_success"),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["status"] == "native_task_success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
