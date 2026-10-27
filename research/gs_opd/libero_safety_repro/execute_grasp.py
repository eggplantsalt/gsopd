# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Execute one validated approach and bounded motor closure with live feedback."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from research.gs_opd.libero_safety_repro.keyposes import (
    inverse_transform,
    transform_from_pose,
)
from research.gs_opd.libero_safety_repro.scene_probe import _read_only_rpc
from research.gs_opd.repair.panda_gripper_state import parse_panda_gripper_closure


class BridgeConnection:
    """Adapt the existing connection to the existing trajectory recorder."""

    def __init__(self, connection):
        self.connection = connection

    def request(self, request):
        return _read_only_rpc(self.connection, request)


def actuator_pair(raw):
    """Normalize robosuite's initial one-DoF zero to its formatted pair."""
    if parse_panda_gripper_closure(raw)["status"] != "pass":
        raise ValueError("Panda gripper motor state unavailable")
    values = np.asarray(raw, dtype=float)
    return np.repeat(values, 2) if values.shape == (1,) else values


def classify_contacts(
    sample: dict,
    geoms: dict,
    pads: set,
    target: set,
    allow_grasp: bool,
    forbidden=frozenset(),
) -> dict:
    """Validate identities and allow only verified pad/target robot contacts."""
    if sample.get("status") not in {"contact", "no_contact", "complete"}:
        return {"complete": False, "illegal": [], "pad_ids": []}
    illegal, contacted = [], set()
    for contact in sample.get("contacts", []):
        pair = []
        for key in ["geom1", "geom2"]:
            item = contact[key]
            gid = item["id"]
            if gid not in geoms or geoms[gid]["geom_name"] != item["name"]:
                return {
                    "complete": False,
                    "illegal": illegal,
                    "pad_ids": sorted(contacted),
                }
            pair.append(gid)
        robot = any(geoms[g]["category"] == "robot" for g in pair)
        pad_target = any(a in pads and b in target for a, b in [pair, pair[::-1]])
        forbidden_target = any(
            a in target and b in forbidden for a, b in [pair, pair[::-1]]
        )
        if (robot and not (allow_grasp and pad_target)) or forbidden_target:
            illegal.append(contact)
        if pad_target:
            contacted.update(set(pair) & pads)
    return {"complete": True, "illegal": illegal, "pad_ids": sorted(contacted)}


def main():
    from research.gs_opd.libero_safety_repro import plan_oa_l0t0 as api
    from research.gs_opd.repair.run_human_l0t2_curobo_transport import EpisodeRecorder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-object", action="append", required=True)
    args = parser.parse_args()
    approach = json.loads(args.approach_report.read_text())
    route = next(r for r in approach["routes"] if r["accepted"])
    source = json.loads(gzip.decompress(Path(approach["source_scene"]).read_bytes()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "libero_safety_real_grasp_episode.v1",
        "status": "running",
        "source_approach": str(args.approach_report),
        "candidate_index": route["candidate_index"],
        "budget": {
            "physical_episodes": 1,
            "standard_horizon": 1000,
            "approach_steps_per_stage": 250,
            "closure_steps": 120,
        },
        "phase_events": [],
        "task_success": False,
        "stable_release": False,
        "safety_certified": False,
        "safety_status": "unknown",
        "new_trainable_episodes": 0,
        "pi05_native_action_compatible": False,
    }
    process, log, connection, observation = api._start_bridge(source, args.output_dir)
    bridge = BridgeConnection(connection)
    recorder = EpisodeRecorder(args.output_dir, bridge)
    native_log = gzip.open(args.output_dir / "native_and_safety.jsonl.gz", "wt")
    geoms = {
        g["geom_id"]: g
        for g in source["collision_scene_export"]["geometry_inventory"]["geometries"]
    }
    pads = {
        g["geom_id"]
        for rows in source["target_context"]["fingertip_pad_geometries"].values()
        for g in rows
    }
    target = {
        g["geom_id"] for g in source["target_context"]["target_collision_geometries"]
    }
    bindings = source["semantic_geometry_bindings"]["bindings"]
    forbidden = set()
    for name in args.forbidden_object:
        forbidden.update(bindings[name]["semantic_geom_ids"])
    report["forbidden_objects"] = args.forbidden_object
    report["forbidden_geom_ids"] = sorted(forbidden)
    complete_steps = 0
    illegal_observed = []
    gripper_current_action = np.array([0.0, 0.0])

    def robot():
        return observation["snapshot_state"]["robot_states"][0]

    def scratch_guard(q_target, jaw_target, allow_grasp):
        q = np.asarray(robot()["joint_qpos"])
        jaw = np.asarray(robot()["gripper_qpos"])
        count = max(2, int(np.ceil(np.max(np.abs(q_target - q)) / 0.01)) + 1)
        qs, jaws = [], []
        for alpha in np.linspace(0.0, 1.0, count):
            for beta in np.linspace(0.0, 1.0, 5):
                qs.append((q + alpha * (q_target - q)).tolist())
                jaws.append((jaw + beta * (jaw_target - jaw)).tolist())
        audit = bridge.request(
            {
                "op": "scratch_robot_path_contact_audit",
                "object_id": source["config"]["target_object_id"],
                "arm_joint_states_rad": qs,
                "gripper_joint_states_rad": jaws,
                "fingertip_contact_scope": "all_states" if allow_grasp else "none",
            }
        )
        if (
            audit.get("status") != "complete"
            or not audit.get("physical_state_unchanged")
            or audit.get("illegal_contact_count") != 0
        ):
            path = (
                args.output_dir / f"rejected_next_step_{recorder.action_count:04d}.json"
            )
            path.write_text(json.dumps(api._json_safe(audit), indent=2) + "\n")
            raise RuntimeError(f"next_motor_step_static_safety_gate:{path}")

    def step(q_target, phase, grip=0.0, allow_grasp=False):
        nonlocal observation, complete_steps, gripper_current_action
        if recorder.action_count >= 1000:
            raise RuntimeError("standard_1000_step_horizon_exhausted")
        next_action = np.clip(
            gripper_current_action + np.array([-1.0, 1.0]) * 0.01 * np.sign(grip),
            -1.0,
            1.0,
        )
        jaw_target = np.array(
            [0.02 * (next_action[0] + 1), 0.02 * (next_action[1] - 1)]
        )
        scratch_guard(q_target, jaw_target, allow_grasp)
        observation = bridge.request(
            {
                "op": "joint_trajectory_step",
                "action_index": recorder.action_count,
                "joint_target_qpos_rad": np.asarray(q_target).tolist(),
                "gripper_action": grip,
                "joint_position_kp": 50.0,
            }
        )
        gripper_current_action = next_action
        recorder.record(observation, np.asarray(q_target), phase, grip)
        substeps = observation.get("substep_contacts", [])
        checks = [
            classify_contacts(s, geoms, pads, target, allow_grasp, forbidden)
            for s in substeps
        ]
        complete = len(checks) == 25 and all(s["complete"] for s in checks)
        if complete:
            complete_steps += 1
        illegal = [c for s in checks for c in s["illegal"]]
        illegal_observed.extend(illegal)
        native = bridge.request({"op": "native_task_status"})
        native_log.write(
            json.dumps(
                api._json_safe(
                    {
                        "step": recorder.action_count,
                        "phase": phase,
                        "native": native,
                        "raw_contact_monitoring_complete": complete,
                        "illegal_contact_record_count": len(illegal),
                    }
                )
            )
            + "\n"
        )
        native_log.flush()
        if not complete:
            raise RuntimeError("physical_substep_contact_monitoring_incomplete")
        if illegal:
            raise RuntimeError("physical_illegal_contact_observed")
        if recorder.action_count % 50 == 0:
            recorder.snapshot(f"step_{recorder.action_count:04d}")

    try:
        recorder.snapshot("initial")
        # A motor command is relative to gripper.current_action, not physical qpos.
        binding = source["target_context"]["target_binding"]
        grasp_state = bridge.request(
            {
                "op": "grasp_state",
                "carried_binding": {
                    "object_id": binding["object_id"],
                    "body_id": binding["root_body_id"],
                    "body_name": binding["root_body_name"],
                    "geom_ids": binding["root_semantic_geom_ids"],
                    "geom_names": [
                        geoms[g]["geom_name"] for g in binding["root_semantic_geom_ids"]
                    ],
                },
                "expected_instruction": source["config"]["instruction"],
            }
        )
        report["initial_gripper_state_query"] = grasp_state
        gripper_current_action = actuator_pair(grasp_state["gripper_current_action"])
        report["initial_gripper_motor_action"] = gripper_current_action.tolist()
        if not np.array_equal(
            observation["snapshot_state"]["mujoco_integration_state"],
            source["initial_observation"]["snapshot_state"]["mujoco_integration_state"],
        ):
            raise RuntimeError("live_initial_state_differs_from_planned_source")
        for stage in route["stages"]:
            start_count = recorder.action_count
            path = np.asarray(stage["curobo"]["valid_joint_trajectory_rad"])
            for index, q_target in enumerate(path[1:]):
                endpoint = index == len(path) - 2
                tolerance = 0.002 if endpoint else 0.015
                for _ in range(20):
                    error = float(
                        np.max(np.abs(np.asarray(robot()["joint_qpos"]) - q_target))
                    )
                    if error <= tolerance:
                        break
                    if recorder.action_count - start_count >= 250:
                        raise RuntimeError(
                            f"{stage['phase']}:stage_action_budget_exhausted"
                        )
                    step(q_target, stage["phase"])
                else:
                    raise RuntimeError(
                        f"{stage['phase']}:waypoint_feedback_stalled:{index}:error={error}"
                    )
            actual_eef = transform_from_pose(robot()["eef_pos"], robot()["eef_quat"])
            actual_hand = (
                inverse_transform(
                    np.asarray(source["planner_frames"]["world_base_transform_m"])
                )
                @ actual_eef
                @ np.asarray(source["planner_frames"]["eef_to_panda_hand_transform"])
            )
            error = api._pose_error(
                np.asarray(stage["target_T_base_panda_hand"]), actual_hand
            )
            report["phase_events"].append(
                {
                    "phase": stage["phase"],
                    "physical_actions": recorder.action_count - start_count,
                    "endpoint_error": error,
                }
            )
            recorder.snapshot(stage["phase"])
            if (
                error["position_error_m"] > 0.003
                or error["orientation_error_rad"] > 0.025
            ):
                raise RuntimeError(
                    f"{stage['phase']}:physical_tool_endpoint_not_reached"
                )
        q_hold = np.asarray(robot()["joint_qpos"])
        stable = 0
        for _ in range(120):
            step(q_hold, "close", grip=1.0, allow_grasp=True)
            current = classify_contacts(
                observation["contacts"], geoms, pads, target, True
            )
            stable = stable + 1 if set(current["pad_ids"]) == pads else 0
            if stable >= 5:
                report["stable_bilateral_contact"] = True
                r = robot()
                o = observation["snapshot_state"]["object_states"][
                    f"objects_dict:{source['config']['target_object_id']}"
                ]
                report["measured_T_object_eef"] = (
                    inverse_transform(
                        transform_from_pose(o["root_pos"], o["root_quat"])
                    )
                    @ transform_from_pose(r["eef_pos"], r["eef_quat"])
                ).tolist()
                report["status"] = "bilateral_grasp_ready_for_live_attachment_planning"
                recorder.snapshot("bilateral_grasp")
                break
        else:
            raise RuntimeError("motor_closure_budget_without_stable_bilateral_contact")
    except Exception as exc:
        report.update(status="stopped", reason=f"{type(exc).__name__}: {exc}")
    finally:
        report.update(
            physical_actions=recorder.action_count,
            captured_physics_substeps=recorder.recorded_physics_substeps,
            raw_contact_complete_steps=complete_steps,
            illegal_contact_records=illegal_observed,
            task_success=bool(observation.get("task_success")),
            safety_monitoring_complete=(
                recorder.action_count > 0 and complete_steps == recorder.action_count
            ),
            terminal_timestep=observation.get("timestep"),
        )
        report["terminal_snapshot"] = recorder.snapshot("terminal")
        report["native_task_status"] = bridge.request({"op": "native_task_status"})
        (args.output_dir / "execution_report.json").write_text(
            json.dumps(api._json_safe(report), indent=2) + "\n"
        )
        native_log.close()
        recorder.close()
        connection.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except Exception:
            process.kill()
            process.wait()
        log.close()


if __name__ == "__main__":
    main()
