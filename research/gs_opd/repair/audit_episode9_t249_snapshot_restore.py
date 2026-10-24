# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only, fail-closed restore probe for the archived Episode 9 t=249 state."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.repair.automatic_recovery_runtime import BridgeProcess


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "task_success_first_safety_reconstruction_l0t2_20261023_attempt1"
)
DEFAULT_PYTHON = Path("/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python")


class _PythonObjectState:
    """Compatibility placeholder for helper states pickled by the Bridge."""


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _summarize_episode9_history(episode_dir: Path) -> dict[str, Any]:
    """Summarize archived action-level and per-substep evidence without replay."""
    trace_path = episode_dir / "observation_action_trace.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        timestep_matches = np.flatnonzero(trace["step_timesteps"] == 249)
        if len(timestep_matches) != 1:
            return {
                "status": "unavailable",
                "trace_path": str(trace_path),
                "reason": "unique_timestep_249_not_found",
                "physical_actions": int(len(trace["step_timesteps"])),
            }
        index = int(timestep_matches[0])
        monitors = [json.loads(str(item)) for item in trace["post_contact_monitoring_json"]]
        total_substeps = 0
        incomplete_substeps = 0
        unsafe_substeps = 0
        limit_violations = 0
        forbidden_contact_events = 0
        unresolved_events = 0
        latest_monitor = monitors[index]
        for monitor in monitors:
            geometry = monitor.get("geometry_monitoring", {})
            substeps = geometry.get("substeps", []) if isinstance(geometry, dict) else []
            total_substeps += len(substeps)
            for substep in substeps:
                incomplete_substeps += int(substep.get("status") not in ("complete", "unsafe"))
                unsafe_substeps += int(substep.get("status") == "unsafe")
                limit_violations += len(substep.get("joint_limit_violations", []))
                forbidden_contact_events += len(substep.get("forbidden_contacts", []))
                unresolved_events += len(substep.get("unresolved_contacts", []))
        latest_geometry = latest_monitor.get("geometry_monitoring", {})
        latest_substeps = latest_geometry.get("substeps", []) if isinstance(latest_geometry, dict) else []
        # The historical geometry substep rows contain verdicts and clearances,
        # but not raw per-substep contact identities. Actual contact identities
        # were saved once at the post-control observation in post_contacts_json.
        latest_has_contact_identities = any("contacts" in item for item in latest_substeps)
        return {
            "status": "complete",
            "trace_path": str(trace_path),
            "physical_actions": int(len(trace["step_timesteps"])),
            "first_timestep": int(trace["step_timesteps"][0]),
            "last_timestep": int(trace["step_timesteps"][-1]),
            "timestep_249_trace_row": index,
            "timestep_249_task_phase": str(trace["step_task_phase"][index]),
            "timestep_249_monitor_status": latest_monitor.get("status"),
            "timestep_249_captured_substeps": latest_monitor.get("captured_physics_substeps"),
            "timestep_249_geometry_substeps": len(latest_substeps),
            "timestep_249_bad_geometry_substeps": sum(item.get("status") != "complete" for item in latest_substeps),
            "timestep_249_raw_contact_identities_saved_per_substep": latest_has_contact_identities,
            "total_geometry_substeps": total_substeps,
            "incomplete_geometry_substeps": incomplete_substeps,
            "unsafe_geometry_substeps": unsafe_substeps,
            "joint_limit_violation_events": limit_violations,
            "forbidden_contact_events": forbidden_contact_events,
            "unresolved_contact_events": unresolved_events,
            "timestep_249_post_control_gripper_qpos": trace["post_robot0_gripper_qpos"][index].tolist(),
            "timestep_249_post_control_contacts_json_saved": True,
            "finger_only_prefix_certification": "not_complete_without_per_substep_contact_identities_under_current_rule",
        }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38772)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    episode_dir = args.archive_root / "episode9"
    snapshot_path = episode_dir / "snapshots/main_timestep_000249.pkl"
    run_config = json.loads((episode_dir / "run_config.json").read_text())
    config_path = Path(run_config["config_path"])
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = json.loads(config_path.read_text())
    payload = snapshot_path.read_bytes()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "gs_opd_episode9_t249_reset_order_audit.v1",
        "read_only": True,
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot_bytes": len(payload),
        "expected_model_geometry_sha256": "1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752",
        "bridge_port": int(args.bridge_port),
        "demonstration_horizon": 1800,
        "physical_action_steps": 0,
        "physical_substeps": 0,
        "native_reset_operations": 0,
        "post_restore_observation_requested": False,
    }
    bridge = BridgeProcess(
        args.output_dir, python=args.python, port=args.bridge_port
    )
    try:
        bridge.start(
            benchmark=config["benchmark"],
            level=int(config["level"]),
            task_id=int(config["task_id"]),
            init_state=int(config["init_state"]),
            seed=int(config["environment_seed"]),
            demonstration_horizon=1800,
        )
        pre_observation = bridge.request(
            {"op": "observe", "include_snapshot_state": True}
        )
        pre_fingerprint = bridge.request(
            {"op": "compiled_model_fingerprint_details"}
        )
        report["bridge_startup"] = {
            "timestep": pre_observation.get("timestep"),
            "model_geometry_sha256": pre_observation.get("scene_metadata", {}).get(
                "model_geometry_sha256"
            ),
            "fingerprint_details": pre_fingerprint,
        }

        reset_response = bridge.request({"op": "reset"})
        report["native_reset_operations"] = 1
        post_observation = bridge.request(
            {"op": "observe", "include_snapshot_state": True}
        )
        post_fingerprint = bridge.request(
            {"op": "compiled_model_fingerprint_details"}
        )
        report["after_formal_native_reset"] = {
            "timestep": reset_response.get("timestep"),
            "model_geometry_sha256": reset_response.get("scene_metadata", {}).get(
                "model_geometry_sha256"
            ),
            "fingerprint_details": post_fingerprint,
            "observed_timestep": post_observation.get("timestep"),
        }
        before_fields = pre_fingerprint.get("component_digests", {})
        after_fields = post_fingerprint.get("component_digests", {})
        report["startup_vs_reset_component_changes"] = {
            name: {
                "before": before_fields.get(name),
                "after": after_fields.get(name),
            }
            for name in sorted(set(before_fields) | set(after_fields))
            if before_fields.get(name) != after_fields.get(name)
        }
        after_hash = report["after_formal_native_reset"]["model_geometry_sha256"]
        report["archive_fingerprint_match_after_reset"] = (
            after_hash == report["expected_model_geometry_sha256"]
        )
        report["reset_timestep_zero_verified"] = (
            reset_response.get("timestep") == 0
            and post_observation.get("timestep") == 0
        )
        if not report["reset_timestep_zero_verified"]:
            report["restore_status"] = "not_attempted_reset_timestep_invalid"
            report["restore_error"] = "native_reset_did_not_leave_timestep_zero"
            report["scene_and_model_validation"] = "failed"
            report["restore_state"] = None
            _write_report(args.output_dir, report)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, default=_json_default))
            return 2
        if not report["archive_fingerprint_match_after_reset"]:
            report["restore_status"] = "not_attempted_post_reset_fingerprint_mismatch"
            report["restore_error"] = (
                "post-reset compiled geometry fingerprint differs from archived "
                "Episode 9 model; strict restore was not attempted"
            )
            report["scene_and_model_validation"] = "failed_at_model_geometry_signature"
            report["restore_state"] = None
            _write_report(args.output_dir, report)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, default=_json_default))
            return 2

        try:
            response = bridge.request(
                {"op": "restore_runtime_snapshot", "payload": payload}
            )
        except RuntimeError as exc:
            report["restore_status"] = "restore_failed"
            report["restore_error"] = str(exc)
            report["scene_and_model_validation"] = "failed"
            report["restore_state"] = None
            _write_report(args.output_dir, report)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, default=_json_default))
            return 2
        report["restore_status"] = response.get("status")
        report["restore_error"] = response.get("error")
        report["scene_and_model_validation"] = (
            "failed_at_model_geometry_signature"
            if response.get("status") == "restore_failed"
            and "model geometry differs" in str(response.get("error", ""))
            else "passed" if response.get("status") == "restored" else "failed"
        )
        if response.get("status") != "restored":
            report["restore_state"] = None
            report["post_restore_observation_requested"] = False
            _write_report(args.output_dir, report)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 2

        report["restore_state"] = response.get("snapshot_state")
        report["post_restore_observation_requested"] = True
        archived_envelope = pickle.loads(payload)
        archived_mujoco_state = np.asarray(
            archived_envelope["snapshot"]["mujoco_state"], dtype=np.float64
        )
        restored_mujoco_state = np.asarray(
            response["snapshot_state"]["mujoco_integration_state"],
            dtype=np.float64,
        )
        report["archived_integration_state_shape"] = list(
            archived_mujoco_state.shape
        )
        report["restored_integration_state_shape"] = list(
            restored_mujoco_state.shape
        )
        report["archived_integration_state_exactly_restored"] = bool(
            archived_mujoco_state.shape == restored_mujoco_state.shape
            and np.array_equal(archived_mujoco_state, restored_mujoco_state)
        )
        report["archived_reward_done"] = {
            "reward": archived_envelope["snapshot"]["bridge_reward"],
            "done": archived_envelope["snapshot"]["bridge_done"],
            "task_bookkeeping": archived_envelope["snapshot"][
                "task_bookkeeping"
            ],
        }
        report["restored_reward_done"] = {
            "reward": response.get("reward"),
            "done": response.get("done"),
            "timestep": response["snapshot_state"]["mujoco_integration_state"][0],
        }
        observation = bridge.request(
            {"op": "observe", "include_snapshot_state": True}
        )
        native = bridge.request({"op": "native_task_status"})
        contacts = bridge.request({"op": "inspect_contacts"})
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        gripper_inventory = bridge.request({"op": "gripper_geometry_inventory"})
        instruction = observation.get("scene_metadata", {}).get("instruction")
        live_grasp = None
        finger_only_current = {"status": "unavailable"}
        if semantic.get("status") == "complete" and gripper_inventory.get("status") == "complete":
            target = semantic.get("bindings", {}).get("moka_pot_2")
            grippers = gripper_inventory.get("grippers", [])
            if target and len(grippers) == 1 and grippers[0].get("status") == "complete":
                target_binding = {
                    "object_id": "moka_pot_2",
                    "body_id": int(target["root_body_id"]),
                    "body_name": str(target["root_body_name"]),
                    "geom_ids": [int(value) for value in target["semantic_geom_ids"]],
                    "geom_names": [str(value) for value in target["semantic_geom_names"]],
                }
                live_grasp = bridge.request({
                    "op": "grasp_state",
                    "expected_instruction": instruction,
                    "carried_binding": target_binding,
                })
                finger_geom_ids = sorted(
                    int(item["geom_id"])
                    for item in grippers[0].get("geometries", [])
                    if item.get("geom_name") in {
                        "gripper0_finger1_collision",
                        "gripper0_finger1_pad_collision",
                        "gripper0_finger2_collision",
                        "gripper0_finger2_pad_collision",
                    }
                )
                robot_contacts = []
                for contact in contacts.get("contacts", []):
                    body_names = (
                        contact.get("body1", {}).get("name", ""),
                        contact.get("body2", {}).get("name", ""),
                    )
                    if any(name.startswith(("robot0_", "gripper0_")) for name in body_names):
                        geom1 = contact.get("geom1", {})
                        geom2 = contact.get("geom2", {})
                        robot_is_geom1 = body_names[0].startswith(("robot0_", "gripper0_"))
                        robot_geom, scene_geom = (geom1, geom2) if robot_is_geom1 else (geom2, geom1)
                        robot_contacts.append({
                            "robot_geom_id": robot_geom.get("id"),
                            "robot_geom_name": robot_geom.get("name"),
                            "scene_geom_id": scene_geom.get("id"),
                            "scene_geom_name": scene_geom.get("name"),
                            "scene_body_name": body_names[1] if robot_is_geom1 else body_names[0],
                        })
                current_pairs_pass = bool(robot_contacts) and all(
                    int(pair["robot_geom_id"]) in finger_geom_ids
                    and pair["scene_body_name"] == target_binding["body_name"]
                    and int(pair["scene_geom_id"]) in target_binding["geom_ids"]
                    for pair in robot_contacts
                )
                finger_only_current = {
                    "status": "pass" if current_pairs_pass else "fail",
                    "finger_geom_ids": finger_geom_ids,
                    "target_body_id": target_binding["body_id"],
                    "target_semantic_geom_ids": target_binding["geom_ids"],
                    "robot_contacts": robot_contacts,
                    "live_grasp_state": live_grasp,
                    "current_contact_identity_only": True,
                }
        exported = bridge.request({"op": "export_runtime_snapshot"})
        report["observed_timestep"] = observation.get("timestep")
        report["native_task_status"] = native
        report["restored_contacts"] = contacts
        report["finger_only_current_contact_audit"] = finger_only_current
        report["episode9_historical_evidence"] = _summarize_episode9_history(episode_dir)
        report["reexport_status"] = exported.get("status")
        report["reexport_sha256"] = exported.get("payload_sha256")
        report["snapshot_payload_byte_identical_after_restore"] = (
            exported.get("payload") == payload
        )
        report["restore_state_matches_observation_timestep"] = (
            observation.get("timestep") == 249
        )
        _write_report(args.output_dir, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, default=_json_default))
        return 0 if report["archived_integration_state_exactly_restored"] else 3
    finally:
        bridge.close()


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "restore_probe.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
