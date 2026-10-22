# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One-shot OSC-assisted L0T1 book grasp with durable recovery evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = Path("/workspace/zy_workspace/LIBERO-Safety")
BRIDGE_PYTHON = LIBERO_ROOT / ".venv/bin/python"
BRIDGE_SCRIPT = REPO_ROOT / "research/gs_opd/smoke/safety_bridge.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BRIDGE_ADDRESS = ("127.0.0.1", 38567)
BRIDGE_AUTHKEY = b"gs-opd-local"
TASK_INSTRUCTION = "pick up the book and place it in the back compartment of the caddy"
BOOK_BINDING = {
    "object_id": "object_1",
    "body_id": 23,
    "body_name": "black_book_1_main",
    "geom_ids": [91],
    "geom_names": ["black_book_1_g0"],
}
NON_TARGET_BODY_IDS = {1, 24, 25, 26, 27}
POSITION_GAIN_M = 0.05
ROTATION_GAIN_RAD = 0.5
MAX_POSITION_COMMAND_M = 0.01
POSITION_TOLERANCE_M = 0.006
ORIENTATION_TOLERANCE_RAD = 0.04
MAX_WAYPOINT_STEPS = 180
MAX_CLOSE_STEPS = 110


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image(reply: dict[str, Any], key: str) -> np.ndarray:
    item = reply["images"][key]
    shape = tuple(item["shape"])
    frame = np.frombuffer(item["data"], dtype=np.uint8).reshape(shape).copy()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Unexpected RGB frame shape: {frame.shape}")
    return frame


def _quat_normalize(q: Any) -> np.ndarray:
    value = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.shape != (4,) or not np.isfinite(value).all() or norm <= 1e-12:
        raise ValueError("Quaternion is malformed")
    return value / norm


def _quat_multiply(a: Any, b: Any) -> np.ndarray:
    aw, ax, ay, az = _quat_normalize(a)
    bw, bx, by, bz = _quat_normalize(b)
    return _quat_normalize(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _rotate_vector(q: Any, vector: Any) -> np.ndarray:
    w, x, y, z = _quat_normalize(q)
    v = np.asarray(vector, dtype=np.float64)
    u = np.asarray([x, y, z], dtype=np.float64)
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _orientation_error(target: Any, current: Any) -> np.ndarray:
    tw, tx, ty, tz = _quat_normalize(target)
    cw, cx, cy, cz = _quat_normalize(current)
    current_inverse = np.asarray([cw, -cx, -cy, -cz])
    delta = _quat_multiply([tw, tx, ty, tz], current_inverse)
    if delta[0] < 0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(np.clip(delta[0], -1.0, 1.0)))
    return delta[1:] * (angle / vector_norm)


def _obs_quat_wxyz(quat_xyzw: Any) -> np.ndarray:
    value = np.asarray(quat_xyzw, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("Observation EEF quaternion is malformed")
    return _quat_normalize([value[3], value[0], value[1], value[2]])


class BridgeSession:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.log_path = output_dir / "safety_bridge.log"
        self.log_handle = None
        self.process: subprocess.Popen[str] | None = None
        self.connection = None

    def start(self) -> None:
        with socket.socket() as sock:
            if sock.connect_ex(BRIDGE_ADDRESS) == 0:
                raise RuntimeError("Safety Bridge port 38567 is already in use")
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"
        env["LIBERO_CONFIG_PATH"] = "/workspace/zy_workspace/.libero_safety_config"
        env["PYTHONPATH"] = os.pathsep.join(
            [
                str(LIBERO_ROOT),
                str(LIBERO_ROOT / "third_party/robosuite-1.4"),
                str(REPO_ROOT),
            ]
        )
        self.log_handle = self.log_path.open("ab", buffering=0)
        command = [
            str(BRIDGE_PYTHON), str(BRIDGE_SCRIPT),
            "--benchmark", "obstacle_avoidance", "--level", "0",
            "--task-id", "1", "--init-state", "0", "--seed", "0",
            "--camera-resolution", "256",
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Safety Bridge exited with code {self.process.returncode}")
            try:
                self.connection = Client(BRIDGE_ADDRESS, authkey=BRIDGE_AUTHKEY)
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)
        raise TimeoutError("Safety Bridge did not become ready within 120 seconds")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.connection is None:
            raise RuntimeError("Safety Bridge is not connected")
        self.connection.send(payload)
        result = self.connection.recv()
        if not isinstance(result, dict):
            raise RuntimeError(f"Bridge returned a non-object response for {payload.get('op')}")
        return result

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


class EvidenceWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.records: list[dict[str, Any]] = []
        self.agentview: list[np.ndarray] = []
        self.wrist: list[np.ndarray] = []
        self.video = cv2.VideoWriter(
            str(output_dir / "osc_assisted_grasp.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (512, 256),
        )
        if not self.video.isOpened():
            raise RuntimeError("OpenCV could not create the grasp video")

    def add_frame(self, reply: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        agentview = _image(reply, "agentview_image")
        wrist = _image(reply, "robot0_eye_in_hand_image")
        self.agentview.append(agentview)
        self.wrist.append(wrist)
        panel = np.concatenate([agentview, wrist], axis=1)
        self.video.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        return agentview, wrist

    def add_step(self, phase: str, action: np.ndarray, reply: dict[str, Any], grasp: dict[str, Any] | None) -> None:
        self.records.append(
            {
                "step_index": len(self.records),
                "phase": phase,
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "state": reply.get("state", {}),
                "snapshot_state": reply.get("snapshot_state", {}),
                "contacts": reply.get("contacts", {}),
                "start_contacts": reply.get("start_contacts", {}),
                "contact_monitoring": reply.get("contact_monitoring", {}),
                "grasp_state": grasp,
                "reward": reply.get("reward", 0.0),
                "done": reply.get("done", False),
            }
        )

    def save(self, metadata: dict[str, Any]) -> None:
        self.video.release()
        states = [item.get("snapshot_state", {}) for item in self.records]
        objects = [item.get("object_states", {}) for item in states]
        robots = [item.get("robot_states", [{}])[0] for item in states]
        arrays = {
            "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
            "step_indices": np.arange(len(self.records), dtype=np.int64),
            "phases": np.asarray([item["phase"] for item in self.records]),
            "actions": np.asarray([item["action"] for item in self.records], dtype=np.float32).reshape(-1, 7),
            "rewards": np.asarray([item["reward"] for item in self.records], dtype=np.float32),
            "dones": np.asarray([item["done"] for item in self.records], dtype=bool),
            "eef_pos": np.asarray([item.get("robot0_eef_pos", [np.nan] * 3) for item in (r.get("state", {}) for r in self.records)], dtype=np.float64).reshape(-1, 3),
            "eef_quat_xyzw": np.asarray([item.get("robot0_eef_quat", [np.nan] * 4) for item in (r.get("state", {}) for r in self.records)], dtype=np.float64).reshape(-1, 4),
            "gripper_qpos": np.asarray([item.get("robot0_gripper_qpos", [np.nan] * 2) for item in (r.get("state", {}) for r in self.records)], dtype=np.float64).reshape(-1, 2),
            "robot_joint_qpos": np.asarray([item.get("joint_qpos", [np.nan] * 7) for item in robots], dtype=np.float64).reshape(-1, 7),
            "book_state_json": np.asarray([json.dumps(next((v for k, v in ob.items() if k.endswith(":black_book_1")), {}), ensure_ascii=False) for ob in objects]),
            "bottle_state_json": np.asarray([json.dumps(next((v for k, v in ob.items() if k.endswith(":bottle_of_beer__7_1")), {}), ensure_ascii=False) for ob in objects]),
            "caddy_state_json": np.asarray([json.dumps(next((v for k, v in ob.items() if k.endswith(":desk_caddy_1")), {}), ensure_ascii=False) for ob in objects]),
            "all_physical_states_json": np.asarray([json.dumps(s, ensure_ascii=False) for s in states]),
            "pre_contacts_json": np.asarray([json.dumps(r.get("start_contacts", {}), ensure_ascii=False) for r in self.records]),
            "post_contacts_json": np.asarray([json.dumps(r.get("contacts", {}), ensure_ascii=False) for r in self.records]),
            "substep_contact_monitoring_json": np.asarray([json.dumps(r.get("contact_monitoring", {}), ensure_ascii=False) for r in self.records]),
            "grasp_state_json": np.asarray([json.dumps(r.get("grasp_state"), ensure_ascii=False) for r in self.records]),
            "agentview_rgb": np.asarray(self.agentview, dtype=np.uint8),
            "wrist_rgb": np.asarray(self.wrist, dtype=np.uint8),
        }
        np.savez_compressed(self.output_dir / "osc_assisted_grasp_trace.npz", **arrays)
        _json(self.output_dir / "step_records.json", self.records)


def _binding_from_existing(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    semantic = json.loads(path.read_text(encoding="utf-8"))
    item = semantic.get("object_1", {})
    association = item.get("physical_association", {})
    decision = item.get("semantic_decision", {})
    if (
        item.get("status") != "resolved"
        or decision.get("status") != "resolved"
        or not str(decision.get("review_status", "")).startswith("reviewed_")
        or association.get("body_id") != BOOK_BINDING["body_id"]
        or association.get("body_name") != BOOK_BINDING["body_name"]
        or association.get("geom_ids") != BOOK_BINDING["geom_ids"]
        or association.get("geom_names") != BOOK_BINDING["geom_names"]
    ):
        raise ValueError("Existing reviewed Book binding does not match Body 23 / Geom 91")
    return BOOK_BINDING.copy(), item


def _book_target(state: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    book = state["object_states"]["objects_dict:black_book_1"]
    root = np.asarray(book["root_pos"], dtype=np.float64)
    q = _quat_normalize(book["root_quat"])
    # Center of the existing LIBERO black_book visual mesh, scaled by its XML 0.4 factor.
    mesh_center_local = np.asarray([-0.0004478, -0.0120432, 0.0690992], dtype=np.float64)
    center = root + _rotate_vector(q, mesh_center_local)
    yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
    return center, float(center[2]), yaw


def _non_target_collision(monitoring: dict[str, Any]) -> dict[str, Any] | None:
    if monitoring.get("status") != "complete":
        return {"reason": "substep_contact_monitoring_incomplete", "monitoring_status": monitoring.get("status")}
    for event in monitoring.get("events", []):
        if event.get("classification") != "robot_scene":
            continue
        body_ids = {item.get("id") for item in event.get("body_pair", [])}
        touched = sorted(body_ids & NON_TARGET_BODY_IDS)
        if touched:
            return {"reason": "robot_contact_with_non_target_body", "body_ids": touched, "event": event}
    return None


def _check_step(reply: dict[str, Any], *, stable: bool) -> str | None:
    if reply.get("done"):
        return "environment_terminated_before_grasp_complete"
    collision = _non_target_collision(reply.get("contact_monitoring", {}))
    if collision is not None:
        return json.dumps(collision, ensure_ascii=False, sort_keys=True)
    snapshot = reply.get("snapshot_state", {})
    robot = (snapshot.get("robot_states") or [{}])[0]
    if robot.get("joint_qpos") and len(robot["joint_qpos"]) == 7:
        # Native MuJoCo limits remain authoritative; this only prevents a grossly invalid command.
        if not np.isfinite(np.asarray(robot["joint_qpos"], dtype=np.float64)).all():
            return "non_finite_panda_joint_state"
    book = snapshot.get("object_states", {}).get("objects_dict:black_book_1", {})
    start_book = getattr(_check_step, "start_book", None)
    if start_book is None:
        _check_step.start_book = np.asarray(book.get("root_pos", [np.nan] * 3), dtype=np.float64)
    elif not stable and np.linalg.norm(np.asarray(book.get("root_pos", [np.nan] * 3), dtype=np.float64) - start_book) > 0.02:
        return "book_displaced_more_than_20mm_before_stable_grasp"
    return None


def _send_step(
    bridge: BridgeSession,
    writer: EvidenceWriter,
    action: np.ndarray,
    phase: str,
    monitor,
    step_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reply = bridge.request(
        {"op": "step_monitored", "action": np.asarray(action, dtype=np.float32).tolist(), "action_index": step_index}
    )
    if reply.get("error"):
        raise RuntimeError(reply["error"])
    writer.add_frame(reply)
    grasp = bridge.request(
        {"op": "grasp_state", "carried_binding": BOOK_BINDING, "expected_instruction": TASK_INSTRUCTION}
    )
    sample = monitor.update(grasp, rollout_step=step_index)
    writer.add_step(phase, action, reply, grasp)
    return reply, sample


def _move_position(
    bridge: BridgeSession,
    writer: EvidenceWriter,
    target: np.ndarray,
    phase: str,
    monitor,
    step_index: list[int],
    target_orientation: np.ndarray,
) -> tuple[dict[str, Any], str | None]:
    latest: dict[str, Any] | None = None
    for _ in range(MAX_WAYPOINT_STEPS):
        observed = bridge.request({"op": "observe", "include_snapshot_state": True})
        current = np.asarray(observed["state"]["robot0_eef_pos"], dtype=np.float64)
        error = np.asarray(target, dtype=np.float64) - current
        distance = float(np.linalg.norm(error))
        if distance <= POSITION_TOLERANCE_M:
            return observed, None
        delta = error * min(1.0, MAX_POSITION_COMMAND_M / max(distance, 1e-12))
        current_orientation = observed["snapshot_state"]["robot_states"][0]["eef_quat"]
        rotation_error = _orientation_error(target_orientation, current_orientation)
        rotation_error = np.clip(rotation_error, -ROTATION_GAIN_RAD, ROTATION_GAIN_RAD)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = (delta / POSITION_GAIN_M).astype(np.float32)
        action[3:6] = (rotation_error / ROTATION_GAIN_RAD).astype(np.float32)
        action[6] = 0.0
        latest, _ = _send_step(bridge, writer, action, phase, monitor, step_index[0])
        issue = _check_step(latest, stable=monitor.stable)
        if issue:
            return latest, issue
        actual_position = np.asarray(latest["snapshot_state"]["robot_states"][0]["eef_pos"], dtype=np.float64)
        actual_delta = actual_position - current
        if float(np.linalg.norm(actual_delta)) > (2.0 * MAX_POSITION_COMMAND_M + 0.005):
            return latest, f"eef_motion_exceeded_guard:{float(np.linalg.norm(actual_delta)):.6f}m"
        step_index[0] += 1
        if phase == "descend_to_book_mid_height":
            contacts = writer.records[-1].get("grasp_state", {}).get("finger_contacts", {})
            if contacts.get("left") is True or contacts.get("right") is True:
                return latest, None
    return latest or {}, f"waypoint_not_reached:{phase}"


def _rotate_and_hold_open(
    bridge: BridgeSession,
    writer: EvidenceWriter,
    target_orientation: np.ndarray,
    monitor,
    step_index: list[int],
) -> tuple[dict[str, Any], str | None]:
    latest: dict[str, Any] | None = None
    for _ in range(MAX_WAYPOINT_STEPS):
        observed = bridge.request({"op": "observe", "include_snapshot_state": True})
        current_orientation = observed["snapshot_state"]["robot_states"][0]["eef_quat"]
        error = _orientation_error(target_orientation, current_orientation)
        if float(np.linalg.norm(error)) <= ORIENTATION_TOLERANCE_RAD:
            return observed, None
        action = np.zeros(7, dtype=np.float32)
        action[3:6] = (np.clip(error, -ROTATION_GAIN_RAD, ROTATION_GAIN_RAD) / ROTATION_GAIN_RAD).astype(np.float32)
        action[6] = 0.0
        latest, _ = _send_step(bridge, writer, action, "align_book_thickness_axis", monitor, step_index[0])
        issue = _check_step(latest, stable=monitor.stable)
        if issue:
            return latest, issue
        step_index[0] += 1
    return latest or {}, "orientation_not_reached"


def _copy_inputs(output_dir: Path, semantic_path: Path, reviewed_path: Path) -> None:
    inputs = output_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for source in (semantic_path, reviewed_path):
        shutil.copy2(source, inputs / source.name)


def run(output_dir: Path, semantic_path: Path, reviewed_path: Path) -> dict[str, Any]:
    from research.gs_opd.repair.trajectory_bundle_acceptance import GraspStabilityMonitor

    output_dir.mkdir(parents=True, exist_ok=False)
    binding, semantic_item = _binding_from_existing(semantic_path)
    reviewed_doc = json.loads(reviewed_path.read_text(encoding="utf-8"))
    if reviewed_doc.get("formal_safety_ir_modified") is not False or reviewed_doc.get("compiler_core_modified") is not False:
        raise ValueError("Reviewed binding provenance reports unexpected Safety IR/Compiler changes")
    _copy_inputs(output_dir, semantic_path, reviewed_path)

    source_snapshots = output_dir / "source"
    source_snapshots.mkdir()
    shutil.copy2(BRIDGE_SCRIPT, source_snapshots / "safety_bridge.py")
    shutil.copy2(Path(__file__), source_snapshots / "osc_assisted_book_grasp.py")

    evidence = EvidenceWriter(output_dir)
    bridge = BridgeSession(output_dir)
    step_index = [0]
    monitor = GraspStabilityMonitor(required_samples=3)
    run_report: dict[str, Any] = {
        "schema_version": "gs_opd_l0t1_osc_assisted_book_grasp.v1",
        "scope": {
            "benchmark": "obstacle_avoidance", "level": 0, "task_id": 1,
            "init_state_index": 0, "environment_seed": 0,
            "task_stage": "grasp_only", "instruction": TASK_INSTRUCTION,
        },
        "status": "in_progress",
        "action_source": "scripted OSC_POSE Cartesian waypoint controller; no VLA action used",
        "controller": {
            "interface": "existing Safety Bridge 7D OSC_POSE step",
            "position_output_gain_m_per_input": POSITION_GAIN_M,
            "rotation_output_gain_rad_per_input": ROTATION_GAIN_RAD,
            "world_frame_delta": True,
            "max_requested_cartesian_delta_m_per_control_step": MAX_POSITION_COMMAND_M,
            "default_controller_config_modified": False,
        },
        "target_binding": binding,
        "target_binding_review_status": semantic_item["semantic_decision"].get("review_status"),
        "grasp_acceptance": "existing GraspStabilityMonitor; 3 consecutive distinct states with closed gripper, bilateral target finger contact, grasp_confirmed, and unchanged physical state",
        "trajectory_safety_verdict": "not_evaluated",
        "b_spline_or_placement_executed": False,
        "training_or_checkpoint_modified": False,
        "non_target_collision_stop_bodies": sorted(NON_TARGET_BODY_IDS),
    }
    frames_before = 0
    final_snapshot_bytes: bytes | None = None
    failure_reason: str | None = None
    try:
        bridge.start()
        initial = bridge.request({"op": "observe", "include_snapshot_state": True})
        scene = initial.get("scene_metadata", {})
        if (
            scene.get("benchmark") != "obstacle_avoidance"
            or scene.get("level") != 0
            or scene.get("task_id") != 1
            or scene.get("init_state_index") != 0
            or scene.get("environment_seed") != 0
            or initial.get("instruction") != TASK_INSTRUCTION
        ):
            raise ValueError("Active Bridge does not match obstacle_avoidance/L0T1/state0/seed0")
        if initial.get("snapshot_state", {}).get("object_states", {}).get("objects_dict:black_book_1", {}).get("body_id") != 23:
            raise ValueError("Active Book object does not match Body 23")
        inventory = bridge.request({"op": "geometry_inventory"})
        grippers = bridge.request({"op": "gripper_geometry_inventory"})
        repair_context = bridge.request({"op": "repair_context"})
        if inventory.get("status") != "complete" or grippers.get("status") != "complete":
            raise ValueError("Active geometry or Panda finger inventory is incomplete")
        geom91 = next((x for x in inventory.get("geometries", []) if x.get("geom_id") == 91), None)
        if not geom91 or geom91.get("body_id") != 23 or geom91.get("geom_name") != "black_book_1_g0":
            raise ValueError("Active Geometry Inventory does not confirm Body 23 / Geom 91")
        scene_start_record = {key: value for key, value in initial.items() if key != "images"}
        scene_start_record["images"] = {
            key: {
                "shape": value.get("shape"),
                "dtype": value.get("dtype"),
                "byte_count": len(value.get("data", b"")),
            }
            for key, value in initial.get("images", {}).items()
        }
        _json(output_dir / "scene_start.json", scene_start_record)
        _json(output_dir / "geometry_inventory.json", inventory)
        _json(output_dir / "gripper_geometry_inventory.json", grippers)
        _json(output_dir / "osc_repair_context.json", repair_context)
        evidence.add_frame(initial)
        book_center, book_mid_z, book_yaw = _book_target(initial["snapshot_state"])
        book_root_quat = _quat_normalize(
            initial["snapshot_state"]["object_states"]["objects_dict:black_book_1"]["root_quat"]
        )
        book_length_axis = _rotate_vector(book_root_quat, [0.0, 1.0, 0.0])
        grasp_offset_m = 0.025
        grasp_center = book_center + grasp_offset_m * book_length_axis
        eef_start = np.asarray(initial["state"]["robot0_eef_pos"], dtype=np.float64)
        eef_quat_start = _quat_normalize(initial["snapshot_state"]["robot_states"][0]["eef_quat"])
        jaw_axis = _rotate_vector(eef_quat_start, [1.0, 0.0, 0.0])
        current_jaw_yaw = math.atan2(float(jaw_axis[1]), float(jaw_axis[0]))
        # The two headings preserve the same jaw-axis line but put the Panda
        # palm on opposite sides of the book. The nearest heading in attempt4
        # drove the palm into the place box, so use the opposite heading here.
        grasp_yaw_candidates = [book_yaw, book_yaw + math.pi]
        grasp_yaw = max(grasp_yaw_candidates, key=lambda angle: abs(_wrap_pi(angle - current_jaw_yaw)))
        yaw_delta = _wrap_pi(grasp_yaw - current_jaw_yaw)
        qz = [math.cos(yaw_delta / 2.0), 0.0, 0.0, math.sin(yaw_delta / 2.0)]
        target_orientation = _quat_multiply(qz, eef_quat_start)
        run_report["initial_state"] = {
            "eef_pos_m": eef_start.tolist(),
            "eef_quat_wxyz": eef_quat_start.tolist(),
            "book_root_pos_m": initial["snapshot_state"]["object_states"]["objects_dict:black_book_1"]["root_pos"],
            "book_root_quat_wxyz": initial["snapshot_state"]["object_states"]["objects_dict:black_book_1"]["root_quat"],
            "derived_book_mesh_center_world_m": book_center.tolist(),
            "book_yaw_rad_from_active_root_quaternion": book_yaw,
            "initial_jaw_axis_yaw_rad": current_jaw_yaw,
            "target_jaw_axis_yaw_rad": grasp_yaw,
            "approach_heading_selection": "opposite_equivalent_heading_to_clear_place_box_after_attempt4_palm_collision",
            "yaw_delta_rad": yaw_delta,
            "book_mid_height_world_m": book_mid_z,
            "book_length_axis_world": book_length_axis.tolist(),
            "grasp_center_world_m": grasp_center.tolist(),
            "grasp_center_offset_along_book_length_axis_m": grasp_offset_m,
            "grasp_offset_rationale": "move 25 mm along book length away from place box after observed Panda palm collision",
        }
        _check_step.start_book = np.asarray(
            initial["snapshot_state"]["object_states"]["objects_dict:black_book_1"]["root_pos"],
            dtype=np.float64,
        )

        start_grasp = bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": TASK_INSTRUCTION})
        monitor.update(start_grasp, rollout_step=-1)

        high_z = max(1.30, float(eef_start[2]) + 0.12)
        target_over_book = np.asarray([grasp_center[0], grasp_center[1], high_z], dtype=np.float64)
        high_side_point = np.asarray([eef_start[0], max(float(eef_start[1]), float(grasp_center[1]) + 0.02), high_z], dtype=np.float64)
        for waypoint, name in (
            (np.asarray([eef_start[0], eef_start[1], high_z]), "raise_clear_of_scene"),
            (high_side_point, "route_on_positive_y_side_of_bottle"),
            (target_over_book, "move_over_book_at_clearance"),
        ):
            _reply, issue = _move_position(bridge, evidence, waypoint, name, monitor, step_index, target_orientation=target_orientation if name == "move_over_book_at_clearance" else eef_quat_start)
            if issue:
                failure_reason = issue
                break
        if failure_reason is None:
            _reply, issue = _rotate_and_hold_open(bridge, evidence, target_orientation, monitor, step_index)
            if issue:
                failure_reason = issue

        if failure_reason is None:
            grasp_pose = np.asarray([grasp_center[0], grasp_center[1], book_mid_z], dtype=np.float64)
            _reply, issue = _move_position(bridge, evidence, grasp_pose, "descend_to_book_mid_height", monitor, step_index, target_orientation)
            if issue:
                failure_reason = issue

        if failure_reason is None:
            for _ in range(MAX_CLOSE_STEPS):
                action = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
                reply, _ = _send_step(bridge, evidence, action, "close_on_book_and_check_bilateral_contact", monitor, step_index[0])
                issue = _check_step(reply, stable=monitor.stable)
                if issue:
                    failure_reason = issue
                    break
                step_index[0] += 1
                if monitor.stable:
                    break

        run_report["stable_grasp"] = monitor.diagnostics()
        run_report["action_count"] = len(evidence.records)
        run_report["task_reward_sum"] = float(sum(float(x["reward"]) for x in evidence.records))
        run_report["environment_done"] = bool(evidence.records[-1]["done"]) if evidence.records else False
        run_report["non_target_collision_events"] = [
            event
            for record in evidence.records
            for event in record.get("contact_monitoring", {}).get("events", [])
            if event.get("classification") == "robot_scene"
            and {body.get("id") for body in event.get("body_pair", [])} & NON_TARGET_BODY_IDS
        ]
        run_report["all_control_steps_contact_coverage_complete"] = bool(evidence.records) and all(
            record.get("contact_monitoring", {}).get("status") == "complete" for record in evidence.records
        )
        if monitor.stable and failure_reason is None:
            export = bridge.request({"op": "export_runtime_snapshot"})
            if export.get("status") != "complete" or not isinstance(export.get("payload"), bytes):
                failure_reason = export.get("error", "runtime_snapshot_export_failed")
            else:
                final_snapshot_bytes = export["payload"]
                (output_dir / "initial_stable_runtime_snapshot.pkl").write_bytes(final_snapshot_bytes)
                run_report["initial_stable_snapshot"] = {
                    "file": "initial_stable_runtime_snapshot.pkl",
                    "sha256": _sha256(output_dir / "initial_stable_runtime_snapshot.pkl"),
                    "schema_version": export.get("schema_version"),
                    "model_geometry_sha256": export.get("model_geometry_sha256"),
                    "payload_sha256": export.get("payload_sha256"),
                    "snapshot_state": export.get("snapshot_state"),
                }
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
    finally:
        bridge.close()

    if final_snapshot_bytes is not None and failure_reason is None and monitor.stable:
        recovery_bridge = BridgeSession(output_dir)
        try:
            recovery_bridge.start()
            restored = recovery_bridge.request({"op": "restore_runtime_snapshot", "payload": final_snapshot_bytes})
            if restored.get("status") != "restored":
                failure_reason = restored.get("error", "runtime_snapshot_restore_failed")
            else:
                expected_state = run_report["initial_stable_snapshot"]["snapshot_state"]
                actual_state = restored.get("snapshot_state", {})
                expected_vec = np.asarray(expected_state.get("mujoco_integration_state", []), dtype=np.float64)
                actual_vec = np.asarray(actual_state.get("mujoco_integration_state", []), dtype=np.float64)
                restore_error = float(np.max(np.abs(expected_vec - actual_vec))) if expected_vec.shape == actual_vec.shape and expected_vec.size else math.inf
                run_report["recovery_restore"] = {
                    "status": "matched" if restore_error == 0.0 else "mismatch",
                    "max_abs_integration_state_error": restore_error,
                    "model_geometry_sha256": restored.get("model_geometry_sha256"),
                }
                if restore_error != 0.0:
                    failure_reason = "runtime_snapshot_restore_state_mismatch"
                else:
                    recovery_monitor = GraspStabilityMonitor(required_samples=3)
                    start_state = recovery_bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": TASK_INSTRUCTION})
                    recovery_monitor.update(start_state, rollout_step=step_index[0] - 1)
                    for _ in range(3):
                        action = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
                        reply, _ = _send_step(recovery_bridge, evidence, action, "post_restore_three_state_grasp_verification", recovery_monitor, step_index[0])
                        issue = _check_step(reply, stable=recovery_monitor.stable)
                        if issue:
                            failure_reason = issue
                            break
                        step_index[0] += 1
                    run_report["recovery_grasp_verification"] = recovery_monitor.diagnostics()
                    if not recovery_monitor.stable and failure_reason is None:
                        failure_reason = "restored_state_did_not_retain_three_consecutive_bilateral_grasp_states"
                    if recovery_monitor.stable and failure_reason is None:
                        recovered_export = recovery_bridge.request({"op": "export_runtime_snapshot"})
                        if recovered_export.get("status") == "complete" and isinstance(recovered_export.get("payload"), bytes):
                            final_snapshot_bytes = recovered_export["payload"]
                            recovery_path = output_dir / "recovered_grasp_runtime_snapshot.pkl"
                            recovery_path.write_bytes(final_snapshot_bytes)
                            run_report["recovered_grasp_snapshot"] = {
                                "file": recovery_path.name,
                                "sha256": _sha256(recovery_path),
                                "schema_version": recovered_export.get("schema_version"),
                                "model_geometry_sha256": recovered_export.get("model_geometry_sha256"),
                                "payload_sha256": recovered_export.get("payload_sha256"),
                                "snapshot_state": recovered_export.get("snapshot_state"),
                            }
                        else:
                            failure_reason = recovered_export.get("error", "recovered_runtime_snapshot_export_failed")
        except Exception as exc:
            failure_reason = f"recovery_{type(exc).__name__}: {exc}"
        finally:
            recovery_bridge.close()

    run_report["status"] = "stable_grasp_recovered" if failure_reason is None and monitor.stable else "failed"
    run_report["failure_reason"] = failure_reason
    run_report["success"] = run_report["status"] == "stable_grasp_recovered"
    run_report["action_count"] = len(evidence.records)
    run_report["stable_grasp"] = monitor.diagnostics()
    run_report["complete_action_trace"] = "osc_assisted_grasp_trace.npz"
    run_report["video"] = "osc_assisted_grasp.mp4"
    run_report["training_data_created"] = False
    run_report["trajectory_safety_verdict"] = "not_evaluated"
    evidence.save(run_report)
    _json(output_dir / "experiment_report.json", run_report)
    manifest = {
        "schema_version": "gs_opd_l0t1_osc_assisted_grasp_manifest.v1",
        "created_local_date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "command": [sys.executable, str(Path(__file__).resolve()), "--out-dir", str(output_dir), "--semantic-results", str(semantic_path), "--reviewed-bindings", str(reviewed_path)],
        "bridge_command": [str(BRIDGE_PYTHON), str(BRIDGE_SCRIPT), "--benchmark", "obstacle_avoidance", "--level", "0", "--task-id", "1", "--init-state", "0", "--seed", "0", "--camera-resolution", "256"],
        "source_sha256": {
            "safety_bridge.py": _sha256(source_snapshots / "safety_bridge.py"),
            "osc_assisted_book_grasp.py": _sha256(source_snapshots / "osc_assisted_book_grasp.py"),
        },
        "outputs_sha256": {},
        "experiment_status": run_report["status"],
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            manifest["outputs_sha256"][str(path.relative_to(output_dir))] = _sha256(path)
    _json(output_dir / "run_manifest.json", manifest)
    return run_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--semantic-results", type=Path,
        default=Path("/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_state0_base_pi05_grasp_smoke_20261022_attempt1/semantic_results_by_object.json"),
    )
    parser.add_argument(
        "--reviewed-bindings", type=Path,
        default=Path("/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_reviewed_semantic_compiler_input_20261022_attempt1/reviewed_semantic_bindings.json"),
    )
    args = parser.parse_args()
    result = run(args.out_dir.resolve(), args.semantic_results.resolve(), args.reviewed_bindings.resolve())
    print(json.dumps({k: result.get(k) for k in ("status", "success", "failure_reason", "action_count", "recovery_restore", "recovery_grasp_verification")}, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
