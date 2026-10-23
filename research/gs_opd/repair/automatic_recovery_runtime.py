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

"""Runtime adapters for the Automatic Safety Recovery Engine."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from research.gs_opd.data.trajectory_writer import TrajectoryWriter
from research.gs_opd.repair.automatic_recovery import (
    GoalSpec,
    JointSpacePlan,
    Observation,
    PrefixPreview,
    RecoveryArtifactWriter,
    RecoveryConfig,
    TaskContext,
)
from research.gs_opd.repair.trajectory_tracking import (
    quaternion_conjugate,
    quaternion_multiply,
)
from research.gs_opd.repair.grasp_retargeting import GraspRetargetingTemplate
from research.gs_opd.repair.task_phase_controller import PhaseEvidence, TaskPhaseController


REPO_ROOT = Path(__file__).resolve().parents[3]
SAFETY_BRIDGE = REPO_ROOT / "research/gs_opd/smoke/safety_bridge.py"
DEFAULT_ENV_PYTHON = Path("/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python")
DEFAULT_PI05_CHECKPOINT = Path("/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_observation(data: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
        .reshape(value["shape"])
        .copy()
        for key, value in data.get("images", {}).items()
    }
    result.update({key: np.asarray(value, dtype=np.float32) for key, value in data.get("state", {}).items()})
    return result


def risk_branch_coverage_is_complete(
    monitor: dict[str, Any], expected_control_steps: int
) -> bool:
    """Separate fully observed unsafe branches from incomplete measurements."""
    expected_substeps = monitor.get("expected_physics_substeps_per_control_step")
    control_steps = monitor.get("control_steps", [])
    geometry = monitor.get("geometry_monitoring", {})
    geometry_substeps = geometry.get("substeps", [])
    total_expected = expected_substeps * expected_control_steps if expected_substeps else 0
    if (
        expected_control_steps <= 0
        or not isinstance(expected_substeps, int)
        or expected_substeps <= 0
        or monitor.get("captured_physics_substeps") != total_expected
        or monitor.get("contact_substeps") != total_expected
        or len(control_steps) != expected_control_steps
        or len(geometry_substeps) != total_expected
        or geometry.get("status") not in {"complete", "unsafe"}
    ):
        return False
    for step in control_steps:
        samples = step.get("contact_samples", {})
        capture_error = step.get("capture_error")
        known_unsafe = capture_error == "substep_geometry_safety_unsafe"
        if (
            step.get("expected_physics_substeps") != expected_substeps
            or step.get("captured_physics_substeps") != expected_substeps
            or step.get("unobserved_expected_substeps") != 0
            or sum(int(value) for value in samples.values()) != expected_substeps
            or int(samples.get("unavailable", 0)) != 0
            or capture_error not in (None, "substep_geometry_safety_unsafe")
            or (step.get("status") != "complete" and not known_unsafe)
        ):
            return False
    return all(item.get("status") in {"complete", "unsafe"} for item in geometry_substeps)


def _quat_matrix_wxyz(value: Any) -> np.ndarray:
    w, x, y, z = np.asarray(value, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("object_pose_quaternion_is_invalid")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_quat_wxyz(matrix: Any) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise ValueError("rotation_matrix_is_invalid")
    trace = float(np.trace(m))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quat = np.asarray(
            [0.25 * scale, (m[2, 1] - m[1, 2]) / scale,
             (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            scale = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            quat = np.asarray(
                [(m[2, 1] - m[1, 2]) / scale, 0.25 * scale,
                 (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale]
            )
        elif index == 1:
            scale = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            quat = np.asarray(
                [(m[0, 2] - m[2, 0]) / scale, (m[0, 1] + m[1, 0]) / scale,
                 0.25 * scale, (m[1, 2] + m[2, 1]) / scale]
            )
        else:
            scale = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            quat = np.asarray(
                [(m[1, 0] - m[0, 1]) / scale, (m[0, 2] + m[2, 0]) / scale,
                 (m[1, 2] + m[2, 1]) / scale, 0.25 * scale]
            )
    quat /= np.linalg.norm(quat)
    if quat[0] < 0:
        quat *= -1
    return quat


def _parse_bddl_entities(
    text: str,
) -> tuple[set[str], set[str], set[str], list[tuple[str, str]], set[str]]:
    fixtures_text = re.search(r"\(:fixtures\s+(.*?)\)\s*\(:objects", text, flags=re.S)
    objects_text = re.search(r"\(:objects\s+(.*?)\)\s*\(:obj_of_interest", text, flags=re.S)
    if fixtures_text is None or objects_text is None:
        raise ValueError("active_BDDL_fixture_or_object_declarations_cannot_be_parsed")

    def declarations(section: str) -> set[str]:
        values = set()
        for names, _kind in re.findall(r"([A-Za-z0-9_\s]+?)\s+-\s+([A-Za-z0-9_]+)", section):
            values.update(names.split())
        if not values:
            raise ValueError("active_BDDL_entity_declarations_are_empty")
        return values

    fixture_names = declarations(fixtures_text.group(1))
    object_names = declarations(objects_text.group(1))
    constraints_match = re.search(r"\(:constraints\s+(.*?)\)\s*\)\s*\)$", text, flags=re.S)
    if constraints_match is None:
        raise ValueError("active_BDDL_safety_constraints_cannot_be_parsed")
    constraints_text = constraints_match.group(1)
    forbidden_robot = set(re.findall(r"\(CheckRobotContact\s+([A-Za-z0-9_]+)\s*\)", constraints_text))
    forbidden_pairs = re.findall(
        r"\(CheckContact\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\)",
        constraints_text,
    )
    if not forbidden_robot and not forbidden_pairs:
        raise ValueError("active_BDDL_does_not_declare_any_safety_contact_constraint")
    goal_start = text.find("(:goal")
    constraint_start = text.find("(:constraints", goal_start)
    if goal_start < 0 or constraint_start < 0:
        raise ValueError("active_BDDL_task_goals_cannot_be_parsed")
    goal_text = text[goal_start:constraint_start]
    goal_targets = set()
    for predicate, first, _second in re.findall(
        r"\(([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)",
        goal_text,
    ):
        if predicate.lower() not in {"and", "not", "or"}:
            goal_targets.add(first)
    for predicate, first in re.findall(
        r"\(([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z0-9_]+)\s*\)", goal_text
    ):
        if predicate.lower() not in {"and", "not", "or"}:
            goal_targets.add(first)
    if not goal_targets:
        raise ValueError("active_BDDL_has_no_resolvable_goal_target_objects")
    return fixture_names, object_names, forbidden_robot, forbidden_pairs, goal_targets


class BridgeProcess:
    """Own the local Safety Bridge process and its authenticated RPC connection."""

    def __init__(self, output_dir: Path, *, python: Path = DEFAULT_ENV_PYTHON, port: int = 38569):
        self.output_dir = output_dir
        self.python = Path(python)
        self.port = int(port)
        if not 1 <= self.port <= 65535:
            raise ValueError("Bridge TCP port must be in 1..65535")
        self.process: subprocess.Popen | None = None
        self.connection = None
        self._bridge_log = None

    def start(
        self,
        *,
        benchmark: str,
        level: int,
        task_id: int,
        init_state: int = 0,
        seed: int = 0,
        demonstration_horizon: int = 1000,
    ):
        if not self.python.is_file():
            raise FileNotFoundError(f"Safety Bridge Python environment is missing: {self.python}")
        if isinstance(demonstration_horizon, bool) or not 1 <= int(demonstration_horizon) <= 1800:
            raise ValueError("demonstration_horizon_must_be_in_1_to_1800")
        ownership = subprocess.run(
            ["ss", "-H", "-ltnp", f"sport = :{self.port}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if ownership.returncode == 0 and ownership.stdout.strip():
            raise RuntimeError(
                f"Bridge_port_is_already_owned:{self.port}:{ownership.stdout.strip()}"
            )
        if ownership.returncode != 0:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind(("127.0.0.1", self.port))
            except OSError as exc:
                raise RuntimeError(f"Bridge_port_is_unavailable:{self.port}:{exc}") from exc
            finally:
                probe.close()
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"
        env["LIBERO_CONFIG_PATH"] = "/workspace/zy_workspace/.libero_safety_config"
        pythonpath = [
            "/workspace/zy_workspace/LIBERO-Safety",
            "/workspace/zy_workspace/LIBERO-Safety/third_party/robosuite-1.4",
            "/workspace/zy_workspace/RLinf",
            str(REPO_ROOT),
        ]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        self._bridge_log = (self.output_dir / "safety_bridge.log").open("wb")
        command = [
            str(self.python), str(SAFETY_BRIDGE), "--benchmark", benchmark,
            "--level", str(level), "--task-id", str(task_id),
            "--init-state", str(init_state), "--seed", str(seed),
            "--port", str(self.port),
            "--demonstration-horizon", str(int(demonstration_horizon)),
        ]
        self.process = subprocess.Popen(
            command,
            cwd="/workspace/zy_workspace/RLinf",
            env=env,
            stdout=self._bridge_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 240
        last_error = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Safety Bridge exited with code {self.process.returncode}; see {self.output_dir / 'safety_bridge.log'}"
                )
            try:
                self.connection = Client(("127.0.0.1", self.port), authkey=b"gs-opd-local")
                return
            except (ConnectionRefusedError, OSError) as exc:
                last_error = exc
                time.sleep(1.0)
        raise TimeoutError(f"Safety Bridge startup timed out: {last_error}")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.connection is None:
            raise RuntimeError("Safety Bridge connection is not active")
        self.connection.send(payload)
        response = self.connection.recv()
        if not isinstance(response, dict):
            raise RuntimeError("Safety Bridge returned a malformed response")
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        return response

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process = None
        if self._bridge_log is not None:
            self._bridge_log.close()
            self._bridge_log = None


class Pi05StudentPolicy:
    """Run the existing checkpoint with the native LIBERO 7D action output."""

    def __init__(self, checkpoint: Path, device: str = "cuda:0"):
        checkpoint = Path(checkpoint)
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"Pi0.5 checkpoint is missing: {checkpoint}")
        os.environ["EMBODIED_PATH"] = "/workspace/zy_workspace/RLinf/examples/embodiment"
        if "/workspace/zy_workspace/RLinf" not in sys.path:
            sys.path.insert(0, "/workspace/zy_workspace/RLinf")
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from rlinf.envs.sim.libero.utils import (
            get_libero_image,
            get_libero_wrist_image,
            quat2axisangle,
        )
        from rlinf.models import get_model

        with initialize_config_dir(
            config_dir="/workspace/zy_workspace/RLinf/evaluations/libero",
            version_base=None,
        ):
            cfg = compose(config_name="libero_spatial_openpi_pi05_smoke")
        cfg.rollout.model.model_path = str(checkpoint)
        OmegaConf.update(cfg.rollout.model, "openpi.torch_compile", False, force_add=True)
        self.torch = torch
        self.get_libero_image = get_libero_image
        self.get_libero_wrist_image = get_libero_wrist_image
        self.quat2axisangle = quat2axisangle
        self.model = get_model(cfg.rollout.model).to(device).eval()
        self.checkpoint = checkpoint
        self.device = device

    def predict(self, bridge_data: dict[str, Any]) -> np.ndarray:
        obs = decode_observation(bridge_data)
        state = np.concatenate(
            [
                obs["robot0_eef_pos"],
                self.quat2axisangle(obs["robot0_eef_quat"].copy()),
                obs["robot0_gripper_qpos"],
            ]
        ).astype(np.float32)
        policy_obs = {
            "main_images": self.torch.from_numpy(
                np.ascontiguousarray(self.get_libero_image(obs))
            ).unsqueeze(0),
            "wrist_images": self.torch.from_numpy(
                np.ascontiguousarray(self.get_libero_wrist_image(obs))
            ).unsqueeze(0),
            "states": self.torch.from_numpy(state).unsqueeze(0),
            "task_descriptions": [bridge_data["instruction"]],
        }
        with self.torch.inference_mode():
            actions, _ = self.model.predict_action_batch(policy_obs, mode="eval")
        chunk = actions[0].detach().float().cpu().numpy()
        if chunk.ndim != 2 or chunk.shape[1:] != (7,) or len(chunk) == 0:
            raise ValueError("native_Pi0.5_prediction_is_not_a_nonempty_N_by_7_chunk")
        if not np.isfinite(chunk).all():
            raise ValueError("native_Pi0.5_prediction_contains_nonfinite_actions")
        return np.asarray(chunk, dtype=np.float32)


class AutomaticRecoveryBridgeBackend:
    """Connect the generic engine to a task-verified LIBERO Safety Bridge."""

    def __init__(
        self,
        bridge: BridgeProcess,
        config_data: dict[str, Any],
        recovery_config: RecoveryConfig,
        output_dir: Path,
        artifacts: RecoveryArtifactWriter,
        *,
        mode: str,
        student: Pi05StudentPolicy | None,
        archived_trace_path: Path,
        grasp_template_path: Path,
        runtime_snapshot_path: Path,
        t10_verification_path: Path,
    ):
        self.bridge = bridge
        self.config_data = config_data
        self.config = recovery_config
        self.output_dir = output_dir
        self.artifacts = artifacts
        self.mode = mode
        self.student = student
        self.archived_trace_path = Path(archived_trace_path)
        self.grasp_template_path = Path(grasp_template_path)
        self.runtime_snapshot_path = Path(runtime_snapshot_path)
        self.t10_verification_path = Path(t10_verification_path)
        self.archive = np.load(self.archived_trace_path, allow_pickle=False)
        self.archived_actions = np.asarray(self.archive["executed_actions"], dtype=np.float32)
        self.archive_sha256 = sha256_file(self.archived_trace_path)
        self.grasp_template = json.loads(self.grasp_template_path.read_text(encoding="utf-8"))
        if self.grasp_template.get("source_trace_sha256") != self.archive_sha256:
            raise ValueError("historical_grasp_record_and_student_trace_fingerprints_do_not_match")
        if int(self.config_data["historical_goal_seed_timestep"]) >= int(
            self.grasp_template["source_first_finger_contact_step_1based"]
        ):
            raise ValueError("historical_pregrasp_state_must_precede_first_finger_contact")
        self.task_grasp_template = GraspRetargetingTemplate.from_dict(self.grasp_template)
        self.task_controller: TaskPhaseController | None = None
        self.task_expert_active = False
        self.task_phase_anchor: tuple[np.ndarray, np.ndarray] | None = None
        self.last_task_phase: str | None = None
        self.phase_plan_attempts: dict[str, int] = {}
        self.current_native_task_status: dict[str, Any] | None = None
        self.current_native_goal_values: dict[tuple[str, ...], bool] = {}
        self.current_region_parent: str | None = None
        self.confirmed_placed_objects: set[str] = set()
        self.task_support_geom_ids: tuple[int, ...] = ()
        self.latest_data: dict[str, Any] | None = None
        self.safety_policy: dict[str, Any] | None = None
        self.context_static: dict[str, Any] | None = None
        self.bindings: dict[str, Any] = {}
        self.student_chunk = np.empty((0, 7), dtype=np.float32)
        self.student_chunk_cursor = 0
        self.current_forecast = np.empty((0, 7), dtype=np.float32)
        self.archived_cursor = 10 if mode == "t10-snapshot" else 0
        self.live_student_handoff = mode != "t10-snapshot"
        self.student_handoff_completed = False
        self.initial_history_safety_verified = False
        self.transition_count = 0
        self.plan_cursor = 0
        self.plan_attempt = 0
        self.plan_counter = 0
        self._current_action_source = "student"
        self.latest_observation: Observation | None = None
        self._video_writers: dict[str, Any] = {}
        self._cv2 = None
        self._ensure_video()

    def _ensure_video(self):
        external_site_packages = Path("/workspace/zy_workspace/RLinf/.venv/lib/python3.11/site-packages")
        if external_site_packages.is_dir() and str(external_site_packages) not in sys.path:
            sys.path.append(str(external_site_packages))
        import cv2

        self._cv2 = cv2

    def _save_video_frame(self, data: dict[str, Any]):
        for key, filename in (
            ("agentview_image", "agentview.mp4"),
            ("robot0_eye_in_hand_image", "wrist.mp4"),
        ):
            image = data.get("images", {}).get(key)
            if not isinstance(image, dict):
                raise ValueError(f"required_video_camera_{key}_is_unavailable")
            frame = np.frombuffer(image["data"], dtype=np.dtype(image["dtype"])).reshape(image["shape"])
            if key not in self._video_writers:
                height, width = frame.shape[:2]
                writer = self._cv2.VideoWriter(
                    str(self.output_dir / filename),
                    self._cv2.VideoWriter_fourcc(*"mp4v"),
                    20.0,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could_not_open_video_output:{filename}")
                self._video_writers[key] = writer
            self._video_writers[key].write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))

    def _load_context_inputs(self, data: dict[str, Any]) -> dict[str, Any]:
        metadata = data.get("scene_metadata")
        if not isinstance(metadata, dict):
            raise ValueError("active_Safety_Bridge_scene_metadata_is_missing")
        bddl_path = Path(metadata["bddl_path"])
        bddl_bytes = bddl_path.read_bytes()
        bddl_hash = hashlib.sha256(bddl_bytes).hexdigest()
        expected_bddl_hash = self.config_data.get("expected_bddl_sha256")
        if expected_bddl_hash and expected_bddl_hash != bddl_hash:
            raise ValueError("active_BDDL_hash_does_not_match_recovery_configuration")
        if metadata.get("model_geometry_sha256") != self.config_data.get("expected_model_geometry_sha256"):
            raise ValueError("compiled_model_geometry_hash_does_not_match_recovery_configuration")
        fixture_names, object_names, forbidden_robot, forbidden_pairs, goal_targets = _parse_bddl_entities(
            bddl_bytes.decode("utf-8")
        )
        semantic = self.bridge.request({"op": "task_semantic_geometry_bindings"})
        if semantic.get("status") != "complete":
            raise ValueError(f"task_semantic_geometry_binding_unknown:{semantic.get('error')}")
        self.bindings = semantic["bindings"]
        declared = fixture_names | object_names
        unbound_declared_entities = sorted(declared - set(self.bindings))
        inventory = self.bridge.request({"op": "geometry_inventory"})
        if inventory.get("status") != "complete":
            raise ValueError("complete_active_geometry_inventory_is_unavailable")
        gripper_inventory = self.bridge.request({"op": "gripper_geometry_inventory"})
        if gripper_inventory.get("status") != "complete" or len(gripper_inventory.get("grippers", [])) != 1:
            raise ValueError("complete_single_Panda_gripper_geometry_inventory_is_unavailable")
        gripper_geom_ids = sorted(
            item["geom_id"] for item in gripper_inventory["grippers"][0]["geometries"]
        )
        if not gripper_geom_ids:
            raise ValueError("Panda_gripper_contact_geometry_is_empty")

        forbidden_names = sorted(forbidden_robot)
        unbound_goal_targets = sorted(goal_targets - set(self.bindings))
        if unbound_goal_targets:
            raise ValueError(f"BDDL_goal_target_entities_are_unbound:{unbound_goal_targets}")
        target_names = sorted(goal_targets)
        if not forbidden_names:
            raise ValueError("BDDL_has_no_CheckRobotContact_objects_for_safety_takeover")
        if set(forbidden_names) & set(target_names):
            raise ValueError("BDDL_object_is_both_robot_forbidden_and_goal_target")
        constraint_names = {name for pair in forbidden_pairs for name in pair} | set(forbidden_names)
        required_semantic_entities = constraint_names | set(goal_targets)
        unknown_constraint_names = sorted(required_semantic_entities - set(self.bindings))
        if unknown_constraint_names:
            raise ValueError(
                f"BDDL_safety_or_goal_entities_are_unbound:{unknown_constraint_names}"
            )
        missing_constraint_geometry = sorted(
            name for name in constraint_names if not self.bindings[name]["geom_ids"]
        )
        if missing_constraint_geometry:
            raise ValueError(
                f"BDDL_contact_constraint_objects_have_no_collision_geometry:{missing_constraint_geometry}"
            )
        target_contact_names = [
            name for name in target_names if self.bindings[name]["geom_ids"]
        ]

        def group(name: str, minimum: float) -> dict[str, Any]:
            binding = self.bindings[name]
            return {
                "object_id": name,
                "body_ids": binding["collision_body_ids"],
                "body_names": binding["collision_body_names"],
                "geom_ids": binding["geom_ids"],
                "minimum_clearance_m": float(minimum),
            }

        inventory_geoms = inventory["geometries"]
        robot_body_ids = {
            int(item["body_id"])
            for item in inventory_geoms
            if item.get("category") == "robot" and item.get("collision_enabled")
        }
        scene_groups = forbidden_names + target_names
        covered_body_ids = {
            int(body_id)
            for name in scene_groups
            for body_id in self.bindings[name]["collision_body_ids"]
        }
        all_scene_collision = {
            int(item["body_id"])
            for item in inventory_geoms
            if item.get("collision_enabled") and item.get("category") == "scene"
        }
        other_body_ids = sorted(all_scene_collision - covered_body_ids)
        body_lookup = {
            int(item["body_id"]): item
            for item in inventory["bodies"]
            if isinstance(item.get("body_id"), int)
        }
        other_geoms = sorted(
            int(item["geom_id"])
            for item in inventory_geoms
            if item.get("collision_enabled") and int(item["body_id"]) in other_body_ids
        )
        other_groups = []
        if other_body_ids:
            other_groups.append({
                "object_id": "unmapped_scene_environment",
                "body_ids": other_body_ids,
                "body_names": [body_lookup[item]["body_name"] for item in other_body_ids],
                "geom_ids": other_geoms,
                "minimum_clearance_m": 0.0,
            })
        if not other_groups and not covered_body_ids.issuperset(all_scene_collision):
            raise ValueError("active_scene_collision_geometry_cannot_be_classified")

        allowed_pairs = []
        for target_name in target_names:
            # Some goal fixtures expose a registered semantic Body whose
            # compiled geometry is visual only. Such a fixture can still be a
            # task target, but cannot produce contact pairs or collision risk.
            if not self.bindings[target_name]["geom_ids"]:
                continue
            for robot_geom in gripper_geom_ids:
                for scene_geom in self.bindings[target_name]["geom_ids"]:
                    allowed_pairs.append({
                        "robot_geom_id": int(robot_geom),
                        "scene_geom_id": int(scene_geom),
                        "minimum_clearance_m": -0.002,
                    })
        forbidden_object_pairs = []
        for object_a, object_b in forbidden_pairs:
            if object_a not in self.bindings or object_b not in self.bindings:
                raise ValueError(f"BDDL_CheckContact_objects_are_unbound:{object_a}:{object_b}")
            forbidden_object_pairs.append({
                "object_id_a": object_a,
                "object_id_b": object_b,
                "body_ids_a": self.bindings[object_a]["collision_body_ids"],
                "body_ids_b": self.bindings[object_b]["collision_body_ids"],
                "geom_ids_a": self.bindings[object_a]["geom_ids"],
                "geom_ids_b": self.bindings[object_b]["geom_ids"],
            })
        self.safety_policy = {
            "source": "active_BDDL_CheckRobotContact_CheckContact_plus_registered_full_collision_geometry",
            "minimum_clearance_m": self.config.hard_minimum_clearance_m,
            "forbidden_objects": [group(name, self.config.hard_minimum_clearance_m) for name in forbidden_names],
            "allowed_target_objects": [
                group(name, 0.010)
                for name in target_names
                if self.bindings[name]["geom_ids"]
            ],
            "noncollidable_goal_objects": [
                name for name in target_names if not self.bindings[name]["geom_ids"]
            ],
            "other_scene_objects": other_groups,
            "allowed_contact_pairs": allowed_pairs,
            "forbidden_object_pairs": forbidden_object_pairs,
            "carried_object": None,
        }
        task_execution = self.config_data.get("task_execution", {})
        if task_execution.get("enabled") is True:
            if task_execution.get("goal_source_is_vlm_teacher") is not False:
                raise ValueError("task_expert_prior_must_not_be_mislabeled_as_VLM_Teacher_output")
            if task_execution.get("fixture_activation_controller") is not None:
                raise ValueError("configured_fixture_activation_controller_has_no_verified_runtime_adapter")
            self.task_controller = TaskPhaseController(task_execution)
            configured_objects = {
                str(item["object_id"]) for item in task_execution.get("objects", [])
            }
            if not configured_objects or not configured_objects.issubset(object_names):
                raise ValueError("task_phase_object_ids_must_bind_to_active_BDDL_objects")
            support_names = task_execution.get("support_contact_geom_names", [])
            if not isinstance(support_names, list) or not support_names:
                raise ValueError("task_phase_program_requires_explicit_support_contact_geometry")
            geometry_by_name: dict[str, list[int]] = {}
            for item in inventory["geometries"]:
                if item.get("collision_enabled") is True:
                    geometry_by_name.setdefault(str(item["geom_name"]), []).append(
                        int(item["geom_id"])
                    )
            resolved_support = []
            for geom_name in support_names:
                matches = geometry_by_name.get(str(geom_name), [])
                if len(matches) != 1:
                    raise ValueError(f"configured_support_geometry_is_not_unique:{geom_name}")
                resolved_support.append(matches[0])
            self.task_support_geom_ids = tuple(resolved_support)
            self.safety_policy["allowed_contact_pairs"] = []
        template_object = self.config_data["goal_object_semantic_id"]
        if template_object not in object_names or template_object not in self.bindings:
            raise ValueError("historical_pregrasp_target_is_not_a_declared_BDDL_object")
        if template_object not in target_names:
            raise ValueError("historical_pregrasp_target_is_not_a_BDDL_goal_target")
        if not self.bindings[template_object]["geom_ids"]:
            raise ValueError("historical_pregrasp_goal_object_has_no_collision_geometry")
        self.context_static = {
            "instruction": metadata["instruction"],
            "task_id": f"{metadata['benchmark']}/L{metadata['level']}T{metadata['task_id']}",
            "bddl_sha256": bddl_hash,
            "model_geometry_sha256": metadata["model_geometry_sha256"],
            "fixture_names": sorted(fixture_names),
            "object_names": sorted(object_names),
            "forbidden_names": forbidden_names,
            "target_names": target_names,
            "allowed_target_contact_names": target_contact_names,
            "unbound_noncritical_bddl_declarations": unbound_declared_entities,
            "forbidden_pairs": forbidden_pairs,
            "goal_object": template_object,
            "gripper_geometry_ids": gripper_geom_ids,
            "source_trace_sha256": self.archive_sha256,
        }
        if self.mode in {"t10-snapshot", "runtime-snapshot"}:
            if self.mode == "t10-snapshot":
                verification = json.loads(self.t10_verification_path.read_text(encoding="utf-8"))
                if verification.get("status") != "verified" or verification.get("timestep") != 10:
                    raise ValueError("archived_t10_replay_snapshot_verification_is_not_valid")
                if verification.get("model_geometry_sha256") != metadata["model_geometry_sha256"]:
                    raise ValueError("archived_t10_model_geometry_fingerprint_mismatch")
                if verification.get("source_trace_sha256") != self.archive_sha256:
                    raise ValueError("archived_t10_student_trace_fingerprint_mismatch")
                obstacle_scope_verified = bool(
                    verification.get("substep_contact_monitoring", {}).get("coverage_complete")
                    and verification.get("substep_contact_monitoring", {}).get("robot_bottle_or_box_contact_events") == 0
                )
                if not obstacle_scope_verified:
                    raise ValueError("archived_t10_history_has_no_complete_obstacle_safety_evidence")
            self.initial_history_safety_verified = False
        else:
            self.initial_history_safety_verified = True
        return self.context_static

    def _grasp_states(self) -> tuple[str | None, bool, dict[str, Any]]:
        if self.context_static is None:
            raise RuntimeError("task_semantics_are_not_initialized")
        states = {}
        held = []
        for object_name in self.context_static["object_names"]:
            binding = self.bindings[object_name]
            response = self.bridge.request({
                "op": "grasp_state",
                "expected_instruction": self.context_static["instruction"],
                "carried_binding": {
                    "object_id": object_name,
                    "body_id": binding["root_body_id"],
                    "body_name": binding["root_body_name"],
                    "geom_ids": binding["root_geom_ids"],
                    "geom_names": binding["root_geom_names"],
                },
            })
            states[object_name] = response
            if response.get("status") == "pass" and response.get("grasp_confirmed") is True:
                held.append(object_name)
            elif response.get("status") != "not_grasped":
                raise ValueError(
                    f"live_grasp_state_is_unknown_for_{object_name}:{response.get('reason') or response.get('error')}"
                )
        if len(held) > 1:
            raise ValueError(f"multiple_objects_are_reported_as_held:{held}")
        return (held[0] if held else None), True, states

    def _refresh_carried_geometry_policy(self, held_object: str | None) -> None:
        """Bind a verified held object so planning and physics checks track it."""
        if self.safety_policy is None:
            raise RuntimeError("verified_safety_policy_is_not_initialized")
        if held_object is None:
            self.safety_policy["carried_object"] = None
            return
        binding = self.bindings.get(held_object)
        if not isinstance(binding, dict) or not binding.get("geom_ids"):
            raise ValueError(f"held_object_has_no_verified_collision_geometry:{held_object}")
        if not binding.get("root_geom_ids"):
            raise ValueError(f"held_object_has_no_verified_root_geometry:{held_object}")
        self.safety_policy["carried_object"] = {
            "object_id": held_object,
            "grasp_binding": {
                "object_id": held_object,
                "body_id": int(binding["root_body_id"]),
                "body_name": binding["root_body_name"],
                "geom_ids": list(binding["root_geom_ids"]),
                "geom_names": list(binding["root_geom_names"]),
            },
            "allowed_contact_pairs": [],
        }

    def _goal_pose(self, data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        target_name = self.context_static["goal_object"]
        binding = self.bindings[target_name]
        key = f"{binding['mapping_name']}:{target_name}"
        object_state = data["snapshot_state"]["object_states"].get(key)
        if not isinstance(object_state, dict):
            raise ValueError(f"live_target_object_state_is_missing:{key}")
        object_position = np.asarray(object_state["root_pos"], dtype=np.float64)
        object_quaternion = np.asarray(object_state["root_quat"], dtype=np.float64)
        self._init_historical_goal_qpos()
        object_to_eef = self.historical_object_to_eef
        if object_position.shape != (3,) or object_quaternion.shape != (4,) or object_to_eef.shape != (4, 4):
            raise ValueError("historical_pregrasp_transform_or_live_object_pose_is_malformed")
        object_rotation = _quat_matrix_wxyz(object_quaternion)
        relative_position = object_to_eef[:3, 3]
        relative_rotation = object_to_eef[:3, :3]
        position = object_position + object_rotation @ relative_position
        quaternion = _matrix_quat_wxyz(object_rotation @ relative_rotation)
        return position, quaternion

    @staticmethod
    def _predicate_key(predicate: Any) -> tuple[str, ...]:
        if not isinstance(predicate, (list, tuple)) or not predicate:
            raise ValueError("native_task_predicate_must_be_a_nonempty_array")
        return tuple(str(value).lower() for value in predicate)

    @staticmethod
    def _pose_matrix(position: Any, quaternion_wxyz: Any) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = _quat_matrix_wxyz(quaternion_wxyz)
        result[:3, 3] = np.asarray(position, dtype=np.float64)
        if result.shape != (4, 4) or not np.isfinite(result).all():
            raise ValueError("task_phase_pose_is_invalid")
        return result

    @staticmethod
    def _matrix_pose(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError("task_phase_transform_is_invalid")
        return value[:3, 3].copy(), _matrix_quat_wxyz(value[:3, :3])

    def _live_object_state(self, data: dict[str, Any], object_name: str) -> dict[str, Any]:
        binding = self.bindings.get(object_name)
        if not isinstance(binding, dict):
            raise ValueError(f"task_phase_object_binding_is_missing:{object_name}")
        key = f"{binding['mapping_name']}:{object_name}"
        state = data.get("snapshot_state", {}).get("object_states", {}).get(key)
        if not isinstance(state, dict):
            raise ValueError(f"live_task_object_state_is_missing:{key}")
        return state

    def _live_eef_pose(self, data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        robots = data.get("snapshot_state", {}).get("robot_states", [])
        if len(robots) != 1:
            raise ValueError("task_phase_requires_exactly_one_live_robot_state")
        robot = robots[0]
        position = np.asarray(robot.get("eef_pos"), dtype=np.float64)
        quaternion = np.asarray(robot.get("eef_quat"), dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("task_phase_live_EEF_pose_is_malformed")
        return position, quaternion / np.linalg.norm(quaternion)

    def _native_region_query(self, object_spec: dict[str, Any]) -> dict[str, Any]:
        predicate = object_spec.get("predicate")
        query = self.bridge.request({"op": "native_goal_query", "predicate": predicate})
        if query.get("status") != "complete" or query.get("predicate_is_native_goal") is not True:
            raise ValueError(
                f"native_task_target_region_is_unknown:{query.get('error') or query.get('status')}"
            )
        region = query.get("region")
        if not isinstance(region, dict) or not region.get("parent_name"):
            raise ValueError("native_task_target_region_frame_is_incomplete")
        self.current_region_parent = str(region["parent_name"])
        return query

    def _placement_object_pose(
        self, data: dict[str, Any], object_spec: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        query = self._native_region_query(object_spec)
        region = query["region"]
        target = object_spec.get("target_pose_in_region")
        if not isinstance(target, dict):
            raise ValueError("task_goal_has_no_explicit_target_pose_in_native_region_frame")
        region_matrix = np.eye(4, dtype=np.float64)
        region_matrix[:3, :3] = np.asarray(region.get("site_rotation_world"), dtype=np.float64)
        region_matrix[:3, 3] = np.asarray(region.get("site_position_world_m"), dtype=np.float64)
        target_matrix = self._pose_matrix(
            target.get("position_m"), target.get("quaternion_wxyz")
        )
        position, quaternion = self._matrix_pose(region_matrix @ target_matrix)
        return position, quaternion, query

    def _placement_eef_pose(
        self, data: dict[str, Any], object_spec: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        target_position, target_quaternion, query = self._placement_object_pose(data, object_spec)
        object_state = self._live_object_state(data, str(object_spec["object_id"]))
        eef_position, eef_quaternion = self._live_eef_pose(data)
        object_world = self._pose_matrix(object_state["root_pos"], object_state["root_quat"])
        eef_world = self._pose_matrix(eef_position, eef_quaternion)
        object_to_eef = np.linalg.inv(object_world) @ eef_world
        target_world = self._pose_matrix(target_position, target_quaternion) @ object_to_eef
        position, quaternion = self._matrix_pose(target_world)
        return position, quaternion, query

    def _task_stage_target(
        self, data: dict[str, Any], phase: str
    ) -> dict[str, Any]:
        if self.task_controller is None:
            raise RuntimeError("task_phase_controller_is_not_enabled")
        object_spec = self.task_controller.current_object
        object_name = str(object_spec["object_id"])
        eef_position, eef_quaternion = self._live_eef_pose(data)
        gripper_open = float(self.config_data.get("open_gripper_action", -1.0))
        close_command = float(self.config_data["task_execution"].get("close_gripper_action", 1.0))
        mode = "osc_closed_loop"
        source = str(object_spec["goal_source"])
        allowed_contacts: tuple[str, ...] = ()
        stride = float(self.config_data["task_execution"].get("default_stride_m", 0.012))

        if phase == "pregrasp":
            # Use the archived verified grasp's object-frame approach vector
            # and distance. The earlier t=118 pose was reached within 9 mm in
            # the archived continuation, where a non-finger robot geom had
            # only 0.419 mm clearance. Treat it as history/IK seed, not a safe
            # pregrasp target. Retarget the verified approach from the current
            # live object pose.
            object_state = self._live_object_state(data, object_name)
            retargeted = self.task_grasp_template.retarget(
                object_state["root_pos"], object_state["root_quat"]
            )
            position = retargeted.pregrasp_position_world_m
            quaternion = retargeted.eef_quaternion_world_wxyz
            source = (
                "archived_verified_object_frame_grasp_and_approach_distance_"
                "retargeted_from_live_object_pose"
            )
            mode = "rrt_connect"
        elif phase in {"second_object_recovery", "approach"}:
            object_state = self._live_object_state(data, object_name)
            retargeted = self.task_grasp_template.retarget(
                object_state["root_pos"], object_state["root_quat"]
            )
            if phase == "second_object_recovery":
                position = retargeted.pregrasp_position_world_m
                quaternion = retargeted.eef_quaternion_world_wxyz
                mode = "rrt_connect"
            else:
                position = retargeted.eef_position_world_m
                quaternion = retargeted.eef_quaternion_world_wxyz
                allowed_contacts = (object_name,)
                stride = float(self.config_data["task_execution"].get("approach_stride_m", 0.010))
            source = "historical_object_frame_grasp_transform_retargeted_from_live_object_pose"
        elif phase in {"close_gripper", "stable_grasp", "settle", "verify_task"}:
            position, quaternion = eef_position, eef_quaternion
            if phase in {"close_gripper", "stable_grasp"}:
                allowed_contacts = (object_name,)
        elif phase == "lift":
            if self.task_phase_anchor is None:
                raise ValueError("lift_phase_has_no_live_EEF_anchor")
            position = self.task_phase_anchor[0] + np.asarray(
                [0.0, 0.0, float(self.config_data["task_execution"].get("lift_height_m", 0.045))]
            )
            quaternion = self.task_phase_anchor[1]
            stride = float(self.config_data["task_execution"].get("lift_stride_m", 0.025))
        elif phase == "transport":
            position, quaternion, _query = self._placement_eef_pose(data, object_spec)
            position = position + np.asarray(
                [0.0, 0.0, float(self.config_data["task_execution"].get("transport_clearance_m", 0.14))]
            )
            mode = "rrt_connect"
        elif phase in {"precise_placement", "release"}:
            position, quaternion, _query = self._placement_eef_pose(data, object_spec)
            if phase == "precise_placement":
                stride = float(self.config_data["task_execution"].get("descent_stride_m", 0.010))
        elif phase == "withdrawal":
            if self.task_phase_anchor is None:
                raise ValueError("withdrawal_phase_has_no_live_EEF_anchor")
            position = self.task_phase_anchor[0] + np.asarray(
                [0.0, 0.0, float(self.config_data["task_execution"].get("withdrawal_height_m", 0.10))]
            )
            quaternion = self.task_phase_anchor[1]
            mode = "rrt_connect"
            allowed_contacts = (object_name,)
        else:
            raise ValueError(f"task_phase_target_is_undefined:{phase}")

        if phase in {"close_gripper", "stable_grasp", "lift", "transport", "precise_placement"}:
            gripper_command = close_command
        elif phase in {"release", "withdrawal", "settle", "verify_task"}:
            gripper_command = gripper_open
        else:
            gripper_command = float(self.task_grasp_template.open_command)
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            raise ValueError("task_phase_target_pose_is_not_finite")
        return {
            "object_spec": object_spec,
            "object_id": object_name,
            "phase": phase,
            "position": np.asarray(position, dtype=np.float64),
            "quaternion": np.asarray(quaternion, dtype=np.float64),
            "gripper_command": float(np.clip(gripper_command, -1.0, 1.0)),
            "mode": mode,
            "source": source,
            "allowed_contacts": allowed_contacts,
            "stride_m": stride,
        }

    def _pose_error(self, data: dict[str, Any], target: dict[str, Any]) -> tuple[float, float]:
        position, quaternion = self._live_eef_pose(data)
        position_error = float(np.linalg.norm(position - target["position"]))
        current = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        desired = Rotation.from_quat(target["quaternion"][[1, 2, 3, 0]])
        rotation_error = float((desired * current.inv()).magnitude())
        return position_error, rotation_error

    def _refresh_task_contact_policy(
        self, phase: str, object_name: str, held_object: str | None
    ) -> None:
        finger_contact_phases = {
            "approach", "close_gripper", "stable_grasp", "withdrawal"
        }
        target = self.bindings[object_name]
        self.safety_policy["allowed_contact_pairs"] = [
            {
                "robot_geom_id": int(robot_geom),
                "scene_geom_id": int(scene_geom),
                "minimum_clearance_m": -0.002,
            }
            for robot_geom in self.context_static["gripper_geometry_ids"]
            for scene_geom in target["geom_ids"]
        ] if phase in finger_contact_phases else []
        self._refresh_carried_geometry_policy(held_object)
        if held_object is None:
            return
        carried = self.safety_policy.get("carried_object")
        if not isinstance(carried, dict):
            raise ValueError("live_held_object_has_no_carried_geometry_policy")
        pairs = []
        if phase in {"precise_placement", "release"}:
            parent = self.current_region_parent
            fixture = self.bindings.get(parent) if parent else None
            if not isinstance(fixture, dict) or not fixture.get("geom_ids"):
                raise ValueError("native_placement_support_has_no_verified_collision_geometry")
            support_ids = [
                geom_id for geom_id in self.task_support_geom_ids
                if geom_id in set(fixture["geom_ids"])
            ]
            if not support_ids:
                raise ValueError("configured_support_geometry_does_not_belong_to_native_target_fixture")
            pairs = [
                {
                    "carried_geom_id": int(carried_geom),
                    "scene_geom_id": int(scene_geom),
                    "minimum_clearance_m": -0.002,
                }
                for carried_geom in self.bindings[held_object]["geom_ids"]
                for scene_geom in support_ids
            ]
        carried["allowed_contact_pairs"] = pairs

    def _task_phase_context(
        self,
        held: str | None,
        grasp_verified: bool,
        grasp_records: dict[str, Any],
        latest: dict[str, Any],
    ) -> TaskContext:
        if self.task_controller is None:
            raise RuntimeError("task_phase_controller_is_not_enabled")
        status = self.bridge.request({"op": "native_task_status"})
        if status.get("status") != "complete" or status.get("task_success") is None:
            raise ValueError(f"native_task_goal_status_is_unknown:{status.get('error')}")
        native_values = {
            self._predicate_key(item.get("predicate")): bool(item.get("value"))
            for item in status.get("native_goal_values", [])
            if isinstance(item, dict) and isinstance(item.get("value"), bool)
        }
        task_config = self.config_data["task_execution"]
        for item in task_config["objects"]:
            predicate_key = self._predicate_key(item["predicate"])
            if predicate_key not in native_values:
                raise ValueError(f"configured_task_goal_is_not_reported_by_native_BDDL:{predicate_key}")
        expected_all = {
            self._predicate_key(item["predicate"]) for item in task_config["objects"]
        }
        if task_config.get("fixture_predicate"):
            expected_all.add(self._predicate_key(task_config["fixture_predicate"]))
        if not expected_all.issubset(native_values):
            raise ValueError("configured_task_goal_set_differs_from_native_BDDL_goal_set")
        self.current_native_task_status = status
        self.current_native_goal_values = native_values

        if held is not None and not self.task_expert_active:
            self.task_controller.adopt_held_object(held)
        elif held is not None and held != self.task_controller.current_object["object_id"]:
            raise ValueError(f"held_object_does_not_match_active_task_phase:{held}")
        for item in task_config["objects"]:
            predicate_key = self._predicate_key(item["predicate"])
            object_name = str(item["object_id"])
            if native_values[predicate_key]:
                self.confirmed_placed_objects.add(object_name)
            elif object_name in self.confirmed_placed_objects:
                raise ValueError(f"previously_placed_native_task_predicate_was_lost:{predicate_key}")

        phase = self.task_controller.phase
        target = self._task_stage_target(latest, phase)
        position_error, rotation_error = self._pose_error(latest, target)
        tolerances = task_config.get("phase_pose_tolerances", {}).get(phase, {})
        goal_reached = bool(
            position_error <= float(tolerances.get("position_m", self.config_data.get("goal_position_tolerance_m", 0.012)))
            and rotation_error <= float(tolerances.get("orientation_rad", self.config_data.get("goal_orientation_tolerance_rad", 0.08)))
        )
        object_state = self._live_object_state(latest, str(target["object_id"]))
        object_qvel = np.asarray(object_state.get("joint_qvel", []), dtype=np.float64)
        eef_position, _eef_quaternion = self._live_eef_pose(latest)
        object_predicate = self._predicate_key(target["object_spec"]["predicate"])
        all_goals_satisfied = all(bool(native_values[key]) for key in expected_all)
        satisfied_object_ids = tuple(
            str(item["object_id"])
            for item in task_config["objects"]
            if bool(native_values[self._predicate_key(item["predicate"])])
        )
        fixture_key = (
            self._predicate_key(task_config["fixture_predicate"])
            if task_config.get("fixture_predicate") else None
        )
        evidence = PhaseEvidence(
            timestep=int(latest.get("timestep", -1)),
            eef_goal_reached=goal_reached,
            held_object=held,
            grasp_confirmed=bool(
                grasp_records.get(str(target["object_id"]), {}).get("grasp_confirmed") is True
            ),
            object_position_world_m=tuple(float(x) for x in object_state["root_pos"]),
            object_quaternion_wxyz=tuple(float(x) for x in object_state["root_quat"]),
            object_qvel=tuple(float(x) for x in object_qvel),
            eef_position_world_m=tuple(float(x) for x in eef_position),
            task_goal_satisfied=bool(native_values[object_predicate]),
            all_task_goals_satisfied=all_goals_satisfied,
            satisfied_object_ids=satisfied_object_ids,
            fixture_predicate_satisfied=(
                bool(native_values[fixture_key]) if fixture_key is not None else None
            ),
            trusted_fixture_activation_available=False,
            task_success=bool(status["task_success"]),
        )
        previous_phase = self.task_controller.phase
        recorded_previous_phase = self.last_task_phase
        transition = self.task_controller.update(evidence)
        phase = self.task_controller.phase
        if phase != self.last_task_phase:
            self.task_phase_anchor = (eef_position.copy(), _eef_quaternion.copy())
            self.last_task_phase = phase
            self.artifacts.event({
                "event": "task_phase_transition",
                "timestep": evidence.timestep,
                "from_phase": previous_phase if transition is not None else recorded_previous_phase,
                "to_phase": phase,
                "object_id": target["object_id"],
                "transition_evidence": transition,
                "predicate": list(object_predicate),
                "predicate_value": evidence.task_goal_satisfied,
                "task_success": evidence.task_success,
                "target_source": target["source"],
            })
        elif self.last_task_phase is None:
            self.task_phase_anchor = (eef_position.copy(), _eef_quaternion.copy())
            self.last_task_phase = phase
            self.artifacts.event({
                "event": "task_phase_entered",
                "timestep": evidence.timestep,
                "phase": phase,
                "object_id": target["object_id"],
                "target_source": target["source"],
            })

        phase = self.task_controller.phase
        target = self._task_stage_target(latest, phase)
        self._refresh_task_contact_policy(phase, target["object_id"], held)
        if phase == "verify_task" and status["task_success"] is not True:
            raise ValueError("all_task_stage_predicates_true_but_native_task_success_is_false")
        source_artifacts = tuple(
            str(value)
            for value in (
                target["object_spec"].get("goal_source_artifact"),
                self.grasp_template_path,
                self.archived_trace_path,
            )
            if value
        )
        goal = GoalSpec(
            goal_id=f"{target['object_id']}:{phase}",
            phase=phase,
            source=target["source"],
            source_artifacts=source_artifacts,
            position_world_m=tuple(float(x) for x in target["position"]),
            quaternion_wxyz=tuple(float(x) for x in target["quaternion"]),
            allowed_target_contacts=target["allowed_contacts"],
        )
        expert_active = self.task_expert_active
        task_expert_only_snapshot = self.mode == "runtime-snapshot"
        return TaskContext(
            task_id=self.context_static["task_id"],
            instruction=self.context_static["instruction"],
            forbidden_contact_objects=tuple(self.context_static["forbidden_names"]),
            allowed_target_contact_objects=tuple(self.context_static["allowed_target_contact_names"]),
            held_object=held,
            phase=phase,
            target_source=target["source"],
            safety_policy_source=self.safety_policy["source"],
            goals=(goal,),
            semantic_bindings_verified=True,
            safety_constraints_verified=True,
            grasp_state_verified=grasp_verified,
            expert_handoff_available=True,
            # Direct runtime-Snapshot continuation has no Student policy
            # instance. If the saved pregrasp pose is even slightly outside
            # tolerance, route it through the configured RRT phase. The t=10
            # replay mode still consumes its archived Student actions before
            # takeover; full episodes load the unchanged live Student.
            handoff_policy=(
                "expert" if expert_active or task_expert_only_snapshot else "student"
            ),
            post_recovery_policy="expert",
            metadata={
                "task_phase_source": task_config["provenance"],
                "task_goal_source": target["source"],
                "goal_source_is_vlm_teacher": False,
                "task_phase": phase,
                "task_object": target["object_id"],
                "phase_execution_mode": target["mode"],
                "requires_rrt_recovery": target["mode"] == "rrt_connect",
                "phase_goal_position_error_m": position_error,
                "phase_goal_orientation_error_rad": rotation_error,
                "native_task_status": status,
                "held_object_state_evidence": grasp_records,
                "allowed_phase_contacts": list(target["allowed_contacts"]),
                "initial_history_safety_verified": self.initial_history_safety_verified,
                "initial_history_safety_scope": (
                    "full_episode_from_timestep_zero" if self.mode == "episode"
                    else "partial_archived_state_after_unverified_history"
                ),
            },
        )

    def current_context(self) -> TaskContext:
        if self.context_static is None:
            data = self.bridge.request({"op": "observe", "include_snapshot_state": True})
            self.latest_data = data
            self._load_context_inputs(data)
        held, grasp_verified, grasp_records = self._grasp_states()
        self._refresh_carried_geometry_policy(held)
        if held is not None:
            phase = self.config_data.get("held_phase", "transport")
        elif self.student_handoff_completed:
            phase = self.config_data.get("post_pregrasp_handoff_phase", "grasp")
        else:
            phase = self.config_data["phase"]
        latest = self.latest_data
        if latest is None:
            latest = self.bridge.request({
                "op": "observe",
                "include_snapshot_state": True,
                "safety_geometry_policy": self.safety_policy,
            })
            self.latest_data = latest
        if self.task_controller is not None:
            return self._task_phase_context(held, grasp_verified, grasp_records, latest)
        position, quaternion = self._goal_pose(latest)
        goal = GoalSpec(
            goal_id=f"{self.context_static['goal_object']}:historical_success_pregrasp",
            phase=self.config_data["phase"],
            source="historical_successful_episode_pregrasp_state_recomputed_from_current_target_pose",
            source_artifacts=(
                str(self.archived_trace_path),
                str(self.output_dir / "historical_pregrasp_transform.json"),
            ),
            position_world_m=tuple(float(value) for value in position),
            quaternion_wxyz=tuple(float(value) for value in quaternion),
            allowed_target_contacts=(self.context_static["goal_object"],),
        )
        unresolved = tuple(self.config_data.get("unresolved_goal_phases", ()))
        return TaskContext(
            task_id=self.context_static["task_id"],
            instruction=self.context_static["instruction"],
            forbidden_contact_objects=tuple(self.context_static["forbidden_names"]),
            allowed_target_contact_objects=tuple(
                self.context_static["allowed_target_contact_names"]
            ),
            held_object=held,
            phase=phase,
            target_source=goal.source,
            safety_policy_source=self.safety_policy["source"],
            goals=(goal,),
            semantic_bindings_verified=True,
            safety_constraints_verified=True,
            grasp_state_verified=grasp_verified,
            expert_handoff_available=False,
            handoff_policy="student",
            unresolved_goal_phases=unresolved,
            metadata={
                "initial_history_safety_verified": self.initial_history_safety_verified,
                "initial_history_safety_scope": (
                    "full_episode_from_timestep_zero"
                    if self.mode == "episode"
                    else "partial_archived_robot_bottle_box_contact_evidence_only"
                ),
                "task_context_source": "native_BDDL_constraints_and_live_registered_geometry",
                "goal_source_is_historical_expert_prior": True,
                "goal_source_is_vlm_teacher": False,
                "held_object_state_evidence": grasp_records,
                "carried_object_geometry_source": (
                    "live_verified_Panda_grasp_and_current_MuJoCo_eef_to_object_transform"
                    if held is not None else None
                ),
                "unresolved_task_stages": list(unresolved),
                "live_student_inference_available": self.student is not None,
            },
        )

    def _observation(self, data: dict[str, Any]) -> Observation:
        snapshot = data.get("snapshot_state")
        if not isinstance(snapshot, dict) or not snapshot.get("robot_states"):
            raise ValueError("Bridge_observation_has_no_complete_robot_snapshot")
        monitor = data.get("contact_monitoring")
        safety = data.get("safety_geometry")
        if monitor is None:
            safety = safety or {}
            geometry_status = safety.get("status")
            complete_now = geometry_status in {"complete", "unsafe"}
            forbidden = tuple(safety.get("forbidden_contacts", [])) + tuple(
                safety.get("forbidden_object_contacts", [])
            ) + tuple(safety.get("unallowed_scene_contacts", []))
            self_contacts = tuple(safety.get("self_contacts", []))
            minimum = safety.get("minimum_forbidden_clearance_m")
            physics_substeps: tuple[dict[str, Any], ...] = ()
            contacts_complete = complete_now
            clearance_complete = complete_now
            expected_substeps = None
        else:
            geometry = monitor.get("geometry_monitoring", {})
            raw_substeps = geometry.get("substeps", [])
            expected_substeps = monitor.get("expected_physics_substeps_per_control_step")
            geometry_complete = bool(
                expected_substeps
                and len(raw_substeps) == expected_substeps
                and all(item.get("status") in {"complete", "unsafe"} for item in raw_substeps)
            )
            contacts_complete = bool(
                monitor.get("status") == "complete"
                and monitor.get("captured_physics_substeps") == expected_substeps
            )
            clearance_complete = geometry_complete
            forbidden = tuple(
                item
                for sample in raw_substeps
                for field in (
                    "forbidden_contacts",
                    "forbidden_object_contacts",
                    "unallowed_scene_contacts",
                )
                for item in sample.get(field, [])
            )
            self_contacts = tuple(
                item for sample in raw_substeps for item in sample.get("self_contacts", [])
            )
            minimums = [
                float(sample["minimum_clearance_m"])
                for sample in raw_substeps
                if sample.get("minimum_clearance_m") is not None
            ]
            minimum = min(minimums) if minimums else None
            physics_substeps = tuple(
                {
                    "status": "complete" if sample.get("status") in {"complete", "unsafe"} else "incomplete",
                    "safety_status": sample.get("status"),
                    "minimum_clearance_m": sample.get("minimum_clearance_m"),
                    "minimum_all_clearance_m": sample.get("minimum_all_clearance_m"),
                    "clearances_by_object": sample.get("clearances_by_object", {}),
                    "forbidden_contacts": sample.get("forbidden_contacts", []),
                    "forbidden_object_contacts": sample.get("forbidden_object_contacts", []),
                    "unallowed_scene_contacts": sample.get("unallowed_scene_contacts", []),
                    "self_contacts": sample.get("self_contacts", []),
                    "allowed_contact_clearances_m": sample.get("allowed_contact_clearances_m", {}),
                }
                for sample in raw_substeps
            )
        if safety is not None and safety.get("status") == "incomplete":
            clearance_complete = False
        state = snapshot["robot_states"][0]
        raw = {
            "timestep": int(data.get("timestep", 0)),
            "expected_physics_substeps": expected_substeps,
            "snapshot_state": snapshot,
            "safety_geometry": safety,
            "contact_monitoring": monitor,
            "task_success": data.get("task_success"),
            "instruction": data.get("instruction"),
            "scene_metadata": data.get("scene_metadata"),
            "native_task_status": data.get("native_task_status"),
        }
        if "reference_qpos" in data:
            raw["reference_qpos"] = data["reference_qpos"]
        if "reference_eef_pose" in data:
            raw["reference_eef_pose"] = data["reference_eef_pose"]
        observation = Observation(
            timestep=int(data.get("timestep", 0)),
            qpos=np.asarray(state["joint_qpos"], dtype=np.float64),
            eef_position_world_m=np.asarray(state["eef_pos"], dtype=np.float64),
            eef_quaternion_wxyz=np.asarray(state["eef_quat"], dtype=np.float64),
            minimum_forbidden_clearance_m=minimum,
            clearance_complete=clearance_complete,
            contacts_complete=contacts_complete,
            forbidden_contacts=forbidden,
            self_contacts=self_contacts,
            physics_substeps=physics_substeps,
            reward=float(data.get("reward", 0.0)),
            done=bool(data.get("done", False)),
            task_success=data.get("task_success"),
            raw=raw,
        )
        self.latest_observation = observation
        return observation

    def observe(self) -> Observation:
        data = self.bridge.request({
            "op": "observe",
            "include_snapshot_state": True,
            "safety_geometry_policy": self.safety_policy,
        })
        if self.task_controller is not None:
            cached = self.current_native_task_status
            if not isinstance(cached, dict) or cached.get("timestep") != data.get("timestep"):
                cached = self.bridge.request({"op": "native_task_status"})
            data["native_task_status"] = cached
            self.current_native_task_status = cached
        self.latest_data = data
        if not self.context_static:
            self._load_context_inputs(data)
        return self._observation(data)

    def student_action(self, observation: Observation) -> np.ndarray:
        self._current_action_source = "student"
        if self.mode == "t10-snapshot" and not self.live_student_handoff:
            if self.archived_cursor >= len(self.archived_actions):
                raise RuntimeError("archived_Student_trace_has_no_action_at_current_state")
            start = self.archived_cursor
            end = min(len(self.archived_actions), start + self.config.segment_control_steps)
            self.current_forecast = self.archived_actions[start:end].copy()
            self.archived_cursor += 1
            return self.current_forecast[0].copy()
        if self.student is None:
            raise RuntimeError("live_Student_inference_is_not_available_after_recovery_handoff")
        if self.student_chunk_cursor >= len(self.student_chunk):
            self.student_chunk = self.student.predict(self.latest_data)
            self.student_chunk_cursor = 0
        start = self.student_chunk_cursor
        end = min(len(self.student_chunk), start + self.config.segment_control_steps)
        self.current_forecast = self.student_chunk[start:end].copy()
        self.student_chunk_cursor += 1
        return self.current_forecast[0].copy()

    def _save_runtime_snapshot(self, payload: bytes, relative: str) -> str:
        path = self.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)

    def predict_student_risk(self, observation: Observation, action: np.ndarray, context: TaskContext) -> dict[str, Any]:
        forecast = np.asarray(self.current_forecast, dtype=np.float32)
        if forecast.ndim != 2 or forecast.shape[1:] != (7,) or len(forecast) == 0:
            return {"status": "unavailable", "reason": "student_prediction_prefix_is_missing"}
        if not np.array_equal(forecast[0], np.asarray(action, dtype=np.float32)):
            return {"status": "unavailable", "reason": "student_action_differs_from_risk_prefix"}
        start = self.bridge.request({"op": "snapshot"})
        snapshot_id = start.get("snapshot_id")
        if not snapshot_id:
            return {"status": "unavailable", "reason": "risk_snapshot_creation_failed"}
        branch = self.bridge.request({
            "op": "branch_rollout",
            "snapshot_id": snapshot_id,
            "candidates": [{"candidate_id": "student-risk-prefix", "actions": forecast.tolist()}],
            "safety_geometry_policy": self.safety_policy,
            "capture_runtime_snapshots": True,
        })
        if not (
            branch.get("main_state_restored_verified") is True
            or branch.get("main_restore_verified") is True
        ):
            return {"status": "unavailable", "reason": "student_risk_branch_main_restore_unverified"}
        candidates = branch.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            return {"status": "unavailable", "reason": "student_risk_branch_result_is_malformed"}
        candidate = candidates[0]
        monitor = candidate.get("contact_monitoring", {})
        steps = candidate.get("steps", [])
        geometry = monitor.get("geometry_monitoring", {})
        substeps = geometry.get("substeps", [])
        coverage = len(steps) == len(forecast) and risk_branch_coverage_is_complete(
            monitor, len(forecast)
        )
        geometry_complete = geometry.get("status") in {"complete", "unsafe"}
        if not coverage or not geometry_complete or candidate.get("status") != "ok":
            return {
                "status": "unavailable",
                "reason": "student_risk_branch_physics_or_geometry_coverage_incomplete",
                "candidate_status": candidate.get("status"),
                "contact_monitoring": monitor,
            }
        minima = [
            float(item["minimum_clearance_m"])
            for item in substeps
            if item.get("minimum_clearance_m") is not None
        ]
        if not minima:
            return {"status": "unavailable", "reason": "student_risk_branch_has_no_forbidden_geometry_distance"}
        snapshot_paths = []
        for index, step in enumerate(steps):
            payload = step.get("terminal_runtime_snapshot")
            if not isinstance(payload, bytes) or not payload:
                return {"status": "unavailable", "reason": "student_risk_branch_snapshot_missing"}
            snapshot_paths.append(
                self._save_runtime_snapshot(
                    payload,
                    f"branch_snapshots/student_risk_step_{observation.timestep:06d}_{index:02d}.pkl",
                )
            )
        return {
            "status": "complete",
            "minimum_predicted_forbidden_clearance_m": (
                0.0 if geometry.get("status") == "unsafe" else min(minima)
            ),
            "predicted_safety_status": geometry.get("status"),
            "prediction_horizon_control_steps": len(forecast),
            "prediction_source": "isolated_Snapshot_rollout_of_native_7D_Student_action_chunk",
            "candidate_student_action_prefix": forecast.tolist(),
            "substep_coverage": monitor,
            "main_state_restored_verified": True,
            "branch_snapshot_paths": snapshot_paths,
        }

    def execute_action(self, action: np.ndarray, context: TaskContext) -> Observation:
        current = self.latest_data
        if current is None:
            current = self.bridge.request({
                "op": "observe",
                "include_snapshot_state": True,
                "safety_geometry_policy": self.safety_policy,
            })
        source = self._current_action_source
        transition_index = self.transition_count
        chunk_index = transition_index
        safety_record = {
            "training_label_valid": False,
            "action_source": source,
            "task_phase": context.phase,
            "risk_prediction_horizon_control_steps": len(self.current_forecast),
            "native_task_status": current.get("native_task_status"),
        }
        predicted_actions = (
            self.current_forecast.copy()
            if source == "student" and len(self.current_forecast)
            else np.asarray(action, dtype=np.float32)[None, :]
        )
        self.trajectory.add_decision(
            chunk_index,
            int(current.get("timestep", 0)),
            current,
            predicted_actions,
            safety_record=safety_record,
        )
        result = self.bridge.request({
            "op": "step_monitored",
            "action": np.asarray(action, dtype=np.float32).tolist(),
            "action_index": int(current.get("timestep", 0)),
            "safety_geometry_policy": self.safety_policy,
        })
        if self.task_controller is not None:
            result["native_task_status"] = self.bridge.request({"op": "native_task_status"})
            self.current_native_task_status = result["native_task_status"]
        self.latest_data = result
        if source == "safety_recovery_osc_pose":
            self.plan_cursor += 1
        self.trajectory.add_step(
            int(result.get("timestep", 0)) - 1,
            chunk_index,
            0,
            np.asarray(action, dtype=np.float32),
            result,
        )
        self._save_video_frame(result)
        snapshot = self.bridge.request({"op": "export_runtime_snapshot"})
        if snapshot.get("status") != "complete" or not isinstance(snapshot.get("payload"), bytes):
            raise RuntimeError("post_transition_runtime_snapshot_export_failed")
        snapshot_path = self._save_runtime_snapshot(
            snapshot["payload"],
            f"snapshots/main_timestep_{int(result.get('timestep', 0)):06d}.pkl",
        )
        observation = self._observation(result)
        observation.raw["snapshot_path"] = snapshot_path
        self.transition_count += 1
        return observation

    def resolve_goal(self, observation: Observation, context: TaskContext) -> GoalSpec:
        if context.phase in context.unresolved_goal_phases:
            raise ValueError(f"task_goal_phase_unresolved:{context.phase}")
        return context.goal_for_phase(context.phase)

    def plan_rrt_connect(self, observation: Observation, goal: GoalSpec, context: TaskContext, config: RecoveryConfig) -> JointSpacePlan:
        if self.task_controller is not None:
            phase_limit = int(
                self.config_data["task_execution"].get("maximum_plan_attempts_per_phase", 3)
            )
            attempts = self.phase_plan_attempts.get(context.phase, 0)
            if attempts >= phase_limit:
                raise RuntimeError(f"task_phase_planning_attempt_budget_exhausted:{context.phase}")
            self.phase_plan_attempts[context.phase] = attempts + 1
        integration_state = np.asarray(
            observation.raw["snapshot_state"]["mujoco_integration_state"], dtype=np.float64
        )
        start_hash = hashlib.sha256(integration_state.tobytes()).hexdigest()
        self.plan_attempt += 1
        rrt = dict(self.config_data.get("rrt_connect", {}))
        rrt["seed"] = int(rrt.get("seed", 20261023)) + self.plan_attempt - 1
        result = self.bridge.request({
            "op": "rrt_connect",
            "expected_instruction": context.instruction,
            "start_state_sha256": start_hash,
            "model_geometry_sha256": self.context_static["model_geometry_sha256"],
            "goal_pose_world": {
                "position_world_m": list(goal.position_world_m),
                "quaternion_wxyz": list(goal.quaternion_wxyz),
            },
            "safety_geometry_policy": self.safety_policy,
            "ik_seed_qpos": [self.historical_goal_qpos.tolist()],
            "planner_config": rrt,
        })
        if result.get("status") != "offline_feasible" or result.get("live_state_unchanged") is not True:
            raise RuntimeError(
                f"planner_status_{result.get('status')}:{result.get('failure_reason') or result.get('route_candidates')}"
            )
        self.plan_counter += 1
        path = np.asarray(result["qpos_path"], dtype=np.float64)
        self._plan_poses = np.asarray(result["eef_reference_poses_world"], dtype=np.float64)
        self.plan_cursor = 0
        self.artifacts.event({
            "event": "planner_result_details",
            "plan_index": self.plan_counter,
            "task_phase": context.phase,
            "phase_plan_attempt": self.phase_plan_attempts.get(context.phase),
            "planner_result": result,
        })
        return JointSpacePlan(
            qpos_path=path,
            minimum_forbidden_clearance_m=float(result["minimum_forbidden_clearance_m"]),
            planner_status=result["status"],
            planner_source=result["planner_source"],
            planning_seconds=float(result["planning_seconds"]),
            diagnostics={
                key: value for key, value in result.items()
                if key not in {"qpos_path", "eef_reference_poses_world"}
            },
        )

    def _init_historical_goal_qpos(self):
        if hasattr(self, "historical_goal_qpos"):
            return
        source_trace = np.load(self.archived_trace_path, allow_pickle=False)
        source_step = int(self.config_data["historical_goal_seed_timestep"]) - 1
        if source_step < 0 or source_step >= len(source_trace["post_physical_state_json"]):
            raise ValueError("historical_successful_grasp_record_step_is_out_of_range")
        source_state = json.loads(source_trace["post_physical_state_json"][source_step])
        target_name = self.config_data["goal_object_semantic_id"]
        source_object = source_state["object_states"].get(f"objects_dict:{target_name}")
        source_robot = source_state["robot_states"][0]
        if not isinstance(source_object, dict):
            raise ValueError("historical_pregrasp_state_has_no_target_object_pose")
        contact_record = json.loads(source_trace["post_contacts_json"][source_step])
        target_bodies = set(self.bindings[target_name]["body_ids"])
        gripper_geoms = set(self.context_static["gripper_geometry_ids"])
        if contact_record.get("status") == "unavailable":
            raise ValueError("historical_pregrasp_contact_evidence_is_unavailable")
        target_gripper_contacts = []
        for contact in contact_record.get("contacts", []):
            body_ids = {
                int(contact["body1"]["id"]),
                int(contact["body2"]["id"]),
            }
            geom_ids = {
                int(contact["geom1"]["id"]),
                int(contact["geom2"]["id"]),
            }
            if body_ids & target_bodies and geom_ids & gripper_geoms:
                target_gripper_contacts.append(contact)
        if target_gripper_contacts:
            raise ValueError("historical_pregrasp_state_already_has_target_finger_contact")
        source_object_rotation = _quat_matrix_wxyz(source_object["root_quat"])
        source_eef_rotation = _quat_matrix_wxyz(source_robot["eef_quat"])
        relative = np.eye(4, dtype=np.float64)
        relative[:3, :3] = source_object_rotation.T @ source_eef_rotation
        relative[:3, 3] = source_object_rotation.T @ (
            np.asarray(source_robot["eef_pos"], dtype=np.float64)
            - np.asarray(source_object["root_pos"], dtype=np.float64)
        )
        if not np.isfinite(relative).all():
            raise ValueError("historical_pregrasp_relative_transform_is_not_finite")
        self.historical_object_to_eef = relative
        self.historical_goal_qpos = np.asarray(
            source_robot["joint_qpos"], dtype=np.float64
        )
        if self.historical_goal_qpos.shape != (7,) or not np.isfinite(self.historical_goal_qpos).all():
            raise ValueError("historical_successful_grasp_record_has_no_finite_7D_qpos_seed")
        derived = {
            "schema_version": "gs_opd_historical_pregrasp_transform.v1",
            "source": "archived_pregrasp_state_before_the_later_stable_grasp_record",
            "source_trace": str(self.archived_trace_path),
            "source_trace_sha256": self.archive_sha256,
            "source_timestep": int(self.config_data["historical_goal_seed_timestep"]),
            "source_first_finger_contact_timestep": int(
                self.grasp_template["source_first_finger_contact_step_1based"]
            ),
            "target_object_semantic_id": target_name,
            "object_to_eef": relative.tolist(),
            "source_eef_pose_world": {
                "position_m": source_robot["eef_pos"],
                "quaternion_wxyz": source_robot["eef_quat"],
            },
            "source_target_object_pose_world": {
                "position_m": source_object["root_pos"],
                "quaternion_wxyz": source_object["root_quat"],
            },
            "historical_qpos_used_only_as_IK_seed": self.historical_goal_qpos.tolist(),
            "VLM_Teacher_generated": False,
        }
        (self.output_dir / "historical_pregrasp_transform.json").write_text(
            json.dumps(derived, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def preview_prefix(self, observation: Observation, plan: JointSpacePlan, context: TaskContext, config: RecoveryConfig) -> PrefixPreview:
        self._current_action_source = "safety_recovery_osc_pose"
        self.current_forecast = np.empty((0, 7), dtype=np.float32)
        if not hasattr(self, "_plan_poses") or len(self._plan_poses) != len(plan.qpos_path):
            return PrefixPreview(np.empty((0, 7)), [], False, False, False, "planner_EEF_reference_poses_are_missing")
        snapshot = self.bridge.request({"op": "snapshot"})
        snapshot_id = snapshot.get("snapshot_id")
        if not snapshot_id:
            return PrefixPreview(np.empty((0, 7)), [], False, False, False, "active_physical_state_Snapshot_failed")
        request = {
            "op": "osc_prefix_probe",
            "snapshot_id": snapshot_id,
            "current_state_sha256": hashlib.sha256(
                np.asarray(observation.raw["snapshot_state"]["mujoco_integration_state"], dtype=np.float64).tobytes()
            ).hexdigest(),
            "jointspace_path_qpos": plan.qpos_path.tolist(),
            "eef_reference_poses_world": self._plan_poses.tolist(),
            "path_start_index": self.plan_cursor,
            "max_control_steps": config.segment_control_steps,
            "gripper_action": float(self.config_data.get("open_gripper_action", -1.0)),
            "risk_trigger_distance_m": config.risk_trigger_distance_m,
            "maximum_joint_path_deviation_rad": config.maximum_joint_path_deviation_rad,
            "maximum_eef_tracking_error_m": config.maximum_eef_tracking_error_m,
            "maximum_eef_orientation_error_rad": config.maximum_eef_orientation_error_rad,
            "current_minimum_forbidden_clearance_m": observation.minimum_forbidden_clearance_m,
            "safety_geometry_policy": self.safety_policy,
        }
        result = self.bridge.request(request)
        main_restored = result.get("main_state_restored_verified") is True
        steps = result.get("steps", [])
        actions = np.asarray(result.get("actions", []), dtype=np.float32)
        observations = []
        snapshot_paths = []
        for index, step in enumerate(steps):
            record = dict(step["observation"])
            record["snapshot_state"] = step["observation"].get("snapshot_state")
            record["contact_monitoring"] = {
                "status": "complete",
                "expected_physics_substeps_per_control_step": 25,
                "captured_physics_substeps": len(step.get("safety_geometry_substeps", [])),
                "geometry_monitoring": {
                    "status": "complete" if all(
                        item.get("status") in {"complete", "unsafe"}
                        for item in step.get("safety_geometry_substeps", [])
                    ) else "incomplete",
                    "substeps": step.get("safety_geometry_substeps", []),
                },
            }
            record["safety_geometry"] = step.get("safety_geometry_substeps", [{}])[-1]
            record["reference_qpos"] = step.get("reference_qpos")
            record["reference_eef_pose"] = step.get("reference_eef_pose")
            observations.append(self._observation(record))
            payload = step.get("terminal_runtime_snapshot")
            if isinstance(payload, bytes) and payload:
                snapshot_paths.append(self._save_runtime_snapshot(
                    payload,
                    f"branch_snapshots/plan_{self.plan_counter:02d}_prefix_{self.plan_cursor + index + 1:04d}.pkl",
                ))
        stop_reason = result.get("stop_reason")
        return PrefixPreview(
            actions=actions,
            observations=observations,
            complete=bool(result.get("complete")),
            main_state_restored_verified=main_restored,
            safe=bool(result.get("safe")),
            failure_reason=result.get("failure_reason") or stop_reason,
            artifacts={
                "plan_index": self.plan_counter,
                "snapshot_id": snapshot_id,
                "stop_reason": stop_reason,
                "contact_monitoring": result.get("contact_monitoring"),
                "branch_snapshot_paths": snapshot_paths,
                "main_state_restored_verified": main_restored,
                "retryable_safety_rejection": bool(
                    result.get("complete")
                    and main_restored
                    and not result.get("safe")
                    and result.get("stop_reason") == "substep_geometry_safety_unsafe"
                    and result.get("contact_monitoring", {})
                    .get("geometry_monitoring", {})
                    .get("status") == "unsafe"
                ),
                "last_path_index": result.get("last_path_index"),
            },
        )

    def task_expert_action(self, observation: Observation, context: TaskContext) -> np.ndarray | None:
        if self.task_controller is None:
            return None
        target = self._task_stage_target(observation.raw, context.phase)
        if target["mode"] == "rrt_connect":
            return None
        robot_position, robot_quaternion = self._live_eef_pose(observation.raw)
        delta = target["position"] - robot_position
        distance = float(np.linalg.norm(delta))
        if distance > target["stride_m"]:
            delta *= target["stride_m"] / distance
        desired_rotation = Rotation.from_quat(target["quaternion"][[1, 2, 3, 0]])
        current_rotation = Rotation.from_quat(robot_quaternion[[1, 2, 3, 0]])
        rotation_delta = (desired_rotation * current_rotation.inv()).as_rotvec()
        position_scale = float(self.config_data["task_execution"].get("position_action_scale_m", 0.05))
        rotation_scale = float(self.config_data["task_execution"].get("rotation_action_scale_rad", 0.5))
        if position_scale <= 0 or rotation_scale <= 0:
            raise ValueError("native_OSC_POSE_action_scales_must_be_positive")
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(delta / position_scale, -1.0, 1.0)
        action[3:6] = np.clip(rotation_delta / rotation_scale, -1.0, 1.0)
        action[6] = target["gripper_command"]
        if not np.isfinite(action).all():
            raise ValueError("task_expert_action_is_not_finite_7D")
        self._current_action_source = "task_expert_osc_pose"
        self.current_forecast = action[None, :].copy()
        return action

    def handoff_to_task_expert(self, observation: Observation, context: TaskContext) -> None:
        if self.task_controller is None:
            raise RuntimeError("task_expert_handoff_requested_without_a_task_phase_program")
        if not self.task_expert_active:
            if self.task_controller.phase == "pregrasp":
                self.task_controller.handoff_after_pregrasp()
            self.task_expert_active = True
            self.student_chunk = np.empty((0, 7), dtype=np.float32)
            self.student_chunk_cursor = 0
            self.current_forecast = np.empty((0, 7), dtype=np.float32)
            self.artifacts.event({
                "event": "task_expert_handoff",
                "timestep": observation.timestep,
                "phase": self.task_controller.phase,
                "source": self.config_data["task_execution"]["provenance"],
                "policy_action_space": "native_7D_OSC_POSE_plus_gripper",
            })
        else:
            self.artifacts.event({
                "event": "task_expert_resumed_after_automatic_replan",
                "timestep": observation.timestep,
                "phase": self.task_controller.phase,
            })

    def handoff_to_student(self, observation: Observation, context: TaskContext) -> None:
        self.student_handoff_completed = True
        self.live_student_handoff = True
        self.student_chunk = np.empty((0, 7), dtype=np.float32)
        self.student_chunk_cursor = 0
        self.current_forecast = np.empty((0, 7), dtype=np.float32)
        self.artifacts.event({
            "event": "student_handoff",
            "timestep": observation.timestep,
            "source": "fresh_Pi0.5_inference_from_current_post_recovery_observation",
            "policy_action_space": "native_7D_OSC_POSE_plus_gripper",
        })

    def is_recovery_goal_reached(self, observation: Observation, goal: GoalSpec) -> bool:
        position_error = float(
            np.linalg.norm(observation.eef_position_world_m - np.asarray(goal.position_world_m))
        )
        a = observation.eef_quaternion_wxyz / np.linalg.norm(observation.eef_quaternion_wxyz)
        b = np.asarray(goal.quaternion_wxyz, dtype=np.float64)
        b /= np.linalg.norm(b)
        rotation_error = 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))
        return bool(
            position_error <= float(self.config_data.get("goal_position_tolerance_m", 0.012))
            and rotation_error <= float(self.config_data.get("goal_orientation_tolerance_rad", 0.08))
        )

    def save_snapshot(self, label: str) -> str | None:
        snapshot = self.bridge.request({"op": "export_runtime_snapshot"})
        if snapshot.get("status") != "complete" or not isinstance(snapshot.get("payload"), bytes):
            return None
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        return self._save_runtime_snapshot(
            snapshot["payload"], f"snapshots/{safe_label}_timestep_{snapshot['snapshot_state']['mujoco_integration_state'][0]}.pkl"
        )

    def record_event(self, event: dict[str, Any]) -> None:
        return None

    def restore_start_state(self) -> dict[str, Any]:
        # Reproduce the native reset/set_init_state sequence used to create the
        # archived runtime snapshot. This is an environment operation on the
        # dedicated Bridge, never a direct qpos edit.
        reset = self.bridge.request({"op": "reset"})
        reset_metadata = reset.get("scene_metadata")
        if not isinstance(reset_metadata, dict):
            raise RuntimeError("native_episode_reset_did_not_return_scene_metadata")
        if reset_metadata.get("model_geometry_sha256") != self.config_data.get(
            "expected_model_geometry_sha256"
        ):
            raise RuntimeError("native_episode_reset_model_geometry_fingerprint_mismatch")
        if int(reset.get("timestep", -1)) != 0:
            raise RuntimeError("native_episode_reset_did_not_return_timestep_zero")
        if self.mode in {"t10-snapshot", "runtime-snapshot"}:
            payload = self.runtime_snapshot_path.read_bytes()
            response = self.bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
            if response.get("status") != "restored":
                raise RuntimeError(f"runtime_snapshot_restore_failed:{response.get('error')}")
            label = "start_t10_archived_snapshot" if self.mode == "t10-snapshot" else "start_runtime_snapshot"
            self._save_runtime_snapshot(payload, f"snapshots/{label}.pkl")
            return response
        response = self.bridge.request({"op": "observe", "include_snapshot_state": True})
        if int(response.get("timestep", -1)) != 0:
            raise RuntimeError("full_episode_start_did_not_begin_at_timestep_zero")
        exported = self.bridge.request({"op": "export_runtime_snapshot"})
        if exported.get("status") != "complete":
            raise RuntimeError("initial_full_episode_snapshot_export_failed")
        self._save_runtime_snapshot(exported["payload"], "snapshots/start_episode_timestep_000000.pkl")
        self.latest_data = response
        return {"status": "initial_episode_state_verified", "snapshot_state": response["snapshot_state"]}

    def initialize_trajectory(self):
        task_execution = self.config_data.get("task_execution", {})
        task_provenance = task_execution.get("provenance") or self.config_data.get(
            "provenance", {}
        ).get("target_pose", "configured_verified_task_goal_sources")
        self.trajectory = TrajectoryWriter(
            self.output_dir / "observation_action_trace.npz",
            {
                "schema": "gs_opd_automatic_safety_recovery_observation_action.v1",
                "benchmark": self.config_data["benchmark"],
                "level": int(self.config_data["level"]),
                "task_id": int(self.config_data["task_id"]),
                "mode": self.mode,
                "source_student_trace": str(self.archived_trace_path),
                "source_student_trace_sha256": self.archive_sha256,
                "goal_provenance": task_provenance,
                "goal_is_VLM_Teacher_output": bool(
                    task_execution.get("goal_source_is_vlm_teacher", False)
                ),
                "model_checkpoint": str(self.student.checkpoint) if self.student else None,
                "model_checkpoint_sha256": sha256_file(self.student.checkpoint / "model.safetensors") if self.student else None,
            },
        )
        self._init_historical_goal_qpos()
        if self.latest_data is None:
            self.latest_data = self.bridge.request({
                "op": "observe",
                "include_snapshot_state": True,
                "safety_geometry_policy": self.safety_policy,
            })
        self._save_video_frame(self.latest_data)

    def finalize(self, result: dict[str, Any]):
        if hasattr(self, "trajectory"):
            self.trajectory.save(steps=self.transition_count, chunks=self.transition_count)
        for writer in self._video_writers.values():
            writer.release()
        self._video_writers.clear()
        summary = {
            "mode": self.mode,
            "task_semantics": self.context_static,
            "safety_policy": self.safety_policy,
            "transitions_recorded": self.transition_count,
            "task_success": bool(result.get("complete_task_success")),
            "complete_safety_success": bool(result.get("complete_safety_success")),
            "local_recovery_success": bool(result.get("local_recovery_success")),
            "standard_horizon_success": bool(result.get("standard_horizon_success")),
            "extended_horizon": bool(result.get("extended_horizon")),
            "result": result,
            "outputs": {
                "trace": str(self.output_dir / "observation_action_trace.npz"),
                "agentview_video": str(self.output_dir / "agentview.mp4"),
                "wrist_video": str(self.output_dir / "wrist.mp4"),
                "main_snapshots": str(self.output_dir / "snapshots"),
                "branch_snapshots": str(self.output_dir / "branch_snapshots"),
            },
        }
        (self.output_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_runtime_config(path: Path) -> tuple[dict[str, Any], RecoveryConfig]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    recovery = RecoveryConfig(**value["recovery"])
    recovery.validate()
    if recovery.hard_minimum_clearance_m < 0.010:
        raise ValueError("runtime_configuration_cannot_reduce_the_10_mm_hard_clearance")
    return value, recovery


__all__ = [
    "AutomaticRecoveryBridgeBackend",
    "BridgeProcess",
    "Pi05StudentPolicy",
    "REPO_ROOT",
    "DEFAULT_ENV_PYTHON",
    "DEFAULT_PI05_CHECKPOINT",
    "load_runtime_config",
    "sha256_file",
]
