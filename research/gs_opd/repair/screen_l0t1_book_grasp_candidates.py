"""Read-only IK and full-geometry screen for the archived L0T1 book candidates.

Run with the LIBERO-Safety virtualenv. A fresh Safety Bridge process restores
the original Student t=0 Runtime Snapshot, then calls only the scratch-only
``screen_grasp_candidates`` operation; it sends no controls and steps no physics.
"""

from __future__ import annotations

import hashlib
import importlib.util
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = Path("/workspace/zy_workspace/LIBERO-Safety")
RAW_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "obstacle_avoidance_L0T1_raw_student_seed0_init0_20261025_234627"
)
CANDIDATE_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "obstacle_avoidance_L0T1_t0_book_cold_grasp_20261026_attempt1"
)
DEFAULT_OUTPUT_DIR = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "obstacle_avoidance_L0T1_t0_book_grasp_candidate_screen_20261026_attempt4"
)
BRIDGE_PYTHON = LIBERO_ROOT / ".venv/bin/python"
BRIDGE_SCRIPT = REPO_ROOT / "research/gs_opd/smoke/safety_bridge.py"
BRIDGE_PORT = 38571
BRIDGE_AUTHKEY = b"gs-opd-local"
EXPECTED_MODEL_SHA256 = "9e2f374b19a09a123e715dcefc0cebaa11d647b4029d4546bc6b52a2944d451e"
EXPECTED_BDDL_SHA256 = "b14943e935d04f7fe07dabce8c097dac3721d2c689965c0abb407df96dfcc844"
EXPECTED_INSTRUCTION = "pick up the book and place it in the back compartment of the caddy"
BOOK_ID = "black_book_1"
BEER_ID = "bottle_of_beer__7_1"
PLACE_BOX_ID = "white_place_box_1"


class _PythonObjectState:
    """Pickle compatibility wrapper used by the archived Runtime Snapshot."""

    def __init__(self, module: str, qualname: str, attributes: dict[str, Any]):
        self.module = module
        self.qualname = qualname
        self.attributes = attributes


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_archive_policy_builder():
    source = CANDIDATE_ARCHIVE / "source/l0t1_book_t0_cold_grasp_search.py"
    spec = importlib.util.spec_from_file_location("l0t1_book_candidate_source", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("archived_candidate_source_cannot_be_loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Bridge:
    def __init__(self, log_path: Path, port: int):
        self.log_path = log_path
        self.port = int(port)
        self.process = None
        self.connection = None
        self.log = None

    def start(self) -> None:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(f"Safety_Bridge_port_{self.port}_is_already_in_use")
        environment = os.environ.copy()
        environment["MUJOCO_GL"] = "egl"
        environment["PYOPENGL_PLATFORM"] = "egl"
        environment["LIBERO_CONFIG_PATH"] = "/workspace/zy_workspace/.libero_safety_config"
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(LIBERO_ROOT), str(LIBERO_ROOT / "third_party/robosuite-1.4"), str(REPO_ROOT)]
        )
        self.log = self.log_path.open("ab", buffering=0)
        command = [
            str(BRIDGE_PYTHON), str(BRIDGE_SCRIPT),
            "--benchmark", "obstacle_avoidance", "--level", "0", "--task-id", "1",
            "--init-state", "0", "--seed", "0", "--camera-resolution", "32",
            "--port", str(self.port),
        ]
        self.process = subprocess.Popen(
            command, cwd=str(REPO_ROOT), env=environment,
            stdout=self.log, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Safety_Bridge_exited_{self.process.returncode}")
            try:
                self.connection = Client(("127.0.0.1", self.port), authkey=BRIDGE_AUTHKEY)
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)
        raise TimeoutError("Safety_Bridge_startup_timeout")

    def request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.connection is None:
            raise RuntimeError("Safety_Bridge_is_not_connected")
        self.connection.send(request)
        reply = self.connection.recv()
        if not isinstance(reply, dict):
            raise RuntimeError(f"malformed_Bridge_reply_for_{request.get('op')}")
        return reply

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
        if self.log is not None:
            self.log.close()
            self.log = None


def run(output_dir: Path = DEFAULT_OUTPUT_DIR, bridge_port: int = BRIDGE_PORT) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing_to_overwrite_candidate_screen_archive:{output_dir}")
    required = [
        RAW_ARCHIVE / "snapshots/initial_runtime_snapshot.pkl",
        RAW_ARCHIVE / "snapshots/runtime_snapshot_manifest.json",
        CANDIDATE_ARCHIVE / "source_initial_runtime_snapshot.pkl",
        CANDIDATE_ARCHIVE / "scene_start.json",
        CANDIDATE_ARCHIVE / "geometry_candidates.json",
        CANDIDATE_ARCHIVE / "source/l0t1_book_t0_cold_grasp_search.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not BRIDGE_PYTHON.is_file() or not BRIDGE_SCRIPT.is_file():
        raise FileNotFoundError(f"required_local_inputs_missing:{missing}")
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_snapshot = RAW_ARCHIVE / "snapshots/initial_runtime_snapshot.pkl"
    copied_snapshot = CANDIDATE_ARCHIVE / "source_initial_runtime_snapshot.pkl"
    manifest = json.loads((RAW_ARCHIVE / "snapshots/runtime_snapshot_manifest.json").read_text())
    source_snapshot_record = manifest["snapshots"]["initial"]
    raw_sha, copied_sha = _sha256_file(raw_snapshot), _sha256_file(copied_snapshot)
    if raw_sha != copied_sha or raw_sha != source_snapshot_record.get("payload_sha256"):
        raise ValueError("original_student_t0_snapshot_hash_mismatch")
    if source_snapshot_record.get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("archived_t0_model_signature_mismatch")
    payload = raw_snapshot.read_bytes()
    import pickle
    envelope = pickle.loads(payload)
    if (
        envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1"
        or envelope.get("model_geometry_sha256") != EXPECTED_MODEL_SHA256
        or envelope.get("scene_metadata", {}).get("instruction") != EXPECTED_INSTRUCTION
    ):
        raise ValueError("archived_Runtime_Snapshot_identity_mismatch")
    expected_state = np.asarray(envelope["snapshot"]["mujoco_state"], dtype=np.float64)
    state_sha = _sha256_bytes(np.ascontiguousarray(expected_state).tobytes())
    if state_sha != source_snapshot_record.get("mujoco_integration_state_sha256"):
        raise ValueError("archived_t0_integration_state_hash_mismatch")

    candidates = json.loads((CANDIDATE_ARCHIVE / "geometry_candidates.json").read_text())
    scene_start = json.loads((CANDIDATE_ARCHIVE / "scene_start.json").read_text())
    if len(candidates) != 24 or len({item.get("candidate_id") for item in candidates}) != 24:
        raise ValueError("expected_exactly_24_unique_archived_book_candidates")
    archived_candidate_hash = _sha256_file(CANDIDATE_ARCHIVE / "geometry_candidates.json")
    archived_model_hash = scene_start.get("model_fingerprint_details", {}).get("model_geometry_sha256")
    if archived_model_hash != EXPECTED_MODEL_SHA256:
        raise ValueError("candidate_archive_model_fingerprint_mismatch")

    policy_source = _load_archive_policy_builder()
    bridge = _Bridge(output_dir / "safety_bridge.log", bridge_port)
    report: dict[str, Any] = {
        "schema_version": "gs_opd_l0t1_book_grasp_candidate_screen.v1",
        "scope": {
            "task": "obstacle_avoidance/L0T1", "seed": 0, "init_state": 0,
            "instruction": EXPECTED_INSTRUCTION,
            "native_goal": ["In", BOOK_ID, "desk_caddy_1_back_contain_region"],
            "candidate_count": len(candidates),
        },
        "inputs": {
            "source_student_snapshot": str(raw_snapshot),
            "source_student_snapshot_sha256": raw_sha,
            "integration_state_sha256": state_sha,
            "model_geometry_sha256": EXPECTED_MODEL_SHA256,
            "candidate_archive": str(CANDIDATE_ARCHIVE),
            "candidate_file_sha256": archived_candidate_hash,
            "screen_implementation_sha256": _sha256_file(REPO_ROOT / "research/gs_opd/repair/mujoco_jointspace_planning.py"),
            "screen_runner_sha256": _sha256_file(Path(__file__).resolve()),
            "official_safety_demo_used": False,
        },
        "method": {
            "isolated_safety_bridge_process": True,
            "scratch_only_screen": True,
            "live_environment_actions": 0,
            "mujoco_physics_substeps": 0,
            "approach_gripper_state": "restored_t0_open_qpos",
            "target_grasp_ik_gripper_state": "geometry_derived_closed_qpos",
            "close_path": "9 linearly interpolated in-range scratch qpos samples including open and closed endpoints",
            "both_fingertip_pad_geometries_checked_at_each_close_sample": True,
            "full_robot_geometry_and_contact_policy_checked_at_each_close_sample": True,
            "rrt_or_physical_grasp_executed": False,
            "training_or_lerobot_conversion": False,
        },
    }
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        reset_meta = reset.get("scene_metadata", {})
        if reset_meta.get("model_geometry_sha256") != EXPECTED_MODEL_SHA256:
            raise RuntimeError("fresh_Bridge_model_geometry_signature_mismatch")
        if reset_meta.get("bddl_sha256") != EXPECTED_BDDL_SHA256:
            raise RuntimeError("fresh_Bridge_native_BDDL_signature_mismatch")
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
        restored_state = np.asarray(
            restored.get("snapshot_state", {}).get("mujoco_integration_state", []),
            dtype=np.float64,
        )
        if (
            restored.get("status") != "restored"
            or restored.get("model_geometry_sha256") != EXPECTED_MODEL_SHA256
            or not np.array_equal(restored_state, expected_state)
        ):
            raise RuntimeError("exact_original_student_t0_Runtime_Snapshot_restore_failed")
        observe = bridge.request({"op": "observe", "include_snapshot_state": True})
        if observe.get("instruction") != EXPECTED_INSTRUCTION:
            raise RuntimeError("restored_instruction_mismatch")
        native = bridge.request({"op": "native_task_status"})
        if native.get("task_success") is not False or native.get("reward") != 0.0 or native.get("done") is not False:
            raise RuntimeError("restored_t0_native_goal_status_mismatch")
        geometry = bridge.request({"op": "geometry_inventory"})
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        gripper = bridge.request({"op": "gripper_geometry_inventory"})
        limits = bridge.request({"op": "robot_joint_limit_inventory"})
        if any(item.get("status") != "complete" for item in (geometry, semantic, gripper, limits)):
            raise RuntimeError("active_geometry_semantics_gripper_or_joint_inventory_incomplete")
        policy, binding_report = policy_source._build_safety_policy(
            semantic["bindings"], geometry, gripper
        )
        initial_gate = bridge.request({
            "op": "safety_geometry_check",
            "safety_geometry_policy": policy,
        })
        if initial_gate.get("status") != "complete":
            raise RuntimeError(f"t0_full_geometry_gate_not_clear:{initial_gate.get('status')}")
        book_state = scene_start["layout"][BOOK_ID]
        book_pose = policy_source._pose_matrix(book_state["root_pos_m"], book_state["root_quat_wxyz"])
        book_binding = semantic["bindings"][BOOK_ID]
        box_half_thickness = float(scene_start["book_collision_geometry"]["half_size_geom_local_m"][0])
        closed_qpos = [box_half_thickness, -box_half_thickness]
        state_now = np.asarray(observe["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
        if not np.array_equal(state_now, expected_state):
            raise RuntimeError("observe_after_restore_did_not_match_t0_integration_state")
        state_now_sha = _sha256_bytes(np.ascontiguousarray(state_now).tobytes())
        screened = bridge.request({
            "op": "screen_grasp_candidates",
            "expected_instruction": EXPECTED_INSTRUCTION,
            "model_geometry_sha256": EXPECTED_MODEL_SHA256,
            "start_state_sha256": state_now_sha,
            "safety_geometry_policy": policy,
            "target_object_id": BOOK_ID,
            "target_root_body_id": int(book_binding["root_body_id"]),
            "clearance_m": 0.010,
            "closed_gripper_qpos": closed_qpos,
            "predicted_stable_object_pose_world": book_pose.tolist(),
            "candidates": candidates,
        })
        after = bridge.request({"op": "observe", "include_snapshot_state": True})
        after_state = np.asarray(after["snapshot_state"]["mujoco_integration_state"], dtype=np.float64)
        live_unchanged = np.array_equal(state_now, after_state)
        if screened.get("status") != "complete" or screened.get("live_state_unchanged") is not True or not live_unchanged:
            raise RuntimeError("candidate_screen_incomplete_or_live_state_changed")

        binding_names = [item["object_id"] for item in policy["forbidden_objects"]]
        by_id = {item["candidate_id"]: item for item in screened.get("candidates", [])}
        if len(by_id) != 24:
            raise RuntimeError(f"screen_returned_{len(by_id)}_candidate_records_instead_of_24")
        ranked_ids = screened.get("ranked_candidate_ids", [])
        ranking = {candidate_id: rank + 1 for rank, candidate_id in enumerate(ranked_ids)}
        summaries = []
        for candidate in candidates:
            row = by_id[str(candidate["candidate_id"])]
            clearance = row.get("minimum_forbidden_clearance_by_object_m", {})
            pregrasp_violations = row.get("pregrasp_check", {}).get("violations", [])
            approach_violations = [
                violation
                for check in row.get("approach_path_checks", [])
                for violation in check.get("violations", [])
            ]
            close_violations = [
                violation
                for sample in row.get("gripper_close_path_checks", [])
                for violation in sample.get("geometry_check", {}).get("violations", [])
            ]
            close_robot_scene_contacts = [
                {"alpha": sample.get("alpha"), **violation}
                for sample in row.get("gripper_close_path_checks", [])
                for violation in sample.get("geometry_check", {}).get("violations", [])
                if "robot_scene_contact_geoms" in violation
            ]
            approach_robot_scene_contacts = [
                {"path_sample_index": index, **violation}
                for index, check in enumerate(row.get("approach_path_checks", []))
                for violation in check.get("violations", [])
                if "robot_scene_contact_geoms" in violation
            ]
            summaries.append({
                "candidate_id": row["candidate_id"],
                "status": row.get("status"),
                "failure_class": row.get("failure_class"),
                "failure_classes": row.get("failure_classes", []),
                "reason": row.get("reason"),
                "failure_reasons": row.get("failure_reasons", [row.get("reason")]),
                "rank": ranking.get(row["candidate_id"]),
                "joint_reachability": row.get("joint_reachability", {}),
                "finger_distance_status": row.get("finger_distance_status"),
                "left_fingertip_to_book_m": row.get("finger_target_distance_m", {}).get("left"),
                "right_fingertip_to_book_m": row.get("finger_target_distance_m", {}).get("right"),
                "bilateral_fingertip_contact": row.get("bilateral_fingertip_contact"),
                "gripper_opening_range": row.get("gripper_opening_range"),
                "beer_min_clearance_m": clearance.get(BEER_ID),
                "place_box_min_clearance_m": clearance.get(PLACE_BOX_ID),
                "book_to_beer_clearance_m": clearance.get(f"book->{BEER_ID}"),
                "book_to_place_box_clearance_m": clearance.get(f"book->{PLACE_BOX_ID}"),
                "t0_robot_beer_clearance_m": row.get("t0_robot_clearance_by_forbidden_object_m", {}).get(BEER_ID),
                "t0_robot_place_box_clearance_m": row.get("t0_robot_clearance_by_forbidden_object_m", {}).get(PLACE_BOX_ID),
                "minimum_forbidden_clearance_m": row.get("minimum_forbidden_clearance_m"),
                "blocking_contact_evidence": {
                    "pregrasp": pregrasp_violations,
                    "approach_first_violation": approach_violations[:1],
                    "closed_gripper_first_violation": close_violations[:1],
                    "approach_robot_scene_contacts": approach_robot_scene_contacts[:4],
                    "closed_gripper_robot_scene_contacts": close_robot_scene_contacts[:8],
                },
            })
        report.update({
            "status": "complete",
            "scene_verification": {
                "instruction": observe.get("instruction"),
                "native_task_status": native,
                "model_geometry_sha256": EXPECTED_MODEL_SHA256,
                "bddl_sha256": EXPECTED_BDDL_SHA256,
                "restored_integration_state_sha256": state_now_sha,
                "full_geometry_gate": initial_gate,
                "static_installation_pair_evidence": initial_gate.get("allowed_static_installation_contact_pairs", []),
                "safety_binding_report": binding_report,
                "book_pose_source": "exact_restored_t0_layout_no_object_motion_hypothesis",
                "closed_gripper_qpos_derived_from_book_collision_half_thickness_m": closed_qpos,
            },
            "screen_result": screened,
            "candidate_summaries": summaries,
            "bilateral_fingertip_geometry_pass_count": sum(
                bool(item.get("bilateral_fingertip_contact", {}).get("pass"))
                for item in screened.get("candidates", [])
            ),
            "ranking": ranked_ids,
            "live_state_unchanged_after_screen": live_unchanged,
            "failure_class_counts": screened.get("failure_class_counts", {}),
            "failure_class_membership_counts": screened.get("failure_class_membership_counts", {}),
        })
    finally:
        bridge.close()
    _write_json(output_dir / "candidate_screening.json", report)
    (output_dir / "candidate_screening_summary.md").write_text(
        "# L0T1 book candidate scratch screen\n\n"
        f"- Status: `{report.get('status')}`\n"
        f"- Candidate count: {report['scope']['candidate_count']}\n"
        f"- Feasible count: {len(report.get('ranking', []))}\n"
        f"- Bilateral fingertip geometry pass count: {report.get('bilateral_fingertip_geometry_pass_count', 0)}\n"
        f"- Ranked candidates: {', '.join(report.get('ranking', [])) or '(none)'}\n"
        f"- Failure classes: `{json.dumps(report.get('failure_class_counts', {}), ensure_ascii=False)}`\n"
        f"- Failure class membership: `{json.dumps(report.get('failure_class_membership_counts', {}), ensure_ascii=False)}`\n"
        "- Actions / physics substeps: 0 / 0\n"
        "- Live state unchanged: "
        f"{report.get('live_state_unchanged_after_screen', False)}\n\n"
        "Complete per-candidate IK, fingertip distance, jaw-gap, and Beer/Place Box clearance evidence is in `candidate_screening.json`.\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bridge-port", type=int, default=BRIDGE_PORT)
    args = parser.parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    result = run(args.output_dir, args.bridge_port)
    print(json.dumps({
        "status": result.get("status"),
        "feasible_count": len(result.get("ranking", [])),
        "ranking": result.get("ranking", []),
        "failure_class_counts": result.get("failure_class_counts", {}),
        "failure_class_membership_counts": result.get("failure_class_membership_counts", {}),
        "archive": str(args.output_dir),
    }, ensure_ascii=False))
