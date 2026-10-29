# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Common, provenance-aware entry point for existing GS-OPD physical engines."""

from __future__ import annotations

import argparse
import errno
import gzip
import hashlib
import json
import multiprocessing.connection
import os
import pickle
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from research.gs_opd.repair.l0t0_success_reference import (
    SuccessReference,
    load_success_reference,
    sha256_file,
)


RLINF_ROOT = Path("/workspace/zy_workspace/RLinf")
LIBERO_ROOT = Path("/workspace/zy_workspace/LIBERO-Safety")
RLINF_PYTHON = RLINF_ROOT / ".venv/bin/python"
LIBERO_PYTHON = LIBERO_ROOT / ".venv/bin/python"
BRIDGE_SCRIPT = REPO_ROOT / "research/gs_opd/smoke/safety_bridge.py"
BRIDGE_AUTHKEY = b"gs-opd-local"
DEFAULT_CHECKPOINT = Path("/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft")
DEFAULT_EXPERIENCE_LIBRARY = Path(
    "/opt/data/private/gs_opd/experience/"
    "gs_opd_task_skill_experience_library_20261026_v2.json"
)
STRICT_FORBIDDEN_CLEARANCE_M = 0.010
STRICT_FINGER_TARGET_PENETRATION_M = 0.002


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _checkpoint_file(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint_weights_missing:{path}")
    return path


def _select_backend(args: argparse.Namespace, reference: SuccessReference | None) -> str:
    if args.backend != "auto":
        return args.backend
    if getattr(args, "cold_start", False):
        if reference is not None or args.snapshot_root is None:
            raise ValueError("cold_start_requires_t0_snapshot_without_success_reference")
        return "experience_task_success"
    if reference is not None and reference.route_prior is not None:
        return "student_bspline_osc"
    if args.experience_library is not None and args.keypose_experience:
        return "experience_task_success"
    raise ValueError("auto_backend_requires_a_valid_route_reference_or_keypose_experience")


def _experience_match(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.experience_library is None:
        return None
    if not args.experience_library.is_file():
        raise FileNotFoundError(f"experience_library_missing:{args.experience_library}")
    if not all((args.experience_object_id, args.target_relation, args.target_region)):
        raise ValueError("experience_matching_requires_object_relation_and_target_region")
    from research.gs_opd.repair.experience_registry import query_experience

    result = query_experience(
        args.experience_object_id,
        args.target_relation,
        args.target_region,
        f"{args.benchmark}/L{args.level}T{args.task_id}",
    )
    return {
        **result,
        "library_sha256": sha256_file(args.experience_library),
        "matched_grasp_prior_qualification": "local_skill_prior_only",
        "source_safety_inherited_by_matched_skill": False,
        "new_scene_physics_verified": False,
    }


def _environment_for_subprocess() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(REPO_ROOT), str(RLINF_ROOT), str(LIBERO_ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _run_bounded_engine(command: list[str], *, log: Any, wall_time_budget_s: float) -> subprocess.CompletedProcess:
    """Stop the engine and its bridge descendants if its wall budget expires."""
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, env=_environment_for_subprocess(),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=wall_time_budget_s)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        # The engine may exit before an uncooperative descendant does.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise TimeoutError(f"engine_wall_time_budget_exhausted:{wall_time_budget_s}s") from exc
    return subprocess.CompletedProcess(command, return_code)


def _bridge_request(connection: Any, payload: dict[str, Any]) -> dict[str, Any]:
    connection.send(payload)
    result = connection.recv()
    if not isinstance(result, dict):
        raise RuntimeError(f"safety_bridge_malformed_response:{payload.get('op')}")
    if result.get("error"):
        raise RuntimeError(f"safety_bridge_{payload.get('op')}_failed:{result['error']}")
    return result


def _launch_bridge_and_preflight(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> tuple[subprocess.Popen[Any], Any, dict[str, Any]]:
    """Start exactly one Bridge and verify task, goal, and semantic geometry before actions."""
    address = ("127.0.0.1", args.bridge_port)
    with socket.socket() as sock:
        try:
            sock.bind(address)
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise RuntimeError(f"safety_bridge_port_already_in_use:{args.bridge_port}") from exc
            raise
    if not LIBERO_PYTHON.is_file() or not BRIDGE_SCRIPT.is_file():
        raise FileNotFoundError("LIBERO_Safety_Bridge_interpreter_or_script_missing")
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["LIBERO_CONFIG_PATH"] = "/workspace/zy_workspace/.libero_safety_config"
    env["PYTHONPATH"] = os.pathsep.join([
        str(LIBERO_ROOT),
        str(LIBERO_ROOT / "third_party/robosuite-1.4"),
        str(REPO_ROOT),
    ])
    command = [
        str(LIBERO_PYTHON), str(BRIDGE_SCRIPT),
        "--benchmark", args.benchmark,
        "--level", str(args.level),
        "--task-id", str(args.task_id),
        "--init-state", str(args.init_state),
        "--seed", str(args.seed),
        "--camera-resolution", "256",
        "--port", str(args.bridge_port),
        "--demonstration-horizon", str(args.max_steps),
    ]
    bridge_log_path = output_dir / "safety_bridge.log"
    if bridge_log_path.exists():
        raise FileExistsError(f"safety_bridge_log_already_exists:{bridge_log_path}")
    bridge_log = bridge_log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=bridge_log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"safety_bridge_exited_before_ready:{process.returncode}")
            if b"SAFETY BRIDGE READY" in bridge_log_path.read_bytes():
                break
            time.sleep(0.5)
        else:
            raise TimeoutError("safety_bridge_not_ready_within_120_seconds")

        # Use the authenticated protocol for the first connection. A raw TCP
        # availability probe is treated as an unauthenticated client by the
        # Bridge's single accept loop and can terminate that listener.
        connection = multiprocessing.connection.Client(address, authkey=BRIDGE_AUTHKEY)
        try:
            observed = _bridge_request(connection, {"op": "observe", "include_snapshot_state": True})
            scene = observed.get("scene_metadata", {})
            if (
                observed.get("instruction") != args.instruction
                or scene.get("benchmark") != args.benchmark
                or scene.get("level") != args.level
                or scene.get("task_id") != args.task_id
                or scene.get("init_state_index") != args.init_state
                or scene.get("environment_seed") != args.seed
            ):
                raise ValueError("active_bridge_scene_does_not_match_requested_task_identity")
            native = _bridge_request(connection, {"op": "native_task_status"})
            values = {
                tuple(row.get("predicate", [])): row.get("value")
                for row in native.get("native_goal_values", [])
                if isinstance(row, dict)
            }
            predicate = tuple(args.target_predicate or [])
            if len(predicate) != 3 or predicate not in values:
                raise ValueError("requested_native_BDDL_goal_not_present_in_active_task")
            if native.get("task_success") is True:
                raise ValueError("fresh_initial_state_already_has_native_task_success")
            inventory = _bridge_request(connection, {"op": "geometry_inventory"})
            semantics = _bridge_request(connection, {"op": "task_semantic_geometry_bindings"})
            task_context = _bridge_request(connection, {
                "op": "task_success_first_context",
                "expected_instruction": args.instruction,
                "target_predicate": list(predicate),
            })
            semantic_json = json.loads(args.semantic_results_json.read_text(encoding="utf-8"))
            target_semantic = semantic_json.get(args.semantic_object_id, {})
            association = target_semantic.get("physical_association", {})
            body_name = association.get("body_name")
            geom_names = set(association.get("geom_names", []))
            live_geoms = [
                row for row in inventory.get("geometries", [])
                if row.get("body_name") == body_name
                and row.get("geom_name") in geom_names
            ]
            if (
                inventory.get("status") != "complete"
                or target_semantic.get("status") != "resolved"
                or association.get("status") != "matched_by_iou"
                or not live_geoms
                or task_context.get("status") != "complete"
                or task_context.get("native_goal_predicate") != list(predicate)
                or task_context.get("target_object_id") != args.experience_object_id
            ):
                raise ValueError("target_semantic_binding_does_not_match_live_compiled_geometry")
            snapshot = _bridge_request(connection, {"op": "export_runtime_snapshot"})
            snapshot_payload = snapshot.get("payload")
            if not isinstance(snapshot_payload, bytes):
                raise RuntimeError("initial_runtime_snapshot_export_missing_payload")
            snapshot_path = output_dir / "preflight_initial_runtime_snapshot.pkl"
            snapshot_path.write_bytes(snapshot_payload)
            preflight = {
                "status": "passed_before_physical_action",
                "instruction": observed["instruction"],
                "task": {key: scene.get(key) for key in (
                    "benchmark", "level", "task_id", "init_state_index", "environment_seed",
                )},
                "scene_metadata": scene,
                "native_task_status": native,
                "requested_native_predicate": list(predicate),
                "requested_predicate_initial_value": values[predicate],
                "model_geometry_sha256": task_context.get("model_geometry_sha256"),
                "bddl_sha256": task_context.get("bddl_sha256"),
                "semantic_bindings_status": semantics.get("status"),
                "target_semantic_object_id": args.semantic_object_id,
                "target_body_name": body_name,
                "target_live_geom_names": sorted(row["geom_name"] for row in live_geoms),
                "geometry_inventory_count": len(inventory.get("geometries", [])),
                "initial_runtime_snapshot": {
                    "path": str(snapshot_path),
                    "sha256": sha256_file(snapshot_path),
                    "integration_state_sha256": snapshot.get("integration_state_sha256"),
                },
                "physical_actions_executed": 0,
                "strict_safety_certified": False,
            }
            return process, bridge_log, preflight
        finally:
            connection.close()
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        bridge_log.close()
        raise


def _student_bspline_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    checkpoint_file: Path,
    route_direction: str,
) -> list[str]:
    if route_direction not in {"left", "right", "above"}:
        raise ValueError(f"student_bspline_backend_unsupported_reference_direction:{route_direction}")
    if args.level != 0 or args.task_id != 0 or args.init_state != 0:
        raise ValueError("student_bspline_backend_is_verified_only_for_obstacle_avoidance_L0T0_state0")
    if args.teacher_json is None or args.semantic_results_json is None:
        raise ValueError("student_bspline_backend_requires_teacher_and_semantic_bindings")
    for path in (args.teacher_json, args.semantic_results_json):
        if not path.is_file():
            raise FileNotFoundError(f"scene_input_missing:{path}")

    script = REPO_ROOT / "research/gs_opd/smoke/pi05_rollout.py"
    if not script.is_file():
        raise FileNotFoundError(f"student_rollout_entry_missing:{script}")
    argv = [
        str(args.rlinf_python), "-u", str(script),
        "--checkpoint", str(checkpoint_file.parent),
        "--seed", str(args.seed),
        "--bridge-port", str(args.bridge_port),
        "--max-steps", str(args.max_steps),
        "--record-physical-state",
        "--record-substep-contacts",
        "--runtime-snapshot-dir", str(output_dir / "runtime_snapshots"),
        "--teacher-json", str(args.teacher_json.resolve()),
        "--semantic-results-json", str(args.semantic_results_json.resolve()),
        "--trajectory-bundle-physical-validation",
        "--trajectory-subject-object-id", args.semantic_object_id,
        "--direct-bspline-osc-left",
        "--trajectory-validation-out", str(output_dir / "direct_bspline_osc_report.json"),
        "--result-out", str(output_dir / "student_rollout_result.json"),
        "--trace-out", str(output_dir / "student_rollout.npz"),
        "--video-out", str(output_dir / "student_and_osc.mp4"),
    ]
    if args.route_candidate_id is not None:
        argv.extend(["--bspline-candidate-id", args.route_candidate_id])
    pre_route_hold_actions = int(getattr(args, "pre_route_hold_actions", 0) or 0)
    if pre_route_hold_actions:
        argv.extend(["--bspline-pre-route-hold-actions", str(pre_route_hold_actions)])
    if args.lora_checkpoint is not None:
        argv.extend(["--lora-checkpoint", str(args.lora_checkpoint.resolve())])
    return argv


def _experience_task_success_command(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    checkpoint_file: Path,
    continuation_snapshot: Path | None = None,
    continuation_placement_report: Path | None = None,
    continuation_target_predicate: list[str] | None = None,
) -> list[str]:
    if args.snapshot_root is None:
        raise ValueError("experience_task_success_backend_requires_an_identity_checked_snapshot_root")
    cold_start = bool(getattr(args, "cold_start", False))
    if continuation_snapshot is not None and continuation_placement_report is not None:
        raise ValueError("cold_start_continuation_accepts_only_one_same_run_resume_source")
    if continuation_snapshot is not None:
        continuation_snapshot = Path(continuation_snapshot).resolve()
        run_root = Path(args.output_dir).resolve()
        if not cold_start:
            raise ValueError("cold_start_continuation_requires_the_cold_start_protocol")
        if not continuation_snapshot.is_file() or not continuation_snapshot.is_relative_to(run_root):
            raise ValueError("cold_start_continuation_snapshot_must_be_generated_inside_the_current_V3_run")
    if continuation_placement_report is not None:
        continuation_placement_report = Path(continuation_placement_report).resolve()
        run_root = Path(args.output_dir).resolve()
        if not cold_start:
            raise ValueError("cold_start_placement_continuation_requires_the_cold_start_protocol")
        if not continuation_placement_report.is_file() or not continuation_placement_report.is_relative_to(run_root):
            raise ValueError("cold_start_placement_report_must_be_generated_inside_the_current_V3_run")
    if args.experience_library is None and not cold_start:
        raise ValueError("experience_task_success_backend_requires_an_experience_library")
    if cold_start and (
        args.experience_library is not None
        or getattr(args, "keypose_experience", [])
        or getattr(args, "previous_search_report", [])
        or getattr(args, "resume_from_runtime_snapshot", None) is not None
        or getattr(args, "grasp_candidate_id", None) is not None
        or getattr(args, "strict_grasp_screen_report", None) is not None
    ):
        raise ValueError("cold_start_cannot_use_historical_experience_or_partial_task_inputs")
    if not args.snapshot_root.is_dir():
        raise FileNotFoundError(f"snapshot_root_missing:{args.snapshot_root}")
    script = REPO_ROOT / "research/gs_opd/repair/task_success_first_executor.py"
    argv = [
        str(args.rlinf_python), "-u", str(script),
        "--snapshot-root", str(args.snapshot_root.resolve()),
        "--output-root", str(output_dir / "task_success_engine"),
        "--checkpoint", str(checkpoint_file.parent),
        "--benchmark", args.benchmark,
        "--level", str(args.level),
        "--task-id", str(args.task_id),
        "--init-state", str(args.init_state),
        "--seed", str(args.seed),
        "--demonstration-horizon", str(args.demonstration_horizon),
        "--maximum-physical-branches", str(args.maximum_physical_branches),
        "--maximum-rrt-preflights", str(args.maximum_rrt_preflights),
        "--rrt-time-budget-s", str(args.rrt_time_budget_s),
        "--candidate-start-index", str(getattr(args, "candidate_start_index", 0)),
        "--geometry-candidate-pool-offset", str(getattr(args, "geometry_candidate_pool_offset", 0)),
        "--transport-strategy", args.transport_strategy,
    ]
    native_region_route_variant = getattr(args, "native_region_route_variant", None)
    if native_region_route_variant is not None:
        argv.extend(["--native-region-route-variant", native_region_route_variant])
    if cold_start:
        port = int(getattr(args, "bridge_port", 38569))
        if not 1 <= port <= 65532:
            raise ValueError("cold_start_bridge_port_requires_three_following_ports")
        argv.extend(["--cold-start", "--port", str(port), "--expert-port", str(port + 1)])
        if continuation_snapshot is not None:
            argv.extend([
                "--cold-start-continuation", "--resume-from-runtime-snapshot", str(continuation_snapshot),
                "--cold-start-continuation-root", str(run_root),
            ])
        elif continuation_placement_report is not None:
            argv.extend([
                "--cold-start-continuation", "--resume-from-placement-report",
                str(continuation_placement_report), "--cold-start-continuation-root", str(run_root),
            ])
    else:
        argv.extend(["--experience-library", str(args.experience_library.resolve())])
    resume_snapshot = getattr(args, "resume_from_runtime_snapshot", None)
    if resume_snapshot is not None:
        if not resume_snapshot.is_file():
            raise FileNotFoundError(
                f"resume_runtime_snapshot_missing:{resume_snapshot}"
            )
        argv.extend([
            "--resume-from-runtime-snapshot",
            str(resume_snapshot.resolve()),
        ])
    grasp_candidate_id = getattr(args, "grasp_candidate_id", None)
    if grasp_candidate_id is not None:
        argv.extend(["--candidate-id", grasp_candidate_id])
    selected_target_predicate = continuation_target_predicate or args.target_predicate
    if selected_target_predicate:
        argv.extend(["--target-predicate", *selected_target_predicate])
    for path in getattr(args, "keypose_experience", []) or []:
        argv.extend(["--keypose-experience", str(path.resolve())])
    for path in getattr(args, "previous_search_report", []) or []:
        if not path.is_file():
            raise FileNotFoundError(f"previous_search_report_missing:{path}")
        argv.extend(["--previous-search-report", str(path.resolve())])
    return argv


def _strict_grasp_screen_gate(args: argparse.Namespace) -> dict[str, Any] | None:
    """Validate a no-action strict grasp screen and pin execution to its candidate."""
    screen_path = getattr(args, "strict_grasp_screen_report", None)
    if screen_path is None:
        return None
    screen_path = Path(screen_path).resolve()
    if not screen_path.is_file():
        raise FileNotFoundError(f"strict_grasp_screen_report_missing:{screen_path}")
    report = json.loads(screen_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("strict_grasp_screen_report_root_must_be_an_object")
    if report.get("schema_version") != "gs_opd_strict_pad_only_grasp_screen.v1":
        raise ValueError("strict_grasp_screen_report_schema_mismatch")
    if (
        report.get("status") != "complete"
        or report.get("strict_safety_candidate_only") is not True
        or report.get("physical_actions") != 0
        or report.get("live_integration_state_unchanged") is not True
        or report.get("safety_certified") is not False
        or report.get("route_preflight_performed") is not False
    ):
        raise ValueError("strict_grasp_screen_report_is_not_a_complete_no_action_screen")

    screening = report.get("screening")
    if (
        not isinstance(screening, dict)
        or screening.get("status") != "complete"
        or screening.get("screening_type") != "offline_scratch_only_not_physical_execution"
    ):
        raise ValueError("strict_grasp_screen_report_is_not_a_complete_scratch_screen")

    scene = report.get("task")
    if not isinstance(scene, dict):
        raise ValueError("strict_grasp_screen_task_identity_malformed")
    expected_identity = {
        "benchmark": args.benchmark,
        "level": args.level,
        "task_id": args.task_id,
        "init_state": args.init_state,
        "environment_seed": args.seed,
    }
    mismatches = {
        key: {"expected": value, "actual": scene.get(key)}
        for key, value in expected_identity.items()
        if scene.get(key) != value
    }
    expected_predicate = list(getattr(args, "target_predicate", []) or [])
    if report.get("native_goal_predicate") != expected_predicate:
        mismatches["native_goal_predicate"] = {
            "expected": expected_predicate,
            "actual": report.get("native_goal_predicate"),
        }
    if mismatches:
        raise ValueError(f"strict_grasp_screen_task_identity_mismatch:{mismatches}")

    policy = screening.get("target_contact_policy")
    if not isinstance(policy, dict):
        raise ValueError("strict_grasp_screen_threshold_contract_mismatch")
    if (
        policy.get("mode") != "pad_only_with_nonpad_clearance"
        or policy.get("strict_pad_only") is not True
        or abs(float(policy.get("minimum_nonpad_robot_to_target_clearance_m", -1.0)) - STRICT_FORBIDDEN_CLEARANCE_M) > 1e-9
        or abs(float(policy.get("maximum_authorized_pad_penetration_m", 1.0)) + STRICT_FINGER_TARGET_PENETRATION_M) > 1e-9
    ):
        raise ValueError("strict_grasp_screen_threshold_contract_mismatch")

    snapshot_root = getattr(args, "snapshot_root", None)
    if snapshot_root is None:
        raise ValueError("strict_grasp_screen_requires_identity_checked_snapshot_root")
    snapshot_root = Path(snapshot_root).resolve()
    manifest_path = snapshot_root / "snapshots/runtime_snapshot_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("strict_grasp_screen_snapshot_manifest_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("scene_metadata"), dict):
        raise ValueError("strict_grasp_screen_snapshot_manifest_malformed")
    active_scene = manifest["scene_metadata"]
    expected_snapshot_identity = {
        "benchmark": args.benchmark,
        "level": args.level,
        "task_id": args.task_id,
        "init_state_index": args.init_state,
        "environment_seed": args.seed,
    }
    snapshot_identity_mismatches = {
        key: {"expected": value, "actual": active_scene.get(key)}
        for key, value in expected_snapshot_identity.items()
        if active_scene.get(key) != value
    }
    if snapshot_identity_mismatches:
        raise ValueError(
            f"strict_grasp_screen_snapshot_identity_mismatch:{snapshot_identity_mismatches}"
        )
    for key in ("model_geometry_sha256", "bddl_sha256"):
        if scene.get(key) != active_scene.get(key):
            raise ValueError(f"strict_grasp_screen_{key}_does_not_match_active_snapshot_root")

    resume_snapshot = getattr(args, "resume_from_runtime_snapshot", None)
    use_resume_snapshot = resume_snapshot is not None
    snapshot_path = (
        Path(resume_snapshot).resolve()
        if use_resume_snapshot
        else snapshot_root / "snapshots/initial_runtime_snapshot.pkl"
    )
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"strict_grasp_screen_start_snapshot_missing:{snapshot_path}")
    snapshot_sha = sha256_file(snapshot_path)
    if report.get("start_snapshot_payload_sha256") != snapshot_sha:
        raise ValueError("strict_grasp_screen_start_snapshot_hash_mismatch")
    instruction = manifest.get("instruction")
    if not isinstance(instruction, str) or not instruction:
        raise ValueError("strict_grasp_screen_snapshot_manifest_instruction_malformed")
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _verified_resume_runtime_snapshot,
    )

    start_snapshot = _verified_resume_runtime_snapshot(
        snapshot_path,
        benchmark=args.benchmark,
        level=args.level,
        task_id=args.task_id,
        seed=args.seed,
        init_state=args.init_state,
        model_sha=str(scene.get("model_geometry_sha256")),
        bddl_sha=str(scene.get("bddl_sha256")),
        instruction=instruction,
    )
    if not use_resume_snapshot:
        initial_snapshot_meta = (
            manifest.get("snapshots", {}).get("initial")
            if isinstance(manifest.get("snapshots"), dict) else None
        )
        if not isinstance(initial_snapshot_meta, dict):
            raise ValueError("strict_grasp_screen_snapshot_manifest_initial_entry_malformed")
        if initial_snapshot_meta.get("payload_sha256") != snapshot_sha:
            raise ValueError("strict_grasp_screen_snapshot_manifest_payload_sha256_mismatch")
        if initial_snapshot_meta.get("model_geometry_sha256") != scene.get("model_geometry_sha256"):
            raise ValueError("strict_grasp_screen_snapshot_manifest_model_sha256_mismatch")
        if initial_snapshot_meta.get("mujoco_integration_state_sha256") != start_snapshot.get(
            "integration_state_sha256"
        ):
            raise ValueError("strict_grasp_screen_snapshot_manifest_integration_state_sha256_mismatch")

    candidate_ids = screening.get("ranked_candidate_ids", [])
    candidates = screening.get("candidates", [])
    if not isinstance(candidates, list) or not isinstance(candidate_ids, list):
        raise ValueError("strict_grasp_screen_candidate_rows_malformed")
    row_candidate_ids = [
        str(row.get("candidate_id", "")) for row in candidates if isinstance(row, dict)
    ]
    if (
        len(row_candidate_ids) != len(candidates)
        or any(not candidate_id for candidate_id in row_candidate_ids)
        or len(set(row_candidate_ids)) != len(row_candidate_ids)
        or any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise ValueError("strict_grasp_screen_candidate_rows_malformed")
    feasible_ids = {
        str(row.get("candidate_id"))
        for row in candidates
        if isinstance(row, dict) and row.get("status") == "offline_feasible_candidate"
    }
    candidate_count_value = screening.get("candidate_count")
    feasible_count_value = screening.get("feasible_count")
    if (
        isinstance(candidate_count_value, bool)
        or not isinstance(candidate_count_value, int)
        or isinstance(feasible_count_value, bool)
        or not isinstance(feasible_count_value, int)
    ):
        raise ValueError("strict_grasp_screen_candidate_counts_malformed")
    candidate_count = candidate_count_value
    feasible_count = feasible_count_value
    if (
        candidate_count != len(candidates)
        or feasible_count != len(feasible_ids)
        or candidate_count < 0
        or feasible_count < 0
        or feasible_count > candidate_count
        or len(candidate_ids) != feasible_count
        or any(candidate_id not in feasible_ids for candidate_id in candidate_ids)
    ):
        raise ValueError("strict_grasp_screen_candidate_counts_malformed")
    candidate_id = None
    status = "blocked_no_strict_grasp_candidate"
    requested_id = getattr(args, "grasp_candidate_id", None)
    if feasible_count > 0:
        candidate_id = str(requested_id) if requested_id is not None else next(
            (str(value) for value in candidate_ids if str(value) in feasible_ids), None
        )
        if candidate_id is None or candidate_id not in feasible_ids:
            raise ValueError("strict_grasp_screen_has_no_ranked_feasible_candidate")
        status = "passed"
    elif requested_id is not None:
        raise ValueError("requested_grasp_candidate_is_not_strictly_feasible")

    return {
        "status": status,
        "screen_report": str(screen_path),
        "screen_report_sha256": sha256_file(screen_path),
        "start_snapshot": str(snapshot_path),
        "start_snapshot_sha256": snapshot_sha,
        "start_snapshot_integration_state_sha256": start_snapshot["integration_state_sha256"],
        "model_geometry_sha256": scene.get("model_geometry_sha256"),
        "bddl_sha256": scene.get("bddl_sha256"),
        "native_goal_predicate": expected_predicate,
        "candidate_count": candidate_count,
        "feasible_count": feasible_count,
        "selected_candidate_id": candidate_id,
        "physical_actions_during_screen": 0,
        "live_state_unchanged": True,
        "safety_certified_by_screen": False,
    }


def _substep_monitoring(trace_path: Path, route_report_path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "incomplete",
        "trace_rows": 0,
        "complete_rows": 0,
        "expected_substeps_per_action": 25,
        "complete": False,
    }
    if not trace_path.is_file():
        return result
    with np.load(trace_path, allow_pickle=False) as archive:
        result["trace_rows"] = int(archive["executed_actions"].shape[0])
        if "post_contact_monitoring_json" not in archive.files:
            result["reason"] = "npz_has_no_per_action_substep_monitoring"
            return result
        rows = archive["post_contact_monitoring_json"]
        parsed = []
        for value in rows:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            row = json.loads(str(value))
            parsed.append(row)
        def row_is_complete(row: dict[str, Any]) -> bool:
            controls = row.get("control_steps")
            if not isinstance(controls, list) or not controls:
                controls = [row]
            return bool(
                row.get("status") == "complete"
                and row.get("captured_physics_substeps", 25) == 25
                and all(
                    isinstance(control, dict)
                    and control.get("status") == "complete"
                    and control.get("expected_physics_substeps") == 25
                    and control.get("captured_physics_substeps") == 25
                    for control in controls
                )
            )
        complete_rows = [
            row for row in parsed
            if row_is_complete(row)
        ]
        result["monitor_rows"] = len(parsed)
        result["complete_rows"] = len(complete_rows)
        result["complete"] = bool(parsed) and len(parsed) == result["trace_rows"] == len(complete_rows)
        result["status"] = "complete" if result["complete"] else "incomplete"
    if route_report_path is not None and route_report_path.is_file():
        route = json.loads(route_report_path.read_text(encoding="utf-8"))
        action_rows = route.get("steps", [])
        route_steps = [row for row in action_rows if row.get("phase", "route") == "route"]
        hold_steps = [row for row in action_rows if row.get("phase") == "pre_route_hold"]
        expected_action_rows = int(route.get("real_execution_steps", len(action_rows)) or 0)
        route_complete = bool(action_rows) and len(action_rows) == expected_action_rows and all(
            row.get("contact_monitoring_status") == "complete"
            for row in action_rows
        )
        result["route_report_steps"] = len(route_steps)
        result["pre_route_hold_steps"] = len(hold_steps)
        result["route_report_action_rows"] = len(action_rows)
        result["route_report_complete"] = route_complete
        result["complete"] = result["complete"] and route_complete
        result["status"] = "complete" if result["complete"] else "incomplete"
    return result


def _outcome(
    *,
    result_path: Path,
    trace_path: Path,
    route_report_path: Path | None,
    strict_report_path: Path | None,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    steps = int(result.get("steps", 0) or 0)
    native_success = result.get("success_once") is True and result.get("done") is True
    monitoring = _substep_monitoring(trace_path, route_report_path)
    strict_report = {}
    if strict_report_path is not None and strict_report_path.is_file():
        strict_report = json.loads(strict_report_path.read_text(encoding="utf-8"))
    strict_pass = bool(
        native_success
        and monitoring["complete"]
        and strict_report.get("strict_10mm_pass") is True
        and strict_report.get("complete_substep_monitoring") is True
        and strict_report.get("native_task_success") is True
        and strict_report.get("source_trace_sha256") == sha256_file(trace_path)
        and strict_report.get("first_violation") is None
    )
    trace_valid = False
    action_count = 0
    if trace_path.is_file():
        with np.load(trace_path, allow_pickle=False) as archive:
            actions = np.asarray(archive["executed_actions"])
            dones = np.asarray(archive["step_dones"], dtype=bool)
            rewards = np.asarray(archive["step_rewards"], dtype=np.float64)
            action_count = len(actions)
            trace_valid = bool(
                actions.ndim == 2 and actions.shape[1] == 7
                and np.isfinite(actions).all()
                and len(actions) == steps == len(dones) == len(rewards)
                and (not native_success or (bool(dones[-1]) and float(rewards[-1]) > 0.0))
            )
    return {
        "native_task_success": native_success,
        "episode_steps": steps,
        "standard_1000_step_success": bool(native_success and steps <= 1000),
        "demonstration_only_success": bool(native_success and steps > 1000),
        "stable_release_verified": False,
        "operational_safety_pass": "not_certified",
        "strict_10mm_pass": strict_pass,
        "strict_safety_status": "pass" if strict_pass else "not_passed_or_not_evaluated",
        "complete_substep_monitoring": monitoring["complete"],
        "substep_monitoring": monitoring,
        "observation_action_trace_valid": trace_valid,
        "actual_action_count": action_count,
        "safe_training_eligible": bool(strict_pass and trace_valid),
        "strict_report_path": str(strict_report_path) if strict_report_path else None,
        "source_safety_inherited": False,
        "classification_note": (
            "native task reward/done, stable release, operational safety, strict 10 mm, "
            "substep coverage, and training eligibility are independent verdicts"
        ),
    }


def _classify_run_status(*, engine_return_code: int, scene_matches: bool, native_task_success: bool) -> str:
    if engine_return_code != 0 or not scene_matches:
        return "incomplete"
    return "complete" if native_task_success else "task_incomplete"


def _classify_experience_run_status(
    *, engine_return_code: int, report_available: bool, scene_matches: bool,
    native_task_success: bool,
) -> str:
    """Keep bounded-executor completion separate from native task completion."""
    if engine_return_code not in (0, 2) or not report_available or not scene_matches:
        return "incomplete"
    return "complete" if native_task_success else "task_incomplete"


def _nested_contact_monitoring_complete(monitoring: Any) -> bool:
    """Validate substep coverage while keeping unsafe measurements observable."""
    if not isinstance(monitoring, dict):
        return False
    expected = 25
    if (
        monitoring.get("status") != "complete"
        or monitoring.get("expected_physics_substeps_per_control_step") != expected
        or monitoring.get("captured_physics_substeps") != expected
        or monitoring.get("unobserved_expected_substeps", 0) != 0
        or monitoring.get("unavailable_contact_substeps", 0) != 0
    ):
        return False
    control_steps = monitoring.get("control_steps")
    if not isinstance(control_steps, list) or len(control_steps) != 1:
        return False
    if any(
        step.get("status") != "complete"
        or step.get("expected_physics_substeps") != expected
        or step.get("captured_physics_substeps") != expected
        or step.get("unobserved_expected_substeps", 0) != 0
        or step.get("capture_error") is not None
        for step in control_steps
    ):
        return False
    geometry = monitoring.get("geometry_monitoring", {})
    substeps = geometry.get("substeps", []) if isinstance(geometry, dict) else []
    measured_statuses = {"complete", "unsafe"}
    return bool(
        geometry.get("status") in measured_statuses
        and len(substeps) == expected
        and all(
            isinstance(row, dict) and row.get("status") in measured_statuses
            for row in substeps
        )
    )


def _localize_cold_start_risks(
    inner_report: dict[str, Any], audit: dict[str, Any], output_dir: Path,
    initial_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Localize the retained episode only; missing evidence never becomes a pass."""
    provenance = inner_report.get("safety_policy_provenance", {})
    policy_path = Path(provenance.get("policy_path") or "")
    archives = [Path(path) for path in audit.get("source_action_record_archives", [])]
    if (not audit.get("success_episode_timestep_continuous") or not archives
            or not policy_path.is_file() or any(not path.is_file() for path in archives)):
        return {"status": "unavailable", "reason": "continuous_episode_or_policy_evidence_missing"}
    from research.gs_opd.repair.safety_risk_localization import localize_risks

    def records():
        for path in archives:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        snapshots = []
        if initial_snapshot is not None and initial_snapshot.is_file():
            snapshots.append({"timestep": 0, "path": str(initial_snapshot.resolve()),
                              "payload_sha256": sha256_file(initial_snapshot),
                              "safe_prefix_verified": False})
        result = localize_risks(records(), policy, snapshots=snapshots, teacher_attested=False)
        result["record_archives"] = [str(path.resolve()) for path in archives]
        result["policy_path"] = str(policy_path.resolve())
        result["policy_sha256"] = sha256_file(policy_path)
        output = output_dir / "observed_safety_risk_windows.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"status": result["status"], "path": str(output), "sha256": sha256_file(output),
                "first_observed_risk": result["first_observed_risk"],
                "strict_safety_pass": result["strict_safety_pass"]}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"status": "unavailable", "reason": f"risk_localization_failed:{type(exc).__name__}:{exc}"}


def _screen_current_episode_reconstruction(
    args: argparse.Namespace, audit: dict[str, Any], output_dir: Path,
) -> dict[str, Any]:
    """Use only the just-generated grasp to screen bounded local alternatives."""
    reports = audit.get("source_search_reports", [])
    if len(reports) != 1:
        return {"status": "unavailable", "reason": "current_episode_grasp_report_not_unique"}
    source = Path(reports[0]).resolve()
    engine_root = (output_dir / "task_success_engine").resolve()
    if not source.is_file() or not source.is_relative_to(engine_root):
        return {"status": "unavailable", "reason": "grasp_report_not_from_current_engine"}
    try:
        context = json.loads((source.parent / "active_target_context.json").read_text(encoding="utf-8"))
        predicate = context["native_goal_predicate"]
        if not isinstance(predicate, list) or len(predicate) != 3 or not all(isinstance(x, str) for x in predicate):
            raise ValueError("current_grasp_native_predicate_invalid")
        screen_dir = output_dir / "safety_reconstruction_screen"
        command = [str(args.rlinf_python), "-m", "research.gs_opd.repair.task_success_first_grasp_search",
                   "--snapshot-root", str(args.snapshot_root), "--output-root", str(screen_dir),
                   "--benchmark", args.benchmark, "--level", str(args.level), "--task-id", str(args.task_id),
                   "--init-state", str(args.init_state), "--seed", str(args.seed),
                   "--port", str(args.bridge_port + 3), "--demonstration-horizon", str(args.demonstration_horizon),
                   "--target-predicate", *predicate, "--strict-grasp-screen-only",
                   "--strict-screen-reference-report", str(source), "--maximum-candidates", "24"]
        with (output_dir / "safety_reconstruction_screen.log").open("wb") as log:
            completed = _run_bounded_engine(command, log=log, wall_time_budget_s=min(args.wall_time_budget_s, 600.0))
        screen_reports = sorted(screen_dir.glob("task_success_first_grasp_*/strict_grasp_geometry_screen.json"))
        if completed.returncode != 0 or len(screen_reports) != 1:
            return {"status": "unavailable", "reason": "bounded_grasp_screen_failed",
                    "return_code": completed.returncode}
        screen = json.loads(screen_reports[0].read_text(encoding="utf-8"))
        return {"status": "screened_not_physically_reconstructed", "path": str(screen_reports[0]),
                "sha256": sha256_file(screen_reports[0]), "source_current_episode_grasp": str(source),
                "feasible_candidates": screen.get("screening", {}).get("feasible_count"),
                "physical_actions": 0, "safety_certified": False}
    except (OSError, ValueError, TypeError, KeyError, TimeoutError) as exc:
        return {"status": "unavailable", "reason": f"reconstruction_screen_failed:{type(exc).__name__}:{exc}"}


def _experience_physical_action_audit(inner_report: dict[str, Any]) -> dict[str, Any]:
    """Summarize archived physical action rows without upgrading safety labels."""
    search_report_paths = []
    record_sources: list[dict[str, Any]] = []
    missing_record_sources = 0
    discarded_student_segments: list[dict[str, Any]] = []
    for segment in inner_report.get("segments", []):
        segment_type = segment.get("type")
        if segment_type in {"student_first", "student_handoff_after_stage_goal"}:
            actions = int(segment.get("actions_executed", 0) or 0)
            if not actions:
                continue
            discarded = (
                segment_type == "student_first" and segment.get("task_success") is not True
            ) or (
                segment_type == "student_handoff_after_stage_goal"
                and isinstance(
                    segment.get("student_handoff_recovery", inner_report.get("student_handoff_recovery")),
                    dict,
                )
            )
            if discarded:
                discarded_student_segments.append({"kind": segment_type, "actions": actions})
                continue
            artifacts = segment.get("artifacts", {})
            records_value = artifacts.get("records") if isinstance(artifacts, dict) else None
            if not records_value:
                missing_record_sources += 1
                continue
            record_sources.append({
                "kind": segment_type,
                "records_path": Path(records_value).resolve(),
                "durable_records_path": None,
                "physical_branches": 0,
                "expected_actions": actions,
                "reported_physics_substeps": None,
            })
        elif segment_type == "bounded_expert_grasp_search":
            report_value = segment.get("report")
            report_path = Path(report_value).resolve() if report_value else None
            if report_path is None or not report_path.is_file():
                missing_record_sources += 1
                continue
            search_report_paths.append(report_path)
            search_report = json.loads(report_path.read_text(encoding="utf-8"))
            branches = search_report.get("physical_branches", [])
            record_sources.append({
                "kind": "bounded_expert_grasp_search",
                "records_path": report_path.parent / "physical_step_records.jsonl.gz",
                "durable_records_path": None,
                "physical_branches": len(branches),
                "expected_actions": sum(
                    int(
                        branch.get(
                            "branch_control_actions",
                            branch.get("attempted_control_actions_archived", 0),
                        ) or 0
                    )
                    for branch in branches
                ),
                "reported_physics_substeps": None,
            })
        elif segment_type == "expert_articulated_fixture_goal":
            actions = int(segment.get("recorded_actions", 0) or 0)
            if not actions:
                continue
            artifacts = segment.get("artifacts", {})
            records_value = artifacts.get("records") if isinstance(artifacts, dict) else None
            if not records_value:
                missing_record_sources += 1
                continue
            record_sources.append({
                "kind": segment_type,
                "records_path": Path(records_value).resolve(),
                "durable_records_path": None,
                "physical_branches": 1,
                "expected_actions": actions,
                "reported_physics_substeps": segment.get("recorded_physics_substeps"),
            })
        elif segment_type == "resumed_expert_release_withdrawal":
            actions = int(segment.get("recorded_actions", 0) or 0)
            if not actions:
                continue
            artifacts = segment.get("artifacts", {})
            records_value = artifacts.get("records") if isinstance(artifacts, dict) else None
            durable_value = artifacts.get("durable_records") if isinstance(artifacts, dict) else None
            if not records_value:
                missing_record_sources += 1
                continue
            record_sources.append({
                "kind": segment_type,
                "records_path": Path(records_value).resolve(),
                "durable_records_path": Path(durable_value).resolve() if durable_value else None,
                "physical_branches": 1,
                "expected_actions": actions,
                "reported_physics_substeps": segment.get("recorded_physics_substeps"),
            })
        elif segment_type == "expert_native_goal_stage_control":
            artifacts = segment.get("artifacts", {})
            records_value = artifacts.get("records") if isinstance(artifacts, dict) else None
            durable_value = artifacts.get("durable_records") if isinstance(artifacts, dict) else None
            archive_value = inner_report.get("archive")
            if not records_value and archive_value:
                placement_root = Path(archive_value) / "expert_native_placement"
                matches = sorted(placement_root.glob("placement_branch_*/physical_step_records.jsonl.gz"))
                if len(matches) == 1:
                    records_value = str(matches[0])
                    durable_candidate = matches[0].with_name("durable_physical_step_records.jsonl")
                    durable_value = str(durable_candidate) if durable_candidate.is_file() else None
            if records_value:
                placement_search = segment.get("placement_search", {})
                record_sources.append({
                    "kind": "expert_native_goal_stage_control",
                    "records_path": Path(records_value).resolve(),
                    "durable_records_path": Path(durable_value).resolve() if durable_value else None,
                    "physical_branches": int(placement_search.get("physical_branch_count", 0) or 0),
                    "expected_actions": int(segment.get("recorded_actions", 0) or 0),
                    "reported_physics_substeps": (
                        int(segment["recorded_physics_substeps"])
                        if segment.get("recorded_physics_substeps") is not None else None
                    ),
                })
            else:
                missing_record_sources += 1
    audit: dict[str, Any] = {
        "status": "not_available",
        "source_search_reports": [str(path) for path in search_report_paths],
        "source_action_record_archives": [str(source["records_path"]) for source in record_sources],
        "durable_action_record_archives": [
            str(source["durable_records_path"])
            for source in record_sources if source["durable_records_path"] is not None
        ],
        "record_source_count": len(record_sources),
        "missing_record_sources": missing_record_sources,
        "discarded_student_segments": discarded_student_segments,
        "record_source_summaries": [],
        "physical_branches": 0,
        "physical_actions_archived": 0,
        "action_record_rows": 0,
        "captured_physics_substeps": 0,
        "complete_substep_monitoring_actions": 0,
        "complete_substep_monitoring": None,
        "action_dim_7_valid": None,
        "state_action_rows_valid": None,
        "branch_action_count_matches_record_rows": None,
        "reported_physics_substeps_match_record_rows": None,
        "all_physical_record_sources_contiguous": None,
        "geometry_monitoring_status_counts": {},
        "phase_action_counts": {},
        "strict_safety_verdict": "not_evaluated_by_this_audit",
        "safety_certified": False,
    }
    if not record_sources:
        return audit

    records_ok = True
    row_indices_ok = True
    actions_ok = True
    state_action_ok = True
    complete_monitor_rows = 0
    recorded_actions = 0
    branch_count = 0
    branch_action_total = 0
    substep_total = 0
    reported_substeps_match = True
    has_reported_substeps = False
    phase_counts: dict[str, int] = {}
    geometry_status_counts: dict[str, int] = {}
    source_summaries = []
    timeline_valid = True
    previous_timestep = 0
    for source in record_sources:
        source_branch_count = int(source["physical_branches"])
        expected_actions = int(source["expected_actions"])
        branch_count += source_branch_count
        branch_action_total += expected_actions
        source_rows = 0
        source_substeps = 0
        source_complete_monitor_rows = 0
        source_indices_ok = True
        records_path = Path(source["records_path"])
        if not records_path.is_file():
            records_ok = False
            source_summaries.append({
                "kind": source["kind"],
                "records_path": str(records_path),
                "physical_branches": source_branch_count,
                "expected_actions": expected_actions,
                "action_record_rows": 0,
                "captured_physics_substeps": 0,
                "status": "missing_records",
            })
            continue
        with gzip.open(records_path, "rt", encoding="utf-8") as handle:
            for expected_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                recorded_actions += 1
                source_rows += 1
                timestep = row.get("timestep")
                if type(timestep) is not int or timestep != previous_timestep + 1:
                    timeline_valid = False
                if type(timestep) is int:
                    previous_timestep = timestep
                action = np.asarray(row.get("action_7d", []), dtype=np.float64)
                actions_ok = actions_ok and action.shape == (7,) and bool(np.isfinite(action).all())
                state_action_ok = state_action_ok and (
                    isinstance(row.get("state"), dict)
                    and isinstance(row.get("snapshot_state"), dict)
                    and isinstance(row.get("reward"), (float, int))
                    and isinstance(row.get("done"), bool)
                )
                if row.get("step_index") != expected_index:
                    records_ok = False
                    row_indices_ok = False
                    source_indices_ok = False
                phase = str(row.get("phase", "unknown"))
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                monitoring = row.get("contact_monitoring", {})
                if not isinstance(monitoring, dict):
                    monitoring = {}
                if _nested_contact_monitoring_complete(monitoring):
                    complete_monitor_rows += 1
                    source_complete_monitor_rows += 1
                captured = int(monitoring.get("captured_physics_substeps", 0) or 0)
                substep_total += captured
                source_substeps += captured
                geometry = monitoring.get("geometry_monitoring", {})
                geometry_status = (
                    geometry.get("status", "missing") if isinstance(geometry, dict) else "missing"
                )
                geometry_status = str(geometry_status)
                geometry_status_counts[geometry_status] = geometry_status_counts.get(geometry_status, 0) + 1
        records_ok = records_ok and source_rows == expected_actions
        expected_substeps = source.get("reported_physics_substeps")
        if expected_substeps is not None:
            has_reported_substeps = True
            reported_substeps_match = reported_substeps_match and source_substeps == expected_substeps
        source_summaries.append({
            "kind": source["kind"],
            "records_path": str(records_path),
            "durable_records_path": (
                str(source["durable_records_path"])
                if source["durable_records_path"] is not None else None
            ),
            "durable_records_available": bool(
                source["durable_records_path"] is not None
                and Path(source["durable_records_path"]).is_file()
            ),
            "physical_branches": source_branch_count,
            "expected_actions": expected_actions,
            "action_record_rows": source_rows,
            "captured_physics_substeps": source_substeps,
            "complete_substep_monitoring_actions": source_complete_monitor_rows,
            "contiguous_step_indices": source_indices_ok,
            "reported_physics_substeps": expected_substeps,
            "status": "complete" if source_rows == expected_actions and source_indices_ok else "incomplete",
        })

    all_monitored = (recorded_actions > 0 and complete_monitor_rows == recorded_actions
                     and missing_record_sources == 0 and records_ok
                     and recorded_actions == branch_action_total)
    count_matches = recorded_actions == branch_action_total and missing_record_sources == 0
    records_ok = records_ok and count_matches
    source_summaries_ok = all(source["status"] == "complete" for source in source_summaries)
    final_native = inner_report.get("final_native_task_status", {})
    success_timeline = bool(
        isinstance(final_native, dict) and final_native.get("task_success") is True
        and type(final_native.get("timestep")) is int
        and timeline_valid and previous_timestep == final_native["timestep"]
        and recorded_actions == final_native["timestep"]
    )
    multi_goal_episode = inner_report.get("multi_goal_episode", {})
    snapshot_boundaries_verified = (
        bool(multi_goal_episode.get("stage_snapshot_boundaries_verified"))
        if isinstance(multi_goal_episode, dict) and multi_goal_episode.get("enabled") is True
        else None
    )
    audit.update({
        "status": "complete" if (
            records_ok and actions_ok and state_action_ok and all_monitored
            and source_summaries_ok and reported_substeps_match
        ) else "incomplete",
        "record_source_summaries": source_summaries,
        "physical_branches": branch_count,
        "physical_actions_archived": branch_action_total,
        "action_record_rows": recorded_actions,
        "captured_physics_substeps": substep_total,
        "complete_substep_monitoring_actions": complete_monitor_rows,
        "complete_substep_monitoring": all_monitored,
        "success_episode_timestep_continuous": success_timeline,
        "success_episode_complete_substep_monitoring": bool(all_monitored and success_timeline),
        "snapshot_boundary_continuity": (
            "verified_by_native_goal_stage_scheduler"
            if snapshot_boundaries_verified is True
            else "failed_native_goal_stage_boundary_verification"
            if snapshot_boundaries_verified is False
            else "not_applicable_single_stage"
        ),
        "snapshot_boundaries_verified": snapshot_boundaries_verified,
        "action_dim_7_valid": actions_ok,
        "state_action_rows_valid": state_action_ok,
        "branch_action_count_matches_record_rows": count_matches,
        "reported_physics_substeps_match_record_rows": (
            reported_substeps_match if has_reported_substeps else None
        ),
        "all_physical_record_sources_contiguous": row_indices_ok,
        "geometry_monitoring_status_counts": geometry_status_counts,
        "phase_action_counts": phase_counts,
        "single_branch_contiguous_records": branch_count == 1 and row_indices_ok and count_matches,
        "monitoring_interpretation": "unsafe geometry samples are observed violations, not missing substep records",
    })
    return audit


def _experience_failure_reasons(inner_report: dict[str, Any]) -> list[str]:
    reasons: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            if value and value not in reasons:
                reasons.append(value)
        elif isinstance(value, (list, tuple)):
            if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int):
                collect(value[0])
                return
            for item in value:
                collect(item)

    collect(inner_report.get("error"))
    for segment in inner_report.get("segments", []):
        collect(segment.get("failure_reasons"))
        for branch in segment.get("physical_branches", []):
            collect(branch.get("failure_reason"))
        collect(segment.get("failure_reason"))
        collect(segment.get("transport_failure_reason"))
        collect(segment.get("post_success_terminal_exception"))
    collect(inner_report.get("failure_reason"))
    return reasons


def _native_goal_value_map(native_status: Any) -> dict[tuple[str, ...], bool | None]:
    if not isinstance(native_status, dict):
        return {}
    rows = native_status.get("native_goal_values")
    if not isinstance(rows, list):
        return {}
    result: dict[tuple[str, ...], bool | None] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("predicate"), list):
            continue
        predicate = tuple(str(token) for token in row["predicate"])
        if predicate:
            result[predicate] = row.get("value") if isinstance(row.get("value"), bool) else None
    return result


def _native_goal_stage_boundary_matches(
    parent_status: Any,
    child_status: Any,
) -> bool:
    """Require the resumed stage to observe the exact native prefix status."""
    if not isinstance(parent_status, dict) or not isinstance(child_status, dict):
        return False
    return bool(
        _native_goal_value_map(parent_status)
        and _native_goal_value_map(parent_status) == _native_goal_value_map(child_status)
        and parent_status.get("timestep") == child_status.get("timestep")
        and parent_status.get("task_success") == child_status.get("task_success")
    )


def _report_terminal_runtime_snapshot(inner_report: dict[str, Any]) -> Path | None:
    for segment in reversed(inner_report.get("segments", [])):
        snapshot = segment.get("terminal_snapshot") if isinstance(segment, dict) else None
        if snapshot and Path(snapshot).is_file():
            return Path(snapshot).resolve()
    return None


def _report_native_goal_contact_snapshot(
    inner_report: dict[str, Any], predicate: list[str] | None,
) -> Path | None:
    """Find an archived still-held contact Snapshot for a true, unsettled native goal."""
    if not predicate:
        return None
    for parent in reversed(inner_report.get("segments", [])):
        if not isinstance(parent, dict) or parent.get("type") != "expert_native_goal_stage_control":
            continue
        attempts = parent.get("placement_search", {}).get("candidate_attempts", [])
        for attempt in reversed(attempts):
            candidate = attempt.get("candidate", {})
            snapshot = attempt.get("terminal_snapshot")
            if (
                candidate.get("predicate") == predicate
                and attempt.get("target_predicate_value") is True
                and snapshot
                and Path(snapshot).is_file()
            ):
                return Path(snapshot).resolve()
    return None


def _run_cold_start_native_goal_continuations(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    checkpoint_file: Path,
    first_report: dict[str, Any],
    first_report_path: Path,
    first_return_code: int,
    wall_time_started: float,
    wall_time_budget_s: float,
    history_reports: list[dict[str, Any]] | None = None,
    history_report_paths: list[Path] | None = None,
    history_stage_records: list[dict[str, Any]] | None = None,
    history_discarded_attempt_reports: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Run pending native goals from same-run exact Runtime Snapshots only."""
    history_reports = history_reports or [first_report]
    history_report_paths = history_report_paths or [first_report_path.resolve()]
    stage_reports = list(history_reports)
    stage_report_paths = [path.resolve() for path in history_report_paths]
    stage_records = list(history_stage_records or [])
    discarded_attempt_reports = list(history_discarded_attempt_reports or [])
    initial_native = history_reports[0].get("task", {}).get("initial_native_task_status", {})
    initial_goals = _native_goal_value_map(initial_native)
    if not stage_records:
        stage_records = [{
            "stage_index": 1,
            "executor_report": str(stage_report_paths[0]),
            "target_predicate": history_reports[0].get("task", {}).get("native_goal_predicate"),
            "input_snapshot": str(args.snapshot_root.resolve() / "snapshots/initial_runtime_snapshot.pkl"),
            "input_snapshot_kind": "identity_checked_legal_t0",
            "initial_native_task_status": initial_native,
            "final_native_task_status": history_reports[0].get("final_native_task_status"),
            "engine_return_code": first_return_code,
        }]
    last_return_code = first_return_code
    stop_reason: str | None = None
    maximum_stages = max(1, len(initial_goals) * 2)

    current = history_reports[-1]
    previous_stage_index = max(
        (int(row.get("stage_index", 0)) for row in stage_records), default=len(stage_reports),
    )
    for stage_index in range(previous_stage_index + 1, maximum_stages + 1):
        if current.get("task_success") is True:
            break
        parent_status = current.get("final_native_task_status")
        parent_values = _native_goal_value_map(parent_status)
        pending = [predicate for predicate, value in parent_values.items() if value is False]
        if not pending:
            stop_reason = "no_pending_supported_native_goal_before_task_success"
            break
        if last_return_code not in (0, 2):
            stop_reason = f"previous_goal_stage_engine_exit_{last_return_code}"
            break

        partial_goal_segments = [
            segment for segment in current.get("segments", [])
            if isinstance(segment, dict) and segment.get("status") == "partial_predicate_completed"
        ]
        run_root = output_dir.resolve()
        recovery_contact = False
        continuation_report_path: Path | None = None
        if partial_goal_segments:
            snapshot_path = _report_terminal_runtime_snapshot(current)
            if snapshot_path is None:
                stop_reason = "same_run_native_goal_terminal_snapshot_missing"
                break
            if not snapshot_path.is_relative_to(run_root):
                stop_reason = "continuation_snapshot_escaped_current_V3_run"
                break
        else:
            active_predicate = current.get("task", {}).get("native_goal_predicate")
            active_value = parent_values.get(tuple(active_predicate or []))
            contact_snapshot = _report_native_goal_contact_snapshot(current, active_predicate)
            continuation_report_path = stage_report_paths[-1]
            if active_value is not True or contact_snapshot is None:
                stop_reason = "previous_stage_did_not_reach_a_settled_native_goal_or_recoverable_contact"
                break
            if (
                not continuation_report_path.is_file()
                or not continuation_report_path.is_relative_to(run_root)
                or not contact_snapshot.is_relative_to(run_root)
            ):
                stop_reason = "placement_contact_resume_source_escaped_current_V3_run"
                break
            snapshot_path = contact_snapshot
            recovery_contact = True

        stage_root = output_dir / "continuation_stages" / f"stage_{stage_index:02d}"
        copied_report_path: Path | None = None
        if recovery_contact:
            copied_report_path = stage_root / "continuation_source_placement_report.json"
            copied_report_path.parent.mkdir(parents=True, exist_ok=True)
            copied_report_path.write_bytes(continuation_report_path.read_bytes())
        command = _experience_task_success_command(
            args,
            output_dir=stage_root,
            checkpoint_file=checkpoint_file,
            continuation_snapshot=None if recovery_contact else snapshot_path,
            continuation_placement_report=copied_report_path,
            continuation_target_predicate=(
                current.get("task", {}).get("native_goal_predicate") if recovery_contact else None
            ),
        )
        remaining = wall_time_budget_s - (time.monotonic() - wall_time_started)
        if remaining <= 1.0:
            stop_reason = "V3_total_wall_time_budget_exhausted_before_native_goal_continuation"
            break
        log_path = stage_root / "execution.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("wb") as log:
                completed = _run_bounded_engine(command, log=log, wall_time_budget_s=remaining)
        except TimeoutError as exc:
            stop_reason = f"{type(exc).__name__}:{exc}"
            last_return_code = 124
            break
        last_return_code = completed.returncode
        reports = sorted((stage_root / "task_success_engine").glob("task_success_first_executor_*/run_report.json"))
        if len(reports) != 1:
            stop_reason = "continuation_executor_report_missing_or_ambiguous"
            break
        stage_report_path = reports[0].resolve()
        try:
            next_report = json.loads(stage_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            stop_reason = f"continuation_executor_report_unreadable:{type(exc).__name__}:{exc}"
            break

        resumed = next_report.get("resumed_from_runtime_snapshot", {})
        resumed_contact = next_report.get("resumed_from_native_goal_contact_snapshot", {})
        expected_start_status = next_report.get("task", {}).get("initial_native_task_status")
        resumed_status = resumed.get("status") if resumed else resumed_contact.get("status")
        resumed_sha = resumed.get("payload_sha256") if resumed else resumed_contact.get("snapshot_sha256")
        boundary_ok = bool(
            resumed_status == "verified_exact"
            and resumed_sha == sha256_file(snapshot_path)
            and _native_goal_stage_boundary_matches(parent_status, expected_start_status)
        )
        new_status = next_report.get("final_native_task_status")
        new_values = _native_goal_value_map(new_status)
        newly_satisfied = [
            predicate for predicate, value in new_values.items()
            if value is True and parent_values.get(predicate) is not True
        ]
        regressed = [
            predicate for predicate, value in new_values.items()
            if value is False and parent_values.get(predicate) is True
        ]
        stage_record = {
            "stage_index": stage_index,
            "executor_report": str(stage_report_path),
            "target_predicate": next_report.get("task", {}).get("native_goal_predicate"),
            "input_snapshot": str(snapshot_path),
            "input_snapshot_sha256": sha256_file(snapshot_path),
            "input_snapshot_kind": "same_run_native_goal_contact_for_release" if recovery_contact else "same_run_settled_native_goal",
            "input_placement_report": str(copied_report_path) if copied_report_path else None,
            "input_native_task_status": parent_status,
            "restored_native_task_status": expected_start_status,
            "exact_stage_boundary_verified": boundary_ok,
            "final_native_task_status": new_status,
            "newly_satisfied_predicates": [list(predicate) for predicate in newly_satisfied],
            "previously_satisfied_predicates_regressed": [list(predicate) for predicate in regressed],
            "engine_return_code": last_return_code,
        }
        recovery_settled = recovery_contact and any(
            isinstance(segment, dict) and segment.get("status") == "partial_predicate_completed"
            for segment in next_report.get("segments", [])
        )
        stage_failure = None
        if not boundary_ok:
            stage_failure = "continuation_stage_snapshot_or_native_boundary_verification_failed"
        elif next_report.get("scope", {}).get("model_geometry_sha256") != first_report.get("scope", {}).get("model_geometry_sha256"):
            stage_failure = "continuation_stage_model_identity_changed"
        elif regressed:
            stage_failure = "continuation_stage_regressed_a_previously_satisfied_native_goal"
        elif not newly_satisfied and not recovery_settled and next_report.get("task_success") is not True:
            stage_failure = "continuation_stage_made_no_native_goal_progress"
        if stage_failure is not None:
            stage_record["episode_state_committed"] = False
            stage_record["attempt_failure_reason"] = stage_failure
            stage_records.append(stage_record)
            discarded_attempt_reports.append({
                "stage_index": stage_index,
                "executor_report": str(stage_report_path),
                "archive": next_report.get("archive"),
                "attempt_status": next_report.get("status"),
                "failure_reason": stage_failure,
                "executor_failure_reason": next_report.get("error") or next_report.get("failure_reason"),
                "actions_excluded_from_continuous_episode": True,
            })
            stop_reason = stage_failure
            break
        stage_record["episode_state_committed"] = True
        stage_records.append(stage_record)
        stage_reports.append(next_report)
        stage_report_paths.append(stage_report_path)
        current = next_report
    else:
        stop_reason = "native_goal_continuation_stage_limit_reached"

    final_report = dict(current)
    flattened_segments: list[dict[str, Any]] = []
    for stage_index, (stage_report_path, stage_report) in enumerate(zip(stage_report_paths, stage_reports), start=1):
        matching_record = next((
            row for row in stage_records
            if Path(str(row.get("executor_report", ""))).resolve() == stage_report_path
        ), {})
        native_stage_index = int(matching_record.get("stage_index", stage_index))
        for segment in stage_report.get("segments", []):
            if isinstance(segment, dict):
                flattened_segments.append({
                    "native_goal_stage_index": native_stage_index,
                    "student_handoff_recovery": stage_report.get("student_handoff_recovery"),
                    **segment,
                })
    final_report["segments"] = flattened_segments
    final_report["cold_start_protocol"] = stage_reports[0].get("cold_start_protocol", {})
    if isinstance(first_report.get("task"), dict):
        final_task = dict(final_report.get("task", {}))
        final_task["initial_native_task_status"] = first_report["task"].get("initial_native_task_status")
        final_task["initial_goal_predicate"] = first_report["task"].get("native_goal_predicate")
        final_report["task"] = final_task
    final_report["student_handoff_recovery"] = any(
        isinstance(stage_report.get("student_handoff_recovery"), dict)
        for stage_report in stage_reports
    )
    final_report["archive"] = str(output_dir)
    episode_success = final_report.get("task_success") is True
    committed_stage_records = [
        row for row in stage_records if row.get("episode_state_committed", True) is True
    ]
    multi_goal_episode = {
        "enabled": True,
        "scheduler": "requery_all_native_BDDL_goals_after_each_settled_stage",
        "initial_native_goals": [
            {"predicate": list(predicate), "value": value}
            for predicate, value in initial_goals.items()
        ],
        "stages": committed_stage_records,
        "stage_count": len(committed_stage_records),
        "stage_attempt_count": len(stage_records),
        "discarded_attempt_reports": discarded_attempt_reports,
        "stage_snapshot_boundaries_verified": all(
            row.get("exact_stage_boundary_verified", True) is True for row in committed_stage_records
        ),
        "previously_completed_goals_rechecked_after_each_stage": True,
        "same_run_snapshot_only": True,
        "target_success_demonstrations_or_keyposes_used": False,
        "official_demonstrations_used": False,
        "historical_actions_replayed": False,
        "native_task_success": episode_success,
        "stop_reason": None if episode_success else stop_reason or "executor_stopped_with_pending_native_goals",
        "executor_report_paths": [str(path) for path in stage_report_paths],
    }
    final_report["multi_goal_episode"] = multi_goal_episode
    if not episode_success and stop_reason:
        final_report["multi_goal_continuation_failure_reason"] = stop_reason
    return final_report, last_return_code, multi_goal_episode


def _load_v3_continuation_history(
    args: argparse.Namespace, output_dir: Path,
) -> dict[str, Any]:
    """Load an incomplete cold-start V3 run for exact same-archive continuation."""
    report_path = getattr(args, "continue_from_v3_report", None)
    if report_path is None:
        raise ValueError("continue_from_v3_report_is_required")
    report_path = Path(report_path).resolve()
    output_dir = output_dir.resolve()
    if report_path != (output_dir / "v3_report.json").resolve():
        raise ValueError("continue_from_v3_report_must_be_the_current_V3_archive_report")
    if not output_dir.is_dir() or not report_path.is_file():
        raise FileNotFoundError("V3_continuation_archive_or_report_is_missing")
    prior = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        prior.get("status") not in {"task_incomplete", "incomplete"}
        or prior.get("backend") != "experience_task_success"
        or prior.get("cold_start_protocol", {}).get("enabled") is not True
        or prior.get("multi_goal_episode", {}).get("native_task_success") is True
    ):
        raise ValueError("V3_continuation_requires_an_incomplete_cold_start_multigoal_archive")
    task = prior.get("task", {})
    expected = {
        "benchmark": args.benchmark,
        "level": args.level,
        "task_id": args.task_id,
        "init_state": args.init_state,
        "seed": args.seed,
        "instruction": args.instruction,
    }
    for key, value in expected.items():
        if task.get(key) != value:
            raise ValueError(f"V3_continuation_{key}_mismatch")
    recorded_t0 = prior.get("inputs", {}).get("snapshot_root")
    if not recorded_t0 or Path(recorded_t0).resolve() != args.snapshot_root.resolve():
        raise ValueError("V3_continuation_legal_t0_snapshot_root_mismatch")
    episode = prior.get("multi_goal_episode", {})
    raw_paths = episode.get("executor_report_paths", [])
    raw_stages = episode.get("stages", [])
    if not raw_paths or len(raw_paths) != len(raw_stages):
        raise ValueError("V3_continuation_executor_stage_history_is_incomplete")
    paths: list[Path] = []
    reports: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    discarded_attempt_reports: list[dict[str, Any]] = []
    model_sha: str | None = None
    for index, (raw_path, raw_stage) in enumerate(zip(raw_paths, raw_stages)):
        path = Path(raw_path).resolve()
        if not path.is_relative_to(output_dir) or not path.is_file():
            raise ValueError("V3_continuation_executor_report_escaped_current_archive")
        report = json.loads(path.read_text(encoding="utf-8"))
        scope = report.get("scope", {})
        for key, value in {
            "benchmark": args.benchmark, "level": args.level, "task_id": args.task_id,
            "init_state": args.init_state, "seed": args.seed,
        }.items():
            if scope.get(key) != value:
                raise ValueError(f"V3_continuation_executor_{key}_mismatch")
        current_model = scope.get("model_geometry_sha256")
        if model_sha is None:
            model_sha = current_model
        elif model_sha != current_model:
            raise ValueError("V3_continuation_model_identity_changed_between_stages")
        final_native = report.get("final_native_task_status")
        stage_record = dict(raw_stage)
        stage_record["executor_report"] = str(path)
        if isinstance(final_native, dict):
            stage_record["episode_state_committed"] = True
            paths.append(path)
            reports.append(report)
            stage_records.append(stage_record)
            continue

        # Preserve a failed attempt as an archive, but resume scheduling from
        # its exact prior settled state only when the attempt restored that
        # same current-run Snapshot and left the native boundary unchanged.
        if index != len(raw_paths) - 1 or not reports:
            raise ValueError("V3_continuation_nonterminal_executor_report_is_not_a_trailing_retry")
        restored = report.get("resumed_from_runtime_snapshot", {})
        previous_native = reports[-1].get("final_native_task_status")
        start_native = report.get("task", {}).get("initial_native_task_status")
        snapshot_path = Path(str(restored.get("path") or "")).resolve()
        if (
            report.get("status") != "bounded_expert_search_exhausted"
            or restored.get("status") != "verified_exact"
            or not snapshot_path.is_file()
            or not snapshot_path.is_relative_to(output_dir)
            or restored.get("payload_sha256") != sha256_file(snapshot_path)
            or not _native_goal_stage_boundary_matches(previous_native, start_native)
        ):
            raise ValueError("V3_continuation_failed_attempt_did_not_restore_its_exact_prior_stage")
        attempt_segment = next((
            row for row in report.get("segments", [])
            if isinstance(row, dict) and row.get("type") == "bounded_expert_grasp_search"
        ), None)
        attempt_archive = Path(str((attempt_segment or {}).get("archive") or ""))
        grasp_reports = sorted(attempt_archive.glob("run_report.json")) if attempt_archive.is_dir() else []
        if len(grasp_reports) != 1:
            raise ValueError("V3_continuation_failed_attempt_grasp_search_archive_is_missing")
        grasp_report = json.loads(grasp_reports[0].read_text(encoding="utf-8"))
        search_start = grasp_report.get("scope", {})
        if (
            Path(str(search_start.get("search_start_runtime_snapshot") or "")).resolve() != snapshot_path
            or search_start.get("search_start_runtime_snapshot_sha256") != sha256_file(snapshot_path)
        ):
            raise ValueError("V3_continuation_failed_attempt_search_did_not_start_from_verified_prior_stage")
        stage_record.update({
            "episode_state_committed": False,
            "attempt_status": report.get("status"),
            "failure_reason": report.get("error"),
            "restored_prior_stage_snapshot": str(snapshot_path),
            "restored_prior_stage_snapshot_sha256": sha256_file(snapshot_path),
        })
        stage_records.append(stage_record)
        discarded_attempt_reports.append({
            "stage_index": raw_stage.get("stage_index"),
            "executor_report": str(path),
            "archive": report.get("archive"),
            "attempt_status": report.get("status"),
            "failure_reason": report.get("error"),
            "actions_excluded_from_continuous_episode": True,
            "exact_prior_stage_snapshot_restored_before_search": True,
        })
    for prior_attempt in episode.get("discarded_attempt_reports", []):
        if not isinstance(prior_attempt, dict):
            raise ValueError("V3_continuation_discarded_attempt_record_is_malformed")
        attempt_path = Path(str(prior_attempt.get("executor_report") or "")).resolve()
        if not attempt_path.is_relative_to(output_dir) or not attempt_path.is_file():
            raise ValueError("V3_continuation_discarded_attempt_report_escaped_current_archive")
        if any(row.get("executor_report") == str(attempt_path) for row in discarded_attempt_reports):
            continue
        attempt_report = json.loads(attempt_path.read_text(encoding="utf-8"))
        attempt_scope = attempt_report.get("scope", {})
        if (
            attempt_scope.get("benchmark") != args.benchmark
            or attempt_scope.get("level") != args.level
            or attempt_scope.get("task_id") != args.task_id
            or attempt_scope.get("init_state") != args.init_state
            or attempt_scope.get("seed") != args.seed
            or attempt_scope.get("model_geometry_sha256") != model_sha
        ):
            raise ValueError("V3_continuation_discarded_attempt_scene_or_model_mismatch")
        discarded_attempt_reports.append(dict(prior_attempt))
        if not any(Path(str(row.get("executor_report", ""))).resolve() == attempt_path for row in stage_records):
            stage_records.append({
                "stage_index": prior_attempt.get("stage_index"),
                "executor_report": str(attempt_path),
                "episode_state_committed": False,
                "attempt_status": attempt_report.get("status"),
                "attempt_failure_reason": prior_attempt.get("failure_reason"),
            })
    stage_records.sort(key=lambda row: int(row.get("stage_index", 0)))
    if not reports:
        raise ValueError("V3_continuation_has_no_committed_native_stage_to_resume")
    if reports[-1].get("task_success") is True:
        raise ValueError("V3_continuation_archive_already_has_native_task_success")
    final_native = reports[-1].get("final_native_task_status", {})
    if not isinstance(final_native, dict) or final_native.get("task_success") is True:
        raise ValueError("V3_continuation_last_executor_native_status_is_not_incomplete")
    return {
        "source_report": prior,
        "history_reports": reports,
        "history_report_paths": paths,
        "history_stage_records": stage_records,
        "discarded_attempt_reports": discarded_attempt_reports,
        "last_return_code": int(next(
            row for row in stage_records
            if Path(str(row.get("executor_report", ""))).resolve() == paths[-1]
        ).get("engine_return_code", 2)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    continuation_history = None
    if getattr(args, "continue_from_v3_report", None) is not None:
        continuation_history = _load_v3_continuation_history(args, output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    reference: SuccessReference | None = None
    if args.reference_trace is not None:
        reference = load_success_reference(
            args.reference_trace,
            args.reference_report,
            expected_suite=args.benchmark,
            expected_level=args.level,
            expected_task_id=args.task_id,
            expected_initial_state_index=args.init_state,
        )
    additional_references = [
        load_success_reference(
            path,
            expected_suite=args.benchmark,
            expected_level=args.level,
            expected_task_id=args.task_id,
            expected_initial_state_index=args.init_state,
        )
        for path in args.additional_reference_trace
    ]
    backend = _select_backend(args, reference)
    experience = None if getattr(args, "cold_start", False) else _experience_match(args)
    try:
        strict_preflight = _strict_grasp_screen_gate(args)
        if strict_preflight is not None and backend != "experience_task_success":
            raise ValueError(
                "strict_grasp_screen_gate_is_supported_only_by_experience_task_success_backend"
            )
    except (OSError, TypeError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        manifest_path = output_dir / "run_manifest.json"
        report_path = output_dir / "v3_report.json"
        rejected = {
            "schema_version": "gs_opd_trajectory_generator_v3.run.v1",
            "status": "rejected_strict_grasp_screen",
            "started_unix_seconds": time.time(),
            "backend": backend,
            "task": {
                "benchmark": args.benchmark,
                "level": args.level,
                "task_id": args.task_id,
                "init_state": args.init_state,
                "seed": args.seed,
                "instruction": args.instruction,
                "native_target_predicate": args.target_predicate,
            },
            "inputs": {
                "strict_grasp_screen_report": str(args.strict_grasp_screen_report),
                "snapshot_root": str(args.snapshot_root) if args.snapshot_root else None,
                "resume_from_runtime_snapshot": (
                    str(args.resume_from_runtime_snapshot)
                    if args.resume_from_runtime_snapshot else None
                ),
            },
            "strict_preflight_validation": {
                "status": "rejected",
                "exception_type": type(exc).__name__,
                "failure_reason": str(exc),
            },
            "engine_launched": False,
            "bridge_started": False,
            "physical_actions": 0,
            "artifacts": {"manifest": str(manifest_path), "report": str(report_path)},
        }
        manifest_path.write_text(
            json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report_path.write_text(
            json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return rejected
    if strict_preflight is not None and strict_preflight["status"] != "passed":
        manifest_path = output_dir / "run_manifest.json"
        report_path = output_dir / "v3_report.json"
        blocked = {
            "schema_version": "gs_opd_trajectory_generator_v3.run.v1",
            "status": "blocked_no_strict_grasp_candidate",
            "started_unix_seconds": time.time(),
            "backend": backend,
            "task": {
                "benchmark": args.benchmark,
                "level": args.level,
                "task_id": args.task_id,
                "init_state": args.init_state,
                "seed": args.seed,
                "instruction": args.instruction,
                "native_target_predicate": args.target_predicate,
            },
            "inputs": {
                "experience_library": str(args.experience_library.resolve()) if args.experience_library else None,
                "experience_match": experience,
                "strict_grasp_screen_report": strict_preflight["screen_report"],
                "strict_grasp_screen_report_sha256": strict_preflight["screen_report_sha256"],
                "snapshot_root": str(args.snapshot_root.resolve()) if args.snapshot_root else None,
                "resume_from_runtime_snapshot": strict_preflight["start_snapshot"],
                "resume_from_runtime_snapshot_sha256": strict_preflight["start_snapshot_sha256"],
            },
            "strict_preflight": strict_preflight,
            "engine_launched": False,
            "bridge_started": False,
            "physical_actions": 0,
            "artifacts": {"manifest": str(manifest_path), "report": str(report_path)},
        }
        manifest_path.write_text(json.dumps(blocked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path.write_text(json.dumps(blocked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return blocked
    if strict_preflight is not None:
        args.grasp_candidate_id = strict_preflight["selected_candidate_id"]
    checkpoint_file = _checkpoint_file(args.checkpoint)
    checkpoint_hash = sha256_file(checkpoint_file)
    if backend == "student_bspline_osc":
        if reference is None or reference.route_prior is None:
            raise ValueError("student_bspline_backend_requires_a_verified_success_route_reference")
        source_checkpoint = Path(str(reference.metadata.get("checkpoint_file", "")))
        if not source_checkpoint.is_file() or sha256_file(source_checkpoint) != checkpoint_hash:
            raise ValueError("active_checkpoint_hash_does_not_match_success_route_reference")
        if args.lora_checkpoint is not None:
            raise ValueError("student_bspline_reference_backend_requires_its_verified_base_checkpoint_only")
        command = _student_bspline_command(
            args,
            output_dir=output_dir,
            checkpoint_file=checkpoint_file,
            route_direction=str(reference.route_prior["preferred_direction"]),
        )
    elif backend == "experience_task_success":
        command = _experience_task_success_command(
            args,
            output_dir=output_dir,
            checkpoint_file=checkpoint_file,
        )
    else:
        raise ValueError(f"unsupported_v3_backend:{backend}")

    manifest_path = output_dir / "run_manifest.json"
    report_path = output_dir / "v3_report.json"
    route_report_path = output_dir / "direct_bspline_osc_report.json"
    result_path = output_dir / "student_rollout_result.json"
    trace_path = output_dir / "student_rollout.npz"
    record = {
        "schema_version": "gs_opd_trajectory_generator_v3.run.v1",
        "status": "running",
        "started_unix_seconds": time.time(),
        "task": {
            "benchmark": args.benchmark,
            "level": args.level,
            "task_id": args.task_id,
            "init_state": args.init_state,
            "seed": args.seed,
            "instruction": args.instruction,
            "target_object_id": args.experience_object_id,
            "semantic_object_id": args.semantic_object_id,
            "native_target_predicate": args.target_predicate,
            "target_relation": args.target_relation,
            "target_region": args.target_region,
            "route_candidate_id": args.route_candidate_id,
            "pre_route_hold_actions": args.pre_route_hold_actions,
        },
        "backend": backend,
        "continuation_of_v3_report": (
            str(args.continue_from_v3_report.resolve())
            if getattr(args, "continue_from_v3_report", None) is not None else None
        ),
        "cold_start_protocol": (
            {
                "enabled": True,
                "no_target_demonstration": True,
                "no_prior_experience": True,
                "target_success_demonstration_used": False,
                "target_success_keyposes_used": False,
                "official_demonstration_used": False,
                "historical_actions_replayed": False,
                "initial_policy": "base_pi05",
                "initial_policy_checkpoint_provenance": (
                    "RLinf/RLinf-Pi05-LIBERO-SFT_revision_45ccfcc4e28634f1576ebf78cab0fbe2fd82432d"
                    if checkpoint_hash == "4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7"
                    else "unverified_by_checkpoint_hash"
                ),
                "unadapted_pretraining_checkpoint_verified": False,
                "experience_library_loaded": False,
                "success_reference_loaded": False,
                "previous_search_reports_loaded": False,
                "start_state": "identity_checked_legal_t0_snapshot",
            }
            if getattr(args, "cold_start", False) else {"enabled": False}
        ),
        "inputs": {
            "checkpoint_file": str(checkpoint_file),
            "checkpoint_sha256": checkpoint_hash,
            "teacher_json": str(args.teacher_json.resolve()) if args.teacher_json else None,
            "teacher_sha256": sha256_file(args.teacher_json) if args.teacher_json else None,
            "semantic_results_json": str(args.semantic_results_json.resolve()) if args.semantic_results_json else None,
            "semantic_results_sha256": sha256_file(args.semantic_results_json) if args.semantic_results_json else None,
            "experience_library": str(args.experience_library.resolve()) if args.experience_library else None,
            "experience_match": experience,
            "success_reference": reference.as_dict() if reference else None,
            "additional_success_references": [row.as_dict() for row in additional_references],
            "snapshot_root": str(args.snapshot_root.resolve()) if args.snapshot_root else None,
            "resume_from_runtime_snapshot": (
                str(args.resume_from_runtime_snapshot.resolve())
                if args.resume_from_runtime_snapshot else None
            ),
            "resume_from_runtime_snapshot_sha256": (
                sha256_file(args.resume_from_runtime_snapshot)
                if args.resume_from_runtime_snapshot else None
            ),
            "grasp_candidate_id": args.grasp_candidate_id,
            "strict_grasp_screen_report": (
                {
                    "path": strict_preflight["screen_report"],
                    "sha256": strict_preflight["screen_report_sha256"],
                }
                if strict_preflight is not None else None
            ),
            "previous_search_reports": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in (getattr(args, "previous_search_report", []) or [])
            ],
        },
        "pipeline_contract": {
            "live_observation_and_snapshot_owner": "selected_existing_physical_engine",
            "active_BDDL_and_geom_binding_owner": "Safety_Bridge_and_existing_task_executor",
            "route_retargeting": "current_state_and_current_scene_only",
            "historical_action_replay": False,
            "target_success_demonstration_used": False if getattr(args, "cold_start", False) else None,
            "target_success_keyposes_used": False if getattr(args, "cold_start", False) else None,
            "official_demonstration_used": False,
            "strict_safety_thresholds": {
                "forbidden_clearance_m": STRICT_FORBIDDEN_CLEARANCE_M,
                "authorized_finger_target_penetration_m": STRICT_FINGER_TARGET_PENETRATION_M,
            },
            "strict_grasp_screen_gate": (
                "passed_and_candidate_pinned_before_engine_launch"
                if strict_preflight is not None else "not_requested"
            ),
            "no_safety_threshold_or_BDDL_changes": True,
        },
        "strict_preflight": strict_preflight,
        "command": command,
        "bridge_command": (
            [
                str(LIBERO_PYTHON), str(BRIDGE_SCRIPT),
                "--benchmark", args.benchmark, "--level", str(args.level),
                "--task-id", str(args.task_id), "--init-state", str(args.init_state),
                "--seed", str(args.seed), "--camera-resolution", "256",
                "--port", str(args.bridge_port), "--demonstration-horizon", str(args.max_steps),
            ] if backend == "student_bspline_osc" else None
        ),
        "artifacts": {
            "manifest": str(manifest_path),
            "report": str(report_path),
            "execution_log": str(output_dir / "execution.log"),
            "trace": str(trace_path) if backend == "student_bspline_osc" else None,
            "runtime_snapshots": str(output_dir / "runtime_snapshots") if backend == "student_bspline_osc" else None,
        },
    }
    manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_path = output_dir / "execution.log"
    bridge_process = None
    bridge_log = None
    engine_launched = False
    wall_time_budget_s = float(getattr(args, "wall_time_budget_s", 3600.0))
    wall_time_started_monotonic = time.monotonic()
    record["engine_wall_time_budget_s"] = wall_time_budget_s
    if continuation_history is not None:
        suffix = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
        source_report_path = output_dir / "v3_report.json"
        source_manifest_path = output_dir / "run_manifest.json"
        (output_dir / f"v3_report_before_continuation_{suffix}.json").write_bytes(
            source_report_path.read_bytes()
        )
        if source_manifest_path.is_file():
            (output_dir / f"run_manifest_before_continuation_{suffix}.json").write_bytes(
                source_manifest_path.read_bytes()
            )
    manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if continuation_history is not None:
        completed = subprocess.CompletedProcess(
            args=command, returncode=continuation_history["last_return_code"], stdout=b"", stderr=b"",
        )
        record["same_episode_resume"] = {
            "status": "validated_existing_V3_archive",
            "executor_report_paths": [str(path) for path in continuation_history["history_report_paths"]],
            "previous_stage_records_preserved": True,
            "legal_t0_and_model_identity_verified": True,
        }
    else:
        try:
            if backend == "student_bspline_osc":
                bridge_process, bridge_log, bridge_preflight = _launch_bridge_and_preflight(
                    args,
                    output_dir=output_dir,
                )
                record["bridge_preflight"] = bridge_preflight
                manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with log_path.open("wb") as log:
                engine_launched = True
                completed = _run_bounded_engine(command, log=log, wall_time_budget_s=wall_time_budget_s)
        except Exception as exc:
            record.update({
                "status": "engine_time_budget_exhausted" if isinstance(exc, TimeoutError)
                else "executor_error" if engine_launched else "preflight_failed",
                "finished_unix_seconds": time.time(),
                "failure_reason": f"{type(exc).__name__}:{exc}",
                "physical_actions_executed": None if engine_launched else 0,
            })
            report_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return record
        finally:
            if bridge_process is not None:
                bridge_process.terminate()
                try:
                    bridge_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    bridge_process.kill()
                    bridge_process.wait(timeout=5)
            if bridge_log is not None:
                bridge_log.close()

    aggregate_inner_report: dict[str, Any] | None = None
    multi_goal_episode: dict[str, Any] | None = None
    if backend == "experience_task_success" and getattr(args, "continue_pending_native_goals", False):
        if continuation_history is not None:
            prior = continuation_history["source_report"]
            history_reports = continuation_history["history_reports"]
            history_report_paths = continuation_history["history_report_paths"]
            history_stage_records = continuation_history["history_stage_records"]
            first_report_path = history_report_paths[0]
            first_report = history_reports[0]
            try:
                aggregate_inner_report, final_return_code, multi_goal_episode = (
                    _run_cold_start_native_goal_continuations(
                        args,
                        output_dir=output_dir,
                        checkpoint_file=checkpoint_file,
                        first_report=first_report,
                        first_report_path=first_report_path,
                        first_return_code=continuation_history["last_return_code"],
                        wall_time_started=wall_time_started_monotonic,
                        wall_time_budget_s=wall_time_budget_s,
                        history_reports=history_reports,
                        history_report_paths=history_report_paths,
                        history_stage_records=history_stage_records,
                        history_discarded_attempt_reports=continuation_history.get(
                            "discarded_attempt_reports", []
                        ),
                    )
                )
                completed = subprocess.CompletedProcess(
                    args=command, returncode=final_return_code, stdout=b"", stderr=b"",
                )
            except Exception as exc:
                aggregate_inner_report = history_reports[-1]
                multi_goal_episode = {
                    "enabled": True,
                    "stage_count": len(history_reports),
                    "native_task_success": aggregate_inner_report.get("task_success") is True,
                    "same_run_snapshot_only": True,
                    "stop_reason": f"multi_goal_continuation_error:{type(exc).__name__}:{exc}",
                    "stages": history_stage_records,
                    "executor_report_paths": [str(path) for path in history_report_paths],
                }
                completed = subprocess.CompletedProcess(
                    args=command, returncode=2, stdout=b"", stderr=b"",
                )
        else:
            first_runs = sorted((output_dir / "task_success_engine").glob("task_success_first_executor_*/run_report.json"))
            if len(first_runs) == 1:
                try:
                    first_report_path = first_runs[0].resolve()
                    first_report = json.loads(first_report_path.read_text(encoding="utf-8"))
                    aggregate_inner_report, final_return_code, multi_goal_episode = (
                        _run_cold_start_native_goal_continuations(
                            args,
                            output_dir=output_dir,
                            checkpoint_file=checkpoint_file,
                            first_report=first_report,
                            first_report_path=first_report_path,
                            first_return_code=completed.returncode,
                            wall_time_started=wall_time_started_monotonic,
                            wall_time_budget_s=wall_time_budget_s,
                        )
                    )
                    completed = subprocess.CompletedProcess(
                        args=command, returncode=final_return_code, stdout=b"", stderr=b"",
                    )
                except Exception as exc:
                    aggregate_inner_report = json.loads(first_runs[0].read_text(encoding="utf-8"))
                    multi_goal_episode = {
                        "enabled": True,
                        "stage_count": 1,
                        "native_task_success": aggregate_inner_report.get("task_success") is True,
                        "same_run_snapshot_only": True,
                        "stop_reason": f"multi_goal_continuation_error:{type(exc).__name__}:{exc}",
                        "executor_report_paths": [str(first_runs[0].resolve())],
                    }
                    completed = subprocess.CompletedProcess(
                        args=command, returncode=2, stdout=b"", stderr=b"",
                    )
            else:
                multi_goal_episode = {
                    "enabled": True,
                    "stage_count": 0,
                    "native_task_success": False,
                    "same_run_snapshot_only": True,
                    "stop_reason": "initial_executor_report_missing_or_ambiguous",
                    "executor_report_paths": [str(path.resolve()) for path in first_runs],
                }
                completed = subprocess.CompletedProcess(
                    args=command, returncode=2, stdout=b"", stderr=b"",
                )

    if backend == "student_bspline_osc":
        inner_result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        inner_route = json.loads(route_report_path.read_text(encoding="utf-8")) if route_report_path.is_file() else {}
        outcome = _outcome(
            result_path=result_path,
            trace_path=trace_path,
            route_report_path=route_report_path,
            strict_report_path=args.strict_safety_report,
        )
        scene = inner_result.get("scene_metadata", {})
        scene_matches = (
            scene.get("benchmark") == args.benchmark
            and scene.get("level") == args.level
            and scene.get("task_id") == args.task_id
            and scene.get("init_state_index") == args.init_state
            and scene.get("environment_seed") == args.seed
        )
        status = _classify_run_status(
            engine_return_code=completed.returncode,
            scene_matches=scene_matches,
            native_task_success=outcome["native_task_success"],
        )
        if not scene_matches and inner_result:
            status = "scene_identity_mismatch_after_engine_return"
        record.update({
            "status": status,
            "finished_unix_seconds": time.time(),
            "engine_return_code": completed.returncode,
            "engine_result": str(result_path) if result_path.is_file() else None,
            "route_report": str(route_report_path) if route_report_path.is_file() else None,
            "route_summary": {
                "status": inner_route.get("status"),
                "route_completed": inner_route.get("route_completed") is True,
                "actual_osc_steps": inner_route.get("real_execution_steps"),
                "pre_route_hold_actions": inner_route.get("pre_route_hold", {}).get("executed_actions"),
                "route_osc_steps": inner_route.get("execution_summary", {}).get("route_osc_steps"),
                "failure_reason": inner_route.get("failure_reason"),
                "selected_candidate_id": inner_route.get("selected_candidate_id"),
                "selected_route_direction": inner_route.get("selected_route_direction"),
                "reference_route_retargeted": True,
                "historical_actions_replayed": False,
            },
            "outcome": outcome,
            "scene_identity_verified": scene_matches,
            "failure_reason": (
                None if status == "complete" else inner_route.get("failure_reason")
                or inner_result.get("failure_reason") or f"engine_exit_{completed.returncode}"
            ),
        })
        if outcome["native_task_success"]:
            candidate = {
                "schema_version": "gs_opd_experience_candidate.v1",
                "source_trace": str(trace_path),
                "source_trace_sha256": sha256_file(trace_path),
                "native_task_success": True,
                "strict_10mm_pass": outcome["strict_10mm_pass"],
                "safe_training_eligible": outcome["safe_training_eligible"],
                "qualification": "task_feasible_only",
                "registered_into_shared_library": False,
            }
            candidate_path = output_dir / "experience_update_candidate.json"
            candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            record["experience_update_candidate"] = str(candidate_path)
    else:
        inner_runs = sorted((output_dir / "task_success_engine").glob("task_success_first_executor_*/run_report.json"))
        inner_report = aggregate_inner_report or (
            json.loads(inner_runs[-1].read_text(encoding="utf-8")) if inner_runs else {}
        )
        native_success = inner_report.get("task_success") is True
        physical_audit = _experience_physical_action_audit(inner_report) if inner_report else {
            "status": "not_available",
            "complete_substep_monitoring": None,
            "state_action_rows_valid": None,
            "physical_actions_archived": 0,
        }
        scope = inner_report.get("scope", {})
        scene_matches = (
            scope.get("benchmark") == args.benchmark
            and scope.get("level") == args.level
            and scope.get("task_id") == args.task_id
            and scope.get("init_state") == args.init_state
            and scope.get("seed") == args.seed
        )
        status = _classify_experience_run_status(
            engine_return_code=completed.returncode,
            report_available=bool(inner_report),
            scene_matches=scene_matches,
            native_task_success=native_success,
        )
        final_native = inner_report.get("final_native_task_status", {})
        success_timestep = final_native.get("timestep") if isinstance(final_native, dict) else None
        standard_1000_success = (
            bool(native_success and isinstance(success_timestep, int) and success_timestep <= 1000)
            if isinstance(success_timestep, int)
            else native_success if args.demonstration_horizon == 1000 else None
        )
        demonstration_only_success = (
            bool(native_success and isinstance(success_timestep, int) and success_timestep > 1000)
            if args.demonstration_horizon > 1000 and isinstance(success_timestep, int)
            else None
        )
        record.update({
            "status": status,
            "finished_unix_seconds": time.time(),
            "engine_return_code": completed.returncode,
            "engine_report": str(multi_goal_episode["executor_report_paths"][-1])
            if multi_goal_episode and multi_goal_episode.get("executor_report_paths") else
            str(inner_runs[-1]) if inner_runs else None,
            "scene_identity_verified": scene_matches,
            "physical_action_audit": physical_audit,
            "outcome": {
                "native_task_success": native_success,
                "native_task_success_timestep": success_timestep,
                "standard_1000_step_success": standard_1000_success,
                "demonstration_only_success": demonstration_only_success,
                "stable_release_verified": "not_evaluated",
                "operational_safety_pass": "not_certified",
                "strict_10mm_pass": False,
                "strict_10mm_assessment": "not_evaluated_without_same_episode_strict_report",
                "complete_substep_monitoring": physical_audit.get("success_episode_complete_substep_monitoring", False),
                "physical_state_action_records_valid": bool(
                    physical_audit.get("state_action_rows_valid")
                    and physical_audit.get("success_episode_timestep_continuous")
                ),
                "observation_action_trace_valid": "not_emitted_by_this_backend",
                "safe_training_eligible": False,
                "source_safety_inherited": False,
            },
            "failure_reasons": _experience_failure_reasons(inner_report),
            "failure_reason": None if native_success else (
                (_experience_failure_reasons(inner_report) or [None])[-1]
                or (multi_goal_episode or {}).get("stop_reason")
                or inner_report.get("error")
                or ("native_task_success_not_reached" if inner_report else "executor_report_missing")
            ),
        })
        if multi_goal_episode is not None:
            record["multi_goal_episode"] = multi_goal_episode
        if getattr(args, "cold_start", False) and native_success and not getattr(args, "continue_pending_native_goals", False):
            initial_snapshot = args.snapshot_root / "snapshots/initial_runtime_snapshot.pkl"
            record["safety_risk_localization"] = _localize_cold_start_risks(
                inner_report, physical_audit, output_dir, initial_snapshot=initial_snapshot,
            )
            report_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if record["safety_risk_localization"].get("first_observed_risk"):
                record["safety_reconstruction"] = _screen_current_episode_reconstruction(args, physical_audit, output_dir)

    report_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "student_bspline_osc", "experience_task_success"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="obstacle_avoidance")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--init-state", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--experience-object-id", default=None)
    parser.add_argument("--semantic-object-id", default=None)
    parser.add_argument("--target-relation", default=None)
    parser.add_argument("--target-region", default=None)
    parser.add_argument("--route-candidate-id", default=None)
    parser.add_argument("--pre-route-hold-actions", type=int, default=0)
    parser.add_argument("--target-predicate", nargs="+", default=None)
    parser.add_argument("--reference-trace", type=Path, default=None)
    parser.add_argument("--reference-report", type=Path, default=None)
    parser.add_argument("--additional-reference-trace", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lora-checkpoint", type=Path, default=None)
    parser.add_argument("--experience-library", type=Path, default=None)
    parser.add_argument("--keypose-experience", type=Path, action="append", default=[])
    parser.add_argument("--previous-search-report", type=Path, action="append", default=[])
    parser.add_argument(
        "--cold-start", action="store_true",
        help="forbid successful-task experience, keyposes, references, resume states, and prior search reports",
    )
    parser.add_argument(
        "--continue-pending-native-goals", action="store_true",
        help="within this cold-start run, continue settled partial goals from same-run exact Runtime Snapshots",
    )
    parser.add_argument(
        "--continue-from-v3-report", type=Path, default=None,
        help="continue an incomplete cold-start V3 archive from its own exact saved stage history",
    )
    parser.add_argument("--snapshot-root", type=Path, default=None)
    parser.add_argument(
        "--resume-from-runtime-snapshot", type=Path, default=None,
        help="resume the experience backend from an identity-checked partial-task MuJoCo Snapshot",
    )
    parser.add_argument(
        "--grasp-candidate-id", default=None,
        help="restrict resumed experience search to one generated grasp candidate ID",
    )
    parser.add_argument(
        "--strict-grasp-screen-report", type=Path, default=None,
        help="require a matching no-action Strict-10mm screen and pin physical search to a passing candidate",
    )
    parser.add_argument("--teacher-json", type=Path, default=None)
    parser.add_argument("--semantic-results-json", type=Path, default=None)
    parser.add_argument("--strict-safety-report", type=Path, default=None)
    parser.add_argument("--rlinf-python", type=Path, default=RLINF_PYTHON)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--demonstration-horizon", type=int, default=1800)
    parser.add_argument("--maximum-physical-branches", type=int, default=2)
    parser.add_argument("--maximum-rrt-preflights", type=int, default=3)
    parser.add_argument("--rrt-time-budget-s", type=float, default=12.0)
    parser.add_argument("--wall-time-budget-s", type=float, default=3600.0,
                        help="wall-clock limit for the physical engine and its subprocesses")
    parser.add_argument(
        "--candidate-start-index", type=int, default=0,
        help="start experience grasp-search preflights at this generated candidate index",
    )
    parser.add_argument(
        "--geometry-candidate-pool-offset", type=int, default=0,
        help="select a later window from deterministic active-scene grasp geometry candidates",
    )
    parser.add_argument("--bridge-port", type=int, default=38567)
    parser.add_argument("--transport-strategy", choices=("native_region_progress_osc", "legacy_rrt"), default="native_region_progress_osc")
    parser.add_argument(
        "--native-region-route-variant",
        choices=(
            "direct_region_progress",
            "keypose_side_region_progress",
            "keypose_side_region_progress_reverse",
        ),
        default=None,
        help="select a region-progress route variant for native-region OSC transport",
    )
    args = parser.parse_args()
    if not 1.0 <= args.wall_time_budget_s <= 86400.0:
        parser.error("--wall-time-budget-s must be in 1..86400 seconds")
    if args.cold_start:
        forbidden_inputs = (
            args.experience_library,
            args.keypose_experience,
            args.previous_search_report,
            args.reference_trace,
            args.reference_report,
            args.additional_reference_trace,
            args.lora_checkpoint,
            args.resume_from_runtime_snapshot,
            args.grasp_candidate_id,
            args.strict_grasp_screen_report,
            args.route_candidate_id,
        )
        if any(value for value in forbidden_inputs):
            parser.error("--cold-start cannot be combined with historical experience, success references, or partial-task inputs")
        if args.backend == "student_bspline_osc":
            parser.error("--cold-start requires task-success search, not a historical route backend")
        if args.snapshot_root is None:
            parser.error("--cold-start requires --snapshot-root for the legal task t0 state")
        if args.candidate_start_index != 0:
            parser.error("--cold-start must begin at generated candidate index 0")
    if args.continue_pending_native_goals and (
        not args.cold_start
        or args.backend not in {"auto", "experience_task_success"}
        or args.target_predicate is not None
        or args.resume_from_runtime_snapshot is not None
    ):
        parser.error("--continue-pending-native-goals requires an unpinned cold-start task-success run")
    if args.continue_from_v3_report is not None:
        if (
            not args.cold_start
            or not args.continue_pending_native_goals
            or args.backend not in {"auto", "experience_task_success"}
            or args.continue_from_v3_report.resolve() != (args.output_dir.resolve() / "v3_report.json").resolve()
        ):
            parser.error(
                "--continue-from-v3-report requires an existing archive, cold-start multigoal mode, "
                "and that archive's own v3_report.json"
            )
    if not args.cold_start and args.experience_library is None:
        args.experience_library = DEFAULT_EXPERIENCE_LIBRARY
    if args.max_steps < 1 or args.demonstration_horizon < 1000:
        parser.error("step budgets must be positive and horizon must be at least 1000")
    if not 0 <= args.pre_route_hold_actions <= 100:
        parser.error("--pre-route-hold-actions must be in [0, 100]")
    if not 1 <= args.maximum_physical_branches <= 3:
        parser.error("--maximum-physical-branches must be in 1..3")
    if not 1 <= args.maximum_rrt_preflights <= 8:
        parser.error("--maximum-rrt-preflights must be in 1..8")
    if args.native_region_route_variant is not None and args.transport_strategy != "native_region_progress_osc":
        parser.error("--native-region-route-variant requires --transport-strategy native_region_progress_osc")
    if args.candidate_start_index < 0:
        parser.error("--candidate-start-index must be nonnegative")
    if not 0 <= args.geometry_candidate_pool_offset <= 32:
        parser.error("--geometry-candidate-pool-offset must fit the 48-candidate active geometry pool")
    if args.grasp_candidate_id is not None and args.candidate_start_index != 0:
        parser.error("--grasp-candidate-id requires --candidate-start-index 0")
    if not 0.5 <= args.rrt_time_budget_s <= 60:
        parser.error("--rrt-time-budget-s must be in 0.5..60")
    if not args.rlinf_python.is_absolute() or not args.rlinf_python.is_file():
        parser.error("--rlinf-python must be an existing absolute interpreter path")
    if not 1024 <= args.bridge_port <= 65535:
        parser.error("--bridge-port must be in 1024..65535")
    if args.backend == "student_bspline_osc" and args.reference_trace is None:
        parser.error("student_bspline_osc requires --reference-trace")
    if args.backend == "student_bspline_osc" and (
        not args.target_predicate or len(args.target_predicate) != 3
    ):
        parser.error("student_bspline_osc requires a three-token --target-predicate")
    if args.reference_report is not None and args.reference_trace is None:
        parser.error("--reference-report requires --reference-trace")
    return args


def main() -> int:
    result = run(_parse_args())
    print(json.dumps({
        "status": result.get("status"),
        "backend": result.get("backend"),
        "archive": result.get("artifacts", {}).get("report"),
        "native_task_success": result.get("outcome", {}).get("native_task_success"),
        "strict_10mm_pass": result.get("outcome", {}).get("strict_10mm_pass"),
        "safe_training_eligible": result.get("outcome", {}).get("safe_training_eligible"),
        "failure_reason": result.get("failure_reason"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
