# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression tests for V3 reference provenance and outcome gates."""

import json
import hashlib
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from research.gs_opd.repair.l0t0_success_reference import load_success_reference
from research.gs_opd.repair.trajectory_generator_v3 import (
    _classify_run_status,
    _classify_experience_run_status,
    _experience_task_success_command,
    _parse_args as _parse_v3_args,
    _select_backend,
    _experience_physical_action_audit,
    _outcome,
    _nested_contact_monitoring_complete,
    _run_bounded_engine,
    _substep_monitoring,
    _student_bspline_command,
    _strict_grasp_screen_gate,
    run as run_v3,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DIRECT_REFERENCE = (
    REPO_ROOT
    / "logs/gs_opd/mink_panda_physical_v3/"
    "live_left_bspline_osc_adaptive_20261019_attempt1/"
    "direct_bspline_osc_trajectory.npz"
)
DIRECT_REPORT = DIRECT_REFERENCE.with_name("live_left_bspline_osc_report.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_engine_wall_timeout_preserves_output_and_stops_process(tmp_path: Path) -> None:
    log_path = tmp_path / "engine.log"
    with log_path.open("wb") as log:
        with pytest.raises(TimeoutError, match="engine_wall_time_budget_exhausted"):
            _run_bounded_engine(
                [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(10)"],
                log=log, wall_time_budget_s=0.5,
            )
    assert log_path.read_text().strip() == "started"
    with log_path.open("wb") as log:
        result = _run_bounded_engine(
            [sys.executable, "-c", "raise SystemExit(7)"], log=log, wall_time_budget_s=5,
        )
    assert result.returncode == 7


def _write_trace(path: Path, *, monitor_status: str) -> None:
    monitoring = [
        json.dumps({
            "status": monitor_status,
            "captured_physics_substeps": 25 if monitor_status == "complete" else 24,
            "control_steps": [{
                "status": monitor_status,
                "expected_physics_substeps": 25,
                "captured_physics_substeps": 25 if monitor_status == "complete" else 24,
            }],
        })
        for _ in range(2)
    ]
    np.savez_compressed(
        path,
        executed_actions=np.zeros((2, 7), dtype=np.float32),
        step_rewards=np.asarray([0.0, 1.0], dtype=np.float32),
        step_dones=np.asarray([False, True]),
        post_contact_monitoring_json=np.asarray(monitoring),
    )


def _strict_screen_fixture(tmp_path: Path, *, feasible: bool) -> tuple[Path, Path]:
    snapshot_root = tmp_path / "snapshot_root"
    (snapshot_root / "snapshots").mkdir(parents=True)
    snapshot = snapshot_root / "snapshots/initial_runtime_snapshot.pkl"
    model_sha = "a" * 64
    bddl_sha = "b" * 64
    instruction = "put both moka pots on the stove"
    integration_state = np.asarray([0.0], dtype=np.float64)
    snapshot.write_bytes(pickle.dumps({
        "schema_version": "gs_opd_runtime_snapshot.v1",
        "scene_metadata": {
            "benchmark": "obstacle_avoidance", "level": 0, "task_id": 2,
            "init_state_index": 0, "environment_seed": 0,
            "bddl_sha256": bddl_sha, "instruction": instruction,
        },
        "model_geometry_sha256": model_sha,
        "snapshot": {"mujoco_state": integration_state.tolist()},
    }))
    manifest = {
        "instruction": instruction,
        "scene_metadata": {
            "benchmark": "obstacle_avoidance", "level": 0, "task_id": 2,
            "init_state_index": 0, "environment_seed": 0,
            "model_geometry_sha256": model_sha, "bddl_sha256": bddl_sha,
            "instruction": instruction,
        },
        "snapshots": {"initial": {
            "payload_sha256": _sha(snapshot),
            "model_geometry_sha256": model_sha,
            "mujoco_integration_state_sha256": hashlib.sha256(
                np.ascontiguousarray(integration_state, dtype=np.float64).tobytes()
            ).hexdigest(),
        }},
    }
    (snapshot_root / "snapshots/runtime_snapshot_manifest.json").write_text(json.dumps(manifest))
    candidate_id = "experience__moka_pot:stable_grasp:t248"
    rows = [{"candidate_id": candidate_id, "status": "offline_feasible_candidate"}] if feasible else []
    screen = {
        "schema_version": "gs_opd_strict_pad_only_grasp_screen.v1",
        "status": "complete",
        "strict_safety_candidate_only": True,
        "physical_actions": 0,
        "live_integration_state_unchanged": True,
        "safety_certified": False,
        "route_preflight_performed": False,
        "task": {
            "benchmark": "obstacle_avoidance", "level": 0, "task_id": 2,
            "init_state": 0, "environment_seed": 0,
            "model_geometry_sha256": model_sha, "bddl_sha256": bddl_sha,
        },
        "native_goal_predicate": ["on", "moka_pot_2", "flat_stove_1_cook_region"],
        "start_snapshot_payload_sha256": _sha(snapshot),
        "screening": {
            "status": "complete",
            "screening_type": "offline_scratch_only_not_physical_execution",
            "target_contact_policy": {
                "mode": "pad_only_with_nonpad_clearance",
                "strict_pad_only": True,
                "minimum_nonpad_robot_to_target_clearance_m": 0.010,
                "maximum_authorized_pad_penetration_m": -0.002,
            },
            "candidate_count": len(rows),
            "feasible_count": len(rows),
            "ranked_candidate_ids": [candidate_id] if feasible else [],
            "candidates": rows,
        },
    }
    report_path = tmp_path / "strict_screen.json"
    report_path.write_text(json.dumps(screen))
    return report_path, snapshot_root


def _screen_gate_args(report: Path, snapshot_root: Path, *, output_dir: Path | None = None):
    from argparse import Namespace

    return Namespace(
        strict_grasp_screen_report=report,
        snapshot_root=snapshot_root,
        resume_from_runtime_snapshot=None,
        benchmark="obstacle_avoidance", level=0, task_id=2, init_state=0, seed=0,
        target_predicate=["on", "moka_pot_2", "flat_stove_1_cook_region"],
        grasp_candidate_id=None,
        output_dir=output_dir or Path("unused"),
        instruction="put both moka pots on the stove",
        experience_library=None,
    )


def test_success_reference_uses_route_structure_without_action_replay() -> None:
    reference = load_success_reference(DIRECT_REFERENCE, DIRECT_REPORT)
    assert reference.native_task_success is True
    assert reference.final_done is True
    assert reference.final_reward == 1.0
    assert reference.route_prior is not None
    assert reference.route_prior["preferred_direction"] == "left"
    assert reference.route_prior["historical_route_steps"] == 749
    assert reference.route_prior["historical_actions_replayed"] is False
    serialized = reference.as_dict()
    assert serialized["strict_10mm_pass"] is False
    assert serialized["safe_training_eligible"] is False
    assert serialized["stable_release_verified"] is False


def test_reference_rejects_a_different_initial_state() -> None:
    with pytest.raises(ValueError, match="initial_state_index_mismatch"):
        load_success_reference(DIRECT_REFERENCE, DIRECT_REPORT, expected_initial_state_index=1)


def test_strict_screen_gate_pins_a_ranked_feasible_candidate(tmp_path: Path) -> None:
    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    gate = _strict_grasp_screen_gate(_screen_gate_args(report, snapshot_root))
    assert gate is not None
    assert gate["status"] == "passed"
    assert gate["selected_candidate_id"] == "experience__moka_pot:stable_grasp:t248"
    assert gate["physical_actions_during_screen"] == 0


def test_strict_screen_selected_candidate_is_forwarded_to_executor(tmp_path: Path) -> None:
    from argparse import Namespace

    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    gate = _strict_grasp_screen_gate(_screen_gate_args(report, snapshot_root))
    assert gate is not None and gate["status"] == "passed"
    experience_library = tmp_path / "experience.json"
    experience_library.write_text("{}")
    args = Namespace(
        rlinf_python=Path("/workspace/zy_workspace/RLinf/.venv/bin/python"),
        snapshot_root=snapshot_root,
        experience_library=experience_library,
        benchmark="obstacle_avoidance", level=0, task_id=2, init_state=0, seed=0,
        demonstration_horizon=1000, maximum_physical_branches=1,
        maximum_rrt_preflights=1, rrt_time_budget_s=12.0,
        candidate_start_index=0, transport_strategy="native_region_progress_osc",
        target_predicate=["on", "moka_pot_2", "flat_stove_1_cook_region"],
        keypose_experience=[], previous_search_report=[],
        resume_from_runtime_snapshot=None,
        grasp_candidate_id=gate["selected_candidate_id"],
    )
    argv = _experience_task_success_command(
        args,
        output_dir=tmp_path / "pinned_candidate",
        checkpoint_file=REPO_ROOT / "logs/gs_opd/mink_panda_physical_v3/live_left_bspline_osc_adaptive_20261019_attempt1/direct_bspline_osc_trajectory.npz",
    )
    assert argv[argv.index("--candidate-id") + 1] == "experience__moka_pot:stable_grasp:t248"


def test_cold_start_command_omits_success_experience_inputs(tmp_path: Path) -> None:
    from argparse import Namespace

    snapshot_root = tmp_path / "snapshot_root"
    snapshot_root.mkdir()
    args = Namespace(
        rlinf_python=Path("/workspace/zy_workspace/RLinf/.venv/bin/python"),
        snapshot_root=snapshot_root,
        experience_library=None,
        cold_start=True,
        benchmark="obstacle_avoidance", level=0, task_id=0, init_state=0, seed=0,
        demonstration_horizon=1000, maximum_physical_branches=1,
        maximum_rrt_preflights=1, rrt_time_budget_s=12.0,
        candidate_start_index=0, transport_strategy="native_region_progress_osc",
        target_predicate=None, keypose_experience=[], previous_search_report=[],
        resume_from_runtime_snapshot=None, grasp_candidate_id=None,
        strict_grasp_screen_report=None,
        bridge_port=39420,
    )
    argv = _experience_task_success_command(
        args,
        output_dir=tmp_path / "cold_start",
        checkpoint_file=tmp_path / "model.safetensors",
    )
    assert "--cold-start" in argv
    assert "--experience-library" not in argv
    assert "--keypose-experience" not in argv
    assert "--previous-search-report" not in argv
    assert "--candidate-id" not in argv
    assert argv[argv.index("--port") + 1] == "39420"
    assert argv[argv.index("--expert-port") + 1] == "39421"


def test_cold_start_backend_selects_task_search_without_experience() -> None:
    from argparse import Namespace

    args = Namespace(
        backend="auto", cold_start=True, snapshot_root=Path("/legal/t0"),
    )
    assert _select_backend(args, reference=None) == "experience_task_success"


def test_cold_start_cli_rejects_success_experience_and_keeps_library_unset(tmp_path: Path) -> None:
    from unittest.mock import patch

    with patch("sys.argv", [
        "trajectory_generator_v3.py", "--output-dir", str(tmp_path / "out"),
        "--level", "0", "--task-id", "0", "--init-state", "0", "--seed", "0",
        "--instruction", "pick up the bowl and place it on the plate",
        "--snapshot-root", str(tmp_path / "snapshot_root"), "--cold-start",
    ]):
        args = _parse_v3_args()
    assert args.cold_start is True
    assert args.experience_library is None
    assert args.keypose_experience == []
    assert args.previous_search_report == []
    assert args.reference_trace is None

    with patch("sys.argv", [
        "trajectory_generator_v3.py", "--output-dir", str(tmp_path / "out"),
        "--level", "0", "--task-id", "0", "--init-state", "0", "--seed", "0",
        "--instruction", "pick up the bowl and place it on the plate",
        "--snapshot-root", str(tmp_path / "snapshot_root"), "--cold-start",
        "--experience-library", str(tmp_path / "experience.json"),
    ]), pytest.raises(SystemExit):
        _parse_v3_args()


def test_strict_screen_gate_blocks_when_no_candidate_passes(tmp_path: Path) -> None:
    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=False)
    gate = _strict_grasp_screen_gate(_screen_gate_args(report, snapshot_root))
    assert gate is not None
    assert gate["status"] == "blocked_no_strict_grasp_candidate"
    assert gate["feasible_count"] == 0
    assert gate["selected_candidate_id"] is None


def test_strict_screen_gate_rejects_a_different_start_snapshot(tmp_path: Path) -> None:
    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    snapshot = snapshot_root / "snapshots/initial_runtime_snapshot.pkl"
    snapshot.write_bytes(b"different physical state")
    with pytest.raises(ValueError, match="start_snapshot_hash_mismatch"):
        _strict_grasp_screen_gate(_screen_gate_args(report, snapshot_root))


def test_strict_screen_gate_checks_initial_snapshot_manifest_payload_hash(tmp_path: Path) -> None:
    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    manifest_path = snapshot_root / "snapshots/runtime_snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["snapshots"]["initial"]["payload_sha256"] = "c" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="snapshot_manifest_payload_sha256_mismatch"):
        _strict_grasp_screen_gate(_screen_gate_args(report, snapshot_root))


def test_strict_screen_gate_accepts_identity_matched_resume_snapshot(tmp_path: Path) -> None:
    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=False)
    resume_snapshot = tmp_path / "stage_runtime_snapshot.pkl"
    resume_snapshot.write_bytes(pickle.dumps({
        "schema_version": "gs_opd_runtime_snapshot.v1",
        "scene_metadata": {
            "benchmark": "obstacle_avoidance", "level": 0, "task_id": 2,
            "init_state_index": 0, "environment_seed": 0,
            "bddl_sha256": "b" * 64, "instruction": "put both moka pots on the stove",
        },
        "model_geometry_sha256": "a" * 64,
        "snapshot": {"mujoco_state": [1.25, -2.5]},
    }))
    screen = json.loads(report.read_text())
    screen["start_snapshot_payload_sha256"] = _sha(resume_snapshot)
    report.write_text(json.dumps(screen))
    args = _screen_gate_args(report, snapshot_root)
    args.resume_from_runtime_snapshot = resume_snapshot

    gate = _strict_grasp_screen_gate(args)
    assert gate is not None
    assert gate["status"] == "blocked_no_strict_grasp_candidate"
    assert gate["start_snapshot"] == str(resume_snapshot.resolve())
    assert gate["start_snapshot_sha256"] == _sha(resume_snapshot)
    assert gate["start_snapshot_integration_state_sha256"] == hashlib.sha256(
        np.ascontiguousarray([1.25, -2.5], dtype=np.float64).tobytes()
    ).hexdigest()


def test_v3_archives_resume_snapshot_identity_mismatch_before_engine_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.gs_opd.repair.trajectory_generator_v3 as v3

    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    resume_snapshot = tmp_path / "wrong_task_runtime_snapshot.pkl"
    resume_snapshot.write_bytes(pickle.dumps({
        "schema_version": "gs_opd_runtime_snapshot.v1",
        "scene_metadata": {
            "benchmark": "obstacle_avoidance", "level": 0, "task_id": 1,
            "init_state_index": 0, "environment_seed": 0,
            "bddl_sha256": "b" * 64, "instruction": "put both moka pots on the stove",
        },
        "model_geometry_sha256": "a" * 64,
        "snapshot": {"mujoco_state": [0.0]},
    }))
    screen = json.loads(report.read_text())
    screen["start_snapshot_payload_sha256"] = _sha(resume_snapshot)
    report.write_text(json.dumps(screen))
    args = _screen_gate_args(report, snapshot_root, output_dir=tmp_path / "wrong_resume_v3")
    args.resume_from_runtime_snapshot = resume_snapshot
    args.reference_trace = None
    args.reference_report = None
    args.additional_reference_trace = []
    monkeypatch.setattr(v3, "_select_backend", lambda _args, _reference: "experience_task_success")
    monkeypatch.setattr(v3, "_experience_match", lambda _args: None)

    result = run_v3(args)
    assert result["status"] == "rejected_strict_grasp_screen"
    assert "resume_Runtime_Snapshot_identity_mismatch" in result[
        "strict_preflight_validation"
    ]["failure_reason"]
    assert result["engine_launched"] is False
    assert result["bridge_started"] is False
    assert result["physical_actions"] == 0
    assert (args.output_dir / "run_manifest.json").is_file()
    assert (args.output_dir / "v3_report.json").is_file()
    assert not (args.output_dir / "execution.log").exists()


def test_strict_screen_gate_checks_snapshot_task_seed_and_initial_state(tmp_path: Path) -> None:
    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    manifest_path = snapshot_root / "snapshots/runtime_snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scene_metadata"]["task_id"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="snapshot_identity_mismatch"):
        _strict_grasp_screen_gate(_screen_gate_args(report, snapshot_root))


def test_strict_screen_gate_rejects_route_preflight_report(tmp_path: Path) -> None:
    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    screen = json.loads(report.read_text())
    screen["route_preflight_performed"] = True
    report.write_text(json.dumps(screen))
    with pytest.raises(ValueError, match="not_a_complete_no_action_screen"):
        _strict_grasp_screen_gate(_screen_gate_args(report, snapshot_root))


def test_strict_screen_gate_rejects_inconsistent_candidate_counts(tmp_path: Path) -> None:
    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    screen = json.loads(report.read_text())
    screen["screening"]["candidate_count"] = 2
    report.write_text(json.dumps(screen))
    with pytest.raises(ValueError, match="candidate_counts_malformed"):
        _strict_grasp_screen_gate(_screen_gate_args(report, snapshot_root))


def test_strict_screen_gate_rejects_non_scratch_screen(tmp_path: Path) -> None:
    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    screen = json.loads(report.read_text())
    screen["screening"]["screening_type"] = "physical_execution"
    report.write_text(json.dumps(screen))
    with pytest.raises(ValueError, match="complete_scratch_screen"):
        _strict_grasp_screen_gate(_screen_gate_args(report, snapshot_root))


def test_v3_does_not_launch_engine_when_strict_screen_has_no_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.gs_opd.repair.trajectory_generator_v3 as v3

    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=False)
    args = _screen_gate_args(report, snapshot_root, output_dir=tmp_path / "blocked_v3")
    args.reference_trace = None
    args.reference_report = None
    args.additional_reference_trace = []
    monkeypatch.setattr(v3, "_select_backend", lambda _args, _reference: "experience_task_success")
    monkeypatch.setattr(v3, "_experience_match", lambda _args: None)

    result = run_v3(args)
    assert result["status"] == "blocked_no_strict_grasp_candidate"
    assert result["engine_launched"] is False
    assert result["bridge_started"] is False
    assert result["physical_actions"] == 0
    assert (args.output_dir / "v3_report.json").is_file()
    assert not (args.output_dir / "execution.log").exists()


def test_v3_archives_rejected_screen_identity_before_engine_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.gs_opd.repair.trajectory_generator_v3 as v3

    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    (snapshot_root / "snapshots/initial_runtime_snapshot.pkl").write_bytes(b"mutated state")
    args = _screen_gate_args(report, snapshot_root, output_dir=tmp_path / "rejected_v3")
    args.reference_trace = None
    args.reference_report = None
    args.additional_reference_trace = []
    monkeypatch.setattr(v3, "_select_backend", lambda _args, _reference: "experience_task_success")
    monkeypatch.setattr(v3, "_experience_match", lambda _args: None)

    result = run_v3(args)
    assert result["status"] == "rejected_strict_grasp_screen"
    assert result["strict_preflight_validation"]["failure_reason"] == (
        "strict_grasp_screen_start_snapshot_hash_mismatch"
    )
    assert result["engine_launched"] is False
    assert result["bridge_started"] is False
    assert result["physical_actions"] == 0
    assert (args.output_dir / "run_manifest.json").is_file()
    assert (args.output_dir / "v3_report.json").is_file()
    assert not (args.output_dir / "execution.log").exists()


def test_v3_archives_malformed_screen_root_before_engine_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.gs_opd.repair.trajectory_generator_v3 as v3

    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    report.write_text("[]")
    args = _screen_gate_args(report, snapshot_root, output_dir=tmp_path / "malformed_v3")
    args.reference_trace = None
    args.reference_report = None
    args.additional_reference_trace = []
    monkeypatch.setattr(v3, "_select_backend", lambda _args, _reference: "experience_task_success")
    monkeypatch.setattr(v3, "_experience_match", lambda _args: None)

    result = run_v3(args)
    assert result["status"] == "rejected_strict_grasp_screen"
    assert result["strict_preflight_validation"]["failure_reason"] == (
        "strict_grasp_screen_report_root_must_be_an_object"
    )
    assert result["engine_launched"] is False
    assert result["bridge_started"] is False
    assert result["physical_actions"] == 0
    assert (args.output_dir / "run_manifest.json").is_file()
    assert (args.output_dir / "v3_report.json").is_file()


def test_v3_archives_malformed_screen_threshold_type_before_engine_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.gs_opd.repair.trajectory_generator_v3 as v3

    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    screen = json.loads(report.read_text())
    screen["screening"]["target_contact_policy"][
        "minimum_nonpad_robot_to_target_clearance_m"
    ] = None
    report.write_text(json.dumps(screen))
    args = _screen_gate_args(report, snapshot_root, output_dir=tmp_path / "threshold_type_v3")
    args.reference_trace = None
    args.reference_report = None
    args.additional_reference_trace = []
    monkeypatch.setattr(v3, "_select_backend", lambda _args, _reference: "experience_task_success")
    monkeypatch.setattr(v3, "_experience_match", lambda _args: None)

    result = run_v3(args)
    assert result["status"] == "rejected_strict_grasp_screen"
    assert result["strict_preflight_validation"]["exception_type"] == "TypeError"
    assert result["engine_launched"] is False
    assert result["bridge_started"] is False
    assert result["physical_actions"] == 0
    assert (args.output_dir / "run_manifest.json").is_file()
    assert (args.output_dir / "v3_report.json").is_file()
    assert not (args.output_dir / "execution.log").exists()


def test_v3_archives_backend_mismatch_before_engine_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.gs_opd.repair.trajectory_generator_v3 as v3

    report, snapshot_root = _strict_screen_fixture(tmp_path, feasible=True)
    args = _screen_gate_args(report, snapshot_root, output_dir=tmp_path / "backend_mismatch_v3")
    args.reference_trace = None
    args.reference_report = None
    args.additional_reference_trace = []
    monkeypatch.setattr(v3, "_select_backend", lambda _args, _reference: "student_bspline_osc")
    monkeypatch.setattr(v3, "_experience_match", lambda _args: None)

    result = run_v3(args)
    assert result["status"] == "rejected_strict_grasp_screen"
    assert result["strict_preflight_validation"]["failure_reason"] == (
        "strict_grasp_screen_gate_is_supported_only_by_experience_task_success_backend"
    )
    assert result["engine_launched"] is False
    assert result["bridge_started"] is False
    assert result["physical_actions"] == 0
    assert (args.output_dir / "run_manifest.json").is_file()
    assert (args.output_dir / "v3_report.json").is_file()
    assert not (args.output_dir / "execution.log").exists()


def test_native_success_does_not_infer_strict_safety(tmp_path: Path) -> None:
    trace = tmp_path / "episode.npz"
    _write_trace(trace, monitor_status="complete")
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"steps": 2, "success_once": True, "done": True}))
    outcome = _outcome(
        result_path=result_path,
        trace_path=trace,
        route_report_path=None,
        strict_report_path=None,
    )
    assert outcome["native_task_success"] is True
    assert outcome["complete_substep_monitoring"] is True
    assert outcome["strict_10mm_pass"] is False
    assert outcome["safe_training_eligible"] is False


def test_incomplete_monitoring_cannot_pass_strict_report(tmp_path: Path) -> None:
    trace = tmp_path / "episode.npz"
    _write_trace(trace, monitor_status="incomplete")
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"steps": 2, "success_once": True, "done": True}))
    strict_path = tmp_path / "strict.json"
    strict_path.write_text(json.dumps({
        "strict_10mm_pass": True,
        "complete_substep_monitoring": True,
        "native_task_success": True,
        "first_violation": None,
    }))
    outcome = _outcome(
        result_path=result_path,
        trace_path=trace,
        route_report_path=None,
        strict_report_path=strict_path,
    )
    assert outcome["strict_10mm_pass"] is False
    assert outcome["safe_training_eligible"] is False


def test_generator_cli_help_works_without_repository_pythonpath() -> None:
    script = REPO_ROOT / "research/gs_opd/repair/trajectory_generator_v3.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--backend" in result.stdout
    assert "--route-candidate-id" in result.stdout
    assert "--pre-route-hold-actions" in result.stdout
    assert "--strict-grasp-screen-report" in result.stdout


def test_engine_completion_does_not_imply_task_completion() -> None:
    assert _classify_run_status(
        engine_return_code=0,
        scene_matches=True,
        native_task_success=False,
    ) == "task_incomplete"
    assert _classify_run_status(
        engine_return_code=0,
        scene_matches=True,
        native_task_success=True,
    ) == "complete"


def test_experience_engine_completion_does_not_imply_native_task_success() -> None:
    assert _classify_experience_run_status(
        engine_return_code=2,
        report_available=True,
        scene_matches=True,
        native_task_success=False,
    ) == "task_incomplete"
    assert _classify_experience_run_status(
        engine_return_code=0,
        report_available=True,
        scene_matches=True,
        native_task_success=True,
    ) == "complete"
    assert _classify_experience_run_status(
        engine_return_code=2,
        report_available=True,
        scene_matches=False,
        native_task_success=True,
    ) == "incomplete"


def test_route_stall_failure_memory_selects_distinct_experience_approaches() -> None:
    from research.gs_opd.repair.task_success_first_search import (
        blocked_candidate_ids_from_report,
        candidate_matches_failure_memory,
    )

    template_id = "akita_black_bowl:stable_grasp:obstacle_avoidance_L0T0_seed0_init0_d364312882"
    report = {
        "search_failure_memory": {
            "events": [{
                "goal_predicate": '["on","akita_black_bowl_1","plate_1"]',
                "candidate_id": f"experience__{template_id}",
                "route_id": "rrthistorical_high_approach:failed",
                "failure_reason": "RuntimeError: migrated_object_relative_approach_action_budget_exhausted",
                "failure_class": "route_stalled",
                "contact_pairs": [],
            }],
        },
    }
    blocked, imported = blocked_candidate_ids_from_report(
        report,
        goal_predicate='["on","akita_black_bowl_1","plate_1"]',
    )
    assert candidate_matches_failure_memory(template_id, blocked)
    assert imported[0]["failure_class"] == "route_stalled"
    assert imported[0]["next_strategy"] == "skip_same_migrated_route_and_try_distinct_geometry_or_route_family"


def test_experience_backend_forwards_snapshot_and_bounded_physical_budgets(tmp_path: Path) -> None:
    from argparse import Namespace

    snapshot_root = tmp_path / "snapshot_root"
    snapshot_root.mkdir()
    experience_library = tmp_path / "experience.json"
    experience_library.write_text("{}")
    previous_search_report = tmp_path / "previous_search.json"
    previous_search_report.write_text("{}")
    args = Namespace(
        rlinf_python=Path("/workspace/zy_workspace/RLinf/.venv/bin/python"),
        snapshot_root=snapshot_root,
        experience_library=experience_library,
        benchmark="obstacle_avoidance",
        level=0,
        task_id=0,
        init_state=0,
        seed=0,
        demonstration_horizon=1000,
        maximum_physical_branches=3,
        maximum_rrt_preflights=3,
        rrt_time_budget_s=12.0,
        candidate_start_index=3,
        transport_strategy="native_region_progress_osc",
        target_predicate=["on", "akita_black_bowl_1", "plate_1"],
        keypose_experience=[],
        previous_search_report=[previous_search_report],
    )
    argv = _experience_task_success_command(
        args,
        output_dir=tmp_path / "attempt8",
        checkpoint_file=REPO_ROOT / "logs/gs_opd/mink_panda_physical_v3/live_left_bspline_osc_adaptive_20261019_attempt1/direct_bspline_osc_trajectory.npz",
    )
    assert argv[argv.index("--snapshot-root") + 1] == str(snapshot_root.resolve())
    assert argv[argv.index("--demonstration-horizon") + 1] == "1000"
    assert argv[argv.index("--maximum-physical-branches") + 1] == "3"
    assert argv[argv.index("--maximum-rrt-preflights") + 1] == "3"
    assert argv[argv.index("--candidate-start-index") + 1] == "3"
    assert argv[argv.index("--previous-search-report") + 1] == str(previous_search_report.resolve())
    assert argv[argv.index("--target-predicate") + 1:argv.index("--target-predicate") + 4] == [
        "on", "akita_black_bowl_1", "plate_1",
    ]


def test_experience_backend_forwards_identity_checked_runtime_resume_and_grasp_id(tmp_path: Path) -> None:
    from argparse import Namespace

    snapshot_root = tmp_path / "snapshot_root"
    snapshot_root.mkdir()
    experience_library = tmp_path / "experience.json"
    experience_library.write_text("{}")
    resume_snapshot = tmp_path / "failed.pkl"
    resume_snapshot.write_bytes(b"runtime snapshot")
    args = Namespace(
        rlinf_python=Path("/workspace/zy_workspace/RLinf/.venv/bin/python"),
        snapshot_root=snapshot_root,
        experience_library=experience_library,
        benchmark="obstacle_avoidance", level=0, task_id=0, init_state=0, seed=0,
        demonstration_horizon=1000, maximum_physical_branches=1,
        maximum_rrt_preflights=1, rrt_time_budget_s=12.0,
        candidate_start_index=0, transport_strategy="native_region_progress_osc",
        target_predicate=["on", "akita_black_bowl_1", "plate_1"],
        keypose_experience=[], previous_search_report=[],
        resume_from_runtime_snapshot=resume_snapshot,
        grasp_candidate_id="geom111_jaw1_approach2_-1_flip+1",
    )
    argv = _experience_task_success_command(
        args,
        output_dir=tmp_path / "attempt12",
        checkpoint_file=REPO_ROOT / "logs/gs_opd/mink_panda_physical_v3/live_left_bspline_osc_adaptive_20261019_attempt1/direct_bspline_osc_trajectory.npz",
    )
    assert argv[argv.index("--resume-from-runtime-snapshot") + 1] == str(resume_snapshot.resolve())
    assert argv[argv.index("--candidate-id") + 1] == "geom111_jaw1_approach2_-1_flip+1"
    assert argv[argv.index("--candidate-start-index") + 1] == "0"


def test_unsafe_geometry_samples_still_count_as_complete_substep_monitoring() -> None:
    from research.gs_opd.repair.task_success_first_grasp_search import (
        _contact_substep_monitoring_complete,
    )

    contact_monitoring = {
        "status": "complete",
        "expected_physics_substeps_per_control_step": 25,
        "captured_physics_substeps": 25,
        "unobserved_expected_substeps": 0,
        "unavailable_contact_substeps": 0,
        "control_steps": [{
            "status": "complete",
            "expected_physics_substeps": 25,
            "captured_physics_substeps": 25,
            "unobserved_expected_substeps": 0,
            "capture_error": None,
        }],
        "geometry_monitoring": {
            "status": "unsafe",
            "substeps": [{"status": "unsafe"} for _ in range(25)],
        },
    }
    assert _contact_substep_monitoring_complete(contact_monitoring) is True
    contact_monitoring["geometry_monitoring"]["substeps"][3]["status"] = "incomplete"
    assert _contact_substep_monitoring_complete(contact_monitoring) is False


def test_experience_audit_counts_unsafe_but_observed_substeps(tmp_path: Path) -> None:
    search_dir = tmp_path / "grasp_search"
    search_dir.mkdir()
    search_report_path = search_dir / "run_report.json"
    record_path = search_dir / "physical_step_records.jsonl.gz"
    monitoring = {
        "status": "complete",
        "expected_physics_substeps_per_control_step": 25,
        "captured_physics_substeps": 25,
        "unobserved_expected_substeps": 0,
        "unavailable_contact_substeps": 0,
        "control_steps": [{
            "status": "complete",
            "expected_physics_substeps": 25,
            "captured_physics_substeps": 25,
            "unobserved_expected_substeps": 0,
            "capture_error": None,
        }],
        "geometry_monitoring": {
            "status": "unsafe",
            "substeps": [{"status": "unsafe"} for _ in range(25)],
        },
    }
    records = [
        {
            "step_index": index,
            "phase": "grasp_approach",
            "action_7d": [0.0] * 7,
            "state": {},
            "snapshot_state": {},
            "reward": 0.0,
            "done": False,
            "contact_monitoring": monitoring,
        }
        for index in range(2)
    ]
    import gzip

    with gzip.open(record_path, "wt", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    search_report_path.write_text(json.dumps({
        "physical_branches": [{"attempted_control_actions_archived": 2}],
    }))
    inner_report = {
        "segments": [{
            "type": "bounded_expert_grasp_search",
            "report": str(search_report_path),
        }],
    }

    audit = _experience_physical_action_audit(inner_report)
    assert audit["status"] == "complete"
    assert audit["complete_substep_monitoring"] is True
    assert audit["physical_actions_archived"] == 2
    assert audit["captured_physics_substeps"] == 50
    assert audit["state_action_rows_valid"] is True
    assert audit["safety_certified"] is False
    assert audit["geometry_monitoring_status_counts"] == {"unsafe": 2}


def test_experience_audit_aggregates_grasp_and_native_placement_archives(tmp_path: Path) -> None:
    import gzip

    search_dir = tmp_path / "grasp_search"
    placement_dir = tmp_path / "placement_branch_00"
    search_dir.mkdir()
    placement_dir.mkdir()
    search_report_path = search_dir / "run_report.json"
    search_report_path.write_text(json.dumps({
        "physical_branches": [{"branch_control_actions": 2}],
    }))

    monitoring = {
        "status": "complete",
        "expected_physics_substeps_per_control_step": 25,
        "captured_physics_substeps": 25,
        "unobserved_expected_substeps": 0,
        "unavailable_contact_substeps": 0,
        "control_steps": [{
            "status": "complete",
            "expected_physics_substeps": 25,
            "captured_physics_substeps": 25,
            "unobserved_expected_substeps": 0,
            "capture_error": None,
        }],
        "geometry_monitoring": {
            "status": "unsafe",
            "substeps": [{"status": "unsafe"} for _ in range(25)],
        },
    }

    def write_records(path: Path, phase: str, count: int) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for index in range(count):
                handle.write(json.dumps({
                    "step_index": index,
                    "phase": phase,
                    "action_7d": [0.0] * 7,
                    "state": {},
                    "snapshot_state": {},
                    "reward": 0.0,
                    "done": False,
                    "contact_monitoring": monitoring,
                }) + "\n")

    write_records(search_dir / "physical_step_records.jsonl.gz", "grasp_approach", 2)
    placement_records = placement_dir / "physical_step_records.jsonl.gz"
    durable_records = placement_dir / "durable_physical_step_records.jsonl"
    write_records(placement_records, "transport_region_progress", 3)
    durable_records.write_text("durable copy present\n")
    inner_report = {
        "archive": str(tmp_path),
        "segments": [
            {
                "type": "bounded_expert_grasp_search",
                "report": str(search_report_path),
            },
            {
                "type": "expert_native_goal_stage_control",
                "recorded_actions": 3,
                "recorded_physics_substeps": 75,
                "placement_search": {"physical_branch_count": 1},
                "artifacts": {
                    "records": str(placement_records),
                    "durable_records": str(durable_records),
                },
            },
        ],
    }

    audit = _experience_physical_action_audit(inner_report)

    assert audit["status"] == "complete"
    assert audit["record_source_count"] == 2
    assert audit["physical_branches"] == 2
    assert audit["physical_actions_archived"] == 5
    assert audit["action_record_rows"] == 5
    assert audit["captured_physics_substeps"] == 125
    assert audit["complete_substep_monitoring_actions"] == 5
    assert audit["branch_action_count_matches_record_rows"] is True
    assert audit["reported_physics_substeps_match_record_rows"] is True
    assert audit["state_action_rows_valid"] is True
    assert audit["safety_certified"] is False
    assert audit["geometry_monitoring_status_counts"] == {"unsafe": 5}
    assert audit["record_source_summaries"][1]["durable_records_available"] is True


def test_experience_audit_includes_retained_student_handoff_and_excludes_rolled_back_actions(tmp_path: Path) -> None:
    import gzip

    monitoring = {
        "status": "complete", "expected_physics_substeps_per_control_step": 25,
        "captured_physics_substeps": 25, "unobserved_expected_substeps": 0,
        "unavailable_contact_substeps": 0,
        "control_steps": [{"status": "complete", "expected_physics_substeps": 25,
                           "captured_physics_substeps": 25, "capture_error": None}],
        "geometry_monitoring": {"status": "unsafe", "substeps": [{"status": "unsafe"}] * 25},
    }

    def write(path: Path, timestep: int) -> None:
        with gzip.open(path, "wt") as handle:
            handle.write(json.dumps({"step_index": 0, "timestep": timestep,
                "phase": "test", "action_7d": [0.0] * 7, "state": {},
                "snapshot_state": {}, "reward": 0.0, "done": False,
                "contact_monitoring": monitoring}) + "\n")

    first = tmp_path / "student_first.gz"
    handoff = tmp_path / "student_handoff.gz"
    write(first, 1)
    write(handoff, 1)
    inner = {"segments": [
        {"type": "student_first", "actions_executed": 1, "task_success": False,
         "artifacts": {"records": str(first)}},
        {"type": "student_handoff_after_stage_goal", "actions_executed": 1,
         "artifacts": {"records": str(handoff)}},
    ], "final_native_task_status": {"task_success": True, "timestep": 1}}
    audit = _experience_physical_action_audit(inner)
    assert audit["physical_actions_archived"] == 1
    assert audit["captured_physics_substeps"] == 25
    assert audit["discarded_student_segments"] == [{"kind": "student_first", "actions": 1}]
    assert audit["success_episode_complete_substep_monitoring"] is True
    inner["student_handoff_recovery"] = {"status": "exact_expert_grasp_snapshot_restored_after_Student_lost_grasp"}
    rolled_back = _experience_physical_action_audit(inner)
    assert rolled_back["status"] == "not_available"
    assert {row["kind"] for row in rolled_back["discarded_student_segments"]} == {
        "student_first", "student_handoff_after_stage_goal",
    }


def test_route_candidate_override_is_forwarded_to_physical_runner(tmp_path: Path) -> None:
    from argparse import Namespace

    args = Namespace(
        level=0,
        task_id=0,
        init_state=0,
        teacher_json=REPO_ROOT / "logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt",
        semantic_results_json=REPO_ROOT / "logs/gs_opd/semantic_bindings_l0t0.json",
        rlinf_python=Path("/workspace/zy_workspace/RLinf/.venv/bin/python"),
        seed=0,
        bridge_port=38567,
        max_steps=1000,
        semantic_object_id="object_1",
        route_candidate_id="trajectory-002",
        pre_route_hold_actions=0,
        lora_checkpoint=None,
    )
    argv = _student_bspline_command(
        args,
        output_dir=tmp_path,
        checkpoint_file=REPO_ROOT / "logs/gs_opd/mink_panda_physical_v3/live_left_bspline_osc_adaptive_20261019_attempt1/direct_bspline_osc_trajectory.npz",
        route_direction="left",
    )
    index = argv.index("--bspline-candidate-id")
    assert argv[index + 1] == "trajectory-002"


def test_pre_route_hold_actions_are_forwarded_to_physical_runner(tmp_path: Path) -> None:
    from argparse import Namespace

    args = Namespace(
        level=0,
        task_id=0,
        init_state=0,
        teacher_json=REPO_ROOT / "logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt",
        semantic_results_json=REPO_ROOT / "logs/gs_opd/semantic_bindings_l0t0.json",
        rlinf_python=Path("/workspace/zy_workspace/RLinf/.venv/bin/python"),
        seed=0,
        bridge_port=38567,
        max_steps=1000,
        semantic_object_id="object_1",
        route_candidate_id="trajectory-004",
        pre_route_hold_actions=20,
        lora_checkpoint=None,
    )
    argv = _student_bspline_command(
        args,
        output_dir=tmp_path,
        checkpoint_file=DIRECT_REFERENCE,
        route_direction="left",
    )
    index = argv.index("--bspline-pre-route-hold-actions")
    assert argv[index + 1] == "20"


def test_substep_aggregation_counts_pre_route_hold_and_route_actions(tmp_path: Path) -> None:
    trace = tmp_path / "episode.npz"
    _write_trace(trace, monitor_status="complete")
    route = tmp_path / "route.json"
    route.write_text(json.dumps({
        "real_execution_steps": 2,
        "steps": [
            {"phase": "pre_route_hold", "contact_monitoring_status": "complete"},
            {"phase": "route", "contact_monitoring_status": "complete"},
        ],
    }))
    monitoring = _substep_monitoring(trace, route)
    assert monitoring["complete"] is True
    assert monitoring["route_report_steps"] == 1
    assert monitoring["pre_route_hold_steps"] == 1
    assert monitoring["route_report_action_rows"] == 2



