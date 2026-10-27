# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Bounded OA L0T0 placement/release/retreat search from a verified held-place Snapshot."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name("complete_candidate9_task.py")
SCENE = Path("/opt/data/private/gs_opd/rollouts/gs_opd_libero_safety_repro_oa_l0t0_stage_b_20261027_attempt04/scene_source.json.gz")
SOURCE_DIR = Path("/opt/data/private/gs_opd/rollouts/oa_l0t0_candidate9_full_task_20261027_attempt07")
SNAPSHOT = SOURCE_DIR / "stage_place_t344_runtime_snapshot.pkl"
SOURCE_REPORT = SOURCE_DIR / "task_report.json"
MAX_CANDIDATES = 8
STANDARD_HORIZON = 1000

# These are bounded engineering search values, not parameters disclosed by LIBERO-Safety.
CANDIDATES = [
    {"candidate_id": "c01_center_radial_retreat", "placement_xy_fraction": [0.0, 0.0], "placement_yaw_delta_deg": 0, "bowl_bottom_clearance_mm": 1.0, "preplace_clearance_mm": 35, "post_release_open_actions_before_retreat": 0, "retreat_direction_region": "away_from_bowl", "retreat_distance_mm": 55, "open_during_retreat": True},
    {"candidate_id": "c02_xminus_vertical_retreat", "placement_xy_fraction": [-0.55, 0.0], "placement_yaw_delta_deg": 0, "bowl_bottom_clearance_mm": 1.0, "preplace_clearance_mm": 50, "post_release_open_actions_before_retreat": 1, "retreat_direction_region": [0.0, 0.0, 1.0], "retreat_distance_mm": 65, "open_during_retreat": True},
    {"candidate_id": "c03_xplus_yaw90_diagonal", "placement_xy_fraction": [0.55, 0.0], "placement_yaw_delta_deg": 90, "bowl_bottom_clearance_mm": 0.0, "preplace_clearance_mm": 30, "post_release_open_actions_before_retreat": 0, "retreat_direction_region": [0.0, 1.0, 0.5], "retreat_distance_mm": 55, "open_during_retreat": True},
    {"candidate_id": "c04_yminus_yawminus90", "placement_xy_fraction": [0.0, -0.55], "placement_yaw_delta_deg": -90, "bowl_bottom_clearance_mm": 2.0, "preplace_clearance_mm": 45, "post_release_open_actions_before_retreat": 2, "retreat_direction_region": [0.0, -1.0, 0.5], "retreat_distance_mm": 55, "open_during_retreat": True},
    {"candidate_id": "c05_yplus_yaw180", "placement_xy_fraction": [0.0, 0.55], "placement_yaw_delta_deg": 180, "bowl_bottom_clearance_mm": 1.0, "preplace_clearance_mm": 25, "post_release_open_actions_before_retreat": 0, "retreat_direction_region": [1.0, 0.0, 0.3], "retreat_distance_mm": 60, "open_during_retreat": True},
    {"candidate_id": "c06_diagonal_yaw45_high_approach", "placement_xy_fraction": [-0.45, 0.45], "placement_yaw_delta_deg": 45, "bowl_bottom_clearance_mm": 0.0, "preplace_clearance_mm": 60, "post_release_open_actions_before_retreat": 3, "retreat_direction_region": [-1.0, 0.0, 0.4], "retreat_distance_mm": 70, "open_during_retreat": True},
    {"candidate_id": "c07_diagonal_yawminus45", "placement_xy_fraction": [0.45, -0.45], "placement_yaw_delta_deg": -45, "bowl_bottom_clearance_mm": 2.0, "preplace_clearance_mm": 40, "post_release_open_actions_before_retreat": 0, "retreat_direction_region": [0.4, 0.0, 1.0], "retreat_distance_mm": 50, "open_during_retreat": True},
    {"candidate_id": "c08_diagonal_yaw135_delayed_open", "placement_xy_fraction": [-0.4, -0.4], "placement_yaw_delta_deg": 135, "bowl_bottom_clearance_mm": 1.0, "preplace_clearance_mm": 35, "post_release_open_actions_before_retreat": 4, "retreat_direction_region": [-0.4, 0.0, 1.0], "retreat_distance_mm": 50, "open_during_retreat": True},
]


def write(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def accepted_complete_success(report: dict) -> bool:
    final = report.get("final_state", {})
    release = report.get("release", {})
    retreat = report.get("retreat_route", {})
    settling = report.get("settling_budget", {})
    return bool(
        report.get("task_success") is True
        and final.get("native_task_success") is True
        and final.get("stable_release") is True
        and release.get("physical_grasp_separated") is True
        and retreat.get("complete") is True
        and report.get("post_retreat_opening_clearance", {}).get("complete") is True
        and settling.get("complete") is True
        and int(settling.get("executed_actions", 0)) == 12
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded eight-candidate OA L0T0 placement search from the exact candidate9 held-place Snapshot."
    )
    parser.parse_args(argv)
    if not all(path.is_file() for path in (RUNNER, SCENE, SNAPSHOT, SOURCE_REPORT)):
        raise FileNotFoundError("runner_or_exact_source_artifact_missing")
    source_sha = __import__("hashlib").sha256(SNAPSHOT.read_bytes()).hexdigest()
    if source_sha != "7e07f1e16d27aeb09066d311d982c14b33623399cb566e40bb6923cb2c057675":
        raise RuntimeError(f"held_place_source_snapshot_sha_mismatch:{source_sha}")
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    root = Path("/opt/data/private/gs_opd/rollouts") / f"oa_l0t0_candidate9_placement_branch_search_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    candidates = CANDIDATES[:MAX_CANDIDATES]
    budget = STANDARD_HORIZON - 344
    manifest = {
        "schema": "gs_opd_oa_l0t0_candidate9_placement_branch_search.v1",
        "source_snapshot": str(SNAPSHOT),
        "source_snapshot_sha256": source_sha,
        "source_timestep": 344,
        "source_state": "real held-place stage snapshot; exact fresh Bridge restore is checked by each branch runner",
        "source_report": str(SOURCE_REPORT),
        "standard_horizon": STANDARD_HORIZON,
        "per_branch_action_budget": budget,
        "maximum_candidates": MAX_CANDIDATES,
        "candidates_evaluated": 0,
        "complete_task_success": False,
        "safety_success": False,
        "new_trainable_episodes": 0,
        "candidate_results": [],
        "engineering_choices_not_paper_values": True,
        "safety_rule_changed": False,
        "bddl_or_state_written": False,
    }
    write(root / "search_report.json", manifest)
    for i, config in enumerate(candidates, 1):
        branch = root / f"candidate_{i:02d}"
        branch.mkdir()
        config = dict(config)
        config["candidate_index"] = i
        config["retreat_planner_seeds"] = [20261120 + 10 * i, 20261121 + 10 * i, 20261122 + 10 * i]
        config_path = branch / "candidate_config.json"
        write(config_path, config)
        output_dir = branch / "run"
        stdout_path = branch / "runner_stdout.log"
        cmd = [
            sys.executable, str(RUNNER),
            "--scene-source", str(SCENE),
            "--snapshot", str(SNAPSHOT),
            "--source-report", str(SOURCE_REPORT),
            "--output-dir", str(output_dir),
            "--branch-config", str(config_path),
            "--max-suffix-actions", str(budget),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(REPO), env.get("PYTHONPATH", "")))
        with stdout_path.open("wb", buffering=0) as log:
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f"candidate {i}/{len(candidates)} started: {config['candidate_id']} pid={proc.pid}", flush=True)
            code = proc.wait()
        report_path = output_dir / "task_report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = {"status": "runner_did_not_write_report", "error": f"runner_exit_code_{code}"}
        success = accepted_complete_success(report)
        item = {
            "index": i,
            "candidate_id": config["candidate_id"],
            "config": config,
            "runner_exit_code": code,
            "status": report.get("status"),
            "task_success": report.get("task_success") is True,
            "native_task_success_at_terminal": report.get("native_task_success") is True or report.get("native_task_success_at_stop") is True,
            "stable_release": report.get("final_state", {}).get("stable_release") is True,
            "release_confirmed": report.get("release", {}).get("physical_grasp_separated") is True,
            "retreat_complete": report.get("retreat_route", {}).get("complete") is True,
            "settling_complete": report.get("settling_budget", {}).get("complete") is True,
            "physical_actions": report.get("physical_actions", 0),
            "monitoring_complete_actions": report.get("monitoring_complete_actions", 0),
            "strict_safe_actions": report.get("safety_certifiable_actions", 0),
            "route_outcomes": [{"phase": r.get("phase"), "status": r.get("execution_status"), "rejection_reason": r.get("rejection_reason")} for r in report.get("routes", [])],
            "error": report.get("error"),
            "report": str(report_path),
            "terminal_snapshot": report.get("terminal_snapshot", {}).get("path"),
            "video": report.get("artifacts", {}).get("dual_camera_video"),
            "accepted_complete_task_success": success,
        }
        manifest["candidate_results"].append(item)
        manifest["candidates_evaluated"] = i
        manifest["complete_task_success"] = success
        manifest["task_success"] = success
        manifest["safety_success"] = False
        manifest["new_trainable_episodes"] = 0
        manifest["best_candidate_so_far"] = item
        manifest["search_stop_reason"] = "complete_stable_task_success_found" if success else None
        write(root / "search_report.json", manifest)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        if success:
            break
    if not manifest.get("complete_task_success"):
        manifest["search_stop_reason"] = "eight_candidate_budget_exhausted" if manifest["candidates_evaluated"] == MAX_CANDIDATES else "search_stopped_after_evaluated_candidates_without_complete_success"
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write(root / "search_report.json", manifest)
    (root / "search_report.md").write_text(
        "# OA L0T0 candidate9 bounded placement branch search\n\n"
        f"- Candidates evaluated: {manifest['candidates_evaluated']}/{MAX_CANDIDATES}\n"
        f"- Complete Task Success: `{manifest['complete_task_success']}`\n"
        f"- Safety Success: `{manifest['safety_success']}`\n"
        f"- New trainable Episodes: `{manifest['new_trainable_episodes']}`\n"
        f"- Stop reason: `{manifest['search_stop_reason']}`\n"
        f"- Source held-place Snapshot: `{SNAPSHOT}`\n",
        encoding="utf-8",
    )
    print(f"search_archive={root}", flush=True)
    return 0 if manifest["complete_task_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
