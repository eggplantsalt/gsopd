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
"""Rebuild the L1T2 success lineage from reset and recapture complete OA evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.gs_opd.repair import task_success_first_grasp_search as grasp_search  # noqa: E402
from research.gs_opd.repair.run53_release_diagnostic import (  # noqa: E402
    EXPECTED_BDDL_SHA,
    EXPECTED_INSTRUCTION,
    EXPECTED_MODEL_SHA,
    INITIAL_SNAPSHOT,
    NATIVE_GOALS,
    _native_values,
    _verified_resume_runtime_snapshot,
    read_gzip_jsonl,
    sha256_file,
)
from research.gs_opd.repair.run53_release_strategy_comparison import (  # noqa: E402
    BRIDGE_SOURCE,
    SUBSTEPS,
    instrument_bridge_source,
)

ROLLOUT_ROOT = Path("/opt/data/private/gs_opd/rollouts")
BOUNDARY_STATE_ABS_TOLERANCE = 1e-7
RUN19_RECORDS = ROLLOUT_ROOT / (
    "task_success_first_cross_level_L1T2_20261026_run19_L0_success_migration/"
    "obstacle_avoidance_L1T2/stage_00/"
    "task_success_first_executor_obstacle_avoidance_L1T2_20261026_065056_331307395/"
    "expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L1T2_20261026_065110_007137754/"
    "physical_step_records.jsonl.gz"
)
SEGMENT_RECORDS = [
    (
        "run19_reset_to_pot1_lift",
        RUN19_RECORDS,
        1,
        182,
    ),
    (
        "run46_pot1_transport_placement",
        ROLLOUT_ROOT
        / "task_success_first_cross_level_L1T2_20261026_run46_shared_placement_transfer/"
        "task_success_first_executor_obstacle_avoidance_L1T2_20261026_133025_720979512/"
        "expert_native_goal_stage_control/placement_branch_03_region_layout_19/"
        "physical_step_records.jsonl.gz",
        183,
        542,
    ),
    (
        "run47_pot1_release_resume",
        ROLLOUT_ROOT
        / "task_success_first_cross_level_L1T2_20261026_run47_release_partial_goal/"
        "task_success_first_executor_obstacle_avoidance_L1T2_20261026_134434_039862217/"
        "expert_native_release_resume/physical_step_records.jsonl.gz",
        543,
        598,
    ),
    (
        "run48_pot2_grasp",
        ROLLOUT_ROOT
        / "task_success_first_cross_level_L1T2_20261026_run48_second_object_from_partial_prefix/"
        "task_success_first_executor_obstacle_avoidance_L1T2_20261026_134731_258088424/"
        "expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L1T2_20261026_134743_113432329/"
        "physical_step_records.jsonl.gz",
        599,
        857,
    ),
    (
        "run52_pot2_transport",
        ROLLOUT_ROOT / "keypose_driven_L1T2_20261026_run52/"
        "task_success_first_executor_obstacle_avoidance_L1T2_20261026_142348_459675599/"
        "expert_native_goal_stage_control/placement_branch_00_region_layout_00/"
        "physical_step_records.jsonl.gz",
        858,
        1217,
    ),
    (
        "run53_native_success_suffix",
        ROLLOUT_ROOT / "keypose_driven_L1T2_20261026_run53_suffix_replan/"
        "task_success_first_executor_obstacle_avoidance_L1T2_20261026_143005_978311542/"
        "expert_native_goal_stage_control/placement_branch_00_region_layout_00/"
        "physical_step_records.jsonl.gz",
        1218,
        1335,
    ),
]
STABLE_RELEASE_ACTIONS = Path(
    "/opt/data/private/gs_opd/rollouts/run53_release_local_placement_20261026_attempt1/"
    "placement_branch_01_25mm/control_actions.jsonl"
)
EXPECTED_SOURCE_HASHES = {
    "run19_reset_to_pot1_lift": "e4ca5d2aa12f2931ed6edbe0a08b7bde99e5fac6d642b7b0fb03505cb809f905",
    "run46_pot1_transport_placement": "9680fe2251aa2e417824dd30d73acb717e7b06482147584ed2a1392e1aab2ea6",
    "run47_pot1_release_resume": "08d2a9cb46940d204f24747da709563da9f339da0282eb4954e3e78640f3813a",
    "run48_pot2_grasp": "ff640ed5b1871e9f51820d2c81318b7fb14d7e649557c2218256ffa8a4c90d60",
    "run52_pot2_transport": "d92682e42f7222c8ce58c800d38d24db461d51010d1b96566d6532c4601f303a",
    "run53_native_success_suffix": "519e1fb90df5c3e3924444e03fa491d55bfd91426ef7bc964e5590dae86e861e",
    "stable_release_actions": "c8401f6d61397cf129f3b147c45a608284a231c930abbfe8fccd8111774854f5",
}


@dataclass(frozen=True)
class EpisodeAction:
    timestep: int
    segment: str
    source_path: str
    source_row_index: int
    phase: str
    action: tuple[float, ...]
    expected_snapshot_state: dict[str, Any] | None
    expected_done: bool | None
    expected_task_success: bool | None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(
        json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), default=_json_default
        )
        + "\n"
    )


def _state_sha(value: Any) -> str:
    state = np.asarray(value, dtype=np.float64)
    if state.ndim != 1 or not state.size or not np.isfinite(state).all():
        raise ValueError("invalid_or_missing_mujoco_integration_state")
    return hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()


def compare_boundary_state(actual_value: Any, expected_value: Any) -> dict[str, Any]:
    """Report bitwise and floating-point-tolerant equality at a source seam."""
    actual = np.asarray(actual_value, dtype=np.float64)
    expected = np.asarray(expected_value, dtype=np.float64)
    same_shape = actual.ndim == 1 and actual.shape == expected.shape
    max_delta = float(np.max(np.abs(actual - expected))) if same_shape else None
    return {
        "bitwise_exact": bool(same_shape and np.array_equal(actual, expected)),
        "same_shape": bool(same_shape),
        "max_abs_state_delta": max_delta,
        "absolute_tolerance": BOUNDARY_STATE_ABS_TOLERANCE,
        "within_tolerance": bool(
            same_shape and max_delta <= BOUNDARY_STATE_ABS_TOLERANCE
        ),
    }


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_actions() -> tuple[list[EpisodeAction], list[dict[str, Any]]]:
    actions: list[EpisodeAction] = []
    sources: list[dict[str, Any]] = []
    expected_timestep = 1
    for name, path, first_timestep, last_timestep in SEGMENT_RECORDS:
        path = Path(path)
        digest = sha256_file(path)
        if digest != EXPECTED_SOURCE_HASHES[name]:
            raise ValueError(f"source_hash_mismatch:{name}:{digest}")
        rows = read_gzip_jsonl(path)
        timesteps = [int(row["timestep"]) for row in rows]
        if timesteps != list(range(first_timestep, last_timestep + 1)):
            raise ValueError(
                f"source_timestep_gap:{name}:{timesteps[:1]}:{timesteps[-1:]}"
            )
        if expected_timestep != first_timestep:
            raise ValueError(f"source_segments_not_contiguous_before:{name}")
        for index, row in enumerate(rows):
            raw_action = np.asarray(row.get("action_7d"), dtype=np.float64)
            if raw_action.shape != (7,) or not np.isfinite(raw_action).all():
                raise ValueError(f"invalid_action:{name}:{index}")
            actions.append(
                EpisodeAction(
                    timestep=int(row["timestep"]),
                    segment=name,
                    source_path=str(path),
                    source_row_index=index,
                    phase=str(row.get("phase", "unspecified")),
                    action=tuple(float(item) for item in raw_action),
                    expected_snapshot_state=row["snapshot_state"],
                    expected_done=row.get("done"),
                    expected_task_success=row.get("task_success"),
                )
            )
        sources.append(
            {
                "name": name,
                "path": str(path),
                "sha256": digest,
                "action_count": len(rows),
                "timestep_range": [first_timestep, last_timestep],
                "is_original_historical_action_source": True,
            }
        )
        expected_timestep = last_timestep + 1

    stable_hash = sha256_file(STABLE_RELEASE_ACTIONS)
    if stable_hash != EXPECTED_SOURCE_HASHES["stable_release_actions"]:
        raise ValueError(f"stable_release_source_hash_mismatch:{stable_hash}")
    release_rows = [
        json.loads(line)
        for line in STABLE_RELEASE_ACTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    release_rows = [row for row in release_rows if "action_7d" in row]
    if len(release_rows) != 77:
        raise ValueError(f"stable_release_action_count_mismatch:{len(release_rows)}")
    for index, row in enumerate(release_rows, start=1336):
        raw_action = np.asarray(row["action_7d"], dtype=np.float64)
        if raw_action.shape != (7,) or not np.isfinite(raw_action).all():
            raise ValueError(f"invalid_release_action:{index}")
        actions.append(
            EpisodeAction(
                timestep=index,
                segment="run53_stable_release_suffix_25mm",
                source_path=str(STABLE_RELEASE_ACTIONS),
                source_row_index=index - 1336,
                phase=str(row.get("phase", "unknown")),
                action=tuple(float(item) for item in raw_action),
                expected_snapshot_state=None,
                expected_done=row.get("done"),
                expected_task_success=row.get("native_task_status", {}).get(
                    "task_success"
                ),
            )
        )
    sources.append(
        {
            "name": "run53_stable_release_suffix_25mm",
            "path": str(STABLE_RELEASE_ACTIONS),
            "sha256": stable_hash,
            "action_count": len(release_rows),
            "timestep_range": [1336, 1412],
            "protocol_note": "real actions from a t=1335 snapshot branch; replayed after native done on the continuous run",
            "is_original_historical_action_source": False,
        }
    )
    if [action.timestep for action in actions] != list(range(1, 1413)):
        raise ValueError("episode_action_sequence_does_not_cover_1_through_1412")
    return actions, sources


def _write_rgb(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"unexpected_RGB_image:{image.shape}:{image.dtype}")
    if not cv2.imwrite(
        str(path),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    ):
        raise RuntimeError(f"camera_png_write_failed:{path}")
    return sha256_file(path)


def _save_observation(
    output: Path,
    observation_handle: Any,
    reply: dict[str, Any],
    timestep: int,
    capture_type: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    agent = grasp_search._image(reply, "agentview_image")
    wrist = grasp_search._image(reply, "robot0_eye_in_hand_image")
    agent_rel = Path("observations/agentview") / f"{timestep:06d}.png"
    wrist_rel = Path("observations/wrist") / f"{timestep:06d}.png"
    item = {
        "timestep": timestep,
        "capture_type": capture_type,
        "agentview_path": str(agent_rel),
        "agentview_sha256": _write_rgb(output / agent_rel, agent),
        "wrist_path": str(wrist_rel),
        "wrist_sha256": _write_rgb(output / wrist_rel, wrist),
        "robot_state": reply.get("state", {}),
        "snapshot_state": reply.get("snapshot_state"),
        "native_task_status": reply.get("native_task_status"),
        "reward": reply.get("reward"),
        "done": reply.get("done"),
        "task_success": reply.get("task_success"),
    }
    _append_jsonl(observation_handle, item)
    return agent, wrist, item


def _export_snapshot(bridge: Any, path: Path) -> dict[str, Any]:
    result = bridge.request({"op": "export_runtime_snapshot"})
    payload = result.get("payload")
    if result.get("status") != "complete" or not isinstance(payload, bytes):
        raise RuntimeError(
            f"snapshot_export_failed:{result.get('error') or result.get('status')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    state = result["snapshot_state"]["mujoco_integration_state"]
    return {
        "path": str(path),
        "payload_sha256": sha256_file(path),
        "integration_state_sha256": _state_sha(state),
        "timestep": result.get("snapshot_state", {})
        .get("task_bookkeeping", {})
        .get("timestep"),
    }


def _pot2_hand_contact(contacts: Iterable[dict[str, Any]]) -> bool:
    if isinstance(contacts, dict):
        contacts = contacts.get("contacts", [])
    for contact in contacts:
        if isinstance(contact, str):
            normalized = contact.lower()
            if "moka_pot_2_g" in normalized and "gripper" in normalized:
                return True
            continue
        if not isinstance(contact, dict):
            continue
        geom1 = str(contact.get("geom1", {}).get("name", "")).lower()
        geom2 = str(contact.get("geom2", {}).get("name", "")).lower()
        if ("moka_pot_2_g" in geom1 and "gripper" in geom2) or (
            "moka_pot_2_g" in geom2 and "gripper" in geom1
        ):
            return True
    return False


def _contact_pairs(contacts: Any) -> list[tuple[str, str]]:
    if isinstance(contacts, dict):
        contacts = contacts.get("contacts", [])
    if not isinstance(contacts, list):
        return []
    pairs = []
    for contact in contacts:
        if not isinstance(contact, dict):
            continue
        first = str(contact.get("geom1", {}).get("name", ""))
        second = str(contact.get("geom2", {}).get("name", ""))
        if first and second:
            pairs.append(tuple(sorted((first, second))))
    return pairs


def _integer_ranges(values: Iterable[int]) -> list[list[int]]:
    ordered = sorted({int(value) for value in values})
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append([start, previous])
            start = value
        previous = value
    ranges.append([start, previous])
    return ranges


def _refresh_archive_hashes(output: Path) -> dict[str, str]:
    paths = [
        output / "reconstruction_input.json",
        output / "reset_state_comparison.json",
        output / "observation_schema.json",
        output / "actions.jsonl",
        output / "observations.jsonl",
        output / "physics_substeps.jsonl.gz",
        output / "snapshot_boundary_checks.jsonl",
        output / "video_frame_index.json",
        output / "dual_camera_episode_20fps.mp4",
    ]
    paths.extend(sorted((output / "snapshots").glob("*.pkl")))
    hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in paths
        if path.is_file()
    }
    _write_json(output / "artifact_hashes.json", hashes)
    report_path = output / "episode_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["artifact_sha256"] = hashes
        _write_json(report_path, report)
    return hashes


def finalize_existing_archive(output: Path) -> dict[str, Any]:
    """Recompute acceptance from durable files after a post-run parser error."""
    output = Path(output).resolve()
    action_rows = [
        json.loads(line)
        for line in (output / "actions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observation_rows = [
        json.loads(line)
        for line in (output / "observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    prior_report_path = output / "episode_report.json"
    prior_report = (
        json.loads(prior_report_path.read_text(encoding="utf-8"))
        if prior_report_path.is_file()
        else {}
    )
    action_timesteps = [int(row["episode_timestep"]) for row in action_rows]
    action_substeps = dict.fromkeys(action_timesteps, 0)
    settling_timesteps = {
        int(row["episode_timestep"])
        for row in action_rows
        if row.get("source_phase") == "placement_settle"
    }
    settling_substeps: list[dict[str, Any]] = []
    substep_count = 0
    contact_substep_count = 0
    pot2_hand_contact_substep_count = 0
    pot2_hand_contact_timesteps: set[int] = set()
    post_success_hand_contact_substep_count = 0
    post_success_hand_contact_timesteps: set[int] = set()
    unique_contact_pairs: set[tuple[str, str]] = set()
    with gzip.open(
        output / "physics_substeps.jsonl.gz", "rt", encoding="utf-8"
    ) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            substep_count += 1
            timestep = int(row["episode_control_timestep"])
            if timestep in action_substeps:
                action_substeps[timestep] += 1
            if timestep in settling_timesteps:
                settling_substeps.append(row)
            sample_contacts = row.get("contacts", {})
            if isinstance(sample_contacts, dict) and sample_contacts.get("contacts"):
                contact_substep_count += 1
            unique_contact_pairs.update(_contact_pairs(sample_contacts))
            if _pot2_hand_contact(sample_contacts):
                pot2_hand_contact_timesteps.add(timestep)
                pot2_hand_contact_substep_count += 1
                if timestep > 1335:
                    post_success_hand_contact_substep_count += 1
                    post_success_hand_contact_timesteps.add(timestep)
    boundary_rows = [
        json.loads(line)
        for line in (output / "snapshot_boundary_checks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    settling = _settling_summary(settling_substeps)
    final_action = action_rows[-1] if action_rows else {}
    final_observation = observation_rows[-1] if observation_rows else {}
    final_status = final_action.get("native_task_status", {})
    final_predicates = _native_values(final_status)
    final_contacts = final_action.get("contacts", {})
    final_hand_contact = _pot2_hand_contact(final_contacts)
    all_boundaries_verified = len(boundary_rows) == 6 and all(
        row.get("within_tolerance") is True for row in boundary_rows
    )
    all_control_steps_have_25_substeps = (
        action_timesteps == list(range(1, 1413))
        and len(action_substeps) == 1412
        and all(count == SUBSTEPS for count in action_substeps.values())
        and substep_count == 1412 * SUBSTEPS
    )
    observation_pairs_complete = (
        len(observation_rows) == 1413
        and [int(row["timestep"]) for row in observation_rows] == list(range(1413))
        and all(
            (output / row[key]).is_file()
            for row in observation_rows
            for key in ("agentview_path", "wrist_path")
        )
    )
    final_native_success = (
        final_status.get("task_success") is True
        and final_observation.get("task_success") is True
    )
    reset_comparison = json.loads(
        (output / "reset_state_comparison.json").read_text(encoding="utf-8")
    )
    final_snapshot_exists = (
        output / "snapshots/t1412_final_stable_release.pkl"
    ).is_file()
    final_video_exists = (output / "dual_camera_episode_20fps.mp4").is_file() and (
        output / "dual_camera_episode_20fps.mp4"
    ).stat().st_size > 0
    stable_release = bool(
        len(action_rows) == 1412
        and all_boundaries_verified
        and reset_comparison.get("exact_match") is True
        and all_control_steps_have_25_substeps
        and observation_pairs_complete
        and final_snapshot_exists
        and final_video_exists
        and all(
            row.get("expected_done") == row.get("actual_done")
            and row.get("expected_native_task_success")
            == row.get("actual_native_task_success")
            for row in boundary_rows
        )
        and final_native_success
        and all(
            final_predicates.get(key) is True
            for key in ("pot1_on", "pot2_on", "stove_on")
        )
        and final_action.get("done") is True
        and not final_hand_contact
        and settling.get("sample_count") == 500
        and settling.get("all_native_predicates_true") is True
        and settling.get("all_substeps_hand_pot2_contact_free") is True
        and settling.get("pot2_position_span_m", float("inf")) <= 0.002
        and settling.get("pot2_max_linear_speed_m_s", float("inf")) <= 0.08
        and settling.get("pot2_max_angular_speed_rad_s", float("inf")) <= 0.8
    )
    post_done_steps = sum(1 for timestep in action_timesteps if timestep > 1335)
    report = {
        **prior_report,
        "schema_version": "gs_opd_l1t2_reconstructed_success_episode.v1",
        "status": "complete_physical_success_episode_reconstructed"
        if stable_release
        else "physical_replay_complete_acceptance_failed",
        "task": "obstacle_avoidance/L1T2 seed=0 init_state=0",
        "archive": str(output),
        "action_count_completed": len(action_rows),
        "expected_action_count": 1412,
        "observation_count": len(observation_rows),
        "observation_action_pairs_complete": observation_pairs_complete,
        "physics_substeps_captured": substep_count,
        "physics_substeps_expected": 1412 * SUBSTEPS,
        "all_captured_control_steps_have_25_physics_substeps": all_control_steps_have_25_substeps,
        "source_snapshot_boundaries": boundary_rows,
        "all_source_snapshot_boundaries_verified_with_tolerance": all_boundaries_verified,
        "boundary_bitwise_exact_count": sum(
            bool(row.get("bitwise_exact")) for row in boundary_rows
        ),
        "boundary_state_absolute_tolerance": BOUNDARY_STATE_ABS_TOLERANCE,
        "fresh_reset_exactly_matches_archived_t0": reset_comparison.get("exact_match")
        is True,
        "boundary_native_done_and_success_values_match_sources": all(
            row.get("expected_done") == row.get("actual_done")
            and row.get("expected_native_task_success")
            == row.get("actual_native_task_success")
            for row in boundary_rows
        ),
        "native_success_first_observed_at_timestep": next(
            (
                int(row["episode_timestep"])
                for row in action_rows
                if row.get("task_success") is True
            ),
            None,
        ),
        "native_success_after_stable_release": final_native_success,
        "final_native_predicates": final_predicates,
        "final_reward": final_action.get("reward"),
        "final_done": final_action.get("done"),
        "final_hand_pot2_contact": final_hand_contact,
        "settling": settling,
        "contact_audit": {
            "physics_substep_rows_with_at_least_one_contact": contact_substep_count,
            "unique_contact_geom_pairs": [
                list(pair) for pair in sorted(unique_contact_pairs)
            ],
            "pot2_gripper_contact_substep_count": pot2_hand_contact_substep_count,
            "pot2_gripper_contact_control_timestep_count": len(
                pot2_hand_contact_timesteps
            ),
            "pot2_gripper_contact_control_timestep_ranges": _integer_ranges(
                pot2_hand_contact_timesteps
            ),
            "post_native_success_pot2_gripper_contact_substep_count": post_success_hand_contact_substep_count,
            "post_native_success_pot2_gripper_contact_timestep_ranges": _integer_ranges(
                post_success_hand_contact_timesteps
            ),
            "pot2_gripper_contact_during_settling_substeps": settling.get(
                "hand_contact_substep_count", 0
            ),
            "final_pot2_gripper_contact": final_hand_contact,
            "all_action_substeps_have_contact_records": substep_count
            == len(action_rows) * SUBSTEPS,
        },
        "stable_release_episode_acceptance": stable_release,
        "final_snapshot_exists": final_snapshot_exists,
        "dual_camera_video_exists": final_video_exists,
        "task_success": final_native_success,
        "safety_certified": False,
        "run53_done_protocol": {
            "done_true_at_source_t1335": True,
            "standard_horizon": 1000,
            "used_demo_only_horizon_extension": True,
            "configured_horizon": 1600,
            "steps_physically_advanced_after_done": post_done_steps,
            "native_done_definition_changed": False,
            "BDDL_changed": False,
            "suffix_replayed_continuously_after_done": post_done_steps == 77,
            "note": "The Bridge used a demo-only horizon of 1600 to cover 1412 controls. After native done=true at t=1335, all 77 archived release suffix controls were physically executed on the same continuous instance. No task-success or BDDL code was altered.",
        },
        "failure": None,
        "post_run_report_parser_issue": prior_report.get("failure"),
        "post_run_archive_audit_recovered_without_physics_replay": True,
        "video": str(output / "dual_camera_episode_20fps.mp4"),
        "actions": str(output / "actions.jsonl"),
        "observations": str(output / "observations.jsonl"),
        "physics_substeps": str(output / "physics_substeps.jsonl.gz"),
        "official_safety_demos_used": False,
        "object_state_direct_write": False,
        "training_started": False,
    }
    _write_json(prior_report_path, report)
    video_index = [{"timestep": 0, "capture_type": "fresh_seeded_reset"}]
    video_index.extend(
        {
            "timestep": int(row["episode_timestep"]),
            "capture_type": "post_action",
            "source_segment": row["source_segment"],
            "source_phase": row["source_phase"],
        }
        for row in action_rows
    )
    _write_json(output / "video_frame_index.json", video_index)
    _refresh_archive_hashes(output)
    return report


def _object_state(snapshot: dict[str, Any], name: str) -> dict[str, Any]:
    key = f"objects_dict:{name}"
    state = snapshot.get("object_states", {}).get(key)
    if not isinstance(state, dict):
        raise KeyError(f"object_state_missing:{key}")
    return state


def _settling_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0, "complete": False}
    native_rows = [_native_values(row["native_task_status"]) for row in samples]
    states = [_object_state(row["snapshot_state"], "moka_pot_2") for row in samples]
    positions = np.asarray([state["root_pos"] for state in states], dtype=np.float64)
    linear = [
        float(
            np.linalg.norm(
                np.asarray(state.get("joint_qvel", []), dtype=np.float64)[:3]
            )
        )
        for state in states
        if len(state.get("joint_qvel", [])) == 6
    ]
    angular = [
        float(
            np.linalg.norm(
                np.asarray(state.get("joint_qvel", []), dtype=np.float64)[3:]
            )
        )
        for state in states
        if len(state.get("joint_qvel", [])) == 6
    ]
    hand_contacts = [_pot2_hand_contact(row.get("contacts", [])) for row in samples]
    return {
        "sample_count": len(samples),
        "all_native_predicates_true": all(
            all(native.get(key) is True for key in ("pot1_on", "pot2_on", "stove_on"))
            for native in native_rows
        ),
        "all_substeps_hand_pot2_contact_free": not any(hand_contacts),
        "pot2_position_span_m": float(
            np.max(np.linalg.norm(positions - positions[0], axis=1))
        ),
        "pot2_max_linear_speed_m_s": max(linear, default=None),
        "pot2_max_angular_speed_rad_s": max(angular, default=None),
        "velocity_samples_complete": len(linear) == len(samples)
        and len(angular) == len(samples),
        "start_position_world_m": positions[0].tolist(),
        "end_position_world_m": positions[-1].tolist(),
        "hand_contact_substep_count": int(sum(hand_contacts)),
    }


def replay_episode(output: Path, demonstration_horizon: int = 1600) -> dict[str, Any]:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not 1412 < demonstration_horizon <= 1800:
        raise ValueError("demonstration_horizon_must_cover_episode_and_be_at_most_1800")
    actions, sources = _load_actions()
    initial = _verified_resume_runtime_snapshot(
        INITIAL_SNAPSHOT,
        benchmark="obstacle_avoidance",
        level=1,
        task_id=2,
        seed=0,
        init_state=0,
        model_sha=EXPECTED_MODEL_SHA,
        bddl_sha=EXPECTED_BDDL_SHA,
        instruction=EXPECTED_INSTRUCTION,
    )
    expected_t0 = np.asarray(
        initial["snapshot_state"]["mujoco_integration_state"], dtype=np.float64
    )
    _write_json(
        output / "reconstruction_input.json",
        {
            "task": "obstacle_avoidance/L1T2",
            "seed": 0,
            "init_state": 0,
            "action_count": len(actions),
            "native_success_timestep_before_release": 1335,
            "release_suffix_action_count": 77,
            "model_geometry_sha256": EXPECTED_MODEL_SHA,
            "bddl_sha256": EXPECTED_BDDL_SHA,
            "instruction": EXPECTED_INSTRUCTION,
            "initial_runtime_snapshot": {
                "path": str(INITIAL_SNAPSHOT),
                "payload_sha256": initial["payload_sha256"],
                "integration_state_sha256": initial["integration_state_sha256"],
            },
            "sources": sources,
            "protocol": {
                "standard_benchmark_horizon": 1000,
                "used_demonstration_only_horizon_extension": demonstration_horizon
                != 1000,
                "demonstration_horizon": demonstration_horizon,
                "run53_native_done_true_at_t1335": True,
                "release_suffix_executed_after_done_on_continuous_environment": True,
                "official_safety_demonstrations_used": False,
                "native_BDDL_modified": False,
                "object_state_direct_write": False,
                "safety_certified": False,
                "training_started": False,
            },
        },
    )

    original_bridge_script = grasp_search.BRIDGE_SCRIPT
    runtime_script = BRIDGE_SOURCE.with_name(f".replay_l1t2_bridge_{os.getpid()}.py")
    runtime_script.write_text(
        instrument_bridge_source(BRIDGE_SOURCE.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    bridge = grasp_search.BridgeSession(
        ("127.0.0.1", _port()),
        output / "safety_bridge.log",
        argparse.Namespace(
            benchmark="obstacle_avoidance",
            level=1,
            task_id=2,
            init_state=0,
            seed=0,
            camera_resolution=256,
            demonstration_horizon=demonstration_horizon,
        ),
    )
    grasp_search.BRIDGE_SCRIPT = runtime_script
    writer = None
    observation_count = 0
    substep_count = 0
    boundary_results: list[dict[str, Any]] = []
    native_success_timestep: int | None = None
    post_done_steps = 0
    failure: str | None = None
    action_path = output / "actions.jsonl"
    observation_path = output / "observations.jsonl"
    substep_path = output / "physics_substeps.jsonl.gz"
    boundary_path = output / "snapshot_boundary_checks.jsonl"
    frame_index: list[dict[str, Any]] = []
    settling_substeps: list[dict[str, Any]] = []
    final_reply: dict[str, Any] | None = None
    latest_snapshot_state: dict[str, Any] | None = None
    try:
        bridge.start()
        reset = bridge.request({"op": "reset"})
        metadata = reset.get("scene_metadata", {})
        if metadata.get("model_geometry_sha256") != EXPECTED_MODEL_SHA:
            raise RuntimeError("reset_model_fingerprint_mismatch")
        if metadata.get("bddl_sha256") != EXPECTED_BDDL_SHA:
            raise RuntimeError("reset_BDDL_fingerprint_mismatch")
        if reset.get("instruction") != EXPECTED_INSTRUCTION:
            raise RuntimeError(f"reset_instruction_mismatch:{reset.get('instruction')}")
        current = bridge.request({"op": "observe", "include_snapshot_state": True})
        current["native_task_status"] = bridge.request({"op": "native_task_status"})
        actual_t0 = np.asarray(
            current["snapshot_state"]["mujoco_integration_state"], dtype=np.float64
        )
        exact_t0 = actual_t0.shape == expected_t0.shape and np.array_equal(
            actual_t0, expected_t0
        )
        _write_json(
            output / "reset_state_comparison.json",
            {
                "reset_was_used_without_snapshot_state_injection": True,
                "archived_t0_payload_sha256": initial["payload_sha256"],
                "archived_t0_integration_state_sha256": _state_sha(expected_t0),
                "reset_integration_state_sha256": _state_sha(actual_t0),
                "exact_match": bool(exact_t0),
                "max_abs_state_delta": float(np.max(np.abs(actual_t0 - expected_t0)))
                if actual_t0.shape == expected_t0.shape
                else None,
            },
        )
        if not exact_t0:
            raise RuntimeError(
                "fresh_seeded_reset_does_not_match_archived_t0; no_snapshot_restore_attempted"
            )
        if current["native_task_status"].get("task_success") is not False:
            raise RuntimeError("reset_native_task_success_not_false")

        observation_handle = observation_path.open("w", encoding="utf-8")
        substep_handle = gzip.open(
            substep_path, "wt", encoding="utf-8", compresslevel=3
        )
        action_handle = action_path.open("w", encoding="utf-8")
        boundary_handle = boundary_path.open("w", encoding="utf-8")
        try:
            agent, wrist, obs_row = _save_observation(
                output, observation_handle, current, 0, "fresh_seeded_reset"
            )
            observation_count += 1
            _write_json(
                output / "observation_schema.json",
                {
                    "observation_fields": [
                        "agentview RGB PNG",
                        "wrist RGB PNG",
                        "robot_state",
                        "MuJoCo snapshot_state",
                        "native_task_status",
                        "reward",
                        "done",
                        "task_success",
                    ],
                    "action_fields": [
                        "action_7d",
                        "source_segment",
                        "source_timestep",
                        "source_phase",
                    ],
                    "camera_images_are_lossless_PNG": True,
                    "action_alignment": "action at control timestep t maps observations[t] to observations[t+1]",
                    "native_predicate_names": list(NATIVE_GOALS),
                },
            )
            video_path = output / "dual_camera_episode_20fps.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                20.0,
                (agent.shape[1] + wrist.shape[1], agent.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError("dual_camera_video_writer_failed")
            writer.write(
                cv2.cvtColor(np.concatenate((agent, wrist), axis=1), cv2.COLOR_RGB2BGR)
            )
            frame_index.append({"timestep": 0, "capture_type": "fresh_seeded_reset"})
            _export_snapshot(bridge, output / "snapshots/t0000_initial_reset.pkl")

            for index, source_action in enumerate(actions):
                actual_timestep_before = int(current.get("timestep", -1))
                if actual_timestep_before != source_action.timestep - 1:
                    failure = f"pre_action_timestep_mismatch:expected={source_action.timestep - 1}:actual={actual_timestep_before}"
                    break
                reply = bridge.request(
                    {
                        "op": "step_monitored",
                        "action": list(source_action.action),
                        "action_index": index,
                        "capture_physical_state_substeps": True,
                    }
                )
                if reply.get("error"):
                    failure = (
                        f"step_error_at_t{source_action.timestep}:{reply['error']}"
                    )
                    break
                samples = reply.get("physical_state_substeps", [])
                monitor = reply.get("contact_monitoring", {})
                if (
                    len(samples) != SUBSTEPS
                    or monitor.get("expected_physics_substeps_per_control_step")
                    != SUBSTEPS
                    or monitor.get("captured_physics_substeps") != SUBSTEPS
                    or reply.get("physical_state_capture_errors")
                ):
                    failure = (
                        f"incomplete_25_substep_capture_at_t{source_action.timestep}"
                    )
                    break
                substep_count += len(samples)
                for sample in samples:
                    record = {
                        "episode_control_timestep": source_action.timestep,
                        "source_segment": source_action.segment,
                        "source_control_timestep": source_action.timestep,
                        "source_phase": source_action.phase,
                        "physics_substep_index": sample["physics_substep_index"],
                        "simulation_time_s": sample["simulation_time_s"],
                        "snapshot_state": sample["snapshot_state"],
                        "native_task_status": sample["native_task_status"],
                        "contacts": sample["contacts"],
                    }
                    _append_jsonl(substep_handle, record)
                    if source_action.phase == "placement_settle":
                        settling_substeps.append(record)

                endpoint = reply.get("snapshot_state", {})
                endpoint_state = np.asarray(
                    endpoint.get("mujoco_integration_state", []), dtype=np.float64
                )
                endpoint_native = bridge.request({"op": "native_task_status"})
                endpoint_contacts = bridge.request({"op": "inspect_contacts"})
                endpoint["mujoco_integration_state_sha256"] = _state_sha(endpoint_state)
                if reply.get("timestep") != source_action.timestep:
                    failure = f"post_action_timestep_mismatch_at_t{source_action.timestep}:actual={reply.get('timestep')}"
                    break
                if actual_timestep_before >= 1335:
                    post_done_steps += 1
                    if int(reply.get("timestep", -1)) != actual_timestep_before + 1:
                        failure = f"post_done_physics_did_not_advance_at_t{source_action.timestep}"
                        break
                if (
                    reply.get("task_success") is True
                    and native_success_timestep is None
                ):
                    native_success_timestep = int(reply.get("timestep", -1))

                source_row = {
                    "episode_timestep": source_action.timestep,
                    "source_segment": source_action.segment,
                    "source_path": source_action.source_path,
                    "source_row_index": source_action.source_row_index,
                    "source_phase": source_action.phase,
                    "action_7d": list(source_action.action),
                    "observation_timestep_before": source_action.timestep - 1,
                    "observation_timestep_after": source_action.timestep,
                    "robot_state": endpoint.get("robot_states", [{}])[0],
                    "object_states": endpoint.get("object_states", {}),
                    "reward": reply.get("reward"),
                    "done": reply.get("done"),
                    "task_success": reply.get("task_success"),
                    "native_task_status": endpoint_native,
                    "contacts": endpoint_contacts,
                    "substep_count": len(samples),
                    "substep_monitoring": monitor,
                    "endpoint_integration_state_sha256": endpoint[
                        "mujoco_integration_state_sha256"
                    ],
                    "safety_certified": False,
                }
                _append_jsonl(action_handle, source_row)

                current = reply
                current["native_task_status"] = endpoint_native
                agent, wrist, _ = _save_observation(
                    output,
                    observation_handle,
                    current,
                    int(reply["timestep"]),
                    "post_action",
                )
                observation_count += 1
                writer.write(
                    cv2.cvtColor(
                        np.concatenate((agent, wrist), axis=1), cv2.COLOR_RGB2BGR
                    )
                )
                frame_index.append(
                    {
                        "timestep": int(reply["timestep"]),
                        "capture_type": "post_action",
                        "source_segment": source_action.segment,
                        "source_phase": source_action.phase,
                    }
                )
                latest_snapshot_state = endpoint

                if source_action.expected_snapshot_state is not None and (
                    index + 1 == len(actions)
                    or actions[index + 1].segment != source_action.segment
                ):
                    expected_state = np.asarray(
                        source_action.expected_snapshot_state[
                            "mujoco_integration_state"
                        ],
                        dtype=np.float64,
                    )
                    comparison = {
                        "segment": source_action.segment,
                        "timestep": source_action.timestep,
                        "expected_integration_state_sha256": _state_sha(expected_state),
                        "actual_integration_state_sha256": _state_sha(endpoint_state),
                        **compare_boundary_state(endpoint_state, expected_state),
                        "expected_native_task_success": source_action.expected_task_success,
                        "actual_native_task_success": endpoint_native.get(
                            "task_success"
                        ),
                        "expected_done": source_action.expected_done,
                        "actual_done": reply.get("done"),
                    }
                    snapshot_meta = _export_snapshot(
                        bridge,
                        output
                        / f"snapshots/t{source_action.timestep:04d}_{source_action.segment}.pkl",
                    )
                    comparison["replay_snapshot"] = snapshot_meta
                    boundary_results.append(comparison)
                    _append_jsonl(boundary_handle, comparison)
                    if not comparison["within_tolerance"]:
                        failure = f"source_snapshot_boundary_mismatch:{source_action.segment}:t{source_action.timestep}"
                        break
                if index % 50 == 49:
                    for handle in (
                        observation_handle,
                        action_handle,
                        substep_handle,
                        boundary_handle,
                    ):
                        handle.flush()
                    with action_path.open("rb") as durable:
                        os.fsync(durable.fileno())
                    print(
                        f"replayed {source_action.timestep}/1412 actions ({substep_count} substeps)",
                        flush=True,
                    )

            final_reply = current
            final_status = bridge.request({"op": "native_task_status"})
            final_contacts = bridge.request({"op": "inspect_contacts"})
            if (
                latest_snapshot_state is not None
                and int(final_reply.get("timestep", -1)) == 1412
            ):
                _export_snapshot(
                    bridge, output / "snapshots/t1412_final_stable_release.pkl"
                )
            boundary_handle.flush()
            action_handle.flush()
            observation_handle.flush()
            substep_handle.flush()
        finally:
            observation_handle.close()
            action_handle.close()
            substep_handle.close()
            boundary_handle.close()

        settling = _settling_summary(settling_substeps)
        final_native = _native_values(final_status)
        final_is_native_success = final_status.get("task_success") is True
        final_hand_contact = _pot2_hand_contact(final_contacts.get("contacts", []))
        all_boundaries_verified = len(boundary_results) == 6 and all(
            row["within_tolerance"] for row in boundary_results
        )
        actions_complete = (
            failure is None
            and final_reply is not None
            and final_reply.get("timestep") == 1412
        )
        stable_release = bool(
            actions_complete
            and final_is_native_success
            and all(
                final_native.get(key) is True
                for key in ("pot1_on", "pot2_on", "stove_on")
            )
            and not final_hand_contact
            and settling.get("sample_count") == 500
            and settling.get("all_native_predicates_true") is True
            and settling.get("all_substeps_hand_pot2_contact_free") is True
            and settling.get("pot2_position_span_m", float("inf")) <= 0.002
            and settling.get("pot2_max_linear_speed_m_s", float("inf")) <= 0.08
            and settling.get("pot2_max_angular_speed_rad_s", float("inf")) <= 0.8
        )
        report = {
            "schema_version": "gs_opd_l1t2_reconstructed_success_episode.v1",
            "status": "complete_physical_success_episode_reconstructed"
            if stable_release and all_boundaries_verified
            else (
                "replay_incomplete"
                if failure
                else "physical_replay_complete_acceptance_failed"
            ),
            "task": "obstacle_avoidance/L1T2 seed=0 init_state=0",
            "archive": str(output),
            "action_count_completed": int(final_reply.get("timestep", 0))
            if final_reply
            else 0,
            "expected_action_count": 1412,
            "observation_count": observation_count,
            "observation_action_pairs_complete": observation_count
            == (int(final_reply.get("timestep", 0)) + 1 if final_reply else 0),
            "physics_substeps_captured": substep_count,
            "physics_substeps_expected": (
                int(final_reply.get("timestep", 0)) if final_reply else 0
            )
            * SUBSTEPS,
            "all_captured_control_steps_have_25_physics_substeps": substep_count
            == (int(final_reply.get("timestep", 0)) if final_reply else 0) * SUBSTEPS,
            "source_snapshot_boundaries": boundary_results,
            "all_source_snapshot_boundaries_verified_with_tolerance": all_boundaries_verified,
            "boundary_bitwise_exact_count": sum(
                bool(row["bitwise_exact"]) for row in boundary_results
            ),
            "boundary_state_absolute_tolerance": BOUNDARY_STATE_ABS_TOLERANCE,
            "native_success_first_observed_at_timestep": native_success_timestep,
            "native_success_after_stable_release": final_is_native_success,
            "final_native_predicates": final_native,
            "final_hand_pot2_contact": final_hand_contact,
            "settling": settling,
            "stable_release_episode_acceptance": stable_release,
            "task_success": final_is_native_success,
            "safety_certified": False,
            "run53_done_protocol": {
                "done_true_at_source_t1335": True,
                "standard_horizon": 1000,
                "used_demo_only_horizon_extension": demonstration_horizon != 1000,
                "configured_horizon": demonstration_horizon,
                "steps_physically_advanced_after_done": post_done_steps,
                "native_done_definition_changed": False,
                "BDDL_changed": False,
                "suffix_replayed_continuously_after_done": post_done_steps == 77,
                "note": "The Bridge was configured with a demo-only extended horizon because the complete sequence is 1412 controls; at t=1335 native done became true, then each archived suffix action was still physically sent and checked. No task-success or BDDL code was altered.",
            },
            "failure": failure,
            "video": str(output / "dual_camera_episode_20fps.mp4"),
            "actions": str(action_path),
            "observations": str(observation_path),
            "physics_substeps": str(substep_path),
            "official_safety_demos_used": False,
            "object_state_direct_write": False,
            "training_started": False,
        }
        _write_json(output / "episode_report.json", report)
        (output / "video_frame_index.json").write_text(
            json.dumps(frame_index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if writer is not None:
            writer.release()
            writer = None
        return finalize_existing_archive(output)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        _write_json(
            output / "episode_report.json",
            {
                "schema_version": "gs_opd_l1t2_reconstructed_success_episode.v1",
                "status": "replay_aborted_with_evidence_preserved",
                "task": "obstacle_avoidance/L1T2 seed=0 init_state=0",
                "archive": str(output),
                "failure": failure,
                "completed_substeps": substep_count,
                "completed_observations": observation_count,
                "completed_boundaries": boundary_results,
                "native_success_timestep": native_success_timestep,
                "safety_certified": False,
            },
        )
        raise
    finally:
        if writer is not None:
            writer.release()
        bridge.close()
        grasp_search.BRIDGE_SCRIPT = original_bridge_script
        runtime_script.unlink(missing_ok=True)
        durable_paths = [
            output / "reconstruction_input.json",
            output / "reset_state_comparison.json",
            output / "actions.jsonl",
            output / "observations.jsonl",
            output / "physics_substeps.jsonl.gz",
            output / "snapshot_boundary_checks.jsonl",
            output / "observation_schema.json",
            output / "video_frame_index.json",
            output / "dual_camera_episode_20fps.mp4",
        ]
        durable_paths.extend(sorted((output / "snapshots").glob("*.pkl")))
        artifact_hashes = {
            str(path.relative_to(output)): sha256_file(path)
            for path in durable_paths
            if path.is_file()
        }
        _write_json(output / "artifact_hashes.json", artifact_hashes)
        report_path = output / "episode_report.json"
        if report_path.is_file():
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
            report_data["artifact_sha256"] = artifact_hashes
            _write_json(report_path, report_data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demonstration-horizon", type=int, default=1600)
    parser.add_argument(
        "--finalize-existing-archive",
        action="store_true",
        help="recompute final integrity/acceptance from durable files without launching MuJoCo",
    )
    args = parser.parse_args()
    report = (
        finalize_existing_archive(args.output)
        if args.finalize_existing_archive
        else replay_episode(args.output, args.demonstration_horizon)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
