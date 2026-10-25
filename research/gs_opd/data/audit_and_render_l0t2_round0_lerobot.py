#!/usr/bin/env python3
"""Read-only visual and alignment audit for the accepted L0T2 LeRobot episode."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET = Path(
    "/opt/data/private/gs_opd/datasets/"
    "lerobot_l0t2_round0_operational_success_v21_20261025"
)
DEFAULT_OUTPUT = Path(
    "/opt/data/private/gs_opd/audits/"
    "l0t2_round0_dataset_visual_audit_20261025"
)
EXPECTED_FRAMES = 1460
FPS = 20
KEY_TIMESTEPS = (0, 10, 249, 329, 649, 1019, 1061, 1459)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def as_hwc_u8(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.shape == (3, 256, 256):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape != (256, 256, 3):
        raise ValueError(f"unexpected_reader_image_shape:{arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=np.uint8)
    return np.ascontiguousarray(arr)


def choose_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def phase_color(phase: str) -> tuple[int, int, int]:
    palette = {
        "raw_pi05_initial_prefix": (82, 159, 247),
        "pregrasp": (86, 184, 220),
        "approach": (255, 205, 86),
        "close_gripper": (255, 163, 72),
        "stable_grasp": (240, 132, 74),
        "lift": (184, 143, 255),
        "transport": (83, 194, 151),
        "precise_placement": (255, 170, 125),
        "release": (242, 117, 127),
        "withdrawal": (148, 176, 255),
        "settle": (189, 194, 205),
        "second_object_recovery": (116, 205, 184),
    }
    return palette.get(phase, (210, 210, 210))


def render_frame(
    agent: np.ndarray,
    wrist: np.ndarray,
    *,
    timestep: int,
    phase: str,
    source_segment: int,
    segment_start: int,
    segment_end: int,
    source_trace: str,
    action: np.ndarray,
    state: np.ndarray,
    timestamp: float,
    large: bool = False,
) -> Image.Image:
    scale = 2 if large else 2
    view_w = view_h = 256 * scale
    header_h = 88 if large else 88
    footer_h = 112 if large else 112
    width, height = 2 * view_w, header_h + view_h + footer_h
    canvas = Image.new("RGB", (width, height), (22, 25, 30))
    draw = ImageDraw.Draw(canvas)
    title_font = choose_font(24 if large else 23)
    body_font = choose_font(17 if large else 16)
    small_font = choose_font(14 if large else 13)
    accent = phase_color(phase)

    draw.rectangle((0, 0, width, header_h), fill=(34, 39, 46))
    draw.rectangle((0, 0, 9, header_h), fill=accent)
    draw.text((22, 9), f"t={timestep:04d}   {phase}", font=title_font, fill=(246, 248, 250))
    draw.text(
        (22, 48),
        f"source seg {source_segment:02d} [{segment_start},{segment_end})  |  "
        f"{source_trace[:60]}  |  pre-action obs @ {timestamp:.3f}s",
        font=small_font,
        fill=(188, 198, 208),
    )

    agent_img = Image.fromarray(agent, mode="RGB").resize((view_w, view_h), Image.Resampling.BILINEAR)
    wrist_img = Image.fromarray(wrist, mode="RGB").resize((view_w, view_h), Image.Resampling.BILINEAR)
    canvas.paste(agent_img, (0, header_h))
    canvas.paste(wrist_img, (view_w, header_h))
    draw.rectangle((0, header_h, view_w, header_h + 34), fill=(18, 23, 28))
    draw.rectangle((view_w, header_h, width, header_h + 34), fill=(18, 23, 28))
    draw.text((12, header_h + 6), "Agentview / stored RGB", font=body_font, fill=(248, 248, 248))
    draw.text((view_w + 12, header_h + 6), "Wrist / stored RGB", font=body_font, fill=(248, 248, 248))

    gripper = float(action[6])
    if gripper > 1e-6:
        command_text = "CLOSE (+1 = close)"
        command_color = (255, 185, 100)
    elif gripper < -1e-6:
        command_text = "OPEN (-1 = open)"
        command_color = (114, 203, 255)
    else:
        command_text = "HOLD (0 = hold)"
        command_color = (197, 204, 213)
    grip_qpos = state[6:8]
    grip_proxy = float(np.sum(np.abs(grip_qpos)))

    draw.rectangle((0, header_h + view_h, width, height), fill=(34, 39, 46))
    draw.text(
        (18, header_h + view_h + 9),
        "Executed 7D OSC_POSE action [dx,dy,dz,drx,dry,drz,grip]: "
        + np.array2string(action, precision=4, suppress_small=True, max_line_width=200),
        font=body_font,
        fill=(241, 244, 247),
    )
    draw.text(
        (18, header_h + view_h + 42),
        f"gripper command: {gripper:+.3f}  {command_text}    "
        f"observed finger qpos: [{grip_qpos[0]:+.4f}, {grip_qpos[1]:+.4f}] "
        f"(|q| sum={grip_proxy:.4f})",
        font=body_font,
        fill=command_color,
    )
    draw.text(
        (18, header_h + view_h + 76),
        f"pre-action EEF xyz: [{state[0]:+.3f}, {state[1]:+.3f}, {state[2]:+.3f}] m   "
        "Images are decoded from LeRobot as stored; no 180-degree rotation applied.",
        font=small_font,
        fill=(184, 194, 204),
    )
    return canvas


def summarize_distribution(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "min": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset_root = args.dataset.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    key_dir = output_root / "keyframes"
    key_dir.mkdir(exist_ok=True)

    info_path = dataset_root / "meta/info.json"
    manifest_path = dataset_root / "data_manifest.json"
    acceptance_path = dataset_root / "round0_lerobot_acceptance.json"
    provenance_path = dataset_root / "per_action_provenance.jsonl"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    provenance = [json.loads(line) for line in provenance_path.read_text(encoding="utf-8").splitlines()]
    if len(provenance) != EXPECTED_FRAMES:
        raise RuntimeError(f"provenance_count_mismatch:{len(provenance)}")

    # Use the official LeRobot v2.1 Reader and preserve its image orientation.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    reader = LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)
    if len(reader) != EXPECTED_FRAMES or int(reader.num_episodes) != 1:
        raise RuntimeError(f"unexpected_dataset_shape:{len(reader)}:{reader.num_episodes}")
    if reader.meta.info.get("codebase_version") != "v2.1":
        raise RuntimeError("unexpected_lerobot_format_version")

    import imageio.v2 as imageio
    import imageio_ffmpeg

    video_path = output_root / "l0t2_round0_expert_dual_camera_20fps.mp4"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    writer = imageio.get_writer(
        str(video_path),
        format="FFMPEG",
        mode="I",
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        ffmpeg_log_level="error",
        output_params=["-crf", "18", "-preset", "medium"],
    )

    timestamps: list[float] = []
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    phases: list[str] = []
    segments: list[int] = []
    source_rows: list[int] = []
    agent_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    images_exact_duplicate_both: list[int] = []
    image_mae_agent: list[float] = []
    image_mae_wrist: list[float] = []
    row_alignment_errors: list[dict[str, Any]] = []
    keyframe_metadata: dict[str, Any] = {}
    try:
        for index in range(EXPECTED_FRAMES):
            item = reader[index]
            row = provenance[index]
            agent = as_hwc_u8(item["image"])
            wrist = as_hwc_u8(item["wrist_image"])
            action = np.asarray(item["actions"], dtype=np.float32).reshape(-1)
            state = np.asarray(item["state"], dtype=np.float32).reshape(-1)
            timestamp = float(np.asarray(item["timestamp"]).reshape(-1)[0])
            frame_index = int(json_scalar(item["frame_index"]))
            episode_index = int(json_scalar(item["episode_index"]))
            task_index = int(json_scalar(item["task_index"]))
            phase = str(row.get("task_phase", "unknown"))
            segment = int(row.get("source_segment_index", -1))
            action_row = int(row.get("source_action_row", -1))
            expected_post = index + 1
            checks = {
                "frame_index": frame_index == index,
                "episode_index": episode_index == 0,
                "provenance_dataset_frame_index": int(row.get("dataset_frame_index", -1)) == index,
                "global_pre_action_timestep": int(row.get("global_pre_action_timestep", -1)) == index,
                "global_post_action_timestep": int(row.get("global_post_action_timestep", -1)) == expected_post,
                "source_recorded_post_action_timestep": int(row.get("source_recorded_post_action_timestep", -1)) == expected_post,
                "timestamp_matches_20hz": abs(timestamp - index / FPS) <= 5e-6,
                "action_is_executed": row.get("action_is_recorded_executed_action") is True,
                "predicted_action_not_used": row.get("predicted_action_used_as_label") is False,
                "observation_is_pre_action": row.get("observation_camera_source", {}).get("pre_action") is True,
                "action_shape_7": action.shape == (7,),
                "state_shape_8": state.shape == (8,),
                "task_matches_acceptance": str(item["task"]) == acceptance.get("task"),
                "task_index_nonnegative": task_index >= 0,
            }
            if not all(checks.values()):
                row_alignment_errors.append({"timestep": index, "failed_checks": [k for k, v in checks.items() if not v]})
            if not np.all(np.isfinite(action)) or not np.all(np.isfinite(state)):
                row_alignment_errors.append({"timestep": index, "failed_checks": ["non_finite_action_or_state"]})

            if agent_frames:
                prev_agent = agent_frames[-1]
                prev_wrist = wrist_frames[-1]
                image_mae_agent.append(float(np.mean(np.abs(agent.astype(np.int16) - prev_agent.astype(np.int16)))))
                image_mae_wrist.append(float(np.mean(np.abs(wrist.astype(np.int16) - prev_wrist.astype(np.int16)))))
                if np.array_equal(agent, prev_agent) and np.array_equal(wrist, prev_wrist):
                    images_exact_duplicate_both.append(index)

            timestamps.append(timestamp)
            actions.append(action.copy())
            states.append(state.copy())
            phases.append(phase)
            segments.append(segment)
            source_rows.append(action_row)
            agent_frames.append(agent)
            wrist_frames.append(wrist)

            if index in KEY_TIMESTEPS:
                row_source_path = str(row.get("source_action_array_file", ""))
                frame = render_frame(
                    agent,
                    wrist,
                    timestep=index,
                    phase=phase,
                    source_segment=segment,
                    segment_start=int(row.get("source_segment_start_timestep", -1)),
                    segment_end=int(row.get("source_segment_end_timestep", -1)),
                    source_trace=Path(row_source_path).name if row_source_path else "source unavailable",
                    action=action,
                    state=state,
                    timestamp=timestamp,
                )
                key_path = key_dir / f"timestep_{index:04d}.png"
                frame.save(key_path)
                keyframe_metadata[str(index)] = {
                    "path": str(key_path),
                    "global_pre_action_timestep": index,
                    "global_post_action_timestep": expected_post,
                    "phase": phase,
                    "source_segment_index": segment,
                    "source_action_row": action_row,
                    "action_7d": action.tolist(),
                    "state_8d": state.tolist(),
                }
            rendered = render_frame(
                agent,
                wrist,
                timestep=index,
                phase=phase,
                source_segment=segment,
                segment_start=int(row.get("source_segment_start_timestep", -1)),
                segment_end=int(row.get("source_segment_end_timestep", -1)),
                source_trace=Path(str(row.get("source_action_array_file", ""))).name,
                action=action,
                state=state,
                timestamp=timestamp,
            )
            writer.append_data(np.asarray(rendered, dtype=np.uint8))
    finally:
        writer.close()

    action_array = np.stack(actions)
    state_array = np.stack(states)
    eef_delta = np.linalg.norm(np.diff(state_array[:, :3], axis=0), axis=1)
    action_translation_norm = np.linalg.norm(action_array[:, :3], axis=1)
    action_rotation_norm = np.linalg.norm(action_array[:, 3:6], axis=1)
    gripper_cmd = action_array[:, 6]
    gripper_unique, gripper_counts = np.unique(np.round(gripper_cmd, 6), return_counts=True)
    out_of_nominal_gripper = np.flatnonzero((gripper_cmd < -1.0) | (gripper_cmd > 1.0))

    phase_counts = dict(Counter(phases))
    source_counts = dict(Counter(str(x) for x in segments))
    source_action_keys = dict(Counter(str(row.get("source_action_key")) for row in provenance))
    source_hashes = dict(Counter(str(row.get("source_action_trace_sha256")) for row in provenance))
    segment_spans: list[dict[str, Any]] = []
    for segment in sorted(set(segments)):
        indices = [i for i, value in enumerate(segments) if value == segment]
        first, last = indices[0], indices[-1]
        row = provenance[first]
        local_rows = source_rows[first : last + 1]
        segment_spans.append(
            {
                "source_segment_index": int(segment),
                "dataset_timestep_start": int(first),
                "dataset_timestep_end_inclusive": int(last),
                "source_segment_start_timestep": int(row.get("source_segment_start_timestep", -1)),
                "source_segment_end_timestep_exclusive": int(row.get("source_segment_end_timestep", -1)),
                "source_rows_contiguous": local_rows == list(range(len(local_rows))),
                "source_action_key": row.get("source_action_key"),
                "source_action_array_file": row.get("source_action_array_file"),
                "source_action_trace_sha256": row.get("source_action_trace_sha256"),
                "row_count": int(len(indices)),
                "task_phases": dict(Counter(phases[first : last + 1])),
            }
        )
    for span in segment_spans:
        expected_length = span["source_segment_end_timestep_exclusive"] - span["source_segment_start_timestep"]
        span["dataset_length_matches_source_timestep_span"] = span["row_count"] == expected_length

    seam_reports: list[dict[str, Any]] = []
    seam_indices = [i for i, row in enumerate(provenance) if row.get("snapshot_seam_before_action")]
    for seam in seam_indices:
        row = provenance[seam]
        metadata = row["snapshot_seam_before_action"]
        prev_i = seam - 1
        seam_reports.append(
            {
                "timestep": seam,
                "previous_observation_timestep": prev_i,
                "current_observation_timestep": seam,
                "previous_phase": phases[prev_i] if prev_i >= 0 else None,
                "current_phase": phases[seam],
                "previous_source_segment": segments[prev_i] if prev_i >= 0 else None,
                "current_source_segment": segments[seam],
                "previous_action_7d": actions[prev_i].tolist() if prev_i >= 0 else None,
                "current_action_7d": actions[seam].tolist(),
                "eef_xyz_delta_between_observations_m": float(eef_delta[prev_i]) if prev_i >= 0 else None,
                "action_translation_norm_previous": float(action_translation_norm[prev_i]) if prev_i >= 0 else None,
                "action_translation_norm_current": float(action_translation_norm[seam]),
                "agentview_pixel_mae_across_transition": image_mae_agent[prev_i] if prev_i >= 0 else None,
                "wrist_pixel_mae_across_transition": image_mae_wrist[prev_i] if prev_i >= 0 else None,
                "agentview_change_above_episode_p99": bool(
                    prev_i >= 0 and image_mae_agent[prev_i] > np.percentile(image_mae_agent, 99)
                ),
                "wrist_change_above_episode_p99": bool(
                    prev_i >= 0 and image_mae_wrist[prev_i] > np.percentile(image_mae_wrist, 99)
                ),
                "manual_review_candidate_due_to_seam_visual_change": bool(
                    prev_i >= 0
                    and (
                        image_mae_agent[prev_i] > np.percentile(image_mae_agent, 99)
                        or image_mae_wrist[prev_i] > np.percentile(image_mae_wrist, 99)
                    )
                ),
                "snapshot_seam_provenance": metadata,
                "visual_transition_is_not_automatically_corruption": True,
            }
        )

    # Flag only unusually large image changes and suspicious exact duplicate pairs.
    def robust_candidates(values: list[float]) -> list[dict[str, float | int]]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return []
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        threshold = max(float(np.percentile(arr, 99.5)), median + 8.0 * max(mad, 1e-6), 25.0)
        return [
            {"transition_to_timestep": int(i + 1), "mean_absolute_pixel_change": float(value)}
            for i, value in enumerate(arr)
            if value >= threshold
        ]

    suspicious_duplicate = []
    for timestep in images_exact_duplicate_both:
        if timestep <= 0:
            continue
        pos_delta = float(eef_delta[timestep - 1])
        translation_action = float(action_translation_norm[timestep - 1])
        if pos_delta > 0.005 or translation_action > 0.005:
            suspicious_duplicate.append(
                {
                    "timestep": timestep,
                    "previous_to_current_eef_delta_m": pos_delta,
                    "previous_action_translation_norm": translation_action,
                    "classification": "review_candidate_visual_duplicate_with_state_or_action_change",
                }
            )

    near_duplicate_review_candidates = []
    for timestep in range(1, len(agent_frames)):
        agent_mae = image_mae_agent[timestep - 1]
        wrist_mae = image_mae_wrist[timestep - 1]
        pos_delta = float(eef_delta[timestep - 1])
        translation_action = float(action_translation_norm[timestep - 1])
        if agent_mae <= 0.25 and wrist_mae <= 0.25 and (pos_delta > 0.005 or translation_action > 0.05):
            near_duplicate_review_candidates.append(
                {
                    "transition_to_timestep": timestep,
                    "agentview_mae": agent_mae,
                    "wrist_mae": wrist_mae,
                    "eef_delta_m": pos_delta,
                    "previous_action_translation_norm": translation_action,
                    "classification": "review_candidate_near_duplicate_with_state_or_action_change",
                }
            )

    alignment_pass = not row_alignment_errors and len(actions) == EXPECTED_FRAMES
    action_gripper_semantics = {
        "source": "/workspace/zy_workspace/LIBERO-Safety/third_party/robosuite-1.4/robosuite/models/grippers/panda_gripper.py",
        "format_action_rule": "PandaGripper accumulates sign(action) into current_action; -1 maps open, +1 maps closed, 0 holds current command",
        "dataset_gripper_action_min": float(np.min(gripper_cmd)),
        "dataset_gripper_action_max": float(np.max(gripper_cmd)),
        "nominal_command_range": [-1.0, 1.0],
        "outside_nominal_range_count": int(len(out_of_nominal_gripper)),
        "outside_nominal_range_rows": [
            {
                "timestep": int(i),
                "raw_command": float(gripper_cmd[i]),
                "phase": phases[int(i)],
                "source_segment_index": int(segments[int(i)]),
                "classification": "small_negative_open_command_overshoot_preserved_from_source",
            }
            for i in out_of_nominal_gripper
        ],
        "negative_open_command_count": int(np.sum(gripper_cmd < -1e-6)),
        "zero_hold_command_count": int(np.sum(np.abs(gripper_cmd) <= 1e-6)),
        "positive_close_command_count": int(np.sum(gripper_cmd > 1e-6)),
        "rounded_value_histogram": {str(float(v)): int(c) for v, c in zip(gripper_unique, gripper_counts)},
        "observed_finger_qpos_is_shown_without_binary_inference": True,
        "finger_qpos_sum_abs_min": float(np.min(np.sum(np.abs(state_array[:, 6:8]), axis=1))),
        "finger_qpos_sum_abs_median": float(np.median(np.sum(np.abs(state_array[:, 6:8]), axis=1))),
        "finger_qpos_sum_abs_max": float(np.max(np.sum(np.abs(state_array[:, 6:8]), axis=1))),
        "interpretation_of_out_of_nominal_range_rows": "The source action label is retained exactly. The inspected PandaGripper format_action uses sign(action), so these slightly-below-minus-one values still select the open command; this report flags them instead of silently clipping training data.",
    }

    # Also create a compact review contact sheet from the eight requested frames.
    contact_sheet_path = output_root / "keyframes_contact_sheet.jpg"
    contact_sheet = Image.new("RGB", (1024, 1360), (24, 28, 33))
    for position, timestep in enumerate(KEY_TIMESTEPS):
        tile = Image.open(key_dir / f"timestep_{timestep:04d}.png").convert("RGB")
        tile = tile.resize((512, 340), Image.Resampling.LANCZOS)
        contact_sheet.paste(tile, ((position % 2) * 512, (position // 2) * 340))
    contact_sheet.save(contact_sheet_path, quality=92)

    seam_review_candidates = [
        seam["timestep"]
        for seam in seam_reports
        if seam["manual_review_candidate_due_to_seam_visual_change"]
    ]
    seam_review_dir = output_root / "snapshot_seam_review"
    seam_review_dir.mkdir(exist_ok=True)
    seam_review_frames: list[Image.Image] = []
    for seam in seam_review_candidates:
        for timestep in (seam - 1, seam):
            row = provenance[timestep]
            rendered = render_frame(
                agent_frames[timestep],
                wrist_frames[timestep],
                timestep=timestep,
                phase=phases[timestep],
                source_segment=segments[timestep],
                segment_start=int(row.get("source_segment_start_timestep", -1)),
                segment_end=int(row.get("source_segment_end_timestep", -1)),
                source_trace=Path(str(row.get("source_action_array_file", ""))).name,
                action=actions[timestep],
                state=states[timestep],
                timestamp=timestamps[timestep],
            )
            rendered.save(seam_review_dir / f"timestep_{timestep:04d}.png")
            seam_review_frames.append(rendered.resize((512, 356), Image.Resampling.LANCZOS))
    seam_review_sheet = None
    if seam_review_frames:
        columns = 2
        rows = math.ceil(len(seam_review_frames) / columns)
        seam_review_sheet = output_root / "snapshot_seam_review_contact_sheet.jpg"
        sheet = Image.new("RGB", (columns * 512, rows * 356), (24, 28, 33))
        for index, frame in enumerate(seam_review_frames):
            sheet.paste(frame, ((index % columns) * 512, (index // columns) * 356))
        sheet.save(seam_review_sheet, quality=92)

    video_reader = imageio.get_reader(str(video_path), format="FFMPEG")
    try:
        decoded_video_frames = sum(1 for _ in video_reader)
        video_metadata = video_reader.get_meta_data()
    finally:
        video_reader.close()

    report = {
        "schema": "gs_opd_l0t2_round0_lerobot_visual_audit.v1",
        "status": "completed_read_only_visual_and_semantic_audit",
        "dataset_path": str(dataset_root),
        "output_path": str(output_root),
        "lerobot_reader": {
            "reader": "lerobot.datasets.lerobot_dataset.LeRobotDataset",
            "package_version": importlib.metadata.version("lerobot"),
            "format_version": reader.meta.info.get("codebase_version"),
            "episode_count": int(reader.num_episodes),
            "frame_count": int(len(reader)),
            "fps": FPS,
            "duration_seconds": EXPECTED_FRAMES / FPS,
            "stored_image_shape": [256, 256, 3],
            "reader_tensor_layout": "CHW float [0,1], converted to HWC uint8 for display",
            "image_rotation_applied": False,
            "image_resized_for_video_display_only": True,
        },
        "evidence_classes": {
            "visual_inspection": "A human-review video and key frames were rendered from LeRobot Reader images; this does not itself certify physical correctness.",
            "source_data_consistency": "Per-frame dataset/provenance indices and source segment metadata were checked; referenced source archive contents were not re-opened in this read-only dataset audit.",
            "observation_action_semantics": "Verified pre-action observation metadata, t -> t+1 action time indices, recorded executed-action flag, finite 7D action/state shapes and 20 Hz timestamps for every row.",
        },
        "checks": {
            "acceptance_status": acceptance.get("status"),
            "manifest_frame_count": manifest.get("total_action_samples"),
            "reader_count_matches_manifest_and_acceptance": len(reader) == manifest.get("total_action_samples") == acceptance.get("sample_count"),
            "one_episode_only": int(reader.num_episodes) == 1,
            "dataset_episode_boundaries": acceptance.get("episode_boundaries"),
            "provenance_rows": len(provenance),
            "all_rows_alignment_pass": alignment_pass,
            "row_alignment_error_count": len(row_alignment_errors),
            "row_alignment_errors": row_alignment_errors[:100],
            "task_consistent": all(str(row.get("task_instruction")) == acceptance.get("task") for row in provenance),
            "phase_count_consistent": phase_counts == acceptance.get("training_windows", {}).get("action_counts_by_phase"),
            "source_segment_count": len(segment_spans),
            "source_segments": segment_spans,
            "snapshot_seam_count": len(seam_reports),
            "snapshot_seams": seam_reports,
            "timestamp_max_error_seconds": float(max(abs(t - i / FPS) for i, t in enumerate(timestamps))),
            "last_dataset_frame": {
                "dataset_frame_index": EXPECTED_FRAMES - 1,
                "global_pre_action_timestep": int(provenance[-1]["global_pre_action_timestep"]),
                "global_post_action_timestep": int(provenance[-1]["global_post_action_timestep"]),
                "is_terminal_observation": False,
            },
            "terminal_archive_evidence": {
                "terminal_timestep": 1460,
                "native_task_success": acceptance.get("source_trajectory", {}).get("native_task_success"),
                "reward": acceptance.get("source_trajectory", {}).get("reward"),
                "done": acceptance.get("source_trajectory", {}).get("done"),
                "included_as_training_frame_or_video_frame": False,
                "provenance_note": "t=1459 stores the pre-action observation and executed action that transitions to t=1460; no t=1460 terminal image is present in this LeRobot episode.",
            },
        },
        "actions_and_gripper": {
            "action_semantics": "Actual executed 7D OSC_POSE action labels from LeRobot; no denormalization or action reconstruction performed.",
            "action_min": action_array.min(axis=0).tolist(),
            "action_max": action_array.max(axis=0).tolist(),
            "action_all_finite": bool(np.all(np.isfinite(action_array))),
            "translation_norm": summarize_distribution(action_translation_norm.tolist()),
            "rotation_norm": summarize_distribution(action_rotation_norm.tolist()),
            "eef_pre_action_delta_between_adjacent_observations_m": summarize_distribution(eef_delta.tolist()),
            "gripper": action_gripper_semantics,
        },
        "visual_change_review_candidates": {
            "adjacent_agentview_mae_0_255": summarize_distribution(image_mae_agent),
            "adjacent_wrist_mae_0_255": summarize_distribution(image_mae_wrist),
            "large_change_threshold_is_robust_99_5_percentile_or_median_plus_8_mad_min_25": True,
            "agentview_large_change_candidates": robust_candidates(image_mae_agent),
            "wrist_large_change_candidates": robust_candidates(image_mae_wrist),
            "exact_duplicate_both_camera_transition_count": len(images_exact_duplicate_both),
            "exact_duplicate_both_camera_transition_timesteps": images_exact_duplicate_both,
            "duplicate_with_state_or_action_change_review_candidates": suspicious_duplicate,
            "near_duplicate_with_state_or_action_change_review_candidates": near_duplicate_review_candidates,
            "snapshot_seam_visual_change_review_candidates": [
                {
                    "timestep": seam["timestep"],
                    "agentview_pixel_mae": seam["agentview_pixel_mae_across_transition"],
                    "wrist_pixel_mae": seam["wrist_pixel_mae_across_transition"],
                    "above_episode_p99_camera_change": {
                        "agentview": seam["agentview_change_above_episode_p99"],
                        "wrist": seam["wrist_change_above_episode_p99"],
                    },
                    "snapshot_state_compatibility": seam["snapshot_seam_provenance"].get("state_level_compatibility"),
                    "classification": "source_seam_for_manual_visual_review_not_automatic_corruption",
                }
                for seam in seam_reports
                if seam["manual_review_candidate_due_to_seam_visual_change"]
            ],
            "interpretation": "Candidates are prompts for human review. Real object contact, arm motion, lighting/view changes, and source transitions can produce large image deltas and are not labelled corruption automatically.",
        },
        "keyframes": keyframe_metadata,
        "artifacts": {
            "video": str(video_path),
            "video_fps": FPS,
            "video_frame_count": EXPECTED_FRAMES,
            "video_duration_seconds": EXPECTED_FRAMES / FPS,
            "video_decode_validation": {
                "decoded_frame_count": int(decoded_video_frames),
                "decoded_fps": float(video_metadata.get("fps", 0.0)),
                "decoded_dimensions": list(video_metadata.get("size", ())),
                "codec": video_metadata.get("codec"),
                "pass": decoded_video_frames == EXPECTED_FRAMES and abs(float(video_metadata.get("fps", 0.0)) - FPS) < 1e-6,
            },
            "video_sha256": sha256_file(video_path),
            "video_size_bytes": video_path.stat().st_size,
            "keyframe_directory": str(key_dir),
            "keyframe_count": len(keyframe_metadata),
            "keyframe_contact_sheet": str(contact_sheet_path),
            "snapshot_seam_review_directory": str(seam_review_dir),
            "snapshot_seam_review_timesteps": [
                timestep for seam in seam_review_candidates for timestep in (seam - 1, seam)
            ],
            "snapshot_seam_review_contact_sheet": str(seam_review_sheet) if seam_review_sheet else None,
            "source_dataset_files_sha256": {
                str(path.relative_to(dataset_root)): sha256_file(path)
                for path in (info_path, manifest_path, acceptance_path, provenance_path, dataset_root / "data/chunk-000/episode_000000.parquet")
            },
        },
    }
    report_path = output_root / "l0t2_round0_lerobot_visual_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path = output_root / "l0t2_round0_lerobot_visual_audit.md"
    summary_path.write_text(
        "# L0T2 Round 0 LeRobot episode visual audit\n\n"
        f"- Read-only source: `{dataset_root}`\n"
        f"- Official LeRobot Reader: v{report['lerobot_reader']['package_version']}, "
        f"{len(reader)} frames, {FPS} FPS, {EXPECTED_FRAMES / FPS:.1f} s, one Episode.\n"
        f"- Video: `{video_path}`\n"
        f"- Key frames: `{key_dir}` (" + ", ".join(str(t) for t in KEY_TIMESTEPS) + ")\n"
        f"- Alignment: {'PASS' if alignment_pass else 'FAIL'}; "
        f"{len(row_alignment_errors)} row issues; {len(seam_reports)} Snapshot seams audited.\n"
        f"- Terminal t=1460 is listed only as provenance evidence, not rendered as a training frame.\n"
        f"- Gripper convention from robosuite PandaGripper: negative=open, positive=close, zero=hold; "
        f"command range [{np.min(gripper_cmd):.3f}, {np.max(gripper_cmd):.3f}].\n"
        f"- Exact adjacent duplicate pairs in both cameras: {len(images_exact_duplicate_both)}; "
        f"near/exact duplicates are review candidates, not automatic corruption labels.\n"
        f"- Source-seam camera-change review candidates: "
        + ", ".join(str(x["timestep"]) for x in report["visual_change_review_candidates"]["snapshot_seam_visual_change_review_candidates"])
        + ".\n\n"
        "Visual appearance, source/provenance consistency, and observation-action time semantics are reported as separate evidence classes in the JSON.\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "video": str(video_path), "keyframes": str(key_dir), "frames": EXPECTED_FRAMES}, ensure_ascii=False))


if __name__ == "__main__":
    main()
