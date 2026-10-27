# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Bounded OA L0T0 placement branches from an exact candidate9 held-place snapshot.

This is task-feasibility exploration. Strict safety evidence is preserved for
every real action and never converted into a safety label. The two supported
task interactions are bowl-to-plate support and gripper opening after reaching
the placement pose; every other physical contact remains a hard stop.
The branch planner maps object-local placement Keyposes through the measured
hand-to-object transform, then uses CuRobo and active MuJoCo path audits before
each closed-loop physical stage. Safety monitoring remains unchanged.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.gs_opd.libero_safety_repro import plan_oa_l0t0 as api
from research.gs_opd.libero_safety_repro import transport_candidate9_above_plate as prior


MODEL_SHA = prior.EXPECTED_MODEL_SHA
OBJECT_ID = prior.TARGET_OBJECT_ID
REGION_ID = prior.TARGET_REGION_ID
INSTRUCTION = prior.INSTRUCTION
SOURCE_SCENE = prior.DEFAULT_SOURCE_SCENE
SOURCE_SNAPSHOT = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "oa_l0t0_candidate9_transport_above_plate_20261027_attempt06/terminal_runtime_snapshot.pkl"
)
PRIOR_TRANSPORT_DIR = SOURCE_SNAPSHOT.parent
HORIZON = 1000
# A bounded 5-step tracking window retains the existing 0.08 rad hard stop;
# attempt 3 plateaued at 0.0376 rad after only three feedback actions.
MAX_ACTIONS_PER_WAYPOINT = 5
MAX_ENDPOINT_FEEDBACK_ACTIONS = 16
MAX_RELEASE_OPEN_ACTIONS = 120
MAX_POST_RELEASE_CLEARANCE_OPEN_ACTIONS = 75
SUBSTEPS = 25
SETTLE_ACTIONS = 12
OPEN_ACTION = -1.0
SEARCH_CONTACT_MAX_PENETRATION_M = 0.0001
SEARCH_CONTACT_ROBOT_GEOM_BODY_PAIRS = {
    ("gripper0_finger1_collision", "gripper0_leftfinger"),
    ("gripper0_finger2_collision", "gripper0_rightfinger"),
}


class _PythonObjectState:
    """Legacy runtime snapshot pickle marker."""


class _Bridge:
    def __init__(self, connection):
        self.connection = connection

    def request(self, request: dict[str, Any]) -> dict[str, Any]:
        return api._rpc(self.connection, request)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, row: Any) -> None:
    path.write_text(json.dumps(api._json_safe(row), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _object_pose(context: dict[str, Any]) -> np.ndarray:
    return prior._object_pose(context)


def _world_pose(context: dict[str, Any], key: str) -> np.ndarray:
    return prior._frame_pose(context[key])


def _strict_contacts(reply: dict[str, Any]) -> dict[str, Any]:
    """Separate actual illegal contacts from unchanged strict-margin findings."""
    raw = reply.get("substep_contacts", [])
    monitor = reply.get("contact_monitoring", {})
    geometry = monitor.get("geometry_monitoring", {})
    geom_steps = geometry.get("substeps", [])
    complete = bool(
        monitor.get("expected_physics_substeps_per_control_step") == SUBSTEPS
        and monitor.get("captured_physics_substeps") == SUBSTEPS
        and monitor.get("unavailable_contact_substeps") == 0
        and len(raw) == SUBSTEPS
        and len(geom_steps) == SUBSTEPS
        and all(row.get("status") in {"complete", "unsafe"} for row in geom_steps)
    )
    self_contacts = []
    unallowed_robot_scene = []
    forbidden_objects = []
    unresolved = []
    margin_only = []
    forbidden_robot_contacts = []
    joint_limit_violations = []
    actual_contacts = []
    for index, row in enumerate(geom_steps):
        actual_contacts.extend({"substep": index, **x} for x in row.get("contact_identities", []))
        self_contacts.extend({"substep": index, **x} for x in row.get("self_contacts", []))
        unallowed_robot_scene.extend({"substep": index, **x} for x in row.get("unallowed_scene_contacts", []))
        forbidden_objects.extend({"substep": index, **x} for x in row.get("forbidden_object_contacts", []))
        unresolved.extend({"substep": index, **x} for x in row.get("unresolved_contacts", []))
        joint_limit_violations.extend({"substep": index, **x} for x in row.get("joint_limit_violations", []))
        for item in row.get("forbidden_contacts", []):
            if "contact_index" in item and "geom1" in item and "geom2" in item:
                forbidden_robot_contacts.append({"substep": index, **item})
            else:
                margin_only.append({"substep": index, **item})
    actual_hard = bool(self_contacts or unallowed_robot_scene or forbidden_objects or unresolved or forbidden_robot_contacts or joint_limit_violations)
    return {
        "complete_25_substep_monitoring": complete,
        "geometry_status": geometry.get("status"),
        "actual_self_contacts": self_contacts,
        "actual_unallowed_robot_scene_contacts": unallowed_robot_scene,
        "actual_forbidden_object_contacts": forbidden_objects,
        "actual_contact_identities": actual_contacts,
        "unresolved_contacts": unresolved,
        "forbidden_robot_contacts": forbidden_robot_contacts,
        "joint_limit_violations": joint_limit_violations,
        "strict_margin_findings": margin_only,
        "actual_hard_contact_present": actual_hard,
        "strict_margin_compliant": bool(complete and geometry.get("status") == "complete" and not margin_only),
        "safety_certifiable_action": bool(complete and not actual_hard and not margin_only and geometry.get("status") == "complete"),
    }


def _point3(context: dict[str, Any], key: str) -> np.ndarray:
    row = context[key]
    out = np.asarray(row["position_m"], dtype=np.float64)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError(f"bad_live_position:{key}")
    return out


def _object_velocity(snapshot_state: dict[str, Any]) -> dict[str, Any]:
    row = snapshot_state.get("object_states", {}).get(f"objects_dict:{OBJECT_ID}")
    if not isinstance(row, dict):
        return {"status": "unknown", "linear_speed_m_s": None, "angular_speed_rad_s": None}
    qvel = np.asarray(row.get("joint_qvel", []), dtype=np.float64)
    if qvel.shape != (6,) or not np.isfinite(qvel).all():
        return {"status": "unknown", "linear_speed_m_s": None, "angular_speed_rad_s": None}
    return {
        "status": "complete",
        "linear_velocity_m_s": qvel[:3].tolist(),
        "angular_velocity_rad_s": qvel[3:].tolist(),
        "linear_speed_m_s": float(np.linalg.norm(qvel[:3])),
        "angular_speed_rad_s": float(np.linalg.norm(qvel[3:])),
    }


def _placement_object_pose(
    context: dict[str, Any],
    collision_scene: dict[str, Any],
    semantic: dict[str, Any],
    world_base: np.ndarray,
    *,
    xy_fraction: tuple[float, float] = (0.0, 0.0),
    yaw_delta_deg: float = 0.0,
    bottom_clearance_m: float = 0.001,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an object-to-plate-region placement transform from live geometry.

    Fractions are normalized within the collision-geometry-derived remaining
    bowl fit on the plate.  They and the optional yaw are experiment choices,
    not values reported by LIBERO-Safety.
    """
    binding = semantic["bindings"].get(REGION_ID)
    if not isinstance(binding, dict):
        raise ValueError("active_plate_semantic_binding_missing")
    body_id, body_name = int(binding["root_body_id"]), str(binding["root_body_name"])
    rows = [r for r in collision_scene.get("scene_collision_geometries", []) if int(r.get("body_id", -1)) == body_id and r.get("body_name") == body_name]
    if not rows or any(r.get("shape") != "cuboid" for r in rows):
        raise ValueError("compiled_plate_collision_geometry_missing_or_unsupported")
    vertices = []
    for row in rows:
        center = np.asarray(row["center_world_m"], dtype=np.float64)
        rot = np.asarray(row["rotation_world"], dtype=np.float64)
        half = np.asarray(row["half_extents_world_axes_m"], dtype=np.float64)
        vertices.extend(prior._box_corners(center, rot, half))
    verts = np.asarray(vertices)
    support_candidates = []
    for row in rows:
        row_rotation = np.asarray(row["rotation_world"], dtype=np.float64)
        row_half = np.asarray(row["half_extents_world_axes_m"], dtype=np.float64)
        if row_rotation.shape != (3, 3) or not np.isfinite(row_rotation).all() or not np.allclose(row_rotation.T @ row_rotation, np.eye(3), atol=1e-5):
            continue
        thin_axis = int(np.argmin(row_half))
        normal = row_rotation[:, thin_axis].copy()
        if normal[2] < 0.0:
            normal *= -1.0
        planar_area = float(np.prod(np.delete(row_half, thin_axis)))
        if normal[2] >= 0.95:
            support_candidates.append((planar_area, row, thin_axis, normal))
    if not support_candidates:
        raise ValueError("active_plate_horizontal_support_collision_geom_unavailable")
    _, support_geom, thin_axis, plate_normal = max(support_candidates, key=lambda item: item[0])
    support_rotation = np.asarray(support_geom["rotation_world"], dtype=np.float64)
    support_half = np.asarray(support_geom["half_extents_world_axes_m"], dtype=np.float64)
    planar_axes = [axis for axis in range(3) if axis != thin_axis]
    x_axis = support_rotation[:, max(planar_axes, key=lambda axis: support_half[axis])].copy()
    x_axis -= plate_normal * float(np.dot(x_axis, plate_normal))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(plate_normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    plate_rotation = np.column_stack((x_axis, y_axis, plate_normal))
    support_center = np.asarray(support_geom["center_world_m"], dtype=np.float64)
    plate_origin = support_center + plate_normal * support_half[thin_axis]
    plate_local_vertices = (verts - plate_origin) @ plate_rotation
    plate_lower, plate_upper = plate_local_vertices.min(axis=0), plate_local_vertices.max(axis=0)
    plate_xy_center = (plate_lower[:2] + plate_upper[:2]) / 2.0
    # Geom 439 is the largest horizontal support slab. Its top face defines
    # the support plane; the surrounding rim is not the bowl's landing height.
    plate_top = 0.0
    context_pose = _object_pose(context)
    xy_fraction = np.asarray(xy_fraction, dtype=np.float64)
    if xy_fraction.shape != (2,) or not np.isfinite(xy_fraction).all() or np.any(np.abs(xy_fraction) > 0.85):
        raise ValueError("placement_xy_fraction_must_be_finite_and_within_0.85")
    if not np.isfinite(yaw_delta_deg) or not np.isfinite(bottom_clearance_m) or bottom_clearance_m < 0.0:
        raise ValueError("placement_yaw_or_clearance_invalid")
    yaw_rotation = Rotation.from_rotvec(plate_rotation[:, 2] * np.deg2rad(yaw_delta_deg)).as_matrix()
    object_rot = yaw_rotation @ context_pose[:3, :3]
    offsets = []
    min_bottom_offset = np.inf
    bowl_rows = context.get("target_collision_geometries", [])
    if not bowl_rows:
        raise ValueError("live_bowl_collision_geometry_missing")
    for row in bowl_rows:
        half = np.asarray(row["size_m"], dtype=np.float64)
        corners = prior._box_corners(np.asarray(row["center_root_m"], dtype=np.float64), np.asarray(row["rotation_root"], dtype=np.float64), half)
        world_offsets = corners @ object_rot.T
        region_offsets = world_offsets @ plate_rotation
        min_bottom_offset = min(min_bottom_offset, float(np.min(region_offsets[:, 2])))
        offsets.extend(region_offsets)
    offsets = np.asarray(offsets)
    bmin, bmax = offsets[:, :2].min(axis=0), offsets[:, :2].max(axis=0)
    remaining = np.minimum(plate_upper[:2] - plate_xy_center - bmax, plate_xy_center + bmin - plate_lower[:2])
    if not np.isfinite(remaining).all() or np.any(remaining < -1e-6):
        raise ValueError("live_bowl_geometry_does_not_fit_plate_collision_frame")
    remaining = np.maximum(remaining, 0.0)
    xy_local = plate_xy_center + xy_fraction * remaining
    pose = context_pose.copy()
    pose[:3, :3] = object_rot
    target_local = np.asarray([xy_local[0], xy_local[1], plate_top + bottom_clearance_m - min_bottom_offset])
    pose[:3, 3] = plate_origin + plate_rotation @ target_local
    frame = np.eye(4, dtype=np.float64)
    frame[:3, :3] = plate_rotation
    frame[:3, 3] = plate_origin
    selected_world = pose[:3, 3].copy()
    return pose, {
        "target_region_id": REGION_ID,
        "target_body_id": body_id,
        "target_body_name": body_name,
        "target_collision_geom_ids": sorted(int(r["geom_id"]) for r in rows),
        "support_surface_geom_id": int(support_geom["geom_id"]),
        "plate_collision_frame_world_m": frame.tolist(),
        "plate_collision_bounds_in_frame_m": {"min": plate_lower.tolist(), "max": plate_upper.tolist()},
        "plate_top_in_frame_m": plate_top,
        "remaining_bowl_center_shift_in_frame_m": remaining.tolist(),
        "selected_xy_fraction_within_fit": xy_fraction.tolist(),
        "selected_object_root_position_world_m": selected_world.tolist(),
        "yaw_delta_about_region_normal_deg": float(yaw_delta_deg),
        "bowl_collision_bottom_offset_region_z_m": min_bottom_offset,
        "requested_bowl_bottom_clearance_m": float(bottom_clearance_m),
        "engineering_choices_not_paper_values": True,
        "target_object_root_pose_world_m": pose.tolist(),
    }


def _build_policy(context, semantic, inventory):
    return prior._collision_policy(context, semantic, inventory)


def _released_safety_policy_for_bridge(policy: dict[str, Any], *, release_transition: bool = False) -> dict[str, Any]:
    """Apply unchanged scene margins after grasp loss, optionally during opening.

    The transition variant keeps only the already-defined exact pad/bowl pairs
    while the open command continues. It has no carried-object binding and does
    not certify release; the fully released variant removes those pairs.
    """
    released = copy.deepcopy(prior._safety_policy_for_bridge(policy))
    released["carried_object"] = None
    if not release_transition:
        released["allowed_contact_pairs"] = []
    released["allowed_carried_support_contact_geom_ids"] = []
    return released


def _target_robot_contacts(evidence: dict[str, Any], bowl_geom_ids: set[int], robot_geom_ids: set[int], *, substep: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in evidence.get("actual_contact_identities", []):
        if substep is not None and row.get("substep") != substep:
            continue
        geom1 = row.get("geom1", {}).get("id")
        geom2 = row.get("geom2", {}).get("id")
        if (geom1 in bowl_geom_ids and geom2 in robot_geom_ids) or (geom2 in bowl_geom_ids and geom1 in robot_geom_ids):
            rows.append(row)
    return rows


def _physical_release_confirmed(grasp_state: dict[str, Any], evidence: dict[str, Any], bowl_geom_ids: set[int], robot_geom_ids: set[int]) -> bool:
    fingers = grasp_state.get("finger_contacts", {})
    pad_separated = bool(
        fingers.get("left") is False
        and fingers.get("right") is False
        and int(fingers.get("contact_count", -1)) == 0
    )
    return bool(
        grasp_state.get("grasp_confirmed") is not True
        and pad_separated
        and evidence.get("complete_25_substep_monitoring") is True
        and not _target_robot_contacts(evidence, bowl_geom_ids, robot_geom_ids, substep=SUBSTEPS - 1)
    )


def _only_expected_plate_support(evidence: dict[str, Any], bowl_geom_ids: set[int], plate_geom_ids: set[int]) -> bool:
    """Task-feasibility exception for exact bowl/plate support; always unsafe."""
    hard_rows = (
        evidence.get("actual_self_contacts", [])
        + evidence.get("actual_unallowed_robot_scene_contacts", [])
        + evidence.get("unresolved_contacts", [])
        + evidence.get("forbidden_robot_contacts", [])
        + evidence.get("joint_limit_violations", [])
    )
    if hard_rows:
        return False
    object_rows = evidence.get("actual_forbidden_object_contacts", [])
    if not object_rows:
        return True
    for row in object_rows:
        a = row.get("geom1", {}).get("id")
        b = row.get("geom2", {}).get("id")
        if not ((a in bowl_geom_ids and b in plate_geom_ids) or (b in bowl_geom_ids and a in plate_geom_ids)):
            return False
    return True


def _task_search_contact_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    rows = evidence.get("forbidden_robot_contacts", [])
    pairs: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        a, b = row.get("geom1", {}), row.get("geom2", {})
        key = tuple(sorted((
            (a.get("id"), a.get("name"), row.get("body1", {}).get("name")),
            (b.get("id"), b.get("name"), row.get("body2", {}).get("name")),
        ), key=lambda item: str(item)))
        pairs.setdefault(key, []).append(row)
    unique = []
    for key, contacts in sorted(pairs.items(), key=lambda item: str(item[0])):
        distances = [float(row["signed_contact_distance_m"]) for row in contacts if row.get("signed_contact_distance_m") is not None]
        unique.append({
            "geom_pair": key,
            "contact_record_count": len(contacts),
            "sampled_substeps": sorted({int(row["substep"]) for row in contacts if row.get("substep") is not None}),
            "deepest_signed_distance_m": min(distances) if distances else None,
            "maximum_penetration_m": max((max(0.0, -distance) for distance in distances), default=None),
        })
    return {"contact_record_count": len(rows), "unique_pair_count": len(unique), "pairs": unique}


def _task_feasibility_search_contact(
    evidence: dict[str, Any], bowl_geom_ids: set[int], phase: str, *, allow_open_release: bool = False,
    allow_expected_plate_support: bool = False, plate_geom_ids: set[int] | None = None,
) -> bool:
    """Continue only this task-search branch through its recorded shallow contact.

    This is deliberately not a safety exception: the unchanged monitor still
    classifies the non-pad geom pair as forbidden and the episode remains
    uncertifiable. All scene, self, unresolved, joint-limit, and other object
    contacts remain hard stops. During release, this continuation is limited to
    actual negative gripper commands.
    """
    phase_root = phase.split("_endpoint_feedback", 1)[0]
    if phase_root not in {"vertical_clear", "above_plate", "place", "retreat"} and not (phase == "release_open" and allow_open_release):
        return False
    for field in (
        "actual_self_contacts",
        "actual_unallowed_robot_scene_contacts",
        "unresolved_contacts",
        "joint_limit_violations",
    ):
        if evidence.get(field):
            return False
    object_contacts = evidence.get("actual_forbidden_object_contacts", [])
    if object_contacts:
        if not allow_expected_plate_support or not plate_geom_ids:
            return False
        for row in object_contacts:
            geom_ids = {row.get("geom1", {}).get("id"), row.get("geom2", {}).get("id")}
            if not (geom_ids & bowl_geom_ids and geom_ids & plate_geom_ids):
                return False
    rows = evidence.get("forbidden_robot_contacts", [])
    if not rows:
        return False
    for row in rows:
        geom1, geom2 = row.get("geom1", {}), row.get("geom2", {})
        body1, body2 = row.get("body1", {}), row.get("body2", {})
        bowl_side = next((geom for geom, body in ((geom1, body1), (geom2, body2)) if geom.get("id") in bowl_geom_ids), None)
        robot_side = next(((geom, body) for geom, body in ((geom1, body1), (geom2, body2)) if geom.get("id") not in bowl_geom_ids), None)
        distance = row.get("signed_contact_distance_m")
        if (
            bowl_side is None
            or robot_side is None
            or (robot_side[0].get("name"), robot_side[1].get("name")) not in SEARCH_CONTACT_ROBOT_GEOM_BODY_PAIRS
            or distance is None
            or float(distance) < -SEARCH_CONTACT_MAX_PENETRATION_M
        ):
            return False
    return True


def _task_feasibility_path_contact(audit: dict[str, Any], bowl_geom_ids: set[int], phase: str) -> dict[str, Any]:
    """Classify exact carried-bowl/finger-shell path contacts for task search only."""
    if phase not in {"vertical_clear", "above_plate", "place", "retreat"}:
        return {"continue_task_search": False, "reason": "phase_not_carrying_target"}
    states = audit.get("states", [])
    rows = [row for state in states for row in state.get("illegal_contacts", [])]
    if not rows:
        return {"continue_task_search": True, "unsafe_contact_present": False, "illegal_contact_count": 0}
    if int(audit.get("illegal_contact_count", -1)) != len(rows):
        return {"continue_task_search": False, "reason": "path_audit_contact_rows_incomplete"}
    for row in rows:
        geom_ids = row.get("geom_ids", [])
        geom_names = row.get("geom_names", [])
        distance = row.get("distance_m")
        if (
            len(geom_ids) != 2
            or len(geom_names) != 2
            or not any(int(geom_id) in bowl_geom_ids for geom_id in geom_ids)
            or not any(pair[0] in geom_names for pair in SEARCH_CONTACT_ROBOT_GEOM_BODY_PAIRS)
            or distance is None
            or float(distance) < -SEARCH_CONTACT_MAX_PENETRATION_M
        ):
            return {"continue_task_search": False, "reason": "path_contains_contact_outside_task_search_allowance"}
    return {
        "continue_task_search": True,
        "unsafe_contact_present": True,
        "policy": "unsafe target-bowl/non-pad left/right finger-shell contact retained for task feasibility only",
        "illegal_contact_count": len(rows),
        "unique_geom_pairs": sorted({tuple(sorted(zip(row["geom_ids"], row["geom_names"]))) for row in rows}, key=str),
        "deepest_signed_distance_m": min(float(row["distance_m"]) for row in rows),
        "maximum_penetration_m": max(max(0.0, -float(row["distance_m"])) for row in rows),
        "safety_policy_changed": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_scene_path = args.scene_source.resolve()
    snapshot_path = args.snapshot.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_scene = json.loads(gzip.decompress(source_scene_path.read_bytes()))
    payload = snapshot_path.read_bytes()
    setattr(sys.modules["__main__"], "_PythonObjectState", _PythonObjectState)
    envelope = pickle.loads(payload)
    if envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
        raise ValueError("candidate9_runtime_snapshot_schema_mismatch")
    if envelope.get("scene_metadata", {}).get("model_geometry_sha256") != MODEL_SHA:
        raise ValueError("candidate9_snapshot_model_signature_mismatch")
    start_t = int(envelope.get("snapshot", {}).get("task_bookkeeping", {}).get("timestep", -1))
    if not 199 <= start_t < HORIZON:
        raise ValueError(f"expected_candidate9_snapshot_timestep_in_[199,{HORIZON})_got_{start_t}")
    expected_state = np.asarray(envelope["snapshot"]["mujoco_state"], dtype=np.float64)
    source_run_report_path = args.source_report.resolve() if args.source_report else snapshot_path.parent / "task_report.json"
    source_run_report = json.loads(source_run_report_path.read_text(encoding="utf-8")) if source_run_report_path.is_file() else None
    source_run_report_matches = False
    source_snapshot_is_terminal = False
    source_snapshot_is_recorded_artifact = False
    if isinstance(source_run_report, dict):
        source_terminal = source_run_report.get("terminal_snapshot", {})
        source_snapshot_is_terminal = bool(
            source_run_report.get("schema") == "gs_opd_oa_l0t0_candidate9_full_task_continuation.v1"
            and Path(source_terminal.get("path", "")).resolve() == snapshot_path
            and source_terminal.get("sha256") == _sha(payload)
            and int(source_terminal.get("timestep", -1)) == start_t
        )
        recorded_snapshot_paths = {
            Path(path).resolve()
            for path in source_run_report.get("artifacts", {}).get("snapshots", [])
        }
        source_snapshot_is_recorded_artifact = snapshot_path in recorded_snapshot_paths
        source_run_report_matches = source_snapshot_is_terminal or source_snapshot_is_recorded_artifact
    prior_branch_actions = 0
    source_branch_actions_before_snapshot = 0
    completed_source_phases: list[str] = []
    source_release_confirmed = False
    source_release_transition_started = False
    if source_run_report_matches:
        previous_chain = source_run_report.get("prior_task_feasibility_prefix_chain", {})
        if source_snapshot_is_terminal:
            source_branch_actions_before_snapshot = int(source_run_report.get("physical_actions", 0))
        else:
            parent_start_t = int(source_run_report.get("source_snapshot", {}).get("timestep", -1))
            if parent_start_t < 0 or start_t < parent_start_t:
                raise ValueError("recorded_artifact_snapshot_precedes_its_parent_run_source")
            source_branch_actions_before_snapshot = start_t - parent_start_t
        prior_branch_actions = int(previous_chain.get("cumulative_continuation_actions_before_source", 0)) + source_branch_actions_before_snapshot
        source_release = source_run_report.get("release", {})
        source_release_confirmed = source_release.get("physical_robot_target_contact_separated") is True
        source_release_transition_started = bool(source_release.get("events"))
        completed_source_phases = list(previous_chain.get("completed_route_phases_in_source_snapshot", []))
        completed_source_phases.extend(
            str(route.get("phase"))
            for route in source_run_report.get("routes", [])
            if route.get("execution_status") == "reached" and route.get("phase")
        )
        completed_source_phases = sorted(set(completed_source_phases))
    if start_t >= HORIZON:
        raise ValueError("standard_task_horizon_already_exhausted_at_branch_start")
    configured_budget = int(getattr(args, "max_suffix_actions", HORIZON - start_t))
    if configured_budget < 1:
        raise ValueError("branch_action_budget_must_be_positive")
    action_budget = min(configured_budget, HORIZON - start_t)
    branch_config = {}
    if getattr(args, "branch_config", None):
        branch_config = json.loads(Path(args.branch_config).read_text(encoding="utf-8"))
        if not isinstance(branch_config, dict):
            raise ValueError("branch_config_must_be_a_json_object")
    candidate_id = str(branch_config.get("candidate_id", output.name))
    report: dict[str, Any] = {
        "schema": "gs_opd_oa_l0t0_candidate9_full_task_continuation.v1",
        "status": "initializing",
        "scope": "task-feasibility completion from one exact candidate9 held-place snapshot; one bounded live branch",
        "candidate": {"candidate_id": candidate_id, "configuration": branch_config, "engineering_choices_not_paper_values": True},
        "source_snapshot": {"path": str(snapshot_path), "sha256": _sha(payload), "timestep": start_t, "model_geometry_sha256": MODEL_SHA},
        "source_run_report": {"path": str(source_run_report_path), "matched_snapshot_sha256_and_timestep": source_run_report_matches},
        "source_snapshot_binding": {
            "exact_terminal_snapshot_match": source_snapshot_is_terminal,
            "recorded_snapshot_artifact_match": source_snapshot_is_recorded_artifact,
            "snapshot_payload_sha256": _sha(payload),
            "snapshot_timestep": start_t,
            "source_run_actions_before_snapshot": source_branch_actions_before_snapshot,
        },
        "stable_release_observed_in_prior_branch": bool(
            isinstance(source_run_report, dict)
            and source_run_report.get("pre_retreat_settling", {}).get("stable_velocity_confirmed_for_three_actions") is True
        ),
        "resumed_after_physical_release": source_release_confirmed,
        "resumed_during_release_transition": source_release_transition_started and not source_release_confirmed,
        "prior_task_feasibility_prefix_chain": {
            "t0_to_t152": "/opt/data/private/gs_opd/rollouts/oa_l0t0_candidate09_physical_RU3wGc/episode",
            "t152_to_t190": "/opt/data/private/gs_opd/rollouts/oa_l0t0_c09_feasible_lift_5sy0p49a",
            "t190_to_t199": str(PRIOR_TRANSPORT_DIR),
            "t152_actions": 152,
            "t153_to_t190_actions": 38,
            "t191_to_t199_actions": 9,
            "cumulative_continuation_actions_before_source": prior_branch_actions,
            "completed_route_phases_in_source_snapshot": completed_source_phases,
            "historical_continuation_actions_before_snapshot": prior_branch_actions,
            "branch_action_budget_total_from_exact_snapshot": action_budget,
            "branch_action_budget_remaining": action_budget,
            "cross_archive_timestep_continuity_claimed": False,
            "t0_to_t152_contact_monitoring_complete": True,
            "t152_to_t190_safety_certified": False,
            "t190_to_t199_safety_monitoring_complete": False,
            "reason": "the t=152 lift-start state was recorded as an input timestep but no exact source snapshot copy is in that archive; later exact snapshot continuations are separately attributable",
        },
        "task": {"benchmark": "obstacle_avoidance", "level": 0, "task_id": 0, "target_object_id": OBJECT_ID, "target_region_id": REGION_ID, "native_predicate": ["on", OBJECT_ID, REGION_ID]},
        "limits": {"standard_horizon": HORIZON, "start_timestep": start_t, "maximum_suffix_actions": action_budget, "mujoco_substeps_per_action": SUBSTEPS, "settling_actions": SETTLE_ACTIONS},
        "task_success": False,
        "safety_success": False,
        "safety_monitoring_complete": False,
        "strict_safety_certified": False,
        "suffix_safety_monitoring_complete": False,
        "suffix_strict_safety_certified": False,
        "new_trainable_episodes": 0,
        "pi05_native_action_compatible": False,
        "control_action_space": "7D_joint_target_qpos_rad plus normalized gripper command",
        "routes": [],
        "physical_actions": 0,
        "task_feasibility_contact_continuations": [],
        "task_feasibility_contact_policy": {
            "purpose": "allow the single task-success search branch to proceed past its observed shallow carry contact",
            "exact_robot_geom_body_pairs": sorted([list(pair) for pair in SEARCH_CONTACT_ROBOT_GEOM_BODY_PAIRS]),
            "object_scope": "only bound target bowl collision geoms",
            "maximum_search_penetration_m": SEARCH_CONTACT_MAX_PENETRATION_M,
            "phases": ["vertical_clear", "above_plate", "place", "release_open_with_negative_gripper_command", "retreat"],
            "safety_rule_changed": False,
            "contact_is_still_classified_unsafe": True,
            "other_contact_types_remain_hard_stops": True,
        },
        "task_feasibility_path_contact_continuations": [],
        "complete_trajectory_reconstruction": False,
    }
    _write(output / "task_report.json", report)
    process = log = connection = recorder = None
    bridge = None
    safety_policy = None
    binding = None
    context = None
    final_obs = None
    monitor_rows: list[dict[str, Any]] = []
    action_log = None
    try:
        process, log, connection, initial = api._start_bridge(source_scene, output)
        bridge = _Bridge(connection)
        if int(initial.get("timestep", -1)) != 0:
            raise RuntimeError("fresh_Bridge_did_not_start_at_zero")
        restored = bridge.request({"op": "restore_runtime_snapshot", "payload": payload})
        actual_state = np.asarray(restored.get("snapshot_state", {}).get("mujoco_integration_state", []), dtype=np.float64)
        if restored.get("status") != "restored" or restored.get("model_geometry_sha256") != MODEL_SHA or not np.array_equal(actual_state, expected_state):
            raise RuntimeError(f"exact_source_snapshot_restore_failed:{restored.get('error') or restored.get('status')}")
        report["restore"] = {"status": restored.get("status"), "exact_integration_state_match": True, "timestep": restored.get("snapshot_state", {}).get("timestep"), "integration_state_sha256": hashlib.sha256(actual_state.tobytes()).hexdigest()}
        live = bridge.request({"op": "observe", "include_snapshot_state": True})
        if int(live.get("timestep", -1)) != start_t:
            raise RuntimeError("restored_timestep_does_not_match_source_snapshot")
        native_before = bridge.request({"op": "native_task_status"})
        if native_before.get("status") != "complete" or native_before.get("task_success") not in {False, True}:
            raise RuntimeError("native_task_status_unavailable_at_exact_continuation_state")
        semantic = bridge.request({"op": "task_semantic_geometry_bindings"})
        context = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
        collision_scene = bridge.request({"op": "collision_scene_export"})
        inventory = bridge.request({"op": "geometry_inventory"})
        if semantic.get("status") != "complete" or context.get("status") != "complete" or collision_scene.get("status") != "complete" or inventory.get("status") != "complete":
            raise RuntimeError("active_scene_context_or_geometry_binding_incomplete")
        target = semantic["bindings"].get(OBJECT_ID)
        binding = {"object_id": OBJECT_ID, "body_id": int(target["root_body_id"]), "body_name": str(target["root_body_name"]), "geom_ids": sorted(int(x) for x in target["geom_ids"]), "geom_names": list(target["geom_names"])}
        bowl_geom_ids = set(int(x) for x in binding["geom_ids"])
        robot_geom_ids = {
            int(item["geom_id"])
            for item in inventory.get("geometries", [])
            if item.get("category") == "robot" and item.get("collision_enabled") is True
        }
        if not bowl_geom_ids or not robot_geom_ids:
            raise RuntimeError("target_bowl_or_full_robot_collision_geometry_inventory_missing")
        grasp = bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": INSTRUCTION})
        live_grasp_confirmed = grasp.get("status") == "pass" and grasp.get("grasp_confirmed") is True
        if not live_grasp_confirmed and not (source_release_confirmed or source_release_transition_started):
            raise RuntimeError("candidate9_bilateral_grasp_not_confirmed_and_release_transition_not_documented")
        arm_q, gripper_q = prior._current_arm_state(live["snapshot_state"])
        fk = bridge.request({"op": "scratch_robot_fk_audit", "arm_joint_positions_rad": [arm_q.tolist()]})
        if fk.get("status") != "complete" or fk.get("physical_state_unchanged") is not True:
            raise RuntimeError("active_robot_base_fk_audit_incomplete")
        fkrow = fk["poses"][0]["bodies"]
        world_base = prior._pose_from_wxyz(fkrow["robot0_base"]["position_world_m"], fkrow["robot0_base"]["quaternion_world_wxyz"])
        world_hand = prior._pose_from_wxyz(fkrow["robot0_right_hand"]["position_world_m"], fkrow["robot0_right_hand"]["quaternion_world_wxyz"])
        world_object = _object_pose(context)
        hand_object = np.linalg.inv(world_hand) @ world_object
        safety_policy = _build_policy(context, semantic, inventory)
        bridge_policy = prior._safety_policy_for_bridge(safety_policy)
        release_transition_bridge_policy = _released_safety_policy_for_bridge(safety_policy, release_transition=True)
        released_bridge_policy = _released_safety_policy_for_bridge(safety_policy)
        if not live_grasp_confirmed:
            pad_contact = grasp.get("finger_contacts", {})
            has_pad_contact = bool(pad_contact.get("left") or pad_contact.get("right") or int(pad_contact.get("contact_count", 0)) > 0)
            bridge_policy = release_transition_bridge_policy if has_pad_contact else released_bridge_policy
        placement_pose, placement_geometry = _placement_object_pose(
            context,
            collision_scene,
            semantic,
            world_base,
            xy_fraction=tuple(branch_config.get("placement_xy_fraction", [0.0, 0.0])),
            yaw_delta_deg=float(branch_config.get("placement_yaw_delta_deg", 0.0)),
            bottom_clearance_m=float(branch_config.get("bowl_bottom_clearance_mm", 1.0)) / 1000.0,
        )
        plate_geom_ids = set(int(x) for x in placement_geometry["target_collision_geom_ids"])
        current_eef = _world_pose(context, "eef_pose_world")
        region_frame = np.asarray(placement_geometry["plate_collision_frame_world_m"], dtype=np.float64)
        above_pose = placement_pose.copy()
        approach_clearance_m = float(branch_config.get("preplace_clearance_mm", 35.0)) / 1000.0
        if not np.isfinite(approach_clearance_m) or approach_clearance_m < 0.010 or approach_clearance_m > 0.100:
            raise ValueError("preplace_clearance_mm_must_be_between_10_and_100")
        above_pose[:3, 3] += region_frame[:3, 2] * approach_clearance_m
        # Each new placement candidate replans both object-relative Keyposes.
        # A bounded same-candidate replan may resume from its exact reached
        # Pre-Place artifact and skip only that already reached phase.
        reached_at_source = {
            str(row.get("phase")): int(row.get("terminal_timestep", -1))
            for row in (source_run_report or {}).get("routes", [])
            if row.get("execution_status") == "reached" and row.get("terminal_timestep") is not None
        }
        route_poses = []
        if reached_at_source.get("above_plate") != start_t:
            route_poses.append(("above_plate", above_pose))
        route_poses.append(("place", placement_pose))
        report["initial_state"] = {"native_task_status": native_before, "grasp_state": grasp, "arm_qpos_rad": arm_q.tolist(), "gripper_qpos_m": gripper_q.tolist(), "world_base_transform_m": world_base.tolist(), "world_panda_hand_transform_m": world_hand.tolist(), "world_eef_transform_m": current_eef.tolist(), "world_bowl_pose_m": world_object.tolist(), "T_panda_hand_bowl_m": hand_object.tolist(), "strict_safety_geometry_check": bridge.request({"op": "safety_geometry_check", "safety_geometry_policy": bridge_policy})}
        report["native_task_success_before_continuation"] = native_before.get("task_success") is True
        report["keypose_targets"] = {
            "object_to_plate_region_place_transform_m": (np.linalg.inv(region_frame) @ placement_pose).tolist(),
            "object_to_plate_region_pre_place_transform_m": (np.linalg.inv(region_frame) @ above_pose).tolist(),
            "world_place_object_pose_m": placement_pose.tolist(),
            "world_pre_place_object_pose_m": above_pose.tolist(),
            "place_geometry": placement_geometry,
            "measured_hand_to_object_transform_at_branch_start_m": hand_object.tolist(),
            "keypose_mapping": "T_world_hand_goal = T_world_object_Keypose @ inverse(T_hand_object_measured); all poses in meters and plate collision-frame Region coordinates",
            "all_offsets_clearances_and_yaw_variants_are_engineering_choices_not_paper_values": True,
        }
        report["safety_policy"] = {"source": safety_policy["source"], "existing_clearance_rules_unchanged": True, "policy_proof": safety_policy["_proof"], "task_feasibility_may_record_expected_goal_support_contact_as_unsafe": True}
        report["safety_policy"]["released_state_policy"] = {
            "carried_object": None,
            "gripper_target_object_contact_pairs": "exact pre-existing verified pad pairs only during release transition; empty after physical separation",
            "minimum_clearances_unchanged": True,
            "selected_at_start_because_grasp_not_confirmed": not live_grasp_confirmed,
        }
        _write(output / "task_report.json", report)

        from research.gs_opd.repair.run_human_l0t2_curobo_transport import EpisodeRecorder
        recorder = EpisodeRecorder(output, bridge, record_schema="gs_opd_oa_l0t0_candidate9_task_step.v1")
        recorder.snapshot(f"resumed_candidate9_t{start_t}")
        action_log = gzip.open(output / "action_monitoring.jsonl.gz", "wt", encoding="utf-8")
        report["physical_execution"] = {"status": "running", "control_mode": "MuJoCo mass-matrix PD joint trajectory", "action_labels": "7D joint target plus normalized gripper command; not pi0.5 native OSC", "action_budget_total_from_exact_snapshot": action_budget, "task_episode_horizon": HORIZON, "strict_policy_is_recorded_not_relaxed": True}
        _write(output / "task_report.json", report)

        def execute(qtarget: np.ndarray, phase: str, waypoint_index: int, *, grip_action: float = 0.0, allow_expected_support: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
            if recorder.action_count >= action_budget or start_t + recorder.action_count >= HORIZON:
                raise RuntimeError("bounded_physical_or_standard_horizon_budget_exhausted")
            grasp_before_action = None
            if phase in {"release_open", "retreat", "settling"} or phase.startswith(("release", "retreat", "settling")):
                grasp_before_action = bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": INSTRUCTION})
            is_released_before_action = grasp_before_action is not None and grasp_before_action.get("grasp_confirmed") is not True
            pad_contact_state = grasp_before_action.get("finger_contacts", {}) if grasp_before_action else {}
            has_pad_contact = bool(pad_contact_state.get("left") or pad_contact_state.get("right") or int(pad_contact_state.get("contact_count", 0)) > 0)
            release_transition = bool(phase in {"release_open", "retreat"} and grip_action < 0.0 and is_released_before_action and has_pad_contact)
            if release_transition:
                action_safety_policy = release_transition_bridge_policy
                policy_mode = "release_transition_exact_pad_pairs_only"
            elif is_released_before_action:
                action_safety_policy = released_bridge_policy
                policy_mode = "released_object_strict_scene_clearance"
            else:
                action_safety_policy = bridge_policy
                policy_mode = "verified_carried_object"
            pre = bridge.request({"op": "preview_joint_trajectory_step", "action_index": recorder.action_count, "joint_target_qpos_rad": qtarget.tolist(), "gripper_action": float(grip_action), "joint_position_kp": 50.0, "safety_geometry_policy": action_safety_policy})
            pre_evidence = _strict_contacts(pre)
            if pre.get("status") != "complete" or pre.get("main_restore_verified") is not True or not pre_evidence["complete_25_substep_monitoring"]:
                _write(output / f"rejected_preflight_{recorder.action_count:04d}.json", {"phase": phase, "waypoint": waypoint_index, "preview": pre, "evidence": pre_evidence})
                raise RuntimeError(f"pre_action_monitor_or_scratch_restore_incomplete:{phase}:{waypoint_index}")
            pre_search_contact = _task_feasibility_search_contact(
                pre_evidence, set(binding["geom_ids"]), phase,
                allow_open_release=(phase in {"release_open", "retreat"} and grip_action < 0.0),
                allow_expected_plate_support=allow_expected_support,
                plate_geom_ids=set(plate_geom_ids),
            )
            pre_expected_support = allow_expected_support and _only_expected_plate_support(pre_evidence, set(binding["geom_ids"]), set(plate_geom_ids))
            if pre_evidence["actual_hard_contact_present"] and not (pre_expected_support or pre_search_contact):
                _write(output / f"rejected_preflight_{recorder.action_count:04d}.json", {"phase": phase, "waypoint": waypoint_index, "preview": pre, "evidence": pre_evidence})
                raise RuntimeError(f"pre_action_actual_contact_rejected:{phase}:{waypoint_index}")
            physical = bridge.request({"op": "joint_trajectory_step", "action_index": recorder.action_count, "joint_target_qpos_rad": qtarget.tolist(), "gripper_action": float(grip_action), "joint_position_kp": 50.0, "safety_geometry_policy": action_safety_policy})
            if physical.get("status") == "failed":
                raise RuntimeError(f"MuJoCo_action_failed:{physical.get('error')}")
            if not np.allclose(np.asarray(pre["predicted_mujoco_integration_state"], dtype=np.float64), np.asarray(physical["snapshot_state"]["mujoco_integration_state"], dtype=np.float64), atol=1e-7, rtol=0):
                raise RuntimeError("scratch_preview_vs_real_physics_mismatch")
            recorder.record(physical, qtarget, phase, float(grip_action))
            evidence = _strict_contacts(physical)
            monitor_rows.append(evidence)
            if not evidence["complete_25_substep_monitoring"]:
                raise RuntimeError(f"physical_25_substep_monitoring_incomplete:{phase}:{recorder.action_count - 1}")
            physical_search_contact = _task_feasibility_search_contact(
                evidence, set(binding["geom_ids"]), phase,
                allow_open_release=(phase in {"release_open", "retreat"} and grip_action < 0.0),
                allow_expected_plate_support=allow_expected_support,
                plate_geom_ids=set(plate_geom_ids),
            )
            physical_expected_support = allow_expected_support and _only_expected_plate_support(evidence, set(binding["geom_ids"]), set(plate_geom_ids))
            if evidence["actual_hard_contact_present"] and not (physical_expected_support or physical_search_contact):
                raise RuntimeError(f"physical_actual_contact_rejected:{phase}:{recorder.action_count - 1}")
            if pre_search_contact or physical_search_contact:
                report["task_feasibility_contact_continuations"].append({
                    "action_index": recorder.action_count - 1,
                    "phase": phase,
                    "waypoint_index": waypoint_index,
                    "safety_label": "unsafe_contact_recorded_task_search_only",
                    "preflight": _task_search_contact_summary(pre_evidence),
                    "physical": _task_search_contact_summary(evidence),
                })
            if phase not in {"release_open", "retreat", "settling"} and not phase.startswith(("release", "retreat", "settling")):
                grasp_after_action = bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": INSTRUCTION})
                if grasp_after_action.get("status") != "pass" or grasp_after_action.get("grasp_confirmed") is not True:
                    raise RuntimeError(f"bowl_dropped_or_grasp_lost_during_{phase}:{recorder.action_count - 1}")
            obs_state = physical["snapshot_state"]
            track = prior._tracking(obs_state, qtarget)
            native = bridge.request({"op": "native_task_status"})
            action_log.write(json.dumps(api._json_safe({"action_index": recorder.action_count - 1, "timestep": physical.get("timestep"), "phase": phase, "waypoint_index": waypoint_index, "safety_monitor_policy_mode": policy_mode, "preview_monitoring": pre_evidence, "physical_monitoring": evidence, "native_task_status": native, "tracking": track}), ensure_ascii=False, separators=(",", ":")) + "\n")
            report["physical_actions"] = recorder.action_count
            report["recorded_physics_substeps"] = recorder.recorded_physics_substeps
            report["suffix_safety_monitoring_complete"] = bool(len(monitor_rows) == recorder.action_count and all(r["complete_25_substep_monitoring"] for r in monitor_rows))
            report["suffix_strict_safety_certified"] = bool(report["suffix_safety_monitoring_complete"] and all(r["safety_certifiable_action"] for r in monitor_rows))
            report["safety_monitoring_complete"] = False
            report["strict_safety_certified"] = False
            if recorder.action_count % 5 == 0:
                recorder.snapshot(f"step_{recorder.action_count:04d}")
                _write(output / "task_report.json", report)
                print(f"task suffix actions={recorder.action_count}/{action_budget} t={physical.get('timestep')} phase={phase} qerr={track['joint_error_linf_rad']:.4f} native={native.get('task_success')}", flush=True)
            return physical, track

        def make_attached_planner(state_context: dict[str, Any], scene_export: dict[str, Any], arm: np.ndarray, jaw: np.ndarray, seed: int):
            ps = copy.deepcopy(source_scene)
            ps["target_context"] = copy.deepcopy(state_context)
            ps["target_context"]["target_binding"] = target
            ps["target_context"]["arm_qpos"] = arm.tolist()
            ps["collision_scene_export"] = scene_export
            planner, device, dtype, state, qstart, planner_info = api._make_planner(ps, world_base, "closed", seed=seed, gripper_qpos_m=jaw)
            attached, coverage = api._attached_spheres(ps)
            if not coverage.get("every_source_box_analytically_covered"):
                raise RuntimeError("held_bowl_attachment_geometry_coverage_failed")
            attached_tensor = api.torch.as_tensor(attached, device=device, dtype=dtype)
            object_pose_base = api._pose_object(np.linalg.inv(world_base) @ _object_pose(state_context), device, dtype)
            # CuRobo 0.8 composes a separate SolverCore for IK and TrajOpt.
            # Populate both attached-object sphere models; the historical
            # MotionPlanner.attachment_manager property only reaches TrajOpt.
            attachment_managers = {}
            for solver_name in ("ik_solver", "trajopt_solver"):
                solver = getattr(planner, solver_name, None)
                core = getattr(solver, "core", None)
                manager = getattr(core, "attachment_manager", None)
                if manager is not None:
                    attachment_managers[solver_name] = manager
            if not attachment_managers:
                manager = getattr(planner, "attachment_manager", None)
                if manager is not None:
                    attachment_managers["motion_planner"] = manager
            if not attachment_managers:
                raise RuntimeError("CuRobo_attachment_manager_unavailable_for_IK_and_TrajOpt")
            for manager in attachment_managers.values():
                manager.update(attached_tensor, state, link_name="attached_object", world_objects_pose_offset=object_pose_base)
            coverage["attached_to_planning_components"] = sorted(attachment_managers)
            world, cubes, meshes = api._make_world(ps, world_base, exclude_target=True)
            planner.update_world(world)
            return ps, planner, device, dtype, state, qstart, planner_info, coverage, cubes, meshes

        # Roll the planner forward from actual observations after each stage.
        for stage_i, (phase, desired_object_world) in enumerate(route_poses):
            context_now = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
            obs_now = bridge.request({"op": "observe", "include_snapshot_state": True})
            arm_now, jaw_now = prior._current_arm_state(obs_now["snapshot_state"])
            scene_now = bridge.request({"op": "collision_scene_export"})
            if scene_now.get("status") != "complete" or scene_now.get("physical_state_unchanged") is not True:
                raise RuntimeError(f"live_collision_scene_export_incomplete_before_{phase}")
            object_now = _object_pose(context_now)
            fk_now = bridge.request({"op": "scratch_robot_fk_audit", "arm_joint_positions_rad": [arm_now.tolist()]})
            if fk_now.get("status") != "complete" or fk_now.get("physical_state_unchanged") is not True:
                raise RuntimeError(f"live_robot_fk_audit_incomplete_before_{phase}")
            fk_bodies = fk_now["poses"][0]["bodies"]
            hand_now = prior._pose_from_wxyz(fk_bodies["robot0_right_hand"]["position_world_m"], fk_bodies["robot0_right_hand"]["quaternion_world_wxyz"])
            hand_object_now = np.linalg.inv(hand_now) @ object_now
            hand_goal_world = desired_object_world @ np.linalg.inv(hand_object_now)
            hand_goal_base = np.linalg.inv(world_base) @ hand_goal_world
            route = {
                "phase": phase,
                "start_timestep": int(obs_now.get("timestep", -1)),
                "start_joint_state_rad": arm_now.tolist(),
                "target_object_pose_world_m": desired_object_world.tolist(),
                "target_panda_hand_pose_base_m": hand_goal_base.tolist(),
                "measured_T_panda_hand_object_m": hand_object_now.tolist(),
                "bounded_planner_seeds": [int(x) for x in branch_config.get("keypose_planner_seeds", [20261103 + stage_i])],
                "planner_candidates": [],
                "execution_status": "not_started",
            }
            report["routes"].append(route)
            _write(output / f"route_{stage_i}_{phase}.json", route)
            selected = None
            for planner_seed in route["bounded_planner_seeds"]:
                _, planner, device, dtype, start_state, qstart, planner_info, coverage, cubes, meshes = make_attached_planner(context_now, scene_now, arm_now, jaw_now, planner_seed)
                ik, planned, path, endpoint = api._plan_stage(planner, device, dtype, start_state, hand_goal_base, grasp_goal=False)
                candidate = {
                    "seed": planner_seed,
                    "ik": ik,
                    "curobo": planned,
                    "valid_path_point_count": len(path) if path is not None else 0,
                    "buffered_path_point_count": planned.get("buffered_trajectory_point_count"),
                    "full_scene_cuboid_count_without_attached_target": cubes,
                    "full_scene_mesh_count_without_attached_target": meshes,
                    "attached_sphere_coverage": coverage,
                    "planner_configuration": planner_info,
                }
                route["planner_candidates"].append(candidate)
                if path is None or planned.get("success") is not True:
                    candidate["rejection_reason"] = planned.get("rejection_reason") or planned.get("status") or "planner_returned_no_success"
                    _write(output / f"route_{stage_i}_{phase}.json", route)
                    continue
                if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 or not np.isfinite(path).all() or np.max(np.abs(path[0] - arm_now)) > 5e-4:
                    candidate["rejection_reason"] = "CuRobo_path_shape_or_live_start_state_invalid"
                    _write(output / f"route_{stage_i}_{phase}.json", route)
                    continue
                if len(path) - 1 + MAX_ENDPOINT_FEEDBACK_ACTIONS > action_budget - recorder.action_count:
                    candidate["rejection_reason"] = "CuRobo_route_exceeds_remaining_physical_budget"
                    _write(output / f"route_{stage_i}_{phase}.json", route)
                    continue
                audit = bridge.request({"op": "scratch_robot_path_contact_audit", "arm_joint_states_rad": path.tolist(), "gripper_joint_states_rad": np.repeat(jaw_now[None, :], len(path), axis=0).tolist(), "object_id": OBJECT_ID, "fingertip_contact_scope": "all_states", "T_eef_object_m": (np.linalg.inv(_world_pose(context_now, "eef_pose_world")) @ object_now).tolist()})
                path_contact = _task_feasibility_path_contact(audit, set(binding["geom_ids"]), phase)
                candidate["active_mujoco_discrete_path_audit"] = audit
                candidate["task_feasibility_path_contact"] = path_contact
                audit_valid = audit.get("status") == "complete" and audit.get("physical_state_unchanged") is True and int(audit.get("state_count", -1)) == len(path)
                if not audit_valid or not path_contact.get("continue_task_search"):
                    candidate["rejection_reason"] = path_contact.get("reason") or "exact_discrete_path_audit_incomplete"
                    _write(output / f"route_{stage_i}_{phase}.json", route)
                    continue
                candidate["selected_for_execution"] = True
                selected = (path, ik, planned, audit, path_contact, planner_info, coverage, cubes, meshes, planner_seed)
                _write(output / f"route_{stage_i}_{phase}.json", route)
                break
            if selected is None:
                route["execution_status"] = "blocked_no_curobo_route"
                route["rejection_reason"] = "all_bounded_seeds_failed_planning_or_active_MuJoCo_path_audit"
                last_candidate = route["planner_candidates"][-1] if route["planner_candidates"] else {}
                route["ik"] = last_candidate.get("ik")
                route["curobo"] = last_candidate.get("curobo")
                route["valid_path_point_count"] = 0
                _write(output / f"route_{stage_i}_{phase}.json", route)
                raise RuntimeError(f"CuRobo_route_unavailable:{phase}:{route['rejection_reason']}")
            path, ik, planned, audit, path_contact, planner_info, coverage, cubes, meshes, selected_seed = selected
            route.update({
                "selected_planner_seed": selected_seed,
                "ik": ik,
                "curobo": planned,
                "valid_path_point_count": len(path),
                "buffered_path_point_count": planned.get("buffered_trajectory_point_count"),
                "full_scene_cuboid_count_without_attached_target": cubes,
                "full_scene_mesh_count_without_attached_target": meshes,
                "attached_sphere_coverage": coverage,
                "planner_configuration": planner_info,
            })
            route["active_mujoco_discrete_path_audit"] = audit
            route["task_feasibility_path_contact"] = path_contact
            if path_contact.get("unsafe_contact_present"):
                report["task_feasibility_path_contact_continuations"].append({"phase": phase, "path_point_count": len(path), **path_contact})
                route["execution_status"] = "executing_with_task_only_unsafe_contact"
            else:
                route["execution_status"] = "executing"
            _write(output / f"route_{stage_i}_{phase}.json", route)
            for waypoint_index, qtarget in enumerate(path[1:], start=1):
                attempts = 0
                while True:
                    _, tracking = execute(qtarget, phase, waypoint_index, allow_expected_support=(phase == "place"))
                    attempts += 1
                    if tracking["joint_error_linf_rad"] <= 0.03:
                        break
                    if tracking["joint_error_linf_rad"] > 0.08 or attempts >= MAX_ACTIONS_PER_WAYPOINT:
                        raise RuntimeError(f"closed_loop_waypoint_tracking_stalled:{phase}:{waypoint_index}:{tracking['joint_error_linf_rad']:.6f}")
            final_q = path[-1]
            stable = False
            endpoint_samples = []
            for _ in range(MAX_ENDPOINT_FEEDBACK_ACTIONS + 1):
                state = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
                tracking = prior._tracking(state, final_q)
                endpoint_samples.append(tracking)
                if tracking["joint_error_linf_rad"] <= 0.005 and tracking["maximum_absolute_joint_velocity_rad_s"] <= 0.03:
                    stable = True
                    break
                if len(endpoint_samples) > MAX_ENDPOINT_FEEDBACK_ACTIONS:
                    break
                execute(final_q, f"{phase}_endpoint_feedback", len(path) - 1, allow_expected_support=(phase == "place"))
            route["endpoint_feedback"] = {"converged": stable, "samples": endpoint_samples}
            if not stable:
                route["execution_status"] = "stopped_endpoint_not_stable"
                raise RuntimeError(f"closed_loop_endpoint_not_stable:{phase}")
            route["execution_status"] = "reached"
            route["terminal_timestep"] = int(bridge.request({"op": "observe"}).get("timestep", -1))
            route["terminal_context"] = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
            route["terminal_grasp_state"] = bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": INSTRUCTION})
            if route["terminal_grasp_state"].get("grasp_confirmed") is not True:
                raise RuntimeError(f"bowl_grasp_lost_after_stage:{phase}")
            recorder.snapshot(f"stage_{phase}_t{route['terminal_timestep']}")
            _write(output / f"route_{stage_i}_{phase}.json", route)

        # The object is now 5 mm above the exact plate collision top. Opening
        # and settling are real actions; the monitor keeps its unchanged strict
        # policy and any plate support contact permanently makes Safety false.
        current_place_obs = bridge.request({"op": "observe", "include_snapshot_state": True})
        place_q, _ = prior._current_arm_state(current_place_obs["snapshot_state"])
        placement_events = []
        released = False
        if source_release_confirmed:
            placement_events = list(source_run_report.get("release", {}).get("events", []))
            released = True
            report["release"] = {
                "open_command": OPEN_ACTION,
                "events": placement_events,
                "physical_grasp_separated": True,
                "pad_contact_separated": True,
                "physical_robot_target_contact_separated": True,
                "inherited_from_exact_source_snapshot": str(snapshot_path),
            }
        else:
            remaining_budget = action_budget - recorder.action_count
            open_budget = min(MAX_RELEASE_OPEN_ACTIONS, remaining_budget)
            report["release"] = {
                "open_command": OPEN_ACTION,
                "maximum_open_actions": open_budget,
                "events": placement_events,
                "physical_grasp_separated": False,
                "pad_contact_separated": False,
                "physical_robot_target_contact_separated": False,
                "release_confirmation_rule": "grasp false plus zero left/right pad contacts and zero target-bowl/all-robot collision-geom contacts at final substep of a complete 25-substep physical action",
            }
            _write(output / "task_report.json", report)
            for i in range(open_budget):
                physical, track = execute(place_q, "release_open", i, grip_action=OPEN_ACTION, allow_expected_support=True)
                grasp_now = bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": INSTRUCTION})
                release_evidence = _strict_contacts(physical)
                physical_release = _physical_release_confirmed(grasp_now, release_evidence, bowl_geom_ids, robot_geom_ids)
                ctx = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
                event = {
                    "timestep": physical.get("timestep"),
                    "gripper_qpos_m": prior._current_arm_state(physical["snapshot_state"])[1].tolist(),
                    "grasp_state": grasp_now,
                    "physical_release_confirmed": physical_release,
                    "pad_contact_separated": bool(
                        grasp_now.get("finger_contacts", {}).get("left") is False
                        and grasp_now.get("finger_contacts", {}).get("right") is False
                        and int(grasp_now.get("finger_contacts", {}).get("contact_count", -1)) == 0
                    ),
                    "target_robot_contacts_last_substep": _target_robot_contacts(release_evidence, bowl_geom_ids, robot_geom_ids, substep=SUBSTEPS - 1),
                    "monitoring": release_evidence,
                    "object_pose_world_m": _object_pose(ctx).tolist(),
                    "native_task_status": bridge.request({"op": "native_task_status"}),
                    "tracking": track,
                }
                placement_events.append(event)
                _write(output / "release_transition_latest.json", event)
                report["release"]["events"] = placement_events
                report["release"]["physical_grasp_separated"] = physical_release
                report["release"]["pad_contact_separated"] = event["pad_contact_separated"]
                report["release"]["physical_robot_target_contact_separated"] = physical_release
                _write(output / "task_report.json", report)
                if physical_release:
                    released = True
                    break
            report["release"] = {
                "open_command": OPEN_ACTION,
                "maximum_open_actions": open_budget,
                "events": placement_events,
                "physical_grasp_separated": released,
                "pad_contact_separated": bool(placement_events and placement_events[-1]["pad_contact_separated"]),
                "physical_robot_target_contact_separated": released,
                "release_confirmation_rule": "grasp false plus zero left/right pad contacts and zero target-bowl/all-robot collision-geom contacts at final substep of a complete 25-substep physical action",
            }
        if not released:
            raise RuntimeError("bounded_gripper_open_budget_expired_without_physical_release_confirmation")
        recorder.snapshot("release_confirmed")

        # Hold/opening delay is a per-candidate variable. After this short
        # bounded interval, the real opening command is coupled to a planned
        # retreat, avoiding dozens of stationary opening actions at bowl level.
        post_release_hold_actions = int(branch_config.get("post_release_open_actions_before_retreat", 0))
        if not 0 <= post_release_hold_actions <= 8:
            raise ValueError("post_release_open_actions_before_retreat_must_be_between_0_and_8")
        post_release_open_rows = []
        for i in range(post_release_hold_actions):
            physical, track = execute(place_q, "release_open", 300 + i, grip_action=OPEN_ACTION, allow_expected_support=True)
            native_after = bridge.request({"op": "native_task_status"})
            row = {
                "timestep": physical.get("timestep"),
                "gripper_qpos_m": prior._current_arm_state(physical["snapshot_state"])[1].tolist(),
                "native_task_status": native_after,
                "tracking": track,
                "monitoring": _strict_contacts(physical),
            }
            post_release_open_rows.append(row)
            if native_after.get("task_success") is not True:
                raise RuntimeError("native_task_predicate_lost_during_candidate_opening_schedule")
        report["release"]["post_release_opening_schedule"] = {
            "fixed_pose_open_actions_before_retreat": post_release_hold_actions,
            "actions": post_release_open_rows,
            "opening_continues_during_retreat": bool(branch_config.get("open_during_retreat", True)),
        }
        recorder.snapshot("pre_retreat_release_schedule")

        released_context = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
        released_object = _object_pose(released_context)
        opened_obs = bridge.request({"op": "observe", "include_snapshot_state": True})
        opened_arm, opened_jaw = prior._current_arm_state(opened_obs["snapshot_state"])
        retreat_origin_world = _world_pose(released_context, "eef_pose_world")
        requested_retreat_direction = branch_config.get("retreat_direction_region", [0.0, 0.0, 1.0])
        if requested_retreat_direction == "away_from_bowl":
            hand_bowl_world = retreat_origin_world[:3, 3] - released_object[:3, 3]
            retreat_direction_region = region_frame[:3, :3].T @ hand_bowl_world
            retreat_direction_region[2] = 0.0
            if np.linalg.norm(retreat_direction_region) < 1e-8:
                retreat_direction_region = np.asarray([0.0, 0.0, 1.0])
        else:
            retreat_direction_region = np.asarray(requested_retreat_direction, dtype=np.float64)
        if retreat_direction_region.shape != (3,) or not np.isfinite(retreat_direction_region).all() or np.linalg.norm(retreat_direction_region) < 1e-8:
            raise ValueError("retreat_direction_region_must_be_finite_nonzero_3d_vector")
        retreat_direction_region /= np.linalg.norm(retreat_direction_region)
        retreat_direction_world = region_frame[:3, :3] @ retreat_direction_region
        retreat_distance_m = float(branch_config.get("retreat_distance_mm", 60.0)) / 1000.0
        if not np.isfinite(retreat_distance_m) or not 0.020 <= retreat_distance_m <= 0.100:
            raise ValueError("retreat_distance_mm_must_be_between_20_and_100")
        retreat_direction_evidence = {
            "source": "candidate configuration in live plate collision frame",
            "region_direction": retreat_direction_region.tolist(),
            "engineering_choice_not_paper_value": True,
        }
        retreat_stage_targets = [
            {
                "name": "candidate_region_relative_retreat",
                "world_offset_m": retreat_direction_world * retreat_distance_m,
            },
        ]
        opened_planner_source = copy.deepcopy(source_scene)
        opened_planner_source["target_context"] = copy.deepcopy(released_context)
        opened_planner_source["target_context"]["target_binding"] = target
        opened_planner_source["target_context"]["arm_qpos"] = opened_arm.tolist()
        opened_collision = bridge.request({"op": "collision_scene_export"})
        opened_planner_source["collision_scene_export"] = opened_collision
        retreat = {
            "start_timestep": int(opened_obs.get("timestep", -1)),
            "retreat_origin_hand_world_m": retreat_origin_world.tolist(),
            "retreat_direction_world": retreat_direction_world.tolist(),
            "retreat_direction_evidence": retreat_direction_evidence,
            "retreat_stage_targets": [{"name": row["name"], "world_offset_m": row["world_offset_m"].tolist()} for row in retreat_stage_targets],
            "planner_seeds": [int(x) for x in branch_config.get("retreat_planner_seeds", [20261120, 20261121, 20261122])],
            "planning_method": "bounded multi-seed CuRobo candidates with per-candidate active MuJoCo path audits and feedback execution",
            "stages": [],
            "complete": False,
        }
        report["retreat_route"] = retreat
        _write(output / "retreat_route.json", retreat)
        for stage_index, target_spec in enumerate(retreat_stage_targets):
            live_obs = bridge.request({"op": "observe", "include_snapshot_state": True})
            live_context = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
            live_arm, live_jaw = prior._current_arm_state(live_obs["snapshot_state"])
            live_object = _object_pose(live_context)
            goal_world = retreat_origin_world.copy()
            goal_world[:3, 3] += target_spec["world_offset_m"]
            goal_base = np.linalg.inv(world_base) @ goal_world
            current_scene = bridge.request({"op": "collision_scene_export"})
            stage_source = copy.deepcopy(opened_planner_source)
            stage_source["target_context"] = copy.deepcopy(live_context)
            stage_source["target_context"]["target_binding"] = target
            stage_source["target_context"]["arm_qpos"] = live_arm.tolist()
            stage_source["collision_scene_export"] = current_scene
            stage = {
                "stage_index": stage_index,
                "target_name": target_spec["name"],
                "start_timestep": int(live_obs.get("timestep", -1)),
                "start_joint_state_rad": live_arm.tolist(),
                "goal_hand_pose_world_m": goal_world.tolist(),
                "goal_hand_pose_base_m": goal_base.tolist(),
                "planner_candidates": [],
                "execution_status": "not_started",
            }
            retreat["stages"].append(stage)
            remaining_before_stage = action_budget - recorder.action_count
            stage["remaining_branch_actions_before_stage"] = remaining_before_stage
            selected_path = None
            selected_planner_seed = None
            for planner_seed in retreat["planner_seeds"]:
                candidate_planner, candidate_device, candidate_dtype, current_state, _, planner_info = api._make_planner(
                    stage_source, world_base, "open", seed=planner_seed, gripper_qpos_m=live_jaw
                )
                candidate_world, candidate_cuboids, candidate_meshes = api._make_world(
                    stage_source, world_base, exclude_target=False, target_world_pose=live_object
                )
                candidate_planner.update_world(candidate_world)
                candidate = {
                    "seed": planner_seed,
                    "planner_configuration": planner_info,
                    "world_cuboids": candidate_cuboids,
                    "world_meshes": candidate_meshes,
                    "attempts": [],
                }
                ik, planned, path, endpoint = api._plan_stage(candidate_planner, candidate_device, candidate_dtype, current_state, goal_base, grasp_goal=False)
                candidate["attempts"].append({"finger_link_collision_disabled": False, "ik": ik, "curobo": planned, "path_point_count": len(path) if path is not None else 0})
                if path is None or planned.get("success") is not True:
                    ik, planned, path, endpoint = api._plan_stage(candidate_planner, candidate_device, candidate_dtype, current_state, goal_base, grasp_goal=True)
                    candidate["attempts"].append({"finger_link_collision_disabled": True, "ik": ik, "curobo": planned, "path_point_count": len(path) if path is not None else 0})
                candidate["ik"] = ik
                candidate["curobo"] = planned
                candidate["valid_path_point_count"] = len(path) if path is not None else 0
                if path is not None and planned.get("success") is True:
                    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 or not np.isfinite(path).all() or np.max(np.abs(path[0] - live_arm)) > 5e-4:
                        candidate["rejection_reason"] = "CuRobo_path_shape_or_live_start_state_invalid"
                    elif len(path) - 1 + min(3, SETTLE_ACTIONS) > remaining_before_stage:
                        candidate["rejection_reason"] = "retreat_path_would_consume_budget_needed_for_settling"
                    else:
                        candidate_audit = bridge.request({
                            "op": "scratch_robot_path_contact_audit",
                            "arm_joint_states_rad": path.tolist(),
                            "gripper_joint_states_rad": np.repeat(live_jaw[None, :], len(path), axis=0).tolist(),
                            "object_id": OBJECT_ID,
                            "fingertip_contact_scope": "none",
                            "T_world_object_m": live_object.tolist(),
                        })
                        candidate["active_mujoco_discrete_path_audit"] = candidate_audit
                        candidate_contact = _task_feasibility_path_contact(candidate_audit, bowl_geom_ids, "retreat")
                        candidate["task_feasibility_path_contact"] = candidate_contact
                        candidate_audit_valid = (
                            candidate_audit.get("status") == "complete"
                            and candidate_audit.get("physical_state_unchanged") is True
                            and int(candidate_audit.get("state_count", -1)) == len(path)
                        )
                        if candidate_audit_valid and candidate_contact.get("continue_task_search"):
                            selected_path = path
                            selected_planner_seed = planner_seed
                            if candidate_contact.get("unsafe_contact_present"):
                                report["task_feasibility_path_contact_continuations"].append({"phase": "retreat", "stage_index": stage_index, "planner_seed": planner_seed, "path_point_count": len(path), **candidate_contact})
                            candidate["selected_for_execution"] = True
                        else:
                            candidate["rejection_reason"] = "active_MuJoCo_path_audit_rejected"
                else:
                    candidate["rejection_reason"] = planned.get("rejection_reason") or planned.get("status")
                stage["planner_candidates"].append(candidate)
                _write(output / "retreat_route.json", retreat)
                if selected_path is not None:
                    break
            if selected_path is None:
                stage["execution_status"] = "blocked_no_curobo_candidate_passed_active_audit"
                stage["rejection_reason"] = "all_bounded_seeds_failed_planning_or_active_MuJoCo_path_audit"
                _write(output / "retreat_route.json", retreat)
                raise RuntimeError(f"post_release_retreat_stage_{stage_index}_no_curobo_candidate_passed_active_audit")
            path = selected_path
            stage["selected_planner_seed"] = selected_planner_seed
            stage["execution_status"] = "executing"
            _write(output / "retreat_route.json", retreat)
            for waypoint_index, qtarget in enumerate(path[1:], start=1):
                attempts_for_endpoint = 0
                while True:
                    keep_opening = bool(branch_config.get("open_during_retreat", True))
                    _, tracking = execute(qtarget, "retreat", waypoint_index, grip_action=OPEN_ACTION if keep_opening else 0.0, allow_expected_support=True)
                    attempts_for_endpoint += 1
                    native_at_waypoint = bridge.request({"op": "native_task_status"})
                    if native_at_waypoint.get("task_success") is not True:
                        raise RuntimeError(f"native_task_predicate_lost_during_retreat:{stage_index}:{waypoint_index}")
                    if tracking["joint_error_linf_rad"] <= 0.03:
                        break
                    if tracking["joint_error_linf_rad"] > 0.08 or attempts_for_endpoint >= MAX_ACTIONS_PER_WAYPOINT:
                        stage["execution_status"] = "stopped_endpoint_not_stable"
                        stage["failed_waypoint_index"] = waypoint_index
                        _write(output / "retreat_route.json", retreat)
                        raise RuntimeError(f"retreat_tracking_stalled:stage_{stage_index}:waypoint_{waypoint_index}")
            end_obs = bridge.request({"op": "observe", "include_snapshot_state": True})
            _, _ = prior._current_arm_state(end_obs["snapshot_state"])
            endpoint_tracking = prior._tracking(end_obs["snapshot_state"], path[-1])
            stage["endpoint_feedback"] = endpoint_tracking
            stage["terminal_timestep"] = int(end_obs.get("timestep", -1))
            if endpoint_tracking["joint_error_linf_rad"] > 0.03:
                stage["execution_status"] = "stopped_endpoint_not_stable"
                _write(output / "retreat_route.json", retreat)
                raise RuntimeError(f"retreat_endpoint_not_stable:stage_{stage_index}")
            stage["execution_status"] = "reached"
            report["retreat_route"] = retreat
            _write(output / "retreat_route.json", retreat)
        retreat["complete"] = True
        report["retreat_route"] = retreat
        recorder.snapshot("retreat_complete")

        # Finish physically opening only after the hand has moved away from the
        # bowl. This check is separate from the earlier release event.
        clearance_opening = {"maximum_actions": 0, "actions": [], "complete": False}
        opened_state = bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"]
        clearance_arm, _ = prior._current_arm_state(opened_state)
        clearance_budget = min(MAX_POST_RELEASE_CLEARANCE_OPEN_ACTIONS, action_budget - recorder.action_count)
        clearance_opening["maximum_actions"] = clearance_budget
        for i in range(clearance_budget + 1):
            grasp_now = bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": INSTRUCTION})
            fingers_now = grasp_now.get("finger_contacts", {})
            closure_now = grasp_now.get("gripper_closure", {}).get("close_command_fraction", np.inf)
            previous_evidence = monitor_rows[-1] if monitor_rows else {}
            last_contacts_clear = bool(monitor_rows) and not _target_robot_contacts(previous_evidence, bowl_geom_ids, robot_geom_ids, substep=SUBSTEPS - 1)
            if float(closure_now) <= 0.05 and fingers_now.get("left") is False and fingers_now.get("right") is False and int(fingers_now.get("contact_count", -1)) == 0 and last_contacts_clear:
                clearance_opening["complete"] = True
                clearance_opening["terminal_grasp_state"] = grasp_now
                break
            if i >= clearance_budget:
                break
            physical, track = execute(clearance_arm, "release_open", 500 + i, grip_action=OPEN_ACTION, allow_expected_support=True)
            clearance_evidence = _strict_contacts(physical)
            grasp_after = bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": INSTRUCTION})
            native_after = bridge.request({"op": "native_task_status"})
            action_row = {
                "timestep": physical.get("timestep"),
                "gripper_qpos_m": prior._current_arm_state(physical["snapshot_state"])[1].tolist(),
                "grasp_state": grasp_after,
                "native_task_status": native_after,
                "target_robot_contact_free_at_final_substep": not _target_robot_contacts(clearance_evidence, bowl_geom_ids, robot_geom_ids, substep=SUBSTEPS - 1),
                "monitoring": clearance_evidence,
                "tracking": track,
            }
            clearance_opening["actions"].append(action_row)
            report["post_retreat_opening_clearance"] = clearance_opening
            _write(output / "task_report.json", report)
            if native_after.get("task_success") is not True:
                raise RuntimeError("native_task_predicate_lost_while_finishing_open_after_retreat")
        report["post_retreat_opening_clearance"] = clearance_opening
        if not clearance_opening["complete"]:
            _write(output / "task_report.json", report)
            raise RuntimeError("post_retreat_gripper_full_open_or_contact_separation_not_confirmed")
        recorder.snapshot("post_retreat_gripper_fully_open")

        q_hold, _ = prior._current_arm_state(bridge.request({"op": "observe", "include_snapshot_state": True})["snapshot_state"])
        settle_rows = []
        remaining_for_settling = action_budget - recorder.action_count
        settle_action_count = min(SETTLE_ACTIONS, remaining_for_settling)
        if settle_action_count < 3:
            raise RuntimeError(f"bounded_branch_has_insufficient_actions_for_settling:{settle_action_count}")
        for i in range(settle_action_count):
            physical, _ = execute(q_hold, "settling", i, allow_expected_support=True)
            status = bridge.request({"op": "native_task_status"})
            ctx = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
            settle_rows.append({"timestep": physical.get("timestep"), "native_task_status": status, "object_pose_world_m": _object_pose(ctx).tolist(), "object_velocity": _object_velocity(physical["snapshot_state"]), "bowl_task_goal_value": status.get("native_goal_values")})
        recorder.snapshot("settled_terminal")
        final_obs = bridge.request({"op": "observe", "include_snapshot_state": True})
        final_native = bridge.request({"op": "native_task_status"})
        final_context = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
        final_grasp = bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": INSTRUCTION})
        final_object_velocity = _object_velocity(final_obs["snapshot_state"])
        stable_positions = [np.asarray(row["object_pose_world_m"], dtype=np.float64)[:3, 3] for row in settle_rows[-3:]]
        last_three_drift_m = float(max((np.linalg.norm(a - b) for i, a in enumerate(stable_positions) for b in stable_positions[i + 1:]), default=np.inf)) if len(stable_positions) == 3 else np.inf
        goal_value = next((item.get("value") for item in final_native.get("native_goal_values", []) if tuple(str(x).lower() for x in item.get("predicate", [])) == ("on", OBJECT_ID, REGION_ID)), None)
        stable_release = bool(
            released
            and final_grasp.get("grasp_confirmed") is not True
            and goal_value is True
            and final_object_velocity.get("status") == "complete"
            and final_object_velocity.get("linear_speed_m_s", np.inf) <= 0.01
            and final_object_velocity.get("angular_speed_rad_s", np.inf) <= 0.10
            and last_three_drift_m <= 0.002
        )
        settle_native_continuous = bool(settle_rows and all(row.get("native_task_status", {}).get("task_success") is True for row in settle_rows))
        complete_task_success = bool(
            final_native.get("task_success") is True
            and released
            and retreat.get("complete") is True
            and clearance_opening.get("complete") is True
            and settle_action_count == SETTLE_ACTIONS
            and settle_native_continuous
        )
        report["settling"] = settle_rows
        report["settling_budget"] = {"requested_actions": SETTLE_ACTIONS, "executed_actions": settle_action_count, "complete": settle_action_count == SETTLE_ACTIONS}
        report["final_state"] = {"timestep": int(final_obs.get("timestep", -1)), "native_task_status": final_native, "native_task_success": bool(final_native.get("task_success") is True), "target_object_pose_world_m": _object_pose(final_context).tolist(), "object_velocity": final_object_velocity, "last_three_settling_pose_max_drift_m": last_three_drift_m, "stable_release_thresholds": {"linear_speed_m_s_max": 0.01, "angular_speed_rad_s_max": 0.10, "last_three_pose_drift_m_max": 0.002}, "grasp_state": final_grasp, "complete_task_success": complete_task_success, "stable_release": stable_release}
        report["native_task_success"] = bool(final_native.get("task_success") is True)
        report["task_success"] = complete_task_success
        report["full_task_episode_complete"] = complete_task_success
        report["stable_release_at_stop"] = stable_release
        report["suffix_safety_monitoring_complete"] = bool(len(monitor_rows) == recorder.action_count and all(r["complete_25_substep_monitoring"] for r in monitor_rows))
        report["suffix_strict_safety_certified"] = bool(report["suffix_safety_monitoring_complete"] and all(r["safety_certifiable_action"] for r in monitor_rows))
        report["safety_monitoring_complete"] = False
        report["strict_safety_certified"] = False
        report["safety_success"] = False
        report["physical_execution"].update({"status": "complete_task_success" if report["task_success"] else "task_success_not_reached", "suffix_actions": recorder.action_count, "full_physics_substeps": recorder.recorded_physics_substeps, "monitoring_complete_actions": sum(r["complete_25_substep_monitoring"] for r in monitor_rows), "strict_safe_actions": sum(r["safety_certifiable_action"] for r in monitor_rows)})
        report["complete_trajectory_reconstruction"] = bool(report["task_success"] and report["prior_task_feasibility_prefix_chain"]["cross_archive_timestep_continuity_claimed"])
        report["new_trainable_episodes"] = 0
        report["episode_label"] = "task_success_unsafe_or_unknown_prefix" if report["task_success"] else "task_success_not_achieved"
        report["status"] = "task_success_only" if report["task_success"] and not report["safety_success"] else ("safe_task_success" if report["safety_success"] else "bounded_task_search_exhausted_without_success")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["status"] = "bounded_branch_stopped_or_blocked"
        if isinstance(report.get("physical_execution"), dict):
            report["physical_execution"]["status"] = "stopped_after_exception"
        if recorder is not None and bridge is not None:
            try:
                final_obs = bridge.request({"op": "observe", "include_snapshot_state": True})
                native_at_stop = bridge.request({"op": "native_task_status"})
                grasp_at_stop = bridge.request({"op": "grasp_state", "carried_binding": binding, "expected_instruction": INSTRUCTION})
                context_at_stop = bridge.request({"op": "task_success_first_context", "expected_instruction": INSTRUCTION, "target_predicate": ["on", OBJECT_ID, REGION_ID]})
                velocity_at_stop = _object_velocity(final_obs["snapshot_state"])
                report["failure_state"] = {
                    "timestep": final_obs.get("timestep"),
                    "native_task_status": native_at_stop,
                    "grasp_state": grasp_at_stop,
                    "target_object_pose_world_m": _object_pose(context_at_stop).tolist(),
                    "object_velocity": velocity_at_stop,
                }
                report["native_task_success"] = native_at_stop.get("task_success") is True
                report["task_success"] = False
                report["native_task_success_at_stop"] = report["native_task_success"]
                if isinstance(report.get("physical_execution"), dict):
                    report["physical_execution"]["status"] = (
                        "stopped_native_task_predicate_lost"
                        if not report["task_success"]
                        else "stopped_before_complete_retreat_and_settling"
                    )
                report["complete_trajectory_reconstruction"] = False
                report["full_task_episode_complete"] = False
                report["stable_release_at_stop"] = False
                report["safety_success"] = False
                report["new_trainable_episodes"] = 0
                report["episode_label"] = "native_predicate_true_but_suffix_incomplete_safety_uncertified" if report["native_task_success"] else "task_success_not_achieved"
            except Exception as state_exc:
                report["failure_state_error"] = f"{type(state_exc).__name__}: {state_exc}"
    finally:
        if action_log is not None:
            action_log.close()
        if bridge is not None:
            try:
                terminal = bridge.request({"op": "export_runtime_snapshot"})
                if terminal.get("status") == "complete" and isinstance(terminal.get("payload"), bytes):
                    p = output / "terminal_runtime_snapshot.pkl"
                    p.write_bytes(terminal["payload"])
                    terminal_envelope = pickle.loads(terminal["payload"])
                    terminal_snapshot = terminal_envelope.get("snapshot", {})
                    terminal_timestep = int(terminal_snapshot.get("task_bookkeeping", {}).get("timestep", -1))
                    terminal_state = np.asarray(terminal_snapshot.get("mujoco_state", []), dtype=np.float64)
                    report["terminal_snapshot"] = {
                        "path": str(p),
                        "sha256": _sha(terminal["payload"]),
                        "timestep": terminal_timestep,
                        "integration_state_sha256": hashlib.sha256(terminal_state.tobytes()).hexdigest() if terminal_state.size else None,
                        "recoverable": bool(
                            terminal_envelope.get("schema_version") == "gs_opd_runtime_snapshot.v1"
                            and terminal_timestep >= start_t
                            and terminal_state.ndim == 1
                            and terminal_state.size == expected_state.size
                            and np.isfinite(terminal_state).all()
                        ),
                    }
            except Exception as snap_exc:
                report["terminal_snapshot"] = {"recoverable": False, "error": f"{type(snap_exc).__name__}: {snap_exc}"}
            if recorder is not None:
                report["physical_actions"] = recorder.action_count
                report["recorded_physics_substeps"] = recorder.recorded_physics_substeps
                report["monitoring_complete_actions"] = sum(r["complete_25_substep_monitoring"] for r in monitor_rows)
                report["safety_certifiable_actions"] = sum(r["safety_certifiable_action"] for r in monitor_rows)
                report["suffix_safety_monitoring_complete"] = bool(recorder.action_count > 0 and report["monitoring_complete_actions"] == recorder.action_count)
                report["suffix_strict_safety_certified"] = bool(report.get("suffix_strict_safety_certified") and report["suffix_safety_monitoring_complete"])
                report["safety_monitoring_complete"] = False
                report["strict_safety_certified"] = False
                report["artifacts"] = {"physical_records": str(output / "physical_step_records.jsonl.gz"), "action_monitoring": str(output / "action_monitoring.jsonl.gz"), "dual_camera_video": str(output / "dual_camera_agentview_wrist_20fps.mp4"), "camera_frames": str(output / "dual_camera_frames"), "snapshots": recorder.snapshot_paths}
                report["physical_execution"] = report.get("physical_execution", {}) | {"suffix_actions": recorder.action_count, "physics_substeps": recorder.recorded_physics_substeps}
                try:
                    recorder.close()
                except Exception as close_exc:
                    report["recorder_close_error"] = f"{type(close_exc).__name__}: {close_exc}"
            try:
                connection.close()
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
            if log is not None:
                log.close()
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write(output / "task_report.json", report)
        report_path = output / "task_report.md"
        report_path.write_text(
            "# OA L0T0 candidate9 bounded full-task continuation\n\n"
            f"- Status: `{report.get('status')}`\n"
            f"- Task success: `{report.get('task_success')}`\n"
            f"- Native task success at stop: `{report.get('native_task_success_at_stop')}`\n"
            f"- Safety success: `{report.get('safety_success')}`\n"
            f"- Full task episode complete: `{report.get('full_task_episode_complete')}`\n"
            f"- Strict safety certified: `{report.get('strict_safety_certified')}`\n"
            f"- Physical suffix actions: `{report.get('physical_actions')}`\n"
            f"- New trainable episodes: `{report.get('new_trainable_episodes')}`\n"
            f"- Error / stop: `{report.get('error')}`\n"
            f"- Terminal snapshot: `{report.get('terminal_snapshot', {}).get('path')}`\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-source", type=Path, default=SOURCE_SCENE)
    parser.add_argument("--snapshot", type=Path, default=SOURCE_SNAPSHOT)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch-config", type=Path)
    parser.add_argument("--max-suffix-actions", type=int, default=HORIZON)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(api._json_safe({k: result.get(k) for k in ("status", "task_success", "safety_success", "physical_actions", "error", "terminal_snapshot")}), ensure_ascii=False, sort_keys=True))
    return 0 if result.get("task_success") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
