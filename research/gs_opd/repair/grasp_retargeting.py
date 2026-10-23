# Copyright 2026 The RLinf Authors.
"""Extract and retarget a physically verified object-relative grasp."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


def _transform(position: Any, quaternion_wxyz: Any) -> np.ndarray:
    position_array = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if position_array.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("object and EEF poses must be a 3-vector and wxyz quaternion")
    if not np.isfinite(position_array).all() or not np.isfinite(quaternion).all():
        raise ValueError("object and EEF poses must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("pose quaternion must have non-zero norm")
    q = quaternion / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    matrix[:3, 3] = position_array
    return matrix


def _json_array(value: np.ndarray) -> list[list[float]] | list[float]:
    return value.tolist()


@dataclass(frozen=True)
class RuntimeObjectBinding:
    """A task object's active Body/Geom identity supplied by the caller."""

    object_name: str
    body_id: int
    body_name: str
    body_ids: tuple[int, ...]
    geom_ids: tuple[int, ...]
    geom_names: tuple[str, ...]

    def as_bridge_binding(self) -> dict[str, Any]:
        return {
            "object_id": self.object_name,
            "body_id": self.body_id,
            "body_name": self.body_name,
            "geom_ids": list(self.geom_ids),
            "geom_names": list(self.geom_names),
        }


@dataclass(frozen=True)
class RetargetedGrasp:
    """Object-frame grasp pose and the source approach/timing profile."""

    eef_position_world_m: np.ndarray
    eef_quaternion_world_wxyz: np.ndarray
    pregrasp_position_world_m: np.ndarray
    approach_direction_object: np.ndarray
    approach_distance_m: float
    object_to_eef: np.ndarray
    translation_offset_object_m: np.ndarray
    rotation_delta_object_rad: np.ndarray


@dataclass(frozen=True)
class GraspRetargetingTemplate:
    """A grasp transform and close timing extracted from a successful trace."""

    source_trace_path: str
    source_trace_sha256: str
    source_object_name: str
    source_body_name: str
    source_body_id: int
    stable_source_steps_1based: tuple[int, ...]
    source_first_finger_contact_step_1based: int
    source_close_start_step_1based: int
    source_stable_confirmation_step_1based: int
    object_to_eef: np.ndarray
    approach_direction_object: np.ndarray
    approach_distance_m: float
    open_command: float
    close_schedule: tuple[float, ...]
    source_stable_contact_evidence: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GraspRetargetingTemplate:
        """Load a previously verified template without importing its actions."""
        if value.get("schema_version") != "gs_opd_grasp_retargeting_template.v1":
            raise ValueError("unsupported_grasp_retargeting_template_schema")
        matrix = np.asarray(value.get("object_to_eef"), dtype=np.float64)
        direction = np.asarray(value.get("approach_direction_object"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("grasp_template_object_to_eef_transform_is_invalid")
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("grasp_template_approach_direction_is_invalid")
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12 or abs(norm - 1.0) > 1e-4:
            raise ValueError("grasp_template_approach_direction_must_be_normalized")
        close_schedule = tuple(float(item) for item in value.get("close_schedule", ()))
        if not close_schedule or not np.isfinite(close_schedule).all():
            raise ValueError("grasp_template_close_schedule_is_invalid")
        source_steps = tuple(int(item) for item in value.get("stable_source_steps_1based", ()))
        evidence = tuple(dict(item) for item in value.get("source_stable_contact_evidence", ()))
        if len(source_steps) < 2 or len(evidence) != len(source_steps):
            raise ValueError("grasp_template_stable_contact_evidence_is_incomplete")
        return cls(
            source_trace_path=str(value["source_trace_path"]),
            source_trace_sha256=str(value["source_trace_sha256"]),
            source_object_name=str(value["source_object_name"]),
            source_body_name=str(value["source_body_name"]),
            source_body_id=int(value["source_body_id_resolved_from_trace"]),
            stable_source_steps_1based=source_steps,
            source_first_finger_contact_step_1based=int(
                value["source_first_finger_contact_step_1based"]
            ),
            source_close_start_step_1based=int(value["source_close_start_step_1based"]),
            source_stable_confirmation_step_1based=int(
                value["source_stable_confirmation_step_1based"]
            ),
            object_to_eef=matrix,
            approach_direction_object=direction / norm,
            approach_distance_m=float(value["approach_distance_m"]),
            open_command=float(value["open_command"]),
            close_schedule=close_schedule,
            source_stable_contact_evidence=evidence,
        )

    @classmethod
    def from_rollout_trace(
        cls,
        trace_path: str | Path,
        *,
        source_object_name: str,
        finger_geom_side_by_name: Mapping[str, str],
        required_stable_states: int = 3,
        close_threshold: float = 0.9,
        approach_lookback_steps: int = 20,
    ) -> GraspRetargetingTemplate:
        """Read the first contiguous bilateral grasp from a recorded trace.

        ``finger_geom_side_by_name`` must come from the active robot model or a
        recorded Geometry Inventory. Body and Geom IDs are resolved from each
        physical-state record rather than embedded in this component.
        """
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional research runtime
            raise RuntimeError("NumPy is required to extract a grasp template") from exc

        path = Path(trace_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if required_stable_states < 2:
            raise ValueError("required_stable_states must be at least two")
        if approach_lookback_steps < 2:
            raise ValueError("approach_lookback_steps must be at least two")
        side_by_name = {str(k): str(v) for k, v in finger_geom_side_by_name.items()}
        if not side_by_name or not set(side_by_name.values()) >= {"left", "right"}:
            raise ValueError("finger geometry mapping must include left and right sides")

        def parse_record(raw: Any, label: str) -> dict[str, Any]:
            value = json.loads(str(raw))
            if not isinstance(value, dict):
                raise ValueError(f"{label} must contain JSON objects")
            return value

        with np.load(path, allow_pickle=False) as archive:
            required_keys = {
                "executed_actions",
                "post_physical_state_json",
                "post_contacts_json",
            }
            if not required_keys <= set(archive.files):
                raise ValueError(
                    f"rollout trace is missing fields: {sorted(required_keys - set(archive.files))}"
                )
            actions = np.asarray(archive["executed_actions"], dtype=np.float64)
            physical_rows = archive["post_physical_state_json"]
            contact_rows = archive["post_contacts_json"]
        count = len(actions)
        if actions.shape != (count, 7) or len(physical_rows) != count or len(contact_rows) != count:
            raise ValueError("action, physical-state, and contact rows are not aligned")
        if not np.isfinite(actions).all():
            raise ValueError("source actions contain non-finite values")

        states = [parse_record(value, "physical state") for value in physical_rows]
        contacts = [parse_record(value, "contact record") for value in contact_rows]

        def object_state(state: dict[str, Any]) -> dict[str, Any]:
            object_states = state.get("object_states", {})
            matches = [
                value
                for key, value in object_states.items()
                if key.endswith(f":{source_object_name}")
                or value.get("body_name", "").removesuffix("_main") == source_object_name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one recorded source object named {source_object_name!r}, found {len(matches)}"
                )
            return matches[0]

        first_object = object_state(states[0])
        source_body_id = int(first_object["body_id"])
        source_body_name = str(first_object["body_name"])

        sides_by_step: list[set[str]] = []
        finger_contact_step: int | None = None
        stable_evidence: list[dict[str, Any]] = []
        for index, record in enumerate(contacts):
            sides: set[str] = set()
            for contact in record.get("contacts", []):
                body1, body2 = contact.get("body1", {}), contact.get("body2", {})
                geom1, geom2 = contact.get("geom1", {}), contact.get("geom2", {})
                body_matches = any(
                    int(body.get("id", -1)) == source_body_id
                    for body in (body1, body2)
                )
                if not body_matches:
                    continue
                other_geom = geom2 if int(body1.get("id", -1)) == source_body_id else geom1
                geom_name = other_geom.get("name")
                side = side_by_name.get(str(geom_name))
                if side in {"left", "right"}:
                    sides.add(side)
                    if finger_contact_step is None:
                        finger_contact_step = index + 1
            sides_by_step.append(sides)

        stable_start: int | None = None
        stable_end: int | None = None
        run = 0
        for index, sides in enumerate(sides_by_step):
            valid = sides >= {"left", "right"} and actions[index, 6] >= close_threshold
            run = run + 1 if valid else 0
            if run >= required_stable_states:
                stable_start = index - required_stable_states + 1
                stable_end = index
                break
        if stable_start is None or stable_end is None:
            raise ValueError("no contiguous closed bilateral grasp was found in the source trace")
        if finger_contact_step is None:
            raise ValueError("source trace has no target-object contact from mapped finger geoms")

        # Select the most recent uninterrupted positive close run ending at the
        # first monitor-confirmed stable state; earlier exploratory opens are excluded.
        close_start = stable_start
        while close_start > 0 and actions[close_start - 1, 6] > 0.0:
            close_start -= 1
        close_schedule = tuple(float(value) for value in actions[close_start : stable_end + 1, 6])
        if not close_schedule or max(close_schedule) < close_threshold:
            raise ValueError("source close timing does not reach the stable-grasp threshold")

        transforms = []
        for index in range(stable_start, stable_end + 1):
            object_record = object_state(states[index])
            robot_states = states[index].get("robot_states", [])
            if len(robot_states) != 1:
                raise ValueError("source trace must contain exactly one robot state")
            robot = robot_states[0]
            world_object = _transform(
                object_record["root_pos"], object_record["root_quat"]
            )
            world_eef = _transform(robot["eef_pos"], robot["eef_quat"])
            transforms.append(np.linalg.inv(world_object) @ world_eef)
            stable_evidence.append(
                {
                    "step_1based": index + 1,
                    "left_contact": True,
                    "right_contact": True,
                    "closed_action": float(actions[index, 6]),
                    "object_position_world_m": object_record["root_pos"],
                    "eef_position_world_m": robot["eef_pos"],
                }
            )
        object_to_eef = np.eye(4, dtype=np.float64)
        object_to_eef[:3, 3] = np.median(
            np.stack([value[:3, 3] for value in transforms]), axis=0
        )
        object_to_eef[:3, :3] = Rotation.from_matrix(
            np.stack([value[:3, :3] for value in transforms])
        ).mean().as_matrix()

        lookback_start = max(0, finger_contact_step - approach_lookback_steps - 1)
        lookback_end = finger_contact_step - 2
        if lookback_end - lookback_start < 1:
            raise ValueError("source trace does not contain enough pre-contact approach states")
        start_robot = states[lookback_start].get("robot_states", [])[0]
        end_robot = states[lookback_end].get("robot_states", [])[0]
        approach_world = (
            np.asarray(end_robot["eef_pos"], dtype=np.float64)
            - np.asarray(start_robot["eef_pos"], dtype=np.float64)
        )
        approach_distance = float(np.linalg.norm(approach_world))
        if not np.isfinite(approach_distance) or approach_distance <= 1e-5:
            raise ValueError("source pre-contact approach motion is too small to retarget")
        approach_reference = object_state(states[lookback_end])
        object_rotation = _transform(
            approach_reference["root_pos"], approach_reference["root_quat"]
        )[:3, :3]
        approach_object = object_rotation.T @ (approach_world / approach_distance)
        approach_object /= np.linalg.norm(approach_object)

        open_samples = actions[:finger_contact_step, 6]
        open_samples = open_samples[open_samples < 0.0]
        if not len(open_samples):
            raise ValueError("source trace has no recorded open-gripper approach command")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(
            source_trace_path=str(path),
            source_trace_sha256=digest,
            source_object_name=source_object_name,
            source_body_name=source_body_name,
            source_body_id=source_body_id,
            stable_source_steps_1based=tuple(
                range(stable_start + 1, stable_end + 2)
            ),
            source_first_finger_contact_step_1based=finger_contact_step,
            source_close_start_step_1based=close_start + 1,
            source_stable_confirmation_step_1based=stable_end + 1,
            object_to_eef=object_to_eef,
            approach_direction_object=approach_object,
            approach_distance_m=approach_distance,
            open_command=float(np.median(open_samples)),
            close_schedule=close_schedule,
            source_stable_contact_evidence=tuple(stable_evidence),
        )

    def retarget(
        self,
        target_position_world_m: Any,
        target_quaternion_world_wxyz: Any,
        *,
        translation_offset_object_m: Any = (0.0, 0.0, 0.0),
        rotation_delta_object_rad: Any = (0.0, 0.0, 0.0),
    ) -> RetargetedGrasp:
        """Map the sampled object-relative grasp onto a runtime object pose."""
        world_object = _transform(target_position_world_m, target_quaternion_world_wxyz)
        offset = np.asarray(translation_offset_object_m, dtype=np.float64)
        rotation_delta = np.asarray(rotation_delta_object_rad, dtype=np.float64)
        if offset.shape != (3,) or rotation_delta.shape != (3,):
            raise ValueError("retarget perturbations must be 3-vectors in object coordinates")
        if not np.isfinite(offset).all() or not np.isfinite(rotation_delta).all():
            raise ValueError("retarget perturbations must be finite")
        object_to_eef = self.object_to_eef.copy()
        object_to_eef[:3, 3] += offset
        object_to_eef[:3, :3] = object_to_eef[:3, :3] @ Rotation.from_rotvec(
            rotation_delta
        ).as_matrix()
        world_eef = world_object @ object_to_eef
        quaternion_xyzw = Rotation.from_matrix(world_eef[:3, :3]).as_quat()
        quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
        approach_world = world_object[:3, :3] @ self.approach_direction_object
        pregrasp = world_eef[:3, 3] - approach_world * self.approach_distance_m
        return RetargetedGrasp(
            eef_position_world_m=world_eef[:3, 3].copy(),
            eef_quaternion_world_wxyz=quaternion_wxyz,
            pregrasp_position_world_m=pregrasp,
            approach_direction_object=self.approach_direction_object.copy(),
            approach_distance_m=self.approach_distance_m,
            object_to_eef=object_to_eef,
            translation_offset_object_m=offset.copy(),
            rotation_delta_object_rad=rotation_delta.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "gs_opd_grasp_retargeting_template.v1",
            "source_trace_path": self.source_trace_path,
            "source_trace_sha256": self.source_trace_sha256,
            "source_object_name": self.source_object_name,
            "source_body_name": self.source_body_name,
            "source_body_id_resolved_from_trace": self.source_body_id,
            "stable_source_steps_1based": list(self.stable_source_steps_1based),
            "source_first_finger_contact_step_1based": self.source_first_finger_contact_step_1based,
            "source_close_start_step_1based": self.source_close_start_step_1based,
            "source_stable_confirmation_step_1based": self.source_stable_confirmation_step_1based,
            "object_to_eef": _json_array(self.object_to_eef),
            "approach_direction_object": _json_array(self.approach_direction_object),
            "approach_distance_m": self.approach_distance_m,
            "open_command": self.open_command,
            "close_schedule": list(self.close_schedule),
            "source_stable_contact_evidence": list(self.source_stable_contact_evidence),
        }


def resolve_runtime_object_binding(
    object_name: str,
    snapshot_state: Mapping[str, Any],
    geometry_inventory: Mapping[str, Any],
) -> RuntimeObjectBinding:
    """Resolve an object name against the current snapshot and geometry inventory."""
    object_states = snapshot_state.get("object_states", {})
    matches = [
        item
        for key, item in object_states.items()
        if key.endswith(f":{object_name}")
        or item.get("body_name", "").removesuffix("_main") == object_name
    ]
    if len(matches) != 1:
        raise ValueError(f"runtime object {object_name!r} resolved to {len(matches)} Bodies")
    target = matches[0]
    body_id, body_name = int(target["body_id"]), str(target["body_name"])
    bodies = geometry_inventory.get("bodies", [])
    body_by_id = {int(body["body_id"]): body for body in bodies}
    if body_id not in body_by_id or body_by_id[body_id].get("body_name") != body_name:
        raise ValueError("runtime target Body does not match the active Geometry Inventory")

    subtree = {body_id}
    changed = True
    while changed:
        changed = False
        for body in bodies:
            current_id = int(body["body_id"])
            parent_id = body.get("parent_body_id")
            if current_id not in subtree and parent_id in subtree:
                subtree.add(current_id)
                changed = True
    geometry_by_id = {int(item["geom_id"]): item for item in geometry_inventory.get("geometries", [])}
    geom_ids = sorted(
        geom_id
        for geom_id, item in geometry_by_id.items()
        if int(item.get("body_id", -1)) == body_id
    )
    if not geom_ids:
        raise ValueError("runtime target Body has no associated Geoms")
    return RuntimeObjectBinding(
        object_name=object_name,
        body_id=body_id,
        body_name=body_name,
        body_ids=tuple(sorted(subtree)),
        geom_ids=tuple(geom_ids),
        geom_names=tuple(str(geometry_by_id[geom_id]["geom_name"]) for geom_id in geom_ids),
    )


def build_standardized_action_record(
    *,
    index: int,
    phase: str,
    action_7d: Any,
    task: str,
    target_object: str,
    snapshot_source: Mapping[str, Any],
    observation_row: int,
    physical_state: Mapping[str, Any],
    native_task_status: Mapping[str, Any],
    contacts: Mapping[str, Any],
    grasp_monitor_sample: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Create a JSONL record paired with image/state arrays in the NPZ."""
    action = np.asarray(action_7d, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("standardized records require a finite 7-D action")
    return {
        "schema_version": "gs_opd_standardized_action_record.v1",
        "index": int(index),
        "phase": phase,
        "task": task,
        "target_object": target_object,
        "action_source": "expert_osc",
        "actual_action_7d": action.tolist(),
        "observation": {
            "storage": "physical_action_observation_contact.npz",
            "row": int(observation_row),
            "camera_keys": ["agentview_image", "robot0_eye_in_hand_image"],
            "state_source": "physical_state_before_action_and_after_action",
        },
        "snapshot_source": dict(snapshot_source),
        "physical_state": dict(physical_state),
        "task_predicates": dict(native_task_status),
        "contacts": dict(contacts),
        "grasp_monitor": dict(grasp_monitor_sample) if grasp_monitor_sample is not None else None,
    }
