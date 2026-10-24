# Copyright 2026 The RLinf Authors.
"""Bounded task-success-first grasp candidate generation and audit helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


DEMO_RECORDS = Path(
    "/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_final_placement_demo_20261022_attempt3/evidence/complete_standardized_action_records.jsonl"
)
TRACE = Path(
    "/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_base_pi05_rollout_400_20261022_attempt2/rollout_trace.npz"
)
TEMPLATE = Path(
    "/opt/data/private/gs_opd/rollouts/l0t2_round1_jointspace_rrt_connect_20261023_attempt1/source/source_grasp_template.json"
)


def rotation_vector_matrix(vector: Any) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return Rotation.from_rotvec(vector).as_matrix()


def read_task_status(row: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any] | None:
    reference = row.get("task_status_ref")
    if not isinstance(reference, dict):
        return None
    path = str(reference.get("npz") or reference.get("archive"))
    if path not in cache:
        cache[path] = np.load(path, allow_pickle=False)
    key = str(reference.get("key", "")).split(":")[-1]
    if key not in cache[path]:
        return None
    try:
        value = json.loads(str(cache[path][key][int(reference["row"])]))
        if not isinstance(value, dict):
            return None
        return value
    except Exception:
        return None


def generate_grasp_candidates(
    template: dict[str, Any],
    stable_qpos: Any,
    approach_qpos: Any,
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    """Generate at most 24 materially distinct object-relative approaches."""
    if limit < 1 or limit > 24:
        raise ValueError("offline_candidate_limit_must_be_between_1_and_24")
    base = np.asarray(template["object_to_eef"], dtype=np.float64)
    direction = np.asarray(template["approach_direction_object"], dtype=np.float64)
    stable = np.asarray(stable_qpos, dtype=np.float64)
    approach = np.asarray(approach_qpos, dtype=np.float64)
    if base.shape != (4, 4) or stable.shape != (7,) or approach.shape != (7,):
        raise ValueError("historical_grasp_candidate_priors_are_malformed")

    # Each family changes the approach side/orientation; paired offset variants
    # test distinct pinch locations. Seeds vary wrist/elbow redundancy without
    # restricting the solver to the historical arm posture.
    specs: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("historical_prior", np.zeros(3), np.zeros(3)),
    ]
    for axis, degrees in ((2, 30), (2, -30), (2, 60), (2, -60), (2, 90), (2, -90), (2, 150), (2, -150), (2, 180),
                          (0, 20), (0, -20), (1, 20), (1, -20), (0, 35), (0, -35), (1, 35), (1, -35)):
        rotvec = np.zeros(3)
        rotvec[axis] = np.deg2rad(degrees)
        specs.append((f"axis{axis}_rot{degrees:+d}", rotvec, np.zeros(3)))
    for name, rotvec, offset in (
        ("yaw30_shiftx+10", [0, 0, np.deg2rad(30)], [0.010, 0, 0]),
        ("yaw30_shiftx-10", [0, 0, np.deg2rad(30)], [-0.010, 0, 0]),
        ("yaw-30_shifty+10", [0, 0, np.deg2rad(-30)], [0, 0.010, 0]),
        ("yaw-30_shifty-10", [0, 0, np.deg2rad(-30)], [0, -0.010, 0]),
        ("pitch20_shiftx+8", [np.deg2rad(20), 0, 0], [0.008, 0, 0]),
        ("pitch-20_shiftx-8", [np.deg2rad(-20), 0, 0], [-0.008, 0, 0]),
    ):
        specs.append((name, np.asarray(rotvec), np.asarray(offset)))
    specs = specs[:limit]

    candidates = []
    seed_deltas = (
        np.zeros(7),
        np.asarray([0.0, -0.35, 0.0, 0.35, 0.0, -0.20, 0.65]),
        np.asarray([0.0, 0.35, 0.0, -0.35, 0.0, 0.20, -0.65]),
        np.asarray([0.0, 0.0, 0.45, 0.0, -0.55, 0.0, 0.35]),
    )
    for index, (name, rotvec, offset) in enumerate(specs):
        delta = Rotation.from_rotvec(rotvec).as_matrix()
        transform = base.copy()
        transform[:3, :3] = base[:3, :3] @ delta
        transform[:3, 3] = base[:3, 3] + offset
        approach_direction = delta.T @ direction
        seed = stable + seed_deltas[index % len(seed_deltas)]
        if index % 2:
            seed = approach + 0.65 * (seed - stable)
        candidates.append({
            "candidate_id": f"tsfs_{index:02d}_{name}",
            "object_to_eef": transform.tolist(),
            "approach_direction_object": (approach_direction / np.linalg.norm(approach_direction)).tolist(),
            "approach_distance_m": float(template["approach_distance_m"]),
            "ik_seed_qpos": [stable.tolist(), seed.tolist()],
            "rotation_delta_object_rad": np.asarray(rotvec).tolist(),
            "translation_offset_object_m": offset.tolist(),
            "search_family": name,
            "source_prior": "verified_bilateral_historical_grasp_transform",
        })
    return candidates


def extract_read_only_reference(output_path: str | Path) -> dict[str, Any]:
    """Summarize the archived 1093-step demo and collision/milestone evidence."""
    with DEMO_RECORDS.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 1093:
        raise ValueError(f"expected_1093_archived_transitions_got_{len(rows)}")
    if any(row["episode_timestep_after"] != index + 1 for index, row in enumerate(rows)):
        raise ValueError("archived_action_chain_is_not_contiguous")

    archive_cache: dict[str, Any] = {}
    array_cache: dict[tuple[str, str], np.ndarray] = {}

    def archive_value(reference: dict[str, Any], key: str, row: int):
        path = str(reference.get("archive") or reference.get("npz"))
        if path not in archive_cache:
            archive_cache[path] = np.load(path, allow_pickle=False)
        array_key = (path, key)
        if array_key not in array_cache:
            array_cache[array_key] = archive_cache[path][key]
        return array_cache[array_key][row]

    def raw_state(step: int) -> dict[str, Any]:
        reference = rows[step - 1]["physical_state_ref"]
        keys = reference.get("keys", [])
        if isinstance(keys, dict):
            keys = list(keys.values())
        row = int(reference["row"])
        physical_key = next((key for key in keys if key.endswith("physical_state_json")), None)
        if physical_key:
            return json.loads(str(archive_value(reference, physical_key, row)))
        robot_key = next((key for key in keys if key.startswith("post_robot_state_json")), None)
        objects_key = next((key for key in keys if key.startswith("post_all_object_states_json") or key == "all_object_states_json"), None)
        if robot_key and objects_key:
            robot = json.loads(str(archive_value(reference, robot_key, row)))
            objects = json.loads(str(archive_value(reference, objects_key, row)))
            return {"robot_states": [robot], "object_states": objects}
        array_keys = set(keys)
        if "post_robot0_joint_qpos" in array_keys:
            robot = {
                "joint_qpos": np.asarray(archive_value(reference, "post_robot0_joint_qpos", row)).tolist(),
                "eef_pos": np.asarray(archive_value(reference, "post_robot0_eef_pos", row)).tolist(),
                "eef_quat": np.asarray(archive_value(reference, "post_robot0_eef_quat_wxyz", row)).tolist(),
                "gripper_qpos": np.asarray(archive_value(reference, "post_robot0_gripper_qpos", row)).tolist(),
            }
            objects = {}
            for object_name in ("moka_pot_2",):
                pos_key, quat_key = f"post_{object_name}_root_pos", f"post_{object_name}_root_quat_wxyz"
                if pos_key in array_keys and quat_key in array_keys:
                    objects[f"objects_dict:{object_name}"] = {
                        "root_pos": np.asarray(archive_value(reference, pos_key, row)).tolist(),
                        "root_quat": np.asarray(archive_value(reference, quat_key, row)).tolist(),
                    }
            return {"robot_states": [robot], "object_states": objects, "available_physical_fields": sorted(array_keys)}
        raise ValueError(f"milestone_physical_state_is_incomplete:{step}:{keys}")

    def object_state(step: int, name: str) -> dict[str, Any]:
        state = raw_state(step)
        matches = [
            value for key, value in state["object_states"].items()
            if key.endswith(":" + name) or value.get("body_name", "").removesuffix("_main") == name
        ]
        if len(matches) > 1:
            raise ValueError(f"object_state_not_unique:{step}:{name}")
        return matches[0] if matches else {"unavailable": "not_recorded_in_source_segment"}

    def pose_matrix(position: Any, quaternion_wxyz: Any) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("milestone_pose_has_invalid_shape")
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(
            [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        ).as_matrix()
        matrix[:3, 3] = position
        return matrix

    def target_gripper_contacts(step: int, object_name: str) -> dict[str, Any]:
        reference = rows[step - 1]["contact_ref"]
        keys = reference.get("keys", [])
        if isinstance(keys, dict):
            keys = list(keys.values())
        contact_key = next((key for key in keys if key.endswith("contacts_json")), None)
        if contact_key is None:
            return {"status": "unavailable", "contacts": []}
        contact_data = json.loads(
            str(archive_value(reference, contact_key, int(reference["row"])))
        )
        matched = []
        for contact in contact_data.get("contacts", []):
            body_names = [
                str(contact.get(f"body{index}", {}).get("name", ""))
                for index in (1, 2)
            ]
            geom_names = [
                str(contact.get(f"geom{index}", {}).get("name", ""))
                for index in (1, 2)
            ]
            if (
                any(object_name in name for name in body_names)
                and any("finger" in name.lower() for name in body_names + geom_names)
            ):
                matched.append({
                    "bodies": body_names,
                    "geometries": geom_names,
                    "force_and_torque": contact.get("force_and_torque"),
                    "position_world_m": contact.get("position"),
                })
        return {"status": contact_data.get("status"), "contacts": matched}

    monitor_rows = []
    coverage_counts = {"complete_25_substep": 0, "incomplete_or_unknown": 0}
    for step, transition in enumerate(rows, 1):
        reference = transition["contact_ref"]
        keys = reference.get("keys", [])
        if isinstance(keys, dict):
            keys = list(keys.values())
        monitor_key = next((key for key in keys if "monitor" in key), None)
        if monitor_key is None:
            coverage_counts["incomplete_or_unknown"] += 1
            continue
        try:
            monitor = json.loads(str(archive_value(reference, monitor_key, int(reference["row"]))))
        except Exception:
            coverage_counts["incomplete_or_unknown"] += 1
            continue
        if monitor.get("captured_physics_substeps") == 25 and monitor.get("contact_substeps") == 25:
            coverage_counts["complete_25_substep"] += 1
        else:
            coverage_counts["incomplete_or_unknown"] += 1
        for event in monitor.get("events", []):
            bodies = event.get("body_pair", [])
            names = sorted(str(item.get("name")) for item in bodies)
            if event.get("classification") == "robot_scene" and any("ginger_beer" in name for name in names):
                monitor_rows.append({"episode_timestep_1based": step, "pair": names, "event": event})
    unsafe_steps = sorted({item["episode_timestep_1based"] for item in monitor_rows})
    intervals = []
    for step in unsafe_steps:
        if not intervals or step > intervals[-1][1] + 1:
            intervals.append([step, step])
        else:
            intervals[-1][1] = step

    def state_delta(first: int, last: int, name: str) -> dict[str, Any]:
        a, b = object_state(first, name), object_state(last, name)
        if "root_pos" not in a or "root_pos" not in b:
            return {"available": False, "reason": "source_segment_did_not_record_this_object_pose"}
        pos_a, pos_b = np.asarray(a["root_pos"]), np.asarray(b["root_pos"])
        quat_a, quat_b = np.asarray(a["root_quat"]), np.asarray(b["root_quat"])
        dot = min(1.0, abs(float(np.dot(quat_a / np.linalg.norm(quat_a), quat_b / np.linalg.norm(quat_b)))))
        return {
            "from_position_m": pos_a.tolist(), "to_position_m": pos_b.tolist(),
            "translation_delta_m": float(np.linalg.norm(pos_b - pos_a)),
            "rotation_delta_rad": float(2 * np.arccos(dot)),
        }

    milestones = {
        "pre_collision_safe_snapshot_t10": 10,
        "last_state_before_first_robot_ginger_beer_contact_t20": 20,
        "first_unsafe_contact_t21": 21,
        "last_unsafe_contact_t56": 56,
        "historical_pregrasp_t118": 118,
        "first_finger_contact_t119": 119,
        "stable_bilateral_grasp_t141": 141,
        "stable_grasp_plus_30mm_lift_t156": 156,
        "native_pot2_placement_success_t719": 719,
        "native_pot1_stable_grasp_t854": 854,
        "pot1_final_placement_branch_start_t1021": 1021,
        "first_complete_task_success_t1053": 1053,
        "final_stable_success_state_t1093": 1093,
    }
    object_names = ["bottle_of_ginger_beer_1", "moka_pot_2", "moka_pot_1", "white_place_box_1"]
    comparisons = {}
    for name in object_names:
        comparisons[name] = {
            "t10_to_t20": state_delta(10, 20, name),
            "t20_to_t56": state_delta(20, 56, name),
            "t20_to_t118": state_delta(20, 118, name),
            "t118_to_t141": state_delta(118, 141, name),
        }
    max_contact_force = max(
        (float(item["event"].get("max_force_norm", 0.0)) for item in monitor_rows),
        default=0.0,
    )

    snapshot_candidates = {
        "safe_t10_start": "/opt/data/private/gs_opd/rollouts/l0t2_round1_safety_recovery_20261023_attempt1/evidence/raw_pi05_t10_runtime_snapshot.pkl",
        "stable_grasp_plus_30mm_lift_t156": None,
        "pot2_native_success_t719": "/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_mokapot2_near_goal_osc_recovery_20261022_attempt1/evidence/moka_pot_2_native_goal_success_runtime_snapshot.pkl",
        "pot1_stable_grasp_t854": "/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_final_placement_demo_20261022_attempt2/evidence/source_t854_retargeted_grasp_runtime_snapshot.pkl",
        "pot1_placement_start_t1021": None,
        "complete_task_success_t1053": None,
        "final_stable_t1093": "/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_final_placement_demo_20261022_attempt3/evidence/moka_pot_1_complete_task_success_runtime_snapshot.pkl",
    }
    snapshots = {}
    for name, path in snapshot_candidates.items():
        file = Path(path) if path else None
        snapshots[name] = {
            "path": str(file) if file else None,
            "available": bool(file and file.is_file()),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest() if file and file.is_file() else None,
            "note": None if file and file.is_file() else "no_exact_milestone_snapshot_archived",
        }

    snapshot_at_timestep = {
        10: "safe_t10_start",
        719: "pot2_native_success_t719",
        854: "pot1_stable_grasp_t854",
        1093: "final_stable_t1093",
    }
    native_predicates = {
        name: read_task_status(rows[step - 1], archive_cache)
        for name, step in milestones.items()
    }
    milestone_records = {}
    for name, step in milestones.items():
        state = raw_state(step)
        robot = state["robot_states"][0]
        object_relative_to_eef = {}
        if robot.get("eef_pos") is not None and robot.get("eef_quat") is not None:
            eef_world = pose_matrix(robot["eef_pos"], robot["eef_quat"])
            for object_name in ("moka_pot_2", "moka_pot_1"):
                target = object_state(step, object_name)
                if target.get("root_pos") is not None and target.get("root_quat") is not None:
                    object_world = pose_matrix(target["root_pos"], target["root_quat"])
                    object_relative_to_eef[object_name] = (
                        np.linalg.inv(object_world) @ eef_world
                    ).tolist()
        snapshot_name = snapshot_at_timestep.get(step)
        milestone_records[name] = {
            "timestep": step,
            "robot_state": robot,
            "objects": {obj: object_state(step, obj) for obj in object_names},
            "object_to_eef_transform_object_frame": object_relative_to_eef,
            "target_finger_contacts": target_gripper_contacts(step, "moka_pot_2"),
            "native_predicate_evidence": native_predicates.get(name),
            "snapshot": (
                snapshots[snapshot_name]
                if snapshot_name is not None
                else {
                    "path": None,
                    "available": False,
                    "sha256": None,
                    "note": "no_exact_milestone_snapshot_archived",
                }
            ),
            "action_status": {
                "reward": rows[step - 1]["reward"],
                "done": rows[step - 1]["done"],
            },
        }

    report = {
        "schema": "gs_opd_task_success_first_reference_audit.v2",
        "read_only": True,
        "transition_count": len(rows),
        "first_native_complete_task_success_timestep": 1053,
        "unsafe_robot_ginger_beer_intervals_1based": intervals,
        "unsafe_action_count": len(unsafe_steps),
        "unsafe_pair_event_row_count": len(monitor_rows),
        "safety_monitor_coverage": {
            "archived_transitions": len(rows),
            "per_action_expected_physics_substeps": 25,
            **coverage_counts,
            "verified_unsafe_interval_actions": len(unsafe_steps),
        },
        "object_pose_deltas": comparisons,
        "milestones": milestone_records,
        "native_predicates": native_predicates,
        "maximum_robot_ginger_beer_contact_force_norm_n": max_contact_force,
        "maximum_robot_ginger_beer_contact_force_norm_n": max_contact_force,
        "snapshots": snapshots,
        "source_sha256": {
            "demo_records": hashlib.sha256(DEMO_RECORDS.read_bytes()).hexdigest(),
            "student_trace": hashlib.sha256(TRACE.read_bytes()).hexdigest(),
            "grasp_template": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
        },
        "robot_ginger_beer_contact_events": monitor_rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
