# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Real LIBERO scene model I/O and Panda state/transform checks.

Shared by the isolated Mink physical execution experiment. Running this module
only exports the real model for its parent process; no environment actions run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE = (
    REPO_ROOT / "logs/gs_opd/trajectory_bundle_l0t0_physical_acceptance_v5.json"
)
DEFAULT_REPORT = REPO_ROOT / "logs/gs_opd/mink_kinematic_feasibility_v5.json"
DEFAULT_LIBERO_PYTHON = Path("/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python")
DEFAULT_LIBERO_ROOT = Path("/workspace/zy_workspace/LIBERO-Safety")
DEFAULT_LIBERO_CONFIG = Path("/workspace/zy_workspace/.libero_safety_config")

FRAME_NAME = "gripper0_grip_site"
FRAME_TYPE = "site"
ARM_VELOCITY_LIMITS_RAD_S = (2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610)
ARM_VELOCITY_LIMIT_SOURCE = "Franka official panda joint_limits.yaml; see source_url"
ARM_VELOCITY_LIMIT_SOURCE_URL = (
    "https://github.com/frankarobotics/franka_ros/blob/develop/"
    "franka_description/robots/panda/joint_limits.yaml"
)
MODEL_SIGNATURE_FIELDS = (
    "body_parentid",
    "body_pos",
    "body_quat",
    "body_mocapid",
    "qpos0",
    "jnt_type",
    "jnt_bodyid",
    "jnt_pos",
    "jnt_axis",
    "jnt_qposadr",
    "geom_bodyid",
    "geom_type",
    "geom_dataid",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_rbound",
    "geom_contype",
    "geom_conaffinity",
    "geom_margin",
    "pair_geom1",
    "pair_geom2",
    "pair_margin",
    "mesh_vertadr",
    "mesh_vertnum",
    "mesh_vert",
    "mesh_face",
    "site_bodyid",
    "site_pos",
    "site_quat",
    "site_size",
    "site_type",
)


def _as_finite_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    shape_matches = array.ndim == len(shape) and all(
        expected == -1 or actual == expected
        for actual, expected in zip(array.shape, shape)
    )
    if not shape_matches or not np.isfinite(array).all():
        raise ValueError(f"{label}_shape_or_values_invalid")
    return array


def quaternion_normalized_wxyz(value: Any, label: str = "quaternion") -> np.ndarray:
    quat = _as_finite_array(value, (4,), label)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{label}_norm_invalid")
    return quat / norm


def quaternion_multiply_wxyz(left: Any, right: Any) -> np.ndarray:
    w1, x1, y1, z1 = quaternion_normalized_wxyz(left, "left_quaternion")
    w2, x2, y2, z2 = quaternion_normalized_wxyz(right, "right_quaternion")
    return quaternion_normalized_wxyz(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_rotate_wxyz(quaternion: Any, vector: Any) -> np.ndarray:
    quat = quaternion_normalized_wxyz(quaternion)
    vec = _as_finite_array(vector, (3,), "vector")
    w, x, y, z = quat
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation @ vec


def derive_eef_target(
    sphere_center_world_m: Any,
    object_quaternion_world_wxyz: Any,
    eef_sphere_center_relative_position_body_local_m: Any,
    eef_body_relative_quaternion_local_wxyz: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the archived rigid object-to-EEF transform to a sphere-center path."""
    sphere_center = _as_finite_array(sphere_center_world_m, (3,), "sphere_center")
    object_quaternion = quaternion_normalized_wxyz(
        object_quaternion_world_wxyz, "object_quaternion"
    )
    local_offset = _as_finite_array(
        eef_sphere_center_relative_position_body_local_m, (3,), "eef_local_offset"
    )
    relative_quaternion = quaternion_normalized_wxyz(
        eef_body_relative_quaternion_local_wxyz, "eef_relative_quaternion"
    )
    return (
        sphere_center + quaternion_rotate_wxyz(object_quaternion, local_offset),
        quaternion_multiply_wxyz(object_quaternion, relative_quaternion),
    )


def compute_model_signature(model: Any) -> str:
    """Mirror Safety Bridge's compiled-model signature without importing Bridge."""
    digest = hashlib.sha256()
    for name in MODEL_SIGNATURE_FIELDS:
        values = np.ascontiguousarray(getattr(model, name))
        digest.update(name.encode())
        digest.update(str((values.dtype, values.shape)).encode())
        digest.update(values.tobytes())
    digest.update(bytes(model.names))
    digest.update(np.asarray(model.opt.gravity, dtype=np.float64).tobytes())
    return digest.hexdigest()


def select_archived_routes(archive: dict[str, Any]) -> list[dict[str, Any]]:
    if archive.get("decision_step") != 55:
        raise ValueError("archive_not_step_55")
    tracking = archive.get("tracking")
    if not isinstance(tracking, dict):
        raise ValueError("tracking_record_missing")
    candidates = tracking.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("tracking_candidates_missing")
    accepted = [
        candidate
        for candidate in candidates
        if candidate.get("budget_status") == "accepted"
        and candidate.get("physical_tracking_performed") is True
        and isinstance(candidate.get("reference_path"), list)
        and len(candidate["reference_path"]) >= 2
    ]
    if len(accepted) != 3:
        raise ValueError(f"expected_three_archived_routes_found_{len(accepted)}")
    directions = [candidate.get("direction") for candidate in accepted]
    if len(set(directions)) != 3 or set(directions) != {"left", "right", "above"}:
        raise ValueError("archived_route_directions_incomplete_or_duplicated")
    return accepted


def model_hash_compatibility(expected: Any, actual: Any) -> tuple[bool, str | None]:
    if not isinstance(expected, str) or len(expected) != 64:
        return False, "archive_model_signature_missing_or_invalid"
    if not isinstance(actual, str) or len(actual) != 64:
        return False, "compiled_model_signature_missing_or_invalid"
    if expected != actual:
        return False, "compiled_model_signature_mismatch"
    return True, None


def _validate_model_and_initial_state(
    mujoco: Any,
    model: Any,
    archive: dict[str, Any],
    export_metadata: dict[str, Any],
) -> dict[str, Any]:
    planning = archive.get("planning_geometry") or {}
    osc = archive.get("active_osc_configuration") or {}
    robot_configs = osc.get("robots") or []
    if len(robot_configs) != 1:
        raise ValueError("archive_active_panda_configuration_missing_or_ambiguous")
    robot_config = robot_configs[0]
    joint_names = robot_config.get("joint_names")
    joint_ranges = robot_config.get("joint_ranges")
    joint_limited = robot_config.get("joint_limited")
    if not (
        isinstance(joint_names, list)
        and len(joint_names) == 7
        and isinstance(joint_ranges, list)
        and len(joint_ranges) == 7
        and isinstance(joint_limited, list)
        and len(joint_limited) == 7
    ):
        raise ValueError("archive_panda_joint_contract_invalid")

    expected_hash = planning.get("model_geometry_sha256")
    actual_hash = compute_model_signature(model)
    hash_matches, hash_error = model_hash_compatibility(expected_hash, actual_hash)
    version_matches = str(mujoco.__version__) == str(planning.get("mujoco_version"))
    task_matches = export_metadata.get("instruction") == planning.get("instruction")
    frame_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, FRAME_NAME))
    if frame_id < 0:
        raise ValueError("archived_eef_site_missing_from_compiled_model")

    integration = _as_finite_array(
        planning.get("mujoco_integration_state"),
        (int(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)),),
        "mujoco_integration_state",
    )
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)

    start_state = None
    candidates = select_archived_routes(archive)
    for candidate in candidates:
        state = candidate.get("start_state")
        if not isinstance(state, dict):
            raise ValueError("candidate_start_state_missing")
        if start_state is None:
            start_state = state
        elif state.get("mujoco_integration_state") != start_state.get(
            "mujoco_integration_state"
        ):
            raise ValueError("candidate_start_states_do_not_match")
    assert start_state is not None
    robot_states = start_state.get("robot_states") or []
    if len(robot_states) != 1:
        raise ValueError("archived_start_robot_state_missing_or_ambiguous")
    recorded_robot = robot_states[0]
    joint_ids = []
    actual_ranges = []
    actual_joint_qpos = []
    actual_joint_qvel = []
    joint_errors = []
    for index, name in enumerate(joint_names):
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if joint_id < 0:
            raise ValueError(f"panda_joint_missing:{name}")
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"panda_joint_not_hinge:{name}")
        if not bool(model.jnt_limited[joint_id]) or joint_limited[index] is not True:
            raise ValueError(f"panda_joint_not_limited:{name}")
        actual_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
        expected_range = _as_finite_array(
            joint_ranges[index], (2,), f"joint_range_{name}"
        )
        if not np.allclose(actual_range, expected_range, rtol=0.0, atol=1e-6):
            raise ValueError(f"panda_joint_range_mismatch:{name}")
        qpos_index = int(model.jnt_qposadr[joint_id])
        dof_index = int(model.jnt_dofadr[joint_id])
        actual_q = float(data.qpos[qpos_index])
        actual_v = float(data.qvel[dof_index])
        recorded_q = float(recorded_robot["joint_qpos"][index])
        recorded_v = float(recorded_robot["joint_qvel"][index])
        joint_ids.append(joint_id)
        actual_ranges.append(actual_range.tolist())
        actual_joint_qpos.append(actual_q)
        actual_joint_qvel.append(actual_v)
        joint_errors.append(
            {
                "joint": name,
                "qpos_abs_error_rad": abs(actual_q - recorded_q),
                "qvel_abs_error_rad_s": abs(actual_v - recorded_v),
            }
        )

    actual_eef_pos = np.asarray(data.site_xpos[frame_id], dtype=np.float64)
    actual_eef_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(actual_eef_quat, data.site_xmat[frame_id])
    recorded_eef_pos = _as_finite_array(
        recorded_robot.get("eef_pos"), (3,), "recorded_eef_pos"
    )
    recorded_eef_quat = quaternion_normalized_wxyz(
        recorded_robot.get("eef_quat"), "recorded_eef_quat"
    )
    eef_position_error = float(np.max(np.abs(actual_eef_pos - recorded_eef_pos)))
    eef_quaternion_error = float(
        min(
            np.max(np.abs(actual_eef_quat - recorded_eef_quat)),
            np.max(np.abs(actual_eef_quat + recorded_eef_quat)),
        )
    )
    max_joint_q_error = max(item["qpos_abs_error_rad"] for item in joint_errors)
    max_joint_v_error = max(item["qvel_abs_error_rad_s"] for item in joint_errors)

    binding_errors: list[str] = []
    scene = archive.get("scene") or {}
    if scene.get("state_id") != planning.get("state_id"):
        binding_errors.append("archive_scene_and_planning_state_ids_differ")
    scene_objects = [scene.get("carried"), *(scene.get("obstacles") or [])]
    if not scene_objects or any(not isinstance(item, dict) for item in scene_objects):
        binding_errors.append("archive_scene_object_bindings_missing")
    else:
        for item in scene_objects:
            body_id = item.get("body_id")
            body_name = item.get("body_name")
            if not isinstance(body_id, int) or not isinstance(body_name, str):
                binding_errors.append("archive_scene_body_identity_invalid")
                continue
            actual_body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if actual_body_name != body_name:
                binding_errors.append(f"scene_body_name_mismatch:{body_name}")
            geom_ids = item.get("geom_ids")
            geom_names = item.get("geom_names")
            if (
                not isinstance(geom_ids, list)
                or not isinstance(geom_names, list)
                or len(geom_ids) != len(geom_names)
                or not geom_ids
            ):
                binding_errors.append(f"scene_geom_binding_invalid:{body_name}")
                continue
            for geom_id, geom_name in zip(geom_ids, geom_names):
                if (
                    not isinstance(geom_id, int)
                    or geom_id < 0
                    or geom_id >= model.ngeom
                ):
                    binding_errors.append(f"scene_geom_id_out_of_range:{body_name}")
                    break
                actual_geom_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                if (
                    actual_geom_name != geom_name
                    or int(model.geom_bodyid[geom_id]) != body_id
                ):
                    binding_errors.append(
                        f"scene_geom_identity_or_owner_mismatch:{body_name}"
                    )
                    break

    carried_binding = scene.get("carried") or {}
    carried_body_id = carried_binding.get("body_id")
    archived_carried_states = [
        value
        for value in (start_state.get("object_states") or {}).values()
        if value.get("body_id") == carried_body_id
    ]
    sphere_center_error = None
    carried_pose_error = None
    if len(archived_carried_states) != 1 or not isinstance(carried_body_id, int):
        binding_errors.append("carried_object_archived_pose_missing_or_ambiguous")
    else:
        carried_state = archived_carried_states[0]
        current_body_pos = np.asarray(data.xpos[carried_body_id], dtype=np.float64)
        current_body_quat = np.asarray(data.xquat[carried_body_id], dtype=np.float64)
        archived_body_pos = _as_finite_array(
            carried_state.get("root_pos"), (3,), "carried_root_pos"
        )
        archived_body_quat = quaternion_normalized_wxyz(
            carried_state.get("root_quat"), "carried_root_quat"
        )
        body_quat_error = float(
            min(
                np.max(np.abs(current_body_quat - archived_body_quat)),
                np.max(np.abs(current_body_quat + archived_body_quat)),
            )
        )
        carried_pose_error = max(
            float(np.max(np.abs(current_body_pos - archived_body_pos))), body_quat_error
        )
        offset = _as_finite_array(
            candidates[0].get("sphere_center_body_local_offset_m"),
            (3,),
            "sphere_center_body_local_offset",
        )
        current_sphere_center = current_body_pos + quaternion_rotate_wxyz(
            current_body_quat, offset
        )
        recorded_sphere_center = _as_finite_array(
            candidates[0].get("start_object_sphere_center_world_m"),
            (3,),
            "recorded_sphere_center",
        )
        sphere_center_error = float(
            np.max(np.abs(current_sphere_center - recorded_sphere_center))
        )
        if carried_pose_error > 1e-6:
            binding_errors.append("carried_root_pose_mismatch")
        if sphere_center_error > 1e-6:
            binding_errors.append("carried_sphere_center_mismatch")

    errors = []
    if not version_matches:
        errors.append("mujoco_version_mismatch")
    if not task_matches:
        errors.append("active_task_instruction_mismatch")
    if hash_error:
        errors.append(hash_error)
    if max_joint_q_error > 1e-6 or max_joint_v_error > 1e-6:
        errors.append("archived_panda_joint_state_mismatch")
    if eef_position_error > 1e-6 or eef_quaternion_error > 1e-6:
        errors.append("archived_eef_site_pose_mismatch")
    errors.extend(binding_errors)
    return {
        "status": "compatible" if not errors else "incompatible",
        "errors": errors,
        "mujoco_version": str(mujoco.__version__),
        "archive_mujoco_version": planning.get("mujoco_version"),
        "task_instruction": export_metadata.get("instruction"),
        "archive_instruction": planning.get("instruction"),
        "task_instruction_matches": task_matches,
        "model_signature": {
            "archive": expected_hash,
            "compiled": actual_hash,
            "matches": hash_matches,
        },
        "compiled_model_dimensions": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "nsite": int(model.nsite),
        },
        "eef_frame": {"name": FRAME_NAME, "type": FRAME_TYPE, "site_id": frame_id},
        "arm_joint_contract": {
            "names": joint_names,
            "joint_ids": joint_ids,
            "actual_position_ranges_rad": actual_ranges,
            "archive_position_ranges_rad": joint_ranges,
            "model_position_ranges_match": True,
            "initial_qpos_rad": actual_joint_qpos,
            "initial_qvel_rad_s": actual_joint_qvel,
            "initial_joint_state_abs_errors": joint_errors,
            "velocity_limits_rad_s": dict(zip(joint_names, ARM_VELOCITY_LIMITS_RAD_S)),
            "velocity_limit_source": ARM_VELOCITY_LIMIT_SOURCE,
            "velocity_limit_source_url": ARM_VELOCITY_LIMIT_SOURCE_URL,
        },
        "initial_state_match": {
            "integration_state_length": int(integration.size),
            "integration_state_expected_length": int(
                mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
            ),
            "max_panda_joint_qpos_error_rad": max_joint_q_error,
            "max_panda_joint_qvel_error_rad_s": max_joint_v_error,
            "eef_position_error_m_max_component": eef_position_error,
            "eef_quaternion_error_max_component": eef_quaternion_error,
            "matches_atol_1e-6": max(
                max_joint_q_error,
                max_joint_v_error,
                eef_position_error,
                eef_quaternion_error,
            )
            <= 1e-6,
        },
        "scene_binding_consistency": {
            "status": "pass" if not binding_errors else "fail",
            "errors": binding_errors,
            "archived_scene_state_id_matches_planning_state_id": scene.get("state_id")
            == planning.get("state_id"),
            "carried_root_pose_max_abs_error": carried_pose_error,
            "carried_sphere_center_max_abs_error_m": sphere_center_error,
        },
        "coordinates": {
            "archive_frame": planning.get("frame"),
            "archive_units": planning.get("units"),
            "path_semantics": "world-frame carried conservative-sphere center in meters",
            "eef_transform_source": "v5 candidate initial grasp evidence",
        },
    }


def _make_mink_limits(
    mink: Any,
    model: Any,
    active_dof_ids: list[int],
    joint_names: list[str],
) -> list[Any]:
    from mink.limits.limit import Constraint, Limit

    active = set(active_dof_ids)
    locked = [index for index in range(int(model.nv)) if index not in active]

    class FreezeInactiveDofs(Limit):
        """Hard equality-via-inequalities lock for scene and gripper DOFs."""

        def compute_qp_inequalities(self, configuration: Any, dt: float) -> Any:
            del configuration, dt
            if not locked:
                return Constraint()
            matrix = np.zeros((2 * len(locked), model.nv), dtype=np.float64)
            for row, dof_id in enumerate(locked):
                matrix[row, dof_id] = 1.0
                matrix[row + len(locked), dof_id] = -1.0
            return Constraint(G=matrix, h=np.zeros(2 * len(locked), dtype=np.float64))

    velocity_map = dict(zip(joint_names, ARM_VELOCITY_LIMITS_RAD_S))
    return [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model, velocities=velocity_map),
        FreezeInactiveDofs(),
    ]


def _quat_angle(left: Any, right: Any) -> float:
    a = quaternion_normalized_wxyz(left)
    b = quaternion_normalized_wxyz(right)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), -1.0, 1.0)))


def _export_real_model(
    output_model: Path,
    output_metadata: Path,
    suite_name: str,
    level: int,
    task_id: int,
) -> dict[str, Any]:
    """LIBERO-Python worker; compiles actual scene without executing actions."""
    import mujoco
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = get_benchmark(suite_name)()
    task = suite.get_task_by_level_id(level, task_id)
    bddl = suite.get_task_bddl_file_path_by_level_id(level, task_id)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    try:
        env.seed(0)
        env.reset()
        initial_states = suite.get_task_init_states_by_level_id(level, task_id)
        env.set_init_state(initial_states[0])
        model = env.env.sim.model._model
        mujoco.mj_saveModel(model, str(output_model), None)
        joint_names = [f"robot0_joint{index}" for index in range(1, 8)]
        joints = []
        for name in joint_names:
            jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if jid < 0:
                raise RuntimeError(f"active LIBERO model is missing Panda joint {name}")
            joints.append(
                {
                    "name": name,
                    "joint_id": jid,
                    "joint_type": int(model.jnt_type[jid]),
                    "limited": bool(model.jnt_limited[jid]),
                    "range_rad": np.asarray(
                        model.jnt_range[jid], dtype=np.float64
                    ).tolist(),
                }
            )
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, FRAME_NAME))
        if site_id < 0:
            raise RuntimeError(
                f"active LIBERO model is missing Panda EEF site {FRAME_NAME}"
            )
        metadata = {
            "source": "current LIBERO-Safety OffScreenRenderEnv compiled scene model",
            "suite": suite_name,
            "level": level,
            "task_id": task_id,
            "instruction": getattr(task, "language", None),
            "mujoco_version": str(mujoco.__version__),
            "robosuite_version": str(__import__("robosuite").__version__),
            "model_signature": compute_model_signature(model),
            "mjb_size_bytes": output_model.stat().st_size,
            "dimensions": {
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nbody": int(model.nbody),
                "njnt": int(model.njnt),
                "nsite": int(model.nsite),
            },
            "eef_site": {"name": FRAME_NAME, "site_id": site_id},
            "arm_joints": joints,
            "actions_executed": 0,
            "physics_steps_executed": 0,
        }
        gripper = env.env.robots[0].gripper
        metadata["finger_geom_groups"] = {
            side: [
                {"id": int(env.sim.model.geom_name2id(name)), "name": name}
                for name in gripper.important_geoms[side]
            ]
            for side in ("left_finger", "right_finger")
        }
        output_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
    finally:
        env.close()


def _compile_real_model(
    python_executable: Path,
    output_model: Path,
    output_metadata: Path,
    suite: str,
    level: int,
    task_id: int,
    libero_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    script = Path(__file__).resolve()
    env = os.environ.copy()
    env["LIBERO_CONFIG_PATH"] = str(config_path)
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    path_items = [
        str(libero_root),
        str(libero_root / "third_party/robosuite-1.4"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(item for item in path_items if item)
    command = [
        str(python_executable),
        str(script),
        "--_export-real-model",
        str(output_model),
        "--_export-metadata",
        str(output_metadata),
        "--suite",
        suite,
        "--level",
        str(level),
        "--task-id",
        str(task_id),
    ]
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "LIBERO model compilation failed: "
            + completed.stderr[-3000:]
            + completed.stdout[-1000:]
        )
    metadata = json.loads(output_metadata.read_text())
    metadata["worker_python"] = str(python_executable)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--_export-real-model", dest="model", required=True)
    parser.add_argument("--_export-metadata", dest="metadata", required=True)
    parser.add_argument("--suite", default="obstacle_avoidance")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()
    result = _export_real_model(
        Path(args.model), Path(args.metadata), args.suite, args.level, args.task_id
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
